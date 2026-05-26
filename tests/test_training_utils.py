import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.training.training_utils import (
    balanced_class_weights,
    balanced_sample_weights,
    build_training_callbacks,
    cutmix_batch,
    expected_calibration_error,
    hard_example_sample_weights,
    mixup_batch,
    save_training_run_metadata,
    set_global_seed,
    split_dataframe_by_group,
    training_run_metadata,
)


def test_set_global_seed_reproducible_for_numpy():
    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_split_dataframe_by_group_has_no_leakage():
    df = pd.DataFrame(
        {
            "brain_id": ["A", "A", "B", "B", "C", "D", "E", "F"],
            "value": list(range(8)),
        }
    )

    split = split_dataframe_by_group(df, group_column="brain_id", seed=7)
    train_groups = set(split["train"]["brain_id"].unique().tolist())
    val_groups = set(split["val"]["brain_id"].unique().tolist())
    test_groups = set(split["test"]["brain_id"].unique().tolist())

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == len(df)


def test_balanced_weight_helpers():
    labels = np.array([0, 0, 0, 1, 1, 2])
    class_weights = balanced_class_weights(labels)
    sample_weights = balanced_sample_weights(labels)

    assert class_weights[2] > class_weights[0]
    assert sample_weights.shape[0] == labels.shape[0]
    assert sample_weights[-1] == pytest.approx(class_weights[2])


def test_hard_example_weights_emphasize_high_losses():
    losses = np.array([0.1, 0.2, 0.3, 1.2, 2.0], dtype=np.float32)
    weights = hard_example_sample_weights(losses, baseline=1.0, high_percentile=60.0, max_scale=4.0)

    assert weights.shape[0] == losses.shape[0]
    assert float(weights[-1]) >= float(weights[0])
    assert np.all(weights >= 1.0)


def test_ece_bounds_and_perfect_case():
    probs = np.array([[0.99, 0.01], [0.02, 0.98], [0.97, 0.03]], dtype=np.float32)
    labels = np.array([0, 1, 0], dtype=int)
    ece = expected_calibration_error(probs, labels, n_bins=10)

    assert 0.0 <= ece <= 1.0
    assert ece < 0.05


def test_mixup_and_cutmix_shapes():
    images = np.random.rand(6, 32, 32, 3).astype(np.float32)
    labels = np.eye(3, dtype=np.float32)[np.array([0, 1, 2, 0, 1, 2])]

    mixed_images, mixed_labels = mixup_batch(images, labels, alpha=0.2, seed=42)
    cutmix_images, cutmix_labels = cutmix_batch(images, labels, alpha=1.0, seed=42)

    assert mixed_images.shape == images.shape
    assert mixed_labels.shape == labels.shape
    assert cutmix_images.shape == images.shape
    assert cutmix_labels.shape == labels.shape


def test_training_metadata_roundtrip(tmp_path):
    metadata = training_run_metadata(
        species="mouse",
        seed=123,
        train_size=10,
        val_size=2,
        test_size=3,
        model_version="v1.2.0",
        preprocessing_options={"clahe": True},
        training_options={"epochs": 20},
    )

    out_path = tmp_path / "training_metadata.json"
    save_training_run_metadata(str(out_path), metadata)

    with out_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    assert loaded["species"] == "mouse"
    assert loaded["seed"] == 123
    assert loaded["preprocessing_options"]["clahe"] is True


def test_build_training_callbacks_creates_checkpoint_template(tmp_path):
    tf = pytest.importorskip("tensorflow")
    callbacks, checkpoint_template = build_training_callbacks(str(tmp_path), patience=4)

    assert isinstance(callbacks, list)
    assert len(callbacks) >= 4
    assert str(tmp_path) in checkpoint_template
    assert "epoch" in checkpoint_template
