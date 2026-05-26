"""Tests for training pipeline: label loading, split manifest, and utility
functions that do not require TensorFlow or real images."""
import json
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.training.training_utils import (
    _random_cutmix_bbox,
    balanced_class_weights,
    cutmix_batch,
    infer_initial_epoch_from_checkpoint,
    latest_checkpoint_path,
    mixup_batch,
    set_global_seed,
    split_dataframe_by_group,
    temperature_scale_logits,
)
from DeepSlice.training.train_runner import (
    _load_labels_dataframe,
    _split_manifest_payload,
    _standardize_labels_dataframe,
    build_split_dataframe,
    discover_image_paths,
    prepare_training_run,
)

TARGET_COLUMNS = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_labels_csv(path: pathlib.Path, n: int = 5, inf_row: int = None) -> pathlib.Path:
    rows = []
    for i in range(n):
        row = {
            "filename": f"brain_s{i + 1:03d}.png",
            "ox": 1.0,
            "oy": float(i * 10),
            "oz": 2.0,
            "ux": 1.0,
            "uy": 0.0,
            "uz": 0.0,
            "vx": 0.0,
            "vy": 1.0,
            "vz": 0.0,
        }
        if inf_row is not None and i == inf_row:
            row["oy"] = float("inf")
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = path / "labels.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _make_images_dir(root: pathlib.Path, structure: dict) -> None:
    """Create directory tree with dummy image files. structure = {folder: [filenames]}"""
    for folder, filenames in structure.items():
        folder_path = root / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            (folder_path / filename).write_bytes(b"fake")


# ---------------------------------------------------------------------------
# split_dataframe_by_group edge cases
# ---------------------------------------------------------------------------

def test_split_tiny_single_group():
    df = pd.DataFrame({"group": ["A", "A", "A"], "val": [1, 2, 3]})
    split = split_dataframe_by_group(df, "group", train_fraction=0.70, val_fraction=0.15)
    total = len(split["train"]) + len(split["val"]) + len(split["test"])
    assert total == len(df)


def test_split_two_groups():
    df = pd.DataFrame({"group": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
    split = split_dataframe_by_group(df, "group", train_fraction=0.60, val_fraction=0.20)
    train_groups = set(split["train"]["group"].unique())
    val_groups = set(split["val"]["group"].unique())
    test_groups = set(split["test"]["group"].unique())
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_split_rejects_too_small_train_fraction():
    df = pd.DataFrame({"group": ["A", "B", "C"], "val": [1, 2, 3]})
    with pytest.raises(ValueError, match="0.10"):
        split_dataframe_by_group(df, "group", train_fraction=0.05)


def test_split_missing_group_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="group_column"):
        split_dataframe_by_group(df, "group")


def test_split_empty_dataframe():
    df = pd.DataFrame({"group": pd.Series([], dtype=str), "val": pd.Series([], dtype=float)})
    split = split_dataframe_by_group(df, "group")
    assert len(split["train"]) == 0
    assert len(split["val"]) == 0
    assert len(split["test"]) == 0


# ---------------------------------------------------------------------------
# _random_cutmix_bbox
# ---------------------------------------------------------------------------

def test_cutmix_bbox_zero_dimensions_returns_zeros():
    rng = np.random.default_rng(0)
    x0, y0, x1, y1 = _random_cutmix_bbox(0, 100, 0.5, rng)
    assert x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0


def test_cutmix_bbox_zero_height_returns_zeros():
    rng = np.random.default_rng(0)
    x0, y0, x1, y1 = _random_cutmix_bbox(0, 0, 0.5, rng)
    assert (x0, y0, x1, y1) == (0, 0, 0, 0)


def test_cutmix_bbox_normal_case():
    rng = np.random.default_rng(42)
    x0, y0, x1, y1 = _random_cutmix_bbox(100, 100, 0.5, rng)
    assert 0 <= x0 <= x1 <= 100
    assert 0 <= y0 <= y1 <= 100


# ---------------------------------------------------------------------------
# mixup_batch empty input
# ---------------------------------------------------------------------------

def test_mixup_batch_empty_input():
    images = np.zeros((0, 32, 32, 3), dtype=np.float32)
    labels = np.zeros((0, 9), dtype=np.float32)
    out_images, out_labels = mixup_batch(images, labels)
    assert out_images.shape == (0, 32, 32, 3)
    assert out_labels.shape == (0, 9)


def test_cutmix_batch_empty_input():
    images = np.zeros((0, 32, 32, 3), dtype=np.float32)
    labels = np.zeros((0, 9), dtype=np.float32)
    out_images, out_labels = cutmix_batch(images, labels)
    assert out_images.shape == (0, 32, 32, 3)


# ---------------------------------------------------------------------------
# temperature_scale_logits
# ---------------------------------------------------------------------------

def test_temperature_scale_logits_scales_correctly():
    logits = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    scaled = temperature_scale_logits(logits, temperature=2.0)
    assert np.allclose(scaled, [1.0, 2.0, 3.0])


def test_temperature_scale_logits_clamps_near_zero_temperature():
    logits = np.array([1.0, 2.0], dtype=np.float32)
    # Near-zero temperature should not cause division by zero
    scaled = temperature_scale_logits(logits, temperature=1e-10)
    assert np.isfinite(scaled).all()


# ---------------------------------------------------------------------------
# balanced_class_weights empty input
# ---------------------------------------------------------------------------

def test_balanced_class_weights_empty():
    result = balanced_class_weights(np.array([]))
    assert result == {}


# ---------------------------------------------------------------------------
# infer_initial_epoch_from_checkpoint
# ---------------------------------------------------------------------------

def test_infer_initial_epoch_none():
    assert infer_initial_epoch_from_checkpoint(None) == 0


def test_infer_initial_epoch_malformed_filename():
    assert infer_initial_epoch_from_checkpoint("/no/epoch/in/name.h5") == 0


def test_infer_initial_epoch_valid():
    assert infer_initial_epoch_from_checkpoint("/checkpoints/ckpt_epoch-042.weights.h5") == 42


def test_infer_initial_epoch_epoch_underscore():
    assert infer_initial_epoch_from_checkpoint("/checkpoints/ckpt_epoch_007.h5") == 7


# ---------------------------------------------------------------------------
# latest_checkpoint_path
# ---------------------------------------------------------------------------

def test_latest_checkpoint_path_missing_dir():
    result = latest_checkpoint_path("/no/such/dir")
    assert result is None


def test_latest_checkpoint_path_empty_dir(tmp_path):
    result = latest_checkpoint_path(str(tmp_path))
    assert result is None


def test_latest_checkpoint_path_finds_highest_epoch(tmp_path):
    (tmp_path / "ckpt_epoch-001.weights.h5").write_bytes(b"x")
    (tmp_path / "ckpt_epoch-005.weights.h5").write_bytes(b"x")
    (tmp_path / "ckpt_epoch-003.weights.h5").write_bytes(b"x")
    result = latest_checkpoint_path(str(tmp_path))
    assert result is not None
    assert "epoch-005" in result


# ---------------------------------------------------------------------------
# _standardize_labels_dataframe
# ---------------------------------------------------------------------------

def _build_labels_df(n=4, bad_row=None, inf_row=None):
    rows = []
    for i in range(n):
        row = {
            "Filenames": f"img_{i:03d}.png",
            **{col: float(i) for col in TARGET_COLUMNS},
        }
        if bad_row is not None and i == bad_row:
            row["oy"] = "not_a_number"
        if inf_row is not None and i == inf_row:
            row["oy"] = float("inf")
        rows.append(row)
    return pd.DataFrame(rows)


def test_standardize_labels_rejects_non_numeric():
    df = _build_labels_df(bad_row=2)
    df = df.rename(columns={"Filenames": "filename"})
    with pytest.raises(ValueError, match="non-numeric"):
        _standardize_labels_dataframe(df)


def test_standardize_labels_rejects_inf():
    df = _build_labels_df(inf_row=1)
    df = df.rename(columns={"Filenames": "filename"})
    with pytest.raises(ValueError, match="infinite"):
        _standardize_labels_dataframe(df)


def test_standardize_labels_rejects_missing_target_columns():
    df = pd.DataFrame({"filename": ["a.png"], "ox": [1.0]})
    with pytest.raises(ValueError, match="missing target columns"):
        _standardize_labels_dataframe(df)


def test_standardize_labels_rejects_missing_identifier():
    row = {col: 1.0 for col in TARGET_COLUMNS}
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="identifier"):
        _standardize_labels_dataframe(df)


def test_standardize_labels_rejects_duplicates():
    rows = [{"filename": "img.png", **{col: 1.0 for col in TARGET_COLUMNS}} for _ in range(2)]
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="duplicate"):
        _standardize_labels_dataframe(df)


def test_standardize_labels_accepts_valid():
    df = _build_labels_df()
    df = df.rename(columns={"Filenames": "filename"})
    standardized = _standardize_labels_dataframe(df)
    assert set(TARGET_COLUMNS).issubset(set(standardized.columns))


# ---------------------------------------------------------------------------
# _load_labels_dataframe
# ---------------------------------------------------------------------------

def test_load_labels_dataframe_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        _load_labels_dataframe(str(tmp_path / "missing.csv"))


def test_load_labels_dataframe_unsupported_extension(tmp_path):
    path = tmp_path / "labels.xlsx"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="CSV/TSV/JSON"):
        _load_labels_dataframe(str(path))


def test_load_labels_dataframe_valid_csv(tmp_path):
    csv_path = _make_labels_csv(tmp_path)
    df = _load_labels_dataframe(str(csv_path))
    assert set(TARGET_COLUMNS).issubset(set(df.columns))
    assert len(df) == 5


def test_load_labels_dataframe_with_inf_raises(tmp_path):
    csv_path = _make_labels_csv(tmp_path, inf_row=2)
    with pytest.raises(ValueError, match="infinite"):
        _load_labels_dataframe(str(csv_path))


# ---------------------------------------------------------------------------
# _split_manifest_payload — leakage detection
# ---------------------------------------------------------------------------

def test_split_manifest_payload_reports_leakage_free():
    train = pd.DataFrame({"group_id": ["A", "A"], "path": ["/a1.png", "/a2.png"]})
    val = pd.DataFrame({"group_id": ["B"], "path": ["/b1.png"]})
    test = pd.DataFrame({"group_id": ["C"], "path": ["/c1.png"]})
    split = {"train": train, "val": val, "test": test}

    payload = _split_manifest_payload(
        split,
        species="mouse",
        group_mode="parent-folder",
        seed=42,
        train_fraction=0.60,
        val_fraction=0.20,
    )
    assert payload["leakage_free"] is True


def test_split_manifest_payload_detects_leakage():
    shared = pd.DataFrame({"group_id": ["A"], "path": ["/a.png"]})
    split = {"train": shared, "val": shared, "test": shared}

    payload = _split_manifest_payload(
        split,
        species="mouse",
        group_mode="parent-folder",
        seed=42,
        train_fraction=0.60,
        val_fraction=0.20,
    )
    assert payload["leakage_free"] is False


# ---------------------------------------------------------------------------
# prepare_training_run — no images raises
# ---------------------------------------------------------------------------

def test_prepare_training_run_no_images_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="No supported images"):
        prepare_training_run(
            image_root=str(empty_dir),
            output_dir=str(out_dir),
        )


# ---------------------------------------------------------------------------
# discover_image_paths
# ---------------------------------------------------------------------------

def test_discover_image_paths_recursive(tmp_path):
    _make_images_dir(
        tmp_path,
        {
            "brainA": ["s001.png", "s002.tif"],
            "brainA/sub": ["s003.jpg"],
            "brainB": ["s004.png"],
        },
    )
    paths = discover_image_paths(str(tmp_path), recursive=True)
    assert len(paths) == 4


def test_discover_image_paths_non_recursive(tmp_path):
    _make_images_dir(
        tmp_path,
        {
            ".": ["top.png"],
            "sub": ["nested.png"],
        },
    )
    paths = discover_image_paths(str(tmp_path), recursive=False)
    assert len(paths) == 1
    assert paths[0].endswith("top.png")


def test_discover_image_paths_filters_non_images(tmp_path):
    (tmp_path / "document.txt").write_text("x")
    (tmp_path / "image.png").write_bytes(b"x")
    (tmp_path / "archive.zip").write_bytes(b"x")
    paths = discover_image_paths(str(tmp_path))
    assert len(paths) == 1


def test_discover_image_paths_missing_dir():
    with pytest.raises(ValueError, match="does not exist"):
        discover_image_paths("/no/such/directory")
