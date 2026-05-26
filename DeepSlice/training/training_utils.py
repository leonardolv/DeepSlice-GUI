from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def set_global_seed(seed: int) -> int:
    """Set Python, NumPy and TensorFlow seeds for reproducible training runs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass
    except Exception:
        # TensorFlow is optional for utility usage in data preparation scripts.
        pass

    return seed


def split_dataframe_by_group(
    dataframe: pd.DataFrame,
    group_column: str,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Split rows into train/val/test while preventing group leakage."""
    if group_column not in dataframe.columns:
        raise ValueError(f"group_column '{group_column}' not found in dataframe columns")

    train_fraction = float(train_fraction)
    val_fraction = float(val_fraction)
    if train_fraction <= 0 or val_fraction < 0:
        raise ValueError("train_fraction must be > 0 and val_fraction must be >= 0")
    if train_fraction < 0.10:
        raise ValueError("train_fraction must be at least 0.10 to produce a meaningful training split")
    if (train_fraction + val_fraction) >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.0")

    unique_groups = dataframe[group_column].dropna().astype(str).unique().tolist()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    if n_groups == 0:
        return {
            "train": dataframe.iloc[0:0].copy(),
            "val": dataframe.iloc[0:0].copy(),
            "test": dataframe.iloc[0:0].copy(),
        }

    if n_groups < 3 and val_fraction > 0:
        # With fewer than 3 groups we cannot guarantee non-empty val and test
        # splits without leakage. Warn loudly but proceed so legacy callers
        # that opt into degenerate behavior aren't broken.
        import warnings as _warnings
        _warnings.warn(
            f"Only {n_groups} distinct group(s) available with val_fraction>0; "
            "val and/or test splits may be empty.",
            stacklevel=2,
        )

    n_train = max(1, int(round(n_groups * train_fraction)))
    n_val = int(round(n_groups * val_fraction))
    if n_train + n_val >= n_groups:
        n_val = max(0, n_groups - n_train - 1)

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val :])

    groups = dataframe[group_column].astype(str)
    train_df = dataframe[groups.isin(train_groups)].reset_index(drop=True)
    val_df = dataframe[groups.isin(val_groups)].reset_index(drop=True)
    test_df = dataframe[groups.isin(test_groups)].reset_index(drop=True)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def balanced_class_weights(labels: Sequence) -> Dict[object, float]:
    labels = np.asarray(labels)
    if labels.size == 0:
        return {}

    classes, counts = np.unique(labels, return_counts=True)
    n_classes = float(len(classes))
    total = float(np.sum(counts))

    weights: Dict[object, float] = {}
    for cls, count in zip(classes.tolist(), counts.tolist()):
        weights[cls] = float(total / (n_classes * float(count)))
    return weights


def balanced_sample_weights(labels: Sequence) -> np.ndarray:
    class_weights = balanced_class_weights(labels)
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray([class_weights[item] for item in labels.tolist()], dtype=np.float32)


def hard_example_sample_weights(
    losses: Sequence[float],
    baseline: float = 1.0,
    high_percentile: float = 75.0,
    max_scale: float = 3.0,
) -> np.ndarray:
    losses_array = np.asarray(losses, dtype=np.float32)
    if losses_array.size == 0:
        return np.zeros(0, dtype=np.float32)

    baseline = float(max(0.0, baseline))
    if max_scale < baseline:
        raise ValueError(
            f"max_scale ({max_scale}) must be >= baseline ({baseline}); "
            "silently clamping would mask configuration mistakes."
        )
    max_scale = float(max_scale)
    threshold = float(np.percentile(losses_array, np.clip(high_percentile, 0.0, 100.0)))
    max_loss = float(np.max(losses_array))

    if max_loss <= threshold + 1e-8:
        return np.full(losses_array.shape[0], baseline, dtype=np.float32)

    excess = np.clip((losses_array - threshold) / (max_loss - threshold), 0.0, 1.0)
    weights = baseline + excess * (max_scale - baseline)
    return weights.astype(np.float32)


def latest_checkpoint_path(checkpoint_dir: str, prefix: str = "ckpt") -> Optional[str]:
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None

    candidates = []
    pattern = re.compile(rf"{re.escape(prefix)}[_-]epoch[_-](\d+)")

    for entry in os.listdir(checkpoint_dir):
        full_path = os.path.join(checkpoint_dir, entry)
        if not os.path.isfile(full_path):
            continue
        match = pattern.search(entry)
        epoch = int(match.group(1)) if match else -1
        mtime = os.path.getmtime(full_path)
        candidates.append((epoch, mtime, full_path))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def infer_initial_epoch_from_checkpoint(checkpoint_path: Optional[str]) -> int:
    if not checkpoint_path:
        return 0
    filename = os.path.basename(str(checkpoint_path))
    match = re.search(r"epoch[_-](\d+)", filename)
    if not match:
        return 0
    return int(match.group(1))


def build_training_callbacks(
    checkpoint_dir: str,
    monitor: str = "val_loss",
    patience: int = 8,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    save_best_only: bool = True,
    use_mixed_precision: bool = False,
):
    """Create common Keras callbacks used for stable and resumable training."""
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is required to build training callbacks") from exc

    os.makedirs(checkpoint_dir, exist_ok=True)

    if use_mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    checkpoint_path = os.path.join(checkpoint_dir, "ckpt_epoch-{epoch:03d}.weights.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=bool(save_best_only),
            save_weights_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=int(max(1, patience)),
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=float(np.clip(lr_factor, 0.05, 0.95)),
            patience=max(1, int(max(1, patience) // 2)),
            min_lr=float(max(0.0, min_lr)),
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        # Timestamp the CSV log so successive runs don't append into one file.
        tf.keras.callbacks.CSVLogger(
            os.path.join(
                checkpoint_dir,
                f"training_log_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv",
            ),
            append=False,
        ),
    ]

    return callbacks, checkpoint_path


def temperature_scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(max(1e-6, temperature))
    return np.asarray(logits, dtype=np.float32) / temperature


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: Sequence,
    n_bins: int = 15,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float32)
    labels = np.asarray(labels)
    if probs.size == 0 or labels.size == 0:
        return 0.0

    if probs.ndim == 1:
        confidence = np.clip(probs, 0.0, 1.0)
        predicted = (confidence >= 0.5).astype(int)
        correctness = (predicted == labels.astype(int)).astype(np.float32)
    elif probs.ndim == 2:
        confidence = np.clip(np.max(probs, axis=1), 0.0, 1.0)
        predicted = np.argmax(probs, axis=1)
        correctness = (predicted == labels.astype(int)).astype(np.float32)
    else:
        raise ValueError("probabilities must be rank-1 or rank-2")

    bin_edges = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    ece = 0.0
    total = float(len(confidence))

    for idx in range(len(bin_edges) - 1):
        left, right = float(bin_edges[idx]), float(bin_edges[idx + 1])
        if idx == len(bin_edges) - 2:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)

        if not np.any(mask):
            continue
        bin_conf = float(np.mean(confidence[mask]))
        bin_acc = float(np.mean(correctness[mask]))
        bin_weight = float(np.sum(mask)) / total
        ece += abs(bin_acc - bin_conf) * bin_weight

    return float(np.clip(ece, 0.0, 1.0))


def mixup_batch(
    images: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.20,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must have the same batch dimension")
    if images.shape[0] == 0:
        return images, labels

    rng = np.random.default_rng(seed)
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    permutation = rng.permutation(images.shape[0])

    mixed_images = (lam * images) + ((1.0 - lam) * images[permutation])
    mixed_labels = (lam * labels) + ((1.0 - lam) * labels[permutation])
    return mixed_images.astype(np.float32), mixed_labels.astype(np.float32)


def _random_cutmix_bbox(height: int, width: int, lam: float, rng) -> Tuple[int, int, int, int]:
    if height <= 0 or width <= 0:
        return 0, 0, 0, 0

    cut_ratio = np.sqrt(max(0.0, 1.0 - lam))
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = int(rng.integers(0, width))
    cy = int(rng.integers(0, height))

    x0 = int(np.clip(cx - (cut_w // 2), 0, width))
    x1 = int(np.clip(cx + (cut_w // 2), 0, width))
    y0 = int(np.clip(cy - (cut_h // 2), 0, height))
    y1 = int(np.clip(cy + (cut_h // 2), 0, height))
    return x0, y0, x1, y1


def cutmix_batch(
    images: np.ndarray,
    labels: np.ndarray,
    alpha: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if images.ndim != 4:
        raise ValueError("cutmix_batch expects images with shape [B, H, W, C]")
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must have the same batch dimension")
    if images.shape[0] == 0:
        return images, labels

    rng = np.random.default_rng(seed)
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    permutation = rng.permutation(images.shape[0])

    mixed = np.array(images, copy=True)
    h, w = images.shape[1], images.shape[2]
    x0, y0, x1, y1 = _random_cutmix_bbox(h, w, lam, rng)

    mixed[:, y0:y1, x0:x1, :] = images[permutation, y0:y1, x0:x1, :]

    patch_area = float(max(0, x1 - x0) * max(0, y1 - y0))
    lam_adjusted = 1.0 - (patch_area / float(max(1, h * w)))
    mixed_labels = (lam_adjusted * labels) + ((1.0 - lam_adjusted) * labels[permutation])
    return mixed.astype(np.float32), mixed_labels.astype(np.float32)


def training_run_metadata(
    *,
    species: str,
    seed: int,
    train_size: int,
    val_size: int,
    test_size: int,
    model_version: str,
    preprocessing_options: Optional[dict] = None,
    training_options: Optional[dict] = None,
) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "species": str(species),
        "seed": int(seed),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "model_version": str(model_version),
        "preprocessing_options": dict(preprocessing_options or {}),
        "training_options": dict(training_options or {}),
    }


def save_training_run_metadata(output_path: str, metadata: dict) -> str:
    if not output_path:
        raise ValueError("output_path is required")
    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return output_path
