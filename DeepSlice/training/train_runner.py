from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..metadata import metadata_loader

_logger = logging.getLogger(__name__)
from ..neural_network.neural_network import XCEPTION_INPUT_SIZE
from . import training_utils

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
GROUP_MODES = {"parent-folder", "filename-prefix", "full-path"}
TRAINER_MODES = {"baseline-cnn", "xception-finetune"}
XCEPTION_INIT_MODES = {"species-primary", "imagenet"}
TARGET_COLUMNS = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]


def _normalize_group_mode(mode: str) -> str:
    normalized = str(mode or "parent-folder").strip().lower()
    if normalized not in GROUP_MODES:
        raise ValueError(
            "group_mode must be one of " + ", ".join(sorted(GROUP_MODES))
        )
    return normalized


def _normalize_trainer_mode(mode: str) -> str:
    normalized = str(mode or "baseline-cnn").strip().lower()
    if normalized not in TRAINER_MODES:
        raise ValueError(
            "trainer must be one of " + ", ".join(sorted(TRAINER_MODES))
        )
    return normalized


def _normalize_xception_init_mode(mode: str) -> str:
    normalized = str(mode or "species-primary").strip().lower()
    if normalized not in XCEPTION_INIT_MODES:
        raise ValueError(
            "xception_init must be one of " + ", ".join(sorted(XCEPTION_INIT_MODES))
        )
    return normalized


def _group_id_from_path(image_path: str, group_mode: str) -> str:
    group_mode = _normalize_group_mode(group_mode)
    absolute_path = os.path.abspath(str(image_path))
    filename = os.path.basename(absolute_path)
    stem = os.path.splitext(filename)[0]

    if group_mode == "parent-folder":
        parent = os.path.basename(os.path.dirname(absolute_path))
        return str(parent or stem)

    if group_mode == "filename-prefix":
        match = re.match(r"(.+?)_s\d{1,5}(?:_|$)", stem, flags=re.IGNORECASE)
        if match is not None and match.group(1).strip() != "":
            return str(match.group(1).strip())
        return stem

    return absolute_path


def discover_image_paths(image_root: str, recursive: bool = True) -> List[str]:
    root = os.path.abspath(str(image_root))
    if not os.path.isdir(root):
        raise ValueError(f"Image directory does not exist: {root}")

    paths: List[str] = []
    if recursive:
        for walk_root, _, filenames in os.walk(root):
            for filename in filenames:
                extension = os.path.splitext(filename)[1].lower()
                if extension in SUPPORTED_IMAGE_FORMATS:
                    paths.append(os.path.abspath(os.path.join(walk_root, filename)))
    else:
        for filename in os.listdir(root):
            absolute = os.path.join(root, filename)
            if not os.path.isfile(absolute):
                continue
            extension = os.path.splitext(filename)[1].lower()
            if extension in SUPPORTED_IMAGE_FORMATS:
                paths.append(os.path.abspath(absolute))

    return sorted(paths)


def build_split_dataframe(image_paths: Sequence[str], group_mode: str) -> pd.DataFrame:
    group_mode = _normalize_group_mode(group_mode)
    records = []
    for image_path in image_paths:
        absolute = os.path.abspath(str(image_path))
        records.append(
            {
                "path": absolute,
                "filename": os.path.basename(absolute),
                "group_id": _group_id_from_path(absolute, group_mode),
            }
        )
    return pd.DataFrame(records)


def _split_manifest_payload(
    split: Dict[str, pd.DataFrame],
    *,
    species: str,
    group_mode: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> Dict[str, object]:
    train_df = split["train"]
    val_df = split["val"]
    test_df = split["test"]

    train_groups = set(train_df["group_id"].astype(str).unique().tolist())
    val_groups = set(val_df["group_id"].astype(str).unique().tolist())
    test_groups = set(test_df["group_id"].astype(str).unique().tolist())

    leakage_free = (
        train_groups.isdisjoint(val_groups)
        and train_groups.isdisjoint(test_groups)
        and val_groups.isdisjoint(test_groups)
    )

    return {
        "species": str(species),
        "group_mode": str(group_mode),
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "counts": {
            "total": int(len(train_df) + len(val_df) + len(test_df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "group_counts": {
            "train": int(len(train_groups)),
            "val": int(len(val_groups)),
            "test": int(len(test_groups)),
        },
        "leakage_free": bool(leakage_free),
        "splits": {
            "train": train_df["path"].astype(str).tolist(),
            "val": val_df["path"].astype(str).tolist(),
            "test": test_df["path"].astype(str).tolist(),
        },
    }


def _resolve_checkpoint_dir(output_dir: str, checkpoint_dir: Optional[str]) -> str:
    if checkpoint_dir:
        return os.path.abspath(str(checkpoint_dir))
    return os.path.join(os.path.abspath(str(output_dir)), "checkpoints")


def prepare_training_run(
    *,
    image_root: str,
    output_dir: str,
    species: str = "mouse",
    group_mode: str = "parent-folder",
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    recursive: bool = True,
    preprocessing_options: Optional[dict] = None,
    training_options: Optional[dict] = None,
    split_manifest_name: str = "training_split_manifest.json",
    metadata_name: str = "training_run_metadata.json",
) -> Dict[str, object]:
    output_dir = os.path.abspath(str(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    training_utils.set_global_seed(int(seed))
    image_paths = discover_image_paths(image_root, recursive=recursive)
    if len(image_paths) == 0:
        raise ValueError("No supported images were found in the provided directory")

    split_df = build_split_dataframe(image_paths, group_mode=group_mode)
    split = training_utils.split_dataframe_by_group(
        dataframe=split_df,
        group_column="group_id",
        train_fraction=float(train_fraction),
        val_fraction=float(val_fraction),
        seed=int(seed),
    )

    split_manifest = _split_manifest_payload(
        split,
        species=species,
        group_mode=_normalize_group_mode(group_mode),
        seed=int(seed),
        train_fraction=float(train_fraction),
        val_fraction=float(val_fraction),
    )

    split_manifest_path = os.path.join(output_dir, str(split_manifest_name))
    with open(split_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2, sort_keys=True)

    config, _ = metadata_loader.load_config()
    model_version = str(
        config.get("DeepSlice_version", {}).get("prerelease", "unknown")
    )

    metadata = training_utils.training_run_metadata(
        species=species,
        seed=int(seed),
        train_size=int(split_manifest["counts"]["train"]),
        val_size=int(split_manifest["counts"]["val"]),
        test_size=int(split_manifest["counts"]["test"]),
        model_version=model_version,
        preprocessing_options=dict(preprocessing_options or {}),
        training_options=dict(training_options or {}),
    )
    metadata["split_group_mode"] = str(split_manifest["group_mode"])
    metadata["split_counts"] = split_manifest["counts"]
    metadata["split_group_counts"] = split_manifest["group_counts"]
    metadata["split_leakage_free"] = bool(split_manifest["leakage_free"])

    metadata_path = os.path.join(output_dir, str(metadata_name))
    training_utils.save_training_run_metadata(metadata_path, metadata)

    return {
        "split_manifest_path": split_manifest_path,
        "metadata_path": metadata_path,
        "split_manifest": split_manifest,
        "metadata": metadata,
        "split": split,
    }


def _standardize_labels_dataframe(labels_df: pd.DataFrame) -> pd.DataFrame:
    if labels_df is None or len(labels_df) == 0:
        raise ValueError("labels file is empty")

    normalized_names = {str(name).strip().lower(): name for name in labels_df.columns}

    missing_targets = [col for col in TARGET_COLUMNS if col not in normalized_names]
    if len(missing_targets) > 0:
        raise ValueError(
            "labels file is missing target columns: " + ", ".join(missing_targets)
        )

    path_aliases = ["path", "image_path", "filepath", "file_path", "source_path"]
    filename_aliases = ["filename", "filenames", "image", "image_name"]

    identifier_kind = None
    identifier_source = None
    for alias in path_aliases:
        if alias in normalized_names:
            identifier_kind = "path"
            identifier_source = normalized_names[alias]
            break
    if identifier_source is None:
        for alias in filename_aliases:
            if alias in normalized_names:
                identifier_kind = "filename"
                identifier_source = normalized_names[alias]
                break
    if identifier_source is None:
        raise ValueError(
            "labels file must include one identifier column: "
            + ", ".join(path_aliases + filename_aliases)
        )

    standardized = pd.DataFrame()
    standardized[identifier_kind] = labels_df[identifier_source].astype(str)
    if identifier_kind == "path":
        standardized["path"] = standardized["path"].apply(
            lambda value: os.path.abspath(str(value))
        )
    else:
        standardized["filename"] = standardized["filename"].apply(
            lambda value: os.path.basename(str(value))
        )

    for target in TARGET_COLUMNS:
        source = normalized_names[target]
        standardized[target] = pd.to_numeric(labels_df[source], errors="coerce")

    if standardized[TARGET_COLUMNS].isna().any().any():
        raise ValueError("labels file contains non-numeric target values")

    non_finite_mask = ~np.isfinite(standardized[TARGET_COLUMNS].to_numpy(dtype=float))
    if non_finite_mask.any():
        raise ValueError("labels file contains infinite target values (Inf/-Inf)")

    key = "path" if "path" in standardized.columns else "filename"
    duplicate_mask = standardized[key].duplicated(keep=False)
    if bool(np.any(duplicate_mask)):
        duplicate_values = (
            standardized.loc[duplicate_mask, key].astype(str).head(3).tolist()
        )
        raise ValueError(
            "labels file has duplicate identifiers for "
            f"{key}: {', '.join(duplicate_values)}"
        )

    return standardized


def _load_labels_dataframe(labels_path: str) -> pd.DataFrame:
    labels_path = os.path.abspath(str(labels_path))
    if not os.path.isfile(labels_path):
        raise ValueError(f"labels file not found: {labels_path}")

    extension = os.path.splitext(labels_path)[1].lower()
    if extension in {".csv", ".tsv"}:
        sep = "\t" if extension == ".tsv" else ","
        labels_df = pd.read_csv(labels_path, sep=sep)
    elif extension == ".json":
        with open(labels_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "records" in payload:
            payload = payload["records"]
        labels_df = pd.DataFrame(payload)
    else:
        raise ValueError("labels file must be CSV/TSV/JSON")

    return _standardize_labels_dataframe(labels_df)


def _attach_labels_to_split(split_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    if "path" in labels_df.columns:
        merged = split_df.merge(
            labels_df[["path"] + TARGET_COLUMNS],
            on="path",
            how="left",
            validate="many_to_one",
        )
    else:
        merged = split_df.merge(
            labels_df[["filename"] + TARGET_COLUMNS],
            on="filename",
            how="left",
            validate="many_to_one",
        )

    missing_targets = merged[TARGET_COLUMNS].isna().any(axis=1)
    if bool(np.any(missing_targets)):
        missing_files = (
            merged.loc[missing_targets, "filename"].astype(str).head(5).tolist()
        )
        raise ValueError(
            "Missing labels for split images: " + ", ".join(missing_files)
        )

    return merged


class _LabeledSequence:
    """Keras Sequence-compatible wrapper that streams (images, labels) batches.

    Avoids loading the entire training set into memory: at 299x299x3 float32
    one image is ~1 MB, so 5k images would consume 5 GB before fit() begins.
    """

    def __init__(self, image_sequence, labels: np.ndarray):
        self._sequence = image_sequence
        self._labels = np.asarray(labels, dtype=np.float32)
        self.batch_size = int(getattr(image_sequence, "batch_size", 1))
        self.n = int(getattr(image_sequence, "n", len(self._labels)))

    def __len__(self):
        return len(self._sequence)

    def __getitem__(self, index):
        batch_x = self._sequence[index]
        start = int(index) * self.batch_size
        end = min(start + batch_x.shape[0], self.n)
        return batch_x, self._labels[start:end]


def _build_labeled_sequence(
    split_with_labels: pd.DataFrame,
    *,
    preprocessing_options: dict,
    batch_size: int,
):
    from ..neural_network import neural_network

    if len(split_with_labels) == 0:
        return None

    sequence = neural_network.PreprocessedImageSequence(
        split_with_labels["path"].astype(str).tolist(),
        batch_size=int(max(1, batch_size)),
        preprocessing_options=preprocessing_options,
    )
    targets = split_with_labels[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    return _LabeledSequence(sequence, targets)


def _validate_learning_rate(value: float, name: str = "learning_rate") -> float:
    """Reject zero/negative/non-finite learning rates that Adam would silently accept."""
    learning_rate = float(value)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return learning_rate


def _build_baseline_regressor(learning_rate: float):
    import tensorflow as tf

    learning_rate = _validate_learning_rate(learning_rate, "baseline learning_rate")

    inputs = tf.keras.Input(shape=XCEPTION_INPUT_SIZE, name="image")
    x = tf.keras.layers.Conv2D(16, 5, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(len(TARGET_COLUMNS), activation="linear", name="coordinates")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="deepslice_baseline_regressor")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def _build_fit_kwargs(
    train_sequence,
    *,
    epochs: int,
    callbacks: list,
    val_sequence,
    initial_epoch: int = 0,
) -> Dict[str, object]:
    fit_kwargs: Dict[str, object] = {
        "x": train_sequence,
        "epochs": int(epochs),
        "callbacks": callbacks,
        "verbose": 2,
    }
    if int(initial_epoch) > 0:
        fit_kwargs["initial_epoch"] = int(initial_epoch)
    if val_sequence is not None and val_sequence.n > 0:
        fit_kwargs["validation_data"] = val_sequence
    return fit_kwargs


def _serialize_history_dict(history_parts: Sequence[Dict[str, Sequence[float]]]) -> Dict[str, List[float]]:
    merged: Dict[str, List[float]] = {}
    for history in history_parts:
        for key, values in history.items():
            merged.setdefault(key, [])
            merged[key].extend(float(item) for item in values)
    return merged


def _save_training_artifacts(
    *,
    model,
    history_dict: Dict[str, List[float]],
    output_dir: str,
) -> Dict[str, str]:
    output_dir = os.path.abspath(str(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "deepslice_training_model.keras")
    history_path = os.path.join(output_dir, "training_history.json")

    model.save(model_path)
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history_dict, handle, indent=2, sort_keys=True)

    return {
        "model_path": model_path,
        "history_path": history_path,
    }


def _build_xception_regressor(species: str, xception_init: str):
    from ..neural_network import neural_network

    xception_init = _normalize_xception_init_mode(xception_init)
    config, metadata_path = metadata_loader.load_config()

    xception_weights = metadata_loader.get_data_path(
        config["weight_file_paths"]["xception_imagenet"],
        metadata_path,
    )
    initial_weights = None
    if xception_init == "species-primary":
        initial_weights = metadata_loader.get_data_path(
            config["weight_file_paths"][str(species)]["primary"],
            metadata_path,
        )

    return neural_network.initialise_network(
        xception_weights=xception_weights,
        weights=initial_weights,
        species=str(species),
    )


def _set_xception_backbone_trainable(model, trainable: bool) -> str:
    import tensorflow as tf

    for layer in getattr(model, "layers", []):
        if isinstance(layer, tf.keras.Model) and str(layer.name).strip().lower().startswith("xception"):
            layer.trainable = bool(trainable)
            return str(layer.name)
    raise RuntimeError("Unable to locate Xception backbone in model")


def _compile_regression_model(model, learning_rate: float):
    import tensorflow as tf

    learning_rate = _validate_learning_rate(learning_rate, "compile learning_rate")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def run_supervised_baseline_training(
    *,
    split: Dict[str, pd.DataFrame],
    labels_path: str,
    output_dir: str,
    preprocessing_options: dict,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    checkpoint_dir: str,
    patience: int,
    lr_factor: float,
    min_lr: float,
    mixed_precision: bool,
) -> Dict[str, object]:
    if int(epochs) <= 0:
        raise ValueError("epochs must be greater than zero for supervised training")

    labels_df = _load_labels_dataframe(labels_path)
    train_df = _attach_labels_to_split(split["train"], labels_df)
    val_df = _attach_labels_to_split(split["val"], labels_df)
    test_df = _attach_labels_to_split(split["test"], labels_df)

    train_sequence = _build_labeled_sequence(
        train_df,
        preprocessing_options=preprocessing_options,
        batch_size=batch_size,
    )
    val_sequence = _build_labeled_sequence(
        val_df,
        preprocessing_options=preprocessing_options,
        batch_size=batch_size,
    )

    if train_sequence is None or train_sequence.n == 0:
        raise ValueError("Training split contains no labeled samples")

    has_val = val_sequence is not None and val_sequence.n > 0
    monitor = "val_loss" if has_val else "loss"
    callbacks, checkpoint_template = training_utils.build_training_callbacks(
        checkpoint_dir=checkpoint_dir,
        monitor=monitor,
        patience=int(max(1, patience)),
        lr_factor=float(lr_factor),
        min_lr=float(min_lr),
        save_best_only=True,
        use_mixed_precision=bool(mixed_precision),
    )

    model = _build_baseline_regressor(float(learning_rate))

    fit_kwargs = _build_fit_kwargs(
        train_sequence,
        epochs=int(epochs),
        callbacks=callbacks,
        val_sequence=val_sequence,
    )

    history = model.fit(**fit_kwargs)

    history_serializable = _serialize_history_dict([history.history])
    artifact_paths = _save_training_artifacts(
        model=model,
        history_dict=history_serializable,
        output_dir=output_dir,
    )

    selected_metric = "val_loss" if "val_loss" in history_serializable else "loss"
    best_loss = float(np.min(history_serializable[selected_metric]))

    latest_checkpoint = training_utils.latest_checkpoint_path(checkpoint_dir)
    resume_epoch = training_utils.infer_initial_epoch_from_checkpoint(latest_checkpoint)

    return {
        "trained": True,
        "trainer": "baseline-cnn",
        "labels_path": os.path.abspath(str(labels_path)),
        "epochs": int(epochs),
        "batch_size": int(max(1, batch_size)),
        "learning_rate": float(learning_rate),
        "monitor": selected_metric,
        "best_loss": best_loss,
        "checkpoint_dir": os.path.abspath(str(checkpoint_dir)),
        "checkpoint_template": str(checkpoint_template),
        "latest_checkpoint": latest_checkpoint,
        "resume_epoch": int(resume_epoch),
        "model_path": artifact_paths["model_path"],
        "history_path": artifact_paths["history_path"],
        "labeled_counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }


def run_supervised_xception_finetune_training(
    *,
    split: Dict[str, pd.DataFrame],
    labels_path: str,
    output_dir: str,
    species: str,
    preprocessing_options: dict,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    finetune_learning_rate: float,
    freeze_base_epochs: int,
    xception_init: str,
    checkpoint_dir: str,
    patience: int,
    lr_factor: float,
    min_lr: float,
    mixed_precision: bool,
) -> Dict[str, object]:
    if int(epochs) <= 0:
        raise ValueError("epochs must be greater than zero for supervised training")

    labels_df = _load_labels_dataframe(labels_path)
    train_df = _attach_labels_to_split(split["train"], labels_df)
    val_df = _attach_labels_to_split(split["val"], labels_df)
    test_df = _attach_labels_to_split(split["test"], labels_df)

    train_sequence = _build_labeled_sequence(
        train_df,
        preprocessing_options=preprocessing_options,
        batch_size=batch_size,
    )
    val_sequence = _build_labeled_sequence(
        val_df,
        preprocessing_options=preprocessing_options,
        batch_size=batch_size,
    )

    if train_sequence is None or train_sequence.n == 0:
        raise ValueError("Training split contains no labeled samples")

    has_val = val_sequence is not None and val_sequence.n > 0
    monitor = "val_loss" if has_val else "loss"
    model = _build_xception_regressor(species=str(species), xception_init=xception_init)

    total_epochs = int(max(1, epochs))
    warmup_epochs = int(np.clip(int(freeze_base_epochs), 0, total_epochs))
    remaining_epochs = int(max(0, total_epochs - warmup_epochs))

    phase_history: List[Dict[str, Sequence[float]]] = []
    checkpoint_template = ""
    backbone_name = ""

    if warmup_epochs > 0:
        backbone_name = _set_xception_backbone_trainable(model, trainable=False)
        _compile_regression_model(model, learning_rate=float(learning_rate))
        warmup_callbacks, checkpoint_template = training_utils.build_training_callbacks(
            checkpoint_dir=checkpoint_dir,
            monitor=monitor,
            patience=int(max(1, patience)),
            lr_factor=float(lr_factor),
            min_lr=float(min_lr),
            save_best_only=True,
            use_mixed_precision=bool(mixed_precision),
        )
        warmup_fit_kwargs = _build_fit_kwargs(
            train_sequence,
            epochs=warmup_epochs,
            callbacks=warmup_callbacks,
            val_sequence=val_sequence,
            initial_epoch=0,
        )
        warmup_history = model.fit(**warmup_fit_kwargs)
        phase_history.append(warmup_history.history)

    if remaining_epochs > 0:
        backbone_name = _set_xception_backbone_trainable(model, trainable=True)
        _compile_regression_model(model, learning_rate=float(finetune_learning_rate))
        finetune_callbacks, checkpoint_template = training_utils.build_training_callbacks(
            checkpoint_dir=checkpoint_dir,
            monitor=monitor,
            patience=int(max(1, patience)),
            lr_factor=float(lr_factor),
            min_lr=float(min_lr),
            save_best_only=True,
            use_mixed_precision=bool(mixed_precision),
        )
        finetune_fit_kwargs = _build_fit_kwargs(
            train_sequence,
            epochs=total_epochs,
            callbacks=finetune_callbacks,
            val_sequence=val_sequence,
            initial_epoch=warmup_epochs,
        )
        finetune_history = model.fit(**finetune_fit_kwargs)
        phase_history.append(finetune_history.history)

    history_serializable = _serialize_history_dict(phase_history)
    artifact_paths = _save_training_artifacts(
        model=model,
        history_dict=history_serializable,
        output_dir=output_dir,
    )

    selected_metric = "val_loss" if "val_loss" in history_serializable else "loss"
    best_loss = float(np.min(history_serializable[selected_metric]))

    latest_checkpoint = training_utils.latest_checkpoint_path(checkpoint_dir)
    resume_epoch = training_utils.infer_initial_epoch_from_checkpoint(latest_checkpoint)

    return {
        "trained": True,
        "trainer": "xception-finetune",
        "labels_path": os.path.abspath(str(labels_path)),
        "species": str(species),
        "xception_init": _normalize_xception_init_mode(xception_init),
        "epochs": int(total_epochs),
        "phase_epochs": {
            "frozen_backbone": int(warmup_epochs),
            "finetune": int(remaining_epochs),
        },
        "backbone_name": str(backbone_name),
        "batch_size": int(max(1, batch_size)),
        "learning_rate": float(learning_rate),
        "finetune_learning_rate": float(finetune_learning_rate),
        "monitor": selected_metric,
        "best_loss": best_loss,
        "checkpoint_dir": os.path.abspath(str(checkpoint_dir)),
        "checkpoint_template": str(checkpoint_template),
        "latest_checkpoint": latest_checkpoint,
        "resume_epoch": int(resume_epoch),
        "model_path": artifact_paths["model_path"],
        "history_path": artifact_paths["history_path"],
        "labeled_counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }


def run_supervised_training(
    *,
    trainer: str,
    split: Dict[str, pd.DataFrame],
    labels_path: str,
    output_dir: str,
    species: str,
    preprocessing_options: dict,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    checkpoint_dir: str,
    patience: int,
    lr_factor: float,
    min_lr: float,
    mixed_precision: bool,
    finetune_learning_rate: float,
    freeze_base_epochs: int,
    xception_init: str,
) -> Dict[str, object]:
    trainer = _normalize_trainer_mode(trainer)
    if trainer == "baseline-cnn":
        return run_supervised_baseline_training(
            split=split,
            labels_path=labels_path,
            output_dir=output_dir,
            preprocessing_options=preprocessing_options,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
            patience=patience,
            lr_factor=lr_factor,
            min_lr=min_lr,
            mixed_precision=mixed_precision,
        )

    return run_supervised_xception_finetune_training(
        split=split,
        labels_path=labels_path,
        output_dir=output_dir,
        species=species,
        preprocessing_options=preprocessing_options,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        finetune_learning_rate=finetune_learning_rate,
        freeze_base_epochs=freeze_base_epochs,
        xception_init=xception_init,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        lr_factor=lr_factor,
        min_lr=min_lr,
        mixed_precision=mixed_precision,
    )


def _bounded_float(name: str, lo: float, hi: float):
    """Argparse type that enforces a closed float interval [lo, hi] up-front."""

    def _parse(value: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"{name} must be a number, got {value!r}")
        if not (lo <= number <= hi):
            raise argparse.ArgumentTypeError(
                f"{name} must be in [{lo}, {hi}], got {number}"
            )
        return number

    return _parse


def _positive_float(name: str):
    """Argparse type that enforces a strictly positive float."""

    def _parse(value: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"{name} must be a number, got {value!r}")
        if number <= 0:
            raise argparse.ArgumentTypeError(
                f"{name} must be > 0, got {number}"
            )
        return number


    return _parse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepSlice training workflow CLI: grouped split preparation and optional supervised training."
    )
    parser.add_argument("--images", required=True, help="Directory containing input training images")
    parser.add_argument("--output-dir", required=True, help="Output directory for manifests and artifacts")
    parser.add_argument("--species", choices=["mouse", "rat"], default="mouse")
    parser.add_argument("--group-mode", choices=sorted(GROUP_MODES), default="parent-folder")
    parser.add_argument(
        "--train-fraction",
        type=_bounded_float("--train-fraction", 0.0, 1.0),
        default=0.70,
    )
    parser.add_argument(
        "--val-fraction",
        type=_bounded_float("--val-fraction", 0.0, 1.0),
        default=0.15,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non-recursive", action="store_true", help="Only scan top-level image directory")
    parser.add_argument("--split-manifest-name", default="training_split_manifest.json")
    parser.add_argument("--metadata-name", default="training_run_metadata.json")

    parser.add_argument("--labels", default=None, help="CSV/TSV/JSON labels file containing target coordinates")
    parser.add_argument("--trainer", choices=sorted(TRAINER_MODES), default="baseline-cnn")
    parser.add_argument("--epochs", type=int, default=0, help="Run supervised training when > 0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--learning-rate",
        type=_positive_float("--learning-rate"),
        default=1e-3,
    )
    parser.add_argument(
        "--finetune-learning-rate",
        type=_positive_float("--finetune-learning-rate"),
        default=1e-4,
    )
    parser.add_argument("--freeze-base-epochs", type=int, default=1)
    parser.add_argument("--xception-init", choices=sorted(XCEPTION_INIT_MODES), default="species-primary")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--lr-factor",
        type=_bounded_float("--lr-factor", 0.0, 1.0),
        default=0.50,
    )
    parser.add_argument(
        "--min-lr",
        type=_positive_float("--min-lr"),
        default=1e-6,
    )
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Generate manifests only")

    parser.add_argument("--no-tissue-crop", action="store_true")
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--stain-normalization", choices=["none", "reinhard"], default="none")
    parser.add_argument("--preserve-color", action="store_true")
    parser.add_argument("--no-percentile-normalization", action="store_true")
    parser.add_argument("--bilateral-denoise", action="store_true")
    parser.add_argument(
        "--gamma",
        type=_bounded_float("--gamma", 0.5, 2.0),
        default=1.0,
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    preprocessing_options = {
        "tissue_crop": not bool(args.no_tissue_crop),
        "clahe": bool(args.clahe),
        "stain_normalization": str(args.stain_normalization),
        "preserve_color": bool(args.preserve_color),
        "percentile_normalization": not bool(args.no_percentile_normalization),
        "bilateral_denoise": bool(args.bilateral_denoise),
        # _build_parser already enforces gamma in [0.5, 2.0].
        "gamma": float(args.gamma),
    }
    checkpoint_dir = _resolve_checkpoint_dir(args.output_dir, args.checkpoint_dir)
    training_options = {
        "seed": int(args.seed),
        "group_mode": str(args.group_mode),
        "trainer": _normalize_trainer_mode(str(args.trainer)),
        "train_fraction": float(args.train_fraction),
        "val_fraction": float(args.val_fraction),
        "batch_size": int(max(1, args.batch_size)),
        "epochs": int(max(0, args.epochs)),
        "learning_rate": float(args.learning_rate),
        "finetune_learning_rate": float(args.finetune_learning_rate),
        "freeze_base_epochs": int(max(0, args.freeze_base_epochs)),
        "xception_init": _normalize_xception_init_mode(str(args.xception_init)),
        "patience": int(max(1, args.patience)),
        "lr_factor": float(args.lr_factor),
        "min_lr": float(args.min_lr),
        "mixed_precision": bool(args.mixed_precision),
        "checkpoint_dir": checkpoint_dir,
    }

    try:
        prepared = prepare_training_run(
            image_root=args.images,
            output_dir=args.output_dir,
            species=args.species,
            group_mode=args.group_mode,
            seed=int(args.seed),
            train_fraction=float(args.train_fraction),
            val_fraction=float(args.val_fraction),
            recursive=not bool(args.non_recursive),
            preprocessing_options=preprocessing_options,
            training_options=training_options,
            split_manifest_name=str(args.split_manifest_name),
            metadata_name=str(args.metadata_name),
        )

        split_manifest = prepared["split_manifest"]
        metadata = dict(prepared["metadata"])

        print(
            "Prepared grouped split: "
            f"train={split_manifest['counts']['train']}, "
            f"val={split_manifest['counts']['val']}, "
            f"test={split_manifest['counts']['test']}"
        )
        print(f"Split manifest: {prepared['split_manifest_path']}")
        print(f"Metadata: {prepared['metadata_path']}")

        training_requested = int(args.epochs) > 0
        metadata["training_requested"] = bool(training_requested)
        metadata["dry_run"] = bool(args.dry_run)

        if bool(args.dry_run):
            training_utils.save_training_run_metadata(prepared["metadata_path"], metadata)
            print("Dry run complete; skipping model fitting.")
            return 0

        if not training_requested:
            training_utils.save_training_run_metadata(prepared["metadata_path"], metadata)
            print("No training run requested (--epochs is 0).")
            return 0

        if not args.labels:
            raise ValueError("--labels is required when --epochs is greater than 0")

        train_count = int(split_manifest["counts"]["train"])
        requested_batch_size = int(max(1, args.batch_size))
        effective_batch_size = requested_batch_size
        if train_count > 0 and requested_batch_size > train_count:
            _logger.warning(
                "Requested --batch-size %s exceeds training-set size %s; "
                "clamping to %s for this run.",
                requested_batch_size,
                train_count,
                train_count,
            )
            effective_batch_size = train_count
        metadata["effective_batch_size"] = effective_batch_size

        training_result = run_supervised_training(
            trainer=str(args.trainer),
            split=prepared["split"],
            labels_path=str(args.labels),
            output_dir=str(args.output_dir),
            species=str(args.species),
            preprocessing_options=preprocessing_options,
            batch_size=effective_batch_size,
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            checkpoint_dir=checkpoint_dir,
            patience=int(max(1, args.patience)),
            lr_factor=float(args.lr_factor),
            min_lr=float(args.min_lr),
            mixed_precision=bool(args.mixed_precision),
            finetune_learning_rate=float(args.finetune_learning_rate),
            freeze_base_epochs=int(max(0, args.freeze_base_epochs)),
            xception_init=str(args.xception_init),
        )

        metadata["training_result"] = training_result
        training_utils.save_training_run_metadata(prepared["metadata_path"], metadata)

        print(
            "Training completed: "
            f"best_{training_result['monitor']}={training_result['best_loss']:.6f}"
        )
        print(f"Model: {training_result['model_path']}")
        print(f"History: {training_result['history_path']}")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
