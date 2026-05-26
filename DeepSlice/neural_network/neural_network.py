from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.utils import Sequence
from glob import glob
from typing import Any
import pandas as pd
import numpy as np
import os
import logging
from skimage import color, exposure, filters, measure, morphology, restoration, transform
from skimage.color import rgb2gray
import warnings
import h5py
from PIL import Image
from ..diagnostics import monitored

_logger = logging.getLogger(__name__)


VALID_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# Input shape expected by the Xception backbone (height, width, channels).
XCEPTION_INPUT_SIZE = (299, 299, 3)

# Full A-P depth of each reference atlas volume in voxels. Loaded lazily from
# metadata/config.json so the source of truth lives in one place; the literal
# dict below is a fallback when config lookup fails (e.g. during unit tests).
ATLAS_DEPTH: dict = {"mouse": 528.0, "rat": 1024.0}


def _resolve_atlas_depth(species: str) -> float:
    try:
        from ..metadata import metadata_loader
        low, high = metadata_loader.get_species_depth_range(str(species).strip().lower())
        return float(max(1.0, high - low))
    except Exception:
        return float(ATLAS_DEPTH.get(str(species).strip().lower(), 528.0))

DEFAULT_PREPROCESSING_OPTIONS = {
    "tissue_crop": True,
    "clahe": False,
    "stain_normalization": "none",
    "mask_background_norm": True,
    "gamma": 1.0,
    "bilateral_denoise": False,
    "percentile_normalization": True,
    "preserve_color": False,
    "min_resolution_px": 500,
    "blur_variance_threshold": 0.0002,
    "dark_intensity_threshold": 0.12,
    "bright_intensity_threshold": 0.93,
    "saturated_fraction_threshold": 0.20,
    "artifact_blank_fraction_threshold": 0.18,
}


def _coerce_preprocessing_options(preprocessing_options=None):
    options = dict(DEFAULT_PREPROCESSING_OPTIONS)
    if isinstance(preprocessing_options, dict):
        options.update(preprocessing_options)
    options["stain_normalization"] = str(options.get("stain_normalization", "none")).strip().lower()
    options["gamma"] = float(np.clip(float(options.get("gamma", 1.0)), 0.5, 2.0))
    return options


def _load_image_rgb(image_path: str) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _estimate_tissue_mask(gray: np.ndarray) -> np.ndarray:
    gray = np.nan_to_num(np.asarray(gray, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if gray.ndim != 2:
        raise ValueError("Tissue-mask estimation expects a grayscale 2D image")

    try:
        threshold = float(filters.threshold_otsu(gray))
    except Exception:
        return np.ones_like(gray, dtype=bool)

    low_mask = gray < threshold
    high_mask = gray > threshold

    candidates = []
    for mask in (low_mask, high_mask):
        fraction = float(np.mean(mask))
        if 0.005 <= fraction <= 0.995:
            candidates.append((abs(fraction - 0.35), mask))

    if len(candidates) == 0:
        mask = low_mask if float(np.mean(low_mask)) < float(np.mean(high_mask)) else high_mask
    else:
        mask = sorted(candidates, key=lambda item: item[0])[0][1]

    min_size = int(max(64, round(mask.size * 0.0005)))
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    if not np.any(mask):
        return np.ones_like(gray, dtype=bool)
    mask = morphology.closing(mask, morphology.disk(3))
    return mask


def _crop_to_tissue_bbox(image: np.ndarray, mask: np.ndarray, pad_fraction: float = 0.03):
    if not np.any(mask):
        return image, mask

    rows, cols = np.where(mask)
    y0, y1 = int(np.min(rows)), int(np.max(rows) + 1)
    x0, x1 = int(np.min(cols)), int(np.max(cols) + 1)

    pad_y = int(round((y1 - y0) * pad_fraction))
    pad_x = int(round((x1 - x0) * pad_fraction))

    y0 = max(0, y0 - pad_y)
    y1 = min(image.shape[0], y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(image.shape[1], x1 + pad_x)

    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _apply_scale_variant(image: np.ndarray, scale_factor: float) -> np.ndarray:
    scale_factor = float(scale_factor)
    if np.isclose(scale_factor, 1.0):
        return image

    h, w = image.shape[:2]
    if scale_factor < 1.0:
        crop_h = int(max(8, round(h * scale_factor)))
        crop_w = int(max(8, round(w * scale_factor)))
        y0 = (h - crop_h) // 2
        x0 = (w - crop_w) // 2
        cropped = image[y0:y0 + crop_h, x0:x0 + crop_w]
        return transform.resize(
            cropped,
            (h, w, image.shape[2]),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)

    pad_h = int(round((scale_factor - 1.0) * h * 0.5))
    pad_w = int(round((scale_factor - 1.0) * w * 0.5))
    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode="reflect",
    )
    return transform.resize(
        padded,
        (h, w, image.shape[2]),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def _apply_reinhard_normalization(image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    lab = color.rgb2lab(np.clip(image, 0.0, 1.0))
    target_mean = np.array([62.0, 0.0, 0.0], dtype=np.float32)
    target_std = np.array([18.0, 9.0, 9.0], dtype=np.float32)

    if mask is None or not np.any(mask):
        source = lab.reshape(-1, 3)
    else:
        source = lab[mask]
    source_mean = np.mean(source, axis=0)
    source_std = np.std(source, axis=0)
    source_std = np.where(source_std < 1e-6, 1.0, source_std)

    normalized_lab = ((lab - source_mean) / source_std) * target_std + target_mean
    rgb = color.lab2rgb(normalized_lab)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _percentile_normalize(image: np.ndarray, mask: np.ndarray = None, low_q: float = 1.0, high_q: float = 99.0):
    if mask is None or not np.any(mask):
        reference = image.reshape(-1, image.shape[-1])
    else:
        reference = image[mask]

    low = np.percentile(reference, low_q, axis=0)
    high = np.percentile(reference, high_q, axis=0)
    span = np.where((high - low) < 1e-6, 1.0, high - low)
    normalized = (image - low) / span
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _flip_image(image: np.ndarray, flip_mode: str):
    key = str(flip_mode or "none").strip().lower()
    if key == "h":
        return np.flip(image, axis=1)
    if key == "v":
        return np.flip(image, axis=0)
    if key in {"hv", "vh", "both"}:
        return np.flip(np.flip(image, axis=1), axis=0)
    return image


def _apply_pixel_dropout(image: np.ndarray, dropout_fraction: float, rng) -> np.ndarray:
    """Inverted pixel dropout: zero a fraction of pixels and rescale the rest
    by 1/(1 - frac) to preserve expected intensity. Matches the standard
    "dropout" formulation used during training; without rescaling the model
    sees a distribution-shifted input."""
    frac = float(np.clip(dropout_fraction, 0.0, 0.95))
    if frac <= 0.0:
        return image
    keep_mask = rng.random(image.shape[:2]) >= frac
    scale = 1.0 / max(1e-6, 1.0 - frac)
    output = np.array(image, copy=True)
    output *= keep_mask[..., None].astype(image.dtype) * scale
    return output


def _compute_quality_metrics(gray: np.ndarray, width: int, height: int, options: dict):
    gray = np.nan_to_num(np.asarray(gray, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lap_var = float(np.var(filters.laplace(gray)))
    mean_intensity = float(np.mean(gray))
    saturated_fraction = float(np.mean((gray <= 0.01) | (gray >= 0.99)))

    blank_mask = gray > 0.985
    largest_blank_fraction = 0.0
    if np.any(blank_mask):
        labels = measure.label(blank_mask)
        if labels.max() > 0:
            counts = np.bincount(labels.ravel())
            if len(counts) > 1:
                largest_blank_fraction = float(np.max(counts[1:]) / gray.size)

    artifact_score = float(np.var(filters.sobel(gray)))

    flags = {
        "low_resolution": bool(min(int(width), int(height)) < int(options["min_resolution_px"])),
        "blurry": bool(lap_var < float(options["blur_variance_threshold"])),
        "underexposed": bool(mean_intensity < float(options["dark_intensity_threshold"])),
        "overexposed": bool(mean_intensity > float(options["bright_intensity_threshold"])),
        "saturated": bool(saturated_fraction > float(options["saturated_fraction_threshold"])),
        "artifact": bool(largest_blank_fraction > float(options["artifact_blank_fraction_threshold"])),
    }

    return {
        "width": int(width),
        "height": int(height),
        "blur_variance": lap_var,
        "mean_intensity": mean_intensity,
        "saturated_fraction": saturated_fraction,
        "largest_blank_fraction": largest_blank_fraction,
        "artifact_gradient_var": artifact_score,
        "flags": flags,
    }


def _preprocess_image_array(
    image_rgb: np.ndarray,
    options: dict,
    scale_factor: float = 1.0,
    flip_mode: str = "none",
    dropout_fraction: float = 0.0,
    rng=None,
):
    rgb = np.clip(np.asarray(image_rgb, dtype=np.float32), 0.0, 1.0)
    gray = rgb2gray(rgb)
    tissue_mask = _estimate_tissue_mask(gray)
    tissue_fraction = float(np.mean(tissue_mask))

    if bool(options.get("tissue_crop", True)):
        rgb, tissue_mask = _crop_to_tissue_bbox(rgb, tissue_mask)
        gray = rgb2gray(rgb)

    if not np.isclose(scale_factor, 1.0):
        rgb = _apply_scale_variant(rgb, float(scale_factor))
        gray = rgb2gray(rgb)
        tissue_mask = _estimate_tissue_mask(gray)

    stain_mode = str(options.get("stain_normalization", "none")).strip().lower()
    if stain_mode == "reinhard":
        rgb = _apply_reinhard_normalization(rgb, tissue_mask)
        gray = rgb2gray(rgb)

    if bool(options.get("clahe", False)):
        if bool(options.get("preserve_color", False)):
            lab = color.rgb2lab(np.clip(rgb, 0.0, 1.0))
            l_channel = np.clip(lab[:, :, 0] / 100.0, 0.0, 1.0)
            l_channel = exposure.equalize_adapthist(l_channel, clip_limit=0.03)
            lab[:, :, 0] = l_channel * 100.0
            rgb = np.clip(color.lab2rgb(lab), 0.0, 1.0).astype(np.float32)
        else:
            gray = exposure.equalize_adapthist(gray, clip_limit=0.03)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
        gray = rgb2gray(rgb)

    if bool(options.get("bilateral_denoise", False)):
        rgb = restoration.denoise_bilateral(
            np.clip(rgb, 0.0, 1.0),
            sigma_color=0.08,
            sigma_spatial=3,
            channel_axis=-1,
        ).astype(np.float32)
        gray = rgb2gray(rgb)

    gamma_value = float(options.get("gamma", 1.0))
    if not np.isclose(gamma_value, 1.0):
        rgb = exposure.adjust_gamma(np.clip(rgb, 0.0, 1.0), gamma=gamma_value).astype(np.float32)
        gray = rgb2gray(rgb)

    if bool(options.get("percentile_normalization", True)):
        rgb = _percentile_normalize(rgb, tissue_mask)
        gray = rgb2gray(rgb)

    if not bool(options.get("preserve_color", False)):
        rgb = np.repeat(gray[:, :, None], 3, axis=2)

    rgb = _flip_image(rgb, flip_mode)
    if dropout_fraction > 0.0 and rng is not None:
        rgb = _apply_pixel_dropout(rgb, dropout_fraction, rng)

    resized = transform.resize(
        rgb,
        XCEPTION_INPUT_SIZE,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    resized = np.clip(resized, 0.0, 1.0)

    if bool(options.get("mask_background_norm", True)):
        resized_mask = transform.resize(
            tissue_mask.astype(np.float32),
            XCEPTION_INPUT_SIZE[:2],
            order=0,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        ) > 0.5
        reference_pixels = resized[resized_mask] if np.any(resized_mask) else resized.reshape(-1, 3)
    else:
        reference_pixels = resized.reshape(-1, 3)

    mean = np.mean(reference_pixels, axis=0, keepdims=True)
    std = np.std(reference_pixels, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (resized - mean) / std
    normalized = np.nan_to_num(normalized.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    preview = normalized.copy()
    preview -= float(np.min(preview))
    max_val = float(np.max(preview))
    if max_val > 1e-6:
        preview /= max_val
    preview_uint8 = np.clip(preview * 255.0, 0, 255).astype(np.uint8)

    return normalized, preview_uint8, {
        "tissue_fraction": tissue_fraction,
    }


def inspect_image_quality(image_path: str, preprocessing_options=None):
    options = _coerce_preprocessing_options(preprocessing_options)
    rgb = _load_image_rgb(image_path)
    h, w = rgb.shape[:2]
    gray = rgb2gray(rgb)
    metrics = _compute_quality_metrics(gray, width=w, height=h, options=options)
    metrics["path"] = str(image_path)
    return metrics


def inspect_image_batch(image_paths: list, preprocessing_options=None):
    options = _coerce_preprocessing_options(preprocessing_options)
    report = {
        "total": len(image_paths),
        "resolution_mismatch": False,
        "issue_count": 0,
        "issue_paths": [],
        "metrics": [],
        "counts": {
            "low_resolution": 0,
            "blurry": 0,
            "underexposed": 0,
            "overexposed": 0,
            "saturated": 0,
            "artifact": 0,
        },
    }
    dimensions = set()

    for image_path in image_paths:
        try:
            metrics = inspect_image_quality(image_path, preprocessing_options=options)
        except Exception as _exc:
            _logger.warning("Skipping image %s during batch inspection: %s", image_path, _exc)
            continue
        report["metrics"].append(metrics)
        dimensions.add((metrics["width"], metrics["height"]))

        flagged = [name for name, value in metrics["flags"].items() if bool(value)]
        if len(flagged) > 0:
            report["issue_count"] += 1
            report["issue_paths"].append({"path": str(image_path), "flags": flagged})
            for flag_name in flagged:
                if flag_name in report["counts"]:
                    report["counts"][flag_name] += 1

    report["resolution_mismatch"] = len(dimensions) > 1
    report["dimensions"] = sorted(list(dimensions))
    return report


def preview_preprocessed_image(
    image_path: str,
    preprocessing_options=None,
    scale_factor: float = 1.0,
    flip_mode: str = "none",
):
    options = _coerce_preprocessing_options(preprocessing_options)
    rgb = _load_image_rgb(image_path)
    model_input, preview_uint8, extra = _preprocess_image_array(
        rgb,
        options=options,
        scale_factor=scale_factor,
        flip_mode=flip_mode,
        dropout_fraction=0.0,
        rng=None,
    )
    quality = inspect_image_quality(image_path, preprocessing_options=options)
    quality.update(extra)
    quality["preview_image"] = preview_uint8
    quality["model_input"] = model_input
    return quality


class PreprocessedImageSequence(Sequence):
    def __init__(
        self,
        image_paths: list,
        batch_size: int = 16,
        preprocessing_options=None,
        scale_factor: float = 1.0,
        flip_mode: str = "none",
        dropout_fraction: float = 0.0,
        dropout_seed_offset: int = 0,
        output_dtype: str = "float32",
    ):
        self.paths = list(image_paths)
        self.n = len(self.paths)
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        self.batch_size = int(batch_size)
        self.preprocessing_options = _coerce_preprocessing_options(preprocessing_options)
        self.scale_factor = float(scale_factor)
        self.flip_mode = str(flip_mode or "none").strip().lower()
        self.dropout_fraction = float(max(0.0, dropout_fraction))
        self.dropout_seed_offset = int(dropout_seed_offset)
        self.output_dtype = str(output_dtype)
        self.filenames = [os.path.basename(path) for path in self.paths]
        self.filepaths = list(self.paths)
        sizes = [get_image_size(path) for path in self.paths]
        self.deepslice_width = [size[0] for size in sizes]
        self.deepslice_height = [size[1] for size in sizes]
        self.deepslice_paths = list(self.paths)

    def __len__(self):
        if self.n == 0:
            return 0
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, index):
        if self.n == 0:
            return np.zeros((0,) + XCEPTION_INPUT_SIZE, dtype=np.float32)

        start = int(index) * self.batch_size
        end = min(start + self.batch_size, self.n)
        batch_paths = self.paths[start:end]
        batch_images = []

        for local_idx, image_path in enumerate(batch_paths):
            rgb = _load_image_rgb(image_path)
            # SeedSequence mixes (base seed, pass offset, sample index) so each
            # cloned pass (TTA/dropout) gets a distinct, reproducible stream.
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    entropy=42,
                    spawn_key=(int(self.dropout_seed_offset), int(start + local_idx)),
                )
            )
            processed, _, _ = _preprocess_image_array(
                rgb,
                options=self.preprocessing_options,
                scale_factor=self.scale_factor,
                flip_mode=self.flip_mode,
                dropout_fraction=self.dropout_fraction,
                rng=rng,
            )
            batch_images.append(processed)

        batch = np.asarray(batch_images, dtype=np.float32)
        if self.output_dtype == "float16":
            return batch.astype(np.float16)
        return batch

    def reset(self):
        return

    def clone_with(
        self,
        scale_factor: float = None,
        flip_mode: str = None,
        dropout_fraction: float = None,
        dropout_seed_offset: int = None,
        output_dtype: str = None,
    ):
        return PreprocessedImageSequence(
            self.paths,
            batch_size=self.batch_size,
            preprocessing_options=self.preprocessing_options,
            scale_factor=self.scale_factor if scale_factor is None else scale_factor,
            flip_mode=self.flip_mode if flip_mode is None else flip_mode,
            dropout_fraction=self.dropout_fraction if dropout_fraction is None else dropout_fraction,
            dropout_seed_offset=self.dropout_seed_offset if dropout_seed_offset is None else dropout_seed_offset,
            output_dtype=self.output_dtype if output_dtype is None else output_dtype,
        )


class EnsemblePartialResultError(RuntimeError):
    """Raised when ensemble secondary inference fails but primary predictions are available."""

    def __init__(self, message: str, partial_predictions: pd.DataFrame):
        super().__init__(message)
        self.partial_predictions = partial_predictions


class PredictionProgressCallback(tf.keras.callbacks.Callback):
    """Keras callback used to expose prediction progress to the GUI."""

    def __init__(self, total_images, phase, progress_callback, cancel_check=None, batch_size: int = 1):
        super().__init__()
        self.total_images = total_images
        self.phase = phase
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.batch_size = max(int(batch_size), 1)
        self._last_completed = 0

    def _raise_if_cancelled(self):
        if self.cancel_check is not None and bool(self.cancel_check()):
            raise RuntimeError("Prediction cancelled by user")

    def on_predict_batch_begin(self, batch, logs=None):
        self._raise_if_cancelled()

    def on_predict_batch_end(self, batch, logs=None):
        self._raise_if_cancelled()
        if self.progress_callback is None:
            return
        completed = min((batch + 1) * self.batch_size, self.total_images)
        if completed > self._last_completed:
            self.progress_callback(completed, self.total_images, self.phase)
            self._last_completed = completed


def gray_scale(img: np.ndarray) -> np.ndarray:
    """
    Convert the image to grayscale

    :param img: The image to convert
    :type img: numpy.ndarray
    :return: The converted image
    :rtype: numpy.ndarray
    """
    h, w = img.shape[:2]
    img = rgb2gray(img).reshape(h, w, 1)
    return img


@monitored("DS-006")
def initialise_network(xception_weights: str, weights: str, species: str) -> Sequential:
    """
    Initialise a neural network with the given weights.

    :param xception_weights: Path to the Xception ImageNet weights file
    :param weights: Path to the DeepSlice dense-layer weights file (loaded on top of Xception)
    :param species: Species of the animal; determines model architecture
    :return: The initialised neural network
    :rtype: keras.models.Sequential
    """
    base_model = Xception(include_top=True, weights=xception_weights)

    if species == "rat":
        inputs = Input(shape=XCEPTION_INPUT_SIZE)
        base_model_layer = base_model(inputs, training=False)
        dense1_layer = Dense(256, activation="relu")(base_model_layer)
        dense2_layer = Dense(256, activation="relu")(dense1_layer)
        output_layer = Dense(9, activation="linear")(dense2_layer)
        model = Model(inputs=inputs, outputs=output_layer)
    else:
        model = Sequential()
        model.add(base_model)
        model.add(Dense(256, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(9, activation="linear"))

    if weights is not None:
        model = load_xception_weights(model, weights, species)
    return model


def load_xception_weights(model, weights, species="mouse"):
    with h5py.File(weights, "r") as new:
        # set weight of each layer manually
        if species == "mouse":
            xception_idx = 0
            dense_idx = 1
        elif species == "rat":
            # RatModelInProgress.h5 has an "input_2" layer at index 0, so we need to adjust the indices<
            xception_idx = 1
            dense_idx = 2
        else:
            raise ValueError("species must be one of 'mouse' or 'rat'")

        dense_layers_updated = 0

        for layer_name, group_name in [
            ("dense", "dense"),
            ("dense_1", "dense_1"),
            ("dense_2", "dense_2"),
        ]:
            if layer_name not in new or group_name not in new[layer_name]:
                raise RuntimeError(
                    f"Expected h5 group '{layer_name}/{group_name}' not found in weights file"
                )

        model.layers[dense_idx].set_weights(
            [new["dense"]["dense"]["kernel:0"], new["dense"]["dense"]["bias:0"]]
        )
        dense_layers_updated += 1
        model.layers[dense_idx + 1].set_weights(
            [new["dense_1"]["dense_1"]["kernel:0"], new["dense_1"]["dense_1"]["bias:0"]]
        )
        dense_layers_updated += 1
        model.layers[dense_idx + 2].set_weights(
            [new["dense_2"]["dense_2"]["kernel:0"], new["dense_2"]["dense_2"]["bias:0"]]
        )
        dense_layers_updated += 1

        if dense_layers_updated != 3:
            raise RuntimeError(
                f"Expected to set 3 dense layers, but set {dense_layers_updated}"
            )

        # Set the weights of the xception model
        weight_names = new["xception"].attrs["weight_names"].tolist()
        weight_names_layers = {
            (name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name)).split("/")[0]
            for name in weight_names
        }
        updated_xception_layers = set()

        for i in range(len(model.layers[xception_idx].layers)):
            name_of_layer = model.layers[xception_idx].layers[i].name
            # if layer name is in the weight names, then we will set weights
            if name_of_layer in weight_names_layers:
                # Get name of weights in the layer
                layer_weight_names = []
                for weight in model.layers[xception_idx].layers[i].weights:
                    try:
                        layer_weight_names.append(weight.name.split("/")[1])
                    except IndexError:
                        layer_weight_names.append(f"{weight.name}:0")

                h5_group = new["xception"][name_of_layer]
                weights_list = [np.array(h5_group[kk]) for kk in layer_weight_names]
                model.layers[xception_idx].layers[i].set_weights(weights_list)
                updated_xception_layers.add(name_of_layer)

        if len(updated_xception_layers) != len(weight_names_layers):
            missing_layers = sorted(weight_names_layers - updated_xception_layers)
            missing_preview = ", ".join(missing_layers[:10])
            if len(missing_layers) > 10:
                missing_preview += f", ... (+{len(missing_layers) - 10} more)"
            raise RuntimeError(
                "Xception weight loading incomplete. "
                f"Updated {len(updated_xception_layers)}/{len(weight_names_layers)} layers. "
                f"Missing layers: {missing_preview}"
            )
    return model


def _create_image_generator(
    images: list,
    batch_size: int = 16,
    preprocessing_options=None,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    images = [i for i in images if os.path.splitext(i)[1].lower() in VALID_IMAGE_FORMATS]
    if len(images) == 0:
        raise ValueError(
            "No images found. Ensure files are one of: "
            + ", ".join(VALID_IMAGE_FORMATS)
        )

    # PreprocessedImageSequence.__init__ already computes per-image sizes via
    # get_image_size(); reuse those instead of opening every file a second time.
    image_generator = PreprocessedImageSequence(
        images,
        batch_size=batch_size,
        preprocessing_options=preprocessing_options,
    )
    width = list(image_generator.deepslice_width)
    height = list(image_generator.deepslice_height)
    return image_generator, width, height


def load_images_from_path(
    image_path: str,
    batch_size: int = 16,
    preprocessing_options=None,
) -> np.ndarray:
    """
    Load the images from the given path
    :param image_path: The path to the images
    :type image_path: str
    :return: an Image generator for the found images
    :rtype: keras.preprocessing.image.ImageDataGenerator
    """
    if image_path is None or not os.path.isdir(image_path):
        raise ValueError("The path provided is not a directory")
    images = glob(os.path.join(image_path, "*"))
    return _create_image_generator(
        images,
        batch_size=batch_size,
        preprocessing_options=preprocessing_options,
    )


def load_images_from_list(
    image_list: list,
    batch_size: int = 16,
    preprocessing_options=None,
) -> np.ndarray:
    """
    Load the images from the given list
    :param image_list: The list of images
    :type image_list: list
    :return: an Image generator for the found images
    :rtype: keras.preprocessing.image.ImageDataGenerator
    """
    if image_list is None:
        raise ValueError("image_list must not be None")
    return _create_image_generator(
        image_list,
        batch_size=batch_size,
        preprocessing_options=preprocessing_options,
    )


def _resolve_generator_metadata(image_generator: Any):
    source_paths = getattr(image_generator, "deepslice_paths", None)
    if source_paths is None or len(source_paths) != len(image_generator.filenames):
        source_paths = getattr(image_generator, "filepaths", None)
    if source_paths is None or len(source_paths) != len(image_generator.filenames):
        directory = getattr(image_generator, "directory", None)
        if directory:
            source_paths = [
                os.path.join(directory, relative_path)
                for relative_path in image_generator.filenames
            ]
        else:
            source_paths = list(image_generator.filenames)
    else:
        source_paths = list(source_paths)

    width = getattr(image_generator, "deepslice_width", None)
    height = getattr(image_generator, "deepslice_height", None)
    if (
        width is None
        or height is None
        or len(width) != len(source_paths)
        or len(height) != len(source_paths)
    ):
        sizes = [get_image_size(path) for path in source_paths]
        width = [size[0] for size in sizes]
        height = [size[1] for size in sizes]

    return source_paths, width, height


def _build_predictions_dataframe(
    image_generator: Any,
    predictions: np.ndarray,
    ensemble_delta_mean_abs: np.ndarray = None,
    ensemble_delta_max_abs: np.ndarray = None,
    tta_coordinate_variance: np.ndarray = None,
    tta_oy_std: np.ndarray = None,
    ensemble_coord_std: np.ndarray = None,
    model_disagreement: np.ndarray = None,
    ensemble_primary_weight: np.ndarray = None,
    primary_oy: np.ndarray = None,
    secondary_oy: np.ndarray = None,
) -> pd.DataFrame:
    source_paths, width, height = _resolve_generator_metadata(image_generator)
    filenames = [os.path.basename(path) for path in source_paths]
    dataframe = pd.DataFrame(
        {
            "Filenames": filenames,
            "width": width,
            "height": height,
            "ox": predictions[:, 0],
            "oy": predictions[:, 1],
            "oz": predictions[:, 2],
            "ux": predictions[:, 3],
            "uy": predictions[:, 4],
            "uz": predictions[:, 5],
            "vx": predictions[:, 6],
            "vy": predictions[:, 7],
            "vz": predictions[:, 8],
        }
    )

    if ensemble_delta_mean_abs is not None:
        dataframe["ensemble_delta_mean_abs"] = np.asarray(ensemble_delta_mean_abs, dtype=float)
    if ensemble_delta_max_abs is not None:
        dataframe["ensemble_delta_max_abs"] = np.asarray(ensemble_delta_max_abs, dtype=float)
    if tta_coordinate_variance is not None:
        dataframe["tta_coordinate_variance"] = np.asarray(tta_coordinate_variance, dtype=float)
    if tta_oy_std is not None:
        dataframe["tta_oy_std"] = np.asarray(tta_oy_std, dtype=float)
    if ensemble_coord_std is not None:
        dataframe["ensemble_coord_std"] = np.asarray(ensemble_coord_std, dtype=float)
    if model_disagreement is not None:
        dataframe["model_disagreement"] = np.asarray(model_disagreement, dtype=bool)
    if ensemble_primary_weight is not None:
        dataframe["ensemble_primary_weight"] = np.asarray(ensemble_primary_weight, dtype=float)
    if primary_oy is not None:
        dataframe["primary_oy"] = np.asarray(primary_oy, dtype=float)
    if secondary_oy is not None:
        dataframe["secondary_oy"] = np.asarray(secondary_oy, dtype=float)

    return dataframe


def _validate_prediction_matrix(predictions: np.ndarray, phase: str):
    matrix = np.asarray(predictions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 9:
        raise RuntimeError(
            f"{phase} inference produced invalid coordinate shape {matrix.shape}; expected (N, 9)"
        )
    if matrix.shape[0] == 0:
        raise RuntimeError(f"{phase} inference produced no predictions")
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"{phase} inference produced non-finite values (NaN/Inf)")
    return matrix


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return vectors / norms


def _combine_prediction_passes(pass_predictions: list):
    if len(pass_predictions) == 0:
        raise ValueError("No prediction passes were provided")
    if len(pass_predictions) == 1:
        n_rows = pass_predictions[0].shape[0]
        return pass_predictions[0], np.zeros(n_rows, dtype=float), np.zeros(n_rows, dtype=float)

    stack = np.stack(pass_predictions, axis=0)
    origins = np.mean(stack[:, :, 0:3], axis=0)

    u_vectors = _normalize_vectors(np.mean(np.array([_normalize_vectors(run[:, 3:6]) for run in pass_predictions]), axis=0))
    v_vectors = _normalize_vectors(np.mean(np.array([_normalize_vectors(run[:, 6:9]) for run in pass_predictions]), axis=0))

    combined = np.concatenate([origins, u_vectors, v_vectors], axis=1)

    coord_variance = np.var(stack, axis=0)
    tta_coordinate_variance = np.mean(coord_variance, axis=1)
    tta_oy_std = np.std(stack[:, :, 1], axis=0)
    return combined, tta_coordinate_variance, tta_oy_std


def _build_inference_pass_specs(tta: bool, multi_scale: bool, section_dropout_passes: int):
    flips = ["none"]
    if bool(tta):
        flips = ["none", "h", "v", "hv"]

    scales = [1.0]
    if bool(multi_scale):
        scales = [0.75, 1.0, 1.25]

    specs = []
    for scale in scales:
        for flip in flips:
            specs.append(
                {
                    "flip_mode": flip,
                    "scale_factor": float(scale),
                    "dropout_fraction": 0.0,
                    "dropout_seed_offset": 0,
                }
            )

    passes = int(max(0, section_dropout_passes))
    for dropout_idx in range(passes):
        specs.append(
            {
                "flip_mode": "none",
                "scale_factor": 1.0,
                "dropout_fraction": 0.10,
                "dropout_seed_offset": dropout_idx + 1,
            }
        )

    return specs


def _run_inference_passes(
    model,
    base_generator,
    phase_label: str,
    progress_callback=None,
    cancel_check=None,
    log_callback=None,
    pass_specs=None,
):
    if pass_specs is None or len(pass_specs) == 0:
        pass_specs = [
            {
                "flip_mode": "none",
                "scale_factor": 1.0,
                "dropout_fraction": 0.0,
                "dropout_seed_offset": 0,
            }
        ]

    pass_predictions = []
    total_passes = len(pass_specs)

    for pass_idx, spec in enumerate(pass_specs):
        if cancel_check is not None and bool(cancel_check()):
            raise RuntimeError("Prediction cancelled by user")

        if hasattr(base_generator, "clone_with"):
            generator = base_generator.clone_with(
                scale_factor=spec["scale_factor"],
                flip_mode=spec["flip_mode"],
                dropout_fraction=spec["dropout_fraction"],
                dropout_seed_offset=spec["dropout_seed_offset"],
            )
        else:
            generator = base_generator

        steps = len(generator)
        callbacks = None
        if progress_callback is not None and pass_idx == 0:
            callbacks = [
                PredictionProgressCallback(
                    total_images=generator.n,
                    phase=phase_label,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    batch_size=generator.batch_size,
                )
            ]

        if log_callback is not None:
            log_callback(
                f"Running {phase_label} inference pass {pass_idx + 1}/{total_passes} "
                f"(scale={spec['scale_factor']:.2f}, flip={spec['flip_mode']}, dropout={spec['dropout_fraction']:.2f})"
            )

        prediction = model.predict(
            generator,
            steps=steps,
            verbose=1,
            callbacks=callbacks,
        )
        pass_predictions.append(_validate_prediction_matrix(prediction, phase=f"{phase_label} pass"))

    combined, tta_coordinate_variance, tta_oy_std = _combine_prediction_passes(pass_predictions)
    return combined, tta_coordinate_variance, tta_oy_std


def _residual_weighted_blend(primary_predictions: np.ndarray, secondary_predictions: np.ndarray):
    n_rows = primary_predictions.shape[0]
    if n_rows == 0:
        return primary_predictions, np.zeros(0, dtype=float)

    x = np.arange(n_rows, dtype=float)

    def residuals(predictions):
        if n_rows < 2:
            return np.ones(n_rows, dtype=float)
        slope, intercept = np.polyfit(x, predictions[:, 1], 1)
        trend = slope * x + intercept
        return np.abs(predictions[:, 1] - trend) + 1e-6

    primary_residual = residuals(primary_predictions)
    secondary_residual = residuals(secondary_predictions)

    inv_primary = 1.0 / primary_residual
    inv_secondary = 1.0 / secondary_residual
    denom = inv_primary + inv_secondary
    denom = np.where(denom < 1e-9, 1.0, denom)

    primary_weight = inv_primary / denom
    secondary_weight = inv_secondary / denom

    blended = np.zeros_like(primary_predictions)
    blended[:, 0] = (primary_weight * primary_predictions[:, 0]) + (secondary_weight * secondary_predictions[:, 0])
    blended[:, 1] = (primary_weight * primary_predictions[:, 1]) + (secondary_weight * secondary_predictions[:, 1])
    blended[:, 2] = (primary_weight * primary_predictions[:, 2]) + (secondary_weight * secondary_predictions[:, 2])

    u_vec = (
        (primary_weight[:, None] * primary_predictions[:, 3:6])
        + (secondary_weight[:, None] * secondary_predictions[:, 3:6])
    )
    v_vec = (
        (primary_weight[:, None] * primary_predictions[:, 6:9])
        + (secondary_weight[:, None] * secondary_predictions[:, 6:9])
    )
    blended[:, 3:6] = _normalize_vectors(u_vec)
    blended[:, 6:9] = _normalize_vectors(v_vec)
    return blended, primary_weight


def predictions_util(
    model: Sequential,
    image_generator: Any,
    primary_weights: str,
    secondary_weights: str,
    ensemble: bool = False,
    species: str = "mouse",
    progress_callback=None,
    log_callback=None,
    cancel_check=None,
    tta: bool = False,
    multi_scale: bool = False,
    fast_fp16: bool = False,
    section_dropout_passes: int = 0,
    confidence_weighted_ensemble: bool = True,
    ensemble_consistency_threshold: float = 0.20,
):
    """
    Predict the image alignments

    :param model: The model to use for prediction
    :param image_generator: The image generator to use for prediction
    :type model: keras.models.Sequential
    :type image_generator: keras.preprocessing.image.ImageDataGenerator
    :return: The predicted alignments
    :rtype: list
    """
    # Do NOT call np.random.seed() / tf.random.set_seed() at module scope: that
    # leaks determinism into unrelated code in the same process. Determinism
    # for this prediction run is provided by the per-sample seed scheme inside
    # PreprocessedImageSequence.__getitem__ (SeedSequence-backed).

    if fast_fp16 and hasattr(image_generator, "clone_with"):
        image_generator = image_generator.clone_with(output_dtype="float16")
        if log_callback is not None:
            log_callback("Fast inference mode enabled (float16 inputs)")

    model = load_xception_weights(model, primary_weights, species)

    pass_specs = _build_inference_pass_specs(
        tta=tta,
        multi_scale=multi_scale,
        section_dropout_passes=section_dropout_passes,
    )

    if cancel_check is not None and bool(cancel_check()):
        raise RuntimeError("Prediction cancelled by user")

    if log_callback is not None:
        log_callback("Running model warm-up pass")
    try:
        warmup_batch = image_generator[0]
        if isinstance(warmup_batch, tuple):
            warmup_batch = warmup_batch[0]
        if warmup_batch is not None and len(warmup_batch) > 0:
            model.predict(warmup_batch[:1], verbose=0)
    except Exception as _warmup_exc:
        # Warmup is best-effort only and should never block inference.
        _logger.warning("Model warmup pass failed (non-fatal): %s", _warmup_exc)

    try:
        predictions, tta_coordinate_variance, tta_oy_std = _run_inference_passes(
            model,
            image_generator,
            phase_label="primary",
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            log_callback=log_callback,
            pass_specs=pass_specs,
        )
    except Exception as exc:
        raise RuntimeError(
            "Primary inference failed. Check image integrity, TensorFlow runtime, and available memory."
        ) from exc

    predictions = _validate_prediction_matrix(predictions, phase="Primary")

    ensemble_delta_mean_abs = None
    ensemble_delta_max_abs = None
    ensemble_coord_std = None
    model_disagreement = None
    ensemble_primary_weight = None
    primary_oy = None
    secondary_oy = None

    if ensemble:
        if secondary_weights is None:
            raise ValueError(
                "secondary_weights is required when ensemble=True"
            )
        if cancel_check is not None and bool(cancel_check()):
            raise RuntimeError("Prediction cancelled by user")

        if hasattr(image_generator, "reset"):
            image_generator.reset()

        model = load_xception_weights(model, secondary_weights, species)

        try:
            secondary_predictions, _, _ = _run_inference_passes(
                model,
                image_generator,
                phase_label="secondary",
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                log_callback=log_callback,
                pass_specs=pass_specs,
            )
        except Exception as exc:
            partial_predictions = _build_predictions_dataframe(image_generator, predictions)
            raise EnsemblePartialResultError(
                "Secondary inference failed during ensemble pass. Primary predictions are available for recovery.",
                partial_predictions,
            ) from exc

        try:
            secondary_predictions = _validate_prediction_matrix(secondary_predictions, phase="Secondary")
        except Exception as exc:
            partial_predictions = _build_predictions_dataframe(image_generator, predictions)
            raise EnsemblePartialResultError(
                "Secondary inference produced invalid outputs. Primary predictions are available for recovery.",
                partial_predictions,
            ) from exc

        primary_predictions = predictions.copy()
        ensemble_abs_delta = np.abs(primary_predictions - secondary_predictions)
        ensemble_delta_mean_abs = np.mean(ensemble_abs_delta, axis=1)
        ensemble_delta_max_abs = np.max(ensemble_abs_delta, axis=1)
        ensemble_coord_std = np.mean(np.std(np.stack([primary_predictions, secondary_predictions], axis=0), axis=0), axis=1)
        primary_oy = primary_predictions[:, 1].copy()
        secondary_oy = secondary_predictions[:, 1].copy()

        atlas_depth = _resolve_atlas_depth(species)
        consistency_cutoff = float(max(0.01, ensemble_consistency_threshold)) * atlas_depth
        model_disagreement = np.abs(primary_oy - secondary_oy) > consistency_cutoff

        if confidence_weighted_ensemble:
            blended, ensemble_primary_weight = _residual_weighted_blend(primary_predictions, secondary_predictions)
            predictions = blended
        else:
            predictions = np.mean([primary_predictions, secondary_predictions], axis=0)

        if model_disagreement is not None and np.any(model_disagreement):
            if ensemble_primary_weight is None:
                primary_choice = np.ones(model_disagreement.shape[0], dtype=bool)
            else:
                primary_choice = ensemble_primary_weight >= 0.5
            use_primary = model_disagreement & primary_choice
            use_secondary = model_disagreement & (~primary_choice)
            predictions[use_primary, :] = primary_predictions[use_primary, :]
            predictions[use_secondary, :] = secondary_predictions[use_secondary, :]

        # Keep basis vectors orthonormal after ensemble blending.
        predictions[:, 3:6] = _normalize_vectors(predictions[:, 3:6])
        predictions[:, 6:9] = _normalize_vectors(predictions[:, 6:9])

    predictions = _validate_prediction_matrix(predictions, phase="Final")

    predictions_df = _build_predictions_dataframe(
        image_generator,
        predictions,
        ensemble_delta_mean_abs=ensemble_delta_mean_abs,
        ensemble_delta_max_abs=ensemble_delta_max_abs,
        tta_coordinate_variance=tta_coordinate_variance,
        tta_oy_std=tta_oy_std,
        ensemble_coord_std=ensemble_coord_std,
        model_disagreement=model_disagreement,
        ensemble_primary_weight=ensemble_primary_weight,
        primary_oy=primary_oy,
        secondary_oy=secondary_oy,
    )

    return predictions_df


def get_image_size(fname):
    """Return width and height for an image file using Pillow."""
    try:
        with Image.open(fname) as image:
            width, height = image.size
    except Exception as exc:
        raise ValueError(f"Unable to read image dimensions for '{fname}': {exc}") from exc
    return width, height
