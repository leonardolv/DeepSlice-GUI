"""Tests for pure utility functions in neural_network.py that do not require
actual model weights or GPU hardware."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.neural_network.neural_network import (
    ATLAS_DEPTH,
    XCEPTION_INPUT_SIZE,
    _build_inference_pass_specs,
    _coerce_preprocessing_options,
    _combine_prediction_passes,
    _normalize_vectors,
    _residual_weighted_blend,
    _validate_prediction_matrix,
    inspect_image_batch,
    inspect_image_quality,
)


# ---------------------------------------------------------------------------
# XCEPTION_INPUT_SIZE and ATLAS_DEPTH module constants
# ---------------------------------------------------------------------------

def test_xception_input_size_is_correct():
    assert XCEPTION_INPUT_SIZE == (299, 299, 3)


def test_atlas_depth_has_mouse_and_rat():
    assert "mouse" in ATLAS_DEPTH
    assert "rat" in ATLAS_DEPTH
    assert ATLAS_DEPTH["mouse"] == pytest.approx(528.0)
    assert ATLAS_DEPTH["rat"] == pytest.approx(1024.0)


# ---------------------------------------------------------------------------
# _coerce_preprocessing_options
# ---------------------------------------------------------------------------

def test_coerce_preprocessing_options_defaults():
    options = _coerce_preprocessing_options(None)
    assert isinstance(options, dict)
    assert "tissue_crop" in options
    assert "clahe" in options
    assert options["gamma"] == pytest.approx(1.0)
    assert options["stain_normalization"] == "none"


def test_coerce_preprocessing_options_overrides():
    options = _coerce_preprocessing_options({"clahe": True, "gamma": 1.8})
    assert options["clahe"] is True
    assert options["gamma"] == pytest.approx(1.8)
    # Defaults still present
    assert "tissue_crop" in options


def test_coerce_preprocessing_options_gamma_clipped():
    options = _coerce_preprocessing_options({"gamma": 0.1})
    assert options["gamma"] >= 0.5

    options = _coerce_preprocessing_options({"gamma": 5.0})
    assert options["gamma"] <= 2.0


def test_coerce_preprocessing_options_stain_normalization_lowercased():
    options = _coerce_preprocessing_options({"stain_normalization": "Reinhard"})
    assert options["stain_normalization"] == "reinhard"


# ---------------------------------------------------------------------------
# _validate_prediction_matrix
# ---------------------------------------------------------------------------

def test_validate_prediction_matrix_passes_valid_input():
    matrix = np.random.randn(5, 9).astype(np.float64)
    result = _validate_prediction_matrix(matrix, "test")
    assert result.shape == (5, 9)


def test_validate_prediction_matrix_rejects_wrong_columns():
    with pytest.raises(RuntimeError, match="shape"):
        _validate_prediction_matrix(np.ones((5, 8)), "test")


def test_validate_prediction_matrix_rejects_empty():
    with pytest.raises(RuntimeError, match="no predictions"):
        _validate_prediction_matrix(np.zeros((0, 9)), "test")


def test_validate_prediction_matrix_rejects_nan():
    matrix = np.ones((3, 9))
    matrix[1, 4] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        _validate_prediction_matrix(matrix, "test")


def test_validate_prediction_matrix_rejects_inf():
    matrix = np.ones((3, 9))
    matrix[0, 0] = np.inf
    with pytest.raises(RuntimeError, match="non-finite"):
        _validate_prediction_matrix(matrix, "test")


# ---------------------------------------------------------------------------
# _normalize_vectors
# ---------------------------------------------------------------------------

def test_normalize_vectors_produces_unit_vectors():
    vectors = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]])
    result = _normalize_vectors(vectors)
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0)


def test_normalize_vectors_handles_zero_vector():
    vectors = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = _normalize_vectors(vectors)
    assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# _combine_prediction_passes
# ---------------------------------------------------------------------------

def test_combine_prediction_passes_single_pass_zero_variance():
    single = np.random.randn(4, 9)
    combined, variance, oy_std = _combine_prediction_passes([single])
    assert combined.shape == (4, 9)
    assert np.allclose(variance, 0.0)
    assert np.allclose(oy_std, 0.0)


def test_combine_prediction_passes_averages_multiple():
    a = np.ones((5, 9)) * 2.0
    b = np.ones((5, 9)) * 4.0
    combined, variance, oy_std = _combine_prediction_passes([a, b])
    assert combined.shape == (5, 9)
    # Origins (columns 0-2) should be averaged
    assert np.allclose(combined[:, :3], 3.0)
    assert np.all(variance >= 0.0)


def test_combine_prediction_passes_raises_on_empty():
    with pytest.raises(ValueError, match="No prediction passes"):
        _combine_prediction_passes([])


# ---------------------------------------------------------------------------
# _residual_weighted_blend
# ---------------------------------------------------------------------------

def test_residual_weighted_blend_returns_correct_shape():
    primary = np.random.randn(6, 9)
    secondary = np.random.randn(6, 9)
    blended, weights = _residual_weighted_blend(primary, secondary)
    assert blended.shape == (6, 9)
    assert weights.shape == (6,)


def test_residual_weighted_blend_weights_sum_to_one():
    primary = np.random.randn(8, 9)
    secondary = np.random.randn(8, 9)
    _, primary_weight = _residual_weighted_blend(primary, secondary)
    secondary_weight = 1.0 - primary_weight
    assert np.allclose(primary_weight + secondary_weight, 1.0)
    assert np.all(primary_weight >= 0.0) and np.all(primary_weight <= 1.0)


def test_residual_weighted_blend_empty_input():
    primary = np.zeros((0, 9))
    secondary = np.zeros((0, 9))
    blended, weights = _residual_weighted_blend(primary, secondary)
    assert blended.shape == (0, 9)
    assert weights.shape == (0,)


def test_residual_weighted_blend_origin_xyz_all_weighted():
    """All three origin columns (ox, oy, oz) must use the weighted blend."""
    # Give primary a perfect linear oy trend, secondary a noisy one.
    n = 10
    x = np.arange(n, dtype=float)
    primary = np.zeros((n, 9))
    primary[:, 1] = x * 5.0 + 100.0   # perfect oy trend -> low residual -> high weight

    secondary = np.zeros((n, 9))
    secondary[:, 1] = x * 5.0 + 100.0 + np.random.default_rng(0).normal(0, 10, n)

    # Make ox different in primary and secondary
    primary[:, 0] = 1.0
    secondary[:, 0] = 3.0

    blended, primary_weight = _residual_weighted_blend(primary, secondary)

    # Since primary has lower residuals it should dominate (weight > 0.5)
    assert float(np.mean(primary_weight)) > 0.5

    # ox must reflect the weighted blend (not simple average)
    expected_ox = primary_weight * 1.0 + (1.0 - primary_weight) * 3.0
    assert np.allclose(blended[:, 0], expected_ox, atol=1e-6)


# ---------------------------------------------------------------------------
# _build_inference_pass_specs
# ---------------------------------------------------------------------------

def test_build_inference_pass_specs_default():
    specs = _build_inference_pass_specs(tta=False, multi_scale=False, section_dropout_passes=0)
    assert len(specs) == 1
    assert specs[0]["flip_mode"] == "none"
    assert specs[0]["scale_factor"] == pytest.approx(1.0)


def test_build_inference_pass_specs_tta_adds_flips():
    specs = _build_inference_pass_specs(tta=True, multi_scale=False, section_dropout_passes=0)
    flip_modes = {spec["flip_mode"] for spec in specs}
    assert "h" in flip_modes
    assert "v" in flip_modes
    assert len(specs) == 4


def test_build_inference_pass_specs_multiscale_adds_scales():
    specs = _build_inference_pass_specs(tta=False, multi_scale=True, section_dropout_passes=0)
    scales = {spec["scale_factor"] for spec in specs}
    assert 0.75 in scales
    assert 1.25 in scales
    assert len(specs) == 3


def test_build_inference_pass_specs_dropout_passes():
    specs = _build_inference_pass_specs(tta=False, multi_scale=False, section_dropout_passes=3)
    dropout_specs = [s for s in specs if s["dropout_fraction"] > 0.0]
    assert len(dropout_specs) == 3


def test_build_inference_pass_specs_combined():
    specs = _build_inference_pass_specs(tta=True, multi_scale=True, section_dropout_passes=2)
    # 4 flips * 3 scales = 12, plus 2 dropout = 14
    assert len(specs) == 14


# ---------------------------------------------------------------------------
# inspect_image_batch — empty list
# ---------------------------------------------------------------------------

def test_inspect_image_batch_empty_list():
    report = inspect_image_batch([])
    assert report["total"] == 0
    assert report["issue_count"] == 0
    assert report["resolution_mismatch"] is False


def test_inspect_image_batch_nonexistent_paths_skipped():
    report = inspect_image_batch(["/no/such/image.png", "/also/missing.jpg"])
    assert report["total"] == 2
    # Both should be skipped gracefully, no crash
    assert len(report["metrics"]) == 0
