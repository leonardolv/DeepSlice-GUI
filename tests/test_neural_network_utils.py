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
    _run_inference_passes,
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
# _run_inference_passes — progress reporting and cancellation must cover
# every pass, not just the first.
# ---------------------------------------------------------------------------

class _FakeGenerator:
    """Minimal stand-in for the Keras image sequence `_run_inference_passes`
    consumes: a fixed image count (`n`) that clone_with must preserve, since
    the fix relies on every pass reporting the same per-pass image count."""

    def __init__(self, n=4, batch_size=2):
        self.n = n
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def clone_with(self, **kwargs):
        return _FakeGenerator(n=self.n, batch_size=self.batch_size)


class _FakeModel:
    """Stand-in for the Keras model: drives the Keras-callback protocol
    (`on_predict_batch_begin`/`on_predict_batch_end`) the same way
    `model.predict(..., callbacks=[...])` would, without needing a real
    network or GPU."""

    def predict(self, generator, steps, verbose=0, callbacks=None):
        callbacks = callbacks or []
        for batch in range(steps):
            for cb in callbacks:
                cb.on_predict_batch_begin(batch)
            for cb in callbacks:
                cb.on_predict_batch_end(batch)
        return np.tile(np.arange(9, dtype=float), (generator.n, 1))


class _TrackingModel(_FakeModel):
    """Like `_FakeModel`, but records how many batches each `predict()` call
    (i.e. each pass) actually got through before returning or raising."""

    def __init__(self):
        self.batches_started: list[dict] = []

    def predict(self, generator, steps, verbose=0, callbacks=None):
        record = {"planned_steps": steps, "batches_completed": 0}
        self.batches_started.append(record)
        callbacks = callbacks or []
        for batch in range(steps):
            for cb in callbacks:
                cb.on_predict_batch_begin(batch)
            for cb in callbacks:
                cb.on_predict_batch_end(batch)
            record["batches_completed"] = batch + 1
        return np.tile(np.arange(9, dtype=float), (generator.n, 1))


def test_run_inference_passes_progress_covers_every_pass_not_just_the_first():
    # tta=True with no multi-scale/dropout gives 4 passes (flip: none/h/v/hv).
    pass_specs = _build_inference_pass_specs(tta=True, multi_scale=False, section_dropout_passes=0)
    assert len(pass_specs) == 4

    generator = _FakeGenerator(n=4, batch_size=2)
    calls = []

    def record_progress(completed, total, phase):
        calls.append((completed, total, phase))

    _run_inference_passes(
        model=_FakeModel(),
        base_generator=generator,
        phase_label="primary",
        progress_callback=record_progress,
        pass_specs=pass_specs,
    )

    assert calls, "progress_callback was never invoked"

    total_images_all_passes = len(pass_specs) * generator.n
    # Every call must report the SAME total — the whole run's image count —
    # not just the first pass's. Before the fix, only pass 0 ever reported,
    # so `total` would have been `generator.n` (4), not 16.
    totals = {total for _, total, _ in calls}
    assert totals == {total_images_all_passes}

    # The bar must not reach 100% until the very last pass — before the fix
    # it hit 100% (completed == generator.n == total) at the end of pass 0
    # and then never moved again for the remaining 3 passes.
    completed_after_first_pass = calls[generator.__len__() - 1][0]
    assert completed_after_first_pass == generator.n
    assert completed_after_first_pass < total_images_all_passes

    # It must reach exactly 100% (not overshoot or undershoot) by the end.
    assert calls[-1][0] == total_images_all_passes


def test_run_inference_passes_cancellation_reaches_every_pass():
    pass_specs = _build_inference_pass_specs(tta=True, multi_scale=False, section_dropout_passes=0)
    generator = _FakeGenerator(n=4, batch_size=2)  # __len__() == 2 batches/pass

    # cancel_check() is called once at each pass's top-of-loop boundary check
    # and twice per batch (on_predict_batch_begin, on_predict_batch_end).
    # Pass 0 (2 batches) therefore makes 1 + 2*2 = 5 calls, and pass 1's own
    # boundary check is call 6. Flip to True starting at call 7 — pass 1's
    # *first batch* callback, not a pass boundary. Before the fix, only pass
    # 0 had a callback attached at all, so nothing would have called
    # cancel_check again until pass 2's boundary check — i.e. only after
    # pass 1 ran to completion for nothing.
    call_count = {"n": 0}

    def cancel_check():
        call_count["n"] += 1
        return call_count["n"] >= 7

    model = _TrackingModel()
    with pytest.raises(RuntimeError, match="cancelled"):
        _run_inference_passes(
            model=model,
            base_generator=generator,
            phase_label="primary",
            progress_callback=lambda *a: None,
            cancel_check=cancel_check,
            pass_specs=pass_specs,
        )

    # Stopped exactly at pass 1's first batch callback, not after running
    # every remaining batch of pass 1 (which would need 5 more calls) to
    # completion first.
    assert call_count["n"] == 7

    # Pass 0 ran to completion (both its batches); pass 1 was interrupted
    # before completing even its first batch. Before the fix, pass 1 had no
    # callback attached at all (only pass 0 did), so it would have run to
    # completion undisturbed (batches_completed == planned_steps) and
    # cancellation would only have been noticed at pass 2's boundary check —
    # after silently discarding a full pass of GPU/CPU work.
    assert len(model.batches_started) == 2
    assert model.batches_started[0]["batches_completed"] == model.batches_started[0]["planned_steps"]
    assert model.batches_started[1]["batches_completed"] == 0


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
    # ...but they must not vanish silently: an unreadable image will fail
    # again during actual inference, so the quality gate needs to be able to
    # tell the user about it before running prediction.
    assert report["unreadable_count"] == 2
    assert report["unreadable_paths"] == ["/no/such/image.png", "/also/missing.jpg"]


def test_inspect_image_batch_mixed_readable_and_unreadable(tmp_path):
    from PIL import Image as PILImage

    good_path = tmp_path / "good.png"
    array = np.random.randint(0, 255, size=(120, 160, 3), dtype=np.uint8)
    PILImage.fromarray(array).save(good_path)

    report = inspect_image_batch([str(good_path), "/no/such/image.png"])
    assert report["total"] == 2
    assert len(report["metrics"]) == 1
    assert report["unreadable_count"] == 1
    assert report["unreadable_paths"] == ["/no/such/image.png"]


# ---------------------------------------------------------------------------
# Efficiency: preview_preprocessed_image should not double-open the file
# (regression test for the redundant inspect_image_quality re-load).
# ---------------------------------------------------------------------------

def test_preview_preprocessed_image_opens_file_only_once(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    from DeepSlice.neural_network import neural_network

    image_path = tmp_path / "sample.png"
    array = np.random.randint(0, 255, size=(120, 160, 3), dtype=np.uint8)
    PILImage.fromarray(array).save(image_path)

    open_call_count = {"count": 0}
    original_open = PILImage.open

    def _counting_open(path, *args, **kwargs):
        if str(path) == str(image_path):
            open_call_count["count"] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(PILImage, "open", _counting_open)

    result = neural_network.preview_preprocessed_image(str(image_path))

    assert open_call_count["count"] == 1
    assert "preview_image" in result
    assert "model_input" in result
    assert result["model_input"].shape == neural_network.XCEPTION_INPUT_SIZE


def test_preprocess_image_array_default_path_minimises_rgb2gray():
    """On the default preprocessing path (no scale, no Reinhard, no CLAHE,
    no bilateral denoise, no gamma change), only two rgb2gray conversions
    should be required: the initial tissue-mask gray and the final
    grayscale-as-3-channels step. This is a regression guard against
    accidentally re-introducing intermediate gray recomputations."""
    from DeepSlice.neural_network import neural_network

    call_count = {"count": 0}
    original_rgb2gray = neural_network.rgb2gray

    def _counting_rgb2gray(image):
        call_count["count"] += 1
        return original_rgb2gray(image)

    neural_network.rgb2gray = _counting_rgb2gray
    try:
        rgb = np.random.rand(120, 160, 3).astype(np.float32)
        options = neural_network._coerce_preprocessing_options(None)
        neural_network._preprocess_image_array(rgb, options=options)
    finally:
        neural_network.rgb2gray = original_rgb2gray

    assert call_count["count"] == 2, (
        f"Default preprocessing path should call rgb2gray exactly twice "
        f"(initial + final), but called it {call_count['count']} times."
    )
