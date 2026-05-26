"""Tests for DSModel validation, pure-logic helpers, and DeepSliceAppState
methods that do not require model weights or image files."""
import json
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.gui.state import DeepSliceAppState, SUPPORTED_IMAGE_FORMATS


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _sample_predictions(n: int = 6) -> pd.DataFrame:
    rows = []
    for idx in range(n):
        rows.append(
            {
                "Filenames": f"brain_s{idx + 1:03d}.png",
                "nr": (idx + 1) * 5,
                "height": 640,
                "width": 1024,
                "ox": 480.0 + idx,
                "oy": 320.0 - (idx * 8.0),
                "oz": 332.0 + (idx * 0.5),
                "ux": -505.0 + (idx * 0.2),
                "uy": 0.72 + (idx * 0.01),
                "uz": 8.5 + (idx * 0.1),
                "vx": -8.0 - (idx * 0.1),
                "vy": 1.30 + (idx * 0.01),
                "vz": -380.0 - (idx * 0.3),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DSModel._parse_bool
# ---------------------------------------------------------------------------

def test_parse_bool_true_values():
    from DeepSlice.main import DSModel
    assert DSModel._parse_bool(True) is True
    assert DSModel._parse_bool("true") is True
    assert DSModel._parse_bool("True") is True
    assert DSModel._parse_bool("TRUE") is True


def test_parse_bool_false_values():
    from DeepSlice.main import DSModel
    assert DSModel._parse_bool(False) is False
    assert DSModel._parse_bool("false") is False
    assert DSModel._parse_bool(0) is False


# ---------------------------------------------------------------------------
# DSModel._validate_prediction_coordinates
# ---------------------------------------------------------------------------

def test_validate_prediction_coordinates_valid():
    from DeepSlice.main import DSModel
    df = _sample_predictions()
    DSModel._validate_prediction_coordinates(df)  # should not raise


def test_validate_prediction_coordinates_missing_column():
    from DeepSlice.main import DSModel
    df = _sample_predictions().drop(columns=["ux"])
    with pytest.raises(RuntimeError, match="ux"):
        DSModel._validate_prediction_coordinates(df)


def test_validate_prediction_coordinates_nan():
    from DeepSlice.main import DSModel
    df = _sample_predictions()
    df.at[2, "oy"] = float("nan")
    with pytest.raises(RuntimeError, match="NaN"):
        DSModel._validate_prediction_coordinates(df)


def test_validate_prediction_coordinates_empty():
    from DeepSlice.main import DSModel
    df = _sample_predictions().iloc[0:0]
    with pytest.raises(RuntimeError, match="empty"):
        DSModel._validate_prediction_coordinates(df)


# ---------------------------------------------------------------------------
# DSModel._append_vector_diagnostics
# ---------------------------------------------------------------------------

def test_append_vector_diagnostics_adds_columns():
    from DeepSlice.main import DSModel

    class _FakeModel:
        log_callback = None
        species = "mouse"
        def _log(self, msg, callback=None): pass

    obj = _FakeModel.__new__(_FakeModel)
    obj.log_callback = None
    obj.species = "mouse"
    obj._log = lambda msg, callback=None: None

    df = _sample_predictions()
    result = DSModel._append_vector_diagnostics(obj, df)
    assert "uv_dot" in result.columns
    assert "uv_cosine" in result.columns
    assert "orthogonality_flag" in result.columns


def test_append_vector_diagnostics_orthogonal_vectors_not_flagged():
    from DeepSlice.main import DSModel

    class _FakeModel:
        log_callback = None
        def _log(self, msg, callback=None): pass

    obj = _FakeModel.__new__(_FakeModel)
    obj.log_callback = None
    obj._log = lambda msg, callback=None: None

    df = pd.DataFrame(
        [
            {
                "Filenames": "s001.png",
                "nr": 1,
                "ox": 0.0, "oy": 0.0, "oz": 0.0,
                # u = (1, 0, 0), v = (0, 1, 0)  → perfectly orthogonal
                "ux": 1.0, "uy": 0.0, "uz": 0.0,
                "vx": 0.0, "vy": 1.0, "vz": 0.0,
            }
        ]
    )
    result = DSModel._append_vector_diagnostics(obj, df)
    assert not bool(result["orthogonality_flag"].iloc[0])


# ---------------------------------------------------------------------------
# DSModel.adjust_angles validation
# ---------------------------------------------------------------------------

def test_dsmodel_adjust_angles_rejects_nan(monkeypatch):
    from DeepSlice.main import DSModel

    class _FakeDSModel(DSModel.__class__):
        pass

    # Build minimal instance without loading model
    obj = object.__new__(DSModel)
    obj.predictions = _sample_predictions()
    obj.species = "mouse"

    with pytest.raises(ValueError, match="finite"):
        obj.adjust_angles(float("nan"), 0.0)


def test_dsmodel_adjust_angles_rejects_out_of_range(monkeypatch):
    from DeepSlice.main import DSModel

    obj = object.__new__(DSModel)
    obj.predictions = _sample_predictions()
    obj.species = "mouse"

    with pytest.raises(ValueError, match="ML angle must be within"):
        obj.adjust_angles(91.0, 0.0)

    with pytest.raises(ValueError, match="DV angle must be within"):
        obj.adjust_angles(0.0, -91.0)


# ---------------------------------------------------------------------------
# DSModel.predict validation
# ---------------------------------------------------------------------------

def test_dsmodel_predict_rejects_zero_batch_size(monkeypatch):
    from DeepSlice.main import DSModel

    obj = object.__new__(DSModel)
    obj.species = "mouse"
    obj.log_callback = None
    obj.download_callback = None
    obj.bad_sections_present = False
    obj.config = {"ensemble_status": {"mouse": False}, "weight_file_paths": {}}
    obj.metadata_path = ""
    obj._model_species = None
    obj.model = None

    with pytest.raises(ValueError, match="positive integer"):
        obj.predict(batch_size=0)


def test_dsmodel_predict_rejects_oversized_batch_size(monkeypatch):
    from DeepSlice.main import DSModel

    obj = object.__new__(DSModel)
    obj.species = "mouse"
    obj.log_callback = None
    obj.download_callback = None
    obj.bad_sections_present = False
    obj.config = {"ensemble_status": {"mouse": False}, "weight_file_paths": {}}
    obj.metadata_path = ""
    obj._model_species = None
    obj.model = None

    with pytest.raises(ValueError, match="512"):
        obj.predict(batch_size=1024)


def test_dsmodel_predict_rejects_invalid_consistency_threshold(monkeypatch):
    from DeepSlice.main import DSModel

    obj = object.__new__(DSModel)
    obj.species = "mouse"
    obj.log_callback = None
    obj.download_callback = None
    obj.bad_sections_present = False
    obj.config = {"ensemble_status": {"mouse": False}, "weight_file_paths": {}}
    obj.metadata_path = ""
    obj._model_species = None
    obj.model = None

    with pytest.raises(ValueError, match="ensemble_consistency_threshold"):
        obj.predict(batch_size=8, ensemble_consistency_threshold=1.5)


# ---------------------------------------------------------------------------
# DSModel.__init__ validation
# ---------------------------------------------------------------------------

def test_dsmodel_init_rejects_none_species():
    from DeepSlice.main import DSModel
    with pytest.raises(ValueError, match="non-empty"):
        DSModel(species=None)


def test_dsmodel_init_rejects_empty_species():
    from DeepSlice.main import DSModel
    with pytest.raises(ValueError, match="non-empty"):
        DSModel(species="   ")


def test_dsmodel_init_rejects_invalid_species():
    from DeepSlice.main import DSModel
    with pytest.raises(ValueError, match="Invalid species"):
        DSModel(species="cat")


# ---------------------------------------------------------------------------
# DSModel.propagate_angles — returns bool
# ---------------------------------------------------------------------------

def test_propagate_angles_returns_bool(monkeypatch):
    """propagate_angles must return a bool indicating convergence."""
    from DeepSlice.main import DSModel
    from DeepSlice.coord_post_processing import angle_methods

    obj = object.__new__(DSModel)
    obj.predictions = _sample_predictions()
    obj.species = "mouse"
    obj.log_callback = None

    # Patch propagate_angles to immediately converge (return unchanged predictions)
    monkeypatch.setattr(
        angle_methods,
        "propagate_angles",
        lambda df, method, species: df.copy(),
    )

    result = obj.propagate_angles()
    assert isinstance(result, bool)
    assert result is True


# ---------------------------------------------------------------------------
# DeepSliceAppState.set_images — extension filtering and deduplication
# ---------------------------------------------------------------------------

def test_set_images_filters_unsupported_extensions(tmp_path):
    state = DeepSliceAppState(species="mouse")

    # Create files with mixed extensions
    png_file = tmp_path / "section_001.png"
    txt_file = tmp_path / "notes.txt"
    jpg_file = tmp_path / "section_002.jpg"
    png_file.write_bytes(b"fake")
    txt_file.write_text("not an image")
    jpg_file.write_bytes(b"fake")

    state.set_images([str(png_file), str(txt_file), str(jpg_file)])

    basenames = [pathlib.Path(p).name for p in state.image_paths]
    assert "section_001.png" in basenames
    assert "section_002.jpg" in basenames
    assert "notes.txt" not in basenames


def test_set_images_deduplicates(tmp_path):
    state = DeepSliceAppState(species="mouse")
    png_file = tmp_path / "section_001.png"
    png_file.write_bytes(b"fake")

    state.set_images([str(png_file), str(png_file), str(png_file)])
    assert len(state.image_paths) == 1


def test_set_images_ignores_nonexistent(tmp_path):
    state = DeepSliceAppState(species="mouse")
    state.set_images([str(tmp_path / "missing.png")])
    assert len(state.image_paths) == 0


# ---------------------------------------------------------------------------
# DeepSliceAppState.image_format_report
# ---------------------------------------------------------------------------

def test_image_format_report_classifies_correctly(tmp_path):
    state = DeepSliceAppState(species="mouse")
    png_file = tmp_path / "a.png"
    tif_file = tmp_path / "b.tif"
    bad_file = tmp_path / "c.bmp"
    png_file.write_bytes(b"x")
    tif_file.write_bytes(b"x")
    bad_file.write_bytes(b"x")

    # set_images now filters unsupported, so bypass it to test report directly
    state.image_paths = [str(png_file), str(tif_file), str(bad_file)]
    report = state.image_format_report()

    supported_names = [pathlib.Path(p).name for p in report["supported"]]
    unsupported_names = [pathlib.Path(p).name for p in report["unsupported"]]

    assert "a.png" in supported_names
    assert "b.tif" in supported_names
    assert "c.bmp" in unsupported_names


# ---------------------------------------------------------------------------
# DeepSliceAppState.undo/redo stack deque behaviour
# ---------------------------------------------------------------------------

def test_undo_stack_max_depth():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions()

    # Push more than 50 snapshots
    for i in range(60):
        state.predictions = _sample_predictions()
        state.snapshot_predictions()

    assert len(state.undo_stack) == 50


def test_undo_redo_stack_is_deque():
    from collections import deque
    state = DeepSliceAppState(species="mouse")
    assert isinstance(state.undo_stack, deque)
    assert isinstance(state.redo_stack, deque)


def test_snapshot_clears_redo_stack():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions()
    state.snapshot_predictions()
    state.predictions = _sample_predictions()
    state.snapshot_predictions()
    state.undo()

    assert len(state.redo_stack) == 1

    state.predictions = _sample_predictions()
    state.snapshot_predictions()

    # Snapshot after undo should clear redo
    assert len(state.redo_stack) == 0


# ---------------------------------------------------------------------------
# DeepSliceAppState.summary_metrics
# ---------------------------------------------------------------------------

def test_summary_metrics_no_predictions():
    state = DeepSliceAppState(species="mouse")
    metrics = state.summary_metrics()
    assert metrics["slice_count"] == 0
    assert metrics["processed"] == 0
    assert metrics["excluded"] == 0


def test_summary_metrics_with_bad_sections():
    state = DeepSliceAppState(species="mouse")
    preds = _sample_predictions()
    preds["bad_section"] = False
    preds.loc[0, "bad_section"] = True
    preds.loc[3, "bad_section"] = True
    state.predictions = preds

    metrics = state.summary_metrics()
    assert metrics["excluded"] == 2
    assert metrics["processed"] == len(preds) - 2
    assert metrics["slice_count"] == len(preds)


# ---------------------------------------------------------------------------
# DeepSliceAppState.set_quality_controls
# ---------------------------------------------------------------------------

def test_set_quality_controls_rejects_non_finite():
    state = DeepSliceAppState(species="mouse")
    with pytest.raises(ValueError):
        state.set_quality_controls(outlier_sigma=float("inf"), confidence_medium=0.5, confidence_high=0.8)


def test_set_quality_controls_rejects_inverted_thresholds():
    state = DeepSliceAppState(species="mouse")
    with pytest.raises(ValueError, match="High confidence threshold"):
        state.set_quality_controls(outlier_sigma=1.5, confidence_medium=0.8, confidence_high=0.5)


# ---------------------------------------------------------------------------
# DeepSliceAppState.interpolate_bad_section_depths — boundary cases
# ---------------------------------------------------------------------------

def test_interpolate_bad_section_depth_no_bad_sections():
    state = DeepSliceAppState(species="mouse")
    preds = _sample_predictions()
    preds["bad_section"] = False
    state.predictions = preds
    replaced = state.interpolate_bad_section_depths()
    assert replaced == 0


def test_interpolate_bad_section_depth_requires_at_least_3():
    state = DeepSliceAppState(species="mouse")
    preds = _sample_predictions(n=2)
    preds["bad_section"] = [True, False]
    state.predictions = preds
    replaced = state.interpolate_bad_section_depths()
    assert replaced == 0


def test_interpolate_bad_section_depth_gap_too_large():
    state = DeepSliceAppState(species="mouse")
    preds = _sample_predictions(n=10)
    preds["bad_section"] = False
    # Mark 7 consecutive middle sections bad (gap > default max_gap=6)
    preds.loc[1:7, "bad_section"] = True
    state.predictions = preds
    replaced = state.interpolate_bad_section_depths(max_gap=4)
    assert replaced == 0


# ---------------------------------------------------------------------------
# DeepSliceAppState.linearity_payload — edge cases
# ---------------------------------------------------------------------------

def test_linearity_payload_single_section():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions(n=1)
    payload = state.linearity_payload()
    assert payload["confidence"].shape[0] == 1


def test_linearity_payload_all_bad_sections():
    state = DeepSliceAppState(species="mouse")
    preds = _sample_predictions()
    preds["bad_section"] = True
    state.predictions = preds
    payload = state.linearity_payload()
    assert np.all(payload["confidence"] == 0.0)


# ---------------------------------------------------------------------------
# DeepSliceAppState.curation_risk_scores — without optional columns
# ---------------------------------------------------------------------------

def test_curation_risk_scores_no_optional_columns():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions()
    # Do NOT annotate — optional columns absent
    scores = state.curation_risk_scores()
    assert scores.shape[0] == len(state.predictions)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


# ---------------------------------------------------------------------------
# DeepSliceAppState.flag_low_confidence_sections
# ---------------------------------------------------------------------------

def test_flag_low_confidence_sections_threshold_zero_flags_none():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions()
    flagged = state.flag_low_confidence_sections(
        threshold=0.0, include_outliers=False, include_high_risk=False
    )
    assert flagged == 0


def test_flag_low_confidence_sections_threshold_one_flags_all():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions()
    flagged = state.flag_low_confidence_sections(
        threshold=1.0, include_outliers=False, include_high_risk=False
    )
    assert flagged == len(state.predictions)


# ---------------------------------------------------------------------------
# DeepSliceAppState.to_session_dict / load_session_dict round-trip
# ---------------------------------------------------------------------------

def test_session_dict_roundtrip_preserves_species():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions()

    payload = state.to_session_dict()
    restored = DeepSliceAppState(species="rat")
    restored.load_session_dict(payload)

    assert restored.species == "mouse"
    assert restored.predictions is not None
    assert len(restored.predictions) == len(state.predictions)


def test_session_dict_roundtrip_preserves_quality_settings():
    state = DeepSliceAppState(species="mouse")
    state.set_quality_controls(outlier_sigma=2.0, confidence_medium=0.45, confidence_high=0.85)

    payload = state.to_session_dict()
    restored = DeepSliceAppState(species="mouse")
    restored.load_session_dict(payload)

    assert restored.outlier_sigma_threshold == pytest.approx(2.0)
    assert restored.confidence_medium_threshold == pytest.approx(0.45)
    assert restored.confidence_high_threshold == pytest.approx(0.85)


def test_session_dict_roundtrip_clears_undo_redo():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions()
    state.snapshot_predictions()

    payload = state.to_session_dict()
    restored = DeepSliceAppState(species="mouse")
    restored.load_session_dict(payload)

    assert len(restored.undo_stack) == 0
    assert len(restored.redo_stack) == 0
