import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.gui.state import DeepSliceAppState


class _MockModel:
    def __init__(self, predictions: pd.DataFrame):
        self.species = "mouse"
        self.predictions = predictions.copy()
        self.spacing_calls = []
        self.bad_section_calls = []

    def enforce_index_order(self):
        self.predictions = self.predictions.sort_values("nr").reset_index(drop=True)

    def enforce_index_spacing(self, section_thickness=None):
        self.spacing_calls.append(section_thickness)
        self.predictions = self.predictions.copy()
        delta = float(section_thickness or 0.0)
        self.predictions["oy"] = self.predictions["oy"].astype(float) + delta

    def set_bad_sections(self, bad_sections, auto=False):
        self.bad_section_calls.append((list(bad_sections), bool(auto)))
        marked = set(str(name) for name in bad_sections)
        self.predictions = self.predictions.copy()
        self.predictions["bad_section"] = self.predictions["Filenames"].astype(str).isin(marked)


class _FailingPredictModel:
    species = "mouse"

    def __init__(self, partial_predictions: pd.DataFrame):
        self.predictions = None
        self._partial_predictions = partial_predictions

    def predict(self, **kwargs):
        class _PartialFailure(Exception):
            def __init__(self, partial_predictions):
                super().__init__("secondary failed")
                self.partial_predictions = partial_predictions

        raise _PartialFailure(self._partial_predictions.copy())


def _sample_predictions() -> pd.DataFrame:
    rows = []
    for idx in range(6):
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


def test_linearity_payload_confidence_outputs_are_valid():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions()

    payload = state.linearity_payload()

    confidence = payload["confidence"]
    assert len(confidence) == len(state.predictions)
    assert np.all(confidence >= 0.0)
    assert np.all(confidence <= 1.0)

    levels = set(payload["confidence_level"].tolist())
    assert levels.issubset({"high", "medium", "low"})

    components = payload["confidence_components"]
    assert set(components.keys()) == {"residual", "angle", "spacing", "center_weight"}


def test_manual_reorder_and_undo_redo_roundtrip():
    state = DeepSliceAppState(species="mouse")
    original = _sample_predictions()
    state.predictions = original.copy()

    ordered_indices = [5, 3, 1, 0, 2, 4]
    state.apply_manual_order(ordered_indices)
    reordered = state.predictions.copy()

    assert reordered.iloc[0]["Filenames"] == original.iloc[5]["Filenames"]
    assert reordered["nr"].astype(int).tolist() == sorted(original["nr"].astype(int).tolist())

    state.undo()
    pdt.assert_frame_equal(
        state.predictions.reset_index(drop=True),
        original.reset_index(drop=True),
        check_dtype=False,
    )

    state.redo()
    pdt.assert_frame_equal(
        state.predictions.reset_index(drop=True),
        reordered.reset_index(drop=True),
        check_dtype=False,
    )


def test_set_bad_sections_supports_undo_and_records_auto_flag(monkeypatch):
    state = DeepSliceAppState(species="mouse")
    original = _sample_predictions()
    state.predictions = original.copy()

    model = _MockModel(state.predictions)
    monkeypatch.setattr(state, "ensure_model", lambda log_callback=None: model)

    selected = [original.iloc[1]["Filenames"], original.iloc[4]["Filenames"]]
    state.set_bad_sections(selected, auto=True)

    assert "bad_section" in state.predictions.columns
    flagged = state.predictions.loc[state.predictions["bad_section"], "Filenames"].tolist()
    assert sorted(flagged) == sorted(selected)
    assert model.bad_section_calls[-1] == (selected, True)

    state.undo()
    pdt.assert_frame_equal(
        state.predictions.reset_index(drop=True),
        original.reset_index(drop=True),
        check_dtype=False,
    )


def test_enforce_index_order_and_undo_roundtrip(monkeypatch):
    state = DeepSliceAppState(species="mouse")
    original = _sample_predictions().iloc[[3, 0, 5, 1, 4, 2]].reset_index(drop=True)
    state.predictions = original.copy()

    model = _MockModel(state.predictions)
    monkeypatch.setattr(state, "ensure_model", lambda log_callback=None: model)

    state.enforce_index_order()
    assert state.predictions["nr"].astype(int).tolist() == sorted(original["nr"].astype(int).tolist())

    state.undo()
    pdt.assert_frame_equal(
        state.predictions.reset_index(drop=True),
        original.reset_index(drop=True),
        check_dtype=False,
    )


def test_enforce_index_spacing_flips_sign_by_direction(monkeypatch):
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions()

    model = _MockModel(state.predictions)
    monkeypatch.setattr(state, "ensure_model", lambda log_callback=None: model)

    state.selected_indexing_direction = "rostro-caudal"
    state.enforce_index_spacing(section_thickness_um=120.0)
    assert model.spacing_calls[-1] == -120.0

    state.selected_indexing_direction = "caudal-rostro"
    state.enforce_index_spacing(section_thickness_um=120.0)
    assert model.spacing_calls[-1] == 120.0


def test_undo_redo_edge_behavior():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions()

    with pytest.raises(ValueError, match="Nothing to undo"):
        state.undo()

    first_order = [5, 4, 3, 2, 1, 0]
    second_order = [0, 2, 4, 1, 3, 5]

    state.apply_manual_order(first_order)
    state.undo()
    state.redo()

    state.undo()
    state.apply_manual_order(second_order)
    with pytest.raises(ValueError, match="Nothing to redo"):
        state.redo()


def test_linearity_payload_marks_bad_sections_low_confidence():
    state = DeepSliceAppState(species="mouse")
    predictions = _sample_predictions().copy()
    predictions["bad_section"] = False
    predictions.loc[[1, 4], "bad_section"] = True
    state.predictions = predictions

    payload = state.linearity_payload()
    n = len(predictions)

    assert payload["x"].shape[0] == n
    assert payload["y"].shape[0] == n
    assert payload["trend"].shape[0] == n
    assert payload["residuals"].shape[0] == n
    assert payload["outliers"].shape[0] == n
    assert payload["weights"].shape[0] == n
    assert payload["confidence"].shape[0] == n
    assert payload["confidence_level"].shape[0] == n

    bad_mask = predictions["bad_section"].astype(bool).values
    assert np.all(payload["confidence"][bad_mask] == 0.0)

    components = payload["confidence_components"]
    for name in ["residual", "angle", "spacing", "center_weight"]:
        assert components[name].shape[0] == n


def test_partial_prediction_candidate_roundtrip_without_model():
    state = DeepSliceAppState(species="mouse")
    state.section_numbers = False
    candidate = _sample_predictions().copy()
    state._partial_prediction_candidate = candidate
    state._partial_prediction_reason = "secondary failed"

    result = state.use_partial_prediction_candidate()

    assert result["partial_recovery"] is True
    assert result["partial_reason"] == "secondary failed"
    assert result["slice_count"] == len(candidate)
    assert state.has_partial_prediction_candidate() is False
    preserved_columns = candidate.columns.tolist()
    pdt.assert_frame_equal(
        state.predictions[preserved_columns].reset_index(drop=True),
        candidate.reset_index(drop=True),
        check_dtype=False,
    )


def test_run_prediction_caches_partial_candidate_on_failure(monkeypatch):
    state = DeepSliceAppState(species="mouse")
    state.image_paths = ["image_001.png"]
    partial = _sample_predictions().copy()

    failing_model = _FailingPredictModel(partial_predictions=partial)
    monkeypatch.setattr(state, "ensure_model", lambda log_callback=None: failing_model)

    # The sentinel-string error was replaced by a typed exception.
    from DeepSlice.gui.state import PartialPredictionAvailable
    with pytest.raises(PartialPredictionAvailable):
        state.run_prediction(
            section_numbers=True,
            legacy_section_numbers=False,
            ensemble=True,
            use_secondary_model=False,
        )

    assert state.has_partial_prediction_candidate() is True
    assert "secondary failed" in state.partial_prediction_reason().lower()


def test_set_quality_controls_validates_ranges():
    state = DeepSliceAppState(species="mouse")

    with pytest.raises(ValueError, match="Outlier sensitivity"):
        state.set_quality_controls(outlier_sigma=0.5, confidence_medium=0.5, confidence_high=0.8)

    with pytest.raises(ValueError, match="High confidence threshold"):
        state.set_quality_controls(outlier_sigma=1.5, confidence_medium=0.7, confidence_high=0.6)

    state.set_quality_controls(outlier_sigma=2.2, confidence_medium=0.45, confidence_high=0.82)
    assert state.outlier_sigma_threshold == pytest.approx(2.2)
    assert state.confidence_medium_threshold == pytest.approx(0.45)
    assert state.confidence_high_threshold == pytest.approx(0.82)


def test_linearity_payload_respects_adjusted_confidence_thresholds():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions()

    baseline = state.linearity_payload()
    baseline_high = int(np.sum(baseline["confidence_level"] == "high"))

    state.set_quality_controls(outlier_sigma=1.5, confidence_medium=0.70, confidence_high=0.92)
    adjusted = state.linearity_payload()
    adjusted_high = int(np.sum(adjusted["confidence_level"] == "high"))

    assert adjusted_high <= baseline_high


def test_partial_prediction_candidate_adds_diagnostic_columns():
    state = DeepSliceAppState(species="mouse")
    state.section_numbers = False
    candidate = _sample_predictions().copy()
    state._partial_prediction_candidate = candidate
    state._partial_prediction_reason = "secondary failed"

    result = state.use_partial_prediction_candidate()

    assert "out_of_bounds_count" in result
    assert "orthogonality_count" in result
    assert "angle_outlier_count" in result
    assert "ap_out_of_bounds" in state.predictions.columns
    assert "orthogonality_flag" in state.predictions.columns
    assert "angle_outlier" in state.predictions.columns


def test_curation_risk_scores_are_bounded_and_aligned():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions().copy()
    state._annotate_prediction_diagnostics()

    scores = state.curation_risk_scores()

    assert scores.shape[0] == len(state.predictions)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_flag_low_confidence_sections_sets_bad_section_column():
    state = DeepSliceAppState(species="mouse")
    state.predictions = _sample_predictions().copy()

    flagged = state.flag_low_confidence_sections(threshold=0.95, include_outliers=False, include_high_risk=False)

    assert flagged > 0
    assert "bad_section" in state.predictions.columns
    assert int(np.sum(state.predictions["bad_section"].astype(bool))) >= flagged


def test_interpolate_bad_section_depths_updates_oy_for_middle_bad_slice():
    state = DeepSliceAppState(species="mouse")
    predictions = _sample_predictions().copy()
    predictions["bad_section"] = False
    predictions.loc[2, "bad_section"] = True
    predictions.loc[2, "oy"] = predictions.loc[2, "oy"] + 30.0
    state.predictions = predictions

    before = state.predictions["oy"].astype(float).copy()
    corrected = state.interpolate_bad_section_depths(max_gap=6)
    after = state.predictions["oy"].astype(float)

    assert corrected >= 1
    assert float(after.iloc[2]) != pytest.approx(float(before.iloc[2]))


def test_training_split_preview_is_group_leakage_free(tmp_path):
    state = DeepSliceAppState(species="mouse")
    image_paths = []
    groups = ["brainA", "brainA", "brainB", "brainB", "brainC", "brainD"]
    for idx, group in enumerate(groups):
        folder = tmp_path / group
        folder.mkdir(parents=True, exist_ok=True)
        image_path = folder / f"{group}_s{idx + 1:03d}.png"
        image_path.write_text("x", encoding="utf-8")
        image_paths.append(str(image_path))

    state.image_paths = image_paths
    state.training_group_mode = "parent-folder"
    state.training_seed = 9
    state.training_train_fraction = 0.60
    state.training_val_fraction = 0.20

    preview = state.build_training_split_preview()

    assert preview["counts"]["total"] == len(image_paths)
    assert preview["counts"]["train"] + preview["counts"]["val"] + preview["counts"]["test"] == len(image_paths)
    assert preview["leakage_free"] is True


def test_training_manifest_exports_include_split_and_metadata(tmp_path):
    state = DeepSliceAppState(species="mouse")
    image_paths = []
    prefixes = ["animal01", "animal01", "animal02", "animal02", "animal03", "animal04"]
    for idx, prefix in enumerate(prefixes):
        image_path = tmp_path / f"{prefix}_s{idx + 1:03d}.png"
        image_path.write_text("x", encoding="utf-8")
        image_paths.append(str(image_path))

    state.image_paths = image_paths
    state.training_group_mode = "filename-prefix"
    state.training_seed = 17
    state.training_train_fraction = 0.67
    state.training_val_fraction = 0.16

    split_path = tmp_path / "training_split_manifest.json"
    metadata_path = tmp_path / "training_run_metadata.json"

    state.save_training_split_manifest(str(split_path))
    state.save_training_metadata_manifest(str(metadata_path))

    with split_path.open("r", encoding="utf-8") as handle:
        split_payload = json.load(handle)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata_payload = json.load(handle)

    assert split_payload["species"] == "mouse"
    assert split_payload["counts"]["total"] == len(image_paths)
    assert set(split_payload["splits"].keys()) == {"train", "val", "test"}
    assert metadata_payload["species"] == "mouse"
    assert metadata_payload["split_counts"]["total"] == len(image_paths)
    assert metadata_payload["split_leakage_free"] is True


def test_training_settings_session_roundtrip():
    state = DeepSliceAppState(species="mouse")
    state.training_seed = 1234
    state.training_group_mode = "filename-prefix"
    state.training_train_fraction = 0.73
    state.training_val_fraction = 0.14
    state.training_patience = 11
    state.training_lr_factor = 0.35
    state.training_min_lr = 2e-6
    state.training_use_mixed_precision = True

    payload = state.to_session_dict()

    restored = DeepSliceAppState(species="rat")
    restored.load_session_dict(payload)

    assert restored.training_seed == 1234
    assert restored.training_group_mode == "filename-prefix"
    assert restored.training_train_fraction == pytest.approx(0.73)
    assert restored.training_val_fraction == pytest.approx(0.14)
    assert restored.training_patience == 11
    assert restored.training_lr_factor == pytest.approx(0.35)
    assert restored.training_min_lr == pytest.approx(2e-6)
    assert restored.training_use_mixed_precision is True
