"""Rat-species coverage for DeepSliceAppState.

These tests mirror the core behaviors already covered for mouse in
test_curation_state.py but run the species-sensitive code paths against
the rat atlas (39 um, depth 0-1024) to catch regressions in rat support.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.gui.state import DeepSliceAppState
from DeepSlice.metadata import metadata_loader


class _RatMockModel:
    def __init__(self, predictions: pd.DataFrame):
        self.species = "rat"
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
        marked = {str(name) for name in bad_sections}
        self.predictions = self.predictions.copy()
        self.predictions["bad_section"] = self.predictions["Filenames"].astype(str).isin(marked)


def _rat_sample_predictions() -> pd.DataFrame:
    """Plane parameters scaled for the WHS rat atlas (depth 0-1024)."""
    rows = []
    for idx in range(6):
        rows.append(
            {
                "Filenames": f"rat_s{idx + 1:03d}.png",
                "nr": (idx + 1) * 5,
                "height": 640,
                "width": 1024,
                "ox": 256.0 + idx,
                "oy": 700.0 - (idx * 20.0),
                "oz": 256.0 + (idx * 0.5),
                "ux": -450.0 + (idx * 0.2),
                "uy": 0.5 + (idx * 0.01),
                "uz": 6.0 + (idx * 0.1),
                "vx": -6.0 - (idx * 0.1),
                "vy": 1.1 + (idx * 0.01),
                "vz": -320.0 - (idx * 0.3),
            }
        )
    return pd.DataFrame(rows)


def test_rat_depth_range_comes_from_config():
    min_depth, max_depth = metadata_loader.get_species_depth_range("rat")
    assert (min_depth, max_depth) == (0, 1024)


def test_rat_state_rejects_invalid_species():
    state = DeepSliceAppState(species="rat")
    with pytest.raises(ValueError, match="Species must be one of"):
        state.set_species("hamster")


def test_rat_supports_ensemble_reflects_config():
    state = DeepSliceAppState(species="rat")
    expected = bool(state._config["ensemble_status"]["rat"])
    assert state.supports_ensemble() is expected


def test_rat_manual_reorder_and_undo_roundtrip():
    state = DeepSliceAppState(species="rat")
    original = _rat_sample_predictions()
    state.predictions = original.copy()

    ordered = [5, 3, 1, 0, 2, 4]
    state.apply_manual_order(ordered)

    assert state.predictions.iloc[0]["Filenames"] == original.iloc[5]["Filenames"]

    state.undo()
    pdt.assert_frame_equal(
        state.predictions.reset_index(drop=True),
        original.reset_index(drop=True),
        check_dtype=False,
    )


def test_rat_bad_sections_roundtrip(monkeypatch):
    state = DeepSliceAppState(species="rat")
    original = _rat_sample_predictions()
    state.predictions = original.copy()

    model = _RatMockModel(state.predictions)
    monkeypatch.setattr(state, "ensure_model", lambda log_callback=None: model)

    selected = [original.iloc[1]["Filenames"], original.iloc[4]["Filenames"]]
    state.set_bad_sections(selected, auto=False)

    assert "bad_section" in state.predictions.columns
    flagged = state.predictions.loc[state.predictions["bad_section"], "Filenames"].tolist()
    assert sorted(flagged) == sorted(selected)

    state.undo()
    pdt.assert_frame_equal(
        state.predictions.reset_index(drop=True),
        original.reset_index(drop=True),
        check_dtype=False,
    )


def test_rat_linearity_confidence_values_are_bounded():
    state = DeepSliceAppState(species="rat")
    state.predictions = _rat_sample_predictions()

    payload = state.linearity_payload()
    confidence = np.asarray(payload["confidence"], dtype=float)

    assert len(confidence) == len(state.predictions)
    assert np.all(confidence >= 0.0)
    assert np.all(confidence <= 1.0)
