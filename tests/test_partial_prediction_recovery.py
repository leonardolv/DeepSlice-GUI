"""Verify the partial-recovery path in DeepSliceAppState.run_prediction.

When ensemble secondary inference fails, the GUI must be able to keep the
primary-model predictions instead of forcing a full re-run.
"""
from __future__ import annotations

import pandas as pd
import pytest

from DeepSlice.gui import state as state_module
from DeepSlice.gui.state import DeepSliceAppState, PartialPredictionAvailable


class _FakeEnsembleError(Exception):
    """Stand-in for neural_network.EnsemblePartialResultError."""

    def __init__(self, message: str, partial_predictions: pd.DataFrame):
        super().__init__(message)
        self.partial_predictions = partial_predictions


def _partial_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Filenames": ["a.png", "b.png"],
            "ox": [0.0, 0.0],
            "oy": [50.0, 150.0],
            "oz": [0.0, 0.0],
            "ux": [1.0, 1.0],
            "uy": [0.0, 0.0],
            "uz": [0.0, 0.0],
            "vx": [0.0, 0.0],
            "vy": [0.0, 0.0],
            "vz": [1.0, 1.0],
            "width": [256, 256],
            "height": [256, 256],
        }
    )


def test_partial_recovery_raises_typed_exception(monkeypatch):
    state = DeepSliceAppState()
    state.image_paths = ["/tmp/a.png", "/tmp/b.png"]
    state.species = "mouse"

    def fake_ensure_model(self, log_callback=None):
        class _Model:
            species = "mouse"
            predictions = None

            def predict(self, **kwargs):
                raise _FakeEnsembleError("secondary failed", _partial_df())

        state.model = _Model()
        return state.model

    monkeypatch.setattr(DeepSliceAppState, "ensure_model", fake_ensure_model)

    with pytest.raises(PartialPredictionAvailable):
        state.run_prediction(
            section_numbers=False,
            legacy_section_numbers=False,
            ensemble=True,
            use_secondary_model=False,
        )

    assert state.has_partial_prediction_candidate()
    assert "secondary failed" in state.partial_prediction_reason()


def test_use_partial_prediction_candidate_clears_state():
    state = DeepSliceAppState()
    state.species = "mouse"
    state._partial_prediction_candidate = _partial_df()
    state._partial_prediction_reason = "secondary failed"

    result = state.use_partial_prediction_candidate()

    assert result["slice_count"] == 2
    assert result["partial_recovery"] is True
    assert state.has_partial_prediction_candidate() is False, (
        "Candidate must be cleared after adoption to avoid double-applying it"
    )
