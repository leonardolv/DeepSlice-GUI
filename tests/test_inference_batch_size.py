"""Verify `_recommended_inference_batch_size` after its GPU-probing
"auto-detect" branch was removed as dead code.

`run_prediction` is the only caller and always passes a concrete
`requested_batch_size` (`self.inference_batch_size`, an `int` field that
defaults to 8 and is never `None`), so the method's job is now just
validating and returning that value - it no longer accepts an unused
`progress_callback` or falls through to a `tensorflow`-probing branch that
no call site could ever reach.
"""
from __future__ import annotations

import inspect

import pytest

from DeepSlice.gui.state import DeepSliceAppState


def test_no_longer_accepts_the_unused_progress_callback():
    """`progress_callback` only ever gated the removed GPU-probing branch;
    keeping it in the signature after that branch is gone would be a
    parameter nothing reads."""
    params = inspect.signature(
        DeepSliceAppState._recommended_inference_batch_size
    ).parameters
    assert "progress_callback" not in params


def test_returns_the_requested_batch_size():
    state = DeepSliceAppState()
    assert state._recommended_inference_batch_size(requested_batch_size=4) == 4


def test_rejects_a_non_positive_batch_size():
    state = DeepSliceAppState()
    with pytest.raises(ValueError):
        state._recommended_inference_batch_size(requested_batch_size=0)
    with pytest.raises(ValueError):
        state._recommended_inference_batch_size(requested_batch_size=-1)


def test_rejects_a_batch_size_over_512():
    state = DeepSliceAppState()
    with pytest.raises(ValueError):
        state._recommended_inference_batch_size(requested_batch_size=513)


def test_logs_the_configured_batch_size():
    state = DeepSliceAppState()
    messages = []
    result = state._recommended_inference_batch_size(
        requested_batch_size=16, log_callback=messages.append
    )
    assert result == 16
    assert messages == ["Using user-configured inference batch size 16"]


def test_run_prediction_passes_inference_batch_size_through_to_the_model(monkeypatch):
    """End-to-end: `run_prediction`'s own call site still resolves to the
    configured `inference_batch_size`, now that the method it calls no
    longer accepts a `progress_callback` keyword."""
    state = DeepSliceAppState()
    state.image_paths = ["/tmp/a.png"]
    state.species = "mouse"
    state.inference_batch_size = 32

    captured = {}

    def fake_ensure_model(self, log_callback=None):
        class _Model:
            species = "mouse"
            predictions = None

            def predict(self, **kwargs):
                captured.update(kwargs)
                import pandas as pd

                self.predictions = pd.DataFrame(
                    {
                        "Filenames": ["a.png"],
                        "ox": [0.0], "oy": [0.0], "oz": [0.0],
                        "ux": [1.0], "uy": [0.0], "uz": [0.0],
                        "vx": [0.0], "vy": [1.0], "vz": [0.0],
                        "width": [256], "height": [256],
                    }
                )

        state.model = _Model()
        return state.model

    monkeypatch.setattr(DeepSliceAppState, "ensure_model", fake_ensure_model)

    state.run_prediction(
        section_numbers=False,
        legacy_section_numbers=False,
        ensemble=False,
        use_secondary_model=False,
    )

    assert captured["batch_size"] == 32
