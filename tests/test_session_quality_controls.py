"""One invalid quality threshold in a session must not discard the valid ones."""
from __future__ import annotations

import pytest

from DeepSlice.gui.state import DeepSliceAppState


def test_valid_values_are_applied():
    state = DeepSliceAppState()
    state.load_session_dict({
        "image_paths": [],
        "outlier_sigma_threshold": 2.5,
        "confidence_medium_threshold": 0.4,
        "confidence_high_threshold": 0.8,
    })
    assert (state.outlier_sigma_threshold, state.confidence_medium_threshold,
            state.confidence_high_threshold) == (2.5, 0.4, 0.8)


@pytest.mark.parametrize("bad", [99.0, "abc", None, float("nan")], ids=repr)
def test_bad_sigma_keeps_valid_confidence_values(bad):
    state = DeepSliceAppState()
    before = state.outlier_sigma_threshold
    state.load_session_dict({
        "image_paths": [],
        "outlier_sigma_threshold": bad,
        "confidence_medium_threshold": 0.4,
        "confidence_high_threshold": 0.8,
    })
    assert state.outlier_sigma_threshold == before
    assert state.confidence_medium_threshold == 0.4
    assert state.confidence_high_threshold == 0.8


def test_bad_confidence_keeps_valid_sigma():
    state = DeepSliceAppState()
    medium, high = state.confidence_medium_threshold, state.confidence_high_threshold
    state.load_session_dict({
        "image_paths": [],
        "outlier_sigma_threshold": 2.5,
        "confidence_medium_threshold": 5.0,
        "confidence_high_threshold": "x",
    })
    assert state.outlier_sigma_threshold == 2.5
    assert (state.confidence_medium_threshold, state.confidence_high_threshold) == (medium, high)
