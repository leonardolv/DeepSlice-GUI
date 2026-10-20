"""Malformed integer fields in a session file must not abort the load."""
from __future__ import annotations

import pytest

from DeepSlice.gui.state import DeepSliceAppState

INT_FIELDS = [
    "inference_batch_size",
    "min_resolution_px",
    "section_dropout_passes",
    "training_seed",
    "training_patience",
]
BAD = [None, "abc", "", float("nan"), float("inf"), True, [], {}]


@pytest.mark.parametrize("field", INT_FIELDS)
@pytest.mark.parametrize("bad", BAD, ids=repr)
def test_bad_value_keeps_current_and_does_not_raise(field, bad):
    state = DeepSliceAppState()
    before = getattr(state, field)
    state.load_session_dict({"image_paths": [], field: bad})
    assert getattr(state, field) == before


@pytest.mark.parametrize("field,raw,expected", [
    ("min_resolution_px", "512", 512),
    ("min_resolution_px", 700.0, 700),
    ("min_resolution_px", 10, 64),
    ("section_dropout_passes", -3, 0),
    ("inference_batch_size", "8.0", 8),
    ("inference_batch_size", 0, 1),
    ("training_patience", 0, 1),
])
def test_numeric_strings_are_accepted_and_clamped(field, raw, expected):
    state = DeepSliceAppState()
    state.load_session_dict({"image_paths": [], field: raw})
    assert getattr(state, field) == expected


def test_later_fields_still_applied_after_a_bad_one():
    state = DeepSliceAppState()
    state.load_session_dict(
        {"image_paths": [], "min_resolution_px": "oops", "tta_enabled": True, "training_seed": 7}
    )
    assert state.tta_enabled is True
    assert state.training_seed == 7
