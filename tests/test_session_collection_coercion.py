"""Malformed image_paths / predictions / stain mode must not abort or corrupt a session load."""
from __future__ import annotations

import pytest

from DeepSlice.gui.state import DeepSliceAppState


@pytest.mark.parametrize("bad", [None, "abc", 5, {"a": 1}, True])
def test_non_list_image_paths_reject_the_load_before_mutating_anything(bad):
    state = DeepSliceAppState()
    state.gamma_correction = 1.0
    state.is_dirty = True
    with pytest.raises(TypeError):
        state.load_session_dict({"image_paths": bad, "gamma_correction": 1.5})
    assert state.gamma_correction == 1.0
    assert state.is_dirty is True


def test_non_string_entries_are_dropped(tmp_path):
    real = tmp_path / "a.png"
    real.write_bytes(b"x")
    state = DeepSliceAppState()
    state.load_session_dict({"image_paths": [None, 5, "", "  ", str(real)]})
    assert state.image_paths == [str(real)]


@pytest.mark.parametrize("bad", ["abc", 5, 1.5])
def test_scalar_predictions_reject_the_load_before_mutating_anything(bad):
    state = DeepSliceAppState()
    state.image_paths = ["keep.png"]
    state.gamma_correction = 1.0
    state.is_dirty = True
    with pytest.raises(ValueError):
        state.load_session_dict({"image_paths": [], "predictions": bad, "gamma_correction": 1.5})
    assert state.image_paths == ["keep.png"]
    assert state.gamma_correction == 1.0
    assert state.is_dirty is True


@pytest.mark.parametrize("bad", ["bogus", None, 5, ""])
def test_unknown_stain_mode_keeps_current(bad):
    state = DeepSliceAppState()
    state.stain_normalization_mode = "reinhard"
    state.load_session_dict({"image_paths": [], "stain_normalization_mode": bad})
    assert state.stain_normalization_mode == "reinhard"


def test_known_stain_mode_applies():
    state = DeepSliceAppState()
    state.load_session_dict({"image_paths": [], "stain_normalization_mode": " Reinhard "})
    assert state.stain_normalization_mode == "reinhard"


@pytest.mark.parametrize("bad", ["sideways", 5, ["rostro-caudal"], {"a": 1}, True, ""])
def test_invalid_indexing_direction_becomes_none(bad):
    state = DeepSliceAppState()
    state.load_session_dict({"detected_indexing_direction": bad, "selected_indexing_direction": bad})
    assert state.detected_indexing_direction is None
    assert state.selected_indexing_direction is None


def test_valid_indexing_direction_is_kept_and_normalised():
    state = DeepSliceAppState()
    state.load_session_dict(
        {"detected_indexing_direction": "Rostro-Caudal ", "selected_indexing_direction": "caudal-rostro"}
    )
    assert state.detected_indexing_direction == "rostro-caudal"
    assert state.selected_indexing_direction == "caudal-rostro"


@pytest.mark.parametrize("bad", [[], [1, 2], "text", 5, None])
def test_non_object_payload_is_rejected_with_a_clear_error(bad):
    state = DeepSliceAppState()
    state.is_dirty = True
    with pytest.raises(TypeError, match="JSON object"):
        state.load_session_dict(bad)
    assert state.is_dirty is True
