"""Round-trip tests for DeepSliceAppState.to_session_dict / load_session_dict.

If load/save drift apart, session files written by one DeepSlice build will
silently fail to restore on the next. These tests pin down the keys that
must survive a save+load cycle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from DeepSlice.gui.state import DeepSliceAppState


def _make_state_with_predictions() -> DeepSliceAppState:
    state = DeepSliceAppState()
    state.species = "mouse"
    state.image_paths = ["/tmp/a_s001.png", "/tmp/a_s002.png"]
    state.section_numbers = True
    state.legacy_section_numbers = False
    state.ensemble = True
    state.use_secondary_model = False
    state.outlier_sigma_threshold = 2.1
    state.confidence_medium_threshold = 0.4
    state.confidence_high_threshold = 0.8
    state.tta_enabled = True
    state.gamma_correction = 1.3

    state.predictions = pd.DataFrame(
        {
            "Filenames": ["a_s001.png", "a_s002.png"],
            "nr": [1, 2],
            "ox": [0.0, 0.0],
            "oy": [100.0, 200.0],
            "oz": [0.0, 0.0],
            "ux": [1.0, 1.0],
            "uy": [0.0, 0.0],
            "uz": [0.0, 0.0],
            "vx": [0.0, 0.0],
            "vy": [0.0, 0.0],
            "vz": [1.0, 1.0],
            "width": [512, 512],
            "height": [512, 512],
        }
    )
    return state


def test_session_roundtrip_preserves_scalar_settings():
    state = _make_state_with_predictions()
    payload = state.to_session_dict()

    restored = DeepSliceAppState()
    restored.load_session_dict(payload)

    assert restored.species == state.species
    assert restored.image_paths == state.image_paths
    assert restored.section_numbers == state.section_numbers
    assert restored.legacy_section_numbers == state.legacy_section_numbers
    assert restored.ensemble == state.ensemble
    assert restored.use_secondary_model == state.use_secondary_model
    assert restored.tta_enabled == state.tta_enabled
    assert abs(restored.outlier_sigma_threshold - state.outlier_sigma_threshold) < 1e-9
    assert abs(restored.confidence_medium_threshold - state.confidence_medium_threshold) < 1e-9
    assert abs(restored.confidence_high_threshold - state.confidence_high_threshold) < 1e-9
    assert abs(restored.gamma_correction - state.gamma_correction) < 1e-9


def test_session_roundtrip_preserves_predictions_table():
    state = _make_state_with_predictions()
    payload = state.to_session_dict()

    restored = DeepSliceAppState()
    restored.load_session_dict(payload)

    assert restored.predictions is not None
    assert len(restored.predictions) == len(state.predictions)
    for col in ["Filenames", "nr", "ox", "oy", "oz"]:
        assert col in restored.predictions.columns


def test_session_load_marks_state_clean():
    state = _make_state_with_predictions()
    state.is_dirty = True
    payload = state.to_session_dict()

    restored = DeepSliceAppState()
    restored.is_dirty = True
    restored.load_session_dict(payload)
    assert restored.is_dirty is False, "load_session_dict must clear the dirty flag"


def test_invalid_species_falls_back_to_mouse():
    payload = {
        "species": "unicorn",
        "image_paths": [],
        "predictions": None,
    }
    restored = DeepSliceAppState()
    restored.load_session_dict(payload)
    assert restored.species == "mouse"


def test_load_session_ignores_non_finite_and_malformed_numeric_settings():
    """NaN/inf/garbage numeric settings keep the current value and never abort the load."""
    state = DeepSliceAppState()
    names = (
        "gamma_correction",
        "training_train_fraction",
        "training_val_fraction",
        "training_lr_factor",
        "training_min_lr",
        "blur_variance_threshold",
        "dark_intensity_threshold",
        "bright_intensity_threshold",
        "saturated_fraction_threshold",
        "artifact_blank_fraction_threshold",
    )
    before = {name: getattr(state, name) for name in names}
    state.load_session_dict(
        {
            "gamma_correction": float("nan"),
            "training_train_fraction": float("inf"),
            "training_val_fraction": float("nan"),
            "training_lr_factor": "oops",
            "training_min_lr": None,
            "blur_variance_threshold": "not-a-number",
            "dark_intensity_threshold": float("nan"),
            "bright_intensity_threshold": float("-inf"),
            "saturated_fraction_threshold": None,
            "artifact_blank_fraction_threshold": [1],
            "training_patience": 7,
        }
    )
    for name, value in before.items():
        assert getattr(state, name) == value, name
    assert state.training_patience == 7  # later keys are still applied
