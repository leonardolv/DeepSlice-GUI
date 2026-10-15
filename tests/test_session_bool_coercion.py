"""Hand-edited / foreign session files may carry booleans as strings."""
from __future__ import annotations

import pytest

from DeepSlice.gui.state import DeepSliceAppState

FLAGS = [
    "tissue_crop_enabled",
    "clahe_enabled",
    "quality_gate_enabled",
    "preserve_color_enabled",
    "tta_enabled",
    "multiscale_enabled",
    "fast_fp16_enabled",
    "bilateral_denoise_enabled",
    "percentile_normalization_enabled",
    "confidence_weighted_ensemble",
    "training_use_mixed_precision",
]


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("raw,expected", [("false", False), ("False", False), ("0", False), ("true", True), (False, False), (True, True)])
def test_string_booleans_are_not_truthy(flag, raw, expected):
    state = DeepSliceAppState()
    state.load_session_dict({"image_paths": [], flag: raw})
    assert getattr(state, flag) is expected


@pytest.mark.parametrize("flag", FLAGS)
def test_missing_flag_keeps_current_value(flag):
    state = DeepSliceAppState()
    before = getattr(state, flag)
    state.load_session_dict({"image_paths": []})
    assert getattr(state, flag) is before
