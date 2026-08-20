"""`is_dirty` must only flip once a mutator's edit actually happens.

`undo()` carries the rule in a comment: "Do not flip is_dirty until we know
the swap can succeed: raising on an empty stack should not mark the session
as modified." Nine other mutators drifted from it by setting the flag as
their first statement — `propagate_angles` was fixed already (see
`test_angle_convergence_reaches_the_user.py`); this file covers the
remaining eight: `run_prediction`, `set_bad_sections`,
`flag_low_confidence_sections`, `interpolate_bad_section_depths`,
`apply_manual_order`, `adjust_angles`, `enforce_index_order` and
`enforce_index_spacing`. Two failure shapes: a validation error before any
edit (a raise), and an edit that turns out to change nothing (a no-op
`return`/`return 0`) — both used to leave the session nagging the user
about unsaved changes it does not have.
"""

import pandas as pd
import pytest

from DeepSlice.gui.state import DeepSliceAppState


COORDS = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]


def sample_predictions(n=3):
    return pd.DataFrame(
        {
            "Filenames": [f"{i}.png" for i in range(n)],
            "nr": list(range(1, n + 1)),
            **{c: [float(i) for i in range(n)] for c in COORDS},
        }
    )


class FakeModel:
    """A stand-in for DSModel that just echoes predictions back, so these
    tests never need real weights, a GPU, or the real coordinate math."""

    species = "mouse"

    def __init__(self):
        self.predictions = None

    def set_bad_sections(self, bad_sections, auto=False):
        self.predictions = self.predictions.copy()

    def adjust_angles(self, ML, DV):
        self.predictions = self.predictions.copy()

    def enforce_index_order(self):
        self.predictions = self.predictions.copy()

    def enforce_index_spacing(self, section_thickness=None):
        self.predictions = self.predictions.copy()


def install_model(monkeypatch, state):
    model = FakeModel()
    monkeypatch.setattr(state, "ensure_model", lambda *a, **k: model)
    return model


@pytest.fixture
def state():
    st = DeepSliceAppState()
    st.predictions = sample_predictions()
    return st


class TestARaiseNeverDirties:
    """The state-not-ready validation error is the same shape as undo() on
    an empty stack: nothing changed, so nothing should look changed."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda st: st.set_bad_sections(["a"]),
            lambda st: st.flag_low_confidence_sections(),
            lambda st: st.interpolate_bad_section_depths(),
            lambda st: st.apply_manual_order([0]),
            lambda st: st.adjust_angles(1.0, 2.0),
            lambda st: st.enforce_index_order(),
            lambda st: st.enforce_index_spacing(),
        ],
    )
    def test_no_predictions_raises_without_dirtying(self, call):
        st = DeepSliceAppState()
        st.predictions = None
        st.is_dirty = False
        with pytest.raises(ValueError):
            call(st)
        assert st.is_dirty is False

    def test_run_prediction_raises_without_dirtying_on_no_images(self):
        st = DeepSliceAppState()
        st.image_paths = []
        st.is_dirty = False
        with pytest.raises(ValueError, match="No images selected"):
            st.run_prediction(
                section_numbers=False,
                legacy_section_numbers=False,
                ensemble=None,
                use_secondary_model=False,
            )
        assert st.is_dirty is False

    def test_apply_manual_order_length_mismatch_does_not_dirty(self, state):
        state.is_dirty = False
        with pytest.raises(ValueError, match="does not match"):
            state.apply_manual_order([0])
        assert state.is_dirty is False


class TestANoOpNeverDirties:
    def test_flag_low_confidence_sections_on_empty_predictions(self):
        st = DeepSliceAppState()
        st.predictions = sample_predictions(n=0)
        st.is_dirty = False
        assert st.flag_low_confidence_sections() == 0
        assert st.is_dirty is False

    def test_flag_low_confidence_sections_when_nothing_new_is_flagged(
        self, monkeypatch, state
    ):
        monkeypatch.setattr(
            state, "linearity_payload", lambda: {"confidence": [1.0] * 3, "outliers": [False] * 3}
        )
        state.is_dirty = False
        assert (
            state.flag_low_confidence_sections(
                threshold=0.0, include_outliers=False, include_high_risk=False
            )
            == 0
        )
        assert state.is_dirty is False

    def test_interpolate_bad_section_depths_with_no_bad_sections_column(self, state):
        state.is_dirty = False
        assert state.interpolate_bad_section_depths() == 0
        assert state.is_dirty is False

    def test_interpolate_bad_section_depths_with_no_bad_sections_flagged(self, state):
        state.predictions["bad_section"] = False
        state.is_dirty = False
        assert state.interpolate_bad_section_depths() == 0
        assert state.is_dirty is False


class TestARealEditDirties:
    def test_set_bad_sections(self, monkeypatch, state):
        install_model(monkeypatch, state)
        state.is_dirty = False
        state.set_bad_sections(["0.png"])
        assert state.is_dirty is True

    def test_apply_manual_order(self, state):
        state.is_dirty = False
        state.apply_manual_order(list(range(len(state.predictions)))[::-1])
        assert state.is_dirty is True

    def test_adjust_angles(self, monkeypatch, state):
        install_model(monkeypatch, state)
        state.is_dirty = False
        state.adjust_angles(1.0, 2.0)
        assert state.is_dirty is True

    def test_enforce_index_order(self, monkeypatch, state):
        install_model(monkeypatch, state)
        state.is_dirty = False
        state.enforce_index_order()
        assert state.is_dirty is True

    def test_enforce_index_spacing(self, monkeypatch, state):
        install_model(monkeypatch, state)
        state.is_dirty = False
        state.enforce_index_spacing(section_thickness_um=25.0)
        assert state.is_dirty is True

    def test_flag_low_confidence_sections(self, monkeypatch, state):
        monkeypatch.setattr(
            state, "linearity_payload", lambda: {"confidence": [0.0] * 3, "outliers": [False] * 3}
        )
        state.is_dirty = False
        flagged = state.flag_low_confidence_sections(
            threshold=1.0, include_outliers=False, include_high_risk=False
        )
        assert flagged == 3
        assert state.is_dirty is True

    def test_interpolate_bad_section_depths(self, monkeypatch, state):
        state.predictions["bad_section"] = [False, True, False]
        monkeypatch.setattr(
            "DeepSlice.gui.state.calculate_brain_center_depths",
            lambda predictions, species=None: [0.0, 5.0, 10.0],
        )
        state.is_dirty = False
        replaced = state.interpolate_bad_section_depths()
        assert replaced == 1
        assert state.is_dirty is True
