"""A non-converging angle normalization must not look like a successful one.

`DSModel.propagate_angles` returns False and logs DS-008 after six
non-converging iterations, and writes its best available estimate into
`predictions` either way. The two layers above it dropped that flag, so
"Normalize Angles" reported success and the user shipped half-normalised
coordinates with no warning.
"""

import ast
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from DeepSlice.gui.state import DeepSliceAppState


COORDS = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]


def sample_predictions():
    return pd.DataFrame(
        {
            "Filenames": ["a.png", "b.png"],
            "nr": [1, 2],
            **{c: [0.0, 1.0] for c in COORDS},
        }
    )


class FakeModel:
    """Stands in for DSModel so this never needs real weights or a GPU."""

    species = "mouse"

    def __init__(self, converged):
        self._converged = converged
        self.predictions = None
        self.calls = 0

    def propagate_angles(self, method="weighted_mean"):
        self.calls += 1
        # A non-converging run still writes its best estimate, which is exactly
        # why the return value is the only way to tell the two apart.
        self.predictions = self.predictions.copy()
        return self._converged


@pytest.fixture
def state(monkeypatch):
    st = DeepSliceAppState()
    st.predictions = sample_predictions()
    return st


def install_model(monkeypatch, state, converged):
    model = FakeModel(converged)
    monkeypatch.setattr(state, "ensure_model", lambda *a, **k: model)
    monkeypatch.setattr(state, "snapshot_predictions", lambda: None)
    return model


class TestTheStateLayerPassesTheFlagOn:
    def test_a_converged_run_returns_true(self, monkeypatch, state):
        install_model(monkeypatch, state, converged=True)
        assert state.propagate_angles() is True

    def test_a_non_converged_run_returns_false(self, monkeypatch, state):
        install_model(monkeypatch, state, converged=False)
        assert state.propagate_angles() is False

    def test_the_predictions_are_updated_either_way(self, monkeypatch, state):
        """The failure mode this guards: a non-converging run looks identical
        from the outside, because it does produce output."""
        model = install_model(monkeypatch, state, converged=False)
        state.propagate_angles()
        assert model.calls == 1
        assert state.predictions is not None

    def test_it_still_refuses_to_run_without_predictions(self, monkeypatch):
        st = DeepSliceAppState()
        st.predictions = None
        with pytest.raises(ValueError, match="No predictions"):
            st.propagate_angles()

    def test_a_refusal_does_not_mark_the_session_dirty(self, monkeypatch):
        """`is_dirty = True` used to be the first statement, so an action that
        raised still nagged the user about unsaved changes. `undo()` already
        carries a comment stating this rule."""
        st = DeepSliceAppState()
        st.predictions = None
        st.is_dirty = False
        with pytest.raises(ValueError):
            st.propagate_angles()
        assert st.is_dirty is False

    def test_a_real_run_does_mark_the_session_dirty(self, monkeypatch, state):
        state.is_dirty = False
        install_model(monkeypatch, state, converged=True)
        state.propagate_angles()
        assert state.is_dirty is True


class TestTheWindowActuallyLooksAtIt:
    """Driving `_normalize_angles` needs a whole QMainWindow; reading the
    handler is what is available without one, and it is the exact thing that
    regressed — the call was there, its result was discarded."""

    @staticmethod
    def handler_source():
        # find_spec locates the file without executing it: main_window imports
        # matplotlib's Qt backend at module scope, which this check does not
        # need and which is not installed everywhere the rest of the suite runs.
        spec = importlib.util.find_spec("DeepSlice.gui.main_window")
        source = Path(spec.origin).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_normalize_angles":
                return node
        raise AssertionError("_normalize_angles not found")

    def test_the_result_is_bound_rather_than_discarded(self):
        node = self.handler_source()
        calls = [
            n for n in ast.walk(node)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "propagate_angles"
        ]
        assert calls, "propagate_angles()'s return value is thrown away again"

    def test_the_handler_branches_on_it(self):
        node = self.handler_source()
        names = {
            n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        assert "converged" in names

        tests = [
            ast.dump(n.test) for n in ast.walk(node) if isinstance(n, ast.If)
        ]
        assert any("converged" in t for t in tests), (
            "the flag is captured but nothing tests it"
        )

    def test_the_user_is_told_inside_that_branch(self):
        """Not just 'the handler has a toast somewhere' — the handler already
        had `_show_logged_exception` for the raising path, which is a different
        failure. The warning has to be in the branch the flag selects."""
        node = self.handler_source()
        branches = [
            n for n in ast.walk(node)
            if isinstance(n, ast.If) and "converged" in ast.dump(n.test)
        ]
        assert branches, "nothing branches on the convergence flag"

        told = [
            c for branch in branches for c in ast.walk(branch)
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr in {"_show_toast", "_show_logged_exception"}
        ]
        assert told, "a non-converging run reaches no user-visible surface"
