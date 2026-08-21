"""The "Enable input quality gate" checkbox controlled nothing, and the scan
it did not gate ran synchronously on the GUI thread.

`_run_alignment` used to call `self.state.screen_input_quality()` (which
decodes every input image) unconditionally, regardless of
`state.quality_gate_enabled` -- the checkbox only changed a dialog's title
and default button, never whether the scan or its modal ran at all. On a
large dataset this froze the UI for minutes with no way to skip it.

The fix moves the scan into a `FunctionWorker` (off the UI thread) and gates
it entirely behind `state.quality_gate_enabled`, matching every other
long-running operation in this window.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

from DeepSlice.gui.state import DeepSliceAppState

# ---------------------------------------------------------------------------
# State layer: the empty-input shortcut must carry the same keys as a real
# scan, or a caller reading `unreadable_count`/`unreadable_paths` off an
# empty-dataset report would KeyError.
# ---------------------------------------------------------------------------

def test_screen_input_quality_empty_report_has_unreadable_fields():
    state = DeepSliceAppState()
    state.image_paths = []
    report = state.screen_input_quality()
    assert report["unreadable_count"] == 0
    assert report["unreadable_paths"] == []


# ---------------------------------------------------------------------------
# `_build_quality_gate_warnings` (a plain staticmethod, importable without a
# QApplication) must actually warn about unreadable images -- this is the
# user-facing half of the neural_network.py fix, and the whole point of
# tracking unreadable images at all.
# ---------------------------------------------------------------------------

def _load_main_window_module():
    # find_spec locates the file without executing it: main_window imports
    # matplotlib's Qt backend at module scope, which this check does not
    # need and which may not be installed everywhere the rest of the suite
    # runs (see test_angle_convergence_reaches_the_user.py for precedent).
    spec = importlib.util.find_spec("DeepSlice.gui.main_window")
    source = Path(spec.origin).read_text(encoding="utf-8")
    return ast.parse(source), source


def _function_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in main_window.py")


class TestBuildQualityGateWarningsMentionsUnreadableImages:
    def test_real_call_warns_about_unreadable_images(self):
        try:
            from DeepSlice.gui.main_window import DeepSliceMainWindow
        except ImportError:
            pytest.skip("PySide6 / matplotlib Qt backend not available in this environment")

        report = {
            "resolution_mismatch": False,
            "issue_count": 0,
            "counts": {},
            "unreadable_count": 2,
            "unreadable_paths": ["/data/a.tif", "/data/b.tif"],
        }
        warnings = DeepSliceMainWindow._build_quality_gate_warnings(report)
        assert any("2 image(s) could not be read" in w for w in warnings)
        assert any("a.tif" in w for w in warnings)

    def test_a_clean_report_has_no_warnings(self):
        try:
            from DeepSlice.gui.main_window import DeepSliceMainWindow
        except ImportError:
            pytest.skip("PySide6 / matplotlib Qt backend not available in this environment")

        report = {
            "resolution_mismatch": False,
            "issue_count": 0,
            "counts": {},
            "unreadable_count": 0,
            "unreadable_paths": [],
        }
        assert DeepSliceMainWindow._build_quality_gate_warnings(report) == []


# ---------------------------------------------------------------------------
# Structural checks on `_run_alignment` / `_run_quality_gate_scan`: driving
# the real button click needs a whole constructed QMainWindow (this file's
# sibling, test_angle_convergence_reaches_the_user.py, documents why that is
# impractical here), so this pins the exact shape of the regression instead
# of merely re-reading the code with confidence.
# ---------------------------------------------------------------------------

class TestTheCheckboxActuallyGatesTheScan:
    def test_run_alignment_no_longer_calls_screen_input_quality_directly(self):
        """The old bug: `screen_input_quality()` (the synchronous, UI-thread
        image-decoding scan) was called unconditionally inside
        `_run_alignment`, regardless of the checkbox. It must not be called
        from there any more -- only from the background-worker path."""
        tree, _ = _load_main_window_module()
        node = _function_node(tree, "_run_alignment")
        calls = [
            n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "screen_input_quality"
        ]
        assert not calls, (
            "_run_alignment still calls screen_input_quality() directly; "
            "the scan must be dispatched through a background worker"
        )

    def test_run_alignment_branches_on_quality_gate_enabled(self):
        tree, _ = _load_main_window_module()
        node = _function_node(tree, "_run_alignment")
        branches = [
            n for n in ast.walk(node)
            if isinstance(n, ast.If) and "quality_gate_enabled" in ast.dump(n.test)
        ]
        assert branches, (
            "_run_alignment does not branch on state.quality_gate_enabled -- "
            "the checkbox controls nothing"
        )

    def test_disabled_branch_skips_straight_to_prediction(self):
        """When the gate is off, the scan must not run at all -- not run and
        be ignored, actually skipped."""
        tree, _ = _load_main_window_module()
        node = _function_node(tree, "_run_alignment")
        gate_if = next(
            n for n in ast.walk(node)
            if isinstance(n, ast.If) and "quality_gate_enabled" in ast.dump(n.test)
        )
        # The else-branch (gate disabled) must go straight to prediction and
        # must not itself invoke the quality scan.
        else_calls = {
            n.func.attr for n in ast.walk(ast.Module(body=gate_if.orelse, type_ignores=[]))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert "_start_prediction_worker" in else_calls
        assert "_run_quality_gate_scan" not in else_calls
        assert "screen_input_quality" not in else_calls


class TestTheScanRunsOffTheUIThread:
    def test_run_quality_gate_scan_uses_a_background_worker(self):
        tree, _ = _load_main_window_module()
        node = _function_node(tree, "_run_quality_gate_scan")
        call_names = {
            n.func.id for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "FunctionWorker" in call_names, (
            "the input quality scan must run inside a FunctionWorker, not "
            "synchronously on the UI thread"
        )

        attr_calls = [
            n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "start"
        ]
        assert attr_calls, "_run_quality_gate_scan never starts its worker"

    def test_the_scan_result_continues_on_to_prediction(self):
        tree, _ = _load_main_window_module()
        node = _function_node(tree, "_on_quality_gate_finished")
        calls = {
            n.func.attr for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert "_start_prediction_worker" in calls, (
            "a completed quality scan must still lead to prediction starting "
            "(after any warning prompt)"
        )

    def test_a_scan_failure_does_not_start_prediction_unconditionally(self):
        """A failed scan must not silently proceed as if nothing happened --
        it has to be reported, and only proceed if the user says so."""
        tree, _ = _load_main_window_module()
        node = _function_node(tree, "_on_quality_gate_error")
        reported = [
            n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"_show_logged_error", "_show_logged_exception", "_show_toast"}
        ]
        assert reported, "a failed quality scan is not reported to the user"

        # _start_prediction_worker must be reachable only inside a branch,
        # not called unconditionally at the top level of the handler.
        top_level_calls = {
            n.value.func.attr for n in node.body
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
        }
        assert "_start_prediction_worker" not in top_level_calls
