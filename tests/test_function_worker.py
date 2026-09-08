"""Pins the `FunctionWorker.run` injection fix (see AGENT_TASK_LOG.md,
2026-09-08).

`inject_callbacks=True` used to add `progress_callback`/`log_callback`/
`cancel_check` to every call unconditionally. That was safe only as long as
every `inject_callbacks=True` target declared all three - true of
`_run_prediction_task`, but `_atlas_preview_task` and `_load_quint_task`
(both added later) declare only the first two and have no `**kwargs`
catch-all. So the blind `cancel_check` injection raised `TypeError` on
*every single call* to either, silently caught by `run`'s own broad
`except Exception` and surfaced to the user as a generic "Failed to load
QuickNII file"/atlas-preview error - loading a QuickNII/QuINT session, and
previewing the atlas, never actually worked.

`FunctionWorker` now only injects a callback a target function actually
declared (or can absorb via `**kwargs`), via `_accepts_kwarg`. The
`cancel_check` auto-injection itself is gone entirely, along with
`request_cancel`/`is_cancel_requested`/`_cancel_event`: nothing anywhere in
the app ever called `request_cancel()`, so the mechanism it fed was
permanently a no-op - deleted rather than wired up, since a real
cancellation path already exists for the one caller that uses `cancel_check`
(`_run_prediction_task`'s own `self._prediction_cancel_event`).

These tests run `FunctionWorker.run()` synchronously (it is a plain method,
not dispatched through a real `QThreadPool` here) against small stand-in
functions shaped like the app's real `inject_callbacks=True` targets, so no
QApplication/thread pool is needed beyond what constructing a `QObject`
(`WorkerSignals`) requires.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from DeepSlice.gui.workers import FunctionWorker


def _run_and_collect(worker: FunctionWorker):
    results = {"finished": None, "error": None}
    worker.signals.finished.connect(lambda r: results.__setitem__("finished", r))
    worker.signals.error.connect(lambda e: results.__setitem__("error", e))
    worker.run()
    return results


class TestInjectCallbacksOnlyAddsWhatTheFunctionDeclares:
    def test_a_function_without_cancel_check_no_longer_crashes(self):
        """Regression: this exact shape (progress_callback/log_callback
        declared, no cancel_check, no **kwargs) is `_load_quint_task`'s and
        `_atlas_preview_task`'s real signature, and used to raise TypeError
        on every call.
        """

        def target(filename, progress_callback=None, log_callback=None):
            return f"loaded {filename}"

        worker = FunctionWorker(target, "session.quint", inject_callbacks=True)
        results = _run_and_collect(worker)

        assert results["error"] is None, results["error"]
        assert results["finished"] == "loaded session.quint"

    def test_a_function_that_declares_cancel_check_still_receives_none(self):
        """`_run_prediction_task`'s shape: cancel_check is still a valid
        parameter for a direct caller to use, it's just never auto-injected
        by the worker any more (nothing ever fed it - see module docstring).
        """

        def target(options, progress_callback=None, log_callback=None, cancel_check=None):
            return {"cancel_check_was": cancel_check}

        worker = FunctionWorker(target, {"x": 1}, inject_callbacks=True)
        results = _run_and_collect(worker)

        assert results["error"] is None, results["error"]
        assert results["finished"] == {"cancel_check_was": None}

    def test_progress_and_log_callbacks_are_still_injected(self):
        seen = {}

        def target(progress_callback=None, log_callback=None):
            progress_callback(1, 2, "phase")
            log_callback("hello")
            seen["called"] = True
            return "ok"

        worker = FunctionWorker(target, inject_callbacks=True)
        progress_events = []
        log_events = []
        worker.signals.progress.connect(lambda c, t, p: progress_events.append((c, t, p)))
        worker.signals.log.connect(log_events.append)
        results = _run_and_collect(worker)

        assert results["error"] is None, results["error"]
        assert seen["called"] is True
        assert progress_events == [(1, 2, "phase")]
        assert log_events == ["hello"]

    def test_a_function_with_kwargs_catch_all_receives_everything(self):
        captured = {}

        def target(**kwargs):
            captured.update(kwargs)
            return "ok"

        worker = FunctionWorker(target, inject_callbacks=True)
        results = _run_and_collect(worker)

        assert results["error"] is None, results["error"]
        assert "progress_callback" in captured
        assert "log_callback" in captured

    def test_a_manually_supplied_cancel_check_is_left_untouched(self):
        """`request_cancel`/`is_cancel_requested` are gone, but a caller
        passing its own `cancel_check` directly through kwargs (as
        `_run_prediction_task`'s own caller does not, but could) must not
        have it clobbered.
        """
        sentinel = object()

        def target(progress_callback=None, log_callback=None, cancel_check=None):
            return cancel_check

        worker = FunctionWorker(target, inject_callbacks=True, cancel_check=sentinel)
        results = _run_and_collect(worker)

        assert results["error"] is None, results["error"]
        assert results["finished"] is sentinel

    def test_inject_callbacks_false_does_not_touch_kwargs_at_all(self):
        def target(a, b):
            return a + b

        worker = FunctionWorker(target, 1, 2)
        results = _run_and_collect(worker)

        assert results["error"] is None, results["error"]
        assert results["finished"] == 3


class TestRequestCancelWasRemoved:
    def test_function_worker_has_no_cancel_api_left(self):
        def target():
            return None

        worker = FunctionWorker(target)
        assert not hasattr(worker, "request_cancel")
        assert not hasattr(worker, "is_cancel_requested")
        assert not hasattr(worker, "_cancel_event")
