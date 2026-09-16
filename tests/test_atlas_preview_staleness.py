"""Pins the atlas-preview staleness guard on `_on_atlas_error`/`_on_atlas_progress`.

`_request_atlas_preview` fires on every curation-row selection change (arrow
key navigation, clicking a row, changing the atlas volume, toggling the
blend overlay) - a user scrubbing quickly through many sections can fire
several overlapping `FunctionWorker` atlas-preview tasks in succession.
`_atlas_request_token` exists precisely to tell a superseded request's
result apart from the current one, and `_on_atlas_ready` already does that
check before touching the atlas viewer. `_on_atlas_error`/`_on_atlas_progress`
did not: a stale request's late failure (e.g. a transient download error on
a row the user already scrubbed past) unconditionally blanked the atlas
viewer and showed "Atlas: failed", discarding whatever a newer, already-
succeeded request had just rendered; a stale progress tick could likewise
overwrite the label with an outdated percentage after the current request
had already finished.

As in test_load_session_file.py/test_drop_event_toast.py, DeepSliceMainWindow
is exercised via the unbound method against a lightweight stub rather than a
real instance - constructing the real window pulls in the full Qt UI, which
nothing else in this suite does either.
"""

from __future__ import annotations

from types import MethodType
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6")

from DeepSlice.gui.main_window import DeepSliceMainWindow
from DeepSlice.gui.workers import FunctionWorker


class _FakePredictions:
    def __init__(self, length: int) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length


class _FakeState:
    def __init__(self, length: int = 5) -> None:
        self.predictions = _FakePredictions(length)


class _SyncThreadPool:
    """Runs a QRunnable synchronously instead of on a real worker thread."""

    def start(self, worker: FunctionWorker) -> None:
        worker.run()


class _StubWindow:
    """Satisfies only what `_request_atlas_preview` and its handlers touch."""

    def __init__(self) -> None:
        self.state = _FakeState()
        self._linearity_payload = None
        self._atlas_request_token = 0
        self.thread_pool = _SyncThreadPool()

        self.enable_atlas_preview_checkbox = MagicMock()
        self.enable_atlas_preview_checkbox.isChecked.return_value = True
        self.atlas_volume_combo = MagicMock()
        self.atlas_volume_combo.currentText.return_value = "MRI"
        self.atlas_slice_info_label = MagicMock()
        self.atlas_viewer = MagicMock()
        self.atlas_coords_label = MagicMock()

        self._latest_atlas_slice = None
        self._latest_atlas_meta = None
        self._record_error = MagicMock()
        self._append_console_log = MagicMock()
        self._on_atlas_ready = MagicMock()
        # The real `_request_atlas_preview` wires its worker's progress/error
        # signals straight to `self._on_atlas_progress`/`self._on_atlas_error`
        # - bind the REAL production methods here so the wiring test below
        # exercises the actual staleness guard, not a mock of it.
        self._on_atlas_progress = MethodType(DeepSliceMainWindow._on_atlas_progress, self)
        self._on_atlas_error = MethodType(DeepSliceMainWindow._on_atlas_error, self)

        # `_track_worker` needs `active_workers`/`_set_global_busy`, neither
        # of which this test cares about - stub it to a no-op like the other
        # main_window.py unbound-method tests do for unrelated bookkeeping.
        self._track_worker = MagicMock()

        # Overridden per-test to control what the "atlas task" does.
        self.atlas_task_behavior = None

    def _atlas_preview_task(self, depth_value, volume_key, request_token, progress_callback=None, log_callback=None):
        return self.atlas_task_behavior(depth_value, volume_key, request_token, progress_callback, log_callback)


def _request(stub: _StubWindow, row_index: int) -> int:
    """Calls the real `_request_atlas_preview` and returns the token it minted."""
    DeepSliceMainWindow._request_atlas_preview(stub, row_index)
    return stub._atlas_request_token


class TestStaleAtlasErrorIsIgnored:
    def test_an_old_requests_failure_does_not_blank_a_newer_successful_preview(self):
        stub = _StubWindow()

        # First request (token 1) will fail, but only once a second request
        # (token 2) has already been minted - simulating the user scrubbing
        # to a new row before the first row's atlas fetch reports back.
        def first_task(depth_value, volume_key, request_token, progress_callback, log_callback):
            # By the time this "network call" fails, a newer request already
            # exists - `_atlas_request_token` on the real window would have
            # moved on. Mint the second request's token here to reproduce
            # that ordering precisely.
            _request(stub, 1)
            raise RuntimeError("atlas download failed")

        stub.atlas_task_behavior = first_task
        DeepSliceMainWindow._request_atlas_preview(stub, 0)

        # The stale (token 1) failure must not have touched the atlas view -
        # it belongs to a request `_atlas_request_token` (now 2) has moved
        # past.
        stub._record_error.assert_not_called()
        stub.atlas_viewer.clear_with_text.assert_not_called()
        # "Atlas: failed" must never have been set from the stale error.
        failed_calls = [
            call for call in stub.atlas_slice_info_label.setText.call_args_list if call.args[0] == "Atlas: failed"
        ]
        assert failed_calls == []

    def test_a_currently_valid_error_still_reports_normally(self):
        stub = _StubWindow()

        def failing_task(depth_value, volume_key, request_token, progress_callback, log_callback):
            raise RuntimeError("atlas download failed")

        stub.atlas_task_behavior = failing_task
        DeepSliceMainWindow._request_atlas_preview(stub, 0)

        # No newer request superseded this one, so the failure is real and
        # must still be surfaced exactly as before this fix.
        stub._record_error.assert_called_once()
        stub.atlas_viewer.clear_with_text.assert_called_once()
        stub.atlas_slice_info_label.setText.assert_any_call("Atlas: failed")


class TestStaleAtlasProgressIsIgnored:
    def test_a_late_progress_tick_from_a_superseded_request_does_not_overwrite_the_label(self):
        stub = _StubWindow()

        def task(depth_value, volume_key, request_token, progress_callback, log_callback):
            # A newer request supersedes this one mid-flight...
            _request(stub, 1)
            # ...and then this (now-stale) request's own progress tick
            # arrives late.
            progress_callback(50, 100, "atlas-download")
            return {
                "image": None,
                "slice_index": 0,
                "depth": 0.0,
                "shape": (1, 1, 1),
                "volume_label": "MRI",
                "request_token": request_token,
            }

        stub.atlas_task_behavior = task
        DeepSliceMainWindow._request_atlas_preview(stub, 0)

        progress_texts = [call.args[0] for call in stub.atlas_slice_info_label.setText.call_args_list]
        assert not any("Atlas download: 50" in text for text in progress_texts)

    def test_a_current_progress_tick_still_updates_the_label(self):
        stub = _StubWindow()

        def task(depth_value, volume_key, request_token, progress_callback, log_callback):
            progress_callback(50, 100, "atlas-download")
            return {
                "image": None,
                "slice_index": 0,
                "depth": 0.0,
                "shape": (1, 1, 1),
                "volume_label": "MRI",
                "request_token": request_token,
            }

        stub.atlas_task_behavior = task
        DeepSliceMainWindow._request_atlas_preview(stub, 0)

        progress_texts = [call.args[0] for call in stub.atlas_slice_info_label.setText.call_args_list]
        assert any("Atlas download: 50" in text for text in progress_texts)


class TestOnAtlasHandlersDirectly:
    """Unit-level checks on the handlers in isolation, independent of the
    `_request_atlas_preview` wiring above."""

    def test_on_atlas_error_ignores_a_stale_token(self):
        stub = _StubWindow()
        stub._atlas_request_token = 2

        DeepSliceMainWindow._on_atlas_error(stub, "boom", 1)

        stub._record_error.assert_not_called()
        stub.atlas_viewer.clear_with_text.assert_not_called()

    def test_on_atlas_error_reports_when_token_matches(self):
        stub = _StubWindow()
        stub._atlas_request_token = 2

        DeepSliceMainWindow._on_atlas_error(stub, "boom", 2)

        stub._record_error.assert_called_once_with("Atlas preview task failed", "boom")
        stub.atlas_viewer.clear_with_text.assert_called_once()

    def test_on_atlas_error_with_no_token_behaves_as_before(self):
        """A caller that never passes a token (defensive default) must not
        regress to always-ignoring - only an explicit mismatch is stale."""
        stub = _StubWindow()
        stub._atlas_request_token = 2

        DeepSliceMainWindow._on_atlas_error(stub, "boom")

        stub._record_error.assert_called_once()

    def test_on_atlas_progress_ignores_a_stale_token(self):
        stub = _StubWindow()
        stub._atlas_request_token = 2

        DeepSliceMainWindow._on_atlas_progress(stub, 10, 100, "atlas-download", 1)

        stub.atlas_slice_info_label.setText.assert_not_called()

    def test_on_atlas_progress_updates_when_token_matches(self):
        stub = _StubWindow()
        stub._atlas_request_token = 2

        DeepSliceMainWindow._on_atlas_progress(stub, 10, 100, "atlas-download", 2)

        stub.atlas_slice_info_label.setText.assert_called_once()
