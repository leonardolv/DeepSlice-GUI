"""Pins DeepSliceMainWindow._load_session_file's ``.json`` branch.

A file that declares ``session_format: "deepslice_gui_v1"`` but fails
partway through ``state.load_session_dict`` used to be swallowed by a bare
``except Exception: pass`` and silently re-parsed as a QuickNII export on
top of whatever state that partial load had already mutated. These tests
call the unbound method against a lightweight stub object rather than a
real ``DeepSliceMainWindow`` - constructing the real window pulls in the
full Qt UI and model loading, which nothing else in this suite does either
(see test_session_roundtrip.py's comment on testing at the state layer).
The stub only needs to satisfy the attributes/methods the method under
test actually touches.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from DeepSlice.gui import main_window as main_window_module
from DeepSlice.gui.main_window import DeepSliceMainWindow
from DeepSlice.gui.state import DeepSliceAppState


class _StubWindow:
    def __init__(self) -> None:
        self.state = DeepSliceAppState()
        self._set_session_io_busy = MagicMock()
        self._load_anchor_targets_from_payload = MagicMock()
        self._apply_state_to_widgets = MagicMock()
        self._curation_modified = None
        self._session_base_text = ""
        self._update_session_status = MagicMock()
        self._refresh_all_views = MagicMock()
        self._add_recent_session = MagicMock()
        self._show_logged_exception = MagicMock()
        self._append_console_log = MagicMock()
        self._on_load_quint_error = MagicMock()
        self._on_load_quint_finished = MagicMock()
        self._load_quint_task = MagicMock()
        self.thread_pool = MagicMock()
        self._track_worker = MagicMock()
        self._baseline_predictions = None


def _load(stub: _StubWindow, filename: str) -> None:
    DeepSliceMainWindow._load_session_file(stub, filename)


def test_a_valid_deepslice_session_loads_without_touching_quicknii(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_worker_cls = MagicMock()
    monkeypatch.setattr(main_window_module, "FunctionWorker", fake_worker_cls)

    session_path = tmp_path / "session.json"
    session_path.write_text(
        json.dumps({"session_format": "deepslice_gui_v1", "image_paths": []}),
        encoding="utf-8",
    )
    stub = _StubWindow()

    _load(stub, str(session_path))

    stub._show_logged_exception.assert_not_called()
    fake_worker_cls.assert_not_called()
    stub._refresh_all_views.assert_called_once()


def test_a_broken_deepslice_session_reports_the_failure_and_does_not_fall_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_session_dict raising must not be silently swallowed and re-tried
    as QuickNII - that was the bug: the file declares itself a DeepSlice
    session, so a failure there is a real error, not "try something else".
    """
    fake_worker_cls = MagicMock()
    monkeypatch.setattr(main_window_module, "FunctionWorker", fake_worker_cls)

    session_path = tmp_path / "broken_session.json"
    # Not a well-formed DeepSlice session payload - load_session_dict is
    # expected to raise on it (missing/invalid required structure).
    session_path.write_text(
        json.dumps({"session_format": "deepslice_gui_v1", "predictions": "not-a-table"}),
        encoding="utf-8",
    )
    stub = _StubWindow()

    _load(stub, str(session_path))

    stub._show_logged_exception.assert_called_once()
    assert stub._show_logged_exception.call_args.kwargs["title"] == "Load Session"
    fake_worker_cls.assert_not_called()


def test_a_non_deepslice_json_file_falls_through_to_quicknii(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .json file with no (or a different) session_format is legitimately
    not ours - QuickNII files are also .json/.xml, so falling through here
    is expected, unchanged behaviour.
    """
    fake_worker_cls = MagicMock()
    monkeypatch.setattr(main_window_module, "FunctionWorker", fake_worker_cls)

    quicknii_path = tmp_path / "not_a_session.json"
    quicknii_path.write_text(json.dumps({"some": "quicknii-shaped-data"}), encoding="utf-8")
    stub = _StubWindow()

    _load(stub, str(quicknii_path))

    stub._show_logged_exception.assert_not_called()
    fake_worker_cls.assert_called_once()


def test_a_file_that_is_not_valid_json_falls_through_to_quicknii(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .json extension with unparsable content (e.g. actually XML/QuickNII
    despite the extension) must not be reported as a "failed session load" -
    it was never claiming to be one.
    """
    fake_worker_cls = MagicMock()
    monkeypatch.setattr(main_window_module, "FunctionWorker", fake_worker_cls)

    not_json_path = tmp_path / "actually_xml.json"
    not_json_path.write_text("<QuickNII><slice/></QuickNII>", encoding="utf-8")
    stub = _StubWindow()

    _load(stub, str(not_json_path))

    stub._show_logged_exception.assert_not_called()
    fake_worker_cls.assert_called_once()
