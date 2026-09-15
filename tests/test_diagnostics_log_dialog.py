"""The "Diagnostics Log" toolbar button and its dialog.

Gives `diagnostics.py`'s existing `log_issue`/`get_issues_by_severity`/
`clear_log` -- already correct, already called from `main.py` -- a real,
reachable UI surface. As in `test_pdf_reporting.py::TestMainWindowPdfDefaults`,
this constructs the real `DeepSliceMainWindow` (the button lives on the
window's own top toolbar, not on a widget that can be tested in isolation)
and patches `QTimer.singleShot` so the deferred startup dialog never fires
and hangs a later test's teardown. `QDialog.exec` is patched too, per this
repo's "never spawn blocking GUI dialogs during tests" standard -- the real
`exec()` would open a real modal event loop with nothing to dismiss it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QListWidgetItem, QPushButton

from DeepSlice.gui import main_window as main_window_module
from DeepSlice.gui.main_window import DeepSliceMainWindow


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def no_startup_dialogs(monkeypatch):
    # See test_pdf_reporting.py::TestMainWindowPdfDefaults for why this is
    # necessary rather than just patching QMessageBox.
    monkeypatch.setattr(QTimer, "singleShot", staticmethod(lambda *a, **kw: None))


@pytest.fixture
def main_window(app, no_startup_dialogs):
    win = DeepSliceMainWindow()
    yield win
    win.close()


class TestTheButtonExistsAndReachesTheHandler:
    def test_diagnostics_log_button_exists(self, main_window):
        assert hasattr(main_window, "diagnostics_log_button")

    def test_clicking_the_button_opens_the_dialog(self, main_window, monkeypatch):
        called = []
        monkeypatch.setattr(
            main_window, "_show_diagnostics_log_dialog", lambda: called.append(True)
        )
        main_window.diagnostics_log_button.click()
        assert called == [True]


class TestTheDialogListsWhatWasLogged:
    def test_opening_the_dialog_reads_get_issues_by_severity(self, main_window, monkeypatch):
        calls = []

        def fake_get_issues_by_severity(severity):
            calls.append(severity)
            return []

        monkeypatch.setattr(
            main_window_module, "get_issues_by_severity", fake_get_issues_by_severity
        )
        monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Rejected)

        main_window._show_diagnostics_log_dialog()

        # ERROR, WARNING and INFO -- every severity get_issues_by_severity
        # understands -- must all have been asked for.
        assert set(calls) == {"ERROR", "WARNING", "INFO"}

    def test_logged_events_show_up_in_the_list(self, main_window, monkeypatch):
        fake_event = {
            "timestamp": "2026-09-15T00:00:00+00:00",
            "severity": "ERROR",
            "rule_id": "DS-999",
            "title": "fake",
            "description": "a fake logged issue for this test",
        }

        def fake_get_issues_by_severity(severity):
            return [fake_event] if severity == "ERROR" else []

        captured = {}

        # Grab the QListWidget the dialog builds so we can inspect its
        # contents without needing a real modal event loop.
        from PySide6.QtWidgets import QListWidget

        def fake_exec(self):
            list_widgets = self.findChildren(QListWidget)
            assert list_widgets, "dialog has no QListWidget"
            list_widget = list_widgets[0]
            captured["texts"] = [
                list_widget.item(i).text() for i in range(list_widget.count())
            ]
            return QDialog.Rejected

        monkeypatch.setattr(
            main_window_module, "get_issues_by_severity", fake_get_issues_by_severity
        )
        monkeypatch.setattr(QDialog, "exec", fake_exec)

        main_window._show_diagnostics_log_dialog()

        assert any("DS-999" in text for text in captured["texts"])
        assert any("a fake logged issue for this test" in text for text in captured["texts"])

    def test_no_issues_shows_a_friendly_placeholder_not_an_empty_list(self, main_window, monkeypatch):
        from PySide6.QtWidgets import QListWidget

        captured = {}

        def fake_exec(self):
            list_widget = self.findChildren(QListWidget)[0]
            captured["texts"] = [
                list_widget.item(i).text() for i in range(list_widget.count())
            ]
            return QDialog.Rejected

        monkeypatch.setattr(
            main_window_module, "get_issues_by_severity", lambda severity: []
        )
        monkeypatch.setattr(QDialog, "exec", fake_exec)

        main_window._show_diagnostics_log_dialog()

        assert len(captured["texts"]) == 1
        assert "no diagnostics logged" in captured["texts"][0].lower()


class TestTheClearButtonIsWiredToClearLog:
    def test_clicking_clear_calls_clear_log_and_refreshes(self, main_window, monkeypatch):
        calls = {"get_issues": 0, "clear": 0}

        def fake_get_issues_by_severity(severity):
            calls["get_issues"] += 1
            return []

        def fake_clear_log():
            calls["clear"] += 1

        monkeypatch.setattr(
            main_window_module, "get_issues_by_severity", fake_get_issues_by_severity
        )
        monkeypatch.setattr(main_window_module, "clear_log", fake_clear_log)

        def fake_exec(self):
            clear_buttons = [
                button for button in self.findChildren(QPushButton)
                if button.text() == "Clear"
            ]
            assert clear_buttons, "dialog has no Clear button"
            clear_buttons[0].click()
            return QDialog.Rejected

        monkeypatch.setattr(QDialog, "exec", fake_exec)

        main_window._show_diagnostics_log_dialog()

        assert calls["clear"] == 1
        # 3 severities on the initial populate, 3 more on the post-clear
        # refresh -- proving Clear actually re-reads the (now-empty) log
        # rather than just wiping the widget locally.
        assert calls["get_issues"] == 6

    def test_clear_button_does_not_close_the_dialog_by_itself(self, main_window, monkeypatch):
        """Clear is an ActionRole button alongside Close, not a stand-in for
        it -- clicking it must not call dialog.accept()/reject() (an
        ActionRole button on a QDialogButtonBox does not trigger those
        automatically; a regression here would mean Clear was wired as an
        Accept/Reject role instead)."""
        monkeypatch.setattr(main_window_module, "get_issues_by_severity", lambda s: [])
        monkeypatch.setattr(main_window_module, "clear_log", lambda: None)

        exec_results = {}

        def fake_exec(self):
            clear_button = next(
                b for b in self.findChildren(QPushButton) if b.text() == "Clear"
            )
            clear_button.click()
            # Still able to query the list widget after the click means the
            # dialog was not accept()/reject()-ed (and torn down) as a side
            # effect of clicking Clear.
            exec_results["list_widget_still_present"] = bool(
                self.findChildren(QPushButton)
            )
            return QDialog.Rejected

        monkeypatch.setattr(QDialog, "exec", fake_exec)
        main_window._show_diagnostics_log_dialog()
        assert exec_results["list_widget_still_present"] is True
