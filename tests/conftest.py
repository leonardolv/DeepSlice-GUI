from __future__ import annotations

import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _mock_qt_dialogs(monkeypatch):
    """Ensure all Qt modal dialogs are non-blocking across all automated test runs."""
    try:
        from PySide6 import QtWidgets
    except Exception:
        return

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *a, **kw: ""),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: ("", "")),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **kw: ("", "")),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda *a, **kw: QtWidgets.QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        staticmethod(lambda *a, **kw: QtWidgets.QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        staticmethod(lambda *a, **kw: QtWidgets.QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        staticmethod(lambda *a, **kw: QtWidgets.QMessageBox.StandardButton.No),
    )
