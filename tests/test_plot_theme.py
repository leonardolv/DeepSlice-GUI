"""Curation plots follow the active theme instead of staying dark in the light theme."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from matplotlib.colors import to_hex
from PySide6.QtWidgets import QApplication

from DeepSlice.gui.main_window import DeepSliceMainWindow


def test_plots_restyle_when_theme_changes():
    QApplication.instance() or QApplication([])
    win = DeepSliceMainWindow()
    try:
        win._theme_name = "dark"
        win._apply_theme()
        dark = DeepSliceMainWindow.PLOT_PALETTES["dark"]
        assert to_hex(win.linearity_figure.get_facecolor()).lower() == dark["figure"].lower()
        assert to_hex(win.hist_axis.get_facecolor()).lower() == dark["axis"].lower()

        win._theme_name = "light"
        win._apply_theme()
        light = DeepSliceMainWindow.PLOT_PALETTES["light"]
        assert to_hex(win.linearity_figure.get_facecolor()).lower() == light["figure"].lower()
        assert to_hex(win.hist_figure.get_facecolor()).lower() == light["figure"].lower()
        assert to_hex(win.linearity_axis.get_facecolor()).lower() == light["axis"].lower()
        assert to_hex(win.linearity_axis.spines["left"].get_edgecolor()).lower() == light["spine"].lower()
    finally:
        win.close()
        win.deleteLater()


def test_palettes_have_same_keys_and_contrast():
    d, l = DeepSliceMainWindow.PLOT_PALETTES["dark"], DeepSliceMainWindow.PLOT_PALETTES["light"]
    assert d.keys() == l.keys()
    assert l["axis"].lower() != d["axis"].lower()
