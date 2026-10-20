import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from DeepSlice.gui.flow_layout import FlowLayout
from DeepSlice.gui.main_window import DeepSliceMainWindow


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def window(app):
    w = DeepSliceMainWindow()
    yield w
    w.close()


@pytest.mark.parametrize(
    "button_name",
    ["curation_prev_button", "auto_flag_low_conf_button", "set_anchor_button"],
)
def test_button_rows_wrap_instead_of_summing(window, button_name):
    button = getattr(window, button_name)
    layout = button.parentWidget().layout()
    row = None
    for i in range(layout.count()):
        child = layout.itemAt(i).layout()
        if child is not None and any(
            child.itemAt(j).widget() is button for j in range(child.count())
        ):
            row = child
    assert isinstance(row, FlowLayout)


def test_left_column_minimum_width_is_bounded(window):
    column = window.curation_prev_button.parentWidget()
    assert column.minimumSizeHint().width() < 350
