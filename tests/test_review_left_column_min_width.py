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
    assert column.minimumSizeHint().width() < column.sizeHint().width()


def test_flags_action_row_is_flow_layout(window):
    button = window.apply_bad_sections_button
    layout = button.parentWidget().layout()
    row = None
    for i in range(layout.count()):
        child = layout.itemAt(i).layout()
        if child is not None and any(
            child.itemAt(j).widget() is button for j in range(child.count())
        ):
            row = child
    assert isinstance(row, FlowLayout)
    widgets = [row.itemAt(j).widget() for j in range(row.count())]
    assert window.detect_outliers_button in widgets
    assert window.reset_flags_button in widgets


def test_collapsible_groups_toggle_contents(window):
    assert window.controls_group.isCheckable()
    assert window.anchor_group.isCheckable()

    window.controls_group.setChecked(False)
    assert window.normalize_angles_button.isHidden()

    window.controls_group.setChecked(True)
    assert not window.normalize_angles_button.isHidden()

    window.anchor_group.setChecked(False)
    assert window.set_anchor_button.isHidden()

    window.anchor_group.setChecked(True)
    assert not window.set_anchor_button.isHidden()
