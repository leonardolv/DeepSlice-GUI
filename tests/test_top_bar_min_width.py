import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from DeepSlice.gui.main_window import DeepSliceMainWindow


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def window(app):
    w = DeepSliceMainWindow()
    yield w
    w.close()


def test_top_bar_minimum_width_does_not_sum_all_actions(window):
    bar = window.findChild(QtWidgets.QFrame, "TopBar")
    total = sum(w.sizeHint().width() for w in window._top_action_widgets)
    assert bar.minimumSizeHint().width() < total / 2
    assert window.minimumSizeHint().width() < 1000


def test_top_bar_actions_stay_inside_bar_at_narrow_and_wide(window, app):
    bar = window.findChild(QtWidgets.QFrame, "TopBar")
    window.show()
    for width, expect_wide in ((800, False), (2200, True)):
        window.resize(width, 800)
        app.processEvents()
        window._update_top_bar_layout()
        app.processEvents()
        assert window._top_bar_wide is expect_wide
        for w in window._top_action_widgets:
            if w.isHidden():
                continue
            origin = w.mapTo(bar, w.rect().topLeft())
            assert origin.x() >= 0
            assert origin.x() + w.width() <= bar.width() + 1
            assert origin.y() + w.height() <= bar.height() + 1
