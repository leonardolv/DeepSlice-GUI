import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from DeepSlice.gui.flow_layout import FlowLayout


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make(app, n=6):
    host = QtWidgets.QWidget()
    lay = FlowLayout(host)
    for i in range(n):
        b = QtWidgets.QPushButton(f"button number {i}")
        lay.addWidget(b)
    return host, lay


def test_wraps_when_narrow_and_single_row_when_wide(app):
    host, lay = _make(app)
    one_row = lay.heightForWidth(5000)
    narrow = lay.heightForWidth(200)
    assert narrow > one_row


def test_minimum_width_is_widest_item_not_sum(app):
    host, lay = _make(app)
    total = sum(lay.itemAt(i).sizeHint().width() for i in range(lay.count()))
    assert lay.minimumSize().width() < total / 2


def test_geometry_keeps_items_inside_width(app):
    host, lay = _make(app)
    host.resize(220, 400)
    host.show()
    app.processEvents()
    for i in range(lay.count()):
        g = lay.itemAt(i).geometry()
        assert g.right() <= 220
    host.close()


def test_review_page_atlas_rows_use_flow_layout(app):
    from DeepSlice.gui.main_window import DeepSliceMainWindow

    w = DeepSliceMainWindow()
    try:
        for widget in (w.blend_slider, w.atlas_scale_slider, w.atlas_offset_x_slider):
            assert widget.minimumWidth() >= 140
            lay = widget.parentWidget().layout()
            assert lay is not None
        flows = [c for c in w.findChildren(QtWidgets.QLayout) if isinstance(c, FlowLayout)]
        assert len(flows) >= 3
    finally:
        w.close()
