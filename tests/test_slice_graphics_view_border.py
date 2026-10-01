"""Regression coverage for `SliceGraphicsView`'s confidence-level border.

`set_image`/`set_array_image` (`gui/main_window.py`) draw a colored border
around the histology/confidence-overlay previews to flag high/medium/low
confidence sections during curation review. Both call sites used to hand
`QGraphicsScene.addRect` a bare `QColor` instead of a `QPen`, which Qt
implicitly widens to a default-constructed, 1px-wide `QPen` - so the
intended bold border (a `pen_width = 4` local was computed in `set_image`
and then never used) rendered as a near-invisible hairline instead. This
file drives the real `SliceGraphicsView` widget (no mocks) and inspects the
actual `QGraphicsRectItem` Qt added to the scene.
"""
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QGraphicsRectItem

from DeepSlice.gui.main_window import SliceGraphicsView

EXPECTED_BORDER_WIDTH = 4


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def _border_rect_items(view: SliceGraphicsView):
    return [item for item in view._scene.items() if isinstance(item, QGraphicsRectItem)]


def _make_png(path):
    image = QImage(12, 8, QImage.Format_RGB888)
    image.fill(QColor("#123456"))
    assert image.save(str(path), "PNG")
    return str(path)


class TestSetImageBorder:
    def test_border_pen_width_matches_the_intended_four_pixels(self, qapp, tmp_path):
        view = SliceGraphicsView()
        try:
            image_path = _make_png(tmp_path / "slice.png")
            border_color = QColor("#D33E56")

            view.set_image(image_path, border_color=border_color)

            rects = _border_rect_items(view)
            assert len(rects) == 1, "expected exactly one border rect item"
            pen = rects[0].pen()
            assert pen.widthF() == EXPECTED_BORDER_WIDTH
            assert pen.color().name() == border_color.name()
        finally:
            view.deleteLater()

    def test_no_border_item_when_border_color_is_none(self, qapp, tmp_path):
        view = SliceGraphicsView()
        try:
            image_path = _make_png(tmp_path / "slice.png")

            view.set_image(image_path, border_color=None)

            assert _border_rect_items(view) == []
        finally:
            view.deleteLater()

    def test_a_thin_hairline_border_is_the_pre_fix_regression_shape(self, qapp, tmp_path):
        """Documents the exact bug: a bare QColor widens to a 1px QPen."""
        from PySide6.QtGui import QPen

        implicit_pen = QPen(QColor("#D33E56"))
        assert implicit_pen.widthF() != EXPECTED_BORDER_WIDTH


class TestSetArrayImageBorder:
    def test_border_pen_width_matches_the_intended_four_pixels(self, qapp):
        view = SliceGraphicsView()
        try:
            array = (np.random.rand(6, 10) * 255).astype(np.uint8)
            border_color = QColor("#2CC784")

            view.set_array_image(array, border_color=border_color)

            rects = _border_rect_items(view)
            assert len(rects) == 1, "expected exactly one border rect item"
            pen = rects[0].pen()
            assert pen.widthF() == EXPECTED_BORDER_WIDTH
            assert pen.color().name() == border_color.name()
        finally:
            view.deleteLater()

    def test_no_border_item_when_border_color_is_none(self, qapp):
        view = SliceGraphicsView()
        try:
            array = (np.random.rand(6, 10) * 255).astype(np.uint8)

            view.set_array_image(array, border_color=None)

            assert _border_rect_items(view) == []
        finally:
            view.deleteLater()
