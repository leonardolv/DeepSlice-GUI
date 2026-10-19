"""A layout that wraps its items onto additional rows when width runs out."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QSizePolicy, QWidgetItem


class FlowLayout(QLayout):
    """Left-to-right layout that wraps; minimum width is the widest single item."""

    def __init__(self, parent=None, spacing: int = 8):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing
        # When True, sizeHint() is the width of all items on one row (the
        # minimum width stays the widest item, so the layout can still wrap).
        self.prefer_single_row = False
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, apply=True)

    def sizeHint(self):
        if not self.prefer_single_row:
            return self.minimumSize()
        size = QSize(0, 0)
        visible = 0
        for item in self._items:
            if item.widget() is not None and item.widget().isHidden():
                continue
            hint = item.sizeHint()
            size = QSize(size.width() + hint.width(), max(size.height(), hint.height()))
            visible += 1
        if visible > 1:
            size.setWidth(size.width() + self._spacing * (visible - 1))
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect, apply):
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_h = area.x(), area.y(), 0
        for item in self._items:
            if item.widget() is not None and item.widget().isHidden():
                continue
            hint = item.sizeHint()
            w = min(hint.width(), max(area.width(), 1))
            if x + w > area.right() + 1 and row_h > 0:
                x = area.x()
                y += row_h + self._spacing
                row_h = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), QSize(w, hint.height())))
            x += w + self._spacing
            row_h = max(row_h, hint.height())
        return y + row_h - rect.y() + m.bottom()
