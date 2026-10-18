"""Interactive per-frame mask count timeline and virtualized change list."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from auto_mosaic.video_analysis import DetectionChange, FrameSummary


def time_text(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    minutes, rest = divmod(milliseconds, 60000)
    seconds, ms = divmod(rest, 1000)
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"


class ChangeTableModel(QAbstractTableModel):
    HEADERS = ("時刻", "変化", "マスク数")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.changes: list[DetectionChange] = []
        self.fps = 1.0

    def set_changes(self, changes, fps):
        self.beginResetModel()
        self.changes, self.fps = changes, fps
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.changes)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 3

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        change = self.changes[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return (time_text(change.frame / self.fps), change.description,
                    f"{change.before} → {change.after}")[index.column()]
        if role == Qt.ItemDataRole.ForegroundRole and change.after == 0:
            return QColor("#ffbc70")
        return None


class MaskCountTimeline(QWidget):
    frame_requested = Signal(int)
    window_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.counts = np.empty(0, np.int32)
        self.areas = np.empty(0, float)
        self.show_counts = self.show_areas = True
        self.fps = 1.0
        self.current = 0
        self.window_seconds = 0
        self.start = 0
        self.edits = {}
        self.propagation_start = None
        self.propagation_range = None
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("時点ごとの検出マスク数と検出面積")

    def set_frames(self, frames: list[FrameSummary], fps: float, areas=None):
        self.counts = np.array([frame.count for frame in frames], dtype=np.int32)
        self.areas = np.asarray(areas if areas is not None else np.zeros(len(frames)), dtype=float)
        self.fps, self.current, self.start = fps, 0, 0
        self.update()

    @property
    def span(self):
        return max(1, min(len(self.counts), round(self.window_seconds * self.fps))) if self.window_seconds else max(1, len(self.counts))

    def set_edits(self, edits):
        self.edits = dict(edits)
        self.update()

    def set_propagation(self, start, bounds):
        self.propagation_start, self.propagation_range = start, bounds
        self.update()

    def set_series_visible(self, counts: bool, areas: bool):
        self.show_counts, self.show_areas = counts, areas
        self.update()

    def set_window_seconds(self, seconds: float):
        self.window_seconds = seconds
        self.start = max(0, min(self.current - self.span // 2, len(self.counts) - self.span))
        self.update()

    def set_current(self, frame: int):
        self.current = frame
        if not self.start <= frame < self.start + self.span:
            self.start = max(0, min(frame - self.span // 2, len(self.counts) - self.span))
        self.update()

    def plot_rect(self):
        return QRectF(36, 38, max(1, self.width() - 96), max(1, self.height() - 68))

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if not len(self.counts) or not delta:
            event.ignore()
            return
        rect = self.plot_rect()
        fraction = max(0.0, min(1.0, (event.position().x() - rect.left()) / rect.width()))
        anchor = self.start + fraction * self.span
        span = max(1, min(len(self.counts), round(self.span * (0.8 if delta > 0 else 1.25))))
        if span == self.span:
            span = max(1, min(len(self.counts), span + (-1 if delta > 0 else 1)))
        self.window_seconds = 0 if span == len(self.counts) else span / self.fps
        self.start = max(0, min(len(self.counts) - span, round(anchor - fraction * span)))
        self.window_changed.emit(self.window_seconds)
        self.update()
        event.accept()

    def frame_at(self, x):
        rect = self.plot_rect()
        offset = int((x - rect.left()) / rect.width() * self.span)
        return max(0, min(len(self.counts) - 1, self.start + max(0, min(self.span - 1, offset))))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and len(self.counts):
            self.frame_requested.emit(self.frame_at(event.position().x()))
            event.accept()

    def mouseMoveEvent(self, event):
        if not len(self.counts):
            return
        frame = self.frame_at(event.position().x())
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.frame_requested.emit(frame)
        QToolTip.showText(event.globalPosition().toPoint(),
                         f"{time_text(frame / self.fps)}  検出マスク {self.counts[frame]}件  面積 {self.areas[frame]:.2f}%  / フレーム {frame + 1}"
                         + (" / 手動編集" if self.edits.get(frame) == "manual" else " / 自動補完" if frame in self.edits else ""), self)

    def keyPressEvent(self, event):
        step = -1 if event.key() == Qt.Key.Key_Left else 1 if event.key() == Qt.Key.Key_Right else 0
        if step and len(self.counts):
            self.frame_requested.emit(max(0, min(len(self.counts) - 1, self.current + step)))
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111319"))
        rect = self.plot_rect()
        if not len(self.counts):
            painter.setPen(QColor("#aab2c2"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "解析後にマスク数・面積を表示します")
            return
        visible = self.counts[self.start:self.start + self.span]
        maximum = max(1, int(self.counts.max()))
        for value in sorted({0, maximum // 2, maximum}):
            y = rect.bottom() - value / maximum * rect.height()
            painter.setPen(QColor("#303744"))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(QColor("#b8c0cf"))
            if self.show_counts:
                painter.drawText(QRectF(0, y - 9, 28, 18), Qt.AlignmentFlag.AlignRight, str(value))
        # Aggregate min AND max per screen pixel, so narrow dropouts are not
        # erased by averaging or by drawing only the maximum count.
        columns = min(max(1, int(rect.width())), len(visible))
        for column in range(columns if self.show_counts else 0):
            left = column * len(visible) // columns
            right = max(left + 1, (column + 1) * len(visible) // columns)
            minimum, peak = int(visible[left:right].min()), int(visible[left:right].max())
            x = rect.left() + column / columns * rect.width()
            width = max(1, rect.width() / columns)
            if peak:
                height = peak / maximum * rect.height()
                painter.fillRect(QRectF(x, rect.bottom() - height, width, height), QColor("#448bbb"))
                painter.setPen(QColor("#8bd1ff"))
                painter.drawLine(QPointF(x, rect.bottom() - height), QPointF(x + width, rect.bottom() - height))
            if minimum == 0:
                painter.fillRect(QRectF(x, rect.bottom() - 3, width, 3), QColor("#ffbc70"))
            elif minimum < peak:
                painter.setPen(QColor("#ffbc70"))
                painter.drawLine(QPointF(x, rect.bottom() - peak / maximum * rect.height()),
                                 QPointF(x, rect.bottom() - minimum / maximum * rect.height()))
        if self.show_areas:
            area_max = max(1.0, float(self.areas.max()))
            painter.setPen(QPen(QColor("#ee91d7"), 2))
            path = QPainterPath()
            visible_areas = self.areas[self.start:self.start + self.span]
            # Retain extrema in time order at each screen pixel, including short dropouts.
            for column in range(columns):
                first = column * len(visible_areas) // columns
                stop = max(first + 1, (column + 1) * len(visible_areas) // columns)
                chunk = visible_areas[first:stop]
                for index in sorted({first, first + int(chunk.argmin()), first + int(chunk.argmax()), stop - 1}):
                    point = QPointF(rect.left() + (index + .5) / self.span * rect.width(),
                                    rect.bottom() - visible_areas[index] / area_max * rect.height())
                    if path.elementCount() == 0:
                        path.moveTo(point)
                    else:
                        path.lineTo(point)
            painter.drawPath(path)
            if len(visible_areas) == 1:
                painter.drawEllipse(path.currentPosition(), 2, 2)
            for fraction in (0, .5, 1):
                y = rect.bottom() - fraction * rect.height()
                painter.drawText(QRectF(rect.right() + 5, y - 9, 55, 18),
                                 Qt.AlignmentFlag.AlignLeft, f"{area_max * fraction:.1f}%")
        painter.setPen(QColor("#b8c0cf"))
        for fraction, alignment in ((0, Qt.AlignmentFlag.AlignLeft), (0.5, Qt.AlignmentFlag.AlignCenter), (1, Qt.AlignmentFlag.AlignRight)):
            x = rect.left() + fraction * rect.width()
            label = time_text((self.start + fraction * self.span) / self.fps)
            box = QRectF(x if fraction == 0 else x - (100 if fraction == 1 else 50), rect.bottom() + 6, 100, 20)
            painter.drawText(box, alignment, label)
        # Keep original detection counts intact; edits occupy a separate strip.
        for frame, kind in self.edits.items():
            if self.start <= frame < self.start + self.span:
                x = rect.left() + (frame - self.start) / self.span * rect.width()
                painter.fillRect(QRectF(x, rect.top() - 12, max(2, rect.width() / self.span), 5), QColor("#63d6a3"))
                if kind == "manual":
                    painter.fillRect(QRectF(x, rect.top() - 14, 3, 9), QColor("#d4ffe9"))
        # The yellow lane is prospective; the green lane above the graph is committed.
        # Use frame edges for the inclusive affected range and centers for cursors.
        if self.propagation_start is not None:
            color = QColor("#f1d36b")
            bounds = self.propagation_range
            if bounds is not None:
                first = max(bounds[0], self.start)
                stop = min(bounds[1] + 1, self.start + self.span)
                if first < stop:
                    left = rect.left() + (first - self.start) / self.span * rect.width()
                    width = (stop - first) / self.span * rect.width()
                    painter.fillRect(QRectF(left, 18, width, 5), color)
                    painter.fillRect(QRectF(left, rect.top(), width, rect.height()), QColor(241, 211, 107, 38))
            source = self.propagation_start
            offscreen = "← " if source < self.start else "→ " if source >= self.start + self.span else ""
            label = f"{offscreen}始点 {source + 1}"
            if bounds is not None:
                label += f"　補完予定 {bounds[0] + 1}～{bounds[1] + 1}"
            painter.setPen(color)
            painter.drawText(QRectF(rect.left(), 0, rect.width(), 17), Qt.AlignmentFlag.AlignLeft, label)
            source_x = rect.left() + (source - self.start + .5) / self.span * rect.width()
            source_x = max(rect.left(), min(rect.right(), source_x))
            painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(source_x, 18), QPointF(source_x, rect.bottom()))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(source_x, 20), 4, 4)
        x = rect.left() + (self.current - self.start + 0.5) / self.span * rect.width()
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(QPointF(x, 23), QPointF(x, rect.bottom()))
