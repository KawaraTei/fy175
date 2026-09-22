"""Interactive per-frame mask count timeline and virtualized change list."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.counts = np.empty(0, np.int32)
        self.fps = 1.0
        self.current = 0
        self.window_seconds = 0
        self.start = 0
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("時点ごとの検出マスク数")

    def set_frames(self, frames: list[FrameSummary], fps: float):
        self.counts = np.array([frame.count for frame in frames], dtype=np.int32)
        self.fps, self.current, self.start = fps, 0, 0
        self.update()

    @property
    def span(self):
        return max(1, min(len(self.counts), round(self.window_seconds * self.fps))) if self.window_seconds else max(1, len(self.counts))

    def set_window_seconds(self, seconds: int):
        self.window_seconds = seconds
        self.start = max(0, min(self.current - self.span // 2, len(self.counts) - self.span))
        self.update()

    def set_current(self, frame: int):
        self.current = frame
        if not self.start <= frame < self.start + self.span:
            self.start = max(0, min(frame - self.span // 2, len(self.counts) - self.span))
        self.update()

    def plot_rect(self):
        return QRectF(36, 18, max(1, self.width() - 52), max(1, self.height() - 48))

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
                         f"{time_text(frame / self.fps)}  マスク {self.counts[frame]}件  / フレーム {frame + 1}", self)

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
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "解析後にマスク数を表示します")
            return
        visible = self.counts[self.start:self.start + self.span]
        maximum = max(1, int(self.counts.max()))
        for value in sorted({0, maximum // 2, maximum}):
            y = rect.bottom() - value / maximum * rect.height()
            painter.setPen(QColor("#303744"))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(QColor("#b8c0cf"))
            painter.drawText(QRectF(0, y - 9, 28, 18), Qt.AlignmentFlag.AlignRight, str(value))
        # Aggregate min AND max per screen pixel, so narrow dropouts are not
        # erased by averaging or by drawing only the maximum count.
        columns = min(max(1, int(rect.width())), len(visible))
        for column in range(columns):
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
        painter.setPen(QColor("#b8c0cf"))
        for fraction, alignment in ((0, Qt.AlignmentFlag.AlignLeft), (0.5, Qt.AlignmentFlag.AlignCenter), (1, Qt.AlignmentFlag.AlignRight)):
            x = rect.left() + fraction * rect.width()
            label = time_text((self.start + fraction * self.span) / self.fps)
            box = QRectF(x if fraction == 0 else x - (100 if fraction == 1 else 50), rect.bottom() + 6, 100, 20)
            painter.drawText(box, alignment, label)
        x = rect.left() + (self.current - self.start + 0.5) / self.span * rect.width()
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(QPointF(x, rect.top() - 5), QPointF(x, rect.bottom()))
