"""Video preview using the image editor's brush/cursor and wheel gestures."""
from __future__ import annotations

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy

from auto_mosaic.ui import AutoMosaicWindow, MaskPreviewLabel, PREVIEW_ZOOM_LEVELS


class VideoPreview(MaskPreviewLabel):
    geometry_changed = Signal()

    def __init__(self):
        super().__init__("動画を開いてください")
        self.image = None
        self.zoom_index = 0
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #0e1015; color: #aab2c2;")
        self.zoom_requested.connect(self._zoom)

    def set_image(self, image):
        if image is None:
            self.image = None
            self.zoom_index = 0
            self.setPixmap(QPixmap())
            self.setText("動画を開いてください")
            self.set_image_rect(QRectF())
            self.set_zoom_enabled(False)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
            self.set_zoom_enabled(True)
            self._layout_image()
        self.update()

    def reset_zoom(self):
        self.zoom_index = 0
        self._layout_image()

    def _zoom(self, position, direction):
        if self.image is None:
            return
        index = max(0, min(len(PREVIEW_ZOOM_LEVELS) - 1, self.zoom_index + (1 if direction > 0 else -1)))
        if index == self.zoom_index:
            return
        rect = self.image_rect
        anchor = position if rect.contains(position) else rect.center()
        normalized = QPointF((anchor.x() - rect.left()) / rect.width(), (anchor.y() - rect.top()) / rect.height())
        self.zoom_index = index
        self._layout_image((anchor, normalized))

    def _layout_image(self, anchor=None):
        if self.image is None:
            return
        base = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        zoom = PREVIEW_ZOOM_LEVELS[self.zoom_index]
        pixmap = QPixmap.fromImage(self.image).scaled(max(1, round(base.width() * zoom)),
                 max(1, round(base.height() * zoom)), Qt.AspectRatioMode.KeepAspectRatio)
        if anchor:
            position, normalized = anchor
            left, top = position.x() - normalized.x() * pixmap.width(), position.y() - normalized.y() * pixmap.height()
        elif self.zoom_index and not self.image_rect.isEmpty():
            left, top = self.image_rect.left(), self.image_rect.top()
        else:
            left, top = (self.width() - pixmap.width()) / 2, (self.height() - pixmap.height()) / 2
        left = AutoMosaicWindow._clamp_preview_axis(left, pixmap.width(), 0, self.width())
        top = AutoMosaicWindow._clamp_preview_axis(top, pixmap.height(), 0, self.height())
        self.set_image_rect(QRectF(left, top, pixmap.width(), pixmap.height()))
        self.setPixmap(pixmap)
        self.geometry_changed.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_image()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.mask_editing:
            painter = QPainter(self)
            painter.setPen(QPen(QColor("#ff3b30"), 2))
            painter.drawRect(self.image_rect.adjusted(1, 1, -1, -1))
