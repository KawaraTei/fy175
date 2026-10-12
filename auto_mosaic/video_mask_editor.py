"""Video-only editing controls; masks are committed by video_mask_edit."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from auto_mosaic.image_ops import paint_mask_stroke
from auto_mosaic.ui_style import section_label
from auto_mosaic.video_mask_edit import propagate_mask, save_frame_mask, undo_mask_edit
from auto_mosaic.video_timeline import time_text


class VideoMaskEditor(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.draft = None
        self.dirty = False
        self.edits = {}
        self.can_undo = False
        self.start = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(7)
        self.frame_label = section_label("未解析")
        self.frame_label.setWordWrap(True)
        frame_row = QHBoxLayout()
        frame_row.addWidget(self.frame_label, 1)
        self.edit_button = QPushButton("マスクを編集")
        self.edit_button.clicked.connect(self.begin_edit)
        frame_row.addWidget(self.edit_button)
        layout.addLayout(frame_row)
        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("ブラシ"))
        self.brush_value = QLabel("40 px")
        self.brush_value.setObjectName("value")
        brush_row.addStretch(1)
        brush_row.addWidget(self.brush_value)
        layout.addLayout(brush_row)
        self.brush = QSlider(Qt.Orientation.Horizontal)
        self.brush.setRange(4, 200)
        self.brush.setValue(40)
        self.brush.valueChanged.connect(self.update_brush)
        layout.addWidget(self.brush)
        hint = QLabel("ドラッグで追加 / Alt＋ドラッグで削除\nホイールで拡大・縮小")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        actions = QHBoxLayout()
        self.apply_button = QPushButton("このフレームに適用")
        self.apply_button.clicked.connect(self.apply)
        self.discard_button = QPushButton("破棄")
        self.discard_button.clicked.connect(self.discard)
        actions.addWidget(self.apply_button, 1)
        actions.addWidget(self.discard_button)
        layout.addLayout(actions)
        layout.addSpacing(8)
        layout.addWidget(section_label("自動補完"))
        self.start_label = QLabel("始点: 未指定")
        self.start_label.setWordWrap(True)
        layout.addWidget(self.start_label)
        self.start_button = QPushButton("現在位置を始点にする")
        self.start_button.clicked.connect(self.set_start)
        layout.addWidget(self.start_button)
        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("終点"))
        self.end = QSpinBox()
        self.end.setRange(1, 1)
        self.end.setSuffix(" フレーム")
        self.end.setKeyboardTracking(False)
        self.end.setAccessibleName("補完する終了フレーム")
        self.end.valueChanged.connect(self.update_controls)
        end_row.addWidget(self.end, 1)
        layout.addLayout(end_row)
        self.end_time = QLabel("00:00.000")
        self.end_time.setObjectName("value")
        self.end_time.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.end_time)
        self.end_button = QPushButton("現在位置を終点にする")
        self.end_button.clicked.connect(lambda: self.end.setValue(self.window.current_frame + 1))
        layout.addWidget(self.end_button)
        note = QLabel("補完マスクは検出結果に追加します。手動で消した領域も追従します。補完後に範囲内を確認してください。")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)
        self.propagate_button = QPushButton("指定フレームまで自動補完")
        self.propagate_button.clicked.connect(self.propagate)
        layout.addWidget(self.propagate_button)
        layout.addSpacing(8)
        self.undo_button = QPushButton("直前のマスク変更を戻す")
        self.undo_button.clicked.connect(self.undo)
        layout.addWidget(self.undo_button)
        self.edit_summary = QLabel("手動編集・補完なし")
        self.edit_summary.setWordWrap(True)
        layout.addWidget(self.edit_summary)
        layout.addStretch(1)

    @property
    def active(self):
        return self.draft is not None

    def reset(self):
        self.draft, self.dirty = None, False
        self.edits, self.can_undo, self.start = {}, False, None
        self.window.preview.set_mask_editing(False)
        self.window.timeline.set_edits({})

    def begin_edit(self):
        window = self.window
        if window.busy or window.analysis is None or self.active:
            return
        window._stop_play()
        try:
            self.draft = window.analysis.cache.full_mask(window.current_frame, window.info)
        except Exception as error:
            window.status.setText(str(error))
            return
        self.dirty = False
        window.view_mode.setCurrentText("検出範囲")
        window._render_preview()
        window._update_controls()
        window.status.setText("マスク編集中: ドラッグで追加 / Alt＋ドラッグで削除。適用してからフレームを移動してください。")

    def update_brush(self):
        self.brush_value.setText(f"{self.brush.value()} px")
        window = self.window
        if window.info is not None:
            window.preview.set_brush_diameter(self.brush.value() * window.preview.image_rect.width() / window.info.width)

    def sync_preview(self):
        window = self.window
        drawing = self.active and not window.busy and window.view_mode.currentText() == "検出範囲"
        if window.preview.mask_editing != drawing:
            window.preview.set_mask_editing(drawing)
        self.update_brush()

    def paint(self, start, end, erase):
        if not self.active or self.window.busy:
            return
        rect = self.window.preview.image_rect
        height, width = self.draft.shape
        def pixel(point):
            return (max(0, min(width - 1, int((point.x() - rect.left()) * width / rect.width()))),
                    max(0, min(height - 1, int((point.y() - rect.top()) * height / rect.height()))))
        if not rect.isEmpty() and paint_mask_stroke(self.draft, pixel(start), pixel(end), self.brush.value(), erase):
            self.dirty = True
            self.window._mark_project_dirty()
            self.window._render_preview()

    def apply(self):
        window = self.window
        if not self.active or window.busy:
            return
        analysis, frame, mask = window.analysis, window.current_frame, self.draft.copy()
        window._start_job("mask_save", lambda cancel, progress: save_frame_mask(analysis, frame, mask, cancel, progress))

    def discard(self):
        if self.dirty and not self._ask_discard("編集中のマスクを破棄"):
            return
        self.draft, self.dirty = None, False
        self.window._mark_project_dirty()
        self.window._render_preview()
        self.window._update_controls()
        self.window.status.setText("編集中のマスクを破棄しました。適用済みの変更は保持しています。")

    def _ask_discard(self, action, all_changes=False):
        detail = "手動編集・自動補完したマスクは破棄されます。" if all_changes else "編集中のマスクは破棄されます。"
        return QMessageBox.warning(self.window, "マスク編集の破棄",
            f"{action}しますか？\n{detail}",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Discard

    def confirm_clear(self, action):
        if not self.window.project_dirty:
            return True
        return QMessageBox.warning(self.window, "未保存の編集",
            f"{action}しますか？\n未保存の編集・設定は失われます。残す場合はキャンセルして「編集を保存」を押してください。",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Discard

    def set_start(self):
        frame = self.window.current_frame
        if self.edits.get(frame) == "manual":
            self.start = frame
            self.end.setRange(min(frame + 2, len(self.window.analysis.frames)), len(self.window.analysis.frames))
            self.update_controls()

    def propagate(self):
        window = self.window
        if window.busy or self.active or self.start is None:
            return
        analysis, start, end = window.analysis, self.start, self.end.value() - 1
        window._start_job("propagate", lambda cancel, progress: propagate_mask(analysis, start, end, cancel, progress))

    def undo(self):
        window = self.window
        if self.can_undo and not window.busy and not self.active:
            analysis = window.analysis
            window._start_job("mask_undo", lambda cancel, progress: undo_mask_edit(analysis, cancel, progress))

    def accept_result(self, kind, result):
        window = self.window
        self.edits, self.can_undo = result.edited, result.can_undo
        if kind == "mask_save":
            self.draft, self.dirty = None, False
            self.start = result.start
            self.end.setRange(min(result.start + 2, len(window.analysis.frames)), len(window.analysis.frames))
        elif self.start is not None and self.edits.get(self.start) != "manual":
            self.start = None
        window.timeline.set_edits(self.edits)
        window.seek(window.current_frame)
        if kind == "propagate":
            window.status.setText(f"フレーム {result.start + 1}～{result.end + 1} に補完しました。タイムラインで確認してください。")
        elif kind == "mask_save":
            window.status.setText(f"フレーム {result.start + 1} のマスクを適用しました。終点を指定すると動きに追従して補完できます。")
        else:
            window.status.setText("直前のマスク変更を戻しました。")

    def update_controls(self):
        window = self.window
        ready = window.analysis is not None and window.analysis.complete and not window.busy
        self.frame_label.setText(f"{'編集中' if self.active else 'フレーム'}: {window.current_frame + 1}" if window.analysis else "未解析")
        self.edit_button.setEnabled(ready and not self.active)
        self.brush.setEnabled(ready and self.active)
        self.apply_button.setEnabled(ready and self.active)
        self.discard_button.setEnabled(ready and self.active)
        idle = ready and not self.active
        self.start_button.setEnabled(idle and self.edits.get(window.current_frame) == "manual")
        valid_start = self.start is not None and window.analysis is not None and self.start < len(window.analysis.frames) - 1
        self.end.setEnabled(idle and valid_start)
        self.end_button.setEnabled(idle and valid_start and window.current_frame > self.start)
        self.propagate_button.setEnabled(idle and valid_start and self.end.value() > self.start + 1)
        self.undo_button.setEnabled(idle and self.can_undo)
        if window.info:
            self.end_time.setText(time_text((self.end.value() - 1) / float(window.info.fps)))
        self.start_label.setText(f"始点: {self.start + 1} フレーム / {time_text(self.start / float(window.info.fps))}" if self.start is not None else "始点: 未指定（編集・適用後に設定）")
        manual = sum(kind == "manual" for kind in self.edits.values())
        propagated = len(self.edits) - manual
        self.edit_summary.setText(f"緑: 手動 {manual} / 補完 {propagated} フレーム" if self.edits else "手動編集・補完なし")
        self.edit_summary.setToolTip("タイムライン上端の緑が編集済み範囲です。")
        self.sync_preview()
