"""Independent layer workflow; ordinary image and video controls stay unchanged."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import replace
from pathlib import Path
import queue

import numpy as np
import cv2
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton, QSpinBox, QSizePolicy,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from auto_mosaic.domain import EffectType, ImageMode, PROCESSING_RANGES, ProcessingSettings, TARGET_LABELS
from auto_mosaic.image_ops import visualize_detection, paint_mask_stroke
from auto_mosaic.layers import LayerDocument, on_white
from auto_mosaic.pipeline import MosaicPipeline
from auto_mosaic.settings_ui import AdvancedSettingsDialog
from auto_mosaic.ui_style import APP_STYLE, section_label
from auto_mosaic.ui import MaskPreviewLabel
from auto_mosaic.platform_support import command_key, option_key


class LayerMosaicWindow(QMainWindow):
    def __init__(self, model_dir: Path, settings: ProcessingSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("レイヤー画像(β) — FY175AutoMosaic")
        self.setStyleSheet(APP_STYLE)
        self.resize(1260, 790)
        self.setMinimumSize(980, 650)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.pipeline = MosaicPipeline(model_dir)
        self.base_settings = settings
        self.document = None
        self.result = None
        self.analysis = None
        self.original = None
        self.mask = None
        self.busy = False
        self.dirty = False
        self.preview_busy = False
        self.preview_generation = 0
        self.editing = False
        self.manual_mask = False
        self.undo_history = deque()
        self.stroke_before = None
        self.embedding = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="layer-image")
        self.events = queue.Queue()
        self._build()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(50)
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.render)
        self._controls()

    def _build(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)
        toolbar = QHBoxLayout()
        self.open_button = QPushButton("レイヤー画像を開く")
        self.open_button.clicked.connect(self.open_file)
        toolbar.addWidget(self.open_button)
        self.filename = QLabel("PSD / CLIP")
        self.filename.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(self.filename, 1)
        layout.addLayout(toolbar)
        body = QHBoxLayout()
        self.panel = QWidget()
        self.panel.setFixedWidth(330)
        left = QVBoxLayout(self.panel)
        left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(section_label("レイヤー"))
        self.layers = QTreeWidget()
        self.layers.setRootIsDecorated(False)
        self.layers.setColumnCount(4)
        self.layers.setHeaderLabels(["レイヤー名", "加工", "検出", "表示"])
        self.layers.header().setStretchLastSection(False)
        self.layers.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            self.layers.header().setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.layers.setColumnWidth(column, 48)
        self.layers.itemChanged.connect(self._layer_changed)
        left.addWidget(self.layers, 1)
        self.layer_note = QLabel("加工：選んだレイヤーのみ変更\n検出：選んだレイヤーを合成して解析\n表示：プレビュー・保存結果に反映")
        self.layer_note.setWordWrap(True)
        left.addWidget(self.layer_note)
        self.warning = QLabel("合成結果は元の編集ソフトと異なる場合があります。\n保存前に処理結果を確認してください。")
        self.warning.setWordWrap(True)
        self.warning.setObjectName("muted")
        left.addWidget(self.warning)
        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItems(["実写", "イラスト"])
        self.mode.setCurrentIndex(int(self.base_settings.mode == ImageMode.ILLUSTRATION))
        self.mode.currentIndexChanged.connect(self._invalidate_detection)
        form.addRow("画像の種類", self.mode)
        targets = QWidget()
        row = QHBoxLayout(targets)
        row.setContentsMargins(0, 0, 0, 0)
        self.target_checks = {}
        for key, label in TARGET_LABELS.items():
            check = QCheckBox(label)
            check.setChecked(key in self.base_settings.targets)
            check.toggled.connect(self._invalidate_detection)
            self.target_checks[key] = check
            row.addWidget(check)
        form.addRow("検出対象", targets)
        self.effect = QComboBox()
        self.effect.addItems(["モザイク", "ぼかし"])
        self.effect.setCurrentIndex(int(self.base_settings.effect == EffectType.BLUR))
        self.effect.currentIndexChanged.connect(self._schedule_render)
        form.addRow("処理の種類", self.effect)
        self.effect_size = QSpinBox()
        self.effect_size.setRange(*PROCESSING_RANGES["effect_size"][:2])
        self.effect_size.setValue(self.base_settings.effect_size)
        self.effect_size.setSuffix(" px")
        self.effect_size.valueChanged.connect(self._schedule_render)
        form.addRow("処理の強さ", self.effect_size)
        left.addLayout(form)
        self.advanced = AdvancedSettingsDialog(self, "レイヤー画像")
        self.advanced_controls = {}
        advanced_form = QFormLayout()
        for key, label in (("confidence_threshold", "検出閾値"), ("mask_threshold", "マスク閾値"),
                           ("mask_expansion", "マスク拡張"), ("mask_smoothing", "マスクの滑らかさ")):
            low, high, step = PROCESSING_RANGES[key]
            control = QSpinBox() if isinstance(step, int) else QDoubleSpinBox()
            control.setRange(low, high)
            control.setSingleStep(step)
            control.setValue(getattr(self.base_settings, key))
            control.setKeyboardTracking(False)
            control.valueChanged.connect(self._invalidate_detection)
            self.advanced_controls[key] = control
            advanced_form.addRow(label, control)
        self.advanced.settings_layout.addLayout(advanced_form)
        self.advanced.settings_layout.addStretch()
        detail = QPushButton("詳細設定")
        detail.clicked.connect(self.advanced.show_settings)
        left.addWidget(detail)
        self.analyze_button = QPushButton("選択したレイヤーで再解析")
        self.analyze_button.clicked.connect(self.analyze)
        left.addWidget(self.analyze_button)
        body.addWidget(self.panel)
        right = QVBoxLayout()
        self.view = QComboBox()
        self.view.addItems(["元画像", "検出範囲", "マスク範囲", "処理結果"])
        self.view.currentIndexChanged.connect(self.refresh_preview)
        right.addWidget(self.view)
        self.preview = MaskPreviewLabel("レイヤー画像を開いてください")
        self.preview.stroke_started.connect(self._begin_stroke)
        self.preview.stroke_requested.connect(self._paint_stroke)
        self.preview.stroke_finished.connect(self._finish_stroke)
        self.preview.region_requested.connect(self._select_region)
        self.preview.setObjectName("preview")
        self.preview.setMinimumSize(1, 1)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        right.addWidget(self.preview, 1)
        # Keep editing controls in the existing left panel; preserve the canvas area.
        self.edit_button = QPushButton("マスクを編集")
        self.edit_button.clicked.connect(self._toggle_edit)
        left.insertWidget(left.count() - 1, self.edit_button)
        self.edit_tools = QWidget()
        edit_layout = QVBoxLayout(self.edit_tools)
        edit_layout.setContentsMargins(0, 0, 0, 0)
        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("ブラシ"))
        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 1000)
        self.brush_size.setValue(20)
        self.brush_size.setSuffix(" px")
        self.brush_size.valueChanged.connect(self.refresh_preview)
        brush_row.addWidget(self.brush_size)
        self.undo_button = QPushButton("元に戻す")
        self.undo_button.clicked.connect(self._undo_mask)
        brush_row.addWidget(self.undo_button)
        edit_layout.addLayout(brush_row)
        hint = QLabel(f"ドラッグ：追加 / {option_key()}：削除\nShift+クリック：領域追加 / {command_key()}+Shift：領域削除")
        hint.setWordWrap(True)
        edit_layout.addWidget(hint)
        left.insertWidget(left.count() - 1, self.edit_tools)
        self.edit_tools.hide()
        self.undo_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Undo), self)
        self.undo_shortcut.activated.connect(self._undo_mask)
        save_row = QHBoxLayout()
        self.flatten_button = QPushButton("統合画像を保存")
        self.flatten_button.clicked.connect(lambda: self.choose_save(False))
        self.layer_save_button = QPushButton("レイヤー付きで保存")
        self.layer_save_button.clicked.connect(lambda: self.choose_save(True))
        save_row.addWidget(self.flatten_button)
        save_row.addWidget(self.layer_save_button)
        right.addLayout(save_row)
        body.addLayout(right, 1)
        layout.addLayout(body, 1)
        self.status = QLabel("PSD / CLIPのラスターレイヤーを加工できます")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def settings(self, require_targets=True):
        targets = frozenset(key for key, check in self.target_checks.items() if check.isChecked())
        if not targets and require_targets:
            raise ValueError("検出対象を1つ以上選択してください。")
        return replace(self.base_settings, mode=ImageMode.PHOTO if self.mode.currentIndex() == 0 else ImageMode.ILLUSTRATION,
                       targets=targets, effect=EffectType.MOSAIC if self.effect.currentIndex() == 0 else EffectType.BLUR,
                       effect_size=self.effect_size.value(),
                       **{key: control.value() for key, control in self.advanced_controls.items()})

    def selected(self, column):
        return frozenset(self.layers.topLevelItem(row).data(0, Qt.ItemDataRole.UserRole)
                         for row in range(self.layers.topLevelItemCount())
                         if self.layers.topLevelItem(row).checkState(column) == Qt.CheckState.Checked)

    def _submit(self, message, work, complete):
        if self.busy:
            return
        self.busy = True
        self.status.setText(message)
        self._controls()
        future = self.executor.submit(work)
        future.add_done_callback(lambda done: self.events.put((done, complete, True)))

    def _poll(self):
        try:
            done, complete, blocking = self.events.get_nowait()
        except queue.Empty:
            return
        if blocking:
            self.busy = False
        try:
            complete(done.result())
        except Exception as error:
            if not blocking:
                self.preview_busy = False
            self.status.setText(f"エラー: {error}")
            QMessageBox.warning(self, "レイヤー画像", str(error))
        self._controls()

    def _controls(self):
        ready = self.document is not None and not self.busy
        self.open_button.setEnabled(not self.busy and not self.preview_busy)
        self.panel.setEnabled(ready)
        self.advanced.setEnabled(ready)
        self.analyze_button.setEnabled(ready)
        self.edit_button.setEnabled(ready and not self.preview_busy)
        self.edit_tools.setVisible(self.editing)
        self.edit_button.setText("マスク編集を終了" if self.editing else "マスクを編集")
        self.undo_button.setEnabled(ready and bool(self.undo_history) and self.stroke_before is None)
        self.undo_shortcut.setEnabled(self.editing and self.undo_button.isEnabled())
        can_paint = (ready and self.editing and not self.preview_busy
                     and self.view.currentText() in {"検出範囲", "マスク範囲"})
        if self.preview.mask_editing != can_paint:
            self.preview.set_mask_editing(can_paint)
        can_save = ready and self.result is not None and not self.preview_busy and not self.editing
        if hasattr(self, "render_timer"):
            can_save = can_save and not self.render_timer.isActive()
        self.flatten_button.setEnabled(can_save)
        self.layer_save_button.setEnabled(can_save)

    def _confirm_discard(self):
        if not self.dirty:
            return True
        return QMessageBox.question(self, "未保存の処理結果", "未保存の処理結果を破棄しますか？",
                                    QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                                    QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Discard

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "レイヤー画像を開く", "", "レイヤー画像 (*.psd *.clip)")
        if path:
            self.load_path(Path(path))

    def load_path(self, path):
        if self.busy or self.preview_busy or not self._confirm_discard():
            return
        self.render_timer.stop()
        previous = self.document

        def load():
            document = LayerDocument(path)
            if previous is not None:
                previous.close()
            return document

        def loaded(document):
            self.document = document
            self.original = document.original
            self.mask = self.result = self.analysis = None
            self.dirty = False
            self.editing = self.manual_mask = False
            self.undo_history.clear()
            self.embedding = None
            self.filename.setText(path.name)
            self.filename.setToolTip(str(path))
            self.layers.blockSignals(True)
            self.layers.clear()
            for info in document.layers:
                item = QTreeWidgetItem([info.name, "", "", ""])
                item.setData(0, Qt.ItemDataRole.UserRole, info.index)
                item.setToolTip(0, info.name + ("\n" + info.reason if info.reason else ""))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                if info.editable:
                    item.setCheckState(1, Qt.CheckState.Checked if info.visible else Qt.CheckState.Unchecked)
                else:
                    item.setText(1, "—")
                    item.setToolTip(1, info.reason)
                for column in (2, 3):
                    item.setCheckState(column, Qt.CheckState.Checked if info.visible else Qt.CheckState.Unchecked)
                item.setToolTip(2, "表示のオン・オフに関係なく、選んだレイヤーを検出に使用します")
                item.setToolTip(3, "プレビューと保存結果の表示を切り替えます。加工・検出の選択は変わりません")
                self.layers.addTopLevelItem(item)
            self.layers.blockSignals(False)
            self.warning.setText(
                "CLIPのベクター・一部の色調補正は合成未対応です。元の編集ソフトでラスタライズしてください。"
                if document.extension == ".clip" else
                "PSDは8bit RGBに対応。特殊レイヤーは加工対象にできません。合成結果を確認して保存してください。"
            )
            self.view.setCurrentText("元画像")
            self.refresh_preview()
            self.status.setText("読み込みました。解析しています…")
            self.analyze()

        self._submit("レイヤー画像を読み込んでいます…", load, loaded)

    def _invalidate_detection(self, *_):
        if self.document is None:
            return
        if self.manual_mask or self.editing:
            self.status.setText("編集中のマスクを保持しています。検出設定は再解析時に反映します。")
            return
        self.render_timer.stop()
        self.mask = self.analysis = self.result = None
        self._request_display()
        self.status.setText("検出設定を変更しました。再解析してください。")
        self.refresh_preview()
        self._controls()

    def _layer_changed(self, item, column):
        if column == 2:
            self._invalidate_detection()
        elif column == 1:
            self._schedule_render()
        elif column == 3:
            self.dirty = True
            self.embedding = None
            self._request_display()

    def _schedule_render(self, *_):
        self.result = None
        if self.document is not None:
            self.dirty = True
            if not self.editing:
                self.status.setText("処理結果を更新しています…")
                self.render_timer.start(150)
        self._controls()

    def analyze(self):
        if self.busy or self.document is None:
            return
        if self.manual_mask and QMessageBox.question(
            self, "再解析", "手動で編集したマスクを破棄して再解析しますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            settings = self.settings()
            detection = self.selected(2)
            if not detection:
                raise ValueError("検出するレイヤーを1つ以上選択してください。")
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.render_timer.stop()
        self.result = None
        self.mask = self.analysis = None
        self.editing = self.manual_mask = False
        self.undo_history.clear()
        self.embedding = None
        processing = self.selected(1)
        document = self.document

        def work():
            analysis = self.pipeline.analyze_image(document.detection_image(detection), settings, source_path=document.source)
            return analysis, document.prepare(analysis.mask, processing, settings)

        def complete(value):
            self.analysis, result = value
            self.mask = self.analysis.mask
            self.view.setCurrentText("処理結果")
            self._rendered(result)

        self._submit("選択レイヤーを解析しています…", work, complete)

    def render(self):
        if self.busy or self.document is None:
            return
        if self.mask is None:
            self._request_display()
            return
        settings, indices, mask = self.settings(False), self.selected(1), self.mask.copy()
        self._submit("処理結果を更新しています…", lambda: self.document.prepare(mask, indices, settings), self._rendered)

    def _rendered(self, result):
        self.result = result
        self.dirty = True
        count = len(self.analysis.detections) if self.analysis is not None else 0
        self.status.setText(f"検出 {count}件 / 加工対象 {len(self.selected(1))}レイヤー — 処理結果を確認して保存できます")
        self._request_display()

    def _request_display(self):
        self.preview_generation += 1
        if self.document is None or self.preview_busy:
            return
        generation = self.preview_generation
        visible, result, document = self.selected(3), self.result, self.document
        self.preview_busy = True
        self._controls()
        def work():
            if result is not None:
                return result.images(visible)
            return document.display.image(visible), None
        def complete(images):
            self.preview_busy = False
            if generation != self.preview_generation or result is not self.result:
                self._request_display()
                return
            self.original = images[0]
            if result is not None:
                result.original_bgra, result.bgra = images
            self.refresh_preview()
        future = self.executor.submit(work)
        future.add_done_callback(lambda done: self.events.put((done, complete, False)))

    def refresh_preview(self, *_):
        if self.original is None:
            return
        image = on_white(self.original)
        if self.view.currentText() == "処理結果" and self.result is not None and self.result.bgra is not None:
            image = on_white(self.result.bgra)
        elif self.view.currentText() in {"検出範囲", "マスク範囲"} and self.mask is not None:
            mask = cv2.resize(self.mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            detections = [] if self.analysis is None else self.analysis.detections
            # Display caches may be smaller than the full-resolution editing mask.
            sx, sy = image.shape[1] / self.document.width, image.shape[0] / self.document.height
            detections = [replace(detection, box=tuple(round(v * (sx if i % 2 == 0 else sy))
                                                       for i, v in enumerate(detection.box))) for detection in detections]
            image = visualize_detection(image, mask, detections,
                                        show_detection_annotations=self.view.currentText() == "検出範囲" and not self.editing)
        rgb = np.ascontiguousarray(image[..., ::-1])
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            max(1, self.preview.width() - 20), max(1, self.preview.height() - 20),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(pixmap)
        self.preview.set_image_rect(QRectF((self.preview.width() - pixmap.width()) / 2,
                                          (self.preview.height() - pixmap.height()) / 2,
                                          pixmap.width(), pixmap.height()))
        self.preview.set_brush_diameter(self.brush_size.value() * pixmap.width() / self.document.width)
        self._controls()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.refresh_preview)

    def _toggle_edit(self):
        if self.busy or self.preview_busy or self.document is None:
            return
        self.render_timer.stop()
        self.editing = not self.editing
        if self.editing:
            if self.mask is None:
                self.mask = np.zeros((self.document.height, self.document.width), bool)
            self.view.setCurrentText("マスク範囲")
            self.status.setText("マスクを編集できます。編集を終了すると処理結果を更新します。")
        else:
            self.view.setCurrentText("処理結果")
            self._schedule_render()
        self.refresh_preview()

    def _mask_point(self, position):
        rect = self.preview.image_rect
        if rect.isEmpty() or not rect.contains(position) or self.mask is None:
            return None
        h, w = self.mask.shape
        return (min(w - 1, max(0, int((position.x() - rect.left()) * w / rect.width()))),
                min(h - 1, max(0, int((position.y() - rect.top()) * h / rect.height()))))

    def _remember_mask(self, mask):
        packed = np.packbits(mask).tobytes()
        self.undo_history.append((mask.shape, packed))
        while len(self.undo_history) > 20 or sum(len(entry[1]) for entry in self.undo_history) > 128 * 1024 * 1024:
            self.undo_history.popleft()

    def _mask_changed(self):
        self.manual_mask = self.dirty = True
        self.result = None
        self.refresh_preview()

    def _begin_stroke(self):
        if self.preview.mask_editing and self.mask is not None:
            self.stroke_before = self.mask.copy()

    def _paint_stroke(self, start, end, erase):
        if not self.preview.mask_editing or self.mask is None:
            return
        first, last = self._mask_point(start), self._mask_point(end)
        if first is not None and last is not None:
            if paint_mask_stroke(self.mask, first, last, self.brush_size.value(), erase):
                self._mask_changed()

    def _finish_stroke(self):
        if self.stroke_before is not None and not np.array_equal(self.stroke_before, self.mask):
            self._remember_mask(self.stroke_before)
        self.stroke_before = None
        self._controls()

    def _undo_mask(self):
        if not self.editing or self.busy or self.stroke_before is not None or not self.undo_history:
            return
        shape, packed = self.undo_history.pop()
        self.mask = np.unpackbits(np.frombuffer(packed, np.uint8), count=shape[0] * shape[1]).reshape(shape).astype(bool)
        self._mask_changed()
        self.status.setText("マスク編集を元に戻しました")

    def _select_region(self, position, erase):
        point = self._mask_point(position)
        if not self.preview.mask_editing or point is None:
            return
        visible, document, embedding = self.selected(3), self.document, self.embedding
        threshold = self.advanced_controls["mask_threshold"].value()
        def work():
            image = document.detection_image(visible)
            return self.pipeline.select_mask_region(image, point, threshold, embedding)
        def complete(value):
            region, self.embedding = value
            candidate = self.mask & ~region if erase else self.mask | region
            if not np.array_equal(candidate, self.mask):
                self._remember_mask(self.mask)
                self.mask = candidate
                self._mask_changed()
            self.status.setText("領域を削除しました" if erase else "領域を追加しました")
        self._submit("クリック位置の領域を選択しています…", work, complete)

    def choose_save(self, layered):
        if self.result is None or self.busy:
            return
        suffix = self.document.extension if layered else ".png"
        suggested = self.document.source.with_name(self.document.source.stem + "_mosaic" + suffix)
        filters = f"{suffix[1:].upper()} (*{suffix})" if layered else "PNG (*.png);;JPEG (*.jpg)"
        path, chosen = QFileDialog.getSaveFileName(self, "レイヤー付きで保存" if layered else "統合画像を保存", str(suggested), filters)
        if path:
            output = Path(path)
            if not output.suffix:
                output = output.with_suffix(".jpg" if chosen.startswith("JPEG") else suffix)
            self.save_to(output)

    def save_to(self, path):
        if self.result is None or self.busy or self.preview_busy or self.editing or self.render_timer.isActive():
            return
        mask, indices, visible, settings = self.mask.copy(), self.selected(1), self.selected(3), self.settings(False)
        document = self.document

        def complete(_):
            self.dirty = False
            self.status.setText(f"保存しました: {path}")

        self._submit("保存しています…", lambda: document.render(mask, indices, settings, visible).save(path), complete)

    def closeEvent(self, event):
        if self.busy or self.preview_busy:
            self.status.setText("処理完了後に閉じてください。")
            event.ignore()
            return
        if not self._confirm_discard():
            event.ignore()
            return
        self.render_timer.stop()
        self.timer.stop()
        if self.document is not None:
            self.executor.submit(self.document.close)
        self.executor.shutdown(wait=False)
        self.advanced.close()
        event.accept()
