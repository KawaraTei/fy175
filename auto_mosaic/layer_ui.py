"""Independent layer workflow; ordinary image and video controls stay unchanged."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import queue

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton, QSpinBox, QSizePolicy,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from auto_mosaic.domain import EffectType, ImageMode, PROCESSING_RANGES, ProcessingSettings, TARGET_LABELS
from auto_mosaic.image_ops import visualize_detection
from auto_mosaic.layers import LayerDocument, on_white
from auto_mosaic.pipeline import MosaicPipeline
from auto_mosaic.settings_ui import AdvancedSettingsDialog
from auto_mosaic.ui_style import APP_STYLE, section_label


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
        self.layers.setColumnCount(3)
        self.layers.setHeaderLabels(["レイヤー名", "加工", "検出"])
        self.layers.header().setStretchLastSection(False)
        self.layers.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2):
            self.layers.header().setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.layers.setColumnWidth(column, 48)
        self.layers.itemChanged.connect(self._layer_changed)
        left.addWidget(self.layers, 1)
        self.layer_note = QLabel("加工：選んだレイヤーのみ変更\n検出：選んだ表示レイヤーを合成して解析")
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
        self.view.addItems(["元画像", "検出範囲", "処理結果"])
        self.view.currentIndexChanged.connect(self.refresh_preview)
        right.addWidget(self.view)
        self.preview = QLabel("レイヤー画像を開いてください")
        self.preview.setObjectName("preview")
        self.preview.setMinimumSize(1, 1)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        right.addWidget(self.preview, 1)
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

    def settings(self):
        targets = frozenset(key for key, check in self.target_checks.items() if check.isChecked())
        if not targets:
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
        future.add_done_callback(lambda done: self.events.put((done, complete)))

    def _poll(self):
        try:
            done, complete = self.events.get_nowait()
        except queue.Empty:
            return
        self.busy = False
        try:
            complete(done.result())
        except Exception as error:
            self.status.setText(f"エラー: {error}")
            QMessageBox.warning(self, "レイヤー画像", str(error))
        self._controls()

    def _controls(self):
        ready = self.document is not None and not self.busy
        self.open_button.setEnabled(not self.busy)
        self.panel.setEnabled(ready)
        self.advanced.setEnabled(ready)
        self.analyze_button.setEnabled(ready)
        can_save = ready and self.result is not None
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
        if self.busy or not self._confirm_discard():
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
            self.filename.setText(path.name)
            self.filename.setToolTip(str(path))
            self.layers.blockSignals(True)
            self.layers.clear()
            for info in document.layers:
                name = info.name + ("（非表示）" if not info.visible else "")
                item = QTreeWidgetItem([name, "", ""])
                item.setData(0, Qt.ItemDataRole.UserRole, info.index)
                item.setToolTip(0, info.name + ("\n" + info.reason if info.reason else ""))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                if info.editable:
                    item.setCheckState(1, Qt.CheckState.Checked if info.visible else Qt.CheckState.Unchecked)
                else:
                    item.setText(1, "—")
                    item.setToolTip(1, info.reason)
                if info.visible:
                    item.setCheckState(2, Qt.CheckState.Checked)
                else:
                    item.setText(2, "—")
                    item.setToolTip(2, "元画像で非表示のレイヤーは検出用合成に含めません")
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
        self.render_timer.stop()
        self.mask = self.analysis = self.result = None
        self.status.setText("検出設定を変更しました。再解析してください。")
        self.refresh_preview()
        self._controls()

    def _layer_changed(self, item, column):
        if column == 2:
            self._invalidate_detection()
        elif column == 1:
            self._schedule_render()

    def _schedule_render(self, *_):
        self.result = None
        if self.mask is not None:
            self.status.setText("処理結果を更新しています…")
            self.render_timer.start(150)
        self._controls()

    def analyze(self):
        if self.busy or self.document is None:
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
        processing = self.selected(1)
        document = self.document

        def work():
            analysis = self.pipeline.analyze_image(document.detection_image(detection), settings, source_path=document.source)
            return analysis, document.render(analysis.mask, processing, settings)

        def complete(value):
            self.analysis, result = value
            self.mask = self.analysis.mask
            self._rendered(result)

        self._submit("選択レイヤーを解析しています…", work, complete)

    def render(self):
        if self.busy or self.document is None or self.mask is None:
            return
        settings, indices, mask = self.settings(), self.selected(1), self.mask.copy()
        self._submit("処理結果を更新しています…", lambda: self.document.render(mask, indices, settings), self._rendered)

    def _rendered(self, result):
        self.result = result
        self.dirty = True
        count = len(self.analysis.detections) if self.analysis is not None else 0
        self.status.setText(f"検出 {count}件 / 加工対象 {len(self.selected(1))}レイヤー — 処理結果を確認して保存できます")
        self.view.setCurrentText("処理結果")
        self.refresh_preview()

    def refresh_preview(self, *_):
        if self.original is None:
            return
        image = on_white(self.original)
        if self.view.currentText() == "処理結果" and self.result is not None:
            image = on_white(self.result.bgra)
        elif self.view.currentText() == "検出範囲" and self.analysis is not None:
            image = visualize_detection(image, self.analysis.mask, self.analysis.detections)
        rgb = np.ascontiguousarray(image[..., ::-1])
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(qimage).scaled(
            max(1, self.preview.width() - 20), max(1, self.preview.height() - 20),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.refresh_preview)

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
        if self.result is None or self.busy or self.render_timer.isActive():
            return
        result = self.result

        def complete(_):
            self.dirty = False
            self.status.setText(f"保存しました: {path}")

        self._submit("保存しています…", lambda: result.save(path), complete)

    def closeEvent(self, event):
        if self.busy:
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
