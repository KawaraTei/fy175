from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from auto_mosaic.domain import EffectType, ImageMode, ProcessingResult, ProcessingSettings
from auto_mosaic.image_ops import (
    apply_effect,
    load_image_bgr,
    paint_mask_stroke,
    visualize_detection,
)
from auto_mosaic.model_catalog import required_model_paths
from auto_mosaic.pipeline import MosaicPipeline


APP_NAME = "FY175AutoMosaic"
MAX_IMAGES = 100
MASK_EDIT_ZOOM_LEVELS = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


class MaskPreviewLabel(QLabel):
    stroke_requested = Signal(QPointF, QPointF, bool)
    zoom_requested = Signal(QPointF, int)

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.mask_editing = False
        self.image_rect = QRectF()
        self.last_position: QPointF | None = None
        self.brush_position: QPointF | None = None
        self.brush_diameter = 0.0
        self.brush_erasing = False
        self.zoom_enabled = False

    def set_zoom_enabled(self, enabled: bool) -> None:
        self.zoom_enabled = enabled

    def wheelEvent(self, event) -> None:
        wheel_delta = event.angleDelta().y()
        if self.zoom_enabled and wheel_delta != 0 and not self.image_rect.isEmpty():
            self.zoom_requested.emit(event.position(), 1 if wheel_delta > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)

    def set_mask_editing(self, editing: bool) -> None:
        self.mask_editing = editing
        self.last_position = None
        self.setMouseTracking(editing)
        if not editing:
            self.brush_position = None
        self.setCursor(
            Qt.CursorShape.CrossCursor if editing else Qt.CursorShape.ArrowCursor
        )
        self.update()

    def set_image_rect(self, rect: QRectF) -> None:
        self.image_rect = rect
        if self.brush_position is not None and not rect.contains(self.brush_position):
            self.brush_position = None
        self.update()

    def set_brush_diameter(self, diameter: float) -> None:
        self.brush_diameter = max(1.0, diameter)
        self.update()

    def _set_brush_position(self, position: QPointF, erase: bool) -> None:
        self.brush_position = position if self.image_rect.contains(position) else None
        self.brush_erasing = erase
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            self.mask_editing
            and event.button() == Qt.MouseButton.LeftButton
            and self.image_rect.contains(event.position())
        ):
            self.last_position = event.position()
            erase = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            self._set_brush_position(self.last_position, erase)
            self.stroke_requested.emit(self.last_position, self.last_position, erase)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.mask_editing:
            current = event.position()
            erase = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            self._set_brush_position(current, erase)
            if (
                self.last_position is not None
                and event.buttons() & Qt.MouseButton.LeftButton
            ):
                self.stroke_requested.emit(self.last_position, current, erase)
                self.last_position = current
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.mask_editing and event.button() == Qt.MouseButton.LeftButton:
            self.last_position = None
            self._set_brush_position(
                event.position(),
                bool(event.modifiers() & Qt.KeyboardModifier.AltModifier),
            )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self.brush_position = None
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull() or self.image_rect.isEmpty():
            super().paintEvent(event)
        else:
            QFrame.paintEvent(self, event)
            image_painter = QPainter(self)
            image_painter.drawPixmap(self.image_rect.topLeft(), pixmap)
            image_painter.end()

        if (
            not self.mask_editing
            or self.brush_position is None
            or self.brush_diameter <= 0
        ):
            return
        radius = self.brush_diameter / 2
        circle = QRectF(
            self.brush_position.x() - radius,
            self.brush_position.y() - radius,
            self.brush_diameter,
            self.brush_diameter,
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        outer_pen = QPen(QColor(0, 0, 0, 210))
        outer_pen.setWidthF(3.0)
        painter.setPen(outer_pen)
        painter.drawEllipse(circle)
        inner_pen = QPen(QColor("#ff6b5f" if self.brush_erasing else "#f5f7fa"))
        inner_pen.setWidthF(1.2)
        painter.setPen(inner_pen)
        painter.drawEllipse(circle)
        painter.end()


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parents[1]


class AutoMosaicWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1260, 790)
        self.setMinimumSize(980, 650)
        self.setAcceptDrops(True)

        self.app_root = application_root()
        self.model_dir = resource_root() / "models"
        self.pipeline = MosaicPipeline(self.model_dir)
        self.image_paths: list[Path] = []
        self.manual_review_paths: list[Path] = []
        self.current_result: ProcessingResult | None = None
        self.preview_rgb: np.ndarray | None = None
        self.mask_edit_active = False
        self.mask_edit_dirty = False
        self.mask_edit_source_path: Path | None = None
        self.edited_mask: np.ndarray | None = None
        self.mask_edit_original_bgr: np.ndarray | None = None
        self.preview_image_rect = QRectF()
        self.mask_edit_zoom_index = 0
        self.preview_render_target_size = (0, 0)
        self.busy = False
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.analysis_generation = 0

        self._build_layout()
        self._apply_style()
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_events)
        self.poll_timer.start(100)
        self._update_model_status()

    def _build_layout(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(central)

        toolbar = QFrame()
        toolbar.setObjectName("toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 10, 14, 10)
        toolbar_layout.setSpacing(8)
        toolbar_layout.addWidget(self._button("画像を追加", self._add_images))
        toolbar_layout.addWidget(self._button("選択を削除", self._remove_selected))
        toolbar_layout.addWidget(
            self._button("選択を除去（メモ）", self._remove_selected_to_manual)
        )
        toolbar_layout.addWidget(self._button("すべて消去", self._clear_images))
        toolbar_layout.addStretch(1)
        self.status_label = QLabel("画像を追加してください")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        toolbar_layout.addWidget(self.status_label, 1)
        root_layout.addWidget(toolbar)

        body = QWidget()
        body_layout = QGridLayout(body)
        body_layout.setContentsMargins(14, 14, 14, 14)
        body_layout.setHorizontalSpacing(12)
        body_layout.setColumnStretch(1, 1)
        root_layout.addWidget(body, 1)

        left = self._panel()
        left.setFixedWidth(230)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)

        target_panel = QWidget()
        target_layout = QVBoxLayout(target_panel)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(self._section_label("処理対象画像"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.file_list.currentRowChanged.connect(self._on_selection_changed)
        target_layout.addWidget(self.file_list, 1)
        self.file_count_label = QLabel("0 / 100")
        self.file_count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.file_count_label.setObjectName("muted")
        target_layout.addWidget(self.file_count_label)

        manual_panel = QWidget()
        manual_layout = QVBoxLayout(manual_panel)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_header = QHBoxLayout()
        manual_header.addWidget(self._section_label("手動対応メモ"))
        manual_header.addStretch(1)
        clear_manual = self._button("クリア", self._clear_manual_review)
        clear_manual.setFixedHeight(26)
        manual_header.addWidget(clear_manual)
        manual_layout.addLayout(manual_header)
        self.manual_review_list = QListWidget()
        manual_layout.addWidget(self.manual_review_list, 1)

        self.list_splitter = QSplitter(Qt.Orientation.Vertical)
        self.list_splitter.setChildrenCollapsible(False)
        self.list_splitter.addWidget(target_panel)
        self.list_splitter.addWidget(manual_panel)
        self.list_splitter.setStretchFactor(0, 3)
        self.list_splitter.setStretchFactor(1, 1)
        self.list_splitter.setSizes([480, 180])
        left_layout.addWidget(self.list_splitter, 1)
        body_layout.addWidget(left, 0, 0)

        center = QFrame()
        center.setObjectName("previewPanel")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        self.preview_label = MaskPreviewLabel("プレビュー")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(320, 320)
        self.preview_label.setObjectName("preview")
        self.preview_label.stroke_requested.connect(self._paint_mask_stroke)
        self.preview_label.zoom_requested.connect(self._zoom_mask_preview)
        center_layout.addWidget(self.preview_label, 1)
        preview_bar = QFrame()
        preview_bar.setObjectName("previewBar")
        preview_bar_layout = QHBoxLayout(preview_bar)
        preview_bar_layout.setContentsMargins(10, 8, 10, 8)
        preview_bar_layout.addWidget(QLabel("表示"))
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItems(["元画像", "検出範囲", "処理結果"])
        self.preview_mode_combo.setCurrentText("処理結果")
        self.preview_mode_combo.currentTextChanged.connect(self._on_preview_mode_changed)
        preview_bar_layout.addWidget(self.preview_mode_combo)
        self.mask_edit_button = self._button("マスクを編集", self._toggle_mask_edit)
        self.mask_edit_button.setEnabled(False)
        preview_bar_layout.addWidget(self.mask_edit_button)
        self.brush_size_label = QLabel("ブラシ")
        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setRange(4, 200)
        self.brush_size_slider.setValue(40)
        self.brush_size_slider.setFixedWidth(100)
        self.brush_size_value = QLabel("40 px")
        self.brush_size_value.setObjectName("value")
        self.brush_size_slider.valueChanged.connect(self._on_brush_size_changed)
        for widget in (
            self.brush_size_label,
            self.brush_size_slider,
            self.brush_size_value,
        ):
            widget.setVisible(False)
            preview_bar_layout.addWidget(widget)
        preview_bar_layout.addStretch(1)
        self.analyze_button = self._button("この画像を再解析", self._analyze_current)
        preview_bar_layout.addWidget(self.analyze_button)
        center_layout.addWidget(preview_bar)
        body_layout.addWidget(center, 0, 1)

        right = self._panel()
        right.setFixedWidth(285)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(14, 12, 14, 12)
        right_layout.setSpacing(7)

        right_layout.addWidget(self._section_label("画像の種類"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["実写", "イラスト"])
        self.mode_combo.setCurrentText("イラスト")
        right_layout.addWidget(self.mode_combo)
        right_layout.addSpacing(8)

        right_layout.addWidget(self._section_label("検出対象"))
        self.penis_check = QCheckBox("penis")
        self.penis_check.setChecked(True)
        self.vagina_check = QCheckBox("vagina / pussy")
        self.vagina_check.setChecked(True)
        right_layout.addWidget(self.penis_check)
        right_layout.addWidget(self.vagina_check)
        right_layout.addSpacing(8)

        right_layout.addWidget(self._section_label("検出閾値"))
        self.threshold_value = QLabel("0.25")
        self.threshold_value.setObjectName("value")
        self.threshold_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_layout.addWidget(self.threshold_value)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(5, 95)
        self.threshold_slider.setValue(25)
        self.threshold_slider.valueChanged.connect(
            lambda value: self.threshold_value.setText(f"{value / 100:.2f}")
        )
        right_layout.addWidget(self.threshold_slider)
        right_layout.addSpacing(8)

        right_layout.addWidget(self._section_label("処理方法"))
        self.effect_combo = QComboBox()
        self.effect_combo.addItems(["モザイク", "ぼかし"])
        self.effect_combo.currentTextChanged.connect(self._on_effect_settings_changed)
        right_layout.addWidget(self.effect_combo)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("サイズ"))
        size_row.addStretch(1)
        self.effect_size_value = QLabel("16 px")
        self.effect_size_value.setObjectName("value")
        size_row.addWidget(self.effect_size_value)
        right_layout.addLayout(size_row)
        self.effect_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.effect_size_slider.setRange(4, 64)
        self.effect_size_slider.setValue(16)
        self.effect_size_slider.valueChanged.connect(self._on_effect_size_changed)
        right_layout.addWidget(self.effect_size_slider)
        right_layout.addSpacing(8)

        right_layout.addWidget(self._section_label("出力フォルダ"))
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit(str(self.app_root / "output"))
        self.output_edit.setReadOnly(True)
        output_row.addWidget(self.output_edit, 1)
        browse = self._button("…", self._choose_output)
        browse.setFixedWidth(38)
        output_row.addWidget(browse)
        right_layout.addLayout(output_row)

        self.open_output_button = self._button(
            "エクスプローラで表示", self._open_output_folder
        )
        right_layout.addWidget(self.open_output_button)

        right_layout.addWidget(self._section_label("出力suffix"))
        self.suffix_edit = QLineEdit("_mosaic")
        self.suffix_edit.setPlaceholderText("空欄可")
        right_layout.addWidget(self.suffix_edit)
        right_layout.addStretch(1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        right_layout.addWidget(self.progress)
        self.remove_after_process_check = QCheckBox("処理後にリストから除去")
        right_layout.addWidget(self.remove_after_process_check)
        self.process_current_button = self._button("表示中の1枚を処理して保存", self._process_current)
        self.process_current_button.setMinimumHeight(36)
        right_layout.addWidget(self.process_current_button)
        self.process_button = self._button("すべて処理して保存", self._process_all)
        self.process_button.setMinimumHeight(42)
        right_layout.addWidget(self.process_button)
        body_layout.addWidget(right, 0, 2)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #17191f; color: #d9dde5; font-family: "Segoe UI"; font-size: 10pt; }
            QFrame#toolbar, QFrame#panel, QFrame#previewBar { background: #20232b; }
            QFrame#previewPanel { background: #0e1015; border: 1px solid #303540; }
            QLabel#preview { color: #6f7787; background: #0e1015; }
            QLabel#section { color: #f0f2f6; font-weight: 600; }
            QLabel#muted { color: #8991a1; }
            QLabel#value { color: #53b8ff; }
            QPushButton { background: #356b92; color: white; border: none; border-radius: 3px; padding: 7px 12px; }
            QPushButton:hover { background: #4f89b2; }
            QPushButton:disabled { background: #3a3d45; color: #858993; }
            QListWidget, QLineEdit, QComboBox { background: #111319; color: #e5e8ef; border: 1px solid #353a46; border-radius: 3px; padding: 6px; }
            QListWidget::item { padding: 5px; }
            QListWidget::item:selected { background: #356b92; color: white; }
            QComboBox QAbstractItemView { background: #111319; color: #e5e8ef; selection-background-color: #356b92; }
            QCheckBox { spacing: 8px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QSlider::groove:horizontal { height: 5px; background: #3a404d; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #53b8ff; border-radius: 2px; }
            QSlider::handle:horizontal { width: 15px; margin: -5px 0; background: #e9edf4; border-radius: 7px; }
            QProgressBar { background: #2a2e38; border: none; height: 7px; border-radius: 3px; }
            QProgressBar::chunk { background: #53b8ff; border-radius: 3px; }
            QSplitter::handle:vertical { background: #303540; height: 6px; margin: 2px 0; }
            QSplitter::handle:vertical:hover { background: #53b8ff; }
            """
        )

    @staticmethod
    def _panel() -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        return panel

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("section")
        return label

    @staticmethod
    def _button(text: str, callback) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def _add_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "処理する画像を選択",
            "",
            "画像 (*.png *.jpg *.jpeg);;すべてのファイル (*.*)",
        )
        self._add_image_paths(Path(raw_path) for raw_path in paths)

    def _add_image_paths(self, paths) -> int:
        existing = set(self.image_paths)
        added = 0
        for raw_path in paths:
            path = Path(raw_path).resolve()
            if (
                path.is_file()
                and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
                and path not in existing
                and len(self.image_paths) < MAX_IMAGES
            ):
                self.image_paths.append(path)
                item = QListWidgetItem(path.name)
                item.setToolTip(str(path))
                self.file_list.addItem(item)
                existing.add(path)
                added += 1
        self.file_count_label.setText(f"{len(self.image_paths)} / {MAX_IMAGES}")
        if self.image_paths and self.file_list.currentRow() < 0:
            self.file_list.setCurrentRow(0)
        return added

    @staticmethod
    def _has_supported_drop(event: QDragEnterEvent | QDropEvent) -> bool:
        return any(
            url.isLocalFile()
            and Path(url.toLocalFile()).suffix.lower() in {".png", ".jpg", ".jpeg"}
            for url in event.mimeData().urls()
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._has_supported_drop(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        added = self._add_image_paths(paths)
        if added:
            self.status_label.setText(f"{added}件の画像を追加しました")
            event.acceptProposedAction()
        else:
            event.ignore()

    def _remove_selected(self) -> None:
        rows = {self.file_list.row(item) for item in self.file_list.selectedItems()}
        self._remove_rows(rows, remember_for_manual_review=False)

    def _remove_selected_to_manual(self) -> None:
        rows = {self.file_list.row(item) for item in self.file_list.selectedItems()}
        self._remove_rows(rows, remember_for_manual_review=True)

    def _remove_processed_paths(self, paths: list[Path]) -> None:
        processed = set(paths)
        rows = {index for index, path in enumerate(self.image_paths) if path in processed}
        self._remove_rows(rows, remember_for_manual_review=False)

    def _remove_rows(self, rows, remember_for_manual_review: bool) -> None:
        rows = sorted(set(rows), reverse=True)
        if not rows:
            return
        next_row = rows[-1]
        if not self._confirm_discard_mask_edit("画像をリストから除去", refresh_after=False):
            return
        self.analysis_generation += 1
        removed_paths = [self.image_paths[row] for row in reversed(rows)]
        if remember_for_manual_review:
            for path in removed_paths:
                if path not in self.manual_review_paths:
                    self.manual_review_paths.append(path)
                    item = QListWidgetItem(path.name)
                    item.setToolTip(str(path))
                    self.manual_review_list.addItem(item)
        self.file_list.blockSignals(True)
        for row in rows:
            self.file_list.takeItem(row)
            del self.image_paths[row]
        self.file_list.clearSelection()
        self.file_list.setCurrentRow(-1)
        self.file_list.blockSignals(False)
        self.current_result = None
        self.file_count_label.setText(f"{len(self.image_paths)} / {MAX_IMAGES}")
        if next_row < len(self.image_paths):
            self.file_list.setCurrentRow(next_row)
        else:
            self.preview_rgb = None
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.set_image_rect(QRectF())
            self.preview_label.setText("プレビュー")
            self.status_label.setText("画像を選択してください")

    def _clear_manual_review(self) -> None:
        self.manual_review_paths.clear()
        self.manual_review_list.clear()

    def _clear_images(self) -> None:
        if not self.image_paths:
            return
        if not self._confirm_discard_mask_edit("画像リストを消去", refresh_after=False):
            return
        self.image_paths.clear()
        self.file_list.clear()
        self.current_result = None
        self.preview_rgb = None
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.set_image_rect(QRectF())
        self.preview_label.setText("プレビュー")
        self.file_count_label.setText(f"0 / {MAX_IMAGES}")

    def _on_selection_changed(self, row: int) -> None:
        if 0 <= row < len(self.image_paths):
            next_path = self.image_paths[row]
            if (
                self.mask_edit_active
                and next_path != self.mask_edit_source_path
                and not self._confirm_discard_mask_edit(
                    "別の画像へ切り替え", refresh_after=False
                )
            ):
                if self.mask_edit_source_path in self.image_paths:
                    source_row = self.image_paths.index(self.mask_edit_source_path)
                    self.file_list.blockSignals(True)
                    self.file_list.clearSelection()
                    self.file_list.setCurrentRow(source_row)
                    self.file_list.item(source_row).setSelected(True)
                    self.file_list.blockSignals(False)
                return
            self.current_result = None
            self._show_original(next_path)
            self.analysis_generation += 1
            generation = self.analysis_generation
            QTimer.singleShot(0, lambda: self._start_analysis(generation, False))

    def _on_preview_mode_changed(self, _value: str) -> None:
        self._update_mask_editor_interaction()
        self._update_mask_edit_controls()
        self._refresh_preview()

    def _on_brush_size_changed(self, value: int) -> None:
        self.brush_size_value.setText(f"{value} px")
        self._update_brush_display_size()

    def _on_effect_size_changed(self, value: int) -> None:
        self.effect_size_value.setText(f"{value} px")
        self._on_effect_settings_changed()

    def _on_effect_settings_changed(self, _value=None) -> None:
        if self.mask_edit_active and self.preview_mode_combo.currentText() == "処理結果":
            self._refresh_preview()

    def _toggle_mask_edit(self) -> None:
        if self.mask_edit_active:
            self._confirm_discard_mask_edit("マスク編集を終了", refresh_after=True)
            return
        path = self._selected_path()
        if (
            path is None
            or self.current_result is None
            or self.current_result.source_path != path
            or self.preview_mode_combo.currentText() != "検出範囲"
        ):
            return
        self.mask_edit_active = True
        self.mask_edit_dirty = False
        self.mask_edit_source_path = path
        self.edited_mask = self.current_result.mask.copy()
        self.mask_edit_original_bgr = load_image_bgr(path)
        self.mask_edit_zoom_index = 0
        self._update_mask_editor_interaction()
        self._update_mask_edit_controls()
        self._refresh_preview()
        self.status_label.setText(
            "マスク編集中: ドラッグで追加 / Alt+ドラッグで削除"
        )

    def _confirm_discard_mask_edit(
        self, action: str, refresh_after: bool = True
    ) -> bool:
        if not self.mask_edit_active:
            return True
        if self.mask_edit_dirty:
            choice = QMessageBox.warning(
                self,
                "マスク編集の破棄",
                f"{action}すると、編集中のマスクは破棄されます。\n"
                "この操作を続けますか？",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if choice != QMessageBox.StandardButton.Discard:
                return False
        self._end_mask_edit(refresh_after)
        return True

    def _end_mask_edit(self, refresh_preview: bool) -> None:
        self.mask_edit_active = False
        self.mask_edit_dirty = False
        self.mask_edit_source_path = None
        self.edited_mask = None
        self.mask_edit_original_bgr = None
        self.mask_edit_zoom_index = 0
        self.preview_label.set_mask_editing(False)
        self.preview_label.set_zoom_enabled(False)
        self._update_mask_edit_controls()
        if refresh_preview:
            self._refresh_preview()

    def _update_mask_edit_controls(self) -> None:
        path = self._selected_path()
        can_start = (
            not self.busy
            and self.current_result is not None
            and self.current_result.source_path == path
            and self.preview_mode_combo.currentText() == "検出範囲"
        )
        self.mask_edit_button.setText(
            "編集を終了（破棄）" if self.mask_edit_active else "マスクを編集"
        )
        self.mask_edit_button.setEnabled(
            not self.busy and (self.mask_edit_active or can_start)
        )
        for widget in (
            self.brush_size_label,
            self.brush_size_slider,
            self.brush_size_value,
        ):
            widget.setVisible(self.mask_edit_active)
        self.preview_mode_combo.setEnabled(not self.busy)
        self.mode_combo.setEnabled(not self.mask_edit_active and not self.busy)
        self.penis_check.setEnabled(not self.mask_edit_active and not self.busy)
        self.vagina_check.setEnabled(not self.mask_edit_active and not self.busy)
        self.threshold_slider.setEnabled(not self.mask_edit_active and not self.busy)
        self.analyze_button.setEnabled(not self.mask_edit_active and not self.busy)
        self.process_current_button.setEnabled(not self.busy)
        self.process_button.setEnabled(not self.mask_edit_active and not self.busy)

    def _update_mask_editor_interaction(self) -> None:
        self.preview_label.set_mask_editing(
            self.mask_edit_active
            and not self.busy
            and self.preview_mode_combo.currentText() == "検出範囲"
        )
        self.preview_label.set_zoom_enabled(self.mask_edit_active and not self.busy)
        self._update_brush_display_size()

    def _zoom_mask_preview(self, position: QPointF, direction: int) -> None:
        if (
            not self.mask_edit_active
            or self.busy
            or self.preview_rgb is None
            or self.preview_image_rect.isEmpty()
            or direction == 0
        ):
            return

        next_index = max(
            0,
            min(
                len(MASK_EDIT_ZOOM_LEVELS) - 1,
                self.mask_edit_zoom_index + (1 if direction > 0 else -1),
            ),
        )
        if next_index == self.mask_edit_zoom_index:
            return

        current_rect = self.preview_image_rect
        if current_rect.contains(position):
            display_anchor = position
            image_anchor = QPointF(
                (position.x() - current_rect.left()) / current_rect.width(),
                (position.y() - current_rect.top()) / current_rect.height(),
            )
        else:
            display_anchor = current_rect.center()
            image_anchor = QPointF(0.5, 0.5)

        self.mask_edit_zoom_index = next_index
        self._render_preview((display_anchor, image_anchor))

    def _update_brush_display_size(self) -> None:
        if self.edited_mask is None or self.preview_image_rect.isEmpty():
            return
        image_width = self.edited_mask.shape[1]
        display_diameter = (
            self.brush_size_slider.value()
            * self.preview_image_rect.width()
            / max(1, image_width)
        )
        self.preview_label.set_brush_diameter(display_diameter)

    def _selected_path(self) -> Path | None:
        row = self.file_list.currentRow()
        return self.image_paths[row] if 0 <= row < len(self.image_paths) else None

    def _settings(self) -> ProcessingSettings:
        targets = set()
        if self.penis_check.isChecked():
            targets.add("penis")
        if self.vagina_check.isChecked():
            targets.add("vagina")
        if not targets:
            raise ValueError("検出対象を1つ以上選択してください。")
        return ProcessingSettings(
            mode=ImageMode.PHOTO if self.mode_combo.currentText() == "実写" else ImageMode.ILLUSTRATION,
            targets=frozenset(targets),
            confidence_threshold=self.threshold_slider.value() / 100,
            effect=EffectType.MOSAIC if self.effect_combo.currentText() == "モザイク" else EffectType.BLUR,
            effect_size=self.effect_size_slider.value(),
        )

    def _analyze_current(self) -> None:
        self.analysis_generation += 1
        self._start_analysis(self.analysis_generation, True)

    def _start_analysis(self, generation: int, report_errors: bool) -> None:
        if generation != self.analysis_generation:
            return
        path = self._selected_path()
        if path is None:
            if report_errors:
                QMessageBox.information(self, APP_NAME, "画像を選択してください。")
            return
        try:
            settings = self._settings()
        except ValueError as error:
            if report_errors:
                QMessageBox.warning(self, "設定", str(error))
            else:
                self.status_label.setText(str(error))
            return
        self._set_busy(True, "解析しています…")

        def work() -> None:
            try:
                result = self.pipeline.analyze(path, settings)
                self.events.put(("analysis", (generation, result)))
            except Exception as error:
                self.events.put(("analysis_error", (generation, error, report_errors)))

        threading.Thread(target=work, daemon=True).start()

    def _filename_suffix(self) -> str:
        suffix = self.suffix_edit.text()
        if any(character in suffix for character in '<>:"/\\|?*'):
            raise ValueError('suffixにファイル名として使えない文字（<>:"/\\|?*）が含まれています。')
        if suffix.endswith((" ", ".")):
            raise ValueError("suffixの末尾に空白またはピリオドは使用できません。")
        return suffix

    def _process_current(self) -> None:
        path = self._selected_path()
        if path is None:
            QMessageBox.information(self, APP_NAME, "画像を選択してください。")
            return
        try:
            settings = self._settings()
            filename_suffix = self._filename_suffix()
        except ValueError as error:
            QMessageBox.warning(self, "設定", str(error))
            return
        output_dir = Path(self.output_edit.text())
        remove_after = self.remove_after_process_check.isChecked()
        edited_mask = self.edited_mask.copy() if self.mask_edit_active else None
        edit_reference = self.current_result if self.mask_edit_active else None
        if self.mask_edit_active and (
            edited_mask is None
            or edit_reference is None
            or self.mask_edit_source_path != path
        ):
            QMessageBox.warning(self, "マスク編集", "編集中のマスクを処理できません。")
            return
        self._set_busy(True, f"{path.name}を処理しています…")

        def work() -> None:
            try:
                if edited_mask is not None and edit_reference is not None:
                    result = self.pipeline.process_with_mask(
                        path,
                        edited_mask,
                        settings,
                        edit_reference.detections,
                        edit_reference.used_box_fallbacks,
                        edit_reference.below_threshold_detections,
                    )
                else:
                    result = self.pipeline.analyze(path, settings)
                output = self.pipeline.save(result, output_dir, filename_suffix)
                self.events.put(
                    (
                        "single_complete",
                        (result, output, remove_after, edited_mask is not None),
                    )
                )
            except Exception as error:
                self.events.put(("error", error))

        threading.Thread(target=work, daemon=True).start()

    def _process_all(self) -> None:
        if self.mask_edit_active:
            QMessageBox.information(
                self,
                "マスク編集",
                "編集中のマスクは「表示中の1枚を処理して保存」でのみ使用できます。",
            )
            return
        if not self.image_paths:
            QMessageBox.information(self, APP_NAME, "画像を追加してください。")
            return
        try:
            settings = self._settings()
            filename_suffix = self._filename_suffix()
        except ValueError as error:
            QMessageBox.warning(self, "設定", str(error))
            return
        output_dir = Path(self.output_edit.text())
        paths = list(self.image_paths)
        remove_after = self.remove_after_process_check.isChecked()
        self._set_busy(True, "一括処理を開始します…")

        def work() -> None:
            outputs: list[Path] = []
            completed_paths: list[Path] = []
            try:
                for index, path in enumerate(paths, start=1):
                    self.events.put(("progress", (index - 1, len(paths), path.name)))
                    result = self.pipeline.analyze(path, settings)
                    outputs.append(self.pipeline.save(result, output_dir, filename_suffix))
                    completed_paths.append(path)
                self.events.put(("complete", (outputs, completed_paths, remove_after)))
            except Exception as error:
                self.events.put(
                    ("batch_error", (error, completed_paths, remove_after))
                )

        threading.Thread(target=work, daemon=True).start()

    def _choose_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "出力フォルダを選択", self.output_edit.text()
        )
        if selected:
            self.output_edit.setText(selected)

    def _open_output_folder(self) -> None:
        output_dir = Path(self.output_edit.text())
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            QMessageBox.warning(
                self,
                "出力フォルダ",
                f"フォルダを作成できません。\n{error}",
            )
            return
        output_url = QUrl.fromLocalFile(str(output_dir.resolve()))
        if not QDesktopServices.openUrl(output_url):
            QMessageBox.warning(
                self,
                "出力フォルダ",
                "エクスプローラでフォルダを開けませんでした。",
            )

    def _show_original(self, path: Path) -> None:
        try:
            self._set_preview_bgr(load_image_bgr(path))
            self.status_label.setText(path.name)
        except Exception as error:
            self.status_label.setText(f"読み込みエラー: {error}")

    def _refresh_preview(self, _value: str | None = None) -> None:
        path = self._selected_path()
        if path is None:
            return
        mode = self.preview_mode_combo.currentText()
        if mode == "元画像" or self.current_result is None:
            self._show_original(path)
        elif mode == "検出範囲":
            original = (
                self.mask_edit_original_bgr
                if self.mask_edit_active and self.mask_edit_original_bgr is not None
                else load_image_bgr(path)
            )
            mask = (
                self.edited_mask
                if self.mask_edit_active and self.edited_mask is not None
                else self.current_result.mask
            )
            self._set_preview_bgr(
                visualize_detection(
                    original,
                    mask,
                    self.current_result.detections,
                    self.current_result.below_threshold_detections,
                )
            )
        elif self.mask_edit_active and self.edited_mask is not None:
            original = (
                self.mask_edit_original_bgr
                if self.mask_edit_original_bgr is not None
                else load_image_bgr(path)
            )
            effect = (
                EffectType.MOSAIC
                if self.effect_combo.currentText() == "モザイク"
                else EffectType.BLUR
            )
            self._set_preview_bgr(
                apply_effect(
                    original,
                    self.edited_mask,
                    effect,
                    self.effect_size_slider.value(),
                )
            )
        else:
            self._set_preview_bgr(self.current_result.image_bgr)

    def _paint_mask_stroke(
        self, start: QPointF, end: QPointF, erase: bool
    ) -> None:
        if not self.mask_edit_active or self.edited_mask is None:
            return
        start_point = self._preview_point_to_mask(start)
        end_point = self._preview_point_to_mask(end)
        if start_point is None or end_point is None:
            return
        if paint_mask_stroke(
            self.edited_mask,
            start_point,
            end_point,
            self.brush_size_slider.value(),
            erase,
        ):
            self.mask_edit_dirty = True
            self._refresh_preview()

    def _preview_point_to_mask(self, position: QPointF) -> tuple[int, int] | None:
        if self.edited_mask is None or self.preview_image_rect.isEmpty():
            return None
        rect = self.preview_image_rect
        height, width = self.edited_mask.shape
        x = int((position.x() - rect.left()) * width / rect.width())
        y = int((position.y() - rect.top()) * height / rect.height())
        return (
            max(0, min(width - 1, x)),
            max(0, min(height - 1, y)),
        )

    def _set_preview_bgr(self, image_bgr: np.ndarray) -> None:
        self.preview_rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        self._render_preview()

    @staticmethod
    def _clamp_preview_axis(
        position: float, image_size: float, viewport_start: float, viewport_size: float
    ) -> float:
        if image_size <= viewport_size:
            return max(
                viewport_start,
                min(viewport_start + viewport_size - image_size, position),
            )
        return max(
            viewport_start + viewport_size - image_size,
            min(viewport_start, position),
        )

    def _render_preview(
        self, zoom_anchor: tuple[QPointF, QPointF] | None = None
    ) -> None:
        if self.preview_rgb is None:
            return
        height, width, channels = self.preview_rgb.shape
        image = QImage(
            self.preview_rgb.data,
            width,
            height,
            width * channels,
            QImage.Format.Format_RGB888,
        ).copy()
        target = self.preview_label.size()
        available_width = max(1, target.width() - 20)
        available_height = max(1, target.height() - 20)
        base_pixmap = QPixmap.fromImage(image).scaled(
            available_width,
            available_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        zoom = (
            MASK_EDIT_ZOOM_LEVELS[self.mask_edit_zoom_index]
            if self.mask_edit_active
            else MASK_EDIT_ZOOM_LEVELS[0]
        )
        pixmap = QPixmap.fromImage(image).scaled(
            max(1, round(base_pixmap.width() * zoom)),
            max(1, round(base_pixmap.height() * zoom)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        target_size = (target.width(), target.height())
        can_reuse_position = (
            self.mask_edit_active
            and self.mask_edit_zoom_index > 0
            and not self.preview_image_rect.isEmpty()
            and self.preview_render_target_size == target_size
        )
        if zoom_anchor is not None:
            display_anchor, image_anchor = zoom_anchor
            left = display_anchor.x() - image_anchor.x() * pixmap.width()
            top = display_anchor.y() - image_anchor.y() * pixmap.height()
        elif can_reuse_position:
            left = self.preview_image_rect.left()
            top = self.preview_image_rect.top()
        else:
            left = (target.width() - pixmap.width()) / 2
            top = (target.height() - pixmap.height()) / 2

        left = self._clamp_preview_axis(
            left, float(pixmap.width()), 10.0, float(available_width)
        )
        top = self._clamp_preview_axis(
            top, float(pixmap.height()), 10.0, float(available_height)
        )
        self.preview_image_rect = QRectF(
            left, top, float(pixmap.width()), float(pixmap.height())
        )
        self.preview_render_target_size = target_size
        self.preview_label.set_image_rect(self.preview_image_rect)
        self._update_brush_display_size()
        if self.mask_edit_active:
            painter = QPainter(pixmap)
            pen = QPen(QColor("#ff3b30"))
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawRect(pixmap.rect().adjusted(2, 2, -3, -3))
            painter.end()
        self.preview_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.preview_rgb is not None:
            QTimer.singleShot(0, self._render_preview)

    def _set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        self.remove_after_process_check.setEnabled(not busy)
        self._update_mask_editor_interaction()
        self._update_mask_edit_controls()
        self.status_label.setText(status)
        if not busy:
            self.progress.setValue(0)

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "analysis":
                    generation, result = payload  # type: ignore[misc]
                    if generation != self.analysis_generation:
                        continue
                    if self._selected_path() == result.source_path:
                        self.current_result = result
                        self.preview_mode_combo.setCurrentText("処理結果")
                        self._refresh_preview()
                    detail = f"{len(result.detections)}件検出"
                    if result.used_box_fallbacks:
                        detail += f" / 矩形補完 {result.used_box_fallbacks}件"
                    self._set_busy(False, detail)
                elif kind == "analysis_error":
                    generation, error, report_errors = payload  # type: ignore[misc]
                    if generation != self.analysis_generation:
                        continue
                    self._set_busy(False, "解析エラー")
                    if report_errors:
                        QMessageBox.critical(self, "解析エラー", str(error))
                elif kind == "progress":
                    index, total, name = payload  # type: ignore[misc]
                    self.progress.setValue(int(index / total * 100))
                    self.status_label.setText(f"{index + 1}/{total}: {name}")
                elif kind == "complete":
                    outputs, completed_paths, remove_after = payload  # type: ignore[misc]
                    self.progress.setValue(100)
                    self._set_busy(False, f"{len(outputs)}件を書き出しました")
                    if remove_after:
                        self._remove_processed_paths(completed_paths)
                    QMessageBox.information(
                        self,
                        "完了",
                        f"{len(outputs)}件を次のフォルダへ保存しました。\n{self.output_edit.text()}",
                    )
                elif kind == "single_complete":
                    result, output, remove_after, used_edited_mask = payload  # type: ignore[misc]
                    if used_edited_mask:
                        self._end_mask_edit(refresh_preview=False)
                    if not remove_after and self._selected_path() == result.source_path:
                        self.current_result = result
                        self.preview_mode_combo.setCurrentText("処理結果")
                        self._refresh_preview()
                    self._set_busy(False, f"{output.name}を書き出しました")
                    if remove_after:
                        self._remove_processed_paths([result.source_path])
                    QMessageBox.information(self, "完了", f"保存しました。\n{output}")
                elif kind == "batch_error":
                    error, completed_paths, remove_after = payload  # type: ignore[misc]
                    self._set_busy(False, "一括処理エラー")
                    if remove_after:
                        self._remove_processed_paths(completed_paths)
                    QMessageBox.critical(
                        self,
                        "処理エラー",
                        f"{len(completed_paths)}件の保存後にエラーが発生しました。\n{error}",
                    )
                elif kind == "error":
                    self._set_busy(False, "エラー")
                    QMessageBox.critical(self, "処理エラー", str(payload))
        except queue.Empty:
            pass

    def _update_model_status(self) -> None:
        missing = [path.name for path in required_model_paths(self.model_dir) if not path.exists()]
        if missing:
            self.status_label.setText(
                f"モデル未配置（{len(missing)}個）: scripts/download_models.py を実行してください"
            )

    def closeEvent(self, event) -> None:
        if self._confirm_discard_mask_edit("アプリを終了", refresh_after=False):
            super().closeEvent(event)
        else:
            event.ignore()


def run() -> None:
    app = QApplication(sys.argv)
    window = AutoMosaicWindow()
    smoke_test = "--smoke-test" in sys.argv
    if not smoke_test:
        window.show()
    if smoke_test:
        missing = [path for path in required_model_paths(window.model_dir) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Packaged models are missing: {missing}")
        from auto_mosaic.smoke import run_model_smoke

        run_model_smoke(window.model_dir)
        QTimer.singleShot(250, app.quit)
    raise SystemExit(app.exec())
