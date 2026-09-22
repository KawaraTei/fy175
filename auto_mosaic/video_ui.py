"""Independent video window. Still-image selection/editor state is never shared."""
from __future__ import annotations

import bisect
import queue
import threading
import time
from pathlib import Path

import cv2
from PySide6.QtCore import QRectF, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QProgressBar,
    QPushButton, QScrollArea, QSlider, QTableView, QVBoxLayout, QWidget,
)

from auto_mosaic.domain import EffectType, ImageMode, ProcessingSettings
from auto_mosaic.image_ops import apply_effect, visualize_detection
from auto_mosaic.ui_style import APP_STYLE, section_label
from auto_mosaic.video_analysis import VideoAnalysis, VideoAnalysisSettings, analyze_video, export_video
from auto_mosaic.video_io import VIDEO_EXTENSIONS, VideoCancelled, decode_frames, probe_video
from auto_mosaic.video_timeline import ChangeTableModel, MaskCountTimeline, time_text


class VideoPreview(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.setMinimumSize(320, 180)

    def set_image(self, image):
        if image is None:
            self.image = None
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0e1015"))
        if self.image is None:
            painter.setPen(QColor("#aab2c2"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "動画を開いてください")
            return
        size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRectF((self.width() - size.width()) / 2, (self.height() - size.height()) / 2,
                        size.width(), size.height())
        painter.drawImage(target, self.image)


class VideoMosaicWindow(QMainWindow):
    def __init__(self, model_dir: Path | None = None, output_dir: Path | None = None):
        super().__init__()
        from auto_mosaic.ui import application_root, resource_root
        self.model_dir = model_dir or resource_root() / "models"
        self.output_dir = output_dir or application_root() / "output"
        self.info = None
        self.analysis: VideoAnalysis | None = None
        self.current_frame = 0
        self.preview_data = None
        self.busy = False
        self.job_kind = ""
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None
        self.closing = False
        self.change_frames = []
        self.last_output = None
        self.setWindowTitle("FY175AutoMosaic — 動画処理")
        self.resize(1340, 940)
        self.setMinimumSize(1060, 760)
        self.setAcceptDrops(True)
        self._build_layout()
        self._apply_style()
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start(80)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)
        self._update_controls()

    def _build_layout(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        self.setCentralWidget(central)
        header = QHBoxLayout()
        self.open_button = QPushButton("動画を開く")
        self.open_button.clicked.connect(self._choose_video)
        header.addWidget(self.open_button)
        self.filename = QLabel("動画未選択")
        self.filename.setWordWrap(True)
        header.addWidget(self.filename, 1)
        self.info_label = QLabel("")
        self.info_label.setObjectName("muted")
        header.addWidget(self.info_label)
        layout.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(14)
        review = QVBoxLayout()
        review.setSpacing(8)
        self.preview = VideoPreview()
        review.addWidget(self.preview, 1)
        transport = QHBoxLayout()
        self.play_button = QPushButton("再生")
        self.play_button.setToolTip("解析済みプレビューを無音で再生")
        self.play_button.clicked.connect(self._toggle_play)
        transport.addWidget(self.play_button)
        self.back_button = QPushButton("−1フレーム")
        self.back_button.clicked.connect(lambda: self._user_seek(self.current_frame - 1))
        self.forward_button = QPushButton("＋1フレーム")
        self.forward_button.clicked.connect(lambda: self._user_seek(self.current_frame + 1))
        transport.addWidget(self.back_button)
        transport.addWidget(self.forward_button)
        self.time_label = QLabel("00:00.000 / 00:00.000")
        transport.addWidget(self.time_label, 1)
        self.view_mode = QComboBox()
        self.view_mode.addItems(["元画像", "検出範囲", "処理結果"])
        self.view_mode.setCurrentText("検出範囲")
        self.view_mode.currentIndexChanged.connect(self._render_preview)
        transport.addWidget(self.view_mode)
        review.addLayout(transport)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.valueChanged.connect(self._user_seek)
        review.addWidget(self.seek_slider)
        self.frame_label = QLabel("解析前")
        review.addWidget(self.frame_label)

        timeline_header = QHBoxLayout()
        timeline_header.addWidget(QLabel("検出マスク数"))
        legend = QLabel("青: マスク数　橙: 0件・短い減少")
        legend.setObjectName("muted")
        timeline_header.addWidget(legend, 1)
        timeline_header.addWidget(QLabel("表示範囲"))
        self.zoom = QComboBox()
        for label, value in (("全体", 0), ("10秒", 10), ("1秒", 1)):
            self.zoom.addItem(label, value)
        self.zoom.currentIndexChanged.connect(lambda: self.timeline.set_window_seconds(self.zoom.currentData()))
        timeline_header.addWidget(self.zoom)
        review.addLayout(timeline_header)
        self.timeline = MaskCountTimeline()
        self.timeline.setFixedHeight(128)
        self.timeline.frame_requested.connect(self._user_seek)
        review.addWidget(self.timeline)
        changes_header = QHBoxLayout()
        self.change_label = QLabel("変化点 0件")
        changes_header.addWidget(self.change_label, 1)
        self.previous_change = QPushButton("前の変化点")
        self.previous_change.clicked.connect(lambda: self._jump_change(-1))
        self.next_change = QPushButton("次の変化点")
        self.next_change.clicked.connect(lambda: self._jump_change(1))
        changes_header.addWidget(self.previous_change)
        changes_header.addWidget(self.next_change)
        review.addLayout(changes_header)
        self.change_model = ChangeTableModel(self)
        self.change_table = QTableView()
        self.change_table.setModel(self.change_model)
        self.change_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.change_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.change_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.change_table.verticalHeader().hide()
        self.change_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.change_table.verticalHeader().setDefaultSectionSize(26)
        self.change_table.horizontalHeader().setFixedHeight(28)
        self.change_table.setFixedHeight(110)
        self.change_table.selectionModel().currentRowChanged.connect(self._select_change)
        review.addWidget(self.change_table)
        body.addLayout(review, 1)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFixedWidth(292)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        panel = QFrame()
        panel.setObjectName("panel")
        settings = QVBoxLayout(panel)
        settings.setContentsMargins(14, 12, 14, 12)
        settings.setSpacing(7)
        settings.addWidget(section_label("画像の種類"))
        self.mode = QComboBox()
        self.mode.addItem("実写", ImageMode.PHOTO.value)
        self.mode.addItem("イラスト", ImageMode.ILLUSTRATION.value)
        self.mode.setCurrentIndex(1)
        settings.addWidget(self.mode)
        settings.addSpacing(8)
        settings.addWidget(section_label("検出対象"))
        self.penis_check = QCheckBox("penis")
        self.vagina_check = QCheckBox("vagina / pussy")
        for check in (self.penis_check, self.vagina_check):
            check.setChecked(True)
            settings.addWidget(check)
        settings.addSpacing(8)
        settings.addWidget(section_label("検出閾値"))
        self.threshold_value = QLabel("0.25")
        self.threshold_value.setObjectName("value")
        self.threshold_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        settings.addWidget(self.threshold_value)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(5, 95)
        self.threshold_slider.setValue(25)
        self.threshold_slider.valueChanged.connect(lambda value: self.threshold_value.setText(f"{value / 100:.2f}"))
        settings.addWidget(self.threshold_slider)
        settings.addSpacing(8)

        settings.addWidget(section_label("処理方法"))
        self.effect = QComboBox()
        self.effect.addItem("モザイク", EffectType.MOSAIC.value)
        self.effect.addItem("ぼかし", EffectType.BLUR.value)
        self.effect.currentIndexChanged.connect(self._render_preview)
        settings.addWidget(self.effect)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("サイズ"))
        size_row.addStretch(1)
        self.effect_size_value = QLabel("16 px")
        self.effect_size_value.setObjectName("value")
        size_row.addWidget(self.effect_size_value)
        settings.addLayout(size_row)
        self.effect_size = QSlider(Qt.Orientation.Horizontal)
        self.effect_size.setRange(4, 64)
        self.effect_size.setValue(16)
        self.effect_size.valueChanged.connect(lambda value: self.effect_size_value.setText(f"{value} px"))
        self.effect_size.valueChanged.connect(self._render_preview)
        settings.addWidget(self.effect_size)
        settings.addSpacing(8)

        settings.addWidget(section_label("輪郭更新"))
        self.interval = QComboBox()
        for label, value in (("毎フレーム（精度優先）", 1), ("3フレームごと（高速）", 3), ("5フレームごと（高速）", 5)):
            self.interval.addItem(label, value)
        settings.addWidget(self.interval)
        self.interval_note = QLabel("検出は全フレームで実行します。")
        self.interval_note.setWordWrap(True)
        self.interval_note.setObjectName("muted")
        settings.addWidget(self.interval_note)
        self.analysis_controls = [self.mode, self.penis_check, self.vagina_check, self.threshold_slider, self.interval]
        self.mode.currentIndexChanged.connect(self._analysis_settings_changed)
        self.interval.currentIndexChanged.connect(self._analysis_settings_changed)
        self.threshold_slider.valueChanged.connect(self._analysis_settings_changed)
        self.penis_check.toggled.connect(self._analysis_settings_changed)
        self.vagina_check.toggled.connect(self._analysis_settings_changed)
        self.analyze_button = QPushButton("動画を解析")
        self.analyze_button.clicked.connect(self._analyze)
        settings.addWidget(self.analyze_button)
        self.summary_label = QLabel("解析後にタイムラインを確認できます。")
        self.summary_label.setWordWrap(True)
        settings.addWidget(self.summary_label)
        settings.addStretch(1)
        settings_scroll.setWidget(panel)
        sidebar = QVBoxLayout()
        sidebar.setSpacing(10)
        sidebar.addWidget(settings_scroll, 1)
        self.format_label = QLabel("MP4 / H.264 / AAC\n元動画の先頭の音声を引き継ぎます。\n可変fpsは固定fpsへ変換します。")
        self.format_label.setWordWrap(True)
        self.format_label.setObjectName("muted")
        sidebar.addWidget(self.format_label)
        self.export_button = QPushButton("動画を書き出す…")
        self.export_button.clicked.connect(self._export)
        sidebar.addWidget(self.export_button)
        self.output_button = QPushButton("出力フォルダを開く")
        self.output_button.clicked.connect(self._open_output)
        sidebar.addWidget(self.output_button)
        body.addLayout(sidebar)
        layout.addLayout(body, 1)
        footer = QHBoxLayout()
        self.status = QLabel("動画を開いてください。")
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        self.cancel_button = QPushButton("中止")
        self.cancel_button.clicked.connect(self._cancel)
        footer.addWidget(self.cancel_button)
        layout.addLayout(footer)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(16)
        layout.addWidget(self.progress)

    def _apply_style(self):
        # The image window owns the common visual language. Only video-specific
        # surfaces and clearer control boundaries are added here.
        self.setStyleSheet(APP_STYLE + """
            QPushButton { border: 1px solid #6386a2; }
            QPushButton:disabled { border-color: #545a66; }
            QPushButton:focus, QComboBox:focus { border-color: #a2d9ff; }
            QComboBox { border-color: #647084; }
            QComboBox:disabled { color: #858993; border-color: #414753; }
            QTableView { background: #111319; gridline-color: #303744; border: 1px solid #303744; selection-background-color: #356b92; }
            QHeaderView::section { background: #262c37; color: #c9d2df; padding: 5px; border: none; }
            QProgressBar { text-align: center; }
            QScrollBar:vertical { background: #20232b; width: 10px; }
            QScrollBar::handle:vertical { background: #556171; min-height: 24px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

    def _settings(self):
        targets = set()
        if self.penis_check.isChecked():
            targets.add("penis")
        if self.vagina_check.isChecked():
            targets.add("vagina")
        # Qt stores str-enums as plain strings: reconstruct them at the UI boundary.
        return VideoAnalysisSettings(ProcessingSettings(ImageMode(self.mode.currentData()), frozenset(targets),
                                                        self.threshold_slider.value() / 100), self.interval.currentData())

    def _effect_type(self) -> EffectType:
        return EffectType(self.effect.currentData())

    def _analysis_settings_changed(self):
        if self.busy:
            return
        self._stop_play()
        if self.analysis is not None:
            self._discard_analysis()
            self.view_mode.setCurrentText("元画像")
            self.progress.setValue(0)
            self.time_label.setText(f"{time_text(self.current_frame / float(self.info.fps))} / {time_text(self.info.duration)}")
            self.frame_label.setText(f"フレーム {self.current_frame + 1:,} / 未解析")
            self.status.setText("検出設定を変更しました。動画を再解析してください。")
            self.summary_label.setText("再解析が必要です。")
        self.interval_note.setText("検出は全フレームで実行します。" if self.interval.currentData() == 1 else
                                  "全フレームで検出。輪郭の移動・拡縮で補間（変形には遅れあり）。")
        self._update_controls()

    def _discard_analysis(self):
        if self.analysis is not None:
            self.analysis.close()
            self.analysis = None
        # Keep the source frame visible when only analysis becomes stale.
        if self.preview_data is not None:
            self.preview.set_image(self.preview_data[0])
        self.preview_data = None
        self.timeline.set_frames([], 1)
        self.change_model.set_changes([], 1)
        self.change_frames = []
        self.change_label.setText("変化点 0件")
        self.frame_label.setText("未解析")
        self.time_label.setText("00:00.000 / 00:00.000")
        self.seek_slider.setRange(0, 0)

    def _choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "処理する動画を選択", "", "動画 (*.mp4 *.mov *.mkv *.webm *.avi *.m4v)")
        if path:
            self.load_video(Path(path))

    def load_video(self, path: Path):
        if self.busy:
            return
        def load(cancel, progress):
            info = probe_video(path, cancel)
            images = list(decode_frames(info, cancel, first_only=True))
            if not images:
                raise ValueError("動画からフレームを読み込めませんでした。")
            return info, images[0]
        self._start_job("load", load)

    def _analyze(self):
        if self.info is None or self.busy:
            return
        try:
            settings = self._settings()
        except ValueError as error:
            self.status.setText(str(error))
            return
        info = self.info
        self._start_job("analyze", lambda cancel, progress: analyze_video(info, settings, self.model_dir, cancel, progress))

    def _export(self):
        if self.analysis is None or self.busy:
            return
        proposed = self.output_dir / f"{self.info.path.stem}_mosaic.mp4"
        sequence = 2
        while proposed.exists():
            proposed = self.output_dir / f"{self.info.path.stem}_mosaic_{sequence}.mp4"
            sequence += 1
        path, _ = QFileDialog.getSaveFileName(self, "動画を書き出す", str(proposed), "MP4 (*.mp4)",
                                             options=QFileDialog.Option.DontConfirmOverwrite)
        if path:
            self.export_to(Path(path))

    def export_to(self, path: Path):
        if self.analysis is None or self.busy:
            return
        if not path.suffix:
            path = path.with_suffix(".mp4")
        analysis, effect, size = self.analysis, self._effect_type(), self.effect_size.value()
        self._start_job("export", lambda cancel, progress: export_video(analysis, path, effect, size, cancel, progress))

    def _start_job(self, kind, action):
        self._stop_play()
        self.busy, self.job_kind = True, kind
        self.cancel = threading.Event()
        self.progress.setValue(0)
        self.status.setText({"load": "動画を読み込んでいます…", "analyze": "動画を解析しています…", "export": "動画を書き出しています…"}[kind])
        self._update_controls()
        def work():
            last_update = 0.0
            def progress(done, total, elapsed):
                nonlocal last_update
                now = time.monotonic()
                if now - last_update >= 0.15 or done == total:
                    self.events.put(("progress", (done, total, elapsed)))
                    last_update = now
            try:
                result = action(self.cancel, progress)
                self.events.put(("success", (kind, result)))
            except VideoCancelled:
                self.events.put(("cancelled", None))
            except Exception as error:
                self.events.put(("error", str(error)))
        self.worker = threading.Thread(target=work, daemon=False)
        self.worker.start()

    def _poll(self):
        while True:
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "progress":
                done, total, elapsed = payload
                self.progress.setValue(min(99, round(done / max(1, total) * 100)))
                speed = done / max(0.01, elapsed)
                remaining = max(0, total - done) / max(0.01, speed)
                verb = "解析" if self.job_kind == "analyze" else "書き出し"
                self.status.setText(f"{verb}中: {done:,} / 約{total:,}フレーム　{speed:.1f} fps　残り約{time_text(remaining)}")
                continue
            self.busy = False
            if event == "success":
                kind, result = payload
                if kind == "load":
                    self._discard_analysis()
                    self.info, image = result
                    self.filename.setText(self.info.path.name)
                    self.filename.setToolTip(str(self.info.path))
                    self.info_label.setText(f"{self.info.width} × {self.info.height}　{float(self.info.fps):.3f} fps　{time_text(self.info.duration)}")
                    self.preview.set_image(image)
                    self.current_frame = 0
                    self.time_label.setText(f"00:00.000 / {time_text(self.info.duration)}")
                    self.frame_label.setText("フレーム 1 / 未解析")
                    self.status.setText("動画を解析すると、マスク数と変化点を確認できます。")
                    self.summary_label.setText("未解析")
                elif kind == "analyze":
                    self._set_analysis(result)
                    self.status.setText("解析完了。タイムラインと検出範囲を確認してから書き出してください。")
                else:
                    self.last_output = result
                    self.output_dir = result.parent
                    self.status.setText(f"書き出しました: {result}")
                self.progress.setValue(100)
            elif event == "cancelled":
                self.status.setText("処理を中止しました。")
                self.progress.setValue(0)
            else:
                self.status.setText(f"処理できませんでした: {payload}")
                self.progress.setValue(0)
            self._update_controls()
            if self.closing:
                QTimer.singleShot(0, self.close)

    def _set_analysis(self, analysis):
        self._discard_analysis()
        self.analysis, self.info = analysis, analysis.info
        self.timeline.set_frames(analysis.frames, float(analysis.info.fps))
        self.timeline.set_window_seconds(self.zoom.currentData())
        changes = analysis.changes
        self.change_frames = [change.frame for change in changes]
        self.change_model.set_changes(changes, float(analysis.info.fps))
        self.change_label.setText(f"変化点 {len(changes):,}件")
        self.seek_slider.setRange(0, len(analysis.frames) - 1)
        count = len(analysis.frames)
        gaps = sum(frame.count == 0 for frame in analysis.frames)
        reused = sum(frame.reused for frame in analysis.frames)
        self.summary_label.setText(f"解析済み {count:,}フレーム\n0件: {gaps:,}フレーム　輪郭再利用: {reused:,}件\n解析時間: {time_text(analysis.elapsed)}")
        self.seek(0)

    def seek(self, frame: int):
        if self.analysis is None:
            return
        self.current_frame = max(0, min(len(self.analysis.frames) - 1, frame))
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(self.current_frame)
        self.seek_slider.blockSignals(False)
        self.timeline.set_current(self.current_frame)
        try:
            self.preview_data = self.analysis.cache.preview(self.current_frame, self.info)
        except Exception as error:
            self._stop_play()
            self.status.setText(str(error))
            return
        summary = self.analysis.frames[self.current_frame]
        self.time_label.setText(f"{time_text(self.current_frame / float(self.info.fps))} / {time_text(self.analysis.duration)}")
        detail = f"フレーム {self.current_frame + 1:,} / {len(self.analysis.frames):,}　マスク {summary.count}件"
        if summary.reused:
            detail += f"　輪郭再利用 {summary.reused}件"
        if summary.fallbacks:
            detail += f"　矩形代替 {summary.fallbacks}件"
        self.frame_label.setText(detail)
        self._render_preview()

    def _user_seek(self, frame: int):
        self._stop_play()
        self.seek(frame)

    def _render_preview(self):
        if self.preview_data is None:
            return
        image, mask, detections = self.preview_data
        mode = self.view_mode.currentText()
        if mode == "検出範囲":
            image = visualize_detection(image, mask, detections)
        elif mode == "処理結果":
            scale = image.shape[1] / self.info.width
            image = apply_effect(image, mask, self._effect_type(), max(1, round(self.effect_size.value() * scale)))
        self.preview.set_image(image)

    def _jump_change(self, direction):
        index = bisect.bisect_left(self.change_frames, self.current_frame) - 1 if direction < 0 else bisect.bisect_right(self.change_frames, self.current_frame)
        if 0 <= index < len(self.change_frames):
            self._stop_play()
            self.change_table.selectRow(index)
            self.seek(self.change_frames[index])

    def _select_change(self, index, previous):
        if index.isValid():
            self._stop_play()
            self.seek(self.change_model.changes[index.row()].frame)

    def _toggle_play(self):
        if self.play_timer.isActive():
            self._stop_play()
        elif self.analysis is not None:
            if self.current_frame == len(self.analysis.frames) - 1:
                self.seek(0)
            self.play_start = time.monotonic() - self.current_frame / float(self.info.fps)
            self.play_button.setText("一時停止")
            self.play_timer.start(max(15, round(1000 / float(self.info.fps))))

    def _stop_play(self):
        if hasattr(self, "play_timer"):
            self.play_timer.stop()
            self.play_button.setText("再生")

    def _play_tick(self):
        if self.analysis is None:
            self._stop_play()
            return
        frame = int((time.monotonic() - self.play_start) * float(self.info.fps))
        if frame >= len(self.analysis.frames):
            self._stop_play()
            frame = len(self.analysis.frames) - 1
        self.seek(frame)

    def _update_controls(self):
        ready = self.analysis is not None and self.analysis.complete and not self.busy
        self.open_button.setEnabled(not self.busy)
        for control in self.analysis_controls:
            control.setEnabled(not self.busy)
        self.analyze_button.setEnabled(self.info is not None and not self.busy)
        self.export_button.setEnabled(ready)
        self.cancel_button.setEnabled(self.busy and not self.cancel.is_set())
        for control in (self.play_button, self.back_button, self.forward_button, self.seek_slider,
                        self.view_mode, self.timeline, self.zoom, self.change_table,
                        self.previous_change, self.next_change):
            control.setEnabled(ready)
        self.effect.setEnabled(not self.busy)
        self.effect_size.setEnabled(not self.busy)
        self.output_button.setEnabled(self.last_output is not None)

    def _cancel(self):
        self.cancel.set()
        self.cancel_button.setEnabled(False)
        self.status.setText("中止しています…")

    def _open_output(self):
        if self.last_output is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output.parent)))

    def dragEnterEvent(self, event):
        if not self.busy and any(url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in VIDEO_EXTENSIONS
                                 for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if self.busy:
            return
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in VIDEO_EXTENSIONS:
                self.load_video(Path(url.toLocalFile()))
                event.acceptProposedAction()
                return

    def closeEvent(self, event):
        self._stop_play()
        if self.busy:
            self.closing = True
            self._cancel()
            event.ignore()
            return
        self._discard_analysis()
        self.info = None
        self.filename.setText("動画未選択")
        self.info_label.setText("")
        self.summary_label.setText("解析後にタイムラインを確認できます。")
        self.status.setText("動画を開いてください。")
        self.preview.set_image(None)
        self.progress.setValue(0)
        self._update_controls()
        self.closing = False
        super().closeEvent(event)
