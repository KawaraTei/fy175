"""Render the real Qt windows using synthetic detection data; not model accuracy QA."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from threading import Event

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from auto_mosaic.ui import AutoMosaicWindow
from auto_mosaic.video_analysis import VideoFrameAnalyzer, analyze_video
from auto_mosaic.video_io import probe_video
from auto_mosaic.video_ui import VideoMosaicWindow
from tests.test_video import settings
from tests.video_fixtures import ShapeDetector, ShapeSegmenter, make_video


def main():
    destination = Path(".codex-qa/2026-09-20")
    destination.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/meiryo.ttc")
    still = AutoMosaicWindow()
    still.show()
    app.processEvents()
    still.grab().save(str(destination / "still-after.png"))
    still.centralWidget().grab().save(str(destination / "still-after-content.png"))
    still.close()

    path = destination / "timeline-fixture.mp4"
    make_video(path)
    info = probe_video(path, Event())
    config = settings(3)
    analyzer = VideoFrameAnalyzer(Path("models"), config, detector=ShapeDetector(), segmenter=ShapeSegmenter())
    analysis = analyze_video(info, config, Path("models"), Event(), lambda *a: None, analyzer=analyzer)
    window = VideoMosaicWindow()
    window.interval.setCurrentIndex(1)
    window.show()
    window.filename.setText(path.name)
    window.info_label.setText("640 × 360　12.000 fps　00:04.000")
    window._set_analysis(analysis)
    window.status.setText("解析完了。タイムラインと検出範囲を確認してから書き出してください。")
    window._update_controls()
    window.progress.setValue(100)
    window.seek(15)
    app.processEvents()
    window.grab().save(str(destination / "video-review.png"))
    window.seek(24)
    app.processEvents()
    window.grab().save(str(destination / "video-gap.png"))
    window.zoom.setCurrentIndex(2)
    window.seek(25)
    window.view_mode.setCurrentText("処理結果")
    app.processEvents()
    window.grab().save(str(destination / "video-detail.png"))
    window.resize(1060, 760)
    app.processEvents()
    window.grab().save(str(destination / "video-minimum.png"))
    window.resize(1340, 940)
    window.threshold_slider.setValue(35)
    app.processEvents()
    window.grab().save(str(destination / "video-needs-analysis.png"))
    window.close()
    print("VIDEO_QA_CAPTURES", destination.resolve())


if __name__ == "__main__":
    main()
