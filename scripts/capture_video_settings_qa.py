"""Capture matched Qt UI states and the actual UI export path using a benign fixture."""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from threading import Event

import cv2
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QScrollArea

from auto_mosaic.ui import AutoMosaicWindow
from auto_mosaic.video_analysis import VideoFrameAnalyzer, analyze_video
from auto_mosaic.video_io import decode_frames, probe_video
from auto_mosaic.video_ui import VideoMosaicWindow
from tests.test_video import settings
from tests.test_video_ui import wait_job
from tests.video_fixtures import ShapeDetector, ShapeSegmenter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("before", "after"))
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    destination = Path(".codex-qa/2026-09-20/video-settings")
    destination.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/meiryo.ttc")
    still = AutoMosaicWindow()
    still.show()
    app.processEvents()
    still.grab().save(str(destination / f"still-{args.stage}.png"))
    still.close()

    path = Path(".codex-qa/2026-09-20/timeline-fixture.mp4").resolve()
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
    window.grab().save(str(destination / f"video-{args.stage}.png"))
    scroll = window.findChild(QScrollArea)
    scroll.grab().save(str(destination / f"settings-{args.stage}.png"))
    geometry = {name: getattr(window, name).geometry().getRect() for name in ("preview", "timeline")}
    print("GEOMETRY", args.stage, json.dumps(geometry))
    window.view_mode.setCurrentText("処理結果")
    app.processEvents()
    window.grab().save(str(destination / f"result-{args.stage}.png"))
    if args.stage == "after" and not args.skip_export:
        for index, name in enumerate(("mosaic", "blur")):
            window.effect.setCurrentIndex(index)
            output = destination / f"ui-{name}.mp4"
            window.export_to(output.resolve())
            wait_job(app, window)
            assert window.last_output == output.resolve(), window.status.text()
            encoded = list(decode_frames(probe_video(output, Event()), Event()))
            cv2.imwrite(str(destination / f"encoded-{name}.png"), encoded[15])
        window.effect.setCurrentIndex(0)
    window.resize(1060, 760)
    app.processEvents()
    window.grab().save(str(destination / f"minimum-{args.stage}.png"))
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    app.processEvents()
    window.grab().save(str(destination / f"minimum-bottom-{args.stage}.png"))
    if args.stage == "after":
        window.resize(1340, 940)
        scroll.verticalScrollBar().setValue(0)
        window.busy = True
        window._update_controls()
        window.status.setText("動画を書き出しています…")
        app.processEvents()
        window.grab().save(str(destination / "video-busy.png"))
        window.busy = False
        window._update_controls()
        window.penis_check.setChecked(False)
        window.effect.setFocus()
        app.processEvents()
        window.grab().save(str(destination / "video-settings-changed.png"))
    window.close()
    print("CAPTURES", destination.resolve())


if __name__ == "__main__":
    main()
