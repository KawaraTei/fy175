"""Capture actual macOS Qt windows and platform-specific paths/shortcuts for review."""
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def main():
    if sys.platform != "darwin":
        raise SystemExit("Run this capture on macOS.")
    os.environ["QT_QPA_PLATFORM"] = "cocoa"
    from scripts.capture_layer_qa import main as capture_layers
    from scripts.capture_clip_geometry_qa import main as capture_coordinates
    from auto_mosaic.platform_support import default_output_dir, command_key
    from auto_mosaic.video_io import tool_path
    from auto_mosaic.ui import AutoMosaicWindow
    from auto_mosaic.video_ui import VideoMosaicWindow
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QKeySequence

    capture_layers()
    capture_coordinates()
    destination = Path(".codex-qa/macos")
    destination.mkdir(parents=True, exist_ok=True)
    coordinate_destination = destination / "coordinates"
    coordinate_destination.mkdir(exist_ok=True)
    for source in Path(".codex-qa/layers/coordinates").iterdir():
        if source.suffix in {".png", ".json"}:
            shutil.copyfile(source, coordinate_destination / source.name)
    for source in Path(".codex-qa/layers").glob("*.png"):
        shutil.copyfile(source, destination / source.name)
    app = QApplication.instance()
    image_window = AutoMosaicWindow()
    video_window = VideoMosaicWindow()
    report = {"macos": platform.mac_ver()[0], "architecture": platform.machine(),
              "qt_platform": app.platformName(), "output": str(default_output_dir()),
              "command_key": command_key(), "ffmpeg": str(tool_path("ffmpeg")),
              "undo": image_window.mask_undo_shortcut.key().toString(QKeySequence.SequenceFormat.NativeText),
              "save": video_window.project_save_shortcut.key().toString(QKeySequence.SequenceFormat.NativeText)}
    assert report["qt_platform"] == "cocoa"
    assert ".app/Contents" not in report["output"]
    for name, window, size in (("image", image_window, (1260, 790)), ("video", video_window, (1340, 940))):
        window.resize(*size)
        window.show()
        app.processEvents()
        window.grab().save(str(destination / f"{name}-main.png"))
        window.advanced_settings.show_settings()
        app.processEvents()
        window.advanced_settings.grab().save(str(destination / f"{name}-settings.png"))
        window.advanced_settings.close()
        window.close()
    (destination / "platform.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("MACOS_QA_OK", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
