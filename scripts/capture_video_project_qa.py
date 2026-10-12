"""Exercise project save/reopen in the real Qt widgets using synthetic video."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QPointF
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from auto_mosaic.video_ui import VideoMosaicWindow
from tests.test_video_mask_edit import make_edit_analysis
from tests.test_video_ui import wait_job


def main():
    destination = Path(".codex-qa/video-project").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/meiryo.ttc")
    window = VideoMosaicWindow()
    window.show()
    window._set_analysis(make_edit_analysis(destination))
    window.filename.setText(window.info.path.name)
    window.info_label.setText("128 × 96　12.000 fps　00:00.667")
    window._update_controls()
    window.seek(1)
    window.settings_tabs.setCurrentIndex(1)
    app.processEvents()
    window.mask_editor.edit_button.click()
    rect = window.preview.image_rect
    point = QPointF(rect.left() + 35 / 128 * rect.width(), rect.top() + 35 / 96 * rect.height())
    window.mask_editor.paint(point, point, False)
    window.mask_editor.apply_button.click()
    wait_job(app, window)
    window.seek(5)
    window.mask_editor.end_button.click()
    window.mask_editor.propagate_button.click()
    wait_job(app, window)
    window.effect_size.setValue(28)
    window.view_mode.setCurrentText("処理結果")
    project = destination / "moving-texture.mp4.fyvideo"
    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(project), "")):
        window.project_save_button.click()
        wait_job(app, window)
    assert window.project_path == project, window.status.text()
    window.close()
    window.show()
    app.processEvents()
    with patch.object(QFileDialog, "getOpenFileName", return_value=(str(project), "")):
        window.project_open_button.click()
        wait_job(app, window)
    assert window.current_frame == 5 and window.mask_editor.edits.get(5) == "propagated", window.status.text()
    geometry = {}
    for name, width, height in (("resumed", 1340, 940), ("resumed-minimum", 1060, 760)):
        window.resize(width, height)
        app.processEvents()
        window.grab().save(str(destination / f"{name}.png"))
        geometry[name] = {"preview": window.preview.geometry().getRect(), "timeline": window.timeline.geometry().getRect()}
    window.resize(1340, 940)
    app.processEvents()
    window.mask_editor.edit_button.click()
    window.mask_editor.paint(point, point, False)
    window.save_project_to(project)
    wait_job(app, window)
    window.close()
    window.show()
    app.processEvents()
    window.load_project_from(project)
    wait_job(app, window)
    assert window.mask_editor.active
    window.grab().save(str(destination / "resumed-draft.png"))
    (destination / "geometry.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard):
        window.close()
    print("VIDEO_PROJECT_QA_OK", destination)
    print(json.dumps(geometry))


if __name__ == "__main__":
    main()
