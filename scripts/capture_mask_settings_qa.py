"""Capture native Qt windows using synthetic segmentation; no private media."""
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea

from auto_mosaic.ui import AutoMosaicWindow
from auto_mosaic.video_ui import VideoMosaicWindow
from tests.test_mask_smoothing import Detector, Segmenter


def main():
    app = QApplication.instance() or QApplication([])
    destination = Path(".codex-qa/2026-11-07/inline-settings").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = destination / "contour-fixture.png"
    yy, xx = np.indices((240, 320))
    image = np.dstack((70 + xx % 100, 80 + yy % 100, 110 + (xx + yy) % 100)).astype(np.uint8)
    Image.fromarray(image).save(source)
    window = AutoMosaicWindow()
    window.move(20, 30)
    window.show()

    def pump():
        for _ in range(8):
            app.processEvents()
            sleep(.015)

    def wait():
        deadline = monotonic() + 10
        while monotonic() < deadline:
            pump()
            if not window.busy and window.current_result is not None:
                return
        raise AssertionError("analysis timeout")

    with patch.object(window.pipeline, "_get_detector", return_value=Detector()), patch.object(
            window.pipeline, "_get_segmenter", return_value=Segmenter()):
        window.mask_expansion_spin.setValue(0)
        window._add_image_paths([source])
        wait()
        window.set_preview_mode("マスク範囲")
        pump()
        window.grab().save(str(destination / "image-expanded.png"))
        QTest.mouseClick(window.advanced_button, Qt.MouseButton.LeftButton)
        pump()
        assert not window.advanced_settings.content.isVisible()
        assert window.suffix_edit.isVisible()
        assert app.activeModalWidget() is None
        QTest.mouseClick(window.preview_mode_buttons["処理結果"], Qt.MouseButton.LeftButton)
        assert window.preview_mode() == "処理結果"
        window.set_preview_mode("マスク範囲")
        pump()
        window.grab().save(str(destination / "image-collapsed.png"))
        window.resize(980, 650)
        pump()
        window.grab().save(str(destination / "image-collapsed-minimum.png"))
        scroll = window.suffix_edit.parentWidget()
        while not isinstance(scroll, QScrollArea):
            scroll = scroll.parentWidget()
        scroll.ensureWidgetVisible(window.suffix_edit)
        pump()
        window.grab().save(str(destination / "image-suffix-minimum.png"))
        scroll.verticalScrollBar().setValue(0)
        window.advanced_button.click()
        pump()
        window.grab().save(str(destination / "image-expanded-minimum.png"))
        window.close()

    video = VideoMosaicWindow()
    video.show()
    pump()
    video.grab().save(str(destination / "video-expanded.png"))
    video.advanced_button.click()
    pump()
    video.grab().save(str(destination / "video-collapsed.png"))
    video.resize(1060, 760)
    pump()
    video.grab().save(str(destination / "video-collapsed-minimum.png"))
    video.advanced_button.click()
    pump()
    video.grab().save(str(destination / "video-expanded-minimum.png"))
    video.close()
    print(f"QA_PLATFORM={app.platformName()} CAPTURES={destination}")


if __name__ == "__main__":
    main()
