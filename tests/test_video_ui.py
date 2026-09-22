from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from auto_mosaic.domain import EffectType, ImageMode
from auto_mosaic.image_ops import apply_effect
from auto_mosaic.model_catalog import detector_for, PHOTO_DETECTOR, ILLUSTRATION_DETECTOR
from auto_mosaic.ui import AutoMosaicWindow
from auto_mosaic.video_analysis import VideoFrameAnalyzer, analyze_video
from auto_mosaic.video_io import decode_frames, probe_video
from auto_mosaic.video_ui import VideoMosaicWindow
from tests.test_video import settings
from tests.video_fixtures import ShapeDetector, ShapeSegmenter, make_video


def wait_job(app, window):
    deadline = time.monotonic() + 15
    while window.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert not window.busy, window.status.text()


class VideoUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/meiryo.ttc")

    def test_timeline_review_export_gate_and_setting_invalidation(self):
        with TemporaryDirectory() as temporary:
            source = make_video(Path(temporary) / "review.mp4")
            window = VideoMosaicWindow()
            window.show()
            self.app.processEvents()
            try:
                self.assertFalse(window.export_button.isEnabled())
                window.load_video(source)
                wait_job(self.app, window)
                self.assertIsNotNone(window.info, window.status.text())
                def synthetic_analyze(info, config, models, cancel, progress):
                    engine = VideoFrameAnalyzer(models, config, detector=ShapeDetector(), segmenter=ShapeSegmenter())
                    return analyze_video(info, config, models, cancel, progress, analyzer=engine)
                with patch("auto_mosaic.video_ui.analyze_video", synthetic_analyze):
                    window.analyze_button.click()
                    wait_job(self.app, window)
                self.assertTrue(window.export_button.isEnabled(), window.status.text())
                self.assertEqual(window.change_model.rowCount(), 6)
                window.seek(23)
                window.next_change.click()
                self.assertEqual(window.current_frame, 24)
                self.assertIn("マスク 0件", window.frame_label.text())
                window.forward_button.click()
                self.assertEqual(window.current_frame, 25)
                self.assertIn("マスク 2件", window.frame_label.text())
                rect = window.timeline.plot_rect()
                x = rect.left() + (15.5 / 48) * rect.width()
                QTest.mouseClick(window.timeline, Qt.MouseButton.LeftButton, pos=QPoint(round(x), round(rect.center().y())))
                self.assertEqual(window.current_frame, 15)
                window.play_button.click()
                self.assertTrue(window.play_timer.isActive())
                window.seek_slider.setValue(18)
                self.assertFalse(window.play_timer.isActive())
                self.assertEqual(window.current_frame, 18)
                old_result = window.analysis
                window.effect_size.setValue(28)
                self.assertIs(window.analysis, old_result)
                self.assertTrue(window.export_button.isEnabled())
                window.threshold_slider.setValue(40)
                self.assertIsNone(window.analysis)
                self.assertFalse(window.export_button.isEnabled())
                self.assertEqual(window.change_model.rowCount(), 0)
                self.assertEqual(window.view_mode.currentText(), "元画像")
                self.assertEqual(window.progress.value(), 0)
                self.assertTrue(window.time_label.text().startswith("00:01.500"))
            finally:
                window.close()

    def test_separate_window_preserves_still_image_state(self):
        still = AutoMosaicWindow()
        still.image_paths = [Path("keep.png")]
        mask = np.zeros((8, 8), bool)
        still.edited_mask = mask
        still.preview_zoom_index = 3
        still._open_video_mode()
        self.app.processEvents()
        try:
            self.assertIsInstance(still.video_window, VideoMosaicWindow)
            self.assertEqual(still.image_paths, [Path("keep.png")])
            self.assertIs(still.edited_mask, mask)
            self.assertEqual(still.preview_zoom_index, 3)
            still.video_window.close()
            self.assertEqual(still.image_paths, [Path("keep.png")])
        finally:
            still.close()

    def test_mode_combo_preserves_enum_and_selects_matching_detector(self):
        window = VideoMosaicWindow()
        try:
            for index, mode, spec in ((0, ImageMode.PHOTO, PHOTO_DETECTOR),
                                      (1, ImageMode.ILLUSTRATION, ILLUSTRATION_DETECTOR)):
                with self.subTest(mode=mode):
                    window.mode.setCurrentIndex(index)
                    selected = window._settings().processing.mode
                    self.assertIs(selected, mode)
                    self.assertIs(detector_for(selected), spec)
        finally:
            window.close()

    def test_effect_combo_drives_preview_and_encoded_export(self):
        # Exercise Qt's real string-enum round trip, not a direct backend enum.
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = make_video(directory / "effects.mp4", audio=False, frames=18)
            info = probe_video(source, Event())
            config = settings(3)
            engine = VideoFrameAnalyzer(Path("models"), config, detector=ShapeDetector(), segmenter=ShapeSegmenter())
            result = analyze_video(info, config, Path("models"), Event(), lambda *a: None, analyzer=engine)
            window = VideoMosaicWindow()
            try:
                window._set_analysis(result)
                window.seek(15)
                window.view_mode.setCurrentText("処理結果")
                source_frame = list(decode_frames(info, Event()))[15]
                preview_frame, mask, _ = window.preview_data
                self.assertTrue(mask.any())
                for index, effect in enumerate((EffectType.MOSAIC, EffectType.BLUR)):
                    with self.subTest(effect=effect):
                        window.effect.setCurrentIndex(index)
                        qimage = window.preview.image
                        pixels = np.frombuffer(qimage.bits(), np.uint8).reshape(qimage.height(), qimage.bytesPerLine())
                        actual_preview = pixels[:, :qimage.width() * 3].reshape(qimage.height(), qimage.width(), 3)[:, :, ::-1]
                        np.testing.assert_array_equal(actual_preview, apply_effect(preview_frame, mask, effect, 16))
                        output = directory / f"{effect.value}.mp4"
                        window.export_to(output)
                        wait_job(self.app, window)
                        self.assertEqual(window.last_output, output, window.status.text())
                        encoded = list(decode_frames(probe_video(output, Event()), Event()))[15].astype(float)
                        expected = apply_effect(source_frame, mask, effect, 16).astype(float)
                        other = EffectType.BLUR if effect is EffectType.MOSAIC else EffectType.MOSAIC
                        wrong = apply_effect(source_frame, mask, other, 16).astype(float)
                        error = np.square(encoded[mask] - expected[mask]).mean()
                        wrong_error = np.square(encoded[mask] - wrong[mask]).mean()
                        self.assertLess(error, wrong_error / 3, "encoded output must match the selected effect")
                        self.assertLess(np.abs(encoded[~mask] - source_frame[~mask]).mean(), 3)
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
