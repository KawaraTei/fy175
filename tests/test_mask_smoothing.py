"""Focused contour and modeless settings checks with a deterministic mask."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from auto_mosaic.domain import Detection, ImageMode, ProcessingSettings
from auto_mosaic.image_ops import refine_mask
from auto_mosaic.pipeline import MosaicPipeline
from auto_mosaic.ui import AutoMosaicWindow
from auto_mosaic.video_analysis import VideoAnalysisSettings, VideoFrameAnalyzer
from auto_mosaic.video_ui import VideoMosaicWindow


def jagged_mask():
    mask = np.zeros((240, 320), np.uint8)
    angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    radii = np.where(np.arange(120) % 2, 64, 73)
    points = np.c_[160 + np.cos(angles) * radii, 120 + np.sin(angles) * radii].astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


class Detector:
    def detect(self, *_args):
        return [Detection("penis", .9, (80, 40, 240, 200))]


class Segmenter:
    def encode(self, _image):
        return None

    def mask_candidates_from_box(self, *_args):
        return [(jagged_mask(), .9)]


class SmoothingTests(unittest.TestCase):
    def test_rounding_reduces_contour_length_and_keeps_tiny_masks(self):
        original = jagged_mask()
        unchanged = original.copy()
        baseline = refine_mask(original, 0)
        rounded = refine_mask(original, 0, 10)
        def perimeter(mask):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            return sum(cv2.arcLength(c, True) for c in contours)
        self.assertLess(perimeter(rounded), perimeter(baseline) * .92)
        self.assertGreater(rounded.sum(), baseline.sum() * .95)
        np.testing.assert_array_equal(original, unchanged)
        np.testing.assert_array_equal(baseline, refine_mask(original, 0, 0))
        tiny = np.zeros((50, 50), bool)
        tiny[25, 25] = True
        self.assertTrue(refine_mask(tiny, 0, 20)[25, 25])
        self.assertFalse(refine_mask(np.zeros_like(tiny), 0, 20).any())

    def test_image_and_video_use_the_same_smoothing_and_manual_masks_stay_exact(self):
        config = ProcessingSettings(ImageMode.ILLUSTRATION, frozenset({"penis"}), .25,
                                    mask_expansion=0, mask_smoothing=10)
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "fixture.png"
            Image.new("RGB", (320, 240), (70, 100, 140)).save(source)
            pipeline = MosaicPipeline(Path("models"))
            with patch.object(pipeline, "_get_detector", return_value=Detector()), patch.object(
                    pipeline, "_get_segmenter", return_value=Segmenter()):
                still = pipeline.analyze(source, config)
                baseline = pipeline.analyze(source, replace(config, mask_smoothing=0))
            self.assertFalse(np.array_equal(still.mask, baseline.mask))
            video = VideoFrameAnalyzer(Path("models"), VideoAnalysisSettings(config, 1),
                                       detector=Detector(), segmenter=Segmenter())
            mask, _, _ = video.analyze(np.zeros((240, 320, 3), np.uint8), Event())
            np.testing.assert_array_equal(mask, still.mask)
            np.testing.assert_array_equal(still.detection_masks[0], still.mask)
            manual = pipeline.process_with_mask(source, jagged_mask(), config, [])
            np.testing.assert_array_equal(manual.mask, jagged_mask())

    def test_dialogs_allow_main_window_input_and_reuse_controls(self):
        app = QApplication.instance() or QApplication([])
        for window in (AutoMosaicWindow(), VideoMosaicWindow()):
            try:
                window.show()
                QTest.mouseClick(window.advanced_button, Qt.MouseButton.LeftButton)
                app.processEvents()
                dialog = window.advanced_settings
                self.assertTrue(dialog.isVisible())
                self.assertIsNone(app.activeModalWidget())
                self.assertIs(window.threshold_slider.window(), dialog)
                slider = getattr(window, "effect_size_slider", None)
                if slider is None:
                    slider = window.effect_size
                value = slider.value()
                QTest.keyClick(slider, Qt.Key.Key_Up)
                self.assertEqual(slider.value(), value + 1)
                window.threshold_slider.setValue(45)
                window.mask_smoothing_spin.setValue(7)
                dialog.close()
                window.advanced_button.click()
                self.assertIs(window.advanced_settings, dialog)
                self.assertEqual(window.threshold_slider.value(), 45)
                config = window._settings()
                config = getattr(config, "processing", config)
                self.assertEqual(config.mask_smoothing, 7)
                self.assertEqual(config.confidence_threshold, .45)
            finally:
                window.close()
        app.processEvents()
