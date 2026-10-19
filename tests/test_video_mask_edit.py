from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sqlite3
import unittest
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QMessageBox

from auto_mosaic.domain import Detection, EffectType
from auto_mosaic.image_ops import apply_effect
from auto_mosaic.video_analysis import FrameCache, FrameSummary, VideoAnalysis, export_video
from auto_mosaic.video_io import VideoCancelled, VideoInfo, decode_frames, encoder_arguments, media_process, probe_video
from auto_mosaic.video_mask_edit import clear_mask_edits, propagate_mask, save_frame_mask, undo_mask_edit
from auto_mosaic.video_project import EditorState, load_project, save_project
from auto_mosaic.video_ui import VideoMosaicWindow
from tests.test_video import settings
from tests.test_video_ui import wait_job


def make_edit_analysis(directory: Path, *, cut=False, gaps=()):
    source = directory / "moving-texture.mp4"
    info = VideoInfo(source, 128, 96, Fraction(12), 8 / 12, 8, False, 0, 0, 0)
    texture = np.random.default_rng(13).integers(30, 225, (96, 128, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    images = [cv2.warpAffine(texture, np.float32([[1, 0, i * 2], [0, 1, 0]]), (128, 96)) for i in range(8)]
    if cut:
        images[4] = np.full_like(texture, 255)
    with media_process(encoder_arguments(info, source), Event(), input_pipe=True) as encoder:
        for image in images:
            encoder.stdin.write(image.tobytes())
    info = probe_video(source, Event())
    result = VideoAnalysis(info, settings(), FrameCache(), complete=True)
    # One automatically detected target, separate from the hand-painted region.
    with closing(sqlite3.connect(result.cache.path)) as connection, connection:
        for index, image in enumerate(decode_frames(info, Event())):
            mask = np.zeros((96, 128), bool)
            mask[50:70, 88 + index * 2:106 + index * 2] = True
            detection = Detection("penis", .9, (88 + index * 2, 50, 106 + index * 2, 70))
            if index in gaps:
                mask[:] = False
            FrameCache.append(connection, index, image, mask, [] if index in gaps else [detection])
            result.frames.append(FrameSummary(0 if index in gaps else 1))
    return result


def manual_mask(result, frame):
    mask = result.cache.full_mask(frame, result.info)
    mask[28:46, 28 + frame * 2:46 + frame * 2] = True
    mask[55:65, 93 + frame * 2:101 + frame * 2] = False
    return mask


class VideoMaskTests(unittest.TestCase):
    def test_unedited_start_fills_gap_and_clear_restores_detection_with_undo(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = make_edit_analysis(directory, gaps=(2, 3, 4, 5))
            try:
                original = result.cache.full_mask(1, result.info)
                propagated = propagate_mask(result, 1, 5, Event(), lambda *args: None)
                self.assertEqual(propagated.edited, dict.fromkeys(range(2, 6), "propagated"))
                for frame in range(2, 6):
                    self.assertTrue(result.cache.full_mask(frame, result.info).any())
                np.testing.assert_array_equal(result.cache.full_mask(1, result.info), original)
                state = EditorState(frame=5, start=1)
                path = save_project(result, directory / "unedited-start.fyvideo", EffectType.MOSAIC,
                                    16, state, None, Event(), lambda *args: None)
                loaded = load_project(path, Event(), lambda *args: None)
                try:
                    self.assertEqual(loaded.state, state)
                finally:
                    loaded.analysis.close()
                clear_mask_edits(result, 3, 3, Event(), lambda *args: None)
                self.assertFalse(result.cache.full_mask(3, result.info).any())
                self.assertTrue(result.cache.full_mask(4, result.info).any())
                restored = undo_mask_edit(result, Event(), lambda *args: None)
                self.assertEqual(restored.edited, propagated.edited)
                clear_mask_edits(result, 2, 5, Event(), lambda *args: None)
                self.assertEqual(result.cache.edit_state()[0], {})
                for frame in range(2, 6):
                    self.assertFalse(result.cache.full_mask(frame, result.info).any())
                np.testing.assert_array_equal(result.cache.full_mask(1, result.info), original)
            finally:
                result.close()

    def test_follow_range_erasures_export_and_undo(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = make_edit_analysis(directory)
            try:
                originals = [result.cache.full_mask(i, result.info) for i in range(8)]
                source_mask = manual_mask(result, 1)
                save_frame_mask(result, 1, source_mask, Event(), lambda *args: None)
                edit = propagate_mask(result, 1, 5, Event(), lambda *args: None)
                self.assertEqual(edit.edited, {1: "manual", 2: "propagated", 3: "propagated", 4: "propagated", 5: "propagated"})
                actual = result.cache.full_mask(5, result.info)
                expected = cv2.warpAffine(source_mask.astype(np.uint8), np.float32([[1, 0, 8], [0, 1, 0]]), (128, 96)) > 0
                self.assertGreater((actual & expected).sum() / (actual | expected).sum(), .9)
                self.assertFalse(actual[60, 110], "a hand-erased region must follow the motion too")
                for frame in (0, 6, 7):
                    np.testing.assert_array_equal(result.cache.full_mask(frame, result.info), originals[frame])
                self.assertEqual([frame.count for frame in result.frames], [1] * 8, "detection history is not edited-mask count")
                output = export_video(result, directory / "edited.mp4", EffectType.MOSAIC, 16, Event(), lambda *args: None)
                source_frame = list(decode_frames(result.info, Event()))[5]
                encoded_frame = list(decode_frames(probe_video(output, Event()), Event()))[5]
                expected_image = apply_effect(source_frame, actual, EffectType.MOSAIC, 16)
                self.assertLess(np.abs(expected_image.astype(float) - encoded_frame).mean(), 3)
                edit = undo_mask_edit(result, Event(), lambda *args: None)
                self.assertEqual(edit.edited, {1: "manual"})
                for frame in range(2, 6):
                    np.testing.assert_array_equal(result.cache.full_mask(frame, result.info), originals[frame])
                # An empty manual override must not fall back to automatic masks.
                save_frame_mask(result, 1, np.zeros_like(source_mask), Event(), lambda *args: None)
                self.assertFalse(result.cache.full_mask(1, result.info).any())
                self.assertFalse(result.cache.preview(1, result.info)[1].any())
            finally:
                result.close()

    def test_cancel_cut_and_manual_keyframe_leave_cache_unchanged(self):
        for cut in (False, True):
            with self.subTest(cut=cut), TemporaryDirectory() as temporary:
                result = make_edit_analysis(Path(temporary), cut=cut)
                try:
                    save_frame_mask(result, 1, manual_mask(result, 1), Event(), lambda *args: None)
                    cancel = Event()
                    def progress(*args):
                        if not cut:
                            cancel.set()
                    with self.assertRaises(ValueError if cut else VideoCancelled):
                        propagate_mask(result, 1, 6, cancel, progress)
                    self.assertEqual(result.cache.edit_state(), ({1: "manual"}, True))
                    undo_mask_edit(result, Event(), lambda *args: None)
                    self.assertEqual(result.cache.edit_state(), ({}, False), "failed propagation must preserve the previous undo")
                    save_frame_mask(result, 1, manual_mask(result, 1), Event(), lambda *args: None)
                    save_frame_mask(result, 3, manual_mask(result, 3), Event(), lambda *args: None)
                    with self.assertRaisesRegex(ValueError, "手動編集済み"):
                        propagate_mask(result, 1, 5, Event(), lambda *args: None)
                    self.assertEqual(result.cache.edit_state()[0], {1: "manual", 3: "manual"})
                finally:
                    result.close()


class VideoMaskUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/meiryo.ttc")

    def test_arrows_from_controls_and_editing_keep_masks_and_settings(self):
        with TemporaryDirectory() as temporary:
            result = make_edit_analysis(Path(temporary))
            window = VideoMosaicWindow()
            window.show()
            window._set_analysis(result)
            window._update_controls()
            editor = window.mask_editor
            try:
                window.seek(1)
                original_settings = window._config_values()
                for control in (window.preview, window.timeline, window.seek_slider,
                                window.effect_size, window.effect, editor.start_button, window.change_table):
                    control.setFocus()
                    QTest.keyClick(control, Qt.Key.Key_Right)
                    self.assertEqual(window.current_frame, 2, type(control).__name__)
                    QTest.keyClick(control, Qt.Key.Key_Left)
                    self.assertEqual(window.current_frame, 1, type(control).__name__)
                self.assertEqual(window._config_values(), original_settings)
                editor.begin_edit()
                editor.draft[20:30, 20:30] = True
                editor.dirty = True
                expected = editor.draft.copy()
                QTest.keyClick(editor.brush, Qt.Key.Key_Right)
                wait_job(self.app, window)
                self.assertEqual(window.current_frame, 2)
                self.assertTrue(editor.active)
                self.assertFalse(editor.dirty)
                np.testing.assert_array_equal(result.cache.full_mask(1, result.info), expected)
                QTest.keyClick(editor.brush, Qt.Key.Key_Left)
                self.assertEqual(window.current_frame, 1)
                np.testing.assert_array_equal(editor.draft, expected)
                self.assertEqual(editor.brush.value(), 40)
                dialog = QDialog(window)
                field = QLineEdit('abc', dialog)
                dialog.setModal(True)
                dialog.show()
                field.setFocus()
                self.app.processEvents()
                QTest.keyClick(field, Qt.Key.Key_Left)
                self.assertEqual(window.current_frame, 1)
                dialog.close()
            finally:
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard):
                    window.close()

    def test_selected_range_keyframe_clear_and_project_reopen(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = make_edit_analysis(directory, gaps=(2, 4, 5))
            save_frame_mask(result, 3, manual_mask(result, 3), Event(), lambda *args: None)
            window = VideoMosaicWindow()
            window.show()
            window._set_analysis(result)
            window._update_controls()
            editor = window.mask_editor
            try:
                window.seek(1)
                editor.start_button.click()
                self.assertEqual(editor.start, 1)
                self.assertNotIn(1, editor.edits)
                window.seek(0)
                self.assertIsNone(window.timeline.propagation_range)
                self.assertFalse(editor.propagate_button.isEnabled())
                window.seek(5)
                self.assertEqual(window.timeline.propagation_start, 1)
                self.assertEqual(window.timeline.propagation_range, (2, 5))
                self.assertFalse(editor.propagate_button.isEnabled(), "manual frame in range is protected")
                self.assertIn("フレーム 4", editor.range_label.text())
                window.seek(3)
                editor.clear_frame_button.click()
                wait_job(self.app, window)
                self.assertNotIn(3, editor.edits)
                self.assertEqual(editor.start, 1)
                window.seek(5)
                self.assertTrue(editor.propagate_button.isEnabled())
                editor.propagate_button.click()
                wait_job(self.app, window)
                self.assertEqual(editor.edits, dict.fromkeys(range(2, 6), "propagated"))
                path = directory / "selection.fyvideo"
                window.save_project_to(path)
                wait_job(self.app, window)
                window.load_project_from(path, confirmed=True)
                wait_job(self.app, window)
                self.assertEqual(window.current_frame, 5)
                self.assertEqual(window.timeline.propagation_start, 1)
                self.assertEqual(window.timeline.propagation_range, (2, 5))
                editor.clear_range_button.click()
                wait_job(self.app, window)
                self.assertEqual(editor.edits, {})
                editor.undo_button.click()
                wait_job(self.app, window)
                self.assertEqual(editor.edits, dict.fromkeys(range(2, 6), "propagated"))
            finally:
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard):
                    window.close()

    def test_brush_zoom_commit_follow_review_and_loss_guards(self):
        with TemporaryDirectory() as temporary:
            result = make_edit_analysis(Path(temporary))
            window = VideoMosaicWindow()
            window.show()
            window._set_analysis(result)
            window._update_controls()
            window.seek(1)
            window.settings_tabs.setCurrentIndex(1)
            self.app.processEvents()
            editor = window.mask_editor
            try:
                editor.edit_button.click()
                self.assertTrue(editor.active)
                self.assertFalse(window.seek_slider.isEnabled())
                self.assertFalse(window.export_button.isEnabled())
                editor.brush.setValue(12)
                def display_pixel(x, y):
                    rect = window.preview.image_rect
                    return QPoint(round(rect.left() + x / 128 * rect.width()), round(rect.top() + y / 96 * rect.height()))
                pos = display_pixel(40, 35)
                window.preview.zoom_requested.emit(QPointF(pos), 1)
                self.assertEqual(window.preview.zoom_index, 1)
                pos = display_pixel(40, 35)
                QTest.mouseClick(window.preview, Qt.MouseButton.LeftButton, pos=pos)
                self.assertTrue(editor.draft[35, 40])
                QTest.mouseClick(window.preview, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier, pos=pos)
                self.assertFalse(editor.draft[35, 40])
                QTest.mouseClick(window.preview, Qt.MouseButton.LeftButton, pos=pos)
                window.view_mode.setCurrentText("処理結果")
                self.assertTrue(editor.active)
                self.assertFalse(window.preview.mask_editing)
                editor.apply_button.click()
                wait_job(self.app, window)
                self.assertFalse(editor.active)
                self.assertTrue(window.export_button.isEnabled())
                self.assertEqual(editor.start, 1)
                window.seek(5)
                self.assertEqual(editor.propagation_range, (2, 5))
                editor.propagate_button.click()
                wait_job(self.app, window)
                self.assertEqual(editor.edits.get(5), "propagated", window.status.text())
                self.assertIn("自動補完済み", window.frame_label.text())
                self.assertEqual(window.timeline.edits, editor.edits)
                previous_config = window._config_values()
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel):
                    window.threshold_slider.setValue(40)
                    self.assertEqual(window._config_values(), previous_config)
                    self.assertIs(window.analysis, result)
                    window.close()
                    self.assertTrue(window.isVisible())
                editor.undo_button.click()
                wait_job(self.app, window)
                self.assertEqual(editor.edits, {1: "manual"})
                self.assertFalse(result.cache.full_mask(5, result.info)[35, 48])
            finally:
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard):
                    window.close()


if __name__ == "__main__":
    unittest.main()
