from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from auto_mosaic.domain import EffectType
from auto_mosaic.video_analysis import export_video
from auto_mosaic.video_io import VideoCancelled, decode_frames, probe_video
from auto_mosaic.video_mask_edit import propagate_mask, save_frame_mask, undo_mask_edit
from auto_mosaic.video_project import EditorState, ProjectSourceUnavailable, load_project, save_project
from auto_mosaic.video_ui import VideoMosaicWindow
from tests.test_video_mask_edit import make_edit_analysis, manual_mask
from tests.test_video_ui import wait_job


def progress(*args):
    pass


class VideoProjectTests(unittest.TestCase):
    def test_roundtrip_preserves_masks_draft_settings_undo_and_export(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            analysis = make_edit_analysis(directory)
            loaded = None
            try:
                save_frame_mask(analysis, 1, manual_mask(analysis, 1), Event(), progress)
                propagate_mask(analysis, 1, 5, Event(), progress)
                expected = [analysis.cache.full_mask(i, analysis.info) for i in range(8)]
                draft = expected[5].copy()
                draft[0:10, 0:10] = True
                state = EditorState(5, "処理結果", 1, 1, 12, 1, True)
                path = save_project(analysis, directory / "編集.fyvideo", EffectType.BLUR, 28, state, draft, Event(), progress)
                before = export_video(analysis, directory / "before.mp4", EffectType.BLUR, 28, Event(), progress)
                frames, edits = list(analysis.frames), analysis.cache.edit_state()
                analysis.close()
                analysis = None
                with patch("auto_mosaic.video_analysis.VideoFrameAnalyzer", side_effect=AssertionError("must not infer")):
                    loaded = load_project(path, Event(), progress)
                restored = loaded.analysis
                self.assertEqual(loaded.state, state)
                np.testing.assert_array_equal(loaded.draft, draft)
                self.assertEqual(restored.frames, frames)
                self.assertEqual(restored.cache.edit_state(), edits)
                self.assertIs(restored.settings.processing.effect, EffectType.BLUR)
                self.assertEqual(restored.settings.processing.effect_size, 28)
                for i, mask in enumerate(expected):
                    np.testing.assert_array_equal(restored.cache.full_mask(i, restored.info), mask)
                after = export_video(restored, directory / "after.mp4", EffectType.BLUR, 28, Event(), progress)
                for a, b in zip(decode_frames(probe_video(before, Event()), Event()), decode_frames(probe_video(after, Event()), Event())):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(undo_mask_edit(restored, Event(), progress).edited, {1: "manual"})
                # Undo in the reopened working cache must not alter the saved project.
                again = load_project(path, Event(), progress)
                try:
                    self.assertEqual(again.analysis.cache.edit_state(), edits)
                finally:
                    again.analysis.close()
            finally:
                if analysis is not None:
                    analysis.close()
                if loaded is not None:
                    loaded.analysis.close()

    def test_cancel_and_write_failure_preserve_existing_save(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            analysis = make_edit_analysis(directory)
            try:
                path = save_project(analysis, directory / "project.fyvideo", EffectType.MOSAIC, 16, EditorState(), None, Event(), progress)
                original = path.read_bytes()
                cancel = Event()
                with self.assertRaises(VideoCancelled):
                    save_project(analysis, path, EffectType.BLUR, 24, EditorState(), None, cancel, lambda *a: cancel.set())
                self.assertEqual(path.read_bytes(), original)
                with patch("auto_mosaic.video_project.os.replace", side_effect=OSError("disk full")):
                    with self.assertRaises(OSError):
                        save_project(analysis, path, EffectType.BLUR, 24, EditorState(), None, Event(), progress)
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse(list(directory.glob("*.tmp")))
                cancel = Event()
                with self.assertRaises(VideoCancelled):
                    load_project(path, cancel, lambda *a: cancel.set())
            finally:
                analysis.close()

    def test_source_relocation_identity_and_corrupt_project(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            analysis = make_edit_analysis(directory)
            try:
                path = save_project(analysis, directory / "project.fyvideo", EffectType.MOSAIC, 16, EditorState(), None, Event(), progress)
                moved = directory / "moved"
                moved.mkdir()
                moved_source = moved / analysis.info.path.name
                analysis.info.path.rename(moved_source)
                with self.assertRaises(ProjectSourceUnavailable):
                    load_project(path, Event(), progress)
                loaded = load_project(path, Event(), progress, source_path=moved_source)
                loaded.analysis.close()
                moved_project = moved / path.name
                shutil.copy2(path, moved_project)
                loaded = load_project(moved_project, Event(), progress)
                loaded.analysis.close()
                original = moved_source.read_bytes()
                stat = moved_source.stat()
                moved_source.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                os.utime(moved_source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                with self.assertRaises(ProjectSourceUnavailable):
                    load_project(moved_project, Event(), progress)
                moved_source.write_bytes(original)
                with closing(sqlite3.connect(moved_project)) as connection, connection:
                    connection.execute("DELETE FROM frames WHERE frame=2")
                with self.assertRaisesRegex(ValueError, "不足"):
                    load_project(moved_project, Event(), progress)
            finally:
                analysis.close()


class VideoProjectUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/meiryo.ttc")

    def test_buttons_reopen_draft_apply_and_failure_preserves_session(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            window = VideoMosaicWindow()
            window.show()
            window._set_analysis(make_edit_analysis(directory))
            window._update_controls()
            self.app.processEvents()
            path = directory / "project.fyvideo"
            try:
                window.seek(1)
                window.settings_tabs.setCurrentIndex(1)
                window.effect.setCurrentIndex(1)
                window.effect_size.setValue(28)
                window.zoom.setCurrentIndex(2)
                window.mask_editor.edit_button.click()
                rect = window.preview.image_rect
                point = QPointF(rect.left() + 35 / 128 * rect.width(), rect.top() + 35 / 96 * rect.height())
                window.mask_editor.paint(point, point, False)
                expected = window.mask_editor.draft.copy()
                self.assertTrue(window.project_dirty)
                with patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "")):
                    window.activateWindow()
                    window.mask_editor.brush.setFocus()
                    self.app.processEvents()
                    QTest.keyClick(window.mask_editor.brush, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
                    wait_job(self.app, window)
                self.assertEqual(window.project_path, path)
                self.assertFalse(window.project_dirty)
                with patch.object(QFileDialog, "getSaveFileName", return_value=("", "")) as dialog:
                    window.project_save_button.click()
                    button_call = dialog.call_args
                    dialog.reset_mock()
                    QTest.keyClick(window.mask_editor.brush, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
                    self.assertEqual(dialog.call_args, button_call)
                    dialog.assert_called_once()
                with patch.object(QMessageBox, "warning", side_effect=AssertionError("saved edits must not prompt")):
                    window.close()
                window.show()
                self.app.processEvents()
                with patch.object(QFileDialog, "getOpenFileName", return_value=(str(path), "")):
                    window.project_open_button.click()
                    wait_job(self.app, window)
                np.testing.assert_array_equal(window.mask_editor.draft, expected)
                self.assertTrue(window.mask_editor.active)
                self.assertEqual(window.current_frame, 1)
                self.assertEqual(window.zoom.currentData(), 1)
                self.assertEqual(window.settings_tabs.currentIndex(), 1)
                self.assertEqual(window._effect_type(), EffectType.BLUR)
                self.assertEqual(window.effect_size.value(), 28)
                self.assertEqual(window._settings(), replace(window.analysis.settings,
                    processing=replace(window.analysis.settings.processing, effect=EffectType.MOSAIC, effect_size=16)))
                self.assertFalse(window.export_button.isEnabled())
                window.mask_editor.apply_button.click()
                wait_job(self.app, window)
                self.assertTrue(window.project_dirty)
                self.assertTrue(window.export_button.isEnabled())
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel):
                    window.close()
                    self.assertTrue(window.isVisible())
                previous = window.analysis
                broken = directory / "broken.fyvideo"
                broken.write_bytes(b"broken")
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard):
                    window.load_project_from(broken)
                    wait_job(self.app, window)
                self.assertIs(window.analysis, previous)
                self.assertTrue(window.project_dirty)
                self.assertIn("復元できません", window.status.text())
                # Saving again clears the dirty flag; changing the effect sets it again.
                window.save_project_to(path)
                wait_job(self.app, window)
                self.assertFalse(window.project_dirty)
                window.effect_size.setValue(32)
                self.assertTrue(window.project_dirty)
            finally:
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard):
                    window.close()


if __name__ == "__main__":
    unittest.main()
