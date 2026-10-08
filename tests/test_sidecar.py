"""Focused persistence and native Qt workflow checks; no detector models required."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication, QMessageBox

from auto_mosaic.domain import Detection, EffectType, ImageMode, ProcessingSettings
from auto_mosaic.image_ops import apply_effect, load_image_bgr
from auto_mosaic.pipeline import MosaicPipeline
from auto_mosaic.sidecar import load_sidecar, save_sidecar, sidecar_path
from auto_mosaic.ui import AutoMosaicWindow


class SidecarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.source = self.folder / "sample.png"
        Image.fromarray(np.random.default_rng(42).integers(0, 255, (64, 96, 3), dtype=np.uint8)).save(self.source)
        self.settings = ProcessingSettings(ImageMode.ILLUSTRATION, frozenset({"penis"}), .4,
                                           effect=EffectType.BLUR, effect_size=12, mask_threshold=1.5,
                                           mask_expansion=5)
        self.mask = np.zeros((64, 96), bool)
        self.mask[10:40, 15:65] = True
        self.mask[20:30, 30:40] = False
        self.mask[60, 90] = True
        self.pipeline = MosaicPipeline(Path("models"))
        self.result = self.pipeline.process_with_mask(
            self.source, self.mask, self.settings, [Detection("penis", .9, (15, 10, 65, 40))],
            detection_masks=[self.mask.copy()],
        )
        self.windows = []

    def tearDown(self):
        for window in self.windows:
            window._end_mask_edit(False)
            window.close()
        self.tmp.cleanup()

    def window(self):
        window = AutoMosaicWindow()
        window.agent_notice = lambda *_args: None
        window.output_edit.setText(str(self.folder / "output"))
        self.windows.append(window)
        window.show()
        self.app.processEvents()
        return window

    def wait(self, window):
        deadline = monotonic() + 5
        # Selection starts its worker on the next Qt event cycle.
        while monotonic() < deadline:
            self.app.processEvents()
            window._poll_events()
            if not window.busy and window.current_result is not None:
                return
            sleep(.01)
        self.fail("Qt operation did not finish")

    def install(self, window):
        window.image_paths = [self.source]
        window.file_list.blockSignals(True)
        window.file_list.addItem(self.source.name)
        window.file_list.setCurrentRow(0)
        window.file_list.blockSignals(False)
        window.current_result = self.result
        window._restore_settings(self.settings)
        window._update_mask_edit_controls()
        window._refresh_preview()

    def test_exact_roundtrip_settings_holes_owned_mask_and_atomic_update(self):
        source_bytes = self.source.read_bytes()
        target = save_sidecar(self.result, self.settings)
        self.assertEqual(target.name, "sample.png.fy")
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertIsInstance(data["mask"], list)
        mask, settings, detections, owned = load_sidecar(self.source)
        np.testing.assert_array_equal(mask, self.mask)
        np.testing.assert_array_equal(owned[0], self.mask)
        self.assertEqual(settings, self.settings)
        self.assertEqual(detections, self.result.detections)
        self.result.mask[:] = False
        self.result.detection_masks[0][:] = False
        save_sidecar(self.result, self.settings)
        self.assertFalse(load_sidecar(self.source)[0].any())
        self.assertFalse(list(self.folder.glob("*.tmp")))
        self.assertEqual(self.source.read_bytes(), source_bytes)

    def test_changed_source_and_malformed_ranges_are_rejected(self):
        target = save_sidecar(self.result, self.settings)
        data = json.loads(target.read_text(encoding="utf-8"))
        data["mask"] = [[0, self.mask.size + 1]]
        target.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "範囲外"):
            load_sidecar(self.source)
        save_sidecar(self.result, self.settings)
        Image.new("RGB", (96, 64), "white").save(self.source)
        with self.assertRaisesRegex(ValueError, "元画像"):
            self.pipeline.restore(self.source)

    def test_mask_only_save_reopen_change_effect_and_export_without_detection(self):
        first = self.window()
        self.install(first)
        self.assertFalse(first.save_sidecar_check.isChecked())
        first._toggle_mask_edit()
        edited = self.mask.copy()
        edited[42:52, 5:20] = True
        first._apply_mask_edit(edited)
        first._save_mask_settings()
        self.wait(first)
        self.assertTrue(sidecar_path(self.source).exists())
        self.assertFalse((self.folder / "output").exists())
        self.assertFalse(first.mask_edit_dirty)
        second = self.window()
        with patch.object(second.pipeline, "analyze", side_effect=AssertionError("must not detect")):
            second._add_image_paths([self.source])
            self.wait(second)
            self.assertTrue(second.current_result.restored_from_sidecar)
            self.assertEqual(second._settings(), self.settings)
            np.testing.assert_array_equal(second.current_result.mask, edited)
            second.effect_combo.setCurrentText("モザイク")
            second.effect_size_slider.setValue(24)
            expected = apply_effect(load_image_bgr(self.source), edited, EffectType.MOSAIC, 24)
            np.testing.assert_array_equal(second.current_result.image_bgr, expected)
            second.save_sidecar_check.setChecked(True)
            second.process_current_button.click()
            self.wait(second)
            np.testing.assert_array_equal(load_image_bgr(self.folder / "output" / "sample_mosaic.png"), expected)
            self.assertEqual(load_sidecar(self.source)[1].effect_size, 24)
            second.process_button.click()
            self.wait(second)
            np.testing.assert_array_equal(load_image_bgr(self.folder / "output" / "sample_mosaic_2.png"), expected)

    def test_processed_save_option_and_failed_sidecar_keep_draft(self):
        window = self.window()
        self.install(window)
        window._toggle_mask_edit()
        with patch.object(window.pipeline, "analyze", side_effect=AssertionError("must not detect")):
            window.process_current_button.click()
            self.wait(window)
        self.assertFalse(sidecar_path(self.source).exists())
        window._toggle_mask_edit()
        edited = self.mask.copy()
        edited[50:55, 50:55] = True
        window._apply_mask_edit(edited)
        window.save_sidecar_check.setChecked(True)
        with patch("auto_mosaic.ui.save_sidecar", side_effect=PermissionError("read only")), patch.object(window, "_notice") as notice:
            window.process_current_button.click()
            self.wait(window)
            self.assertIn("画像は保存済み", notice.call_args.args[2])
        self.assertTrue(window.mask_edit_active and window.mask_edit_dirty)
        np.testing.assert_array_equal(window.edited_mask, edited)
        window.process_current_button.click()
        self.wait(window)
        np.testing.assert_array_equal(load_sidecar(self.source)[0], edited)
        self.assertTrue(window.current_result.restored_from_sidecar)
        window.effect_size_slider.setValue(28)
        with patch.object(window.pipeline, "analyze", side_effect=AssertionError("must preserve saved edit")):
            window.process_current_button.click()
            self.wait(window)
        np.testing.assert_array_equal(load_sidecar(self.source)[0], edited)
        self.assertEqual(load_sidecar(self.source)[1].effect_size, 28)

    def test_invalid_sidecar_does_not_fall_back_or_overwrite(self):
        target = sidecar_path(self.source)
        target.write_text("broken JSON", encoding="utf-8")
        window = self.window()
        notices = []
        window.agent_notice = lambda *args: notices.append(args)
        with patch.object(window.pipeline, "analyze", side_effect=AssertionError("must not detect")):
            window._add_image_paths([self.source])
            deadline = monotonic() + 5
            while monotonic() < deadline and not notices:
                self.app.processEvents()
                window._poll_events()
                sleep(.01)
            self.assertTrue(notices)
            self.assertIsNone(window.current_result)
            window.save_sidecar_check.setChecked(True)
            window.process_current_button.click()
            deadline = monotonic() + 5
            while window.busy and monotonic() < deadline:
                self.app.processEvents()
                window._poll_events()
                sleep(.01)
            self.assertFalse(window.busy)
        self.assertEqual(target.read_text(encoding="utf-8"), "broken JSON")
        self.assertFalse((self.folder / "output").exists())

    def test_restore_button_uses_adjacent_file_or_file_picker_and_keeps_draft_on_cancel(self):
        target = save_sidecar(self.result, self.settings)
        window = self.window()
        self.install(window)
        window.effect_size_slider.setValue(28)
        with patch("auto_mosaic.ui.QFileDialog.getOpenFileName") as picker, patch.object(
            window.pipeline, "analyze", side_effect=AssertionError("must not detect"),
        ):
            window.restore_sidecar_button.click()
            self.wait(window)
            picker.assert_not_called()
        self.assertEqual(window._settings(), self.settings)
        np.testing.assert_array_equal(window.current_result.mask, self.mask)
        moved = self.folder / "saved-mask.fy"
        target.rename(moved)
        window._toggle_mask_edit()
        edited = self.mask.copy()
        edited[50:55, 50:55] = True
        window._apply_mask_edit(edited)
        with patch("auto_mosaic.ui.QFileDialog.getOpenFileName", return_value=("", "")) as picker:
            window.restore_sidecar_button.click()
            picker.assert_called_once()
        self.assertTrue(window.mask_edit_dirty)
        np.testing.assert_array_equal(window.edited_mask, edited)
        moved.write_text("broken JSON", encoding="utf-8")
        with patch("auto_mosaic.ui.QFileDialog.getOpenFileName", return_value=(str(moved), "")), patch.object(
            window, "_notice",
        ) as notice:
            window.restore_sidecar_button.click()
            self.wait(window)
            self.assertEqual(notice.call_args.args[1], "マスクの復元")
        self.assertTrue(window.mask_edit_dirty)
        np.testing.assert_array_equal(window.edited_mask, edited)
        save_sidecar(self.result, self.settings).replace(moved)
        with patch("auto_mosaic.ui.QFileDialog.getOpenFileName", return_value=(str(moved), "")), patch(
            "auto_mosaic.ui.QMessageBox.warning", return_value=QMessageBox.StandardButton.Discard,
        ) as warning:
            window.restore_sidecar_button.click()
            self.wait(window)
            warning.assert_called_once()
        self.assertFalse(window.mask_edit_active)
        np.testing.assert_array_equal(window.current_result.mask, self.mask)
        self.assertEqual(window._settings(), self.settings)


if __name__ == "__main__":
    unittest.main()
