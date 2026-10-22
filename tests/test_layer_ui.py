import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from auto_mosaic.domain import Detection, ProcessingResult
from auto_mosaic.layer_ui import LayerMosaicWindow
from tests.layer_fixtures import make_layers
from tests.test_layers import SETTINGS


def wait_idle(app, window):
    deadline = time.monotonic() + 20
    while window.busy or window.render_timer.isActive():
        app.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError(window.status.text())
        time.sleep(.01)
    app.processEvents()


def synthetic_analysis(image, settings, *, source_path):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), bool)
    mask[height//4:height*3//4, width//4:width*3//4] = True
    return ProcessingResult(source_path, image, mask,
                            [Detection("penis", .9, (width//4, height//4, width*3//4, height*3//4))])


class LayerUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_checks_invalidate_then_save_current_result(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "art.psd"
            background, _ = make_layers(path)
            window = LayerMosaicWindow(Path("models"), SETTINGS)
            window.pipeline.analyze_image = Mock(side_effect=synthetic_analysis)
            window.show()
            with patch.object(QMessageBox, "warning") as warnings:
                window.load_path(path)
                wait_idle(self.app, window)
                self.assertFalse(warnings.called)
                self.assertIsNotNone(window.result)
                self.assertTrue(window.flatten_button.isEnabled())
                foreground = window.layers.topLevelItem(0)
                foreground.setCheckState(3, Qt.CheckState.Unchecked)
                self.assertFalse(window.flatten_button.isEnabled())
                self.assertEqual(foreground.checkState(1), Qt.CheckState.Checked)
                self.assertEqual(foreground.checkState(2), Qt.CheckState.Checked)
                wait_idle(self.app, window)
                self.assertEqual(window.pipeline.analyze_image.call_count, 1)
                np.testing.assert_array_equal(window.result.bgra, background)
                np.testing.assert_array_equal(window.original, background)
                foreground.setCheckState(3, Qt.CheckState.Checked)
                wait_idle(self.app, window)
                self.assertTrue(np.any(window.result.bgra != background))
                window.layers.topLevelItem(0).setCheckState(1, Qt.CheckState.Unchecked)
                self.assertFalse(window.flatten_button.isEnabled())
                wait_idle(self.app, window)
                self.assertEqual(window.pipeline.analyze_image.call_count, 1)
                # A detection edit must not cancel an accepted, pending visibility update.
                foreground.setCheckState(3, Qt.CheckState.Unchecked)
                window.layers.topLevelItem(0).setCheckState(2, Qt.CheckState.Unchecked)
                self.assertIsNone(window.result)
                self.assertFalse(window.flatten_button.isEnabled())
                wait_idle(self.app, window)
                np.testing.assert_array_equal(window.original, background)
                self.assertIsNone(window.result)
                self.assertFalse(window.flatten_button.isEnabled())
                window.analyze_button.click()
                wait_idle(self.app, window)
                window.save_to(root / "saved.png")
                wait_idle(self.app, window)
                self.assertTrue((root / "saved.png").exists())
                self.assertFalse(window.dirty)
                self.assertFalse(warnings.called)
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
