from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from auto_mosaic.domain import Detection, ProcessingResult
from auto_mosaic.image_ops import box_mask, load_image_bgr
from auto_mosaic.mask_selection import alternative_prompt_points, rank_alternative_masks
from auto_mosaic.pipeline import MosaicPipeline
from auto_mosaic.sidecar import load_sidecar
from auto_mosaic.ui import AutoMosaicWindow
from tests.test_segmenter import _segmenter_and_embedding


def regions():
    original = np.zeros((120, 160), bool)
    original[20:100, 65:95] = True
    alternative = np.zeros_like(original)
    alternative[30:90, 25:60] = True
    alternative[30:90, 100:135] = True
    return original, alternative, (20, 20, 140, 100)


class MaskCandidateTests(unittest.TestCase):
    def test_alternatives_keep_both_sides_and_reject_original_duplicates(self):
        original, alternative, box = regions()
        candidates = rank_alternative_masks([
            (original, .99), (original | alternative, .98),
            (alternative, .9), (alternative.copy(), .8),
            (np.ones_like(original), .7),
        ], original, box, 0, 0)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(candidates[1][50, 40] and candidates[1][50, 120])
        self.assertFalse(candidates[1][50, 80])
        self.assertFalse(candidates[1][:20].any())
        points, negative = alternative_prompt_points(original, box)
        self.assertTrue(original[negative[1], negative[0]])
        self.assertTrue(any(x < 65 for x, y in points))
        self.assertTrue(any(x >= 95 for x, y in points))
        self.assertTrue(all(not original[y, x] for x, y in points))

    def test_rectangle_fallback_can_switch_to_model_region(self):
        original, alternative, box = regions()
        original = box_mask(original.shape, box)
        self.assertEqual(alternative_prompt_points(original, box), ([], None))
        self.assertEqual(len(rank_alternative_masks([(alternative, .9)], original, box, 0, 0)), 2)

    def test_alternative_prompt_does_not_force_positive_center(self):
        segmenter, embedding = _segmenter_and_embedding()
        segmenter.mask_candidates_from_box_points(embedding, (1, 1, 7, 7), 0)
        self.assertEqual(segmenter.decoder.feed['point_labels'].tolist(), [[2, 3]])
        segmenter.mask_candidates_from_box_points(embedding, (1, 1, 7, 7), 0, (2, 4), (4, 4))
        self.assertEqual(segmenter.decoder.feed['point_labels'].tolist(), [[2, 3, 1, 0]])
        self.assertEqual(segmenter.decoder.feed['point_coords'].tolist(), [[[1, 1], [7, 7], [2, 4], [4, 4]]])

    def test_pipeline_reuses_embedding_and_settings(self):
        original, alternative, box = regions()
        pipeline = MosaicPipeline(Path('models'))
        segmenter = Mock()
        segmenter.mask_candidates_from_box_points.side_effect = lambda *args: [(alternative, .9)]
        pipeline._segmenter = segmenter
        from auto_mosaic.domain import ProcessingSettings, ImageMode
        settings = ProcessingSettings(ImageMode.ILLUSTRATION, frozenset({'penis'}), .25,
                                      mask_threshold=1.5, mask_expansion=0)
        image = np.zeros((*original.shape, 3), np.uint8)
        candidates, embedding = pipeline.detection_mask_alternatives(image, box, original, settings)
        pipeline.detection_mask_alternatives(image, box, original, settings, embedding)
        segmenter.encode.assert_called_once()
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(call.args[2] == 1.5 for call in segmenter.mask_candidates_from_box_points.call_args_list))

    def test_button_cycle_undo_overlap_save_and_no_candidate(self):
        app = QApplication.instance() or QApplication([])
        window = AutoMosaicWindow()
        window.show()
        def wait():
            deadline = monotonic() + 5
            while monotonic() < deadline:
                app.processEvents()
                window._poll_events()
                if not window.busy and window.current_result is not None:
                    return
                sleep(.01)
            self.fail('worker timeout')

        try:
            self.assertFalse(window.next_mask_button.isEnabled())
            with TemporaryDirectory() as folder:
                path = Path(folder) / 'sample.png'
                Image.new('RGB', (160, 120), (100, 130, 150)).save(path)
                original, alternative, box = regions()
                other = np.zeros_like(original)
                other[75:105, 80:115] = True
                result = ProcessingResult(path, load_image_bgr(path), original | other,
                    [Detection('penis', .9, (80, 75, 115, 105)), Detection('penis', .85, box)],
                    detection_masks=[other, original])
                window.pipeline.analyze = lambda *_: result
                window.pipeline.detection_mask_alternatives = Mock(return_value=([original, alternative], object()))
                window._add_image_paths([path])
                wait()
                window.mask_target_combo.setCurrentIndex(1)
                self.assertTrue(window.next_mask_button.isEnabled())
                QTest.mouseClick(window.next_mask_button, Qt.MouseButton.LeftButton)
                self.assertFalse(window.next_mask_button.isEnabled())
                wait()
                self.assertEqual(window.mask_target_combo.currentIndex(), 1)
                np.testing.assert_array_equal(window.edited_mask, alternative | other)
                self.assertTrue(window.mask_edit_dirty)
                window._undo_mask_edit()
                np.testing.assert_array_equal(window.edited_mask, original | other)
                window.next_mask_button.click()
                np.testing.assert_array_equal(window.edited_mask, alternative | other)
                window.next_mask_button.click()
                np.testing.assert_array_equal(window.edited_mask, original | other)
                window.pipeline.detection_mask_alternatives.assert_called_once()
                window.mask_target_combo.setCurrentIndex(0)
                window.pipeline.detection_mask_alternatives.return_value = ([other], object())
                window.next_mask_button.click()
                wait()
                window.mask_target_combo.setCurrentIndex(1)
                window.next_mask_button.click()
                self.assertEqual(window.pipeline.detection_mask_alternatives.call_count, 2)
                np.testing.assert_array_equal(window.edited_mask, alternative | other)
                with patch.object(window, '_notice'):
                    window._save_mask_settings()
                    wait()
                restored = load_sidecar(path)
                np.testing.assert_array_equal(restored[0], alternative | other)
                np.testing.assert_array_equal(restored[3][1], alternative)
                window._mask_candidates.clear()
                window.pipeline.detection_mask_alternatives.return_value = ([alternative], object())
                before = window.edited_mask.copy()
                window.next_mask_button.click()
                wait()
                np.testing.assert_array_equal(window.edited_mask, before)
                self.assertIn('見つかりません', window.status_label.text())
                window._mask_candidates.clear()
                window.pipeline.detection_mask_alternatives.side_effect = RuntimeError('test error')
                with patch.object(window, '_notice'):
                    window.next_mask_button.click()
                    wait()
                np.testing.assert_array_equal(window.edited_mask, before)
                self.assertTrue(window.next_mask_button.isEnabled())
        finally:
            window._end_mask_edit(False)
            window.close()
            app.processEvents()


if __name__ == '__main__':
    unittest.main()
