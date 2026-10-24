import threading
import unittest
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from auto_mosaic.domain import ProcessingResult
from auto_mosaic.ui import AutoMosaicWindow


class ImageWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = AutoMosaicWindow()
        self.addCleanup(self.window.close)
        self.notices = patch.object(self.window, "_notice").start()
        self.addCleanup(patch.stopall)
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.source = self.folder / "source.png"
        with Image.new("RGB", (32, 32), (100, 120, 140)) as image:
            image.save(self.source)
        self.window.output_edit.setText(str(self.folder / "output"))
        self.window.remove_after_process_check.setChecked(False)

    def wait_idle(self):
        deadline = monotonic() + 5
        while monotonic() < deadline:
            self.app.processEvents()
            self.window._poll_events()
            if not self.window.busy:
                return
            sleep(.005)
        self.fail("image worker did not finish")

    def test_analysis_save_batch_and_region_reuse_one_thread_and_release_results(self):
        threads = []
        batch_result_refs = []
        checking_batch = False

        def analyze(path, settings):
            threads.append(threading.current_thread())
            if checking_batch:
                # A saved result must be released before the next image starts.
                self.assertTrue(all(ref() is None for ref in batch_result_refs))
            result = ProcessingResult(path, np.zeros((32, 32, 3), np.uint8),
                                      np.zeros((32, 32), bool))
            if checking_batch:
                batch_result_refs.append(weakref.ref(result))
            return result

        self.window.pipeline.analyze = analyze
        self.window._add_image_paths([self.source])
        self.wait_idle()
        for _ in range(3):
            self.window._analyze_current()
            self.wait_idle()
        self.window._save_current()
        self.wait_idle()

        other = self.folder / "other.png"
        other.write_bytes(self.source.read_bytes())
        self.window._add_image_paths([other])
        checking_batch = True
        self.window._process_all()
        self.wait_idle()
        self.assertEqual(len(batch_result_refs), 2)
        self.assertTrue(all(ref() is None for ref in batch_result_refs))
        checking_batch = False

        self.window._toggle_mask_edit()
        def select_region(image, point, threshold, embedding):
            threads.append(threading.current_thread())
            return np.ones((32, 32), bool), object()
        self.window.pipeline.select_mask_region = select_region
        self.window._select_mask_region(self.window.preview_image_rect.center(), False)
        self.wait_idle()
        self.assertTrue(self.window.edited_mask.all())
        self.window._end_mask_edit(refresh_preview=False)

        self.assertEqual(len(threads), 8)
        self.assertEqual(len(set(threads)), 1)
        self.assertIsNot(threads[0], threading.current_thread())
        self.assertFalse(any(call.args[0] == "critical" for call in self.notices.call_args_list))

    def test_close_cancels_queued_work_and_invalidates_scheduled_analysis(self):
        started = threading.Event()
        release = threading.Event()
        def running():
            started.set()
            release.wait(5)
        active = self.window._worker.submit(running)
        try:
            self.assertTrue(started.wait(2))
            queued = self.window._worker.submit(lambda: None)
            self.window._add_image_paths([self.source])
            self.window.close()
            self.app.processEvents()
            self.assertTrue(queued.cancelled())
            self.assertFalse(self.window.poll_timer.isActive())
        finally:
            release.set()
            active.result(timeout=2)

    def test_close_stops_batch_before_next_image(self):
        started = threading.Event()
        release = threading.Event()
        processed = []
        def prepare(path, settings):
            processed.append(path)
            started.set()
            release.wait(5)
            return ProcessingResult(path, np.zeros((32, 32, 3), np.uint8),
                                      np.zeros((32, 32), bool))
        self.window.pipeline.prepare = prepare
        self.window.image_paths = [self.source, self.folder / "not-started.png"]
        try:
            self.window._process_all()
            self.assertTrue(started.wait(2))
            self.window.close()
        finally:
            release.set()
            self.window._worker.shutdown(wait=True)
        self.assertEqual(processed, [self.source])


if __name__ == "__main__":
    unittest.main()
