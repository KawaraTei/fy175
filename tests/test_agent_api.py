"""Focused HTTP-to-Qt integration checks, using deterministic local media."""
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from auto_mosaic.agent_api import AgentServer
from auto_mosaic.agent_client import request
from auto_mosaic.domain import Detection, ProcessingResult
from auto_mosaic.image_ops import apply_effect, load_image_bgr, save_image_bgr
from auto_mosaic.ui import AutoMosaicWindow


class DeterministicPipeline:
    def __init__(self, original):
        self.original = original

    def analyze(self, path, settings):
        image = load_image_bgr(path)
        mask = np.zeros(image.shape[:2], bool)
        mask[25:60, 35:90] = True
        return ProcessingResult(path, apply_effect(image, mask, settings.effect, settings.effect_size), mask,
                                [Detection("penis", .9, (35, 25, 90, 60))], detection_masks=[mask.copy()])

    def process_with_mask(self, *args, **kwargs):
        return self.original.process_with_mask(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.original.save(*args, **kwargs)

    def restore(self, *args, **kwargs):
        return self.original.restore(*args, **kwargs)

    def prepare(self, path, settings):
        restored = self.restore(path, settings)
        return restored[0] if restored else self.analyze(path, settings)


class AgentApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.executor = ThreadPoolExecutor(max_workers=2)

    @classmethod
    def tearDownClass(cls):
        cls.executor.shutdown()

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = self.root / "fixture.png"
        rng = np.random.default_rng(42)
        save_image_bgr(self.image, rng.integers(0, 255, (100, 160, 3), dtype=np.uint8))
        self.window = AutoMosaicWindow()
        self.window.pipeline = DeterministicPipeline(self.window.pipeline)
        self.window.show()
        self.server = AgentServer(self.window, 0)
        self.url = self.server.url

    def tearDown(self):
        self.server.close()
        self.window._end_mask_edit(False)
        self.window.close()
        self.tmp.cleanup()

    def http(self, endpoint="/", body=None):
        future = self.executor.submit(request, self.url, endpoint, body)
        deadline = time.monotonic() + 20
        while not future.done() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertTrue(future.done(), "HTTP request timed out")
        return future.result()

    def call(self, name, wait=True, **args):
        result, _ = self.http("/commands", {"operation": name, "arguments": args})
        if wait and result["job"]:
            deadline = time.monotonic() + 30
            job = result["job"]
            while job["status"] == "running" and time.monotonic() < deadline:
                time.sleep(.02)
                job, _ = self.http("/jobs/" + job["id"])
            self.assertEqual(job["status"], "succeeded", job)
            return job
        return result

    def decoded(self, endpoint):
        content, headers = self.http(endpoint)
        return cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED), headers

    def test_sidecar_save_restore_settings_and_export_provenance(self):
        from auto_mosaic.sidecar import sidecar_path

        self.call("settings.update", values={"output_dir": str(self.root / "out")})
        self.call("images.add", paths=[str(self.image)])
        self.call("mask.edit", shape="rectangle", points=[[5, 5], [20, 20]], action="add")
        expected, _ = self.decoded("/mask")
        saved = self.call("image.save_mask_settings")
        self.assertEqual(saved["result"]["outputs"], [str(sidecar_path(self.image))])
        self.call("images.clear", discard=True)
        with patch.object(self.window.pipeline, "analyze", side_effect=AssertionError("must not detect")):
            restored = self.call("images.add", paths=[str(self.image)])
            self.assertEqual(restored["result"]["mask_source"], "restored")
            self.call("settings.update", values={"effect_size": 24, "save_mask_settings": True})
            metadata, _ = self.http("/preview/metadata")
            self.assertEqual(metadata["mask_source"], "restored")
            mask, _ = self.decoded("/mask")
            np.testing.assert_array_equal(mask, expected)
            exported = self.call("image.save")
            self.assertEqual(exported["result"]["mask_source"], "restored")
            self.assertFalse(exported["result"]["detection_rerun"])
            batch = self.call("images.save_all")
            self.assertEqual(batch["result"]["saved_images"][0]["mask_source"], "restored")
        self.call("settings.update", values={"effect_size": 32})
        restored = self.call("image.restore_mask_settings")
        self.assertEqual(restored["result"]["mask_source"], "restored")
        self.assertEqual(self.window.effect_size_slider.value(), 24)
        relocated = self.root / "elsewhere.fy"
        sidecar_path(self.image).rename(relocated)
        self.call("image.restore_mask_settings", path=str(relocated))

    def test_discovery_image_preview_edit_and_export(self):
        description, _ = self.http()
        self.assertIn("mask.edit", description["operations"])
        self.assertEqual(description["settings_schema"]["image"]["effect_size"]["maximum"], self.window.effect_size_slider.maximum())
        self.call("settings.update", values={"output_dir": str(self.root / "out"), "suffix": "_edited"})
        self.call("images.add", paths=[str(self.image)])
        self.call("mask.edit", shape="rectangle", points=[[5, 5], [20, 20]], action="add")
        self.call("mask.edit", shape="stroke", points=[[50, 40], [65, 40]], diameter=12, action="erase")
        self.call("mask.edit", shape="polygon", points=[[120, 70], [145, 70], [135, 90]], action="add")
        mask, _ = self.decoded("/mask")
        self.assertEqual(mask.shape, (100, 160))
        self.assertEqual(mask[10, 10], 255)
        self.assertEqual(mask[40, 55], 0)
        preview, metadata = self.decoded("/preview?view=result&max_size=80&crop=0,0,160,100")
        self.assertEqual(preview.shape[:2], (50, 80))
        self.assertEqual(json.loads(metadata["X-Agent-Metadata"])["source_units_per_pixel"], [2, 2])
        full_preview, _ = self.decoded("/preview?view=result")
        self.call("settings.update", values={"effect": "blur", "effect_size": 8})
        changed, _ = self.decoded("/preview?view=result")
        self.assertFalse(np.array_equal(full_preview, changed))
        job = self.call("image.save")
        self.assertEqual(job["result"]["mask_source"], "edited")
        self.assertTrue(job["result"]["used_edited_mask"])
        self.assertFalse(job["result"]["detection_rerun"])
        exported = load_image_bgr(Path(job["result"]["outputs"][0]))
        self.assertTrue(np.array_equal(exported, changed))
        source = load_image_bgr(self.image)
        self.assertTrue(np.array_equal(source[mask == 0], exported[mask == 0]))
        self.assertEqual(self.window.agent_notice is not None, True)
        snapshot, _ = self.decoded("/snapshot")
        self.assertGreater(snapshot.shape[1], 1000)

    def test_save_provenance_and_http11(self):
        def persistent_requests():
            connection = HTTPConnection("127.0.0.1", self.server.http.server_port, timeout=10)
            try:
                connection.request("GET", "/")
                first = connection.getresponse()
                description = json.loads(first.read())
                socket = connection.sock
                connection.request("GET", "/state")
                second = connection.getresponse()
                state = json.loads(second.read())
                return first.version, second.version, description, state, socket is connection.sock
            finally:
                connection.close()
        future = self.executor.submit(persistent_requests)
        deadline = time.monotonic() + 15
        while not future.done() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        versions = future.result(timeout=1)
        self.assertEqual(versions[:2], (11, 11))
        self.assertEqual(versions[2]["pid"], os.getpid())
        self.assertTrue(versions[4], "Connection should remain persistent")
        self.call("settings.update", values={"output_dir": str(self.root / "out")})
        self.call("images.add", paths=[str(self.image)])
        job = self.call("image.save")
        self.assertEqual(job["result"]["mask_source"], "reanalyzed")
        self.assertTrue(job["result"]["detection_rerun"])
        self.assertFalse(job["result"]["used_edited_mask"])
        batch = self.call("images.save_all")["result"]
        self.assertEqual(batch["saved_images"][0]["path"], str(self.image))
        self.assertEqual(batch["saved_images"][0]["mask_source"], "reanalyzed")

    def test_indexed_preview_preserves_draft_and_json_transform(self):
        second = self.root / "second.png"
        source = np.full((100, 160, 3), (50, 110, 190), np.uint8)
        save_image_bgr(second, source)
        self.call("images.add", paths=[str(self.image), str(second)])
        self.call("mask.edit", shape="rectangle", points=[[5, 5], [15, 15]])
        before, _ = self.http("/state")
        mask = self.window.edited_mask.copy()
        original, _ = self.decoded("/preview?image_index=1&view=original")
        self.assertTrue(np.array_equal(original, source))
        endpoint = "/preview?image_index=1&view=mask_overlay&crop=20,10,100,80&max_size=64"
        with patch.object(self.window.pipeline, "analyze", wraps=self.window.pipeline.analyze) as analyze:
            started, _ = self.http(endpoint)
            self.assertEqual(started["retry_url"], endpoint)
            job = started["job"]
            deadline = time.monotonic() + 15
            while job["status"] == "running" and time.monotonic() < deadline:
                job, _ = self.http("/jobs/" + job["id"])
            self.assertEqual(job["status"], "succeeded", job)
            preview, headers = self.decoded(endpoint)
            metadata, _ = self.http(endpoint.replace("/preview?", "/preview/metadata?"))
            self.assertEqual(metadata, json.loads(headers["X-Agent-Metadata"]))
            self.assertEqual(metadata["source_crop"], [20, 10, 100, 80])
            self.assertEqual(metadata["image_index"], 1)
            self.assertEqual(preview.shape[:2], (51, 64))
            self.assertEqual(analyze.call_count, 1)
        after, _ = self.http("/state")
        for key in ("selected_index", "undo_depth", "editing", "preview_view"):
            self.assertEqual(after["image"][key], before["image"][key])
        self.assertTrue(np.array_equal(self.window.edited_mask, mask))
        self.call("mask.undo")
        with patch.object(self.window.pipeline, "analyze", side_effect=RuntimeError("indexed failure")):
            self.call("settings.update", values={"effect_size": 32})
            started, _ = self.http(endpoint)
            job = started["job"]
            while job["status"] == "running":
                job, _ = self.http("/jobs/" + job["id"])
            self.assertEqual(job["status"], "failed")
            self.assertIn("indexed failure", job["error"])

    def test_invalid_requests_and_async_errors_are_data(self):
        before = self.window.effect_size_slider.value()
        with self.assertRaises(HTTPError) as error:
            self.call("settings.update", values={"effect_size": 32, "mask_expansion": 999})
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.window.effect_size_slider.value(), before)
        self.call("images.add", paths=[str(self.image)])
        self.call("mask.edit", points=[[10, 10]], diameter=10)
        with self.assertRaises(HTTPError) as error:
            self.call("image.analyze")
        self.assertEqual(error.exception.code, 409)
        self.call("mask.discard")
        with patch.object(self.window.pipeline, "analyze", side_effect=RuntimeError("expected inference failure")):
            started = self.call("image.analyze", wait=False)
            job = started["job"]
            while job["status"] == "running":
                job, _ = self.http("/jobs/" + job["id"])
            self.assertEqual(job["status"], "failed")
            self.assertIn("expected inference failure", job["error"])

    def test_image_only_scope(self):
        description, _ = self.http()
        self.assertEqual(description["supported_media"], ["image"])
        self.assertFalse(any(name.startswith("video.") for name in description["operations"]))
        with self.assertRaises(HTTPError) as error:
            self.http("/preview?workspace=video")
        self.assertEqual(error.exception.code, 400)
        with self.assertRaises(HTTPError) as error:
            self.call("video.open", path=str(self.image))
        self.assertEqual(error.exception.code, 404)

    def test_machine_state_utf8_and_detection_details(self):
        state, headers = self.http("/state")
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(state["image"]["status"], "empty")
        self.assertEqual(state["image"]["preview_view"], "result")
        self.assertIn("画像", state["image"]["status_text"])
        job = self.call("images.add", paths=[str(self.image)])
        self.assertEqual(job["result"]["detections"][0]["index"], 0)
        self.assertEqual(job["result"]["detections"][0]["class_name"], "penis")
        self.assertEqual(job["result"]["detection_count"], 1)

    def test_gui_and_api_share_mask_undo_history(self):
        self.call("images.add", paths=[str(self.image)])
        original = self.window.current_result.mask.copy()
        self.call("mask.edit", shape="rectangle", points=[[5, 5], [15, 15]])
        after_api = self.window.edited_mask.copy()
        point = self.window.preview_image_rect.center()
        self.window._paint_mask_stroke(point, point, True)
        self.assertFalse(np.array_equal(self.window.edited_mask, after_api))
        self.call("mask.undo")
        self.assertTrue(np.array_equal(self.window.edited_mask, after_api))
        self.window.activateWindow()
        QTest.qWait(30)
        QTest.keyClick(self.window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        self.assertTrue(np.array_equal(self.window.edited_mask, original))
        state, _ = self.http("/state")
        self.assertEqual(state["image"]["undo_depth"], 0)
        self.assertFalse(state["image"]["mask_dirty"])

    def test_detection_edits_preserve_overlap_and_undo(self):
        self.call("images.add", paths=[str(self.image)])
        reference = self.window.current_result
        second = np.zeros(reference.mask.shape, bool)
        second[40:75, 70:110] = True
        reference.detections.append(Detection("penis", .8, (70, 40, 110, 75)))
        reference.detection_masks.append(second)
        reference.mask |= second
        original = reference.mask.copy()
        self.call("mask.remove_detection", index=0)
        edited, _ = self.decoded("/mask")
        self.assertEqual(edited[30, 40], 0)
        self.assertEqual(edited[45, 75], 255)
        owned, _ = self.decoded("/mask?index=0")
        self.assertFalse(owned.any())
        self.call("mask.undo")
        restored, _ = self.decoded("/mask")
        self.assertTrue(np.array_equal(restored > 0, original))
        self.call("mask.dilate", index=0, px=3)
        grown, _ = self.decoded("/mask")
        self.assertEqual(grown[23, 40], 255)
        self.call("mask.undo")
        self.call("mask.erode", index=0, px=3)
        shrunk, _ = self.decoded("/mask")
        self.assertEqual(shrunk[26, 40], 0)
        self.assertEqual(shrunk[45, 75], 255)
        self.call("mask.undo")
        self.call("mask.edit", shape="rectangle", points=[[5, 5], [15, 15]])
        self.call("mask.undo")
        restored, _ = self.decoded("/mask")
        self.assertTrue(np.array_equal(restored > 0, original))
        self.call("mask.edit", shape="rectangle", points=[[5, 5], [15, 15]])
        overlay, _ = self.decoded("/preview?view=mask_overlay")
        expected = self.window.preview_rgb[:, :, ::-1]
        self.assertTrue(np.array_equal(overlay, expected))
        self.assertEqual(self.window.preview_mode(), "マスク範囲")
        output = self.root / "任意の名前.png"
        job = self.call("image.save", path=str(output))
        self.assertEqual(job["result"]["outputs"], [str(output)])
        self.assertTrue(output.is_file())
        state, _ = self.http("/state")
        self.assertEqual(state["image"]["undo_depth"], 0)
        with self.assertRaises(HTTPError) as error:
            self.call("image.save", path=str(output))
        self.assertEqual(error.exception.code, 409)


if __name__ == "__main__":
    unittest.main()
