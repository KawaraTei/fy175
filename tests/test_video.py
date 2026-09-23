from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import replace
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import numpy as np

from auto_mosaic.domain import Detection, EffectType, ImageMode, ProcessingSettings
from auto_mosaic.video_analysis import (
    FrameSummary, VideoAnalysisSettings, VideoFrameAnalyzer, analyze_video,
    detection_changes, export_video,
)
from auto_mosaic.video_io import VideoCancelled, decode_frames, probe_video, media_process, tool_path
from tests.video_fixtures import ShapeDetector, ShapeSegmenter, make_video


def settings(interval=1):
    return VideoAnalysisSettings(ProcessingSettings(ImageMode.ILLUSTRATION, frozenset({"penis"}), 0.25), interval)


class VideoTests(unittest.TestCase):
    def test_timeline_keeps_single_frame_gap_and_count_changes(self):
        changes = detection_changes([FrameSummary(n) for n in [0, 1, 1, 2, 0, 2, 1, 0]])
        self.assertEqual([(c.frame, c.before, c.after) for c in changes],
                         [(1, 0, 1), (3, 1, 2), (4, 2, 0), (5, 0, 2), (6, 2, 1), (7, 1, 0)])
        self.assertEqual(changes[2].description, "検出途切れ")

    def test_contour_reuse_does_not_skip_detection_or_hide_gaps(self):
        class Detector:
            def __init__(self):
                self.calls = 0
            def detect(self, *_args):
                self.calls += 1
                if self.calls == 4:
                    return []
                return [Detection("penis", .9, (20, 20, 60, 60))]
        detector, segmenter = Detector(), ShapeSegmenter()
        config = settings(5)
        config = replace(config, processing=replace(config.processing, mask_threshold=1.5))
        engine = VideoFrameAnalyzer(Path("models"), config, detector=detector, segmenter=segmenter)
        image = np.zeros((100, 100, 3), np.uint8)
        summaries = [engine.analyze(image, Event())[2] for _ in range(6)]
        self.assertEqual(detector.calls, 6)
        self.assertEqual(segmenter.calls, 2)
        self.assertEqual(segmenter.mask_threshold, 1.5)
        self.assertEqual([f.count for f in summaries], [1, 1, 1, 0, 1, 1])
        engine.analyze(np.full_like(image, 255), Event())
        self.assertEqual(segmenter.calls, 3, "scene cut must refresh the contour")

    def test_media_analysis_export_audio_and_cache(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = make_video(directory / "入力 テスト.mp4")
            info = probe_video(source, Event())
            detector, segmenter = ShapeDetector(), ShapeSegmenter()
            engine = VideoFrameAnalyzer(Path("models"), settings(3), detector=detector, segmenter=segmenter)
            result = analyze_video(info, settings(3), Path("models"), Event(), lambda *a: None, analyzer=engine)
            try:
                self.assertEqual(len(result.frames), 48)
                self.assertEqual([result.frames[i].count for i in (0, 6, 15, 24, 25, 33, 42)], [0, 1, 2, 0, 2, 1, 0])
                self.assertEqual(detector.calls, 48)
                self.assertLess(engine.encoder_calls, 30)
                output = export_video(result, directory / "処理済み.mp4", EffectType.MOSAIC, 24, Event(), lambda *a: None)
                exported = probe_video(output, Event())
                self.assertTrue(exported.has_audio)
                self.assertEqual((exported.width, exported.height, exported.fps), (640, 360, info.fps))
                self.assertAlmostEqual(exported.duration, result.duration, delta=1 / float(info.fps))
                original_frames = list(decode_frames(info, Event()))
                output_frames = list(decode_frames(exported, Event()))
                self.assertEqual(len(output_frames), len(result.frames))
                # Within a cached mask the exported pixels must actually change.
                with closing(sqlite3.connect(result.cache.path)) as connection:
                    row = connection.execute("SELECT mask FROM frames WHERE frame=15").fetchone()
                mask = result.cache.decode_mask(row[0], (360, 640))
                difference = np.abs(output_frames[15].astype(float) - original_frames[15]).mean(axis=2)
                self.assertGreater(difference[mask].mean(), difference[~mask].mean() + 1)
                self.assertEqual(detector.calls, 48, "export must use reviewed masks without reinference")
                with self.assertRaises(FileExistsError):
                    export_video(result, output, EffectType.MOSAIC, 24, Event(), lambda *a: None)
                cancel = Event()
                def stop_after_frame(*_args):
                    cancel.set()
                with self.assertRaises(VideoCancelled):
                    export_video(result, directory / "cancelled.mp4", EffectType.MOSAIC, 24, cancel, stop_after_frame)
                self.assertFalse((directory / "cancelled.mp4").exists())
                self.assertFalse(list(directory.glob(".fy175-export-*")))
            finally:
                result.close()

    def test_source_change_is_rejected(self):
        with TemporaryDirectory() as temporary:
            path = make_video(Path(temporary) / "silent.mp4", audio=False, frames=12)
            info = probe_video(path, Event())
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                info.verify_source()

    def test_variable_rate_rotated_silent_video_uses_identical_frame_sequence(self):
        class EmptyDetector:
            def detect(self, *_args):
                return []
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = make_video(directory / "source.mp4", audio=False)
            variable = directory / "variable.mp4"
            expression = "setpts=if(lt(N,24),N/(12*TB),(24+2*(N-24))/(12*TB))".replace(",", "\\,")
            with media_process([str(tool_path("ffmpeg")), "-v", "error", "-y", "-i", str(source),
                                "-vf", expression, "-fps_mode", "vfr", "-c:v", "libx264", str(variable)], Event()):
                pass
            rotated = directory / "portrait.mp4"
            with media_process([str(tool_path("ffmpeg")), "-v", "error", "-y", "-display_rotation", "90", "-i", str(variable),
                                "-c", "copy", str(rotated)], Event()):
                pass
            info = probe_video(rotated, Event())
            self.assertEqual((info.width, info.height), (360, 640))
            engine = VideoFrameAnalyzer(Path("models"), settings(), detector=EmptyDetector())
            result = analyze_video(info, settings(), Path("models"), Event(), lambda *a: None, analyzer=engine)
            try:
                output = export_video(result, directory / "output.mp4", EffectType.MOSAIC, 16, Event(), lambda *a: None)
                exported = probe_video(output, Event())
                self.assertFalse(exported.has_audio)
                original = list(decode_frames(info, Event()))
                processed = list(decode_frames(exported, Event()))
                self.assertEqual(len(original), len(processed))
                self.assertEqual(len(original), len(result.frames))
                for a, b in zip(original, processed):
                    self.assertLess(np.abs(a.astype(float) - b).mean(), 3)
            finally:
                result.close()


if __name__ == "__main__":
    unittest.main()
