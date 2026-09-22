"""Video inference, compressed frame cache, and reviewed-mask export; no UI state."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import cv2
import numpy as np

from auto_mosaic.detector import YoloOnnxDetector
from auto_mosaic.domain import Detection, EffectType, ProcessingSettings
from auto_mosaic.image_ops import apply_effect, bounded_mask
from auto_mosaic.mask_selection import select_detection_mask
from auto_mosaic.model_catalog import SAM_DECODER_FILENAME, SAM_ENCODER_FILENAME, detector_for
from auto_mosaic.segmenter import Sam2OnnxSegmenter
from auto_mosaic.video_io import (
    VideoInfo, check_cancel, decode_frames, encoder_arguments, media_process, mux_audio,
)


@dataclass(frozen=True)
class VideoAnalysisSettings:
    processing: ProcessingSettings
    contour_interval: int = 1

    def __post_init__(self):
        if self.contour_interval not in {1, 3, 5}:
            raise ValueError("輪郭更新間隔は1、3、5フレームから選択してください。")
        if not self.processing.targets:
            raise ValueError("検出対象を1つ以上選択してください。")


@dataclass(frozen=True)
class FrameSummary:
    count: int
    reused: int = 0
    fallbacks: int = 0


@dataclass(frozen=True)
class DetectionChange:
    frame: int
    before: int
    after: int

    @property
    def description(self) -> str:
        if self.before == 0:
            return "検出開始"
        if self.after == 0:
            return "検出途切れ"
        return "検出数の増加" if self.after > self.before else "検出数の減少"


def detection_changes(frames: list[FrameSummary]) -> list[DetectionChange]:
    changes = []
    before = 0
    for index, frame in enumerate(frames):
        if frame.count != before:
            changes.append(DetectionChange(index, before, frame.count))
        before = frame.count
    return changes


def _encode(extension: str, array: np.ndarray, parameters=()) -> bytes:
    ok, result = cv2.imencode(extension, array, list(parameters))
    if not ok:
        raise RuntimeError("解析キャッシュの圧縮に失敗しました。")
    return result.tobytes()


class FrameCache:
    """Disk-backed cache. Connections are local to each worker/UI operation."""
    def __init__(self):
        self.temporary = TemporaryDirectory(prefix="fy175-video-")
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "frames.sqlite"
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("CREATE TABLE frames (frame INTEGER PRIMARY KEY, mask BLOB, preview BLOB, detections TEXT)")

    def close(self):
        self.temporary.cleanup()

    @staticmethod
    def append(connection, index, image, mask, detections):
        height, width = image.shape[:2]
        scale = min(1.0, 960 / max(width, height))
        preview = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                             interpolation=cv2.INTER_AREA) if scale < 1 else image
        mask_bytes = _encode(".png", mask.astype(np.uint8) * 255, [cv2.IMWRITE_PNG_COMPRESSION, 3]) if mask.any() else None
        preview_bytes = _encode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
        boxes = json.dumps([(d.class_name, d.confidence, d.box) for d in detections])
        connection.execute("INSERT INTO frames VALUES (?, ?, ?, ?)", (index, mask_bytes, preview_bytes, boxes))

    @staticmethod
    def decode_mask(data, shape):
        if data is None:
            return np.zeros(shape, dtype=bool)
        mask = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != shape:
            raise RuntimeError("保存されたマスクを読み込めません。再解析してください。")
        return mask > 0

    def preview(self, index: int, info: VideoInfo):
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute("SELECT mask, preview, detections FROM frames WHERE frame=?", (index,)).fetchone()
        if row is None:
            raise RuntimeError("このフレームの解析結果がありません。")
        image = cv2.imdecode(np.frombuffer(row[1], np.uint8), cv2.IMREAD_COLOR)
        mask = self.decode_mask(row[0], (info.height, info.width))
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        sx, sy = image.shape[1] / info.width, image.shape[0] / info.height
        detections = [Detection(name, confidence, tuple(round(v * (sx if j % 2 == 0 else sy))
                      for j, v in enumerate(box))) for name, confidence, box in json.loads(row[2])]
        return image, mask, detections


@dataclass
class VideoAnalysis:
    info: VideoInfo
    settings: VideoAnalysisSettings
    cache: FrameCache
    frames: list[FrameSummary] = field(default_factory=list)
    elapsed: float = 0
    encoder_calls: int = 0
    complete: bool = False

    @property
    def duration(self):
        return len(self.frames) / float(self.info.fps)

    @property
    def changes(self):
        return detection_changes(self.frames)

    def close(self):
        self.cache.close()


@dataclass
class _Track:
    detection: Detection
    mask: np.ndarray
    age: int
    fallback: bool


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - intersection
    return intersection / max(1, union)


class VideoFrameAnalyzer:
    def __init__(self, model_dir: Path, settings: VideoAnalysisSettings, *, detector=None, segmenter=None):
        self.model_dir, self.settings = model_dir, settings
        self.detector, self.segmenter = detector, segmenter
        self.tracks: list[_Track] = []
        self.previous_small = None
        self.encoder_calls = 0

    def analyze(self, image, cancel: Event):
        check_cancel(cancel)
        settings = self.settings.processing
        if self.detector is None:
            spec = detector_for(settings.mode)
            self.detector = YoloOnnxDetector(self.model_dir / spec.filename, spec)
        # Detection is never skipped: even a one-frame dropout remains visible.
        # Below-threshold candidates are not masks and are not included.
        detections = [d for d in self.detector.detect(image, settings.targets, settings.confidence_threshold,
                                                     settings.iou_threshold)
                      if d.confidence >= settings.confidence_threshold]
        small = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        scene_cut = self.previous_small is not None and np.mean(np.abs(small - self.previous_small)) > 35
        self.previous_small = small
        previous = [] if scene_cut else self.tracks
        available = set(range(len(previous)))
        next_tracks = []
        embedding = None
        combined = np.zeros(image.shape[:2], dtype=bool)
        reused = fallbacks = 0
        for detection in detections:
            check_cancel(cancel)
            matches = [(i, _iou(detection.box, previous[i].detection.box)) for i in available
                       if previous[i].detection.class_name == detection.class_name]
            match, overlap = max(matches, key=lambda item: item[1], default=(-1, 0))
            old = previous[match] if overlap >= 0.65 else None
            if old is not None:
                available.remove(match)
            # Refresh after cuts, lost matches, large movement, or the interval.
            if old is not None and old.age + 1 < self.settings.contour_interval:
                a, b = old.detection.box, detection.box
                sx, sy = (b[2]-b[0]) / (a[2]-a[0]), (b[3]-b[1]) / (a[3]-a[1])
                transform = np.float32([[sx, 0, b[0] - sx*a[0]], [0, sy, b[1] - sy*a[1]]])
                mask = cv2.warpAffine(old.mask.astype(np.uint8), transform, (image.shape[1], image.shape[0]),
                                      flags=cv2.INTER_NEAREST) > 0
                mask = bounded_mask(mask, detection.box)
                fallback, age = old.fallback, old.age + 1
                reused += 1
            else:
                if self.segmenter is None:
                    self.segmenter = Sam2OnnxSegmenter(self.model_dir / SAM_ENCODER_FILENAME,
                                                       self.model_dir / SAM_DECODER_FILENAME)
                if embedding is None:
                    embedding = self.segmenter.encode(image)
                    self.encoder_calls += 1
                check_cancel(cancel)
                mask, fallback = select_detection_mask(
                    self.segmenter.mask_candidates_from_box(embedding, detection.box),
                    image.shape[:2], detection.box, settings.mask_expansion)
                age = 0
            combined |= mask
            fallbacks += int(fallback)
            next_tracks.append(_Track(detection, mask, age, fallback))
        self.tracks = next_tracks
        return combined, detections, FrameSummary(len(detections), reused, fallbacks)


def analyze_video(info: VideoInfo, settings: VideoAnalysisSettings, model_dir: Path,
                  cancel: Event, progress, *, analyzer=None) -> VideoAnalysis:
    info.verify_source()
    result = VideoAnalysis(info, settings, FrameCache())
    engine = analyzer or VideoFrameAnalyzer(model_dir, settings)
    started = time.perf_counter()
    try:
        with closing(sqlite3.connect(result.cache.path)) as connection, connection:
            for index, image in enumerate(decode_frames(info, cancel)):
                mask, detections, summary = engine.analyze(image, cancel)
                FrameCache.append(connection, index, image, mask, detections)
                result.frames.append(summary)
                if index % 30 == 0:
                    connection.commit()
                progress(index + 1, info.estimated_frames, time.perf_counter() - started)
        check_cancel(cancel)
        info.verify_source()
        if not result.frames:
            raise ValueError("動画からフレームを読み込めませんでした。")
        result.complete = True
        result.elapsed = time.perf_counter() - started
        result.encoder_calls = engine.encoder_calls
        return result
    except BaseException:
        result.close()
        raise


def export_video(analysis: VideoAnalysis, destination: Path, effect: EffectType, size: int,
                 cancel: Event, progress) -> Path:
    if not analysis.complete:
        raise ValueError("解析を完了してから書き出してください。")
    info = analysis.info
    info.verify_source()
    destination = destination.resolve()
    if destination == info.path or destination.exists():
        raise FileExistsError("既存ファイルへは上書きできません。別のファイル名を指定してください。")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("出力ファイルの拡張子は .mp4 を指定してください。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Only this operation's temporary directory is cleaned up, also on failure.
    with TemporaryDirectory(prefix=".fy175-export-", dir=destination.parent) as temporary:
        silent = Path(temporary) / "video.mp4"
        complete = Path(temporary) / "complete.mp4"
        with closing(sqlite3.connect(analysis.cache.path)) as connection:
            with media_process(encoder_arguments(info, silent), cancel, input_pipe=True) as encoder:
                count = 0
                for index, image in enumerate(decode_frames(info, cancel)):
                    row = connection.execute("SELECT mask FROM frames WHERE frame=?", (index,)).fetchone()
                    if row is None:
                        raise RuntimeError("動画と解析結果のフレーム数が一致しません。再解析してください。")
                    mask = FrameCache.decode_mask(row[0], image.shape[:2])
                    encoder.stdin.write(apply_effect(image, mask, effect, size).tobytes())
                    count += 1
                    progress(count, len(analysis.frames), time.perf_counter() - started)
                if count != len(analysis.frames):
                    raise RuntimeError("動画が途中で途切れました。書き出しを中止しました。")
        check_cancel(cancel)
        mux_audio(info, silent, complete, analysis.duration, cancel)
        check_cancel(cancel)
        info.verify_source()
        if destination.exists():
            raise FileExistsError("出力先にファイルが作成されたため書き出しを中止しました。")
        complete.rename(destination)
    return destination
