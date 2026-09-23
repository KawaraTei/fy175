"""Deterministic, non-sensitive media for video workflow tests and UI QA."""
from fractions import Fraction
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from auto_mosaic.domain import Detection
from auto_mosaic.video_io import VideoInfo, encoder_arguments, media_process, tool_path


class ShapeDetector:
    """Deliberately synthetic detector; never used by the application."""
    def __init__(self):
        self.calls = 0

    def detect(self, image, *_args):
        self.calls += 1
        green = ((image[:, :, 1].astype(float) > image[:, :, 0] * 1.4)
                 & (image[:, :, 1].astype(float) > image[:, :, 2] * 1.4)
                 & (image[:, :, 1] > 95)).astype(np.uint8)
        contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 100:
                result.append(Detection("penis", 0.92, (x, y, x + w, y + h)))
        return sorted(result, key=lambda item: item.box)


class ShapeSegmenter:
    def __init__(self):
        self.calls = 0

    def encode(self, image):
        self.calls += 1
        return image.shape[:2]

    def mask_candidates_from_box(self, shape, box, mask_threshold):
        self.mask_threshold = mask_threshold
        mask = np.zeros(shape, np.uint8)
        x1, y1, x2, y2 = box
        cv2.ellipse(mask, ((x1 + x2) // 2, (y1 + y2) // 2),
                    (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)), 0, 0, 360, 1, -1)
        return [(mask > 0, 0.98)]


def make_video(destination: Path, *, audio=True, frames=48, fps=12):
    width, height = 640, 360
    info = VideoInfo(destination, width, height, Fraction(fps), frames / fps, frames, False, 0, 0, 0)
    silent = destination.with_name(destination.stem + "-silent.mp4")
    cancel = Event()
    with media_process(encoder_arguments(info, silent), cancel, input_pipe=True) as process:
        for index in range(frames):
            y, x = np.indices((height, width))
            image = np.empty((height, width, 3), np.uint8)
            image[:, :, 0] = 38 + x // 12
            image[:, :, 1] = 30 + y // 15
            image[:, :, 2] = 45 + x // 16
            # 0 -> 1 -> 2 -> 0 (single-frame gap) -> 2 -> 1 -> 0.
            count = 0 if index < 6 or index == 24 or index >= 42 else 1 if index < 15 or index >= 33 else 2
            for target in range(count):
                cx, cy = 160 + index * 2 + target * 220, 180 + target * 30
                cv2.ellipse(image, (cx, cy), (46, 62), 0, 0, 360, (80, 205, 105), -1)
                for line in range(-30, 35, 10):
                    cv2.line(image, (cx - 20, cy + line), (cx + 20, cy + line + 6), (75, 145, 85), 2)
            cv2.putText(image, f"FRAME {index + 1:03d}", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 235), 1, cv2.LINE_AA)
            process.stdin.write(image.tobytes())
    if audio:
        with media_process([str(tool_path("ffmpeg")), "-v", "error", "-y", "-i", str(silent),
                            "-f", "lavfi", "-i", f"sine=frequency=440:duration={frames / fps}",
                            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", str(destination)], cancel):
            pass
        silent.unlink()
    else:
        silent.rename(destination)
    return destination
