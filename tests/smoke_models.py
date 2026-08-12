from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from auto_mosaic.detector import YoloOnnxDetector
from auto_mosaic.model_catalog import (
    ILLUSTRATION_DETECTOR,
    PHOTO_DETECTOR,
    SAM_DECODER_FILENAME,
    SAM_ENCODER_FILENAME,
)
from auto_mosaic.segmenter import Sam2OnnxSegmenter


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def synthetic_image() -> np.ndarray:
    image = np.zeros((360, 480, 3), dtype=np.uint8)
    image[:] = (35, 45, 70)
    cv2.ellipse(image, (240, 180), (100, 70), 0, 0, 360, (120, 180, 220), -1)
    cv2.rectangle(image, (190, 140), (290, 220), (210, 130, 90), -1)
    return image


def main() -> None:
    image = synthetic_image()
    for spec in (PHOTO_DETECTOR, ILLUSTRATION_DETECTOR):
        started = time.perf_counter()
        detector = YoloOnnxDetector(MODEL_DIR / spec.filename, spec)
        detections = detector.detect(
            image,
            frozenset({"penis", "vagina"}),
            confidence_threshold=0.95,
            iou_threshold=0.35,
        )
        elapsed = time.perf_counter() - started
        print(f"detector {spec.filename}: {len(detections)} detections, {elapsed:.2f}s")

    started = time.perf_counter()
    segmenter = Sam2OnnxSegmenter(
        MODEL_DIR / SAM_ENCODER_FILENAME,
        MODEL_DIR / SAM_DECODER_FILENAME,
    )
    embedding = segmenter.encode(image)
    mask = segmenter.mask_from_box(embedding, (180, 120, 300, 240))
    elapsed = time.perf_counter() - started
    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.bool_
    assert np.any(mask)
    print(f"SAM2 mask: {int(mask.sum())} pixels, {elapsed:.2f}s")


if __name__ == "__main__":
    main()
