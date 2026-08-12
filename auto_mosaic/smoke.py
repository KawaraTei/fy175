from __future__ import annotations

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


def run_model_smoke(model_dir: Path) -> None:
    image = np.zeros((360, 480, 3), dtype=np.uint8)
    image[:] = (35, 45, 70)
    cv2.ellipse(image, (240, 180), (100, 70), 0, 0, 360, (120, 180, 220), -1)
    cv2.rectangle(image, (190, 140), (290, 220), (210, 130, 90), -1)

    for spec in (PHOTO_DETECTOR, ILLUSTRATION_DETECTOR):
        detector = YoloOnnxDetector(model_dir / spec.filename, spec)
        detector.detect(
            image,
            frozenset({"penis", "vagina"}),
            confidence_threshold=0.95,
            iou_threshold=0.35,
        )

    segmenter = Sam2OnnxSegmenter(
        model_dir / SAM_ENCODER_FILENAME,
        model_dir / SAM_DECODER_FILENAME,
    )
    embedding = segmenter.encode(image)
    mask = segmenter.mask_from_box(embedding, (180, 120, 300, 240))
    if mask.shape != image.shape[:2] or not np.any(mask):
        raise RuntimeError("Packaged SAM2 inference did not produce a valid mask")
