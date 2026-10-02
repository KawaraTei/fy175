from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


class ImageMode(str, Enum):
    PHOTO = "photo"
    ILLUSTRATION = "illustration"


class EffectType(str, Enum):
    MOSAIC = "mosaic"
    BLUR = "blur"


PREVIEW_VIEWS = {"original": "元画像", "detection": "検出範囲", "mask_overlay": "マスク範囲", "result": "処理結果"}


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class ProcessingSettings:
    mode: ImageMode
    targets: frozenset[str]
    confidence_threshold: float
    iou_threshold: float = 0.35
    effect: EffectType = EffectType.MOSAIC
    effect_size: int = 16
    mask_expansion: int = 3
    mask_threshold: float = 0.0


@dataclass
class ProcessingResult:
    source_path: Path
    image_bgr: np.ndarray
    mask: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    below_threshold_detections: list[Detection] = field(default_factory=list)
    used_box_fallbacks: int = 0
    # Index matches detections; each mask retains overlap with other detections.
    detection_masks: list[np.ndarray] = field(default_factory=list)
