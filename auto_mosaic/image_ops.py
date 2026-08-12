from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from auto_mosaic.domain import Detection, EffectType


def load_image_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        rgb = np.asarray(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_image_bgr(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=95, subsampling=0)
    else:
        image.save(path, compress_level=4)


def apply_effect(
    image_bgr: np.ndarray, mask: np.ndarray, effect: EffectType, size: int
) -> np.ndarray:
    if not np.any(mask):
        return image_bgr.copy()
    if effect is EffectType.MOSAIC:
        height, width = image_bgr.shape[:2]
        reduced_width = max(1, int(np.ceil(width / max(1, size))))
        reduced_height = max(1, int(np.ceil(height / max(1, size))))
        reduced = cv2.resize(
            image_bgr, (reduced_width, reduced_height), interpolation=cv2.INTER_AREA
        )
        effected = cv2.resize(
            reduced, (width, height), interpolation=cv2.INTER_NEAREST
        )
    else:
        kernel = max(3, size * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        effected = cv2.GaussianBlur(image_bgr, (kernel, kernel), 0)

    result = image_bgr.copy()
    result[mask] = effected[mask]
    return result


def refine_mask(mask: np.ndarray, expansion: int) -> np.ndarray:
    binary = mask.astype(np.uint8) * 255
    if expansion > 0:
        size = expansion * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.dilate(binary, kernel, iterations=1)
    return binary > 0


def bounded_mask(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    margin_ratio: float = 0.12,
) -> np.ndarray:
    height, width = mask.shape
    x1, y1, x2, y2 = box
    margin_x = max(2, int((x2 - x1) * margin_ratio))
    margin_y = max(2, int((y2 - y1) * margin_ratio))
    limit = np.zeros_like(mask, dtype=bool)
    limit[
        max(0, y1 - margin_y) : min(height, y2 + margin_y),
        max(0, x1 - margin_x) : min(width, x2 + margin_x),
    ] = True
    return mask & limit


def center_anchored_component(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """Keep only the component attached to the center area of a detection box."""
    binary = mask.astype(np.uint8)
    component_count, labels = cv2.connectedComponents(binary, connectivity=8)
    if component_count <= 1:
        return np.zeros_like(mask, dtype=bool)

    height, width = mask.shape
    x1, y1, x2, y2 = box
    center_x = max(0, min(width - 1, int(round((x1 + x2 - 1) / 2))))
    center_y = max(0, min(height - 1, int(round((y1 + y2 - 1) / 2))))
    radius = max(2, int(round(min(x2 - x1, y2 - y1) * 0.08)))
    anchor_labels = labels[
        max(0, center_y - radius) : min(height, center_y + radius + 1),
        max(0, center_x - radius) : min(width, center_x + radius + 1),
    ]
    counts = np.bincount(anchor_labels.reshape(-1), minlength=component_count)
    counts[0] = 0
    selected_label = int(np.argmax(counts))
    if counts[selected_label] == 0:
        return np.zeros_like(mask, dtype=bool)
    return labels == selected_label


def box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = box
    result[y1:y2, x1:x2] = True
    return result


def visualize_detection(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    result = image_bgr.copy()
    overlay = result.copy()
    overlay[mask] = (40, 90, 230)
    result = cv2.addWeighted(result, 0.66, overlay, 0.34, 0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (40, 220, 255), 2)
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(result, (x1, y1), (x2, y2), (80, 220, 255), 2)
        label = f"{detection.class_name} {detection.confidence:.0%}"
        cv2.putText(
            result,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 240, 255),
            1,
            cv2.LINE_AA,
        )
    return result
