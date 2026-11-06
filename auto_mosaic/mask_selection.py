"""Mask selection shared by still images and video frames."""
from __future__ import annotations

import cv2
import numpy as np

from auto_mosaic.image_ops import bounded_mask, box_mask, center_anchored_component, refine_mask


def select_detection_mask(candidates, shape, box, expansion: int, smoothing: int = 0) -> tuple[np.ndarray, bool]:
    box_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    minimum_area = max(16, int(box_area * 0.04))
    maximum_area = int(box_area * 1.2)
    candidate = None
    best_score = float("-inf")
    for raw_mask, score in candidates:
        anchored = center_anchored_component(bounded_mask(raw_mask, box), box)
        area = int(anchored.sum())
        if minimum_area <= area <= maximum_area and score > best_score:
            candidate = anchored
            best_score = score
    if candidate is None:
        return box_mask(shape, box), True
    return bounded_mask(refine_mask(candidate, expansion, smoothing), box), False


def alternative_prompt_points(mask, box):
    """Sample up to four well-separated interiors outside the original mask."""
    inside = box_mask(mask.shape, box)
    remaining = inside & ~mask
    # A rectangle fallback has no reliable negative point.
    if not remaining.any():
        return [], None
    original = bounded_mask(mask, box)
    negative = None
    if original.any():
        distance = cv2.distanceTransform(original.astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(distance.argmax(), distance.shape)
        negative = (int(x), int(y))
    distance = cv2.distanceTransform(remaining.astype(np.uint8), cv2.DIST_L2, 5)
    points = []
    radius = max(4, min(box[2] - box[0], box[3] - box[1]) // 3)
    for _ in range(4):
        y, x = np.unravel_index(distance.argmax(), distance.shape)
        if distance[y, x] < 2:
            break
        points.append((int(x), int(y)))
        cv2.circle(distance, (int(x), int(y)), radius, 0, -1)
    return points, negative


def rank_alternative_masks(candidates, original, box, expansion, smoothing):
    """Retain distinct SAM regions, including disconnected visible parts."""
    minimum_area = max(16, int((box[2] - box[0]) * (box[3] - box[1]) * .04))
    result = [original.copy()]
    original_area = np.count_nonzero(original)
    rectangle_fallback = np.array_equal(original, box_mask(original.shape, box))
    for raw, score in sorted(candidates, key=lambda item: item[1], reverse=True):
        if not np.isfinite(score):
            continue
        mask = bounded_mask(raw, box)
        if mask.sum() < minimum_area:
            continue
        if (original_area and not rectangle_fallback
                and np.count_nonzero(mask & original) / original_area >= .5):
            continue
        mask = bounded_mask(refine_mask(mask, expansion, smoothing), box)
        if any(mask_iou(mask, previous) >= .9 for previous in result):
            continue
        result.append(mask)
        if len(result) == 7:
            break
    return result


def mask_iou(first, second):
    union = np.count_nonzero(first | second)
    return np.count_nonzero(first & second) / union if union else 1.0
