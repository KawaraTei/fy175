"""Mask selection shared by still images and video frames."""
from __future__ import annotations

import numpy as np

from auto_mosaic.image_ops import bounded_mask, box_mask, center_anchored_component, refine_mask


def select_detection_mask(candidates, shape, box, expansion: int) -> tuple[np.ndarray, bool]:
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
    return bounded_mask(refine_mask(candidate, expansion), box), False
