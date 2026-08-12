from __future__ import annotations

import numpy as np

from auto_mosaic.domain import EffectType
from auto_mosaic.image_ops import (
    apply_effect,
    bounded_mask,
    center_anchored_component,
    refine_mask,
)


def test_mosaic_changes_only_masked_pixels() -> None:
    image = np.arange(24 * 24 * 3, dtype=np.uint8).reshape(24, 24, 3)
    mask = np.zeros((24, 24), dtype=bool)
    mask[6:18, 6:18] = True
    result = apply_effect(image, mask, EffectType.MOSAIC, 6)
    assert np.array_equal(result[~mask], image[~mask])
    assert np.any(result[mask] != image[mask])


def test_bounded_mask_clips_distant_pixels() -> None:
    mask = np.ones((100, 100), dtype=bool)
    clipped = bounded_mask(mask, (40, 40, 60, 60), margin_ratio=0.1)
    assert clipped[50, 50]
    assert not clipped[0, 0]


def test_refine_mask_expands_mask() -> None:
    mask = np.zeros((30, 30), dtype=bool)
    mask[15, 15] = True
    refined = refine_mask(mask, 2)
    assert int(refined.sum()) > 1


def test_center_anchored_component_discards_larger_unrelated_region() -> None:
    mask = np.zeros((80, 100), dtype=bool)
    mask[5:35, 5:35] = True
    mask[37:45, 47:55] = True
    selected = center_anchored_component(mask, (40, 30, 62, 52))
    assert selected[41, 51]
    assert not selected[20, 20]


def test_center_anchored_component_rejects_mask_outside_center_area() -> None:
    mask = np.zeros((80, 100), dtype=bool)
    mask[30:38, 40:48] = True
    selected = center_anchored_component(mask, (40, 30, 62, 52))
    assert not np.any(selected)
