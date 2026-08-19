from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from auto_mosaic.domain import (
    Detection,
    EffectType,
    ImageMode,
    ProcessingResult,
    ProcessingSettings,
)
from auto_mosaic.image_ops import load_image_bgr
from auto_mosaic.pipeline import MosaicPipeline


def _result(source: Path) -> ProcessingResult:
    return ProcessingResult(
        source_path=source,
        image_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
        mask=np.zeros((4, 4), dtype=bool),
    )


def test_save_uses_custom_suffix() -> None:
    with TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        pipeline = MosaicPipeline(Path("models"))
        saved = pipeline.save(_result(Path("sample.png")), output_dir, "_hidden")
        assert saved.name == "sample_hidden.png"
        assert saved.exists()


def test_save_allows_empty_suffix_without_overwrite() -> None:
    with TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        (output_dir / "sample.png").write_bytes(b"existing")
        pipeline = MosaicPipeline(Path("models"))
        saved = pipeline.save(_result(Path("sample.png")), output_dir, "")
        assert saved.name == "sample_2.png"
        assert (output_dir / "sample.png").read_bytes() == b"existing"


def test_process_with_mask_uses_edited_mask_without_detection() -> None:
    with TemporaryDirectory() as temporary:
        source = Path(temporary) / "source.png"
        pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        Image.fromarray(pixels, "RGB").save(source)
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        detection = Detection("penis", 0.9, (8, 8, 24, 24))
        pipeline = MosaicPipeline(Path("models"))
        result = pipeline.process_with_mask(
            source,
            mask,
            ProcessingSettings(
                mode=ImageMode.ILLUSTRATION,
                targets=frozenset({"penis"}),
                confidence_threshold=0.25,
                effect=EffectType.MOSAIC,
                effect_size=8,
            ),
            [detection],
        )
        original = load_image_bgr(source)
        assert np.array_equal(result.mask, mask)
        assert result.mask is not mask
        assert result.detections == [detection]
        assert np.array_equal(result.image_bgr[~mask], original[~mask])
        assert np.any(result.image_bgr[mask] != original[mask])


class _FakeDetector:
    def detect(self, *_args):
        return [Detection("penis", 0.9, (20, 20, 40, 40))]


class _FakeSegmenter:
    def __init__(self, candidates):
        self.candidates = candidates

    def encode(self, _image):
        return object()

    def mask_candidates_from_box(self, _embedding, _box):
        return self.candidates


def _analyze_with_candidates(candidates):
    with TemporaryDirectory() as temporary:
        source = Path(temporary) / "source.png"
        Image.new("RGB", (64, 64), (120, 100, 80)).save(source)
        pipeline = MosaicPipeline(Path("models"))
        pipeline._get_detector = lambda _mode: _FakeDetector()  # type: ignore[method-assign]
        pipeline._get_segmenter = lambda: _FakeSegmenter(candidates)  # type: ignore[method-assign]
        return pipeline.analyze(
            source,
            ProcessingSettings(
                mode=ImageMode.PHOTO,
                targets=frozenset({"penis"}),
                confidence_threshold=0.25,
                mask_expansion=0,
            ),
        )


def test_analyze_prefers_centered_mask_over_higher_scored_unrelated_mask() -> None:
    unrelated = np.zeros((64, 64), dtype=bool)
    unrelated[21:27, 21:27] = True
    centered = np.zeros((64, 64), dtype=bool)
    centered[28:34, 28:34] = True
    result = _analyze_with_candidates([(unrelated, 0.99), (centered, 0.5)])
    assert result.mask[30, 30]
    assert not result.mask[23, 23]
    assert result.used_box_fallbacks == 0


def test_analyze_falls_back_to_detection_box_when_no_mask_is_centered() -> None:
    unrelated = np.zeros((64, 64), dtype=bool)
    unrelated[21:27, 21:27] = True
    result = _analyze_with_candidates([(unrelated, 0.99)])
    assert result.mask[30, 30]
    assert result.mask[23, 23]
    assert not result.mask[19, 19]
    assert result.used_box_fallbacks == 1
