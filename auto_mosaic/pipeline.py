from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np

from auto_mosaic.detector import YoloOnnxDetector
from auto_mosaic.domain import Detection, ImageMode, ProcessingResult, ProcessingSettings
from auto_mosaic.image_ops import (
    apply_effect,
    bounded_mask,
    box_mask,
    center_anchored_component,
    load_image_bgr,
    refine_mask,
    save_image_bgr,
)
from auto_mosaic.model_catalog import (
    SAM_DECODER_FILENAME,
    SAM_ENCODER_FILENAME,
    detector_for,
)
from auto_mosaic.segmenter import Sam2OnnxSegmenter


class MosaicPipeline:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._detectors: dict[ImageMode, YoloOnnxDetector] = {}
        self._segmenter: Sam2OnnxSegmenter | None = None
        self._lock = Lock()

    def analyze(self, path: Path, settings: ProcessingSettings) -> ProcessingResult:
        with self._lock:
            image = load_image_bgr(path)
            detector = self._get_detector(settings.mode)
            detections = detector.detect(
                image,
                settings.targets,
                settings.confidence_threshold,
                settings.iou_threshold,
            )
            combined_mask = np.zeros(image.shape[:2], dtype=bool)
            fallback_count = 0
            if detections:
                segmenter = self._get_segmenter()
                embedding = segmenter.encode(image)
                for detection in detections:
                    box_area = max(
                        1,
                        (detection.box[2] - detection.box[0])
                        * (detection.box[3] - detection.box[1]),
                    )
                    minimum_area = max(16, int(box_area * 0.04))
                    maximum_area = int(box_area * 1.2)
                    candidate = None
                    best_score = float("-inf")
                    for raw_mask, score in segmenter.mask_candidates_from_box(
                        embedding, detection.box
                    ):
                        anchored = center_anchored_component(
                            bounded_mask(raw_mask, detection.box), detection.box
                        )
                        area = int(anchored.sum())
                        if minimum_area <= area <= maximum_area and score > best_score:
                            candidate = anchored
                            best_score = score

                    if candidate is None:
                        candidate = box_mask(image.shape[:2], detection.box)
                        fallback_count += 1
                    else:
                        candidate = refine_mask(candidate, settings.mask_expansion)
                        candidate = bounded_mask(candidate, detection.box)
                    combined_mask |= candidate

            effected = apply_effect(
                image, combined_mask, settings.effect, settings.effect_size
            )
            return ProcessingResult(
                source_path=path,
                image_bgr=effected,
                mask=combined_mask,
                detections=detections,
                used_box_fallbacks=fallback_count,
            )

    def save(
        self,
        result: ProcessingResult,
        output_dir: Path,
        filename_suffix: str = "_mosaic",
    ) -> Path:
        extension = result.source_path.suffix.lower()
        if extension not in {".png", ".jpg", ".jpeg"}:
            extension = ".png"
        output_stem = f"{result.source_path.stem}{filename_suffix}"
        output_path = output_dir / f"{output_stem}{extension}"
        sequence = 2
        while output_path.exists():
            output_path = output_dir / f"{output_stem}_{sequence}{extension}"
            sequence += 1
        save_image_bgr(output_path, result.image_bgr)
        return output_path

    def process_with_mask(
        self,
        path: Path,
        mask: np.ndarray,
        settings: ProcessingSettings,
        detections: list[Detection],
        used_box_fallbacks: int = 0,
    ) -> ProcessingResult:
        image = load_image_bgr(path)
        if mask.shape != image.shape[:2]:
            raise ValueError("編集中のマスクサイズが画像と一致しません。")
        edited_mask = mask.astype(bool, copy=True)
        return ProcessingResult(
            source_path=path,
            image_bgr=apply_effect(
                image, edited_mask, settings.effect, settings.effect_size
            ),
            mask=edited_mask,
            detections=list(detections),
            used_box_fallbacks=used_box_fallbacks,
        )

    def _get_detector(self, mode: ImageMode) -> YoloOnnxDetector:
        if mode not in self._detectors:
            spec = detector_for(mode)
            path = self.model_dir / spec.filename
            if not path.exists():
                raise FileNotFoundError(f"検出モデルがありません: {path}")
            self._detectors[mode] = YoloOnnxDetector(path, spec)
        return self._detectors[mode]

    def _get_segmenter(self) -> Sam2OnnxSegmenter:
        if self._segmenter is None:
            encoder = self.model_dir / SAM_ENCODER_FILENAME
            decoder = self.model_dir / SAM_DECODER_FILENAME
            if not encoder.exists() or not decoder.exists():
                raise FileNotFoundError(
                    "SAM2モデルがありません。scripts/download_models.pyを実行してください。"
                )
            self._segmenter = Sam2OnnxSegmenter(encoder, decoder)
        return self._segmenter
