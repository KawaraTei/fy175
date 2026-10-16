"""CPU/CUDA comparison with identical synthetic inputs; not an accuracy benchmark.

Run: python -m scripts.benchmark_inference --output .codex-qa/gpu/benchmark.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from auto_mosaic.detector import YoloOnnxDetector
from auto_mosaic.inference import DEVICE_ENV
from auto_mosaic.model_catalog import (
    PHOTO_DETECTOR, ILLUSTRATION_DETECTOR, SAM_ENCODER_FILENAME, SAM_DECODER_FILENAME,
)
from auto_mosaic.segmenter import Sam2OnnxSegmenter
from tests.smoke_models import synthetic_image


def measure(operation, repeats):
    result = operation()  # warm up; startup is measured separately
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        timings.append(time.perf_counter() - started)
    return result, {"median_ms": statistics.median(timings) * 1000, "seconds": timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    models = Path(__file__).resolve().parents[1] / "models"
    image = synthetic_image()
    report = {
        "method": "480x360 synthetic input, FP32, 1 warmup, sequential CPU then CUDA; excludes model initialization",
        "python": platform.python_version(), "onnxruntime": ort.__version__,
        "model_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in models.glob("*.onnx")},
        "runs": {}, "comparison": {},
    }
    outputs = {}
    previous_device = os.environ.get(DEVICE_ENV)
    try:
        for device in ("cpu", "cuda"):
            os.environ[DEVICE_ENV] = device
            run, outputs[device] = {}, {}
            for spec in (PHOTO_DETECTOR, ILLUSTRATION_DETECTOR):
                started = time.perf_counter()
                detector = YoloOnnxDetector(models / spec.filename, spec)
                startup = time.perf_counter() - started
                # Include preprocessing and postprocessing, as the app does.
                _, timing = measure(lambda: detector.detect(image, frozenset({"penis", "vagina"}), .1, .35), args.repeats)
                tensor = detector._prepare_input(image)[0]
                outputs[device][spec.filename] = detector.session.run(detector.output_names, {detector.input_name: tensor})[0]
                run[spec.filename] = {**timing, "startup_seconds": startup, "providers": detector.session.get_providers()}
                del detector
                gc.collect()
            started = time.perf_counter()
            segmenter = Sam2OnnxSegmenter(models / SAM_ENCODER_FILENAME, models / SAM_DECODER_FILENAME)
            startup = time.perf_counter() - started
            embedding, encode_timing = measure(lambda: segmenter.encode(image), args.repeats)
            candidates, decode_timing = measure(lambda: segmenter.mask_candidates_from_box(embedding, (180, 120, 300, 240)), args.repeats)
            points = segmenter.mask_candidates_from_point(embedding, (240, 180))
            outputs[device]["masks"] = candidates + points
            run["sam_encoder"] = {**encode_timing, "startup_seconds": startup, "providers": segmenter.encoder.get_providers()}
            run["sam_decoder"] = {**decode_timing, "providers": segmenter.decoder.get_providers()}
            report["runs"][device] = run
            print(json.dumps({device: run}), flush=True)
            del segmenter, embedding
            gc.collect()
        for name in report["runs"]["cpu"]:
            report["comparison"][name] = {"speedup": report["runs"]["cpu"][name]["median_ms"] / report["runs"]["cuda"][name]["median_ms"]}
        for spec in (PHOTO_DETECTOR, ILLUSTRATION_DETECTOR):
            diff = np.abs(outputs["cpu"][spec.filename] - outputs["cuda"][spec.filename])
            report["comparison"][spec.filename].update(max_abs_difference=float(diff.max()), mean_abs_difference=float(diff.mean()))
        report["mask_comparison"] = []
        for (cpu_mask, cpu_score), (gpu_mask, gpu_score) in zip(outputs["cpu"]["masks"], outputs["cuda"]["masks"], strict=True):
            union = np.count_nonzero(cpu_mask | gpu_mask)
            report["mask_comparison"].append({
                "iou": np.count_nonzero(cpu_mask & gpu_mask) / union if union else 1.,
                "different_pixels": int(np.count_nonzero(cpu_mask != gpu_mask)),
                "score_difference": abs(cpu_score - gpu_score),
            })
    finally:
        if previous_device is None:
            os.environ.pop(DEVICE_ENV, None)
        else:
            os.environ[DEVICE_ENV] = previous_device
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2))
    print(json.dumps(report["mask_comparison"], indent=2))


if __name__ == "__main__":
    main()
