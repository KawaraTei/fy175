"""Short CPU comparison on a synthetic clip; not a real-world accuracy benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Event

from auto_mosaic.model_catalog import SAM_DECODER_FILENAME, SAM_ENCODER_FILENAME
from auto_mosaic.segmenter import Sam2OnnxSegmenter
from auto_mosaic.video_analysis import VideoFrameAnalyzer, analyze_video, export_video
from auto_mosaic.video_io import probe_video
from auto_mosaic.domain import EffectType
from tests.test_video import settings
from tests.video_fixtures import ShapeDetector, make_video


def main():
    import time
    destination = Path(".codex-qa/2026-09-20")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "benchmark-fixture.mp4"
    make_video(path)
    info = probe_video(path, Event())
    models = Path("models")
    segmenter = Sam2OnnxSegmenter(models / SAM_ENCODER_FILENAME, models / SAM_DECODER_FILENAME)
    results = []
    for interval in (1, 3, 5):
        config = settings(interval)
        engine = VideoFrameAnalyzer(models, config, detector=ShapeDetector(), segmenter=segmenter)
        analysis = analyze_video(info, config, models, Event(), lambda *a: None, analyzer=engine)
        try:
            record = {"interval": interval, "frames": len(analysis.frames), "seconds": analysis.elapsed,
                      "fps": len(analysis.frames) / analysis.elapsed, "sam_encoder_calls": analysis.encoder_calls,
                      "reused_masks": sum(f.reused for f in analysis.frames),
                      "fallback_masks": sum(f.fallbacks for f in analysis.frames),
                      "counts": [f.count for f in analysis.frames]}
            output = destination / f"benchmark-output-{interval}.mp4"
            # This fixture output is deliberately named; do not replace existing files.
            if not output.exists():
                started = time.perf_counter()
                export_video(analysis, output, EffectType.MOSAIC, 24, Event(), lambda *a: None)
                record["export_seconds"] = time.perf_counter() - started
            results.append(record)
            print(json.dumps(record), flush=True)
        finally:
            analysis.close()
    (destination / "benchmark.json").write_text(json.dumps({
        "method": "640x360 12fps 4s synthetic clip, synthetic detector, real CPU SAM2 tiny; model initialization excluded",
        "runs": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
