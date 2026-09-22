"""Small final source/packaging checks; does not mutate project content."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import numpy as np
from PIL import Image

from auto_mosaic.video_analysis import VideoAnalysisSettings, analyze_video
from auto_mosaic.video_io import probe_video
from auto_mosaic.domain import ImageMode, ProcessingSettings


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / ".codex-qa/2026-09-20"
    before = np.asarray(Image.open(evidence / "still-before.png").convert("RGB"))
    after = np.asarray(Image.open(evidence / "still-after-content.png").convert("RGB"))
    assert before.shape == after.shape, (before.shape, after.shape)
    difference = int(np.count_nonzero(before != after))
    print("STILL_CONTENT_PIXEL_DIFFERENCES", difference)
    assert difference == 0

    info = probe_video(evidence / "timeline-fixture.mp4", Event())
    config = VideoAnalysisSettings(ProcessingSettings(ImageMode.ILLUSTRATION, frozenset({"penis", "vagina"}), .25))
    analysis = analyze_video(info, config, root / "models", Event(), lambda *a: None)
    try:
        print("REAL_DETECTOR_VIDEO", json.dumps({"frames": len(analysis.frames), "seconds": analysis.elapsed,
              "max_masks": max(f.count for f in analysis.frames), "sam_calls": analysis.encoder_calls}))
    finally:
        analysis.close()

    for name in ("ffmpeg.exe", "ffprobe.exe"):
        path = root / "dist/FY175AutoMosaic/_internal/ffmpeg" / name
        assert path.is_file(), path
        print("PACKAGED_TOOL", name, path.stat().st_size)


if __name__ == "__main__":
    main()
