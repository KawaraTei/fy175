"""Small packaged-platform check: native bindings, PSD roundtrip and FFmpeg I/O."""
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import subprocess

import numpy as np

from auto_mosaic.domain import ImageMode, ProcessingSettings
from auto_mosaic.layers import LayerDocument
from auto_mosaic.video_io import decode_frames, probe_video, tool_path


def run_platform_smoke():
    import psdparse
    import clipparse

    assert clipparse.ClipFile() is not None
    with TemporaryDirectory(prefix="fy175-smoke-") as temporary:
        root = Path(temporary)
        source = root / "sample.psd"
        pixels = np.zeros((32, 32, 4), np.uint8)
        pixels[..., 3] = 255
        pixels[::2, ::2, :3] = 255
        psd = psdparse.PSDFile()
        psd.create_blank(32, 32)
        psd.add_layer("test", 0, 0, pixels.tobytes(), 32, 32)
        psd.set_merged_image(psd.composite()[0])
        assert psd.save(str(source))
        del psd
        document = LayerDocument(source)
        try:
            result = document.render(np.ones((32, 32), bool), frozenset({0}),
                                     ProcessingSettings(ImageMode.ILLUSTRATION, frozenset({"penis"}), .25))
            result.save(root / "processed.psd")
            assert np.any(result.bgra != pixels)
        finally:
            document.close()
        video = root / "sample.mp4"
        subprocess.run([str(tool_path("ffmpeg")), "-v", "error", "-f", "lavfi", "-i",
                        "color=c=blue:s=64x48:r=3:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
                       check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        info = probe_video(video, Event())
        frames = list(decode_frames(info, Event()))
        assert len(frames) == 3
    print("PACKAGED_PLATFORM_SMOKE_OK", flush=True)
