from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


@dataclass(frozen=True)
class Download:
    filename: str
    url: str
    sha256: str | None = None


DOWNLOADS = (
    Download(
        "anime-censor-detect-v1.0-n.onnx",
        "https://huggingface.co/deepghs/anime_censor_detection/resolve/main/censor_detect_v1.0_n/model.onnx?download=true",
        "029de0a116f6c3c73bde62d2a8354c78664795579858f3c8e28fc1b4633a891c",
    ),
    Download(
        "sam2_hiera_tiny.encoder.onnx",
        "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_tiny.encoder.onnx?download=true",
        "4cc015ee18520e93f8c7ddfeaca7436039daaaaf19721b4b96a8810a805e82f7",
    ),
    Download(
        "sam2_hiera_tiny.decoder.onnx",
        "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_tiny.decoder.onnx?download=true",
        "f5a4bd656c143899fb7f52d64ed81e6f6aeb37d477a0b6da50146ac7cf2187bf",
    ),
)

NUDENET_FILENAME = "nudenet-320n.onnx"
NUDENET_SHA256 = "c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(item: Download) -> None:
    destination = MODEL_DIR / item.filename
    if destination.exists() and (item.sha256 is None or file_sha256(destination) == item.sha256):
        print(f"[skip] {item.filename}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"[download] {item.filename}")

    last_percent = -5

    def report(blocks: int, block_size: int, total: int) -> None:
        nonlocal last_percent
        if total > 0:
            percent = min(100, int(blocks * block_size * 100 / total))
            if percent >= last_percent + 5 or percent == 100:
                print(f"  {percent:3d}%")
                last_percent = percent

    urllib.request.urlretrieve(item.url, temporary, reporthook=report)
    if item.sha256 is not None:
        actual = file_sha256(temporary)
        if actual != item.sha256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {item.filename}: expected {item.sha256}, got {actual}"
            )
    temporary.replace(destination)


def copy_nudenet_model() -> None:
    destination = MODEL_DIR / NUDENET_FILENAME
    if destination.exists() and file_sha256(destination) == NUDENET_SHA256:
        print(f"[skip] {NUDENET_FILENAME}")
        return
    package = importlib.util.find_spec("nudenet")
    if package is None or package.origin is None:
        raise RuntimeError("nudenet package is not installed. Run pip install -r requirements.txt")
    source = Path(package.origin).resolve().parent / "320n.onnx"
    if not source.exists():
        raise RuntimeError(f"NudeNet model was not found: {source}")
    if file_sha256(source) != NUDENET_SHA256:
        raise RuntimeError("Bundled NudeNet model SHA-256 did not match the expected file")
    print(f"[copy] {NUDENET_FILENAME}")
    shutil.copy2(source, destination)


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        copy_nudenet_model()
        for item in DOWNLOADS:
            download(item)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("All models are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
