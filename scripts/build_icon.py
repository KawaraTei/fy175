"""Package the source PNG into Windows/Qt and macOS application icons."""
from pathlib import Path

from PIL import Image


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--macos", action="store_true")
    args = parser.parse_args()
    assets = Path(__file__).resolve().parents[1] / "assets"
    with Image.open(assets / "app-icon.png") as source:
        source.convert("RGBA").save(
            assets / "app.ico",
            format="ICO",
            sizes=[(size, size) for size in (16, 24, 32, 48, 64, 128, 256)],
        )
        if args.macos:
            source.convert("RGBA").save(assets / "app.icns", format="ICNS")


if __name__ == "__main__":
    main()
