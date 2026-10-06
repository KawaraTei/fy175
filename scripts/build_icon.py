"""Package the source PNG into the Windows/Qt multi-resolution application icon."""
from pathlib import Path

from PIL import Image


def main() -> None:
    assets = Path(__file__).resolve().parents[1] / "assets"
    with Image.open(assets / "app-icon.png") as source:
        source.convert("RGBA").save(
            assets / "app.ico",
            format="ICO",
            sizes=[(size, size) for size in (16, 24, 32, 48, 64, 128, 256)],
        )


if __name__ == "__main__":
    main()
