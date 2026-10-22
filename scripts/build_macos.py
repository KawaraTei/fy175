"""Build a native-architecture .app on macOS (FFmpeg is installed externally)."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codesign-identity", default="-", help="Default: ad-hoc local signing")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("macOS builds must run on a Mac; cross-compilation from Windows is unsupported.")
    arch = platform.machine()
    if arch not in {"arm64", "x86_64"}:
        parser.error(f"Unsupported architecture: {arch}")
    from auto_mosaic.model_catalog import required_model_paths
    missing = [str(path) for path in required_model_paths(ROOT / "models") if not path.is_file()]
    if missing:
        parser.error("Run bash scripts/setup_macos.sh first. Missing models: " + ", ".join(missing))
    staging = ROOT / "build" / f"macos-{arch}" / "notices"
    staging.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for distribution in ("onnxruntime", "numpy", "Pillow", "opencv-python-headless",
                         "PySide6_Essentials", "shiboken6", "psdparse", "clipparse"):
        installed = importlib.metadata.distribution(distribution)
        inventory[distribution] = installed.version
        for entry in installed.files or []:
            if any(word in entry.name.lower() for word in ("license", "copying", "notice")):
                source = Path(installed.locate_file(entry))
                if source.is_file():
                    target = staging / distribution / Path(str(entry))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
    (staging / "BUILD.json").write_text(json.dumps({
        "macos": platform.mac_ver()[0], "architecture": arch,
        "python": platform.python_version(), "packages": inventory,
        "ffmpeg": "External installation; not bundled",
    }, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, "scripts/build_icon.py", "--macos"], cwd=ROOT, check=True)
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed", "--onedir",
               "--name", "FY175AutoMosaic", "--icon", "assets/app.icns",
               "--osx-bundle-identifier", "com.github.KawaraTei.fy175",
               "--target-arch", arch, "--codesign-identity", args.codesign_identity,
               "--distpath", f"dist/macos-{arch}", "--workpath", f"build/macos-{arch}/pyinstaller",
               "--specpath", f"build/macos-{arch}",
               "--add-data", f"{ROOT / 'models'}:models",
               "--add-data", f"{ROOT / 'assets' / 'app.ico'}:assets",
               "--add-data", f"{ROOT / 'LICENSES'}:LICENSES",
               "--add-data", f"{staging}:LICENSES/macos-python"]
    for filename in ("README.md", "MACOS.md", "LICENSE", "NOTICE", "MODEL_LICENSES.md",
                     "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_SOURCES.md", "AGENT_INTERFACE.md"):
        command += ["--add-data", f"{ROOT / filename}:docs"]
    command.append(str(ROOT / "auto_mosaic" / "app.py"))
    subprocess.run(command, cwd=ROOT, check=True)
    bundle = ROOT / "dist" / f"macos-{arch}" / "FY175AutoMosaic.app"
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(bundle)], check=True)
    print(f"Built: {bundle}")


if __name__ == "__main__":
    main()
