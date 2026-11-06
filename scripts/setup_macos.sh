#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != Darwin ]]; then
    echo "This setup script requires macOS." >&2
    exit 1
fi
PYTHON_COMMAND="${PYTHON:-python3.13}"
if [[ ! -x .venv/bin/python ]]; then
    "$PYTHON_COMMAND" -m venv .venv
fi
.venv/bin/python -c '
import platform, sys
if sys.version_info[:2] != (3, 13):
    raise SystemExit("Python 3.13 is required; create a fresh .venv with Python 3.13.")
arch = platform.machine()
if arch not in ("arm64", "x86_64"):
    raise SystemExit(f"Unsupported CPU architecture: {arch}")
minimum = 14 if arch == "arm64" else 13
if int(platform.mac_ver()[0].split(".")[0]) < minimum:
    raise SystemExit(f"macOS {minimum} or later is required on {arch}.")
print(f"macOS {platform.mac_ver()[0]} / {arch} / Python {platform.python_version()}")
'
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
.venv/bin/python scripts/download_models.py
echo 'Setup complete. Run: .venv/bin/python -m auto_mosaic.app'
echo 'Video mode also requires FFmpeg: brew install ffmpeg'
