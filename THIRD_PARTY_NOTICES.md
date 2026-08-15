# Third-Party Notices

This document records the third-party software and model files used by
FY175AutoMosaic. Copyright in each component remains with its respective owner.

The license of FY175AutoMosaic itself is GNU Affero General Public License version
3 only (`AGPL-3.0-only`). See [`LICENSE`](LICENSE). Third-party components keep
their own licenses; the application license does not replace them.

## Runtime software

| Component | Version | License used for this distribution | License / notice file |
| --- | ---: | --- | --- |
| Python | 3.13 | Python Software Foundation License | `LICENSES/Python-3.13.txt` |
| NumPy | 2.2.6 | BSD-3-Clause and bundled component licenses | `LICENSES/NumPy-2.2.6.txt` |
| ONNX Runtime | 1.28.0 | MIT; bundled third-party components have their own terms | `LICENSES/ONNX-Runtime-1.28.0.txt`, `LICENSES/ONNX-Runtime-ThirdPartyNotices-1.28.0.txt` |
| OpenCV / opencv-python-headless | 4.12.0.88 | Apache-2.0; bundled third-party components have their own terms | `LICENSES/OpenCV-4.12.0.txt`, `LICENSES/OpenCV-ThirdParty-4.12.0.txt` |
| Pillow | 11.3.0 | HPND and bundled component licenses | `LICENSES/Pillow-11.3.0.txt` |
| PySide6 Essentials | 6.11.1 | LGPL-3.0-only (the open-source option selected here) | `LICENSES/LGPL-3.0.txt`, `LICENSES/GPL-3.0.txt` |
| shiboken6 | 6.11.1 | LGPL-3.0-only (the open-source option selected here) | `LICENSES/LGPL-3.0.txt`, `LICENSES/GPL-3.0.txt` |
| PyInstaller | 6.15.0 | GPL-2.0-or-later with the PyInstaller bootloader exception | `LICENSES/PyInstaller-6.15.0.txt` |

The Windows folder build keeps the Qt libraries as separate DLL files. Do not
merge or statically link them without reviewing the resulting LGPL obligations.
Recipients must be allowed to replace/relink the LGPL libraries and to reverse
engineer the combined work for debugging modifications to those libraries.

Qt contains third-party software in addition to Qt's own code. The applicable
notices depend on the exact Qt modules included in a build. The current build
uses Qt Core, GUI, Network, SVG, and Widgets. Before publishing a binary release,
retain the generated `LICENSES` directory and verify the corresponding entries
in Qt's official list:

- https://doc.qt.io/qt-6/licenses-used-in-qt.html
- https://doc.qt.io/qt-6/qtmodules.html

The exact pinned Python dependencies are listed in `requirements.txt`.

## Models and model tooling

| File / component | Source | Declared or applicable terms | Included text |
| --- | --- | --- | --- |
| `nudenet-320n.onnx` | [NudeNet](https://github.com/notAI-tech/NudeNet) | The model metadata and NudeNet v3 repository license identify AGPL-3.0. PyPI metadata currently says MIT; this project follows the stricter AGPL notice. | `LICENSES/NudeNet-AGPL-3.0.txt` |
| `anime-censor-detect-v1.0-n.onnx` | [DeepGHS anime censor detection](https://huggingface.co/deepghs/anime_censor_detection) | Model card: MIT. The serialized training metadata identifies Ultralytics YOLOv8; this project is therefore distributed under AGPL-3.0-only as the conservative compatible choice. | `LICENSE`, `LICENSES/DeepGHS-Model-MIT-NOTICE.md` |
| `sam2_hiera_tiny.encoder.onnx`, `sam2_hiera_tiny.decoder.onnx` | [Segment Anything 2 ONNX models](https://huggingface.co/vietanhdev/segment-anything-2-onnx-models) | Apache-2.0 | `LICENSES/Apache-2.0.txt` |
| SAM2 export/interface reference | [SAM Exporter](https://github.com/vietanhdev/samexporter) | MIT | `LICENSES/SAMExporter-MIT.txt` |

See [`MODEL_LICENSES.md`](MODEL_LICENSES.md) for model checksums, the upstream
metadata conflict, and rules for replacing or redistributing model files.

## Source availability

FY175AutoMosaic is intended to be published with its complete corresponding source
code. If you redistribute a binary, provide recipients with the exact source
for that binary and preserve all license and notice files. See
[`DISTRIBUTION.md`](DISTRIBUTION.md) for the release checklist.
