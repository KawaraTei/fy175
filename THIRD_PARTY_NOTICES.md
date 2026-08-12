# Third-party models and runtime

This prototype does not modify or claim ownership of the model weights below.

## NudeNet 320n

- Source: https://pypi.org/project/nudenet/
- Upstream: https://github.com/notAI-tech/NudeNet
- File: `320n.onnx`, renamed locally to `nudenet-320n.onnx`
- Package metadata license: MIT
- Classes used by this application: `MALE_GENITALIA_EXPOSED`, `FEMALE_GENITALIA_EXPOSED`

## DeepGHS anime censor detection

- Source: https://huggingface.co/deepghs/anime_censor_detection
- File: `censor_detect_v1.0_n/model.onnx`, renamed locally
- Model card license: MIT
- Labels used by this application: `penis`, `pussy`

## Segment Anything 2 ONNX

- Source: https://huggingface.co/vietanhdev/segment-anything-2-onnx-models
- Upstream: https://github.com/facebookresearch/sam2
- Files: `sam2_hiera_tiny.encoder.onnx`, `sam2_hiera_tiny.decoder.onnx`
- License stated by the ONNX model repository: Apache License 2.0

The SAM2 preprocessing and prompt handling in this project follows the public interface demonstrated by:

- https://github.com/vietanhdev/samexporter

## Runtime libraries

Runtime dependency names and pinned versions are listed in `requirements.txt`. Their respective license texts are included by PyInstaller where packages provide them; review package licenses again before any distribution.

The desktop UI uses PySide6 Essentials (Qt for Python), distributed under LGPL-3.0-only, GPL-2.0-only, or GPL-3.0-only terms. This folder build keeps the Qt libraries as separate dynamic libraries.
