# Model Sources and Licenses

FY175AutoMosaic does not train or own the model weights listed below. The download
script retrieves fixed files and verifies their SHA-256 values. Model weights
are intentionally excluded from the Git repository by `.gitignore`.

## Model inventory

| Local file | Upstream file | SHA-256 | Terms followed by this project |
| --- | --- | --- | --- |
| `nudenet-320n.onnx` | NudeNet package `320n.onnx` | `c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f` | AGPL-3.0 |
| `anime-censor-detect-v1.0-n.onnx` | `deepghs/anime_censor_detection`, `censor_detect_v1.0_n/model.onnx` | `029de0a116f6c3c73bde62d2a8354c78664795579858f3c8e28fc1b4633a891c` | MIT model-card notice plus conservative AGPL-3.0 treatment for the Ultralytics YOLOv8 lineage |
| `sam2_hiera_tiny.encoder.onnx` | `vietanhdev/segment-anything-2-onnx-models` | `4cc015ee18520e93f8c7ddfeaca7436039daaaaf19721b4b96a8810a805e82f7` | Apache-2.0 |
| `sam2_hiera_tiny.decoder.onnx` | `vietanhdev/segment-anything-2-onnx-models` | `f5a4bd656c143899fb7f52d64ed81e6f6aeb37d477a0b6da50146ac7cf2187bf` | Apache-2.0 |

The authoritative download URLs are in `scripts/download_models.py`.

## NudeNet metadata conflict

The NudeNet PyPI page currently labels the package as MIT. In contrast, the
NudeNet v3 repository contains an AGPL-3.0 license, and the bundled ONNX file's
embedded metadata says `license: AGPL-3.0 License` and identifies Ultralytics.
For publication, this project preserves the repository's AGPL text and treats
the weight as AGPL-3.0. This avoids relying on the less restrictive PyPI label
while the upstream metadata is inconsistent.

Sources:

- https://pypi.org/project/nudenet/
- https://github.com/notAI-tech/NudeNet
- `LICENSES/NudeNet-AGPL-3.0.txt`

## DeepGHS / Ultralytics qualification

The DeepGHS model card declares MIT, but the model and training metadata identify
Ultralytics YOLOv8. Ultralytics states that its trained models are provided under
AGPL-3.0 by default unless an enterprise license applies. Because the respective
scope of those notices is not completely resolved by the model card, this
repository uses AGPL-3.0-only for the complete application and preserves the MIT
declaration as well. This is a conservative publication policy, not a legal
determination that the model card is invalid.

Sources:

- https://huggingface.co/deepghs/anime_censor_detection
- https://www.ultralytics.com/license
- `LICENSES/DeepGHS-Model-MIT-NOTICE.md`

## SAM2 ONNX

The ONNX model repository declares Apache-2.0. The project uses the public
preprocessing and prompt interface demonstrated by SAM Exporter, whose license
is MIT. The corresponding texts are stored as:

- `LICENSES/Apache-2.0.txt`
- `LICENSES/SAMExporter-MIT.txt`

## Redistribution rules

When publishing a source repository without model binaries, retain this document,
the download script, checksums, and all referenced license texts.

When publishing binaries that include the weights:

1. Distribute the entire application under AGPL-3.0-only unless you have obtained
   different rights from every relevant rights holder.
2. Include `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, this document, and the
   complete `LICENSES` directory.
3. Provide the exact corresponding application source for the binary.
4. Do not remove model metadata or upstream attribution.
5. Re-check upstream model cards and licenses whenever a model file or checksum
   changes.

If a future model has unclear, non-commercial, research-only, or custom terms,
do not add it to a public release until its redistribution rights are confirmed.
