from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from auto_mosaic.domain import ImageMode


@dataclass(frozen=True)
class DetectorSpec:
    filename: str
    class_names: tuple[str, ...]
    target_indices: dict[str, tuple[int, ...]]
    input_size: int
    centered_letterbox: bool = True
    padding_value: int = 114
    swap_red_blue: bool = True
    pad_before_resize: bool = False


PHOTO_DETECTOR = DetectorSpec(
    filename="nudenet-320n.onnx",
    class_names=(
        "FEMALE_GENITALIA_COVERED",
        "FACE_FEMALE",
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "ANUS_EXPOSED",
        "FEET_EXPOSED",
        "BELLY_COVERED",
        "FEET_COVERED",
        "ARMPITS_COVERED",
        "ARMPITS_EXPOSED",
        "FACE_MALE",
        "BELLY_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_COVERED",
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED",
    ),
    target_indices={"penis": (14,), "vagina": (4,)},
    input_size=320,
    centered_letterbox=False,
    padding_value=0,
    swap_red_blue=False,
    pad_before_resize=True,
)

ILLUSTRATION_DETECTOR = DetectorSpec(
    filename="anime-censor-detect-v1.0-n.onnx",
    class_names=("nipple_f", "penis", "pussy"),
    target_indices={"penis": (1,), "vagina": (2,)},
    input_size=640,
)

SAM_ENCODER_FILENAME = "sam2_hiera_tiny.encoder.onnx"
SAM_DECODER_FILENAME = "sam2_hiera_tiny.decoder.onnx"


def detector_for(mode: ImageMode) -> DetectorSpec:
    return PHOTO_DETECTOR if mode is ImageMode.PHOTO else ILLUSTRATION_DETECTOR


def required_model_paths(model_dir: Path) -> tuple[Path, ...]:
    return (
        model_dir / PHOTO_DETECTOR.filename,
        model_dir / ILLUSTRATION_DETECTOR.filename,
        model_dir / SAM_ENCODER_FILENAME,
        model_dir / SAM_DECODER_FILENAME,
    )
