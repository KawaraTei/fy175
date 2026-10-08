"""Text sidecars containing exact masks in original-image pixel coordinates."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, fields
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from auto_mosaic.domain import Detection, EffectType, ImageMode, PROCESSING_RANGES, ProcessingResult, ProcessingSettings
from auto_mosaic.image_ops import load_image_bgr


def sidecar_path(source: Path) -> Path:
    # Keep the image extension so sample.jpg and sample.png never share a mask.
    return source.with_name(source.name + ".fy")


def _fingerprint(source: Path) -> str:
    with source.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _encode_mask(mask: np.ndarray) -> list[list[int]]:
    flat = mask.astype(bool).ravel()
    edges = np.flatnonzero(np.diff(np.r_[False, flat, False]))
    return [[int(start), int(end - start)] for start, end in edges.reshape(-1, 2)]


def _decode_mask(runs, shape) -> np.ndarray:
    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    if not isinstance(runs, list):
        raise ValueError("マスク範囲の形式が不正です。")
    previous_end = 0
    for run in runs:
        if not isinstance(run, list) or len(run) != 2 or any(type(v) is not int for v in run):
            raise ValueError("マスク範囲の形式が不正です。")
        start, length = run
        if start < previous_end or length <= 0 or start + length > mask.size:
            raise ValueError("マスク範囲が画像の範囲外または重複しています。")
        mask[start:start + length] = True
        previous_end = start + length
    return mask.reshape(shape)


def _read_settings(data) -> ProcessingSettings:
    if not isinstance(data, dict):
        raise ValueError("設定の形式が不正です。")
    values = dict(data)
    if set(values) != {field.name for field in fields(ProcessingSettings)}:
        raise ValueError("保存された設定の項目が不正です。")
    values["mode"] = ImageMode(values["mode"])
    values["effect"] = EffectType(values["effect"])
    targets = values["targets"]
    if not isinstance(targets, list) or not targets or any(v not in {"penis", "vagina"} for v in targets):
        raise ValueError("検出対象が不正です。")
    values["targets"] = frozenset(targets)
    settings = ProcessingSettings(**values)
    if settings.iou_threshold != ProcessingSettings.iou_threshold:
        raise ValueError("対応していないIOU閾値です。")
    for name, (low, high, step) in PROCESSING_RANGES.items():
        value = getattr(settings, name)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("保存された設定値が範囲外です。")
        if abs(value / step - round(value / step)) > 1e-6:
            raise ValueError("保存された設定値の刻みが不正です。")
    if type(settings.effect_size) is not int or type(settings.mask_expansion) is not int:
        raise ValueError("サイズは整数で指定してください。")
    return settings


def save_sidecar(result: ProcessingResult, settings: ProcessingSettings) -> Path:
    source = result.source_path
    shape = load_image_bgr(source).shape[:2]
    if result.mask.shape != shape or any(mask.shape != shape for mask in result.detection_masks):
        raise ValueError("保存するマスクのサイズが元画像と一致しません。")
    values = asdict(settings)
    values["targets"] = sorted(settings.targets)
    data = {
        "format": "FY175AutoMosaic", "version": 1,
        "source_sha256": _fingerprint(source),
        "width": shape[1], "height": shape[0],
        "settings": values,
        "mask_encoding": "flat-runs-start-length",
        "mask": _encode_mask(result.mask),
        "detections": [asdict(detection) for detection in result.detections],
        "detection_masks": [_encode_mask(mask) for mask in result.detection_masks],
    }
    target = sidecar_path(source)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                prefix=target.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def load_sidecar(source: Path, *, sidecar_file: Path | None = None
                 ) -> tuple[np.ndarray, ProcessingSettings, list[Detection], list[np.ndarray]] | None:
    target = sidecar_file if sidecar_file is not None else sidecar_path(source)
    if not target.exists():
        if sidecar_file is not None:
            raise FileNotFoundError(f"復元する.fyファイルがありません: {target}")
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if (data["format"] != "FY175AutoMosaic" or data["version"] != 1
                or data["mask_encoding"] != "flat-runs-start-length"):
            raise ValueError("対応していない.fy形式です。")
        image_shape = load_image_bgr(source).shape[:2]
        if (data["height"], data["width"]) != image_shape or data["source_sha256"] != _fingerprint(source):
            raise ValueError("元画像が保存時と異なるため、マスクを復元できません。")
        settings = _read_settings(data["settings"])
        mask = _decode_mask(data["mask"], image_shape)
        detections = [Detection(item["class_name"], float(item["confidence"]), tuple(item["box"]))
                      for item in data["detections"]]
        owned = [_decode_mask(runs, image_shape) for runs in data["detection_masks"]]
        if owned and len(owned) != len(detections):
            raise ValueError("検出範囲とマスクの数が一致しません。")
        return mask, settings, detections, owned
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise ValueError(f"{target.name}を復元できません: {error}") from error
