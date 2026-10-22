"""Layer document snapshots and selective raster processing (PSD / CLIP)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import clipparse
import psdparse

from auto_mosaic.domain import ProcessingSettings


LAYER_EXTENSIONS = frozenset({".psd", ".clip"})


@dataclass(frozen=True)
class LayerInfo:
    index: int
    name: str
    depth: int
    visible: bool
    editable: bool
    reason: str


@dataclass
class LayerRender:
    bgra: np.ndarray
    document: bytes
    extension: str

    def save(self, path: Path) -> None:
        """Never overwrite an existing artwork, including the input."""
        suffix = path.suffix.lower()
        if suffix not in {self.extension, ".png", ".jpg", ".jpeg"}:
            raise ValueError("PNG・JPEGまたは元画像と同じレイヤー形式を選んでください。")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            try:
                if suffix == self.extension:
                    stream.write(self.document)
                elif suffix == ".png":
                    Image.fromarray(self.bgra[..., [2, 1, 0, 3]]).save(stream, format="PNG")
                else:
                    Image.fromarray(on_white(self.bgra)[..., ::-1]).save(stream, format="JPEG", quality=95)
            except Exception:
                stream.close()
                path.unlink(missing_ok=True)
                raise


def on_white(bgra: np.ndarray) -> np.ndarray:
    alpha = bgra[..., 3:4].astype(np.float32) / 255
    return np.rint(bgra[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)


def effect_pixels(bgra: np.ndarray, mask: np.ndarray, settings: ProcessingSettings) -> np.ndarray:
    """Filter premultiplied color; preserve exact alpha and all unmasked pixels."""
    from auto_mosaic.image_ops import apply_effect

    alpha = bgra[..., 3:4].astype(np.float32) / 255
    premultiplied = np.concatenate((bgra[..., :3].astype(np.float32) * alpha, alpha), axis=2)
    filtered = apply_effect(premultiplied, mask, settings.effect, settings.effect_size)
    color = np.divide(filtered[..., :3], filtered[..., 3:4],
                      out=np.zeros_like(filtered[..., :3]), where=filtered[..., 3:4] > 0)
    result = bgra.copy()
    selected = mask & (bgra[..., 3] > 0)
    result[selected, :3] = np.clip(np.rint(color[selected]), 0, 255).astype(np.uint8)
    return result


class LayerDocument:
    """Use on one worker only. The temporary snapshot keeps the original untouched."""

    def __init__(self, source: Path):
        self.source = source
        self.extension = source.suffix.lower()
        if self.extension not in LAYER_EXTENSIONS:
            raise ValueError("PSDまたはCLIPファイルを選んでください。")
        self._temporary = TemporaryDirectory(prefix="fy175-layers-")
        self.root = Path(self._temporary.name)
        self.snapshot = self.root / ("source" + self.extension)
        try:
            shutil.copyfile(source, self.snapshot)
            self._initialize()
        except Exception as error:
            message = str(error)
        else:
            return
        # Release native readers held by exception tracebacks before removing mapped files.
        self.close()
        raise ValueError(message)

    def _initialize(self):
        reader = self._read(self.snapshot)
        self.width, self.height = self._size(reader)
        if self.extension == ".psd" and (reader.header.depth != 8 or reader.header.mode != 3):
            raise ValueError("レイヤー加工は8bit RGBのPSDに対応しています。RGB・8bitへ変換してください。")
        self.layers = self._layer_infos(reader)
        self.original = self._composite(reader)

    def close(self):
        self._temporary.cleanup()

    def _read(self, path):
        reader = psdparse.PSDFile() if self.extension == ".psd" else clipparse.ClipFile()
        if not reader.load(str(path)):
            raise ValueError(f"レイヤー画像を読み込めません: {path.name}")
        return reader

    def _size(self, reader):
        header = reader.header if self.extension == ".psd" else reader
        return header.width, header.height

    def _composite(self, reader):
        if self.extension == ".psd":
            pixels, stats = reader.composite()
            if any(stats.values()):
                raise ValueError("再現できないPSDの合成効果があります。該当する効果をラスタライズしてください。")
        else:
            pixels = reader.merged_image()
        return np.frombuffer(pixels, np.uint8).reshape(self.height, self.width, 4).copy()

    def _layer_infos(self, reader):
        infos = []
        pending = [(index, (), True) for index in reader.children(-1)]
        while pending:
            index, ancestors, visible = pending.pop()
            layer = reader.layers[index]
            name = layer.name_unicode or f"レイヤー {index + 1}"
            shown = visible and layer.visible
            if layer.is_group:
                pending.extend((child, (*ancestors, name), shown) for child in reader.children(index))
                continue
            reason = ""
            if layer.width <= 0 or layer.height <= 0:
                reason = "画素なし"
            elif self.extension == ".psd":
                if (layer.layer_type != psdparse.LayerType.NORMAL or layer.smart_object is not None
                        or layer.vector_mask is not None or layer.pixel_data_irrelevant):
                    reason = "加工にはラスタライズが必要"
            elif layer.is_text or layer.is_filter:
                reason = "加工にはラスタライズが必要"
            elif (layer.left, layer.top, layer.width, layer.height) != (0, 0, self.width, self.height):
                reason = "加工にはキャンバス全体のラスターレイヤーが必要"
            infos.append(LayerInfo(index, " / ".join((*ancestors, name)),
                                   len(ancestors), shown, not reason, reason))
        return infos

    def detection_image(self, indices: frozenset[int]) -> np.ndarray:
        reader = self._read(self.snapshot)
        self._validate_indices(indices)
        if self.extension == ".psd":
            for info in self.layers:
                if info.index not in indices:
                    reader.layers[info.index].visible = False
            return on_white(self._composite(reader))
        writer = clipparse.ClipWriter()
        writer.load(str(self.snapshot))
        for info in self.layers:
            if info.index not in indices:
                writer.set_layer_attr(reader.layers[info.index].main_id, visible=0)
        target = self.root / "detection.clip"
        writer.save(str(target))
        return on_white(self._composite(self._read(target)))

    def _validate_indices(self, indices):
        if not indices <= {info.index for info in self.layers}:
            raise ValueError("存在しないレイヤーが選択されています。")

    def render(self, mask: np.ndarray, indices: frozenset[int], settings: ProcessingSettings) -> LayerRender:
        self._validate_indices(indices)
        if mask.shape != (self.height, self.width):
            raise ValueError("マスクのサイズがキャンバスと一致しません。")
        if any(not info.editable and info.index in indices for info in self.layers):
            raise ValueError("加工対象にはラスターレイヤーを選んでください。")
        reader = self._read(self.snapshot)
        writer = reader if self.extension == ".psd" else clipparse.ClipWriter()
        if self.extension == ".clip":
            writer.load(str(self.snapshot))
        for index in sorted(indices):
            layer = reader.layers[index]
            pixels = np.frombuffer(reader.layer_image(index, "image"), np.uint8).reshape(layer.height, layer.width, 4)
            local_mask = np.zeros((layer.height, layer.width), dtype=bool)
            x1, y1 = max(0, layer.left), max(0, layer.top)
            x2, y2 = min(self.width, layer.right), min(self.height, layer.bottom)
            if x2 <= x1 or y2 <= y1:
                continue
            local_mask[y1-layer.top:y2-layer.top, x1-layer.left:x2-layer.left] = mask[y1:y2, x1:x2]
            if not local_mask.any():
                continue
            edited = effect_pixels(pixels, local_mask, settings)
            if self.extension == ".psd":
                writer.set_layer_pixels(index, edited.tobytes(), layer.width, layer.height)
            else:
                writer.set_pixels(layer.main_id, edited.tobytes(), layer.width, layer.height)
        target = self.root / ("result" + self.extension)
        if self.extension == ".psd":
            merged = self._composite(writer)
            writer.set_merged_image(merged.tobytes())
            if not writer.save(str(target)):
                raise OSError("PSDの保存に失敗しました。")
        else:
            writer.save(str(target))
            merged = self._composite(self._read(target))
            writer = clipparse.ClipWriter()
            writer.load(str(target))
            writer.set_canvas_preview(merged.tobytes(), self.width, self.height)
            writer.save(str(target))
            problems = clipparse.validate(str(target))
            if problems:
                raise ValueError("CLIP保存データの検証に失敗しました: " + "; ".join(problems))
        return LayerRender(merged, target.read_bytes(), self.extension)
