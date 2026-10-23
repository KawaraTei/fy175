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
    original_bgra: np.ndarray

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
        from auto_mosaic.layer_preview import ClipDisplay, PsdDisplay
        self.display = (PsdDisplay(self, self._memory_psd()) if self.extension == ".psd"
                        else ClipDisplay(self, reader))
        self.layers = self._layer_infos(reader)
        self.original = (self._composite(reader) if self.extension == ".psd" else
                         self.composite_image(frozenset(info.index for info in self.layers if info.visible)))

    def _memory_psd(self):
        reader = psdparse.PSDFile()
        if not reader.load_bytes(self.snapshot.read_bytes()):
            raise ValueError("PSDの表示データを読み込めません。")
        return reader

    def close(self):
        self.display = None
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
        """PSD native composition. CLIP always uses the shared coordinate path."""
        pixels, stats = reader.composite()
        if any(stats.values()):
            raise ValueError("再現できないPSDの合成効果があります。該当する効果をラスタライズしてください。")
        return np.frombuffer(pixels, np.uint8).reshape(self.height, self.width, 4).copy()

    def _layer_infos(self, reader):
        infos = []
        self._ancestors = {}
        pending = [(index, (), True) for index in reader.children(-1)]
        while pending:
            index, ancestors, visible = pending.pop()
            layer = reader.layers[index]
            name = layer.name_unicode or f"レイヤー {index + 1}"
            shown = visible and layer.visible
            if layer.is_group:
                pending.extend((child, (*ancestors, index), shown) for child in reader.children(index))
                continue
            self._ancestors[index] = ancestors
            reason = ""
            if self.extension == ".clip":
                placement = self.display.placements.get(index)
                if placement is None:
                    reason = "画素なし"
                elif not placement.editable:
                    reason = "加工にはラスタライズが必要"
            elif layer.width <= 0 or layer.height <= 0:
                reason = "画素なし"
            elif self.extension == ".psd":
                if (layer.layer_type != psdparse.LayerType.NORMAL or layer.smart_object is not None
                        or layer.vector_mask is not None or layer.pixel_data_irrelevant):
                    reason = "加工にはラスタライズが必要"
            names = tuple(reader.layers[parent].name_unicode or f"レイヤー {parent + 1}" for parent in ancestors)
            infos.append(LayerInfo(index, " / ".join((*names, name)),
                                   len(ancestors), shown, not reason, reason))
        return infos

    def _set_visibility(self, reader, writer, indices):
        self._validate_indices(indices)
        if indices == frozenset(info.index for info in self.layers if info.visible):
            return  # Preserve original group/local flags when effective visibility is unchanged.
        enabled_groups = {parent for index in indices for parent in self._ancestors[index]}
        flags = {info.index: info.index in indices for info in self.layers}
        flags.update({parent: True for parent in enabled_groups})
        for index, visible in flags.items():
            if self.extension == ".psd":
                writer.layers[index].visible = visible
            else:
                writer.set_layer_attr(reader.layers[index].main_id, visible=int(visible))

    def composite_image(self, indices: frozenset[int]) -> np.ndarray:
        self._validate_indices(indices)
        reader = self._read(self.snapshot)
        if self.extension == ".psd":
            self._set_visibility(reader, reader, indices)
            return self._composite(reader)
        from auto_mosaic.layer_preview import ClipDisplay
        return ClipDisplay(self, reader, full_size=True).image(indices)

    def detection_image(self, indices: frozenset[int]) -> np.ndarray:
        return on_white(self.composite_image(indices))

    def _validate_indices(self, indices):
        if not indices <= {info.index for info in self.layers}:
            raise ValueError("存在しないレイヤーが選択されています。")

    def _edited_pixels(self, reader, mask, indices, settings):
        self._validate_indices(indices)
        if mask.shape != (self.height, self.width):
            raise ValueError("マスクのサイズがキャンバスと一致しません。")
        if any(not info.editable and info.index in indices for info in self.layers):
            raise ValueError("加工対象にはラスターレイヤーを選んでください。")
        if self.extension == ".clip":
            from auto_mosaic.clip_geometry import ClipReader, read_pixels
            with ClipReader(self.snapshot) as source:
                for index in sorted(indices):
                    placement = self.display.placements[index]
                    local_mask = placement.local_mask(mask)
                    if local_mask.any():
                        pixels = read_pixels(source, reader, index, placement)
                        yield index, effect_pixels(pixels, local_mask, settings)
            return
        for index in sorted(indices):
            layer = reader.layers[index]
            pixels = np.frombuffer(reader.layer_image(index, "image"), np.uint8).reshape(layer.height, layer.width, 4)
            local_mask = np.zeros((layer.height, layer.width), dtype=bool)
            x1, y1 = max(0, layer.left), max(0, layer.top)
            x2, y2 = min(self.width, layer.right), min(self.height, layer.bottom)
            if x2 <= x1 or y2 <= y1:
                continue
            local_mask[y1-layer.top:y2-layer.top, x1-layer.left:x2-layer.left] = mask[y1:y2, x1:x2]
            if local_mask.any():
                yield index, effect_pixels(pixels, local_mask, settings)

    def prepare(self, mask, indices, settings):
        """Prepare edited display pixels without writing a PSD/CLIP document."""
        from auto_mosaic.layer_preview import PreparedLayers, PsdDisplay
        reader = self._memory_psd() if self.extension == ".psd" else self._read(self.snapshot)
        changes = self._edited_pixels(reader, mask, indices, settings)
        if self.extension == ".psd":
            for index, pixels in changes:
                reader.set_layer_pixels(index, pixels.tobytes(), pixels.shape[1], pixels.shape[0])
            processed = PsdDisplay(self, reader)
        else:
            processed = self.display.with_pixels(changes)
        return PreparedLayers(self.display, processed)

    def render(self, mask: np.ndarray, indices: frozenset[int], settings: ProcessingSettings,
               visible_indices: frozenset[int] | None = None) -> LayerRender:
        self._validate_indices(indices)
        if mask.shape != (self.height, self.width):
            raise ValueError("マスクのサイズがキャンバスと一致しません。")
        if any(not info.editable and info.index in indices for info in self.layers):
            raise ValueError("加工対象にはラスターレイヤーを選んでください。")
        if self.extension == ".clip":
            return self._render_clip(mask, indices, settings, visible_indices)
        reader = writer = self._read(self.snapshot)
        original = self.original
        if visible_indices is not None:
            self._set_visibility(reader, writer, visible_indices)
            original = self.composite_image(visible_indices)
        for index, edited in self._edited_pixels(reader, mask, indices, settings):
            layer = reader.layers[index]
            writer.set_layer_pixels(index, edited.tobytes(), layer.width, layer.height)
        target = self.root / ("result" + self.extension)
        merged = self._composite(writer)
        writer.set_merged_image(merged.tobytes())
        if not writer.save(str(target)):
            raise OSError("PSDの保存に失敗しました。")
        return LayerRender(merged, target.read_bytes(), self.extension, original)

    def _render_clip(self, mask, indices, settings, visible_indices):
        from auto_mosaic.layer_preview import ClipDisplay
        visible = (frozenset(info.index for info in self.layers if info.visible)
                   if visible_indices is None else visible_indices)
        self._validate_indices(visible)
        reader = self._read(self.snapshot)
        original = ClipDisplay(self, reader, full_size=True).image(visible)
        writer = clipparse.ClipWriter()
        writer.load(str(self.snapshot))
        self._set_visibility(reader, writer, visible)
        for index, pixels in self._edited_pixels(reader, mask, indices, settings):
            # Write in the original offscreen's coordinates and dimensions.
            # The saved offsets and transform metadata remain untouched.
            writer.set_pixels(reader.layers[index].main_id, pixels.tobytes(), pixels.shape[1], pixels.shape[0])
        # Decode the written local rasters through the same geometry/compositor.
        # This temporary native file is export-only and avoids holding all edited
        # full-resolution layers in memory simultaneously.
        edited_path = self.root / "edited.clip"
        writer.save(str(edited_path))
        edited_reader = self._read(edited_path)
        merged = ClipDisplay(self, edited_reader, full_size=True, source_path=edited_path).image(visible)
        del edited_reader
        writer.set_canvas_preview(merged.tobytes(), self.width, self.height)
        target = self.root / "result.clip"
        writer.save(str(target))
        problems = clipparse.validate(str(target))
        if problems:
            raise ValueError("CLIP保存データの検証に失敗しました: " + "; ".join(problems))
        return LayerRender(merged, target.read_bytes(), self.extension, original)
