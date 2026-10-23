"""CLIP raster coordinates shared by preview, analysis and export.

clipparse 0.2.0 does not expose the SQL layer/mask offsets. Read those
alongside its pixels; never rewrite or bake them into the saved document.
Format references: animeops/clip-tools (ClipLayer.bbox and resizable image
attributes), dobrokot/clip_to_psd (independent mask placement).
"""
from dataclasses import dataclass
import mmap
import sqlite3
import struct

import cv2
import numpy as np
from clipparse_tools import clip_lazy_demo as reference


class ClipReader(reference.ClipFile):
    """Reference block decoder with a read-only mapping and in-memory SQL."""
    def __init__(self, path):
        self.stream = open(path, "rb")
        self.raw = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        self.db = sqlite3.connect(":memory:")
        try:
            header = struct.unpack_from(">Q", self.raw, 16)[0]
            offset = struct.unpack_from(">Q", self.raw, header + 24)[0]
            kind, length = struct.unpack_from(">8sQ", self.raw, offset)
            if kind != b"CHNKSQLi":
                raise ValueError("CLIPのメタデータを読み込めません。")
            self.db.deserialize(self.raw[offset + 16:offset + 16 + length])
            self.cur = self.db.cursor()
            self.ext_offset = {reference.as_str(e): o for e, o in
                               self.cur.execute("SELECT ExternalID, Offset FROM ExternalChunk")}
            self.attr_cache = {}
            query = self.cur.execute("SELECT * FROM Layer")
            names = [column[0] for column in query.description]
            self.rows = {row["MainId"]: row for row in
                         (dict(zip(names, values)) for values in query.fetchall())}
        except Exception:
            self.close()
            raise

    def close(self):
        self.db.close()
        self.raw.close()
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def offset(row, *prefixes):
    return tuple(sum(int(row.get(prefix + axis) or 0) for prefix in prefixes) for axis in "XY")


@dataclass(frozen=True)
class RasterPlacement:
    origin: tuple[int, int]
    size: tuple[int, int]
    offscreen: int
    editable: bool = False
    transform: np.ndarray | None = None

    def canvas(self, pixels, size, fill=0):
        if self.transform is not None:
            # Interpolate premultiplied color to keep transparent RGB out of edges.
            alpha = pixels[..., 3:4].astype(np.float32) / 255
            premultiplied = np.concatenate((pixels[..., :3] * alpha, alpha), axis=2)
            warped = cv2.warpPerspective(premultiplied, self.transform, size)
            color = np.divide(warped[..., :3], warped[..., 3:4],
                              out=np.zeros_like(warped[..., :3]), where=warped[..., 3:4] > 0)
            return np.clip(np.rint(np.concatenate((color, warped[..., 3:4] * 255), axis=2)), 0, 255).astype(np.uint8)
        w, h = size
        result = np.full((h, w, *pixels.shape[2:]), fill, pixels.dtype)
        x, y = self.origin
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + pixels.shape[1]), min(h, y + pixels.shape[0])
        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = pixels[y1-y:y2-y, x1-x:x2-x]
        return result

    def local_mask(self, mask):
        # Editable rasters only have integer translations. Preserve off-canvas data.
        x, y = self.origin
        inverse = RasterPlacement((-x, -y), mask.shape[1::-1], 0)
        return inverse.canvas(mask, self.size)


def raster_placement(reader, layer):
    row = reader.rows[layer.main_id]
    base = offset(row, "LayerOffset")
    rendered = row.get("DrawToRenderOffscreenType") is not None
    render_offset = (0, 0) if rendered else offset(row, "LayerRenderOffscrOffset")
    origin = tuple(a + b for a, b in zip(base, render_offset))
    off = reader.top_offscreen(layer.main_id)
    attr = reader.attribute(off) if off else None
    blob = row.get("ResizableImageInfo")
    if blob and not rendered:
        # Only transform an unbaked source. A refreshed render already contains it.
        if len(blob) < 184 or struct.unpack_from(">I", blob)[0] != 120:
            raise ValueError(f"未対応のCLIP変形情報です: {layer.name}")
        width, height = struct.unpack_from(">II", blob, 32)
        if not width or not height:
            raise ValueError(f"CLIP変形元のサイズが不正です: {layer.name}")
        original = row.get("ResizableOriginalMipmap")
        chain = reader.mipmap_chain(original) if original else []
        if chain:
            off = chain[0][1]
            attr = reader.attribute(off)
        if not attr or (attr["width"], attr["height"]) != (width, height):
            raise ValueError(f"CLIP変形元の画像を特定できません: {layer.name}")
        corners = np.array(struct.unpack_from(">8d", blob, 120), np.float32).reshape(4, 2)[[0, 1, 3, 2]]
        corners += base
        source = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
        if not np.isfinite(corners).all() or abs(cv2.contourArea(corners)) < 1e-6:
            raise ValueError(f"CLIP変形先の座標が不正です: {layer.name}")
        return RasterPlacement(origin, (width, height), off, transform=cv2.getPerspectiveTransform(source, corners))
    if attr and (reader.has_pixels(off) or attr["has_init_color"]):
        editable = attr["num_channels"] == 4 and not layer.is_text and not blob
        return RasterPlacement(origin, (attr["width"], attr["height"]), off, editable)
    if layer.is_text:
        obj = reader.object_offscreen(layer.main_id)
        if obj:
            attr = reader.attribute(obj)
            # Native parser has already located the text bounding box (including negatives).
            origin = (layer.left + origin[0], layer.top + origin[1])
            return RasterPlacement(origin, (attr["width"], attr["height"]), obj)
    if off and attr and not blob:
        return RasterPlacement(origin, (attr["width"], attr["height"]), off,
                               attr["num_channels"] == 4 and not layer.is_text)
    return None


def read_pixels(reader, native, index, placement):
    layer = native.layers[index]
    # The native path is fast, but crops to its inferred bounds. Use actual
    # offscreen pixels when those bounds differ or a transformed source is used.
    if (placement.transform is None and placement.size == (layer.width, layer.height)
            and placement.offscreen == reader.top_offscreen(layer.main_id)):
        raw = native.layer_image(index, "image")
        if raw:
            return np.frombuffer(raw, np.uint8).reshape(placement.size[1], placement.size[0], 4)
    return reader.offscreen_image(placement.offscreen)[..., [2, 1, 0, 3]]


def read_mask(reader, layer, size):
    off = reader.top_offscreen(layer.main_id, mask=True)
    if not off:
        return None
    attr = reader.attribute(off)
    row = reader.rows[layer.main_id]
    placement = RasterPlacement(offset(row, "LayerOffset", "LayerMaskOffset", "LayerMaskOffscrOffset"),
                                (attr["width"], attr["height"]), off)
    fill = attr["init_color"] & 255 if attr["has_init_color"] else 0
    return placement.canvas(reader.offscreen_image(off)[..., 3], size, fill)
