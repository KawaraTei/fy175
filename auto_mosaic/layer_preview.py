"""In-memory display caches. Native document serialization belongs to export only."""
from collections import OrderedDict
import copy
from pathlib import Path
import sqlite3
import struct
from types import SimpleNamespace

import cv2
import numpy as np
from clipparse_tools import clip_lazy_demo as clip_blend
from PySide6.QtGui import QImage, QPainter

DISPLAY_MAX_SIDE = 1024


def display_size(width, height):
    scale = min(1, DISPLAY_MAX_SIDE / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))

def normal_over(dst, src):
    h, w = dst.shape[:2]
    bottom = QImage(np.ascontiguousarray(dst).data, w, h, w * 4, QImage.Format.Format_ARGB32).copy()
    top = QImage(np.ascontiguousarray(src).data, w, h, w * 4, QImage.Format.Format_ARGB32).copy()
    painter = QPainter(bottom)
    painter.drawImage(0, 0, top)
    painter.end()
    return np.frombuffer(bottom.constBits(), np.uint8).reshape(h, w, 4).copy()


def thumbnail(pixels, size):
    if pixels.shape[1::-1] == size:
        return pixels.copy()
    # Resize premultiplied pixels to avoid hidden RGB bleeding into translucent edges.
    alpha = pixels[..., 3:4].astype(np.float32) / 255
    scaled = cv2.resize(np.concatenate((pixels[..., :3] * alpha, alpha), axis=2), size,
                        interpolation=cv2.INTER_AREA)
    color = np.divide(scaled[..., :3], scaled[..., 3:4], out=np.zeros_like(scaled[..., :3]),
                      where=scaled[..., 3:4] > 0)
    return np.clip(np.rint(np.concatenate((color, scaled[..., 3:4] * 255), axis=2)), 0, 255).astype(np.uint8)


class DisplayCache:
    def __init__(self):
        self.cache = OrderedDict()

    def image(self, visible):
        if visible not in self.cache:
            self.cache[visible] = self.compose(visible)
            while len(self.cache) > 1 and (len(self.cache) > 4 or sum(x.nbytes for x in self.cache.values()) > 128 * 1024 * 1024):
                self.cache.popitem(last=False)
        self.cache.move_to_end(visible)
        return self.cache[visible]


class PsdDisplay(DisplayCache):
    def __init__(self, document, reader):
        super().__init__()
        self.document = document
        self.reader = reader
        self.flags = [layer.visible for layer in reader.layers]

    def compose(self, visible):
        for layer, flag in zip(self.reader.layers, self.flags):
            layer.visible = flag
        self.document._set_visibility(self.reader, self.reader, visible)
        return thumbnail(self.document._composite(self.reader), display_size(self.document.width, self.document.height))


class ClipDisplay(DisplayCache):
    """Cache decoded pixels; use clipparse's reference blend equations and tree rules."""
    def __init__(self, document, reader):
        super().__init__()
        self.document = document
        # Native LayerInfo objects borrow the reader's storage; retain value copies.
        fields = ("is_group", "is_filter", "width", "height", "left", "top", "main_id",
                  "has_mask", "composite_raw", "opacity_raw", "clipping")
        self.layers = [SimpleNamespace(**{key: getattr(layer, key) for key in fields}) for layer in reader.layers]
        self.children = {-1: list(reader.children(-1))}
        self.children.update({i: list(reader.children(i)) for i, layer in enumerate(self.layers) if layer.is_group})
        self.size = display_size(document.width, document.height)
        self.pixels = {}
        self.masks = {}
        self.filters = {}
        # Only metadata is read here. No temporary CLIP or SQLite files are written.
        with document.snapshot.open("rb") as stream:
            header = stream.read(24)
            header_length = struct.unpack_from(">Q", header, 16)[0]
            stream.seek(header_length + 24)
            sql_offset = struct.unpack(">Q", stream.read(8))[0]
            stream.seek(sql_offset)
            kind, length = struct.unpack(">8sQ", stream.read(16))
            if kind != b"CHNKSQLi":
                raise ValueError("CLIPのメタデータを読み込めません。")
            db = sqlite3.connect(":memory:")
            try:
                db.deserialize(stream.read(length))
                columns = {row[1] for row in db.execute("PRAGMA table_info(Layer)")}
                if "FilterLayerInfo" in columns:
                    self.filters = dict(db.execute("SELECT MainId, FilterLayerInfo FROM Layer WHERE FilterLayerInfo IS NOT NULL"))
            finally:
                db.close()
        mask_reader = None
        try:
            for index, layer in enumerate(self.layers):
                if layer.width > 0 and layer.height > 0 and not layer.is_group and not layer.is_filter:
                    raw = reader.layer_image(index, "image")
                    if raw:
                        pixels = np.frombuffer(raw, np.uint8).reshape(layer.height, layer.width, 4)
                        self.pixels[index] = self.place(pixels, layer)
                if layer.has_mask:
                    # Group masks are not exposed by the native layer_image API.
                    if mask_reader is None:
                        mask_reader = clip_blend.ClipFile(str(document.snapshot))
                    off = mask_reader.top_offscreen(layer.main_id, mask=True)
                    if off is not None and mask_reader.has_pixels(off):
                        mask = clip_blend.fit_canvas(mask_reader.offscreen_image(off), document.width, document.height)
                        self.masks[index] = cv2.resize(mask[..., 3], self.size, interpolation=cv2.INTER_AREA)
        finally:
            if mask_reader is not None:
                mask_reader.db.close()
                Path(mask_reader.tmp).unlink(missing_ok=True)

    def place(self, pixels, layer):
        canvas = clip_blend.place(pixels, (layer.left, layer.top), self.document.width, self.document.height)
        return thumbnail(canvas, self.size)

    def with_pixels(self, changes):
        result = copy.copy(self)
        result.cache = OrderedDict()
        result.pixels = self.pixels.copy()
        for index, pixels in changes:
            result.pixels[index] = self.place(pixels, self.layers[index])
        return result

    def compose(self, visible):
        enabled = visible | {parent for index in visible for parent in self.document._ancestors[index]}
        def draw(parent, dst=None):
            if dst is None:
                dst = np.zeros((self.size[1], self.size[0], 4), np.uint8)
            clip_base = None
            for index in self.children[parent]:
                if index not in enabled:
                    continue
                layer = self.layers[index]
                if layer.is_group and layer.composite_raw == 30:
                    dst = draw(index, dst)
                    clip_base = None
                    continue
                if layer.is_filter:
                    blob = self.filters.get(layer.main_id)
                    if blob:
                        rgba = clip_blend.apply_filter(dst[..., [2, 1, 0, 3]], bytes(blob))
                        dst = rgba[..., [2, 1, 0, 3]]
                    continue
                pixels = draw(index) if layer.is_group else self.pixels.get(index)
                if pixels is None:
                    continue
                pixels = pixels.copy()
                if index in self.masks:
                    pixels[..., 3] = pixels[..., 3].astype(np.uint32) * self.masks[index] // 255
                if layer.opacity_raw < 256:
                    pixels[..., 3] = pixels[..., 3].astype(np.uint32) * layer.opacity_raw // 256
                if layer.clipping and clip_base is not None:
                    pixels[..., 3] = (pixels[..., 3].astype(np.uint32) * clip_base + 127) // 255
                elif not layer.clipping:
                    clip_base = pixels[..., 3].copy()
                # Reference equations expect RGB ordering for luminance-based blend modes.
                if layer.composite_raw == 0:
                    dst = normal_over(dst, pixels)
                else:
                    dst = clip_blend.blend_over(dst[..., [2, 1, 0, 3]], pixels[..., [2, 1, 0, 3]],
                                               layer.composite_raw)[..., [2, 1, 0, 3]]
            return dst
        return draw(-1)


class PreparedLayers:
    def __init__(self, original, processed):
        self.original = original
        self.processed = processed
        self.bgra = self.original_bgra = None

    def images(self, visible):
        return self.original.image(visible), self.processed.image(visible)
