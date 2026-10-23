"""In-memory display caches. Native document serialization belongs to export only."""
from collections import OrderedDict
from contextlib import nullcontext
import copy
from types import SimpleNamespace

import cv2
import numpy as np
from clipparse_tools import clip_lazy_demo as clip_blend
from PySide6.QtGui import QImage, QPainter
from PySide6.QtCore import Qt

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
    # Use Qt's native premultiplied scaler; avoid full-canvas float temporaries.
    h, w = pixels.shape[:2]
    pixels = np.ascontiguousarray(pixels)
    image = QImage(pixels.data, w, h, w * 4, QImage.Format.Format_ARGB32)
    scaled = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied).scaled(
        *size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
    ).convertToFormat(QImage.Format.Format_ARGB32)
    return np.frombuffer(scaled.constBits(), np.uint8).reshape(size[1], size[0], 4).copy()


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
    def __init__(self, document, reader, *, full_size=False, source_path=None):
        super().__init__()
        self.document = document
        self.full_size = full_size
        self.source_path = source_path or document.snapshot
        self.native = reader if full_size else None
        # Native LayerInfo objects borrow the reader's storage; retain value copies.
        fields = ("is_group", "is_filter", "width", "height", "left", "top", "main_id",
                  "has_mask", "composite_raw", "opacity_raw", "clipping")
        self.layers = [SimpleNamespace(**{key: getattr(layer, key) for key in fields}) for layer in reader.layers]
        self.children = {-1: list(reader.children(-1))}
        self.children.update({i: list(reader.children(i)) for i, layer in enumerate(self.layers) if layer.is_group})
        self.size = ((document.width, document.height) if full_size
                     else display_size(document.width, document.height))
        self.pixels = {}
        self.masks = {}
        self.filters = {}
        self.placements = {}
        if full_size:
            # Geometry is authoritative in the initial snapshot. Full-resolution
            # detection/export decode one layer at a time instead of retaining
            # a canvas-sized buffer for every layer in a complex document.
            self.filters = document.display.filters
            self.placements = document.display.placements
            return
        from auto_mosaic.clip_geometry import ClipReader, raster_placement, read_pixels, read_mask
        with ClipReader(document.snapshot) as source:
            for index, layer in enumerate(reader.layers):
                self.filters[layer.main_id] = source.rows[layer.main_id].get("FilterLayerInfo")
                if not layer.is_group and not layer.is_filter:
                    placement = raster_placement(source, layer)
                    if placement is not None:
                        self.placements[index] = placement
                        self.pixels[index] = self.place(read_pixels(source, reader, index, placement), index)
                if layer.has_mask:
                    mask = read_mask(source, layer, (document.width, document.height))
                    if mask is not None:
                        self.masks[index] = cv2.resize(mask, self.size, interpolation=cv2.INTER_AREA)

    def place(self, pixels, index):
        canvas = self.placements[index].canvas(pixels, (self.document.width, self.document.height))
        return canvas if self.full_size else thumbnail(canvas, self.size)

    def with_pixels(self, changes):
        result = copy.copy(self)
        result.cache = OrderedDict()
        result.pixels = self.pixels.copy()
        for index, pixels in changes:
            result.pixels[index] = pixels if self.full_size else self.place(pixels, index)
        return result

    def compose(self, visible):
        from auto_mosaic.clip_geometry import ClipReader, read_pixels, read_mask
        enabled = visible | {parent for index in visible for parent in self.document._ancestors[index]}
        def pixels_for(index):
            if not self.full_size:
                return self.pixels.get(index)
            placement = self.placements.get(index)
            if placement is None:
                return None
            raw = self.pixels.get(index)
            if raw is None:
                raw = read_pixels(source, self.native, index, placement)
            return self.place(raw, index)

        def draw(parent, dst=None):
            if dst is None:
                dst = np.zeros((self.size[1], self.size[0], 4), np.uint8)
            clip_base = None
            for index in self.children[parent]:
                if index not in enabled:
                    continue
                layer = self.layers[index]
                mask = (read_mask(source, layer, self.size) if self.full_size and layer.has_mask
                        else self.masks.get(index))
                if layer.is_group and layer.composite_raw == 30:
                    if mask is None and layer.opacity_raw == 256:
                        dst = draw(index, dst)
                    else:
                        before = dst
                        after = draw(index, dst.copy())
                        coverage = np.full(dst.shape[:2], layer.opacity_raw / 256, np.float32)
                        if mask is not None:
                            coverage *= mask / 255
                        dst = interpolate_composite(before, after, coverage)
                    clip_base = None
                    continue
                if layer.is_filter:
                    blob = self.filters.get(layer.main_id)
                    if blob:
                        rgba = clip_blend.apply_filter(dst[..., [2, 1, 0, 3]], bytes(blob))
                        dst = rgba[..., [2, 1, 0, 3]]
                    continue
                pixels = draw(index) if layer.is_group else pixels_for(index)
                if pixels is None:
                    continue
                pixels = pixels.copy()
                if mask is not None:
                    pixels[..., 3] = pixels[..., 3].astype(np.uint32) * mask // 255
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
        try:
            with ClipReader(self.source_path) if self.full_size else nullcontext() as source:
                return draw(-1)
        finally:
            # Recursive closures otherwise retain the native file mapping until
            # cyclic GC, preventing a subsequent export/cleanup on Windows.
            draw = None


def interpolate_composite(before, after, coverage):
    """Mask a pass-through group's contribution, keeping its backdrop intact."""
    weight = coverage[..., None]
    a0, a1 = before[..., 3:4] / 255, after[..., 3:4] / 255
    alpha = a0 * (1 - weight) + a1 * weight
    color = before[..., :3] * a0 * (1 - weight) + after[..., :3] * a1 * weight
    color = np.divide(color, alpha, out=np.zeros_like(color), where=alpha > 0)
    return np.clip(np.rint(np.concatenate((color, alpha * 255), axis=2)), 0, 255).astype(np.uint8)


class PreparedLayers:
    def __init__(self, original, processed):
        self.original = original
        self.processed = processed
        self.bgra = self.original_bgra = None

    def images(self, visible):
        return self.original.image(visible), self.processed.image(visible)
