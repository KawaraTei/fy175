from dataclasses import replace
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import clipparse
import psdparse

from auto_mosaic.domain import EffectType, ImageMode, ProcessingSettings
from auto_mosaic.layers import LayerDocument, effect_pixels, on_white
from tests.layer_fixtures import make_layers


SETTINGS = ProcessingSettings(ImageMode.ILLUSTRATION, frozenset({"penis", "vagina"}), .3)


class LayerTests(unittest.TestCase):
    def test_cached_display_does_not_read_write_or_reprocess(self):
        for extension in (".psd", ".clip"):
            with self.subTest(extension=extension), TemporaryDirectory() as temporary:
                source = Path(temporary) / ("cached" + extension)
                make_layers(source)
                document = LayerDocument(source)
                try:
                    mask = np.ones((96, 128), bool)
                    prepared = document.prepare(mask, frozenset({1}), SETTINGS)
                    selections = (frozenset({0}), frozenset({1}), frozenset(), frozenset({0, 1}))
                    expected = {visible: document.render(mask, frozenset({1}), SETTINGS, visible).bgra
                                for visible in selections}
                    before = {p.name: p.stat().st_mtime_ns for p in document.root.iterdir()}
                    with patch.object(document, "_read", side_effect=AssertionError("display read")), patch(
                        "auto_mosaic.layers.effect_pixels", side_effect=AssertionError("display processing")
                    ), patch.object(document, "render", side_effect=AssertionError("display export")):
                        for visible in selections * 2:
                            original, processed = prepared.images(visible)
                            # Qt's integer alpha rounding may differ by one from native CLIP composition.
                            self.assertLessEqual(np.abs(processed.astype(int) - expected[visible]).max(), 1)
                    self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in document.root.iterdir()})
                finally:
                    document.close()

    def test_clip_preview_retains_blend_opacity_and_clipping(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "blend.clip"
            make_layers(source)
            reader = clipparse.ClipFile()
            reader.load(str(source))
            writer = clipparse.ClipWriter()
            writer.load(str(source))
            writer.set_layer_attr(reader.layers[1].main_id, opacity=128, composite=2, clipping=1)
            changed = root / "changed.clip"
            writer.save(str(changed))
            del reader, writer
            document = LayerDocument(changed)
            try:
                visible = frozenset({0, 1})
                actual = document.display.image(visible)
                self.assertLessEqual(np.abs(actual.astype(int) - document.original).max(), 1)
            finally:
                document.close()

    def test_visibility_preserves_hidden_pixels_and_detection_selection(self):
        for extension in (".psd", ".clip"):
            with self.subTest(extension=extension), TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / ("visible" + extension)
                background, top = make_layers(source)
                document = LayerDocument(source)
                try:
                    mask = np.ones(background.shape[:2], bool)
                    detection = document.detection_image(frozenset({1}))
                    result = document.render(mask, frozenset({1}), SETTINGS, frozenset({0}))
                    np.testing.assert_array_equal(result.bgra, background)
                    np.testing.assert_array_equal(result.original_bgra, background)
                    np.testing.assert_array_equal(document.detection_image(frozenset({1})), detection)
                    output = root / ("hidden" + extension)
                    result.save(output)
                    reader = document._read(output)
                    self.assertFalse(reader.layers[1].visible)
                    hidden_pixels = np.frombuffer(reader.layer_image(1, "image"), np.uint8).reshape(top.shape)
                    np.testing.assert_array_equal(hidden_pixels, effect_pixels(top, mask, SETTINGS))
                    del reader
                    reopened = LayerDocument(output)
                    try:
                        self.assertFalse(next(info.visible for info in reopened.layers if info.index == 1))
                        shown = reopened.render(mask, frozenset(), SETTINGS, frozenset({0, 1}))
                        self.assertTrue(np.any(shown.bgra != background))
                        np.testing.assert_array_equal(reopened.detection_image(frozenset({1})), on_white(hidden_pixels))
                        empty = reopened.render(mask, frozenset(), SETTINGS, frozenset())
                        self.assertFalse(empty.bgra[..., 3].any())
                    finally:
                        reopened.close()
                finally:
                    document.close()

    def test_show_child_of_hidden_group_without_showing_sibling(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "group.psd"
            psd = psdparse.PSDFile()
            psd.create_blank(32, 32)
            first = np.full((32, 32, 4), (10, 20, 30, 255), np.uint8)
            second = np.full_like(first, (40, 50, 60, 255))
            psd.add_layer("first", 0, 0, first.tobytes(), 32, 32)
            psd.add_layer("second", 0, 0, second.tobytes(), 32, 32)
            psd.add_folder("group", 0, 2)
            next(layer for layer in psd.layers if layer.is_group).visible = False
            psd.save(str(source))
            del psd
            document = LayerDocument(source)
            try:
                self.assertTrue(all(not info.visible for info in document.layers))
                index = next(info.index for info in document.layers if info.name == "group / first")
                result = document.render(np.zeros((32, 32), bool), frozenset(), SETTINGS, frozenset({index}))
                np.testing.assert_array_equal(result.bgra, first)
                np.testing.assert_array_equal(document.detection_image(frozenset({index})), on_white(first))
            finally:
                document.close()

    def test_selective_roundtrip_psd_and_clip(self):
        for extension in (".psd", ".clip"):
            with self.subTest(extension=extension), TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / ("日本語" + extension)
                background, top = make_layers(source)
                digest = hashlib.sha256(source.read_bytes()).digest()
                document = LayerDocument(source)
                try:
                    indices = frozenset(info.index for info in document.layers)
                    np.testing.assert_array_equal(document.detection_image(indices), on_white(document.original))
                    np.testing.assert_array_equal(document.detection_image(frozenset({0})), on_white(background))
                    # Detection exclusions do not affect the exported layer visibility.
                    mask = np.zeros((96, 128), dtype=bool)
                    mask[24:72, 32:96] = True
                    result = document.render(mask, frozenset({1}), SETTINGS)
                    output = root / ("result" + extension)
                    result.save(output)
                    reader = psdparse.PSDFile() if extension == ".psd" else clipparse.ClipFile()
                    self.assertTrue(reader.load(str(output)))
                    actual_bg = np.frombuffer(reader.layer_image(0, "image"), np.uint8).reshape(background.shape)
                    actual_top = np.frombuffer(reader.layer_image(1, "image"), np.uint8).reshape(top.shape)
                    np.testing.assert_array_equal(actual_bg, background)
                    np.testing.assert_array_equal(actual_top[..., 3], top[..., 3])
                    np.testing.assert_array_equal(actual_top[~mask], top[~mask])
                    self.assertTrue(np.any(actual_top[mask] != top[mask]))
                    merged = reader.composite()[0] if extension == ".psd" else reader.merged_image()
                    np.testing.assert_array_equal(np.frombuffer(merged, np.uint8).reshape(result.bgra.shape), result.bgra)
                    if extension == ".clip":
                        self.assertEqual(clipparse.validate(str(output)), [])
                    del reader
                    result.save(root / "result.png")
                    with Image.open(root / "result.png") as image:
                        np.testing.assert_array_equal(np.asarray(image), result.bgra[..., [2, 1, 0, 3]])
                    with self.assertRaises(FileExistsError):
                        result.save(source)
                    self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), digest)
                    no_layers = document.render(mask, frozenset(), SETTINGS)
                    np.testing.assert_array_equal(no_layers.bgra, document.original)
                finally:
                    document.close()

    def test_offset_group_hidden_and_masked_psd(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "offset.psd"
            psd = psdparse.PSDFile()
            psd.create_blank(80, 60)
            y, x = np.mgrid[:30, :40]
            pixels = np.zeros((30, 40, 4), np.uint8)
            pixels[..., :3] = ((x + y) % 2 * 255)[..., None]
            pixels[..., 3] = 255
            psd.add_layer("offset", -10, 20, pixels.tobytes(), 40, 30)
            psd.add_folder("group", 0, 1)
            psd.set_layer_mask_pixels(1, bytes([128])*1200, -10, 20, 40, 30)
            psd.add_layer("hidden", 0, 0, pixels.tobytes(), 40, 30)
            psd.layers[-1].visible = False
            psd.save(str(source))
            del psd
            document = LayerDocument(source)
            try:
                self.assertEqual(next(info.depth for info in document.layers if info.index == 1), 1)
                self.assertFalse(document.layers[0].visible)
                mask = np.ones((60, 80), bool)
                result = document.render(mask, frozenset({1}), SETTINGS)
                output = Path(temporary) / "changed.psd"
                result.save(output)
                psd = psdparse.PSDFile()
                psd.load(str(output))
                self.assertEqual((psd.layers[1].left, psd.layers[1].top), (-10, 20))
                self.assertIsNotNone(psd.layers[1].mask)
                self.assertTrue(any(layer.is_group for layer in psd.layers))
                self.assertFalse(psd.layers[-1].visible)
                del psd
            finally:
                document.close()

    def test_transparent_color_does_not_bleed(self):
        pixels = np.zeros((32, 32, 4), np.uint8)
        pixels[..., :3] = (255, 0, 0)
        pixels[8:24, 8:24] = (0, 0, 200, 128)
        mask = np.ones((32, 32), bool)
        for effect in EffectType:
            result = effect_pixels(pixels, mask, replace(SETTINGS, effect=effect))
            np.testing.assert_array_equal(result[..., 3], pixels[..., 3])
            np.testing.assert_array_equal(result[8:24, 8:24, :3], pixels[8:24, 8:24, :3])


if __name__ == "__main__":
    unittest.main()
