"""Independent canvas-coordinate expectations, not native-compositor parity."""
from pathlib import Path
from tempfile import TemporaryDirectory
import struct
import unittest
from unittest.mock import patch

import clipparse
import numpy as np

from auto_mosaic.layers import LayerDocument, effect_pixels, on_white
from auto_mosaic.clip_geometry import ClipReader
from tests.layer_fixtures import make_layers, move_clip_layer, edit_clip_metadata, group_clip_with_mask
from tests.test_layers import SETTINGS


def translated(pixels, dx, dy):
    expected = np.zeros_like(pixels)
    # Deliberately simple pixel mapping, independent of the implementation.
    for y, x in np.ndindex(pixels.shape[:2]):
        if 0 <= x + dx < pixels.shape[1] and 0 <= y + dy < pixels.shape[0]:
            expected[y + dy, x + dx] = pixels[y, x]
    expected[expected[..., 3] == 0] = 0
    return expected


class ClipGeometryTests(unittest.TestCase):
    def test_non_canvas_raster_size_preserved_on_save(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "small.clip"
            _, pixels = make_layers(path)
            move_clip_layer(path, 100, -20)
            def edit(db):
                from clipparse_tools.clip_build import retarget_attribute
                row = db.execute("SELECT MainId, Attribute FROM Offscreen WHERE MainId=("
                                 "SELECT Offscreen FROM MipmapInfo WHERE MainId=("
                                 "SELECT BaseMipmapInfo FROM Mipmap WHERE MainId=("
                                 "SELECT LayerRenderMipmap FROM Layer WHERE LayerName=?)))",
                                 ("加工対象の模様",)).fetchone()
                from clipparse_tools.clip_lazy_demo import parse_attribute
                sizes = parse_attribute(row[1])["block_sizes"]
                db.execute("UPDATE Offscreen SET Attribute=? WHERE MainId=?",
                           (retarget_attribute(row[1], 80, 60, sizes), row[0]))
            edit_clip_metadata(path, edit)
            doc = LayerDocument(path)
            try:
                expected = np.zeros_like(pixels)
                expected[:40, 100:] = pixels[20:60, :28]
                np.testing.assert_array_equal(doc.detection_image(frozenset({1})), on_white(expected))
                mask = np.zeros(pixels.shape[:2], bool)
                mask[12:28, 108:120] = True
                local = np.zeros((60, 80), bool)
                local[32:48, 8:20] = True
                edited = effect_pixels(pixels[:60, :80], local, SETTINGS)
                result = doc.render(mask, frozenset({1}), SETTINGS, frozenset({1}))
                expected[:40, 100:] = edited[20:60, :28]
                np.testing.assert_array_equal(on_white(result.bgra), on_white(expected))
                output = Path(temp) / "small-saved.clip"
                result.save(output)
                with ClipReader(output) as check:
                    placement = doc.display.placements[1]
                    actual = check.offscreen_image(placement.offscreen)[..., [2, 1, 0, 3]]
                    np.testing.assert_array_equal(actual, edited)
            finally:
                doc.close()

    def test_group_mask_has_its_own_origin_in_preview_detection_and_export(self):
        for pass_through in (False, True):
            with self.subTest(pass_through=pass_through), TemporaryDirectory() as temp:
                path = Path(temp) / "group.clip"
                _, pixels = make_layers(path)
                mask = np.zeros(pixels.shape[:2], np.uint8)
                mask[15:55, 20:70] = 255
                group_clip_with_mask(path, mask, pass_through=pass_through)
                doc = LayerDocument(path)
                try:
                    visible = frozenset(info.index for info in doc.layers)
                    expected = pixels.copy()
                    # Mask origin = layer(7,-3) + mask(5,9) + mask surface(-2,1).
                    expected[..., 3] = 0
                    expected[22:62, 30:80, 3] = pixels[22:62, 30:80, 3] // 2
                    actual = doc.display.image(visible)
                    np.testing.assert_array_equal(on_white(actual), on_white(expected))
                    np.testing.assert_array_equal(doc.detection_image(visible), on_white(expected))
                    result = doc.render(np.zeros(mask.shape, bool), frozenset(), SETTINGS)
                    np.testing.assert_array_equal(on_white(result.bgra), on_white(expected))
                finally:
                    doc.close()

    def test_offsets_detection_edit_export_and_baked_flag(self):
        for x, y, rx, ry, baked in ((23, -17, 5, 3, None), (-21, 13, 8, -2, 1)):
            with self.subTest(baked=baked), TemporaryDirectory() as temp:
                path = Path(temp) / "moved.clip"
                _, pixels = make_layers(path)
                move_clip_layer(path, x, y, rx, ry, baked)
                dx, dy = (x + rx, y + ry) if baked is None else (x, y)
                expected = translated(pixels, dx, dy)
                document = LayerDocument(path)
                try:
                    visible = frozenset({1})
                    # Demonstrate the dependency bug: its coordinates ignore metadata.
                    native = document._read(path)
                    self.assertEqual((native.layers[1].left, native.layers[1].top), (0, 0))
                    del native
                    np.testing.assert_array_equal(document.display.image(visible), expected)
                    np.testing.assert_array_equal(document.detection_image(visible), on_white(expected))
                    mask = np.zeros(expected.shape[:2], bool)
                    mask[32:48, 44:60] = True
                    local = np.zeros_like(mask)
                    for yy, xx in np.ndindex(local.shape):
                        if 0 <= xx + dx < mask.shape[1] and 0 <= yy + dy < mask.shape[0]:
                            local[yy, xx] = mask[yy + dy, xx + dx]
                    edited = effect_pixels(pixels, local, SETTINGS)
                    expected_edit = translated(edited, dx, dy)
                    prepared = document.prepare(mask, visible, SETTINGS)
                    with patch.object(document, "_read", side_effect=AssertionError("display I/O")), patch(
                        "auto_mosaic.layers.effect_pixels", side_effect=AssertionError("display effect")
                    ), patch("auto_mosaic.clip_geometry.ClipReader", side_effect=AssertionError("display decode")):
                        for selected in (visible, frozenset(), visible):
                            actual = prepared.images(selected)[1]
                        np.testing.assert_array_equal(actual, expected_edit)
                    result = document.render(mask, visible, SETTINGS, visible)
                    np.testing.assert_array_equal(result.bgra, expected_edit)
                    saved = Path(temp) / "saved.clip"
                    result.save(saved)
                    reopened = LayerDocument(saved)
                    try:
                        np.testing.assert_array_equal(reopened.original, expected_edit)
                        reader = reopened._read(saved)
                        raw = np.frombuffer(reader.layer_image(1, "image"), np.uint8).reshape(pixels.shape)
                        np.testing.assert_array_equal(raw, edited)
                        png, w, h = reader.preview_png()
                        import io
                        from PIL import Image
                        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(png)))[..., [2, 1, 0, 3]], expected_edit)
                        del reader
                        with ClipReader(path) as a, ClipReader(saved) as b:
                            for mid, row in a.rows.items():
                                for field in ("LayerOffsetX", "LayerOffsetY", "LayerRenderOffscrOffsetX",
                                              "LayerRenderOffscrOffsetY", "DrawToRenderOffscreenType"):
                                    self.assertEqual(row.get(field), b.rows[mid].get(field))
                    finally:
                        reopened.close()
                finally:
                    document.close()

    def test_unbaked_transform_and_baked_render_are_not_applied_twice(self):
        for baked in (None, 1):
            with self.subTest(baked=baked), TemporaryDirectory() as temp:
                path = Path(temp) / "transformed.clip"
                _, pixels = make_layers(path)
                move_clip_layer(path, 0, 0, baked=baked)
                blob = bytearray(184)
                struct.pack_into(">I", blob, 0, 120)
                struct.pack_into(">II", blob, 32, 128, 96)
                struct.pack_into(">8d", blob, 120, 16, 8, 144, 8, 16, 104, 144, 104)
                def edit(db):
                    if "ResizableImageInfo" not in {r[1] for r in db.execute("PRAGMA table_info(Layer)")}:
                        db.execute("ALTER TABLE Layer ADD COLUMN ResizableImageInfo BLOB")
                    db.execute("UPDATE Layer SET ResizableImageInfo=? WHERE LayerName=?", (blob, "加工対象の模様"))
                edit_clip_metadata(path, edit)
                doc = LayerDocument(path)
                try:
                    expected = pixels if baked else translated(pixels, 16, 8)
                    actual = doc.display.image(frozenset({1}))
                    # Zero-alpha RGB is not meaningful after a premultiplied transform.
                    np.testing.assert_array_equal(on_white(actual), on_white(expected))
                    np.testing.assert_array_equal(doc.detection_image(frozenset({1})), on_white(expected))
                    self.assertFalse(next(info.editable for info in doc.layers if info.index == 1))
                finally:
                    doc.close()


if __name__ == "__main__":
    unittest.main()
