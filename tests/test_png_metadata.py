from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
import zlib

import numpy as np
from PIL import Image

from auto_mosaic.domain import ProcessingResult
from auto_mosaic.image_ops import load_image_bgr
from auto_mosaic.pipeline import MosaicPipeline


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data)))


class PngMetadataTests(unittest.TestCase):
    def test_processed_save_preserves_raw_generation_text_and_processed_pixels(self):
        chunks = [
            _chunk(b"tEXt", b'prompt\0{"seed": 12345}'),
            _chunk(b"zTXt", b"workflow\0\0" + zlib.compress(b'{"nodes": [1, 2]}')),
            _chunk(b"iTXt", b"parameters\0\0\0ja\0" + "生成情報\0日本語のプロンプト".encode()),
            _chunk(b"iTXt", b"compressed\0\1\0\0\0" + zlib.compress("圧縮テキスト".encode())),
            _chunk(b"tEXt", b"Comment\0first"),
            _chunk(b"tEXt", b"Comment\0second"),
        ]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.PNG"
            Image.new("RGB", (4, 4), "white").save(source)
            # Some generators place metadata after the image data, before IEND.
            original = source.read_bytes()
            original = original[:-12] + b"".join(chunks) + original[-12:]
            source.write_bytes(original)
            pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
            result = ProcessingResult(source_path=source, image_bgr=pixels,
                                      mask=np.ones((4, 4), dtype=bool))
            pipeline = MosaicPipeline(root / "models")
            for explicit_path in (None, root / "output" / "explicit.png"):
                with self.subTest(explicit_path=explicit_path):
                    output = pipeline.save(result, root / "output", output_path=explicit_path)
                    output_bytes = output.read_bytes()
                    for chunk in chunks:
                        self.assertEqual(output_bytes.count(chunk), 1)
                    with Image.open(output) as saved:
                        saved.verify()
                    with Image.open(output) as saved:
                        self.assertEqual(saved.text["prompt"], '{"seed": 12345}')
                        self.assertEqual(saved.text["workflow"], '{"nodes": [1, 2]}')
                        self.assertEqual(saved.text["parameters"], "日本語のプロンプト")
                        self.assertEqual(saved.text["compressed"], "圧縮テキスト")
                    np.testing.assert_array_equal(load_image_bgr(output), pixels)
                    self.assertEqual(source.read_bytes(), original)

    def test_save_without_text_metadata_and_jpeg_still_work(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for extension in (".png", ".jpg"):
                with self.subTest(extension=extension):
                    source = root / f"source{extension}"
                    Image.new("RGB", (4, 4), "white").save(source)
                    result = ProcessingResult(source_path=source,
                                              image_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
                                              mask=np.ones((4, 4), dtype=bool))
                    output = MosaicPipeline(root / "models").save(result, root / "output")
                    np.testing.assert_array_equal(load_image_bgr(output), result.image_bgr)

    def test_corrupt_metadata_fails_before_creating_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (4, 4)).save(source)
            original = source.read_bytes()
            chunk = _chunk(b"tEXt", b"prompt\0value")
            broken = chunk[:-1] + bytes([chunk[-1] ^ 1])
            source.write_bytes(original[:-12] + broken + original[-12:])
            result = ProcessingResult(source_path=source,
                                      image_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
                                      mask=np.ones((4, 4), dtype=bool))
            output = root / "output.png"
            with self.assertRaisesRegex(ValueError, "PNGメタデータ"):
                MosaicPipeline(root / "models").save(result, root, output_path=output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
