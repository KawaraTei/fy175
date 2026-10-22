"""Small deterministic textured layers; no personal artwork is used."""
from pathlib import Path

import numpy as np
import clipparse
import psdparse


def make_layers(path: Path, width=128, height=96):
    destination = path
    path = path.with_name("fixture" + path.suffix)
    y, x = np.mgrid[:height, :width]
    background = np.full((height, width, 4), (180, 120, 60, 255), dtype=np.uint8)
    top = np.zeros_like(background)
    top[..., 0] = (x * 17 + y * 3) % 256
    top[..., 1] = (x * 7 + y * 13) % 256
    top[..., 2] = (x * 11 + y * 19) % 256
    top[height//8:height*7//8, width//8:width*7//8, 3] = 180
    if path.suffix == ".psd":
        reader = psdparse.PSDFile()
        reader.create_blank(width, height)
        reader.add_layer("背景", 0, 0, background.tobytes(), width, height)
        reader.add_layer("加工対象の模様", 0, 0, top.tobytes(), width, height)
        reader.set_merged_image(reader.composite()[0])
        assert reader.save(str(path))
    else:
        template = Path(__file__).parent / "fixtures" / "emptyimage.clip"
        reader = clipparse.ClipFile()
        assert reader.load(str(template))
        ids = [layer.main_id for layer in reader.layers]
        writer = clipparse.ClipWriter()
        writer.load(str(template))
        writer.resize_canvas(width, height)
        for layer_id, name, pixels in zip(ids, ("背景", "加工対象の模様"), (background, top)):
            writer.set_layer_attr(layer_id, name=name)
            writer.set_pixels(layer_id, pixels.tobytes(), width, height)
        writer.save(str(path))
        check = clipparse.ClipFile()
        assert check.load(str(path))
        merged = check.merged_image()
        del check
        writer.set_canvas_preview(merged, width, height)
        writer.save(str(path))
        assert clipparse.validate(str(path)) == []
    del reader
    if path != destination:
        destination.write_bytes(path.read_bytes())
    return background, top
