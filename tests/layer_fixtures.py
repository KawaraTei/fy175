"""Small deterministic textured layers; no personal artwork is used."""
from pathlib import Path

import numpy as np
import clipparse
import psdparse


def edit_clip_metadata(path, edit):
    """Modify a synthetic fixture's SQL, retaining the actual pixel chunks."""
    from clipparse_tools.clip_write import ClipFile
    file = ClipFile(str(path))
    try:
        edit(file.db)
        file.save(str(path))
    finally:
        file.close()


def move_clip_layer(path, x, y, render_x=0, render_y=0, baked=None):
    def edit(db):
        columns = {r[1] for r in db.execute("PRAGMA table_info(Layer)")}
        if "DrawToRenderOffscreenType" not in columns:
            db.execute("ALTER TABLE Layer ADD COLUMN DrawToRenderOffscreenType INTEGER")
        db.execute("UPDATE Layer SET LayerOffsetX=?, LayerOffsetY=?, LayerRenderOffscrOffsetX=?, "
                   "LayerRenderOffscrOffsetY=?, DrawToRenderOffscreenType=? WHERE LayerName=?",
                   (x, y, render_x, render_y, baked, "加工対象の模様"))
    edit_clip_metadata(path, edit)


def group_clip_with_mask(path, mask, *, pass_through=False):
    """Turn the background fixture into a folder with a single-channel mask."""
    import struct
    from clipparse_tools.clip_write import ClipFile, as_str
    from clipparse_tools.clip_encode import build_block_record, build_trailers, patch_block_sizes
    native = clipparse.ClipFile()
    native.load(str(path))
    group, child = [layer.main_id for layer in native.layers]
    del native
    file = ClipFile(str(path))
    try:
        db = file.db
        mip = db.execute("SELECT LayerRenderMipmap FROM Layer WHERE MainId=?", (group,)).fetchone()[0]
        off = db.execute("SELECT Offscreen FROM MipmapInfo WHERE MainId=(SELECT BaseMipmapInfo FROM Mipmap WHERE MainId=?)", (mip,)).fetchone()[0]
        attr, external = db.execute("SELECT Attribute, BlockData FROM Offscreen WHERE MainId=?", (off,)).fetchone()
        attr = bytearray(attr)
        # Parameter section: mode=mask, no color planes, one 8-bit alpha block.
        for field, value in ((4, 1), (6, 0), (7, 1), (8, 0), (9, 0), (10, 0)):
            struct.pack_into(">I", attr, 38 + field * 4, value)
        block = np.zeros((256, 256), np.uint8)
        block[:mask.shape[0], :mask.shape[1]] = mask
        record = bytearray(build_block_record(0, block.tobytes(), 256, 256))
        struct.pack_into(">I", record, 50, block.size)
        attr = patch_block_sizes(bytes(attr), [len(record)])
        db.execute("UPDATE Offscreen SET Attribute=? WHERE MainId=?", (attr, off))
        for index, (eid, _) in enumerate(file.externals):
            if as_str(eid) == as_str(external):
                file.externals[index] = (eid, bytes(record) + build_trailers([1], [0]))
                break
        db.execute("UPDATE Layer SET LayerFolder=17, LayerType=LayerType|2, LayerComposite=?, "
                   "LayerOpacity=128, LayerFirstChildIndex=?, LayerNextIndex=0, "
                   "LayerLayerMaskMipmap=LayerRenderMipmap, LayerRenderMipmap=0, "
                   "LayerOffsetX=7, LayerOffsetY=-3, LayerMaskOffsetX=5, LayerMaskOffsetY=9, "
                   "LayerMaskOffscrOffsetX=-2, LayerMaskOffscrOffsetY=1 WHERE MainId=?",
                   (30 if pass_through else 0, child, group))
        file.save(str(path))
    finally:
        file.close()


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
