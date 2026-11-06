"""Render the layer UI with deterministic detection, then check real model inference."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import json
import time
import numpy as np
from unittest.mock import patch

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtTest import QTest

from auto_mosaic.layer_ui import LayerMosaicWindow
from auto_mosaic.layers import on_white
from auto_mosaic.pipeline import MosaicPipeline
from auto_mosaic.ui import AutoMosaicWindow
from tests.layer_fixtures import make_layers
from tests.test_layers import SETTINGS
from tests.test_layer_ui import synthetic_analysis, wait_idle


def main():
    root = Path(".codex-qa/layers").resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    if Path("C:/Windows/Fonts/meiryo.ttc").is_file():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/meiryo.ttc")
    report = {}
    for extension in (".psd", ".clip"):
        source = root / ("layer-sample" + extension)
        make_layers(source, 640, 480)
        window = LayerMosaicWindow(Path("models"), SETTINGS)
        window.pipeline.analyze_image = synthetic_analysis
        window.show()
        with patch.object(QMessageBox, "warning", side_effect=lambda *args: (_ for _ in ()).throw(AssertionError(args))):
            window.load_path(source)
            wait_idle(app, window)
            # Leave the background layer untouched.
            window.layers.topLevelItem(1).setCheckState(1, Qt.CheckState.Unchecked)
            wait_idle(app, window)
            for name, size in (("normal", (1260, 790)), ("minimum", (980, 650))):
                window.resize(*size)
                app.processEvents()
                window.refresh_preview()
                window.grab().save(str(root / f"{extension[1:]}-{name}.png"))
            # Visibility is independent of processing/detection and needs no reanalysis.
            window.layers.topLevelItem(0).setCheckState(3, Qt.CheckState.Unchecked)
            wait_idle(app, window)
            window.grab().save(str(root / f"{extension[1:]}-hidden.png"))
            window.layers.topLevelItem(0).setCheckState(3, Qt.CheckState.Checked)
            wait_idle(app, window)
            window.edit_button.click()
            rect = window.preview.image_rect
            point = (rect.topLeft() + QPointF(rect.width() * .18, rect.height() * .18)).toPoint()
            QTest.mouseClick(window.preview, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
            app.processEvents()
            window.grab().save(str(root / f"{extension[1:]}-mask-edit.png"))
            window.edit_button.click()
            wait_idle(app, window)
            window.resize(1260, 790)
            window.view.setCurrentText("元画像")
            app.processEvents()
            window.refresh_preview()
            window.grab().save(str(root / f"{extension[1:]}-original.png"))
            if extension == ".psd":
                window.advanced.show_settings()
                app.processEvents()
                window.advanced.grab().save(str(root / "layer-settings.png"))
                window.advanced.close()
            target = root / ("processed" + extension)
            if target.exists():
                target.unlink()
            window.save_to(target)
            wait_idle(app, window)
            report[extension] = {"source": str(source), "saved": str(target), "layers": len(window.document.layers)}
            if extension == ".psd":
                real_pipeline = MosaicPipeline(Path("models"))
                actual = real_pipeline.analyze_image(on_white(window.original), SETTINGS, source_path=source)
                report["real_model"] = {"detections": len(actual.detections), "shape": list(actual.mask.shape)}
                region, _ = real_pipeline.select_mask_region(actual.image_bgr, (320, 240), SETTINGS.mask_threshold)
                report["real_model"]["point_region_pixels"] = int(region.sum())
                assert region.shape == actual.mask.shape
            window.close()
            app.processEvents()
    main_window = AutoMosaicWindow()
    main_window.show()
    main_window._set_preview_bgr(actual.image_bgr)
    for name, size in (("normal", (1260, 790)), ("minimum", (980, 650))):
        main_window.resize(*size)
        app.processEvents()
        main_window._render_preview()
        main_window.grab().save(str(root / f"image-entry-{name}.png"))
    main_window.close()
    # A focused large-image comparison against the old display path (native export).
    from auto_mosaic.layers import LayerDocument
    for extension in (".psd", ".clip"):
        source = root / ("latency" + extension)
        make_layers(source, 2048, 1536)
        document = LayerDocument(source)
        try:
            mask = np.zeros((1536, 2048), bool)
            mask[384:1152, 512:1536] = True
            prepared = document.prepare(mask, frozenset({1}), SETTINGS)
            visible = frozenset({0})
            start = time.perf_counter()
            document.render(mask, frozenset({1}), SETTINGS, visible)
            old = time.perf_counter() - start
            start = time.perf_counter()
            prepared.images(visible)
            cold = time.perf_counter() - start
            start = time.perf_counter()
            prepared.images(visible)
            warm = time.perf_counter() - start
            report[extension]["display_seconds"] = {"old_export": old, "first_cached_composite": cold, "repeat": warm}
        finally:
            document.close()
    (root / "qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("LAYER_QA_OK", root)


if __name__ == "__main__":
    main()
