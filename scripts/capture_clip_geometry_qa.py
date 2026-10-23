"""Focused coordinate, brush and display-latency evidence (synthetic CLIP)."""
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from auto_mosaic.layer_ui import LayerMosaicWindow
from auto_mosaic.layers import LayerDocument, on_white
from tests.layer_fixtures import make_layers, move_clip_layer
from tests.test_layer_ui import synthetic_analysis, wait_idle
from tests.test_layers import SETTINGS


def main():
    root = Path(".codex-qa/layers/coordinates").resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    if Path("C:/Windows/Fonts/meiryo.ttc").is_file():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/meiryo.ttc")
    source = root / "moved.clip"
    make_layers(source, 640, 480)
    move_clip_layer(source, 100, -40, 20, 10)
    window = LayerMosaicWindow(Path("models"), SETTINGS)
    inputs = []
    def analysis(image, settings, *, source_path):
        inputs.append(image.copy())
        return synthetic_analysis(image, settings, source_path=source_path)
    window.pipeline.analyze_image = analysis
    window.show()
    with patch.object(QMessageBox, "warning", side_effect=lambda *args: (_ for _ in ()).throw(AssertionError(args))):
        window.load_path(source)
        wait_idle(app, window)
        reader = window.document._read(source)
        old = np.frombuffer(reader.merged_image(), np.uint8).reshape(480, 640, 4)
        del reader
        Image.fromarray(on_white(old)[..., ::-1]).save(root / "before-native.png")
        Image.fromarray(on_white(window.original)[..., ::-1]).save(root / "after-composite.png")
        np.testing.assert_array_equal(inputs[0], on_white(window.original))
        window.resize(1260, 790)
        window.view.setCurrentText("元画像")
        app.processEvents()
        window.grab().save(str(root / "after-preview.png"))
        window.layers.topLevelItem(1).setCheckState(1, Qt.CheckState.Unchecked)
        wait_idle(app, window)
        window.edit_button.click()
        window.brush_size.setValue(40)
        rect = window.preview.image_rect
        point = (rect.topLeft() + QPointF(rect.width() * .78, rect.height() * .35)).toPoint()
        QTest.mouseClick(window.preview, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
        app.processEvents()
        assert window.mask[168, 499]
        window.grab().save(str(root / "after-mask-edit.png"))
        window.edit_button.click()
        wait_idle(app, window)
        window.grab().save(str(root / "after-processed.png"))
        output = root / "saved.clip"
        output.unlink(missing_ok=True)
        window.save_to(output)
        wait_idle(app, window)
        reopened = LayerDocument(output)
        np.testing.assert_array_equal(reopened.original, window.result.bgra)
        reopened.close()
    window.close()
    app.processEvents()
    source = root / "latency.clip"
    make_layers(source, 2048, 1536)
    move_clip_layer(source, -137, 79, 12, -9)
    doc = LayerDocument(source)
    times = []
    states = (frozenset({0}), frozenset({1}), frozenset(), frozenset({0, 1}))
    first = []
    for state in states:
        start = time.perf_counter()
        doc.display.image(state)
        first.append(time.perf_counter() - start)
    with patch.object(doc, "_read", side_effect=AssertionError("visibility read")), patch(
        "auto_mosaic.clip_geometry.ClipReader", side_effect=AssertionError("visibility decode")
    ):
        for i in range(100):
            start = time.perf_counter()
            doc.display.image(states[i % len(states)])
            times.append(time.perf_counter() - start)
    report = {"qt_platform": app.platformName(), "canvas": [2048, 1536], "layers": 2,
              "first_composition_seconds": first, "cached_median_seconds": float(np.median(times)),
              "cached_max_seconds": max(times), "comparison": "same selected layers, 640x480",
              "checks": ["detector input equals displayed canvas", "QTest brush at canvas (499,168)",
                         "saved CLIP reopened equals processed preview"],
              "limits": "Synthetic fixtures only; CSP and reported file not available. Timings exclude screen presentation."}
    doc.close()
    (root / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    main()
