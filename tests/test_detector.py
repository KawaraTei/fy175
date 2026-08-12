from __future__ import annotations

import numpy as np

from auto_mosaic.detector import YoloOnnxDetector


def test_as_rows_transposes_yolo_channel_first_output() -> None:
    output = np.zeros((1, 9, 8400), dtype=np.float32)
    rows = YoloOnnxDetector._as_rows(output)
    assert rows.shape == (8400, 9)


def test_as_rows_keeps_prediction_rows() -> None:
    output = np.zeros((1, 8400, 9), dtype=np.float32)
    rows = YoloOnnxDetector._as_rows(output)
    assert rows.shape == (8400, 9)
