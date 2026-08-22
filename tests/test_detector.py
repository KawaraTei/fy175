from __future__ import annotations

import numpy as np

from auto_mosaic.detector import YoloOnnxDetector
from auto_mosaic.model_catalog import DetectorSpec


class _FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def run(self, _output_names, _feed):
        return [self.output]


def test_as_rows_transposes_yolo_channel_first_output() -> None:
    output = np.zeros((1, 9, 8400), dtype=np.float32)
    rows = YoloOnnxDetector._as_rows(output)
    assert rows.shape == (8400, 9)


def test_as_rows_keeps_prediction_rows() -> None:
    output = np.zeros((1, 8400, 9), dtype=np.float32)
    rows = YoloOnnxDetector._as_rows(output)
    assert rows.shape == (8400, 9)


def test_detect_returns_only_five_best_below_threshold_candidates() -> None:
    detector = object.__new__(YoloOnnxDetector)
    detector.spec = DetectorSpec(
        filename="fake.onnx",
        class_names=("penis",),
        target_indices={"penis": (0,)},
        input_size=100,
    )
    detector.input_name = "images"
    detector.output_names = ["output"]
    scores = [0.9, 0.49, 0.4, 0.3, 0.2, 0.1, 0.05]
    rows = np.array(
        [[10 + index * 12, 10, 8, 8, score] for index, score in enumerate(scores)],
        dtype=np.float32,
    )
    detector.session = _FakeSession(rows[None, ...])
    detector._prepare_input = lambda _image: (  # type: ignore[method-assign]
        np.zeros((1, 3, 100, 100), dtype=np.float32),
        1.0,
        0,
        0,
    )

    detections = detector.detect(
        np.zeros((100, 100, 3), dtype=np.uint8),
        frozenset({"penis"}),
        confidence_threshold=0.5,
        iou_threshold=0.35,
        below_threshold_limit=5,
    )

    assert len(detections) == 6
    assert detections[0].confidence >= 0.5
    assert [round(item.confidence, 2) for item in detections[1:]] == [
        0.49,
        0.4,
        0.3,
        0.2,
        0.1,
    ]
