from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from auto_mosaic.domain import Detection
from auto_mosaic.model_catalog import DetectorSpec


class YoloOnnxDetector:
    def __init__(self, model_path: Path, spec: DetectorSpec) -> None:
        self.spec = spec
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        self.input_height = int(shape[2]) if isinstance(shape[2], int) else spec.input_size
        self.input_width = int(shape[3]) if isinstance(shape[3], int) else spec.input_size
        self.output_names = [item.name for item in self.session.get_outputs()]

    def detect(
        self,
        image_bgr: np.ndarray,
        targets: frozenset[str],
        confidence_threshold: float,
        iou_threshold: float,
    ) -> list[Detection]:
        tensor, scale, pad_x, pad_y = self._prepare_input(image_bgr)
        raw = self.session.run(self.output_names, {self.input_name: tensor})[0]
        predictions = self._as_rows(raw)

        wanted_indices = {
            index
            for name, indices in self.spec.target_indices.items()
            if name in targets
            for index in indices
        }
        boxes_xywh: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        original_height, original_width = image_bgr.shape[:2]

        for row in predictions:
            if row.size < 4 + len(self.spec.class_names):
                continue

            class_scores = row[4 : 4 + len(self.spec.class_names)]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if class_id not in wanted_indices or confidence < confidence_threshold:
                continue

            center_x, center_y, width, height = (float(value) for value in row[:4])
            x1 = int(round((center_x - width / 2 - pad_x) / scale))
            y1 = int(round((center_y - height / 2 - pad_y) / scale))
            x2 = int(round((center_x + width / 2 - pad_x) / scale))
            y2 = int(round((center_y + height / 2 - pad_y) / scale))
            x1 = max(0, min(original_width - 1, x1))
            y1 = max(0, min(original_height - 1, y1))
            x2 = max(x1 + 1, min(original_width, x2))
            y2 = max(y1 + 1, min(original_height, y2))

            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(confidence)
            class_ids.append(class_id)

        if not boxes_xywh:
            return []

        selected = cv2.dnn.NMSBoxes(
            boxes_xywh, scores, confidence_threshold, iou_threshold
        )
        selected_indices = np.asarray(selected).reshape(-1).tolist()
        return [
            Detection(
                class_name=self._public_name(class_ids[index]),
                confidence=scores[index],
                box=(
                    boxes_xywh[index][0],
                    boxes_xywh[index][1],
                    boxes_xywh[index][0] + boxes_xywh[index][2],
                    boxes_xywh[index][1] + boxes_xywh[index][3],
                ),
            )
            for index in selected_indices
        ]

    def _prepare_input(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        original_height, original_width = image_bgr.shape[:2]
        if self.spec.pad_before_resize:
            square_size = max(original_height, original_width)
            square = np.full(
                (square_size, square_size, 3),
                self.spec.padding_value,
                dtype=np.uint8,
            )
            square[:original_height, :original_width] = image_bgr
            canvas = cv2.resize(
                square,
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_LINEAR,
            )
            scale = self.input_width / square_size
            return self._to_tensor(canvas), scale, 0, 0

        scale = min(
            self.input_width / original_width, self.input_height / original_height
        )
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        resized = cv2.resize(
            image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )

        canvas = np.full(
            (self.input_height, self.input_width, 3),
            self.spec.padding_value,
            dtype=np.uint8,
        )
        pad_x = (
            (self.input_width - resized_width) // 2
            if self.spec.centered_letterbox
            else 0
        )
        pad_y = (
            (self.input_height - resized_height) // 2
            if self.spec.centered_letterbox
            else 0
        )
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        return self._to_tensor(canvas), scale, pad_x, pad_y

    def _to_tensor(self, image_bgr: np.ndarray) -> np.ndarray:
        model_image = (
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if self.spec.swap_red_blue
            else image_bgr
        )
        return model_image.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0

    @staticmethod
    def _as_rows(output: np.ndarray) -> np.ndarray:
        array = np.asarray(output)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"Unexpected YOLO output shape: {output.shape}")
        if array.shape[0] < array.shape[1] and array.shape[0] <= 128:
            array = array.T
        return array

    def _public_name(self, class_id: int) -> str:
        raw_name = self.spec.class_names[class_id]
        if raw_name in {"pussy", "FEMALE_GENITALIA_EXPOSED"}:
            return "vagina"
        if raw_name == "MALE_GENITALIA_EXPOSED":
            return "penis"
        return raw_name
