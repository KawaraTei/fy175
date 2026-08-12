from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class Sam2OnnxSegmenter:
    """SAM2 ONNX box-prompt segmentation with one cached embedding per image."""

    def __init__(self, encoder_path: Path, decoder_path: Path) -> None:
        providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.encoder = ort.InferenceSession(
            str(encoder_path), sess_options=options, providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(decoder_path), sess_options=options, providers=providers
        )
        encoder_input = self.encoder.get_inputs()[0]
        self.encoder_input_name = encoder_input.name
        shape = encoder_input.shape
        self.input_height = int(shape[2])
        self.input_width = int(shape[3])
        self.encoder_output_names = [item.name for item in self.encoder.get_outputs()]
        self.decoder_input_names = [item.name for item in self.decoder.get_inputs()]
        self.decoder_output_names = [item.name for item in self.decoder.get_outputs()]

    def encode(self, image_bgr: np.ndarray) -> tuple[list[np.ndarray], tuple[int, int]]:
        original_size = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = ((resized / 255.0 - mean) / std).transpose(2, 0, 1)[None, ...]
        outputs = self.encoder.run(
            self.encoder_output_names, {self.encoder_input_name: tensor}
        )
        return outputs, original_size

    def mask_from_box(
        self,
        embedding: tuple[list[np.ndarray], tuple[int, int]],
        box: tuple[int, int, int, int],
    ) -> np.ndarray:
        candidates = self.mask_candidates_from_box(embedding, box)
        return max(candidates, key=lambda item: item[1])[0]

    def mask_candidates_from_box(
        self,
        embedding: tuple[list[np.ndarray], tuple[int, int]],
        box: tuple[int, int, int, int],
    ) -> list[tuple[np.ndarray, float]]:
        encoder_outputs, original_size = embedding
        original_height, original_width = original_size
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        point_coords = np.array(
            [[[center_x, center_y], [x1, y1], [x2, y2]]], dtype=np.float32
        )
        point_coords[..., 0] *= self.input_width / original_width
        point_coords[..., 1] *= self.input_height / original_height
        point_labels = np.array([[1, 2, 3]], dtype=np.float32)
        mask_input = np.zeros(
            (1, 1, self.input_height // 4, self.input_width // 4), dtype=np.float32
        )
        has_mask_input = np.array([0], dtype=np.float32)

        high_res_0, high_res_1, image_embedding = encoder_outputs[:3]
        values = (
            image_embedding,
            high_res_0,
            high_res_1,
            point_coords,
            point_labels,
            mask_input,
            has_mask_input,
        )
        feed = {
            name: value for name, value in zip(self.decoder_input_names, values, strict=True)
        }
        outputs = self.decoder.run(self.decoder_output_names, feed)
        masks = outputs[0][0]
        scores = np.asarray(outputs[1]).reshape(-1)
        candidates: list[tuple[np.ndarray, float]] = []
        for mask_logits, score in zip(masks, scores, strict=True):
            resized_mask = cv2.resize(
                mask_logits,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            )
            candidates.append((resized_mask > 0.0, float(score)))
        return candidates
