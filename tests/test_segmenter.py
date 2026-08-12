import numpy as np

from auto_mosaic.segmenter import Sam2OnnxSegmenter


class _FakeDecoder:
    def __init__(self) -> None:
        self.feed = None

    def run(self, _output_names, feed):
        self.feed = feed
        masks = np.zeros((1, 3, 8, 8), dtype=np.float32)
        masks[0, :, 3:6, 3:6] = 1.0
        scores = np.array([[0.2, 0.8, 0.4]], dtype=np.float32)
        return [masks, scores]


def test_box_prompt_includes_positive_center_point() -> None:
    segmenter = Sam2OnnxSegmenter.__new__(Sam2OnnxSegmenter)
    segmenter.input_width = 8
    segmenter.input_height = 8
    segmenter.decoder_input_names = [
        "image_embed",
        "high_res_0",
        "high_res_1",
        "point_coords",
        "point_labels",
        "mask_input",
        "has_mask_input",
    ]
    segmenter.decoder_output_names = ["masks", "scores"]
    segmenter.decoder = _FakeDecoder()
    encoder_outputs = [
        np.zeros((1, 1, 1, 1), dtype=np.float32),
        np.zeros((1, 1, 1, 1), dtype=np.float32),
        np.zeros((1, 1, 1, 1), dtype=np.float32),
    ]

    candidates = segmenter.mask_candidates_from_box((encoder_outputs, (8, 8)), (2, 2, 6, 6))

    feed = segmenter.decoder.feed
    assert feed is not None
    assert feed["point_labels"].tolist() == [[1.0, 2.0, 3.0]]
    assert feed["point_coords"].tolist() == [[[4.0, 4.0], [2.0, 2.0], [6.0, 6.0]]]
    assert len(candidates) == 3
    assert candidates[1][1] > candidates[0][1]
