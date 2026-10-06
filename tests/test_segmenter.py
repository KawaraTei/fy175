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


def _segmenter_and_embedding():
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

    return segmenter, (encoder_outputs, (8, 8))


def test_box_prompt_includes_positive_center_point() -> None:
    segmenter, embedding = _segmenter_and_embedding()
    candidates = segmenter.mask_candidates_from_box(embedding, (2, 2, 6, 6))

    feed = segmenter.decoder.feed
    assert feed is not None
    assert feed["point_labels"].tolist() == [[1.0, 2.0, 3.0]]
    assert feed["point_coords"].tolist() == [[[4.0, 4.0], [2.0, 2.0], [6.0, 6.0]]]
    assert len(candidates) == 3
    assert candidates[1][1] > candidates[0][1]


def test_mask_threshold_changes_pixel_inclusion_without_changing_scores() -> None:
    segmenter, embedding = _segmenter_and_embedding()
    box = (2, 2, 6, 6)
    wide = segmenter.mask_candidates_from_box(embedding, box, -0.5)
    default = segmenter.mask_candidates_from_box(embedding, box)
    narrow = segmenter.mask_candidates_from_box(embedding, box, 1.0)
    for index in range(3):
        assert wide[index][0].sum() == 64
        assert default[index][0].sum() == 9
        assert narrow[index][0].sum() == 0
        assert wide[index][1] == default[index][1] == narrow[index][1]
    assert np.array_equal(segmenter.mask_from_box(embedding, box, -0.5), wide[1][0])


def test_point_prompt_scales_coordinates_and_uses_mask_threshold() -> None:
    segmenter, embedding = _segmenter_and_embedding()
    embedding = (embedding[0], (16, 32))
    wide = segmenter.mask_candidates_from_point(embedding, (16, 8), -0.5)
    narrow = segmenter.mask_candidates_from_point(embedding, (16, 8), 1.0)
    assert segmenter.decoder.feed["point_coords"].tolist() == [[[4, 4], [0, 0]]]
    assert segmenter.decoder.feed["point_labels"].tolist() == [[1, -1]]
    assert wide[0][0].shape == (16, 32)
    assert wide[0][0].all()
    assert not narrow[0][0].any()
