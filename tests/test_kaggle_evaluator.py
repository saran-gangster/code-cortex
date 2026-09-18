import pytest

from aeroguard.evaluation.predictions import normalize_model_prediction


def test_normalize_model_prediction_keeps_valid_output():
    assert normalize_model_prediction([1, 2, 4, 8], 2, 0.75) == {
        "box_xyxy": [1.0, 2.0, 4.0, 8.0],
        "label": 2,
        "score": 0.75,
    }


@pytest.mark.parametrize("box", ([1, 2, 1, 8], [1, 2, 4, 2]))
def test_normalize_model_prediction_drops_only_degenerate_output(box):
    assert normalize_model_prediction(box, 1, 0.5) is None


@pytest.mark.parametrize(
    ("box", "label", "score", "message"),
    [
        ([1, 2, float("nan"), 8], 1, 0.5, "finite"),
        ([1, 2, 4, 8], 0, 0.5, "positive"),
        ([1, 2, 4, 8], 1, 1.5, r"\[0, 1\]"),
    ],
)
def test_normalize_model_prediction_rejects_other_invalid_output(box, label, score, message):
    with pytest.raises(ValueError, match=message):
        normalize_model_prediction(box, label, score)
