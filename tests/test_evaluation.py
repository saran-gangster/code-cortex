from aeroguard.evaluation import match_detections

TARGET = {"box_xyxy": [10, 10, 20, 20], "label": 2}
PREDICTION = {"box_xyxy": [10, 10, 20, 20], "label": 2, "score": 0.9}


def test_perfect_detection_fixture():
    result = match_detections([PREDICTION], [TARGET])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 0, 0)
    assert result.precision == result.recall == 1.0


def test_duplicate_prediction_is_false_positive():
    duplicate = {**PREDICTION, "score": 0.8}
    result = match_detections([PREDICTION, duplicate], [TARGET])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 1, 0)


def test_wrong_class_cannot_match():
    result = match_detections([{**PREDICTION, "label": 3}], [TARGET])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 1, 1)


def test_empty_predictions_and_targets_are_explicit():
    missed = match_detections([], [TARGET])
    empty = match_detections([], [])
    assert (missed.true_positives, missed.false_positives, missed.false_negatives) == (0, 0, 1)
    assert (empty.precision, empty.recall) == (0.0, 0.0)
