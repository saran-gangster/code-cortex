import pytest

from aeroguard.evaluation import evaluate_operating_points, select_best_f1_point


def prediction(score: float, box=(0, 0, 10, 10), label: int = 1) -> dict:
    return {"box_xyxy": list(box), "label": label, "score": score}


def target(box=(0, 0, 10, 10), label: int = 1) -> dict:
    return {"box_xyxy": list(box), "label": label}


def test_threshold_sweep_exposes_counts_and_detection_accuracy():
    records = [
        {
            "predictions": [
                prediction(0.9),
                prediction(0.4, box=(20, 20, 30, 30)),
            ],
            "targets": [target()],
        }
    ]

    low, high = evaluate_operating_points(records, [0.3, 0.8])

    assert low["true_positives"] == 1
    assert low["false_positives"] == 1
    assert low["false_negatives"] == 0
    assert low["precision"] == pytest.approx(0.5)
    assert low["recall"] == pytest.approx(1.0)
    assert low["f1"] == pytest.approx(2 / 3)
    assert low["detection_accuracy"] == pytest.approx(0.5)
    assert high["f1"] == pytest.approx(1.0)
    assert select_best_f1_point([low, high]) == high


def test_operating_point_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="score thresholds"):
        evaluate_operating_points([], [1.1])
    with pytest.raises(ValueError, match="positive"):
        evaluate_operating_points([], [0.5], max_detections_per_image=0)
    with pytest.raises(ValueError, match="score_floor"):
        evaluate_operating_points([], [0.2], score_floor=0.3)


def test_operating_point_supports_evaluator_aliases_and_floor():
    records = [
        {
            "detections": [prediction(0.9), prediction(0.1, box=(20, 20, 30, 30))],
            "ground_truth": [target()],
        }
    ]

    point = evaluate_operating_points(records, [0.3], score_floor=0.2)[0]

    assert point["true_positives"] == 1
    assert point["false_positives"] == 0
    assert point["f1"] == pytest.approx(1.0)
