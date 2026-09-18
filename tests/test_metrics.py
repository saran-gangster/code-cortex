import numpy as np
import pytest

from aeroguard.evaluation.metrics import IOU_THRESHOLDS, evaluate_detections


def target(label=1, box=(0, 0, 10, 10)):
    return {"box_xyxy": list(box), "label": label}


def prediction(score=0.9, label=1, box=(0, 0, 10, 10)):
    return {"box_xyxy": list(box), "label": label, "score": score}


def record(predictions, targets, root="r1", image_id="f1"):
    return {"image_id": image_id, "recording_root": root, "predictions": predictions, "targets": targets}


def test_perfect_detection_has_unit_ap_and_recall():
    result = evaluate_detections([record([prediction()], [target()])])
    assert result.pooled.ap50 == pytest.approx(1.0)
    assert result.pooled.ap50_95 == pytest.approx(1.0)
    assert result.pooled.recall == pytest.approx(1.0)
    assert result.pooled.precision == pytest.approx(1.0)
    assert result.pooled.f1 == pytest.approx(1.0)
    assert result.pooled.detection_accuracy == pytest.approx(1.0)
    assert result.pooled.per_class[1].support == 1
    assert tuple(result.iou_thresholds) == tuple(IOU_THRESHOLDS)

    payload = result.as_dict()["pooled"]
    assert payload["precision"] == pytest.approx(1.0)
    assert payload["f1"] == pytest.approx(1.0)
    assert payload["detection_accuracy"] == pytest.approx(1.0)


def test_empty_predictions_have_zero_metrics_and_support_is_target_count():
    result = evaluate_detections([record([], [target()])])
    assert result.pooled.support == 1
    assert result.pooled.ap50 == result.pooled.ap50_95 == result.pooled.recall == 0.0
    assert result.pooled.false_negatives == 1


def test_duplicate_is_one_true_positive_and_one_false_positive():
    result = evaluate_detections([record([prediction(0.9), prediction(0.8)], [target()])])
    assert result.pooled.true_positives == 1
    assert result.pooled.false_positives == 1
    assert result.pooled.ap50 == pytest.approx(1.0)
    assert result.pooled.precision == pytest.approx(0.5)
    assert result.pooled.f1 == pytest.approx(2 / 3)
    assert result.pooled.detection_accuracy == pytest.approx(0.5)


def test_wrong_class_is_not_a_match():
    result = evaluate_detections([record([prediction(label=2)], [target(label=1)])])
    assert result.pooled.true_positives == 0
    assert result.pooled.false_positives == 1
    assert result.pooled.false_negatives == 1


def test_score_floor_and_per_image_cap_are_applied_before_matching():
    detections = [prediction(0.95, label=2), prediction(0.90), prediction(0.80)]
    result = evaluate_detections([record(detections, [target()])], score_floor=0.85, max_detections_per_image=1)
    assert result.pooled.true_positives == 0
    assert result.pooled.false_positives == 1
    assert result.pooled.false_negatives == 1


def test_score_order_controls_duplicate_matching_and_ap():
    detections = [prediction(0.9, box=(20, 20, 30, 30)), prediction(0.8)]
    result = evaluate_detections([record(detections, [target()])])
    assert result.pooled.true_positives == 1
    assert result.pooled.false_positives == 1
    assert result.pooled.ap50 == pytest.approx(0.5)


def test_prediction_only_class_does_not_change_supported_class_macro_ap():
    result = evaluate_detections([
        record([prediction(), prediction(0.7, label=2)], [target()])
    ])

    assert result.pooled.ap50 == pytest.approx(1.0)
    assert result.pooled.per_class[2].support == 0
    assert result.pooled.false_positives == 1


def test_per_recording_root_aggregation_is_separate_from_pooled():
    result = evaluate_detections([
        record([prediction()], [target()], root="a", image_id="a1"),
        record([], [target()], root="b", image_id="b1"),
    ])
    assert set(result.by_recording_root) == {"a", "b"}
    assert result.by_recording_root["a"].recall == pytest.approx(1.0)
    assert result.by_recording_root["b"].recall == pytest.approx(0.0)
    assert result.pooled.recall == pytest.approx(0.5)


def test_single_recording_root_matches_pooled_values_and_keeps_root_name():
    result = evaluate_detections([
        record([prediction()], [target()], root="flight-a", image_id="a1")
    ])
    root = result.by_recording_root["flight-a"]

    assert root.name == "flight-a"
    assert root.support == result.pooled.support
    assert root.ap50 == result.pooled.ap50
    assert root.ap50_95 == result.pooled.ap50_95
    assert root.true_positives == result.pooled.true_positives


def test_fixed_operating_point_can_use_higher_score_threshold():
    result = evaluate_detections(
        [record([prediction(0.6)], [target()])], score_floor=0.0, fixed_score_threshold=0.7
    )
    assert result.pooled.ap50 == pytest.approx(1.0)
    assert result.pooled.recall == pytest.approx(0.0)


def test_fixed_threshold_cannot_recover_predictions_below_evaluation_floor():
    with pytest.raises(ValueError, match="cannot be lower"):
        evaluate_detections(
            [record([prediction(0.6)], [target()])],
            score_floor=0.7,
            fixed_score_threshold=0.5,
        )


@pytest.mark.parametrize("field,value", [("score", np.nan), ("box_xyxy", [0, 0, np.inf, 1])])
def test_nonfinite_detection_values_are_rejected(field, value):
    bad = prediction()
    bad[field] = value
    with pytest.raises(ValueError, match="finite"):
        evaluate_detections([record([bad], [target()])])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("score", 1.1, r"in \[0, 1\]"), ("box_xyxy", [0, 0, 0, 1], "positive area")],
)
def test_out_of_contract_detection_values_are_rejected(field, value, message):
    bad = prediction()
    bad[field] = value
    with pytest.raises(ValueError, match=message):
        evaluate_detections([record([bad], [target()])])
