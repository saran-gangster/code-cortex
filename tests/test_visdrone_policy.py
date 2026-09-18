import pytest

from aeroguard.evaluation.visdrone import (
    VISDRONE_TO_AUAIR,
    VisDroneRow,
    map_visdrone_category,
    match_visdrone_detections,
    parse_visdrone_det_row,
)


def prediction(box, label="Car", score=0.9):
    return {"box_xyxy": box, "label": label, "score": score}


def test_strict_parser_accepts_det_row_and_converts_box():
    row = parse_visdrone_det_row("10,20,30,40,0,4,0,2", line_number=3)
    assert row == VisDroneRow(10.0, 20.0, 30.0, 40.0, 0.0, 4, 0, 2)
    assert row.box_xyxy == (10.0, 20.0, 40.0, 60.0)
    assert row.mapped_label == "Car"


def test_parser_accepts_one_observed_trailing_comma_but_not_two():
    assert parse_visdrone_det_row("10,20,30,40,0,4,0,2,").mapped_label == "Car"
    with pytest.raises(ValueError, match="at most one trailing comma"):
        parse_visdrone_det_row("10,20,30,40,0,4,0,2,,")


@pytest.mark.parametrize(
    "row",
    [
        "1,2,3,4,0,4,0",
        "1,2,3,4,0,4,0,0,extra",
        "1,2,-3,4,0,4,0,0",
        "1,2,3,4,nan,4,0,0",
        "1,2,3,4,0,4.0,0,0",
        "1,2,3,4,0,12,0,0",
        "1,2,3,4,0,4,2,0",
    ],
)
def test_malformed_rows_are_rejected(row):
    with pytest.raises((TypeError, ValueError)):
        parse_visdrone_det_row(row)


def test_frozen_seven_concept_mapping_and_explicit_ignore_policy():
    assert VISDRONE_TO_AUAIR == {
        1: "Human", 2: "Human", 3: "Bicycle", 4: "Car",
        5: "Van", 6: "Truck", 9: "Bus", 10: "Motorbike",
    }
    assert map_visdrone_category(0) is None
    assert map_visdrone_category(7) is None
    assert map_visdrone_category(8) is None
    assert map_visdrone_category(11) is None


def test_valid_target_has_precedence_over_overlapping_ignore_region():
    # Category 0 overlaps the valid Car.  The valid target must win.
    result = match_visdrone_detections(
        [prediction((0, 0, 10, 10))],
        ["0,0,10,10,0,4,0,0", "0,0,10,10,0,0,0,0"],
    )
    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.ignored_predictions == 0
    assert result.prediction_status == ("tp",)


def test_prediction_substantially_overlapping_ignored_region_is_suppressed():
    result = match_visdrone_detections(
        [prediction((1, 1, 9, 9))],
        ["0,0,10,10,0,0,0,0"],
    )
    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.ignored_predictions == 1


def test_non_overlapping_prediction_remains_false_positive():
    result = match_visdrone_detections(
        [prediction((20, 20, 30, 30))],
        ["0,0,10,10,0,0,0,0", "0,0,10,10,0,11,0,0"],
    )
    assert result.false_positives == 1
    assert result.ignored_predictions == 0
