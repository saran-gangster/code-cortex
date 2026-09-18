import math

import pytest

from aeroguard.evaluation.calibration import evaluate_calibration


def test_perfectly_calibrated_binary_scores_have_zero_error():
    summary = evaluate_calibration([1.0, 0.0], [True, False], bin_count=2)

    assert summary.sample_count == 2
    assert summary.brier_score == 0.0
    assert summary.expected_calibration_error == 0.0
    assert summary.bins[-1].count == 1  # score 1.0 belongs to the final bin
    assert summary.scope == "emitted_detections_only"


def test_empty_emissions_are_unavailable_not_perfect():
    summary = evaluate_calibration([], [], bin_count=3)

    assert summary.sample_count == 0
    assert summary.brier_score is None
    assert summary.expected_calibration_error is None
    assert all(item.count == 0 for item in summary.bins)


def test_overconfident_error_has_expected_brier_and_ece():
    summary = evaluate_calibration([0.9], [False], bin_count=10)

    assert summary.brier_score == pytest.approx(0.81)
    assert summary.expected_calibration_error == pytest.approx(0.9)


@pytest.mark.parametrize("scores", [[math.nan], [math.inf], [-0.1], [1.1]])
def test_invalid_scores_are_rejected(scores):
    with pytest.raises(ValueError, match="finite values"):
        evaluate_calibration(scores, [True])


def test_shape_types_and_bin_count_are_strict():
    with pytest.raises(ValueError, match="same length"):
        evaluate_calibration([0.5], [])
    with pytest.raises(ValueError, match="booleans"):
        evaluate_calibration([0.5], [1])
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_calibration([], [], bin_count=True)
