"""Calibration diagnostics for emitted detections.

These metrics say nothing about objects for which the detector emitted no
prediction.  Correctness must therefore come from a declared one-to-one
matching policy (normally same-class IoU >= 0.5).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    empirical_accuracy: float | None
    absolute_gap: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationSummary:
    sample_count: int
    brier_score: float | None
    expected_calibration_error: float | None
    bins: tuple[ReliabilityBin, ...]
    scope: str = "emitted_detections_only"

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "bins": [item.to_dict() for item in self.bins],
            "scope": self.scope,
        }


def evaluate_calibration(
    scores: Iterable[float],
    correctness: Iterable[bool],
    *,
    bin_count: int = 10,
) -> CalibrationSummary:
    """Compute Brier score and equal-width expected calibration error.

    Bins are ``[lower, upper)`` except for the final bin, which includes 1.0.
    Empty input produces explicit unavailable metrics rather than numeric zero.
    """

    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count <= 0:
        raise ValueError("bin_count must be a positive integer")

    score_values = tuple(float(value) for value in scores)
    correctness_values = tuple(correctness)
    if len(score_values) != len(correctness_values):
        raise ValueError("scores and correctness must have the same length")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in score_values):
        raise ValueError("scores must be finite values in [0, 1]")
    if any(type(value) is not bool for value in correctness_values):
        raise ValueError("correctness values must be booleans")

    buckets: list[list[int]] = [[] for _ in range(bin_count)]
    for index, score in enumerate(score_values):
        bucket_index = min(int(score * bin_count), bin_count - 1)
        buckets[bucket_index].append(index)

    bins: list[ReliabilityBin] = []
    weighted_gap = 0.0
    total = len(score_values)
    for bucket_index, members in enumerate(buckets):
        lower = bucket_index / bin_count
        upper = (bucket_index + 1) / bin_count
        if not members:
            bins.append(ReliabilityBin(lower, upper, 0, None, None, None))
            continue
        confidence = sum(score_values[index] for index in members) / len(members)
        accuracy = sum(correctness_values[index] for index in members) / len(members)
        gap = abs(confidence - accuracy)
        weighted_gap += len(members) * gap
        bins.append(ReliabilityBin(lower, upper, len(members), confidence, accuracy, gap))

    if total == 0:
        return CalibrationSummary(0, None, None, tuple(bins))

    brier = sum(
        (score - float(correct)) ** 2
        for score, correct in zip(score_values, correctness_values, strict=True)
    ) / total
    return CalibrationSummary(total, brier, weighted_gap / total, tuple(bins))
