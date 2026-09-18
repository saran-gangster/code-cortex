"""Judge-friendly fixed-threshold metrics for object detection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .matching import match_detections


def evaluate_operating_points(
    records: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
    *,
    score_floor: float = 0.0,
    iou_threshold: float = 0.5,
    max_detections_per_image: int = 300,
) -> list[dict[str, int | float]]:
    """Aggregate TP/FP/FN and derived metrics at each score threshold.

    The returned ``detection_accuracy`` is the Jaccard-style
    ``TP / (TP + FP + FN)``. It is intentionally named so it cannot be
    mistaken for image-classification accuracy.
    """

    if max_detections_per_image < 1:
        raise ValueError("max_detections_per_image must be positive")
    if not 0.0 <= score_floor <= 1.0:
        raise ValueError("score_floor must be in [0, 1]")
    prepared: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    for record in records:
        predictions = [
            item
            for item in record.get("predictions", record.get("detections", []))
            if float(item["score"]) >= score_floor
        ]
        predictions = sorted(
            predictions,
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:max_detections_per_image]
        targets = list(record.get("targets", record.get("ground_truth", [])))
        prepared.append((predictions, targets))
    points: list[dict[str, int | float]] = []
    for threshold_value in thresholds:
        threshold = float(threshold_value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("score thresholds must be in [0, 1]")
        if threshold < score_floor:
            raise ValueError("score thresholds cannot be lower than score_floor")
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for predictions, targets in prepared:
            retained = [item for item in predictions if float(item["score"]) >= threshold]
            match = match_detections(
                retained,
                targets,
                iou_threshold=iou_threshold,
            )
            true_positives += match.true_positives
            false_positives += match.false_positives
            false_negatives += match.false_negatives

        precision_denominator = true_positives + false_positives
        recall_denominator = true_positives + false_negatives
        f1_denominator = 2 * true_positives + false_positives + false_negatives
        accuracy_denominator = true_positives + false_positives + false_negatives
        points.append(
            {
                "score_threshold": threshold,
                "iou_threshold": float(iou_threshold),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": (
                    true_positives / precision_denominator if precision_denominator else 0.0
                ),
                "recall": true_positives / recall_denominator if recall_denominator else 0.0,
                "f1": 2 * true_positives / f1_denominator if f1_denominator else 0.0,
                "detection_accuracy": (
                    true_positives / accuracy_denominator if accuracy_denominator else 0.0
                ),
            }
        )
    return points


def select_best_f1_point(points: Sequence[dict[str, int | float]]) -> dict[str, int | float]:
    """Select the development operating point with deterministic tie-breaking."""

    if not points:
        raise ValueError("at least one operating point is required")
    return dict(
        max(
            points,
            key=lambda point: (
                float(point["f1"]),
                float(point["detection_accuracy"]),
                float(point["precision"]),
                float(point["score_threshold"]),
            ),
        )
    )


__all__ = ["evaluate_operating_points", "select_best_f1_point"]
