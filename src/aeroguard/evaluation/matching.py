"""Small deterministic detection matcher used by metric/calibration fixtures."""

from __future__ import annotations

from dataclasses import dataclass


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class MatchSummary:
    true_positives: int
    false_positives: int
    false_negatives: int
    prediction_correctness: tuple[bool, ...]

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.true_positives + self.false_positives + self.false_negatives
        return 2 * self.true_positives / denominator if denominator else 0.0

    @property
    def detection_accuracy(self) -> float:
        """Return TP / (TP + FP + FN), not image-classification accuracy."""

        denominator = self.true_positives + self.false_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0


def match_detections(
    predictions: list[dict],
    targets: list[dict],
    *,
    iou_threshold: float = 0.5,
) -> MatchSummary:
    """Greedily match score-sorted predictions to same-class targets once."""

    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    ordered = sorted(predictions, key=lambda item: float(item["score"]), reverse=True)
    unmatched = set(range(len(targets)))
    correctness: list[bool] = []
    for prediction in ordered:
        candidates = [
            (index, _iou(prediction["box_xyxy"], target["box_xyxy"]))
            for index, target in enumerate(targets)
            if index in unmatched and target["label"] == prediction["label"]
        ]
        if not candidates:
            correctness.append(False)
            continue
        index, overlap = max(candidates, key=lambda item: item[1])
        if overlap >= iou_threshold:
            unmatched.remove(index)
            correctness.append(True)
        else:
            correctness.append(False)
    true_positives = sum(correctness)
    return MatchSummary(
        true_positives=true_positives,
        false_positives=len(correctness) - true_positives,
        false_negatives=len(unmatched),
        prediction_correctness=tuple(correctness),
    )
