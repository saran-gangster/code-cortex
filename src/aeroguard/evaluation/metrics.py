"""Pure NumPy detection metrics used by AeroGuard.

The public entry point is :func:`evaluate_detections`.  Its input is a sequence
of frame records.  Each record contains ``predictions`` and ``targets`` (or the
aliases ``detections`` and ``ground_truth``), whose items contain
``box_xyxy`` and ``label``; predictions additionally contain ``score``.

Predictions below ``score_floor`` are discarded, then the highest scoring
``max_detections_per_image`` predictions are retained independently per image.
Ties are resolved by original input order.  Matching is one-to-one, same-image,
same-class, greedy in descending score order.  AP uses the deterministic COCO
101-point interpolated precision envelope at IoU 0.50 and 0.50:0.05:0.95.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05, dtype=np.float64)


@dataclass(frozen=True)
class ClassMetrics:
    """Metrics for one class and one aggregate scope."""

    label: int
    support: int
    ap50: float
    ap50_95: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.true_positives + self.false_positives + self.false_negatives
        return 2 * self.true_positives / denominator if denominator else 0.0

    @property
    def detection_accuracy(self) -> float:
        """Return TP / (TP + FP + FN), the detection analogue requested by judges."""

        denominator = self.true_positives + self.false_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0


@dataclass(frozen=True)
class AggregateMetrics:
    """Pooled or per-recording-root metrics."""

    name: str
    support: int
    ap50: float
    ap50_95: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    per_class: dict[int, ClassMetrics]

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
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


@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation, including pooled and recording-root aggregates."""

    score_floor: float
    max_detections_per_image: int
    iou_thresholds: tuple[float, ...]
    fixed_score_threshold: float
    fixed_iou_threshold: float
    pooled: AggregateMetrics
    by_recording_root: dict[str, AggregateMetrics]

    @property
    def ap50(self) -> float:
        return self.pooled.ap50

    @property
    def ap50_95(self) -> float:
        return self.pooled.ap50_95

    @property
    def fixed_operating_point_recall(self) -> float:
        return self.pooled.recall

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-friendly metrics without losing the configured semantics."""

        def aggregate(value: AggregateMetrics) -> dict[str, Any]:
            def class_metrics(item: ClassMetrics) -> dict[str, Any]:
                return {
                    **vars(item),
                    "precision": item.precision,
                    "f1": item.f1,
                    "detection_accuracy": item.detection_accuracy,
                }

            return {
                "name": value.name,
                "support": value.support,
                "ap50": value.ap50,
                "ap50_95": value.ap50_95,
                "precision": value.precision,
                "recall": value.recall,
                "f1": value.f1,
                "detection_accuracy": value.detection_accuracy,
                "true_positives": value.true_positives,
                "false_positives": value.false_positives,
                "false_negatives": value.false_negatives,
                "per_class": {str(k): class_metrics(v) for k, v in value.per_class.items()},
            }

        return {
            "score_floor": self.score_floor,
            "max_detections_per_image": self.max_detections_per_image,
            "iou_thresholds": list(self.iou_thresholds),
            "fixed_operating_point": {
                "score_threshold": self.fixed_score_threshold,
                "iou_threshold": self.fixed_iou_threshold,
            },
            "pooled": aggregate(self.pooled),
            "by_recording_root": {k: aggregate(v) for k, v in self.by_recording_root.items()},
        }


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _box(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite [x1, y1, x2, y2] box") from exc
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite [x1, y1, x2, y2] box")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{name} must have positive area")
    return result


def _label(value: Any, name: str) -> int:
    number = _finite_number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection_wh = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(intersection_wh))
    area_a = float(np.prod(np.maximum(0.0, a[2:] - a[:2])))
    area_b = float(np.prod(np.maximum(0.0, b[2:] - b[:2])))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _item(item: Mapping[str, Any], *, prediction: bool, index: int) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise TypeError(f"detection/target {index} must be a mapping")
    box_value = item.get("box_xyxy", item.get("bbox"))
    if box_value is None:
        raise ValueError(f"detection/target {index} is missing box_xyxy")
    result: dict[str, Any] = {
        "box": _box(box_value, f"detection/target {index}.box_xyxy"),
        "label": _label(item.get("label", item.get("class_id")), f"detection/target {index}.label"),
        "index": index,
    }
    if prediction:
        result["score"] = _finite_number(item.get("score"), f"detection {index}.score")
        if not 0.0 <= result["score"] <= 1.0:
            raise ValueError(f"detection {index}.score must be in [0, 1]")
    if result["label"] < 1:
        raise ValueError(f"detection/target {index}.label must be a positive model label")
    return result


def _records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {record_index} must be a mapping")
        image_id = record.get("image_id", record.get("frame_id", record_index))
        root = str(record.get("recording_root", record.get("root", "default")))
        predictions = record.get("predictions", record.get("detections", []))
        targets = record.get("targets", record.get("ground_truth", []))
        if predictions is None or targets is None:
            raise ValueError(f"record {record_index} predictions and targets must be sequences")
        normalized.append({
            "image_id": str(image_id),
            "recording_root": root,
            "predictions": [_item(x, prediction=True, index=i) for i, x in enumerate(predictions)],
            "targets": [_item(x, prediction=False, index=i) for i, x in enumerate(targets)],
        })
    return normalized


def _filter_and_cap(records: list[dict[str, Any]], score_floor: float, cap: int) -> list[dict[str, Any]]:
    result = []
    for record in records:
        kept = [x for x in record["predictions"] if x["score"] >= score_floor]
        kept = sorted(kept, key=lambda x: (-x["score"], x["index"]))[:cap]
        result.append({**record, "predictions": kept})
    return result


def _ap(correct: np.ndarray, scores: np.ndarray, support: int) -> float:
    if support == 0 or correct.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    tp = np.cumsum(correct[order], dtype=np.float64)
    fp = np.cumsum(~correct[order], dtype=np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(support)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    sampled = [envelope[recall >= point][0] if np.any(recall >= point) else 0.0 for point in np.linspace(0, 1, 101)]
    return float(np.mean(sampled))


def _match_label_records(
    records: list[dict[str, Any]], threshold: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Match records whose prediction and target lists already contain one class."""

    entries: list[tuple[float, int, bool]] = []
    support = 0
    for record_index, record in enumerate(records):
        targets = record["targets"]
        support += len(targets)
        unmatched = set(range(len(targets)))
        for prediction in sorted(
            record["predictions"], key=lambda x: (-x["score"], x["index"])
        ):
            candidates = [(j, _iou(prediction["box"], target["box"])) for j, target in enumerate(targets) if j in unmatched]
            match = max(candidates, key=lambda pair: pair[1]) if candidates else (-1, 0.0)
            correct = match[1] >= threshold
            if correct:
                unmatched.remove(match[0])
            entries.append((prediction["score"], record_index, correct))
    entries.sort(key=lambda x: (-x[0], x[1]))
    if not entries:
        return np.empty(0, dtype=bool), np.empty(0, dtype=np.float64), support
    return np.asarray([x[2] for x in entries], dtype=bool), np.asarray([x[0] for x in entries], dtype=np.float64), support


def _aggregate(
    records: list[dict[str, Any]],
    name: str,
    thresholds: np.ndarray,
    fixed_iou: float,
    fixed_floor: float,
) -> AggregateMetrics:
    labels = sorted({x["label"] for r in records for x in r["targets"]} | {x["label"] for r in records for x in r["predictions"]})
    per_class: dict[int, ClassMetrics] = {}
    for label in labels:
        label_records = [
            {
                **record,
                "predictions": [
                    item for item in record["predictions"] if item["label"] == label
                ],
                "targets": [item for item in record["targets"] if item["label"] == label],
            }
            for record in records
        ]
        fixed_records = _filter_and_cap(
            label_records,
            fixed_floor,
            max((len(r["predictions"]) for r in label_records), default=1),
        )
        aps = []
        fixed_correct, _, support = _match_label_records(fixed_records, fixed_iou)
        for threshold in thresholds:
            correct, scores, _ = _match_label_records(label_records, float(threshold))
            aps.append(_ap(correct, scores, support))
        tp = int(fixed_correct.sum())
        fp = int(fixed_correct.size - tp)
        per_class[label] = ClassMetrics(label, support, aps[0], float(np.mean(aps)), tp / support if support else 0.0, tp, fp, support - tp)
    support = sum(x.support for x in per_class.values())
    tp = sum(x.true_positives for x in per_class.values())
    fp = sum(x.false_positives for x in per_class.values())
    fn = sum(x.false_negatives for x in per_class.values())
    supported = [item for item in per_class.values() if item.support > 0]
    return AggregateMetrics(
        name,
        support,
        float(np.mean([item.ap50 for item in supported])) if supported else 0.0,
        float(np.mean([item.ap50_95 for item in supported])) if supported else 0.0,
        tp / support if support else 0.0,
        tp,
        fp,
        fn,
        per_class,
    )


def evaluate_detections(
    records: Sequence[Mapping[str, Any]],
    *,
    score_floor: float = 0.0,
    max_detections_per_image: int = 100,
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
    fixed_score_threshold: float | None = None,
    fixed_iou_threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate detections with pooled and per-recording-root reporting.

    ``fixed_score_threshold`` controls the reported operating-point recall. It
    defaults to ``score_floor``; unlike AP, it is a single recall value at
    ``fixed_iou_threshold`` after the same per-image cap.  Empty prediction
    sets and classes with no support have AP/recall zero.  Invalid or
    non-finite boxes and scores raise ``ValueError`` instead of being silently
    dropped.
    """
    floor = _finite_number(score_floor, "score_floor")
    fixed_floor = floor if fixed_score_threshold is None else _finite_number(fixed_score_threshold, "fixed_score_threshold")
    fixed_iou = _finite_number(fixed_iou_threshold, "fixed_iou_threshold")
    if not 0 <= floor <= 1 or not 0 <= fixed_floor <= 1:
        raise ValueError("score floors must be in [0, 1]")
    if fixed_floor < floor:
        raise ValueError("fixed_score_threshold cannot be lower than score_floor")
    if not 0 < fixed_iou <= 1:
        raise ValueError("fixed_iou_threshold must be in (0, 1]")
    if not isinstance(max_detections_per_image, (int, np.integer)) or max_detections_per_image < 1:
        raise ValueError("max_detections_per_image must be a positive integer")
    thresholds = np.asarray(iou_thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or thresholds.size == 0 or not np.isfinite(thresholds).all() or np.any((thresholds <= 0) | (thresholds > 1)):
        raise ValueError("iou_thresholds must be finite values in (0, 1]")
    normalized = _filter_and_cap(_records(records), floor, int(max_detections_per_image))
    pooled = _aggregate(normalized, "pooled", thresholds, fixed_iou, fixed_floor)
    roots = sorted({r["recording_root"] for r in normalized})
    if len(roots) == 1:
        # Pooled and per-root records are identical for a one-flight partition.
        by_root = {roots[0]: replace(pooled, name=roots[0])}
    else:
        by_root = {
            root: _aggregate(
                [r for r in normalized if r["recording_root"] == root],
                root,
                thresholds,
                fixed_iou,
                fixed_floor,
            )
            for root in roots
        }
    return EvaluationResult(floor, int(max_detections_per_image), tuple(float(x) for x in thresholds), fixed_floor, fixed_iou, pooled, by_root)


evaluate = evaluate_detections
evaluate_detection = evaluate_detections

__all__ = [
    "IOU_THRESHOLDS",
    "AggregateMetrics",
    "ClassMetrics",
    "EvaluationResult",
    "evaluate",
    "evaluate_detection",
    "evaluate_detections",
]
