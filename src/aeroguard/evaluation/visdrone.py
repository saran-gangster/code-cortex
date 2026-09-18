"""Frozen AU-AIR to VisDrone DET evaluation policy.

VisDrone DET annotations are eight comma-separated fields::

    left, top, width, height, score, category, truncation, occlusion

This module deliberately has no torch dependency.  It contains only the
dataset contract, parsing, and the small matching policy used by the
external RGB-only evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# VisDrone's official category ids.  This is the frozen seven-concept adapter;
# Trailer has no VisDrone counterpart, and tricycle variants are not silently
# forced into Bicycle or Motorbike.
VISDRONE_TO_AUAIR: dict[int, str] = {
    1: "Human",      # pedestrian
    2: "Human",      # people
    3: "Bicycle",
    4: "Car",
    5: "Van",
    6: "Truck",
    9: "Bus",
    10: "Motorbike",  # motor
}

IGNORED_VISDRONE_CATEGORIES = frozenset({0, 7, 8, 11})
VISDRONE_IGNORED_REGION = 0
VISDRONE_OTHERS = 11
_MAX_CATEGORY = 11


@dataclass(frozen=True)
class VisDroneRow:
    """One validated VisDrone DET annotation row."""

    left: float
    top: float
    width: float
    height: float
    score: float
    category_id: int
    truncation: int
    occlusion: int

    @property
    def box_xyxy(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)

    @property
    def mapped_label(self) -> str | None:
        return VISDRONE_TO_AUAIR.get(self.category_id)

    @property
    def is_ignored(self) -> bool:
        return self.category_id in IGNORED_VISDRONE_CATEGORIES


def _parse_float(value: str, field: str, line_number: int | None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        location = f" on line {line_number}" if line_number is not None else ""
        raise ValueError(f"invalid {field}{location}: {value!r}") from exc
    if not math.isfinite(result):
        location = f" on line {line_number}" if line_number is not None else ""
        raise ValueError(f"{field} must be finite{location}")
    return result


def _parse_int(value: str, field: str, line_number: int | None) -> int:
    # int("1.0") must not pass: accepting it hides damaged DET files.
    if not value or value.strip() != value or value.lstrip("+-").isdigit() is False:
        location = f" on line {line_number}" if line_number is not None else ""
        raise ValueError(f"invalid integer {field}{location}: {value!r}")
    try:
        return int(value)
    except ValueError as exc:
        location = f" on line {line_number}" if line_number is not None else ""
        raise ValueError(f"invalid integer {field}{location}: {value!r}") from exc


def parse_visdrone_det_row(row: str, *, line_number: int | None = None) -> VisDroneRow:
    """Parse one strict VisDrone DET row.

    Blank/comment rows are not annotation rows and are rejected here so a
    caller cannot accidentally treat a malformed file as a valid empty scene.
    Coordinates use the source ``left, top, width, height`` convention.
    """

    if not isinstance(row, str):
        raise TypeError("VisDrone DET row must be a string")
    if not row or not row.strip() or row.lstrip().startswith("#"):
        location = f" on line {line_number}" if line_number is not None else ""
        raise ValueError(f"blank/comment VisDrone DET row{location}")
    fields = row.rstrip("\r\n").split(",")
    if len(fields) == 9 and fields[-1] == "":
        fields.pop()
    if len(fields) != 8:
        location = f" on line {line_number}" if line_number is not None else ""
        raise ValueError(
            f"VisDrone DET row requires 8 fields with at most one trailing comma{location}"
        )
    if any(field.strip() != field or field == "" for field in fields):
        location = f" on line {line_number}" if line_number is not None else ""
        raise ValueError(f"VisDrone DET fields must not be empty or padded{location}")

    left = _parse_float(fields[0], "left", line_number)
    top = _parse_float(fields[1], "top", line_number)
    width = _parse_float(fields[2], "width", line_number)
    height = _parse_float(fields[3], "height", line_number)
    score = _parse_float(fields[4], "score", line_number)
    category_id = _parse_int(fields[5], "category", line_number)
    truncation = _parse_int(fields[6], "truncation", line_number)
    occlusion = _parse_int(fields[7], "occlusion", line_number)
    if left < 0 or top < 0 or width <= 0 or height <= 0:
        raise ValueError("VisDrone box must have non-negative origin and positive size")
    if not 0 <= category_id <= _MAX_CATEGORY:
        raise ValueError(f"VisDrone category must be in [0, {_MAX_CATEGORY}]")
    if truncation not in (0, 1) or occlusion not in (0, 1, 2):
        raise ValueError("VisDrone truncation must be 0/1 and occlusion must be 0/1/2")
    return VisDroneRow(left, top, width, height, score, category_id, truncation, occlusion)


# Short aliases keep call sites readable while retaining one canonical parser.
parse_visdrone_row = parse_visdrone_det_row


def map_visdrone_category(category_id: int) -> str | None:
    """Return the AU-AIR concept or ``None`` under the explicit ignore policy."""

    if not isinstance(category_id, int) or isinstance(category_id, bool):
        raise TypeError("VisDrone category_id must be an integer")
    if category_id < 0 or category_id > _MAX_CATEGORY:
        raise ValueError(f"VisDrone category must be in [0, {_MAX_CATEGORY}]")
    return VISDRONE_TO_AUAIR.get(category_id)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _prediction_box(prediction: Mapping[str, object]) -> Sequence[float]:
    box = prediction.get("box_xyxy")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("prediction requires box_xyxy with four coordinates")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values) or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("prediction box_xyxy must be finite and have positive area")
    return values


def _prediction_label(prediction: Mapping[str, object]) -> str:
    label = prediction.get("label", prediction.get("class_name"))
    if not isinstance(label, str) or not label:
        raise ValueError("prediction requires a non-empty string label")
    return label


def _as_row(annotation: VisDroneRow | str | Mapping[str, object]) -> VisDroneRow:
    if isinstance(annotation, VisDroneRow):
        return annotation
    if isinstance(annotation, str):
        return parse_visdrone_det_row(annotation)
    if not isinstance(annotation, Mapping):
        raise TypeError("annotation must be a VisDroneRow, DET row, or mapping")
    if "category_id" not in annotation:
        raise ValueError("annotation mapping requires category_id")
    box = annotation.get("box_xyxy")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("annotation mapping requires box_xyxy with four coordinates")
    left, top, right, bottom = (float(value) for value in box)
    return VisDroneRow(left, top, right - left, bottom - top, 1.0, int(annotation["category_id"]), 0, 0)


@dataclass(frozen=True)
class VisDroneMatchSummary:
    true_positives: int
    false_positives: int
    false_negatives: int
    ignored_predictions: int
    prediction_status: tuple[str, ...]


def match_visdrone_detections(
    predictions: Iterable[Mapping[str, object]],
    annotations: Iterable[VisDroneRow | str | Mapping[str, object]],
    *,
    iou_threshold: float = 0.5,
    ignore_overlap_threshold: float = 0.5,
) -> VisDroneMatchSummary:
    """Match AU-AIR-labelled predictions against the frozen VisDrone policy.

    All valid mapped targets are considered before ignore suppression.  Thus a
    prediction intersecting both a valid target and an ignored box remains a
    true positive when it satisfies the valid target match.  Ignore overlap is
    intersection divided by *prediction area*, which suppresses detections
    substantially contained in an ignored region without hiding a large,
    unrelated false positive.
    """

    if not 0 < iou_threshold <= 1 or not 0 < ignore_overlap_threshold <= 1:
        raise ValueError("overlap thresholds must be in (0, 1]")
    rows = [_as_row(annotation) for annotation in annotations]
    valid = [row for row in rows if row.mapped_label is not None]
    ignored = [row for row in rows if row.is_ignored or row.mapped_label is None]
    ordered = sorted(predictions, key=lambda item: float(item["score"]), reverse=True)
    unmatched = set(range(len(valid)))
    statuses: list[str] = []
    for prediction in ordered:
        box = _prediction_box(prediction)
        label = _prediction_label(prediction)
        candidates = [
            (index, _iou(box, target.box_xyxy))
            for index, target in enumerate(valid)
            if index in unmatched and target.mapped_label == label
        ]
        if candidates:
            index, overlap = max(candidates, key=lambda item: item[1])
            if overlap >= iou_threshold:
                unmatched.remove(index)
                statuses.append("tp")
                continue
        prediction_area = (box[2] - box[0]) * (box[3] - box[1])
        suppressed = any(
            _intersection(box, row.box_xyxy) / prediction_area >= ignore_overlap_threshold
            for row in ignored
        )
        statuses.append("ignored" if suppressed else "fp")
    return VisDroneMatchSummary(
        true_positives=statuses.count("tp"),
        false_positives=statuses.count("fp"),
        false_negatives=len(unmatched),
        ignored_predictions=statuses.count("ignored"),
        prediction_status=tuple(statuses),
    )


def _intersection(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


evaluate_visdrone_detections = match_visdrone_detections


__all__ = [
    "IGNORED_VISDRONE_CATEGORIES",
    "VISDRONE_IGNORED_REGION",
    "VISDRONE_OTHERS",
    "VISDRONE_TO_AUAIR",
    "VisDroneMatchSummary",
    "VisDroneRow",
    "evaluate_visdrone_detections",
    "map_visdrone_category",
    "match_visdrone_detections",
    "parse_visdrone_det_row",
    "parse_visdrone_row",
]
