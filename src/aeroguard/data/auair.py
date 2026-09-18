"""Small, strict helpers for the AU-AIR annotation contract.

The archive is JSON rather than COCO-shaped data.  In particular, two source
keys are literal typos (``image_width:`` and ``longtitude``), labels are
zero-based, and the millisecond field is an offset rather than a fractional
second.  Keeping those rules here makes the detector and replay code consume
one canonical representation.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SOURCE_CLASS_NAMES = (
    "Human",
    "Car",
    "Truck",
    "Van",
    "Motorbike",
    "Bicycle",
    "Bus",
    "Trailer",
)
STATE_FEATURE_NAMES = (
    "altitude_m",
    "linear_x_mps",
    "linear_y_mps",
    "linear_z_mps",
    "roll_rad",
    "pitch_rad",
    "yaw_sin",
    "yaw_cos",
)

_IMAGE_RE = re.compile(
    r"^frame_(?P<root>\d{14})_(?P<stream>x|xx)_(?P<index>\d+)\.(?P<suffix>[^.]+)$",
    re.IGNORECASE,
)


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present key, including keys whose value is ``None``."""

    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _float_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def image_name(record: Mapping[str, Any]) -> str:
    value = _first(record, "image_name", "filename", "image")
    if value is None:
        raise ValueError("AU-AIR record has no image_name")
    return Path(str(value)).name


def recording_root_from_image(name: str) -> str:
    """Extract the shared 14-digit recording root from an AU-AIR filename."""

    match = _IMAGE_RE.match(Path(str(name)).name)
    if match is None:
        raise ValueError(f"not an AU-AIR frame filename: {name!r}")
    return match.group("root")


def stream_from_image(name: str) -> str:
    match = _IMAGE_RE.match(Path(str(name)).name)
    if match is None:
        raise ValueError(f"not an AU-AIR frame filename: {name!r}")
    return match.group("stream").lower()


def frame_index_from_image(name: str) -> int:
    match = _IMAGE_RE.match(Path(str(name)).name)
    if match is None:
        raise ValueError(f"not an AU-AIR frame filename: {name!r}")
    return int(match.group("index"))


def recording_root(record_or_name: Mapping[str, Any] | str) -> str:
    if isinstance(record_or_name, Mapping):
        explicit = _first(record_or_name, "recording_root", "recording_id")
        if explicit is not None:
            return str(explicit)
        return recording_root_from_image(image_name(record_or_name))
    return recording_root_from_image(str(record_or_name))


def reconstruct_timestamp(record_or_time: Mapping[str, Any]) -> datetime:
    """Reconstruct source time while preserving millisecond offsets > 999."""

    time = _mapping(record_or_time.get("time")) if isinstance(record_or_time, Mapping) else {}
    source = time if time else record_or_time
    fields = {
        "year": int(_first(source, "year", default=0)),
        "month": int(_first(source, "month", default=1)),
        "day": int(_first(source, "day", default=1)),
        "hour": int(_first(source, "hour", default=0)),
        "minute": int(_first(source, "min", "minute", default=0)),
        "second": int(_first(source, "sec", "second", default=0)),
    }
    milliseconds = _float_or_nan(_first(source, "ms", "milliseconds", default=0))
    if not math.isfinite(milliseconds):
        raise ValueError("AU-AIR timestamp ms must be finite")
    # AU-AIR has no timezone field; preserve the documented naive source time.
    return datetime(**fields) + timedelta(milliseconds=milliseconds)  # noqa: DTZ001


def canonicalize_state(record_or_state: Mapping[str, Any]) -> dict[str, float]:
    """Return the eight detector state features in documented SI/radian units.

    Missing or malformed fields remain ``nan`` so callers can create an
    explicit availability mask.  They are not silently replaced here; the
    model-side encoder performs its own pre-MLP sanitization.
    """

    nested = _mapping(record_or_state.get("state")) if isinstance(record_or_state, Mapping) else {}
    source = nested if nested else record_or_state
    if "altitude" in source or "altitude_mm" in source:
        altitude_mm = _float_or_nan(_first(source, "altitude", "altitude_mm"))
        altitude_m = altitude_mm * 0.001 if math.isfinite(altitude_mm) else math.nan
    else:
        altitude_m = _float_or_nan(_first(source, "altitude_m"))
    yaw = _float_or_nan(_first(source, "angle_psi", "yaw", "yaw_rad"))
    return {
        "altitude_m": altitude_m,
        "linear_x_mps": _float_or_nan(_first(source, "linear_x", "linear_x_mps")),
        "linear_y_mps": _float_or_nan(_first(source, "linear_y", "linear_y_mps")),
        "linear_z_mps": _float_or_nan(_first(source, "linear_z", "linear_z_mps")),
        "roll_rad": _float_or_nan(_first(source, "angle_phi", "roll", "roll_rad")),
        "pitch_rad": _float_or_nan(_first(source, "angle_theta", "pitch", "pitch_rad")),
        "yaw_sin": math.sin(yaw) if math.isfinite(yaw) else math.nan,
        "yaw_cos": math.cos(yaw) if math.isfinite(yaw) else math.nan,
    }


def state_vector(record_or_state: Mapping[str, Any]) -> list[float]:
    if not isinstance(record_or_state, Mapping):
        values = list(record_or_state)  # type: ignore[arg-type]
        if len(values) != len(STATE_FEATURE_NAMES):
            raise ValueError("numeric state vectors must have eight features")
        return [float(value) for value in values]
    state = record_or_state if all(name in record_or_state for name in STATE_FEATURE_NAMES) else canonicalize_state(record_or_state)
    return [float(state[name]) for name in STATE_FEATURE_NAMES]


def state_is_finite(record_or_state: Mapping[str, Any]) -> bool:
    return all(math.isfinite(value) for value in state_vector(record_or_state))


def source_to_model_label(source_label: Any, *, offset: int = 1, num_classes: int = 8) -> int:
    """Map AU-AIR's foreground IDs to a background-reserving model ID."""

    try:
        numeric = int(source_label)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid AU-AIR class: {source_label!r}") from exc
    try:
        exact_numeric = numeric == float(source_label)
    except (TypeError, ValueError):
        exact_numeric = False
    if not exact_numeric or not 0 <= numeric < num_classes:
        raise ValueError(f"AU-AIR class outside 0..{num_classes - 1}: {source_label!r}")
    return numeric + offset


@dataclass(frozen=True)
class RejectedBox:
    index: int
    box: Any
    reasons: tuple[str, ...]


@dataclass
class BoxCleaningResult:
    boxes: list[tuple[float, float, float, float]]
    labels: list[int]
    rejected: list[RejectedBox]

    def __iter__(self) -> Iterator[Any]:
        # Convenient for callers that prefer ``boxes, labels, rejected = ...``.
        yield self.boxes
        yield self.labels
        yield self.rejected


def clean_boxes(
    raw_boxes: Iterable[Mapping[str, Any]] | None,
    *,
    image_width: float | None = None,
    image_height: float | None = None,
    label_offset: int = 1,
    num_classes: int = len(SOURCE_CLASS_NAMES),
    clip_to_image: bool = True,
) -> BoxCleaningResult:
    """Convert ``top,left,height,width`` boxes to finite XYXY targets.

    Malformed, non-finite, non-positive and out-of-range annotations are
    rejected individually.  Clipping is conservative and keeps a frame even
    when all of its boxes are rejected, which is important for empty-target
    detector paths.
    """

    boxes: list[tuple[float, float, float, float]] = []
    labels: list[int] = []
    rejected: list[RejectedBox] = []
    for index, raw in enumerate(raw_boxes or []):
        reasons: list[str] = []
        if not isinstance(raw, Mapping):
            rejected.append(RejectedBox(index, raw, ("malformed",)))
            continue
        top = _float_or_nan(_first(raw, "top"))
        left = _float_or_nan(_first(raw, "left"))
        height = _float_or_nan(_first(raw, "height"))
        width = _float_or_nan(_first(raw, "width"))
        if not all(math.isfinite(value) for value in (top, left, height, width)):
            reasons.append("non_finite")
        if math.isfinite(width) and math.isfinite(height) and (width <= 0 or height <= 0):
            reasons.append("non_positive_size")
        try:
            label = source_to_model_label(
                _first(raw, "class", "label"), offset=label_offset, num_classes=num_classes
            )
        except ValueError:
            label = -1
            reasons.append("invalid_class")
        if reasons:
            rejected.append(RejectedBox(index, dict(raw), tuple(dict.fromkeys(reasons))))
            continue

        x1, y1, x2, y2 = left, top, left + width, top + height
        if clip_to_image:
            if image_width is not None and math.isfinite(float(image_width)):
                x1, x2 = max(0.0, min(x1, float(image_width))), max(0.0, min(x2, float(image_width)))
            if image_height is not None and math.isfinite(float(image_height)):
                y1, y2 = max(0.0, min(y1, float(image_height))), max(0.0, min(y2, float(image_height)))
        if x2 <= x1 or y2 <= y1:
            rejected.append(RejectedBox(index, dict(raw), ("non_positive_size_after_clipping",)))
            continue
        boxes.append((x1, y1, x2, y2))
        labels.append(label)
    return BoxCleaningResult(boxes, labels, rejected)


@dataclass
class FrameRecord:
    image_name: str
    image_path: str | None
    recording_root: str
    stream_id: str
    frame_index: int
    timestamp: datetime
    image_width: int
    image_height: int
    boxes: list[tuple[float, float, float, float]]
    labels: list[int]
    rejected_boxes: list[RejectedBox]
    raw_state: dict[str, Any]
    state: dict[str, float]

    @property
    def frame_id(self) -> str:
        return self.image_name

    @property
    def state_available(self) -> bool:
        return state_is_finite(self.state)

    @property
    def longitude(self) -> float:
        return _float_or_nan(_first(self.raw_state, "longtitude", "longitude"))

    @property
    def latitude(self) -> float:
        return _float_or_nan(_first(self.raw_state, "latitude"))

    def target(self) -> dict[str, Any]:
        return {"boxes": self.boxes, "labels": self.labels}


def parse_auair_record(record: Mapping[str, Any], image_root: str | Path | None = None) -> FrameRecord:
    name = image_name(record)
    width = _float_or_nan(_first(record, "image_width:", "image_width", "width"))
    height = _float_or_nan(_first(record, "image_height", "height"))
    if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
        raise ValueError(f"invalid image dimensions for {name!r}")
    cleaned = clean_boxes(
        _first(record, "bbox", "boxes", default=[]),
        image_width=width,
        image_height=height,
    )
    root = recording_root(record)
    path = str(Path(image_root) / name) if image_root is not None else None
    raw_state = dict(record)
    return FrameRecord(
        image_name=name,
        image_path=path,
        recording_root=root,
        stream_id=stream_from_image(name),
        frame_index=frame_index_from_image(name),
        timestamp=reconstruct_timestamp(record),
        image_width=int(width),
        image_height=int(height),
        boxes=cleaned.boxes,
        labels=cleaned.labels,
        rejected_boxes=cleaned.rejected,
        raw_state=raw_state,
        state=canonicalize_state(record),
    )


parse_record = parse_auair_record
parse_timestamp = reconstruct_timestamp
canonical_state = canonicalize_state
clean_annotation_boxes = clean_boxes
group_recording_root = recording_root


def load_auair_records(path: str | Path, image_root: str | Path | None = None) -> list[FrameRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        for key in ("records", "annotations", "data"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise TypeError("AU-AIR annotation JSON must contain a list of records")
    return [parse_auair_record(record, image_root=image_root) for record in payload]


parse_auair_annotations = load_auair_records
