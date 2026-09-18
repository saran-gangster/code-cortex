"""Strict, versioned inference records shared by the API and replay UI."""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CLASS_NAMES = ("Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer")
HexSha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class StrictModel(BaseModel):
    # JSON enum tokens are deliberately accepted at the transport boundary; all
    # semantic fields remain closed-world and are validated below.
    model_config = ConfigDict(extra="forbid")


class PredictionSource(str, Enum):
    computed = "computed"
    cached = "cached"
    fixture = "fixture"


class InputMode(str, Enum):
    paired_state = "paired_state"
    state_masked = "state_masked"
    separate_rgb_fallback = "separate_rgb_fallback"


class MetadataAlignment(str, Enum):
    paired_annotation = "paired_annotation"
    independently_timestamped = "independently_timestamped"
    unavailable = "unavailable"
    injected_delay = "injected_delay"
    injected_permutation = "injected_permutation"


class OriginalSize(StrictModel):
    width: Annotated[int, Field(ge=1)]
    height: Annotated[int, Field(ge=1)]

    @field_validator("width", "height", mode="before")
    @classmethod
    def strict_integer(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            # Pydantic converts ValueError into a structured 422 response; TypeError escapes FastAPI.
            raise ValueError("image dimensions must be integers")  # noqa: TRY004
        return value


class Detection(StrictModel):
    class_name: Literal["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]
    box_xyxy: tuple[float, float, float, float]
    raw_score: float = Field(ge=0, le=1)
    calibrated_score: float | None = Field(default=None, ge=0, le=1)
    track_id: str | None = None

    @field_validator("raw_score", "calibrated_score", mode="before")
    @classmethod
    def strict_number(cls, value: Any) -> Any:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("scores must be numbers")
        return value

    @field_validator("box_xyxy", "raw_score", "calibrated_score")
    @classmethod
    def finite(cls, value: Any) -> Any:
        values = value if isinstance(value, tuple) else (value,)
        if any(not math.isfinite(float(item)) for item in values if item is not None):
            raise ValueError("inference values must be finite")
        return value


class Latency(StrictModel):
    preprocess_ms: float | None = Field(default=None, ge=0)
    model_ms: float | None = Field(default=None, ge=0)
    postprocess_ms: float | None = Field(default=None, ge=0)
    end_to_end_ms: float | None = Field(default=None, ge=0)

    @field_validator("preprocess_ms", "model_ms", "postprocess_ms", "end_to_end_ms", mode="before")
    @classmethod
    def strict_number(cls, value: Any) -> Any:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("latency values must be numbers")
        return value

    @field_validator("preprocess_ms", "model_ms", "postprocess_ms", "end_to_end_ms")
    @classmethod
    def finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("latency values must be finite")
        return value


QUALITY_FLAGS = {
    "STATE_MISSING", "STATE_INVALID", "INJECTED_DELAY", "INJECTED_PERMUTATION",
    "REVIEW_REQUESTED", "FRAME_GAP", "CALIBRATION_UNAVAILABLE",
}


class InferenceRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    frame_id: Annotated[str, Field(min_length=1)]
    model_id: Annotated[str, Field(min_length=1)]
    checkpoint_sha256: HexSha256
    protocol_sha256: HexSha256
    prediction_source: PredictionSource
    source_time_ms: int | None
    original_size: OriginalSize
    input_mode: InputMode
    metadata_alignment: MetadataAlignment
    detections: list[Detection]
    quality_flags: list[str]
    latency: Latency

    @field_validator("source_time_ms")
    @classmethod
    def valid_time(cls, value: int | None) -> int | None:
        if value is not None and not isinstance(value, int):
            raise ValueError("source_time_ms must be an integer or null")
        return value

    @field_validator("quality_flags")
    @classmethod
    def valid_flags(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(flag not in QUALITY_FLAGS for flag in value):
            raise ValueError("quality_flags must be unique known flags")
        return value

    @model_validator(mode="after")
    def validate_boxes(self) -> InferenceRecord:
        width, height = self.original_size.width, self.original_size.height
        for detection in self.detections:
            x1, y1, x2, y2 = detection.box_xyxy
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("detection boxes must be finite and within original image bounds")
        return self


class InferenceRequest(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    run_id: Annotated[str, Field(min_length=1)] = "live"
    image_base64: str | None = Field(default=None, max_length=8_000_000)
    original_size: OriginalSize
    source_time_ms: int | None = None
    input_mode: InputMode = InputMode.separate_rgb_fallback
    metadata_alignment: MetadataAlignment = MetadataAlignment.unavailable
    state: list[float] | None = None
    review_requested: bool = False

    @field_validator("state", mode="before")
    @classmethod
    def finite_state(cls, value: list[float] | None) -> list[float] | None:
        if value is not None:
            if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
                raise ValueError("state values must be numbers")
            if any(not math.isfinite(float(item)) for item in value):
                raise ValueError("state values must be finite")
        return value


class ReviewRequest(StrictModel):
    run_id: Annotated[str, Field(min_length=1)]
    frame_id: Annotated[str, Field(min_length=1)]
    decision: Literal["accept", "reject", "needs_review"]
    comment: Annotated[str, Field(max_length=4000)] = ""


class Review(ReviewRequest):
    review_id: Annotated[str, Field(min_length=1)]
    created_at: Annotated[str, Field(min_length=1)]
