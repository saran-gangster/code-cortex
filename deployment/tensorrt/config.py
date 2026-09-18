"""Strict, dependency-free validation for the TensorRT deployment manifest."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


class ConfigError(ValueError):
    """Raised when a deployment manifest violates the supported contract."""


@dataclass(frozen=True)
class ShapeProfile:
    minimum: tuple[int, int, int, int]
    optimum: tuple[int, int, int, int]
    maximum: tuple[int, int, int, int]

    @property
    def dynamic(self) -> bool:
        return not (self.minimum == self.optimum == self.maximum)


@dataclass(frozen=True)
class DeploymentConfig:
    source: Path
    raw: dict[str, Any]
    min_size: int
    max_size: int
    size_divisible: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    state_mean: tuple[float, ...]
    state_std: tuple[float, ...]
    normalizer_sha256: str
    precision: str
    workspace_mib: int
    image_profile: ShapeProfile
    score_threshold: float
    nms_threshold: float
    topk_candidates: int
    detections_per_image: int

    @property
    def state_dim(self) -> int:
        return len(self.state_mean)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be an object")
    return value


def _integer(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_vector(value: Any, field: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ConfigError(f"{field} must contain exactly {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ConfigError(f"{field} must contain only finite values")
    return result


def _shape(value: Any, field: str, divisor: int) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ConfigError(f"{field} must be [1, 3, H, W]")
    shape = tuple(_integer(item, field) for item in value)
    if shape[:2] != (1, 3):
        raise ConfigError(f"{field} must fix batch/channels to [1, 3]")
    if shape[2] % divisor or shape[3] % divisor:
        raise ConfigError(f"{field} H and W must be multiples of {divisor}")
    return shape


def _probability(value: Any, field: str, *, upper_inclusive: bool = False) -> float:
    number = float(value)
    upper_ok = number <= 1.0 if upper_inclusive else number < 1.0
    if not math.isfinite(number) or number < 0.0 or not upper_ok:
        suffix = "]" if upper_inclusive else ")"
        raise ConfigError(f"{field} must be in [0, 1{suffix}")
    return number


def validate_config(raw: Any, *, source: Path = Path("<memory>")) -> DeploymentConfig:
    root = _mapping(raw, "manifest")
    if root.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")

    model = _mapping(root.get("model"), "model")
    if model.get("architecture") != "aeroguard_fcos_resnet50_fpn_gated_film":
        raise ConfigError("model.architecture is unsupported")
    if _integer(model.get("num_classes"), "model.num_classes") != 9:
        raise ConfigError("this repository checkpoint contract requires exactly 9 classes")
    if _integer(model.get("state_dim"), "model.state_dim") != 8:
        raise ConfigError("this repository checkpoint contract requires state_dim=8")
    if _integer(model.get("feature_levels"), "model.feature_levels") != 5:
        raise ConfigError("this FCOS adapter requires exactly 5 feature levels")
    min_size = _integer(model.get("min_size"), "model.min_size")
    max_size = _integer(model.get("max_size"), "model.max_size")
    if min_size > max_size:
        raise ConfigError("model.min_size cannot exceed model.max_size")
    size_divisible = _integer(model.get("size_divisible"), "model.size_divisible")

    preprocessing = _mapping(root.get("preprocessing"), "preprocessing")
    image_mean = _finite_vector(preprocessing.get("image_mean"), "preprocessing.image_mean", 3)
    image_std = _finite_vector(preprocessing.get("image_std"), "preprocessing.image_std", 3)
    if any(item <= 0 for item in image_std):
        raise ConfigError("preprocessing.image_std values must be positive")
    state_mean = _finite_vector(preprocessing.get("state_mean"), "preprocessing.state_mean", 8)
    state_std = _finite_vector(preprocessing.get("state_std"), "preprocessing.state_std", 8)
    if tuple(preprocessing.get("state_feature_names", ())) != STATE_FEATURE_NAMES:
        raise ConfigError(
            "preprocessing.state_feature_names must exactly match the trained feature order"
        )
    if any(item <= 0 for item in state_std):
        raise ConfigError("preprocessing.state_std values must be positive")
    normalizer_sha256 = preprocessing.get("normalizer_sha256")
    if not isinstance(normalizer_sha256, str) or len(normalizer_sha256) != 64:
        raise ConfigError("preprocessing.normalizer_sha256 must be a 64-character digest")
    try:
        int(normalizer_sha256, 16)
    except ValueError as exc:
        raise ConfigError("preprocessing.normalizer_sha256 must be hexadecimal") from exc

    build = _mapping(root.get("build"), "build")
    precision = build.get("precision")
    if precision not in {"fp32", "fp16"}:
        raise ConfigError("build.precision must be fp32 or fp16; INT8 has no calibration path")
    workspace_mib = _integer(build.get("workspace_mib"), "build.workspace_mib")
    profiles = _mapping(build.get("profiles"), "build.profiles")
    if profiles.get("state") != [1, 8]:
        raise ConfigError("build.profiles.state must be fixed at [1, 8]")
    if profiles.get("state_mask") != [1, 1]:
        raise ConfigError("build.profiles.state_mask must be fixed at [1, 1]")
    images = _mapping(profiles.get("images"), "build.profiles.images")
    profile = ShapeProfile(
        _shape(images.get("min"), "build.profiles.images.min", size_divisible),
        _shape(images.get("opt"), "build.profiles.images.opt", size_divisible),
        _shape(images.get("max"), "build.profiles.images.max", size_divisible),
    )
    for axis, label in ((2, "H"), (3, "W")):
        if not profile.minimum[axis] <= profile.optimum[axis] <= profile.maximum[axis]:
            raise ConfigError(f"image profile must satisfy min <= opt <= max for {label}")
    if max(profile.maximum[2:]) > math.ceil(max_size / size_divisible) * size_divisible:
        raise ConfigError("image profile exceeds the configured FCOS max_size padding envelope")

    postprocess = _mapping(root.get("postprocess"), "postprocess")
    score_threshold = _probability(postprocess.get("score_threshold"), "postprocess.score_threshold")
    nms_threshold = _probability(
        postprocess.get("nms_threshold"), "postprocess.nms_threshold", upper_inclusive=True
    )
    topk_candidates = _integer(postprocess.get("topk_candidates"), "postprocess.topk_candidates")
    detections = _integer(
        postprocess.get("detections_per_image"), "postprocess.detections_per_image"
    )

    return DeploymentConfig(
        source=source,
        raw=root,
        min_size=min_size,
        max_size=max_size,
        size_divisible=size_divisible,
        image_mean=image_mean,
        image_std=image_std,
        state_mean=state_mean,
        state_std=state_std,
        normalizer_sha256=normalizer_sha256,
        precision=precision,
        workspace_mib=workspace_mib,
        image_profile=profile,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        topk_candidates=topk_candidates,
        detections_per_image=detections,
    )


def load_config(path: str | Path) -> DeploymentConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"deployment config is not a regular file: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read deployment config {source}: {exc}") from exc
    return validate_config(raw, source=source)


__all__ = [
    "STATE_FEATURE_NAMES",
    "ConfigError",
    "DeploymentConfig",
    "ShapeProfile",
    "load_config",
    "validate_config",
]
