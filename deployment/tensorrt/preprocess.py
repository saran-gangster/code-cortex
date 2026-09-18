"""Deterministic CPU preprocessing for the exported AeroGuard contract."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DeploymentConfig


@dataclass(frozen=True)
class ImageTransform:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    padded_height: int
    padded_width: int
    scale_y: float
    scale_x: float


def _target_size(height: int, width: int, minimum: int, maximum: int) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(minimum / min(height, width), maximum / max(height, width))
    return max(1, int(height * scale)), max(1, int(width * scale))


def preprocess_image(path: str | Path, config: DeploymentConfig) -> tuple[np.ndarray, ImageTransform]:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise ValueError(f"image is not a regular file: {image_path}")
    try:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            original_width, original_height = image.size
            resized_height, resized_width = _target_size(
                original_height, original_width, config.min_size, config.max_size
            )
            image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot decode RGB image {image_path}: {exc}") from exc

    chw = np.transpose(pixels, (2, 0, 1))
    mean = np.asarray(config.image_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(config.image_std, dtype=np.float32)[:, None, None]
    chw = (chw - mean) / std
    divisor = config.size_divisible
    padded_height = math.ceil(resized_height / divisor) * divisor
    padded_width = math.ceil(resized_width / divisor) * divisor
    profile = config.image_profile
    if not (
        profile.minimum[2] <= padded_height <= profile.maximum[2]
        and profile.minimum[3] <= padded_width <= profile.maximum[3]
    ):
        raise ValueError(
            f"preprocessed shape [1,3,{padded_height},{padded_width}] is outside the TensorRT profile"
        )
    batch = np.zeros((1, 3, padded_height, padded_width), dtype=np.float32)
    batch[0, :, :resized_height, :resized_width] = chw
    transform = ImageTransform(
        original_height,
        original_width,
        resized_height,
        resized_width,
        padded_height,
        padded_width,
        resized_height / original_height,
        resized_width / original_width,
    )
    return np.ascontiguousarray(batch), transform


def preprocess_state(
    values: Sequence[float], state_valid: bool | float, config: DeploymentConfig
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) != config.state_dim:
        raise ValueError(f"flight state must contain exactly {config.state_dim} values")
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (config.state_dim,) or not np.isfinite(state).all():
        raise ValueError("flight state must contain only finite scalar values")
    if isinstance(state_valid, bool):
        valid = float(state_valid)
    else:
        valid = float(state_valid)
        if valid not in {0.0, 1.0}:
            raise ValueError("state-valid must be exactly 0 or 1")
    normalized = (state - np.asarray(config.state_mean, dtype=np.float32)) / np.asarray(
        config.state_std, dtype=np.float32
    )
    return normalized.reshape(1, -1).astype(np.float32), np.asarray([[valid]], dtype=np.float32)


__all__ = ["ImageTransform", "preprocess_image", "preprocess_state"]
