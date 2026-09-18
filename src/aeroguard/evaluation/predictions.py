"""Validation at the boundary between raw detector output and strict metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def normalize_model_prediction(box: Sequence[float], label: int, score: float) -> dict[str, Any] | None:
    """Normalize one detector output, dropping only finite zero-area boxes."""
    values = [float(value) for value in box]
    numeric_score = float(score)
    numeric_label = int(label)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("model prediction box must contain four finite values")
    if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
        raise ValueError("model prediction score must be finite and in [0, 1]")
    if numeric_label < 1:
        raise ValueError("model prediction label must be positive")
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return {"box_xyxy": values, "label": numeric_label, "score": numeric_score}


__all__ = ["normalize_model_prediction"]
