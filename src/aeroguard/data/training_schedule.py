"""Deterministic helpers for matched training schedules."""

from __future__ import annotations

import hashlib


def deterministic_state_mask(
    base_mask: float,
    dropout_probability: float,
    *,
    seed: int,
    schedule_index: int,
) -> float:
    """Return a stable per-step state mask without worker RNG dependence."""

    if base_mask == 0.0 or dropout_probability == 0.0:
        return base_mask
    payload = f"{seed}:{schedule_index}".encode()
    unit_interval = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    return 0.0 if unit_interval < dropout_probability else base_mask


__all__ = ["deterministic_state_mask"]
