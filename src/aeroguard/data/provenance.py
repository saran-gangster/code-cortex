"""Fail-closed validation for the frozen split and normalizer provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrozenPartitions:
    train: frozenset[str]
    development: frozenset[str]
    final_test: frozenset[str]


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_protocol_bundle(
    protocol: Mapping[str, Any],
    normalizer: Mapping[str, Any],
) -> FrozenPartitions:
    """Validate partition disjointness and the train-only normalizer hash."""

    selection = protocol.get("selection")
    if not isinstance(selection, Mapping):
        raise TypeError("protocol selection must be a mapping")
    values: dict[str, frozenset[str]] = {}
    for name in ("train", "development", "final_test"):
        raw = selection.get(name)
        if not isinstance(raw, list) or not raw or any(not isinstance(item, str) for item in raw):
            raise TypeError(f"protocol partition {name} must be a non-empty string list")
        if len(raw) != len(set(raw)):
            raise ValueError(f"protocol partition {name} contains duplicate roots")
        values[name] = frozenset(raw)
    if (
        values["train"] & values["development"]
        or values["train"] & values["final_test"]
        or values["development"] & values["final_test"]
    ):
        raise ValueError("protocol recording-root partitions overlap")
    if protocol.get("final_test_unsealed") is not False:
        raise ValueError("this development workflow requires final-test to remain sealed")
    if normalizer.get("fit_partition") != "train":
        raise ValueError("state normalizer must be fitted on train only")
    fit_roots = normalizer.get("fit_recording_roots")
    if not isinstance(fit_roots, list) or frozenset(fit_roots) != values["train"]:
        raise ValueError("normalizer fit roots do not match the frozen training roots")
    if normalizer.get("protocol_sha256") != protocol.get("protocol_sha256"):
        raise ValueError("normalizer/protocol hash mismatch")
    expected_hash = normalizer.get("normalizer_sha256")
    unhashed = dict(normalizer)
    unhashed.pop("normalizer_sha256", None)
    if not isinstance(expected_hash, str) or canonical_sha256(unhashed) != expected_hash:
        raise ValueError("state normalizer content hash mismatch")
    return FrozenPartitions(
        train=values["train"],
        development=values["development"],
        final_test=values["final_test"],
    )


__all__ = ["FrozenPartitions", "canonical_sha256", "validate_protocol_bundle"]
