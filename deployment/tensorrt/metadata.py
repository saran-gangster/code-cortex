"""Artifact hashing and atomic provenance sidecars."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"artifact is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def require_nonempty_artifact(path: str | Path, kind: str) -> Path:
    artifact = Path(path).resolve()
    if not artifact.is_file() or artifact.stat().st_size <= 0:
        raise RuntimeError(f"{kind} was not produced as a non-empty regular file: {artifact}")
    return artifact


__all__ = ["canonical_sha256", "require_nonempty_artifact", "sha256_file", "write_json_atomic"]
