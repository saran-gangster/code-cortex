"""Deterministic, non-benchmark evaluation evidence reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = "1.0"


def _validate_report_guards(report: Mapping[str, Any]) -> None:
    """Reject report states that would overstate engineering evidence."""

    if report.get("benchmark_claim", False):
        raise ValueError("benchmark_claim=True is not allowed for engineering reports")
    if report.get("partition") == "final_test" and report.get("final_test_unsealed") is not True:
        raise ValueError("final_test evidence cannot be reported while final test is sealed")


def _json_text(report: Mapping[str, Any]) -> str:
    _validate_report_guards(report)
    try:
        return json.dumps(
            report,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation report must contain only finite JSON values") from exc


def build_evaluation_report(
    *,
    artifact_kind: str,
    partition: str,
    protocol_sha256: str | None = None,
    final_test_unsealed: bool,
    model_id: str,
    checkpoint_id: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    limitations: Sequence[str] = (),
    schema_version: str = _SCHEMA_VERSION,
    benchmark_claim: bool = False,
    protocol_hash: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe engineering/development evaluation report.

    ``final_test`` is intentionally available only after an explicit unseal, and
    this path can never produce a benchmark claim.  ``protocol_hash`` is accepted
    as a convenience alias; reports use the repository's ``protocol_sha256`` key.
    """

    if protocol_sha256 is None:
        protocol_sha256 = protocol_hash
    if protocol_sha256 is None:
        raise ValueError("protocol_sha256 is required")

    report: dict[str, Any] = {
        "schema_version": schema_version,
        "artifact_kind": artifact_kind,
        "benchmark_claim": benchmark_claim,
        "partition": partition,
        "protocol_sha256": protocol_sha256,
        "final_test_unsealed": final_test_unsealed,
        "model_id": model_id,
        "checkpoint_id": checkpoint_id,
        "config": dict(config),
        "metrics": dict(metrics),
        "limitations": list(limitations),
    }
    _json_text(report)
    return report


def write_evaluation_report(
    path: str | PathLike[str],
    report: Mapping[str, Any] | None = None,
    **build_kwargs: Any,
) -> dict[str, Any]:
    """Write an evaluation report as stable sorted JSON with a trailing newline."""

    if report is not None and build_kwargs:
        raise TypeError("provide either report or builder arguments, not both")
    materialized = (
        dict(report) if report is not None else build_evaluation_report(**build_kwargs)
    )
    text = _json_text(materialized)
    destination = Path(path)
    destination.write_text(text, encoding="utf-8", newline="\n")
    return materialized


# Short aliases keep the module convenient for scripts without changing the
# explicit names used by the public API.
build_report = build_evaluation_report
write_report = write_evaluation_report


__all__ = [
    "build_evaluation_report",
    "build_report",
    "write_evaluation_report",
    "write_report",
]
