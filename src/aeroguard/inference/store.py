"""Deterministic, visibly labelled offline replay records."""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema

from .contracts import (
    InferenceRecord,
    InputMode,
    Latency,
    MetadataAlignment,
    OriginalSize,
    PredictionSource,
)

_ZERO = "0" * 64


class FixtureReplayStore:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], InferenceRecord] = {}
        self._runs: dict[str, dict[str, object]] = {}
        self._lock = threading.RLock()
        self._seed()

    def _seed(self) -> None:
        record = InferenceRecord(
            frame_id="fixture-frame-001", model_id="aeroguard-fixture-v1",
            checkpoint_sha256=_ZERO, protocol_sha256=_ZERO,
            prediction_source=PredictionSource.fixture, source_time_ms=0,
            original_size=OriginalSize(width=1280, height=720), input_mode=InputMode.separate_rgb_fallback,
            metadata_alignment=MetadataAlignment.unavailable, detections=[], quality_flags=["STATE_MISSING"],
            latency=Latency(preprocess_ms=None, model_ms=None, postprocess_ms=None, end_to_end_ms=None),
        )
        self._records[("fixture-run", record.frame_id)] = record
        self._runs["fixture-run"] = {"run_id": "fixture-run", "prediction_source": "fixture", "frame_count": 1}
        artifact = Path(
            os.getenv("AEROGUARD_SMOKE_RECORD", "reports/fcos_overfit_gate_inference.json")
        )
        if artifact.is_file():
            computed = InferenceRecord.model_validate_json(artifact.read_text(encoding="utf-8"))
            self.put("fcos-overfit-engineering-gate", computed)

    def list_runs(self) -> list[dict[str, object]]:
        with self._lock:
            ordered = sorted(self._runs, key=lambda key: (key != "fixture-run", key))
            return deepcopy([self._runs[key] for key in ordered])

    def get(self, run_id: str, frame_id: str) -> InferenceRecord | None:
        with self._lock:
            record = self._records.get((run_id, frame_id))
            return deepcopy(record) if record else None

    def list_frames(self, run_id: str) -> list[InferenceRecord]:
        with self._lock:
            records = [record for (stored_run, _), record in self._records.items() if stored_run == run_id]
            return deepcopy(sorted(records, key=lambda record: record.frame_id))

    def list_models(self) -> list[dict[str, object]]:
        with self._lock:
            models: dict[str, dict[str, object]] = {}
            for record in self._records.values():
                models.setdefault(record.model_id, {
                    "model_id": record.model_id,
                    "prediction_source": record.prediction_source,
                })
            return deepcopy([models[key] for key in sorted(models)])

    def put(self, run_id: str, record: InferenceRecord) -> None:
        with self._lock:
            self._records[(run_id, record.frame_id)] = deepcopy(record)
            frame_count = sum(1 for stored_run, _ in self._records if stored_run == run_id)
            self._runs[run_id] = {
                "run_id": run_id,
                "prediction_source": record.prediction_source.value,
                "frame_count": frame_count,
            }


class EvaluationReportStore:
    """Load and validate local evaluation reports once at application startup.

    Report ids are the report's ``model_id``.  This is stable across file
    renames and deterministic for a given model artifact.  The store is
    intentionally immutable after construction so a malformed file cannot be
    silently introduced between discovery and retrieval.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        schema_path: str | os.PathLike[str],
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.schema_path = Path(schema_path).expanduser().resolve()
        self._reports = self._discover()

    def _discover(self) -> dict[str, dict[str, Any]]:
        if not self.directory.exists():
            return {}
        if not self.directory.is_dir():
            raise ValueError(f"evaluation report directory is not a directory: {self.directory}")
        try:
            schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
            jsonschema.Draft202012Validator.check_schema(schema)
        except (OSError, json.JSONDecodeError, jsonschema.SchemaError) as exc:
            raise ValueError(f"invalid evaluation report schema: {self.schema_path}") from exc
        validator = jsonschema.Draft202012Validator(schema)
        reports: dict[str, dict[str, Any]] = {}
        for path in sorted(self.directory.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                validator.validate(report)
            except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
                raise ValueError(f"invalid evaluation report: {path}") from exc
            if not isinstance(report, dict):
                raise TypeError(f"evaluation report must be an object: {path}")
            if report["partition"] == "final_test":
                if report["final_test_unsealed"] is not True:
                    raise ValueError(f"final-test seal violation: {path}")
            elif report["final_test_unsealed"] is not False:
                raise ValueError(f"non-final report cannot unseal final test: {path}")
            report_id = report["model_id"]
            if not isinstance(report_id, str) or not report_id.strip() or report_id != report_id.strip() or any(
                character in report_id for character in ("/", "\\", "..")
            ):
                raise ValueError(f"unsafe evaluation report id {report_id!r}: {path}")
            if report_id in reports:
                raise ValueError(f"duplicate evaluation report id {report_id!r}: {path}")
            materialized = deepcopy(report)
            materialized["report_id"] = report_id
            reports[report_id] = materialized
        return reports

    def list(self) -> list[dict[str, Any]]:
        return deepcopy([self._reports[key] for key in sorted(self._reports)])

    def get(self, report_id: str) -> dict[str, Any] | None:
        report = self._reports.get(report_id)
        return deepcopy(report) if report is not None else None
