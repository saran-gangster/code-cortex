from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aeroguard.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "contracts" / "evaluation_report.schema.json"


def report(model_id: str = "fcos-film-dev") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "artifact_kind": "development_evaluation",
        "benchmark_claim": False,
        "partition": "development",
        "protocol_sha256": "a" * 64,
        "final_test_unsealed": False,
        "model_id": model_id,
        "checkpoint_id": "checkpoint-1",
        "config": {"steps": 10},
        "metrics": {"pooled": {"ap50": 0.1}},
        "limitations": ["Engineering report only"],
    }


def write_report(directory: Path, value: dict[str, object], name: str = "report.json") -> None:
    (directory / name).write_text(json.dumps(value), encoding="utf-8")


def test_valid_reports_list_and_get_by_stable_model_id(tmp_path: Path) -> None:
    write_report(tmp_path, report())
    client = TestClient(create_app(report_dir=tmp_path, report_schema_path=SCHEMA))

    listed = client.get("/reports")
    assert listed.status_code == 200
    assert listed.json()["reports"][0]["report_id"] == "fcos-film-dev"
    fetched = client.get("/reports/fcos-film-dev")
    assert fetched.status_code == 200
    assert fetched.json()["model_id"] == "fcos-film-dev"
    assert fetched.json()["report_id"] == "fcos-film-dev"


def test_existing_review_report_fallback_is_preserved(tmp_path: Path) -> None:
    client = TestClient(create_app(report_dir=tmp_path, report_schema_path=SCHEMA))
    assert client.get("/reports/fixture-run").status_code == 200
    assert client.get("/reports/fixture-run").json()["review_count"] == 0


def test_duplicate_model_ids_fail_at_startup(tmp_path: Path) -> None:
    write_report(tmp_path, report(), "one.json")
    write_report(tmp_path, report(), "two.json")
    with pytest.raises(ValueError, match="duplicate evaluation report id"):
        create_app(report_dir=tmp_path, report_schema_path=SCHEMA)


def test_invalid_report_fails_at_startup(tmp_path: Path) -> None:
    invalid = report()
    invalid["benchmark_claim"] = True
    write_report(tmp_path, invalid)
    with pytest.raises(ValueError, match="invalid evaluation report"):
        create_app(report_dir=tmp_path, report_schema_path=SCHEMA)
