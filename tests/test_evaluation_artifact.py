import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_evaluation_artifact_is_schema_valid_and_sealed():
    schema = json.loads((ROOT / "contracts" / "evaluation_report.schema.json").read_text())
    report = json.loads((ROOT / "reports" / "evaluation_pipeline_fixture.json").read_text())
    protocol = json.loads((ROOT / "manifests" / "protocol.json").read_text())

    jsonschema.validate(report, schema)
    assert report["partition"] == "synthetic_fixture"
    assert report["benchmark_claim"] is False
    assert report["final_test_unsealed"] is False
    assert report["protocol_sha256"] == protocol["protocol_sha256"]
    assert report["metrics"]["pooled"]["support"] == 2
    assert "Synthetic boxes" in report["limitations"][0]


def test_committed_development_reports_are_valid_unique_and_sealed():
    schema = json.loads((ROOT / "contracts" / "evaluation_report.schema.json").read_text())
    protocol = json.loads((ROOT / "manifests" / "protocol.json").read_text())
    report_paths = sorted((ROOT / "reports" / "evaluations").glob("*.json"))

    assert report_paths
    model_ids = []
    for report_path in report_paths:
        report = json.loads(report_path.read_text())
        jsonschema.validate(report, schema)
        assert report["partition"] == "development"
        assert report["benchmark_claim"] is False
        assert report["final_test_unsealed"] is False
        assert report["protocol_sha256"] == protocol["protocol_sha256"]
        assert report["config"]["training_schedule_sha256"]
        assert report["config"]["shared_warmstart_sha256"]
        assert report["metrics"]["pooled"]["support"] > 0
        model_ids.append(report["model_id"])

    assert len(model_ids) == len(set(model_ids))
