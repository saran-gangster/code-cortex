import json

import pytest

from aeroguard.evaluation.report import build_evaluation_report, write_evaluation_report


def _report(**overrides):
    values = {
        "artifact_kind": "synthetic_development_evaluation",
        "partition": "development",
        "protocol_sha256": "protocol-123",
        "final_test_unsealed": False,
        "model_id": "fcos-film-dev",
        "checkpoint_id": "checkpoint-123",
        "config": {"seed": 17, "learning_rate": 0.001},
        "metrics": {"loss": 1.25, "frames": 8},
        "limitations": ["Synthetic/development evidence only."],
    }
    values.update(overrides)
    return build_evaluation_report(**values)


def test_report_is_deterministic_and_carries_required_evidence_fields(tmp_path):
    first = _report(metrics={"recall": 0.5, "precision": 0.75})
    second = _report(metrics={"recall": 0.5, "precision": 0.75})
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    write_evaluation_report(first_path, first)
    write_evaluation_report(second_path, second)

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_text(encoding="utf-8").endswith("\n")
    assert list(json.loads(first_path.read_text(encoding="utf-8"))) == sorted(first)
    assert first["benchmark_claim"] is False
    assert first["protocol_sha256"] == "protocol-123"
    assert first["final_test_unsealed"] is False


def test_sealed_final_test_is_refused():
    with pytest.raises(ValueError, match="final test.*sealed"):
        _report(partition="final_test")


def test_benchmark_claim_is_refused():
    with pytest.raises(ValueError, match="benchmark_claim"):
        _report(benchmark_claim=True)


def test_non_finite_metric_is_refused():
    with pytest.raises(ValueError, match="finite JSON"):
        _report(metrics={"loss": float("nan")})
