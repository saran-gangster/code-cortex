from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aeroguard.api.app import create_app
from aeroguard.inference.contracts import (
    InferenceRecord,
    InferenceRequest,
    Latency,
    PredictionSource,
    Review,
)
from aeroguard.inference.runtime import InferenceRuntime


def _payload(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "run_id": "run-surface",
        "frame_id": "frame-surface",
        "original_size": {"width": 100, "height": 80},
        "input_mode": "separate_rgb_fallback",
        "metadata_alignment": "unavailable",
    }
    value.update(changes)
    return value


def _runtime(request: InferenceRequest) -> InferenceRecord:
    return InferenceRecord(
        frame_id=request.frame_id,
        model_id="surface-model",
        checkpoint_sha256="1" * 64,
        protocol_sha256="2" * 64,
        prediction_source=PredictionSource.computed,
        source_time_ms=request.source_time_ms,
        original_size=request.original_size,
        input_mode=request.input_mode,
        metadata_alignment=request.metadata_alignment,
        detections=[],
        quality_flags=[],
        latency=Latency(preprocess_ms=1, model_ms=2, postprocess_ms=1, end_to_end_ms=4),
    )


def test_readiness_capabilities_and_models_are_frontend_friendly(tmp_path: Path) -> None:
    client = TestClient(create_app(InferenceRuntime(_runtime), review_path=tmp_path / "reviews.jsonl"))

    assert client.get("/health").status_code == 200
    assert client.get("/readiness").json()["computed_inference_ready"] is True
    capabilities = client.get("/capabilities").json()
    assert capabilities["computed_inference"] is True
    assert capabilities["fixture_replay"] is True
    assert client.get("/models/surface-model").status_code == 404


def test_image_and_state_validation_and_error_envelope(tmp_path: Path) -> None:
    client = TestClient(create_app(review_path=tmp_path / "reviews.jsonl"))
    invalid_state = client.post("/infer", json=_payload(input_mode="paired_state"))
    assert invalid_state.status_code == 422
    assert invalid_state.json()["error"]["code"] == "validation_error"
    invalid_image = client.post("/infer", json=_payload(image_base64="not-base64"))
    assert invalid_image.status_code == 422
    valid_image = base64.b64encode(b"fixture-image-bytes").decode("ascii")
    unavailable = client.post("/infer", json=_payload(image_base64=valid_image))
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "runtime_unavailable"


def test_routing_errors_use_the_same_error_envelope(tmp_path: Path) -> None:
    client = TestClient(create_app(review_path=tmp_path / "reviews.jsonl"))

    missing = client.get("/definitely-not-a-route")
    wrong_method = client.put("/health")

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "http_error"
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["code"] == "http_error"


def test_review_is_deterministic_idempotent_and_retrievable(tmp_path: Path) -> None:
    client = TestClient(create_app(InferenceRuntime(_runtime), review_path=tmp_path / "reviews.jsonl"))
    assert client.post("/infer", json=_payload()).status_code == 200
    body = {"run_id": "run-surface", "frame_id": "frame-surface", "decision": "accept", "comment": "ok"}
    first = client.post("/reviews", json=body)
    second = client.post("/reviews", json=body)
    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["review_id"] == second.json()["review_id"]
    review_id = first.json()["review_id"]
    assert client.get(f"/reviews/{review_id}").json() == first.json()
    assert len(client.get("/reviews", params={"run_id": "run-surface"}).json()["reviews"]) == 1
    assert len((tmp_path / "reviews.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_review_store_treats_server_timestamp_as_nonsemantic(tmp_path: Path) -> None:
    from aeroguard.api.app import ReviewStore

    store = ReviewStore(tmp_path / "reviews.jsonl")
    first_time = datetime.now(timezone.utc)
    values = {
        "review_id": "stable-id",
        "run_id": "run",
        "frame_id": "frame",
        "decision": "accept",
        "comment": "same request",
    }
    first = Review(created_at=first_time.isoformat(), **values)
    duplicate = Review(created_at=(first_time + timedelta(seconds=1)).isoformat(), **values)

    assert store.append(first) == first
    assert store.append(duplicate) == first
    assert len((tmp_path / "reviews.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_cors_is_configurable_without_default_wildcard(tmp_path: Path) -> None:
    client = TestClient(create_app(review_path=tmp_path / "reviews.jsonl", cors_origins=["http://ui.test:5173"]))
    allowed = client.options(
        "/health",
        headers={
            "Origin": "http://ui.test:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    denied = client.options(
        "/health",
        headers={
            "Origin": "http://evil.test",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == "http://ui.test:5173"
    assert "access-control-allow-origin" not in denied.headers


def test_review_store_rejects_path_traversal_and_report_duplicate_ids(tmp_path: Path) -> None:
    from aeroguard.api.app import ReviewStore

    with pytest.raises(ValueError):
        ReviewStore(tmp_path / ".." / "outside.jsonl", tmp_path / "reviews")

    report = {
        "schema_version": "1.0",
        "artifact_kind": "development_evaluation",
        "benchmark_claim": False,
        "partition": "development",
        "protocol_sha256": "a" * 64,
        "final_test_unsealed": False,
        "model_id": "same-model",
        "checkpoint_id": "checkpoint",
        "config": {},
        "metrics": {},
        "limitations": ["fixture"],
    }
    (tmp_path / "one.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "two.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate evaluation report id"):
        create_app(report_dir=tmp_path, report_schema_path=Path(__file__).parents[1] / "contracts/evaluation_report.schema.json")


def test_non_final_report_cannot_claim_final_test_is_unsealed(tmp_path: Path) -> None:
    report = {
        "schema_version": "1.0",
        "artifact_kind": "development_evaluation",
        "benchmark_claim": False,
        "partition": "development",
        "protocol_sha256": "a" * 64,
        "final_test_unsealed": True,
        "model_id": "unsafe-report",
        "checkpoint_id": "checkpoint",
        "config": {},
        "metrics": {},
        "limitations": ["fixture"],
    }
    (tmp_path / "unsafe.json").write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot unseal"):
        create_app(
            report_dir=tmp_path,
            report_schema_path=Path(__file__).parents[1] / "contracts/evaluation_report.schema.json",
        )
