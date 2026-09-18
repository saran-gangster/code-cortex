from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aeroguard.api.app import ReviewStore, create_app
from aeroguard.inference.contracts import (
    Detection,
    InferenceRecord,
    InferenceRequest,
    Latency,
    PredictionSource,
)
from aeroguard.inference.runtime import InferenceRuntime, RuntimeErrorInfo


def computed(request: InferenceRequest) -> InferenceRecord:
    return InferenceRecord(
        frame_id=request.frame_id, model_id="test-model", checkpoint_sha256="1" * 64,
        protocol_sha256="2" * 64, prediction_source=PredictionSource.computed,
        source_time_ms=request.source_time_ms, original_size=request.original_size,
        input_mode=request.input_mode, metadata_alignment=request.metadata_alignment,
        detections=[{"class_name": "Car", "box_xyxy": (1.0, 2.0, 20.0, 30.0),
                     "raw_score": 0.8, "calibrated_score": None}],
        quality_flags=[], latency=Latency(preprocess_ms=1.0, model_ms=2.0, postprocess_ms=1.0, end_to_end_ms=4.0),
    )


def request_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "frame_id": "frame-1", "run_id": "run-1", "original_size": {"width": 100, "height": 80},
        "input_mode": "paired_state", "metadata_alignment": "paired_annotation", "state": [1.0, 2.0],
    }
    payload.update(changes)
    return payload


def test_fixture_is_explicit_and_replayable(tmp_path: Path) -> None:
    client = TestClient(create_app(review_path=tmp_path / "reviews.jsonl"))
    assert client.get("/health").json()["runtime_mode"] == "fixture_replay"
    assert client.get("/runs").json()["runs"][0]["prediction_source"] == "fixture"
    response = client.get("/runs/fixture-run/frames/fixture-frame-001")
    assert response.status_code == 200
    assert response.json()["prediction_source"] == "fixture"


def test_runtime_success_and_review_jsonl(tmp_path: Path) -> None:
    client = TestClient(create_app(InferenceRuntime(computed), review_path=tmp_path / "reviews.jsonl"))
    response = client.post("/infer", json=request_payload())
    assert response.status_code == 200
    assert response.json()["prediction_source"] == "computed"
    review = client.post("/reviews", json={"run_id": "run-1", "frame_id": "frame-1", "decision": "accept", "comment": "ok"})
    assert review.status_code == 201
    report = client.get("/reports/run-1")
    assert report.json()["review_count"] == 1
    saved = (tmp_path / "reviews.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(saved[0])["frame_id"] == "frame-1"


def test_bad_box_and_nonfinite_values_are_rejected(tmp_path: Path) -> None:
    client = TestClient(create_app(InferenceRuntime(computed), review_path=tmp_path / "reviews.jsonl"))
    assert client.post("/infer", json=request_payload(state=["NaN"])).status_code == 422
    assert client.post("/infer", json=request_payload(state=["1.0"])).status_code == 422
    assert client.post("/infer", json=request_payload(original_size={"width": "100", "height": 80})).status_code == 422
    assert client.post("/infer", json=request_payload(original_size={"width": 0, "height": 80})).status_code == 422

    def bad_box(request: InferenceRequest) -> InferenceRecord:
        record = computed(request)
        invalid = Detection(class_name="Car", box_xyxy=(0.0, 0.0, 101.0, 10.0), raw_score=0.8)
        return record.model_copy(update={"detections": [invalid]})

    bad_client = TestClient(create_app(InferenceRuntime(bad_box), review_path=tmp_path / "bad.jsonl"))
    assert bad_client.post("/infer", json=request_payload()).status_code == 503


def test_runtime_failure_is_honest_and_no_dummy_record(tmp_path: Path) -> None:
    def failure(_: InferenceRequest) -> InferenceRecord:
        raise RuntimeErrorInfo("checkpoint unavailable")

    client = TestClient(create_app(InferenceRuntime(failure), review_path=tmp_path / "reviews.jsonl"))
    response = client.post("/infer", json=request_payload())
    assert response.status_code == 503
    assert client.get("/runs/run-1/frames/frame-1").status_code == 404


def test_review_path_cannot_escape_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ReviewStore(tmp_path / "escape.jsonl", tmp_path / "nested")
