"""A small local FastAPI service for computed inference and offline replay."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from aeroguard.inference.contracts import (
    InferenceRecord,
    InferenceRequest,
    Review,
    ReviewRequest,
)
from aeroguard.inference.runtime import InferenceRuntime, RuntimeErrorInfo
from aeroguard.inference.store import EvaluationReportStore, FixtureReplayStore

API_VERSION = "0.1.0"


class ReviewStore:
    def __init__(self, path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> None:
        configured = Path(path).expanduser().resolve()
        base = Path(root or configured.parent).expanduser().resolve()
        try:
            configured.relative_to(base)
        except ValueError as exc:
            raise ValueError("review path must remain inside the configured review directory") from exc
        configured.parent.mkdir(parents=True, exist_ok=True)
        self.path = configured
        self._lock = threading.Lock()

    def append(self, review: Review) -> Review:
        line = json.dumps(review.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return review

    def list(self, run_id: str | None = None) -> list[Review]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        result: list[Review] = []
        for line in lines:
            if not line.strip():
                continue
            review = Review.model_validate_json(line)
            if run_id is None or review.run_id == run_id:
                result.append(review)
        return result


def create_app(
    runtime: InferenceRuntime | None = None,
    replay_store: FixtureReplayStore | None = None,
    review_path: str | os.PathLike[str] | None = None,
    report_dir: str | os.PathLike[str] | None = None,
    report_schema_path: str | os.PathLike[str] | None = None,
    max_concurrency: int = 1,
) -> FastAPI:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    app = FastAPI(title="AeroGuard API", version=API_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:4173", "http://localhost:4173", "http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.state.runtime = runtime or InferenceRuntime()
    app.state.replay = replay_store or FixtureReplayStore()
    default_report_dir = Path(os.getenv("AEROGUARD_REPORT_DIR", "reports/evaluations"))
    default_schema_path = Path(__file__).resolve().parents[3] / "contracts" / "evaluation_report.schema.json"
    app.state.evaluation_reports = EvaluationReportStore(
        report_dir or default_report_dir,
        report_schema_path or default_schema_path,
    )
    if review_path is None:
        review_root = Path(os.getenv("AEROGUARD_REVIEW_DIR", ".aeroguard")).resolve()
        configured_review_path = review_root / "reviews.jsonl"
    else:
        configured_review_path = Path(review_path).expanduser()
        review_root = configured_review_path.parent.resolve()
    app.state.reviews = ReviewStore(configured_review_path, review_root)
    app.state.inference_gate = asyncio.Semaphore(max_concurrency)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "version": API_VERSION, "model_ready": app.state.runtime.ready,
                "runtime_mode": "computed" if app.state.runtime.ready else "fixture_replay"}

    @app.get("/runs")
    async def runs() -> dict[str, object]:
        return {"runs": app.state.replay.list_runs()}

    @app.post("/infer", response_model=InferenceRecord)
    async def infer(request: InferenceRequest) -> InferenceRecord:
        async with app.state.inference_gate:
            try:
                record = await asyncio.to_thread(app.state.runtime.infer, request)
            except RuntimeErrorInfo as exc:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
            app.state.replay.put(request.run_id, record)
            return record

    @app.get("/runs/{run_id}/frames/{frame_id}", response_model=InferenceRecord)
    async def frame(run_id: str, frame_id: str) -> InferenceRecord:
        record = app.state.replay.get(run_id, frame_id)
        if record is None:
            raise HTTPException(status_code=404, detail="frame not found")
        return record

    @app.post("/reviews", response_model=Review, status_code=201)
    async def review(request: ReviewRequest) -> Review:
        record = app.state.replay.get(request.run_id, request.frame_id)
        if record is None:
            raise HTTPException(status_code=404, detail="cannot review an unknown frame")
        result = Review(review_id=uuid.uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(), **request.model_dump())
        return await asyncio.to_thread(app.state.reviews.append, result)

    @app.get("/reports/{run_id}")
    async def report(run_id: str) -> dict[str, object]:
        evaluation = app.state.evaluation_reports.get(run_id)
        if evaluation is not None:
            return evaluation
        runs = {item["run_id"]: item for item in app.state.replay.list_runs()}
        if run_id not in runs:
            raise HTTPException(status_code=404, detail="run not found")
        reviews = await asyncio.to_thread(app.state.reviews.list, run_id)
        return {"run_id": run_id, "provenance": runs[run_id]["prediction_source"],
                "review_count": len(reviews), "reviews": [item.model_dump(mode="json") for item in reviews]}

    @app.get("/reports")
    async def reports() -> dict[str, object]:
        return {"reports": app.state.evaluation_reports.list()}

    return app


app = create_app()
