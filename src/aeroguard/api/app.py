"""A small local FastAPI service for computed inference and offline replay."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from aeroguard.inference.contracts import (
    CapabilityResponse,
    EvaluationReportResponse,
    HealthResponse,
    InferenceRecord,
    InferenceRequest,
    ModelDetail,
    ModelListResponse,
    ModelSummary,
    PredictionSource,
    ReadinessResponse,
    ReportsListResponse,
    Review,
    ReviewListResponse,
    ReviewRequest,
    RunListResponse,
    RunReportResponse,
    RunSummary,
)
from aeroguard.inference.runtime import InferenceRuntime, RuntimeErrorInfo
from aeroguard.inference.store import EvaluationReportStore, FixtureReplayStore

API_VERSION = "0.2.0"
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:4173",
    "http://localhost:4173",
)


class ReviewStoreError(RuntimeError):
    """A safe, actionable failure while reading or writing review data."""


class ReviewStore:
    """Append-only local JSONL storage with deterministic, idempotent reviews."""

    def __init__(self, path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> None:
        raw_path = Path(path).expanduser()
        if raw_path.is_symlink():
            raise ValueError("review path must not be a symbolic link")
        configured = raw_path.resolve()
        base = Path(root or raw_path.parent).expanduser().resolve()
        try:
            configured.relative_to(base)
        except ValueError as exc:
            raise ValueError("review path must remain inside the configured review directory") from exc
        configured.parent.mkdir(parents=True, exist_ok=True)
        self.path = configured
        self._lock = threading.RLock()

    def _read_unlocked(self) -> list[Review]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ReviewStoreError(f"cannot read review store: {self.path}") from exc
        result: list[Review] = []
        seen: set[str] = set()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                review = Review.model_validate_json(line)
            except Exception as exc:
                raise ReviewStoreError(f"invalid review record at line {line_number}") from exc
            if review.review_id in seen:
                raise ReviewStoreError(f"duplicate review id in store: {review.review_id}")
            seen.add(review.review_id)
            result.append(review)
        return result

    def put(self, review: Review) -> tuple[Review, bool]:
        """Atomically insert a review, returning the stored value and whether it was new."""

        line = json.dumps(review.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)
        with self._lock:
            existing = {item.review_id: item for item in self._read_unlocked()}
            previous = existing.get(review.review_id)
            if previous is not None:
                same_request = (
                    previous.run_id == review.run_id
                    and previous.frame_id == review.frame_id
                    and previous.decision == review.decision
                    and previous.comment == review.comment
                )
                if not same_request:
                    raise ReviewStoreError(f"review id already exists with different content: {review.review_id}")
                return previous, False
            try:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise ReviewStoreError(f"cannot write review store: {self.path}") from exc
        return review, True

    def append(self, review: Review) -> Review:
        """Compatibility wrapper for callers that only need the stored review."""

        return self.put(review)[0]

    def list(self, run_id: str | None = None, frame_id: str | None = None) -> list[Review]:
        with self._lock:
            reviews = self._read_unlocked()
        return [
            review for review in reviews
            if (run_id is None or review.run_id == run_id)
            and (frame_id is None or review.frame_id == frame_id)
        ]

    def get(self, review_id: str) -> Review | None:
        with self._lock:
            return next((item for item in self._read_unlocked() if item.review_id == review_id), None)


def _configured_origins(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        value = os.getenv("AEROGUARD_CORS_ORIGINS")
    if value is None:
        return list(DEFAULT_CORS_ORIGINS)
    if isinstance(value, str):
        origins = [item.strip() for item in value.split(",")]
    else:
        origins = [item.strip() for item in value]
    return [origin for origin in origins if origin]


def _review_id(request: ReviewRequest) -> str:
    canonical = json.dumps(
        request.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _api_error(code: str, message: str, details: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        payload["details"] = details
    return {"error": payload}


def create_app(
    runtime: InferenceRuntime | None = None,
    replay_store: FixtureReplayStore | None = None,
    review_path: str | os.PathLike[str] | None = None,
    report_dir: str | os.PathLike[str] | None = None,
    report_schema_path: str | os.PathLike[str] | None = None,
    max_concurrency: int = 1,
    cors_origins: Sequence[str] | str | None = None,
) -> FastAPI:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    app = FastAPI(title="AeroGuard API", version=API_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_configured_origins(cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Authorization", "Content-Type"],
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

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            detail = exc.detail
        else:
            detail = {"code": "http_error", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={"error": detail})

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_api_error("validation_error", "request validation failed", jsonable_encoder(exc.errors())),
        )

    @app.exception_handler(ReviewStoreError)
    async def review_store_error(_: Request, exc: ReviewStoreError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_api_error("review_store_error", str(exc)),
        )

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_api_error("internal_error", "an unexpected server error occurred"),
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> dict[str, object]:
        ready = app.state.runtime.ready
        return {
            "status": "ok",
            "version": API_VERSION,
            "model_ready": ready,
            "runtime_mode": "computed" if ready else "fixture_replay",
        }

    @app.get("/readiness", response_model=ReadinessResponse)
    @app.get("/ready", response_model=ReadinessResponse, include_in_schema=False)
    @app.get("/health/readiness", response_model=ReadinessResponse, include_in_schema=False)
    async def readiness() -> ReadinessResponse:
        computed_ready = bool(app.state.runtime.ready)
        return ReadinessResponse(
            ready=True,
            status="ready" if computed_ready else "degraded",
            computed_inference_ready=computed_ready,
            replay_ready=True,
        )

    @app.get("/capabilities", response_model=CapabilityResponse)
    @app.get("/metadata", response_model=CapabilityResponse, include_in_schema=False)
    async def capabilities() -> CapabilityResponse:
        return CapabilityResponse(
            api_version=API_VERSION,
            computed_inference=bool(app.state.runtime.ready),
            fixture_replay=True,
            review_persistence=True,
            supported_prediction_sources=list(PredictionSource),
            endpoints={
                "health": "/health",
                "readiness": "/readiness",
                "inference": "/infer",
                "runs": "/runs",
                "models": "/models",
                "reports": "/reports",
                "reviews": "/reviews",
            },
        )

    @app.get("/runs", response_model=RunListResponse)
    async def runs() -> dict[str, object]:
        return {"runs": app.state.replay.list_runs()}

    @app.get("/runs/{run_id}", response_model=RunSummary)
    async def run_detail(run_id: str) -> dict[str, object]:
        summary = next((item for item in app.state.replay.list_runs() if item["run_id"] == run_id), None)
        if summary is None:
            raise HTTPException(status_code=404, detail={"code": "run_not_found", "message": "run not found"})
        return summary

    @app.get("/runs/{run_id}/frames", response_model=list[InferenceRecord])
    async def frames(run_id: str) -> list[InferenceRecord]:
        if not any(item["run_id"] == run_id for item in app.state.replay.list_runs()):
            raise HTTPException(status_code=404, detail={"code": "run_not_found", "message": "run not found"})
        return app.state.replay.list_frames(run_id)

    @app.post("/infer", response_model=InferenceRecord)
    async def infer(request: InferenceRequest) -> InferenceRecord:
        async with app.state.inference_gate:
            try:
                record = await asyncio.to_thread(app.state.runtime.infer, request)
            except RuntimeErrorInfo as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={"code": "runtime_unavailable", "message": str(exc)},
                ) from exc
            app.state.replay.put(request.run_id, record)
            return record

    @app.get("/runs/{run_id}/frames/{frame_id}", response_model=InferenceRecord)
    async def frame(run_id: str, frame_id: str) -> InferenceRecord:
        record = app.state.replay.get(run_id, frame_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "frame_not_found", "message": "frame not found"})
        return record

    @app.post("/reviews", response_model=Review, status_code=201)
    async def review(request: ReviewRequest, response: Response) -> Review:
        record = app.state.replay.get(request.run_id, request.frame_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "frame_not_found", "message": "cannot review an unknown frame"},
            )
        result = Review(
            review_id=_review_id(request),
            created_at=datetime.now(timezone.utc).isoformat(),
            **request.model_dump(),
        )
        stored, created = await asyncio.to_thread(app.state.reviews.put, result)
        response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
        return stored

    @app.get("/reviews", response_model=ReviewListResponse)
    async def reviews(run_id: str | None = None, frame_id: str | None = None) -> dict[str, object]:
        return {"reviews": await asyncio.to_thread(app.state.reviews.list, run_id, frame_id)}

    @app.get("/reviews/{review_id}", response_model=Review)
    async def review_detail(review_id: str) -> Review:
        if not review_id or "/" in review_id or "\\" in review_id:
            raise HTTPException(status_code=404, detail={"code": "review_not_found", "message": "review not found"})
        result = await asyncio.to_thread(app.state.reviews.get, review_id)
        if result is None:
            raise HTTPException(status_code=404, detail={"code": "review_not_found", "message": "review not found"})
        return result

    @app.get("/models", response_model=ModelListResponse)
    async def models() -> dict[str, object]:
        reports = {item["model_id"]: item for item in app.state.evaluation_reports.list()}
        catalog: dict[str, ModelSummary] = {}
        for model_id, report in reports.items():
            catalog[model_id] = ModelSummary(
                model_id=model_id,
                report_id=report["report_id"],
                checkpoint_id=report["checkpoint_id"],
                partition=report["partition"],
                available_for_inference=bool(app.state.runtime.ready),
            )
        for item in app.state.replay.list_models():
            model_id = str(item["model_id"])
            catalog.setdefault(
                model_id,
                ModelSummary(
                    model_id=model_id,
                    prediction_source=item["prediction_source"],
                    available_for_inference=bool(app.state.runtime.ready),
                ),
            )
        return {"models": [catalog[key] for key in sorted(catalog)]}

    @app.get("/models/{model_id}", response_model=ModelDetail)
    async def model_detail(model_id: str) -> ModelDetail:
        report = app.state.evaluation_reports.get(model_id)
        if report is not None:
            return ModelDetail(
                model_id=model_id,
                report_id=report["report_id"],
                checkpoint_id=report["checkpoint_id"],
                partition=report["partition"],
                available_for_inference=bool(app.state.runtime.ready),
                report=report,
            )
        replay_model = next((item for item in app.state.replay.list_models() if item["model_id"] == model_id), None)
        if replay_model is None:
            raise HTTPException(status_code=404, detail={"code": "model_not_found", "message": "model not found"})
        return ModelDetail(
            model_id=model_id,
            prediction_source=replay_model["prediction_source"],
            available_for_inference=bool(app.state.runtime.ready),
        )

    @app.get("/reports", response_model=ReportsListResponse)
    async def reports() -> dict[str, object]:
        return {"reports": app.state.evaluation_reports.list()}

    @app.get("/reports/{report_id}", response_model=EvaluationReportResponse | RunReportResponse)
    async def report(report_id: str) -> dict[str, object]:
        if not report_id or "/" in report_id or "\\" in report_id:
            raise HTTPException(status_code=404, detail={"code": "report_not_found", "message": "report not found"})
        evaluation = app.state.evaluation_reports.get(report_id)
        if evaluation is not None:
            return evaluation
        runs = {item["run_id"]: item for item in app.state.replay.list_runs()}
        if report_id not in runs:
            raise HTTPException(status_code=404, detail={"code": "report_not_found", "message": "report not found"})
        reviews = await asyncio.to_thread(app.state.reviews.list, report_id)
        return {
            "run_id": report_id,
            "provenance": runs[report_id]["prediction_source"],
            "review_count": len(reviews),
            "reviews": [item.model_dump(mode="json") for item in reviews],
        }

    return app


def run() -> None:
    import uvicorn

    uvicorn.run("aeroguard.api.app:app", host="127.0.0.1", port=int(os.getenv("AEROGUARD_PORT", "8000")))


app = create_app()
