# AeroGuard backend API

The FastAPI service gives the dashboard one stable interface for model evidence, fixture replay, live inference, and human review. It does not invent model output: when no computed runtime is configured, `/infer` returns a clear `503 runtime_unavailable` response while fixture replay and saved reports remain usable.

## Start locally

```powershell
python -m pip install -e ".[dev]"
python -m uvicorn aeroguard.api.app:app --host 127.0.0.1 --port 8000
```

Set the frontend environment variable to `VITE_API_URL=http://127.0.0.1:8000`. Local Vite origins on ports 5173 and 4173 are allowed by default. Other origins can be supplied through `AEROGUARD_CORS_ORIGINS` as a comma-separated list.

## Main routes

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Service status and current runtime mode |
| GET | `/readiness` | Computed-inference and fixture readiness |
| GET | `/capabilities` | Features and endpoint discovery for the UI |
| GET | `/runs` | Replay run summaries |
| GET | `/runs/{run_id}/frames` | Frames available in one replay run |
| GET | `/runs/{run_id}/frames/{frame_id}` | One replayed or computed inference record |
| GET | `/models` | Stable model catalogue built from reports and replay data |
| GET | `/models/{model_id}` | Model details and linked evaluation report |
| GET | `/reports` | Schema-validated evaluation reports |
| GET | `/reports/{report_id}` | One model report, or a run review summary |
| POST | `/infer` | Validated computed inference request |
| POST | `/reviews` | Save an idempotent operator decision |
| GET | `/reviews` | List reviews, optionally filtered by run or frame |
| GET | `/reviews/{review_id}` | Retrieve one review |

## Response guarantees

- Request and response bodies are described in the generated OpenAPI schema at `/docs`.
- Validation, missing-resource, unavailable-runtime, unknown-route, and wrong-method failures use one `{"error": {"code": ..., "message": ...}}` envelope.
- Review IDs are deterministic from the reviewed run, frame, decision, and comment. Repeating the same request returns the saved review without adding a duplicate line.
- Evaluation reports are checked against the repository JSON schema before the service starts. Development reports cannot claim that the final test was unsealed.
- Review storage is append-only JSONL inside the configured review directory, with traversal, symbolic-link, duplicate-ID, and malformed-record checks.

## Current runtime boundary

Fixture replay is ready immediately and is always labelled `prediction_source: fixture`. A production checkpoint adapter can be injected through `InferenceRuntime`; until that adapter is configured, the API truthfully reports that computed inference is unavailable. This keeps UI development unblocked without presenting fixture values as model results.
