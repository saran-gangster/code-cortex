import type {
  CapabilityResponse,
  EvaluationReport,
  HealthResponse,
  InferenceRecord,
  ModelSummary,
  ReadinessResponse,
  Review,
  ReviewRequest,
  RunSummary,
} from './types'

export const API_BASE = ((import.meta.env.VITE_API_URL as string | undefined) ?? 'http://127.0.0.1:8000').replace(/\/$/, '')

export class ApiError extends Error {
  status: number
  code: string

  constructor(message: string, status = 0, code = 'network_error') {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new ApiError(`Invalid ${label} response`, 0, 'invalid_response')
  return value as Record<string, unknown>
}

function string(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value) throw new ApiError(`Invalid ${label}`, 0, 'invalid_response')
  return value
}

function number(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new ApiError(`Invalid ${label}`, 0, 'invalid_response')
  return value
}

function optionalString(value: unknown): string | null {
  return typeof value === 'string' && value ? value : null
}

function predictionSource(value: unknown): 'computed' | 'cached' | 'fixture' {
  if (value === 'computed' || value === 'cached' || value === 'fixture') return value
  throw new ApiError('Invalid prediction source', 0, 'invalid_response')
}

async function requestJson<T>(path: string, init: RequestInit | undefined, parse: (value: unknown) => T): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...(init?.body ? { 'Content-Type': 'application/json' } : {}), ...init?.headers },
    })
  } catch {
    throw new ApiError(`Cannot reach AeroGuard API at ${API_BASE}`)
  }

  let payload: unknown = null
  try {
    payload = await response.json()
  } catch {
    if (response.ok) throw new ApiError('API returned a non-JSON response', response.status, 'invalid_response')
  }

  if (!response.ok) {
    const envelope = payload && typeof payload === 'object' ? (payload as Record<string, unknown>).error : null
    const detail = envelope && typeof envelope === 'object' ? envelope as Record<string, unknown> : {}
    throw new ApiError(
      typeof detail.message === 'string' ? detail.message : `API request failed (${response.status})`,
      response.status,
      typeof detail.code === 'string' ? detail.code : 'http_error',
    )
  }
  return parse(payload)
}

function parseHealth(value: unknown): HealthResponse {
  const item = object(value, 'health')
  if (item.status !== 'ok' || (item.runtime_mode !== 'computed' && item.runtime_mode !== 'fixture_replay')) throw new ApiError('Invalid health response')
  return { status: 'ok', version: string(item.version, 'version'), model_ready: Boolean(item.model_ready), runtime_mode: item.runtime_mode }
}

function parseReadiness(value: unknown): ReadinessResponse {
  const item = object(value, 'readiness')
  if (item.status !== 'ready' && item.status !== 'degraded') throw new ApiError('Invalid readiness response')
  return { ready: Boolean(item.ready), status: item.status, computed_inference_ready: Boolean(item.computed_inference_ready), replay_ready: Boolean(item.replay_ready) }
}

function parseCapabilities(value: unknown): CapabilityResponse {
  const item = object(value, 'capabilities')
  const sources = Array.isArray(item.supported_prediction_sources) ? item.supported_prediction_sources.map(predictionSource) : []
  const endpoints = object(item.endpoints, 'capability endpoints')
  return {
    api_version: string(item.api_version, 'API version'),
    computed_inference: Boolean(item.computed_inference),
    fixture_replay: Boolean(item.fixture_replay),
    review_persistence: Boolean(item.review_persistence),
    supported_prediction_sources: sources,
    endpoints: Object.fromEntries(Object.entries(endpoints).filter((entry): entry is [string, string] => typeof entry[1] === 'string')),
  }
}

function parseRun(value: unknown): RunSummary {
  const item = object(value, 'run')
  return { run_id: string(item.run_id, 'run id'), prediction_source: predictionSource(item.prediction_source), frame_count: number(item.frame_count, 'frame count') }
}

function parseDetection(value: unknown): InferenceRecord['detections'][number] {
  const item = object(value, 'detection')
  const box = item.box_xyxy
  if (!Array.isArray(box) || box.length !== 4 || box.some((coordinate) => typeof coordinate !== 'number' || !Number.isFinite(coordinate))) {
    throw new ApiError('Invalid detection box', 0, 'invalid_response')
  }
  const allowed = ['Human', 'Car', 'Truck', 'Van', 'Motorbike', 'Bicycle', 'Bus', 'Trailer'] as const
  if (!allowed.includes(item.class_name as typeof allowed[number])) throw new ApiError('Invalid detection class', 0, 'invalid_response')
  return {
    class_name: item.class_name as typeof allowed[number],
    box_xyxy: box as [number, number, number, number],
    raw_score: number(item.raw_score, 'raw score'),
    calibrated_score: typeof item.calibrated_score === 'number' ? item.calibrated_score : null,
    track_id: optionalString(item.track_id),
  }
}

function parseFrame(value: unknown): InferenceRecord {
  const item = object(value, 'frame')
  const size = object(item.original_size, 'original size')
  const latency = object(item.latency, 'latency')
  const inputModes = ['paired_state', 'state_masked', 'separate_rgb_fallback'] as const
  const alignments = ['paired_annotation', 'independently_timestamped', 'unavailable', 'injected_delay', 'injected_permutation'] as const
  if (!inputModes.includes(item.input_mode as typeof inputModes[number])) throw new ApiError('Invalid input mode')
  if (!alignments.includes(item.metadata_alignment as typeof alignments[number])) throw new ApiError('Invalid metadata alignment')
  if (!Array.isArray(item.detections) || !Array.isArray(item.quality_flags)) throw new ApiError('Invalid frame evidence')
  return {
    schema_version: '1.0',
    frame_id: string(item.frame_id, 'frame id'),
    model_id: string(item.model_id, 'model id'),
    checkpoint_sha256: string(item.checkpoint_sha256, 'checkpoint hash'),
    protocol_sha256: string(item.protocol_sha256, 'protocol hash'),
    prediction_source: predictionSource(item.prediction_source),
    source_time_ms: typeof item.source_time_ms === 'number' ? item.source_time_ms : null,
    original_size: { width: number(size.width, 'image width'), height: number(size.height, 'image height') },
    input_mode: item.input_mode as InferenceRecord['input_mode'],
    metadata_alignment: item.metadata_alignment as InferenceRecord['metadata_alignment'],
    detections: item.detections.map(parseDetection),
    quality_flags: item.quality_flags.filter((flag): flag is string => typeof flag === 'string'),
    latency: {
      preprocess_ms: typeof latency.preprocess_ms === 'number' ? latency.preprocess_ms : null,
      model_ms: typeof latency.model_ms === 'number' ? latency.model_ms : null,
      postprocess_ms: typeof latency.postprocess_ms === 'number' ? latency.postprocess_ms : null,
      end_to_end_ms: typeof latency.end_to_end_ms === 'number' ? latency.end_to_end_ms : null,
    },
    image_url: optionalString(item.image_url) ?? undefined,
    image_data_url: optionalString(item.image_data_url) ?? undefined,
  }
}

function parseModel(value: unknown): ModelSummary {
  const item = object(value, 'model')
  return {
    model_id: string(item.model_id, 'model id'),
    report_id: optionalString(item.report_id),
    checkpoint_id: optionalString(item.checkpoint_id),
    partition: optionalString(item.partition),
    prediction_source: item.prediction_source == null ? null : predictionSource(item.prediction_source),
    available_for_inference: Boolean(item.available_for_inference),
    report: item.report ? parseReport(item.report) : null,
  }
}

function parseReport(value: unknown): EvaluationReport {
  const item = object(value, 'report')
  const partitions = ['synthetic_fixture', 'development', 'final_test', 'external_test'] as const
  if (!partitions.includes(item.partition as typeof partitions[number])) throw new ApiError('Invalid report partition')
  if (!Array.isArray(item.limitations)) throw new ApiError('Invalid report limitations')
  return {
    schema_version: '1.0',
    artifact_kind: string(item.artifact_kind, 'artifact kind'),
    benchmark_claim: false,
    partition: item.partition as EvaluationReport['partition'],
    protocol_sha256: string(item.protocol_sha256, 'protocol hash'),
    final_test_unsealed: Boolean(item.final_test_unsealed),
    model_id: string(item.model_id, 'model id'),
    report_id: optionalString(item.report_id) ?? string(item.model_id, 'model id'),
    checkpoint_id: string(item.checkpoint_id, 'checkpoint id'),
    config: object(item.config, 'report config'),
    metrics: object(item.metrics, 'report metrics'),
    limitations: item.limitations.filter((limitation): limitation is string => typeof limitation === 'string'),
  }
}

function parseReview(value: unknown): Review {
  const item = object(value, 'review')
  if (item.decision !== 'accept' && item.decision !== 'reject' && item.decision !== 'needs_review') throw new ApiError('Invalid review decision')
  return {
    review_id: string(item.review_id, 'review id'), created_at: string(item.created_at, 'review date'),
    run_id: string(item.run_id, 'run id'), frame_id: string(item.frame_id, 'frame id'),
    decision: item.decision, comment: typeof item.comment === 'string' ? item.comment : '',
  }
}

export const api = {
  health: () => requestJson('/health', undefined, parseHealth),
  readiness: () => requestJson('/readiness', undefined, parseReadiness),
  capabilities: () => requestJson('/capabilities', undefined, parseCapabilities),
  runs: () => requestJson('/runs', undefined, (value) => {
    const payload = object(value, 'runs')
    if (!Array.isArray(payload.runs)) throw new ApiError('Invalid runs response')
    return payload.runs.map(parseRun)
  }),
  frames: (runId: string) => requestJson(`/runs/${encodeURIComponent(runId)}/frames`, undefined, (value) => {
    if (!Array.isArray(value)) throw new ApiError('Invalid frames response')
    return value.map(parseFrame)
  }),
  frame: (runId: string, frameId: string) => requestJson(`/runs/${encodeURIComponent(runId)}/frames/${encodeURIComponent(frameId)}`, undefined, parseFrame),
  models: () => requestJson('/models', undefined, (value) => {
    const payload = object(value, 'models')
    if (!Array.isArray(payload.models)) throw new ApiError('Invalid models response')
    return payload.models.map(parseModel)
  }),
  model: (modelId: string) => requestJson(`/models/${encodeURIComponent(modelId)}`, undefined, parseModel),
  reports: () => requestJson('/reports', undefined, (value) => {
    const payload = object(value, 'reports')
    if (!Array.isArray(payload.reports)) throw new ApiError('Invalid reports response')
    return payload.reports.map(parseReport)
  }),
  report: (reportId: string) => requestJson(`/reports/${encodeURIComponent(reportId)}`, undefined, parseReport),
  reviews: (runId?: string, frameId?: string) => {
    const query = new URLSearchParams()
    if (runId) query.set('run_id', runId)
    if (frameId) query.set('frame_id', frameId)
    const suffix = query.size ? `?${query}` : ''
    return requestJson(`/reviews${suffix}`, undefined, (value) => {
      const payload = object(value, 'reviews')
      if (!Array.isArray(payload.reviews)) throw new ApiError('Invalid reviews response')
      return payload.reviews.map(parseReview)
    })
  },
  review: (reviewId: string) => requestJson(`/reviews/${encodeURIComponent(reviewId)}`, undefined, parseReview),
  saveReview: (review: ReviewRequest) => requestJson('/reviews', { method: 'POST', body: JSON.stringify(review) }, parseReview),
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Unexpected error'
}
