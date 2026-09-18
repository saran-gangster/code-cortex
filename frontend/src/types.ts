export type PredictionSource = 'computed' | 'cached' | 'fixture' | 'annotation'
export type InputMode = 'paired_state' | 'state_masked' | 'separate_rgb_fallback'
export type MetadataAlignment =
  | 'paired_annotation'
  | 'independently_timestamped'
  | 'unavailable'
  | 'injected_delay'
  | 'injected_permutation'

export type Detection = {
  class_name: 'Human' | 'Car' | 'Truck' | 'Van' | 'Motorbike' | 'Bicycle' | 'Bus' | 'Trailer'
  box_xyxy: [number, number, number, number]
  raw_score: number | null
  calibrated_score: number | null
  track_id: string | null
}

export type InferenceRecord = {
  schema_version: '1.0'
  frame_id: string
  model_id: string
  checkpoint_sha256: string
  protocol_sha256: string
  prediction_source: PredictionSource
  source_time_ms: number | null
  original_size: { width: number; height: number }
  input_mode: InputMode
  metadata_alignment: MetadataAlignment
  detections: Detection[]
  quality_flags: string[]
  latency: {
    preprocess_ms: number | null
    model_ms: number | null
    postprocess_ms: number | null
    end_to_end_ms: number | null
  }
  image_url?: string
  image_data_url?: string
  state?: number[]
  rgb_detections?: Detection[]
  state_detections?: Detection[]
  rgb_model_id?: string
  state_model_id?: string
  ground_truth?: Array<{ class_name: string; box_xyxy: number[] }>
}

export type HealthResponse = {
  status: 'ok'
  version: string
  model_ready: boolean
  runtime_mode: 'computed' | 'fixture_replay'
}

export type ReadinessResponse = {
  ready: boolean
  status: 'ready' | 'degraded'
  computed_inference_ready: boolean
  replay_ready: boolean
}

export type CapabilityResponse = {
  api_version: string
  computed_inference: boolean
  fixture_replay: boolean
  review_persistence: boolean
  supported_prediction_sources: PredictionSource[]
  endpoints: Record<string, string>
}

export type RunSummary = {
  run_id: string
  prediction_source: PredictionSource
  frame_count: number
  offline?: boolean
}

export type ModelSummary = {
  model_id: string
  report_id: string | null
  checkpoint_id: string | null
  partition: string | null
  prediction_source: PredictionSource | null
  available_for_inference: boolean
  report?: EvaluationReport | null
}

export type EvaluationReport = {
  schema_version: '1.0'
  artifact_kind: string
  benchmark_claim: false
  partition: 'synthetic_fixture' | 'development' | 'final_test' | 'external_test'
  protocol_sha256: string
  final_test_unsealed: boolean
  model_id: string
  report_id: string
  checkpoint_id: string
  config: Record<string, unknown>
  metrics: Record<string, unknown>
  limitations: string[]
  documentedFallback?: boolean
}

export type ReviewDecision = 'accept' | 'reject' | 'needs_review'

export type ReviewRequest = {
  run_id: string
  frame_id: string
  decision: ReviewDecision
  comment: string
}

export type Review = ReviewRequest & {
  review_id: string
  created_at: string
}

export type ServiceMode = 'loading' | 'online' | 'degraded' | 'offline'
export type Intervention = 'recorded' | 'missing' | 'invalid' | 'delayed'
export type ModelView = 'e1' | 'e2' | 'split'
export type Page = 'live' | 'results' | 'architecture'
