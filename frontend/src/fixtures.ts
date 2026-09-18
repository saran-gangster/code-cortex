import type { Detection, EvaluationReport, InferenceRecord, ModelSummary, RunSummary } from './types'
import developmentReports from './development-reports.json'
import developmentSequence from '../public/assets/auair-demo/sequence.json'

export const OFFLINE_RUN_ID = 'auair-development-replay'
const ZERO = '0'.repeat(64)

const boxes: Array<[Detection['class_name'], [number, number, number, number], number]> = [
  ['Human', [610, 170, 770, 545], 0.88],
  ['Car', [215, 395, 390, 520], 0.78],
  ['Human', [985, 340, 1095, 590], 0.69],
]

function detection([class_name, box_xyxy, raw_score]: typeof boxes[number]): Detection {
  return { class_name, box_xyxy, raw_score, calibrated_score: null, track_id: null }
}

export const syntheticFrames: InferenceRecord[] = Array.from({ length: 6 }, (_, index) => ({
  schema_version: '1.0',
  frame_id: `fixture-frame-0${180 + index}`,
  model_id: 'aeroguard-offline-fixture-e1',
  checkpoint_sha256: ZERO,
  protocol_sha256: ZERO,
  prediction_source: 'fixture',
  source_time_ms: 36_000 + index * 200,
  original_size: { width: 1280, height: 720 },
  input_mode: index === 5 ? 'state_masked' : 'paired_state',
  metadata_alignment: index === 2 ? 'injected_delay' : index === 5 ? 'unavailable' : 'paired_annotation',
  detections: boxes.slice(0, index === 2 ? 2 : 3).map((item, detectionIndex) => ({
    ...detection(item),
    raw_score: Math.max(0.51, item[2] - index * 0.012 - detectionIndex * 0.01),
  })),
  quality_flags: index === 2 ? ['INJECTED_DELAY'] : index === 5 ? ['STATE_INVALID', 'REVIEW_REQUESTED'] : [],
  latency: { preprocess_ms: null, model_ms: null, postprocess_ms: null, end_to_end_ms: null },
}))

export const offlineFrames: InferenceRecord[] = developmentSequence.frames.map((frame) => ({
  schema_version: '1.0', frame_id: frame.frame_id,
  model_id: developmentSequence.state_model_id,
  checkpoint_sha256: ZERO, protocol_sha256: ZERO,
  prediction_source: 'cached', source_time_ms: frame.source_time_ms,
  original_size: frame.original_size, input_mode: 'paired_state', metadata_alignment: 'paired_annotation',
  detections: frame.state_detections as Detection[], rgb_detections: frame.rgb_detections as Detection[],
  state_detections: frame.state_detections as Detection[], rgb_model_id: developmentSequence.rgb_model_id,
  state_model_id: developmentSequence.state_model_id, state: frame.state, ground_truth: frame.ground_truth,
  quality_flags: [], latency: { preprocess_ms: null, model_ms: null, postprocess_ms: null, end_to_end_ms: null },
  image_url: `/assets/auair-demo/${frame.image_filename}`,
}))

export const offlineRun: RunSummary = {
  run_id: OFFLINE_RUN_ID,
  prediction_source: 'cached',
  frame_count: offlineFrames.length,
  offline: true,
}

// Compact snapshots extracted from the committed development evaluation reports.
export const documentedReports = developmentReports as EvaluationReport[]

export function mergeDocumentedReports(apiReports: EvaluationReport[]): EvaluationReport[] {
  return [...apiReports, ...documentedReports.filter((snapshot) =>
    !apiReports.some((report) => report.model_id === snapshot.model_id && report.partition === snapshot.partition),
  )]
}

export const offlineModels: ModelSummary[] = documentedReports.map((report) => ({
  model_id: report.model_id,
  report_id: report.report_id,
  checkpoint_id: report.checkpoint_id,
  partition: report.partition,
  prediction_source: null,
  available_for_inference: false,
  report,
}))

export function fixtureDetections(frame: InferenceRecord, kind: 'e2' | 'e1'): Detection[] {
  if (kind === 'e2' && frame.rgb_detections) return frame.rgb_detections
  if (kind === 'e1' && frame.state_detections) return frame.state_detections
  if (kind === 'e1') return frame.detections
  return frame.detections.slice(0, Math.max(1, frame.detections.length - 1)).map((item) => ({ ...item, raw_score: Math.max(0, (item.raw_score ?? 0) - 0.08) }))
}
