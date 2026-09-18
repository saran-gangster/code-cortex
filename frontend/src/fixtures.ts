import type { Detection, EvaluationReport, InferenceRecord, ModelSummary, RunSummary } from './types'

export const OFFLINE_RUN_ID = 'offline-demo-run'
const ZERO = '0'.repeat(64)

const boxes: Array<[Detection['class_name'], [number, number, number, number], number]> = [
  ['Human', [610, 170, 770, 545], 0.88],
  ['Car', [215, 395, 390, 520], 0.78],
  ['Human', [985, 340, 1095, 590], 0.69],
]

function detection([class_name, box_xyxy, raw_score]: typeof boxes[number]): Detection {
  return { class_name, box_xyxy, raw_score, calibrated_score: null, track_id: null }
}

export const offlineFrames: InferenceRecord[] = Array.from({ length: 6 }, (_, index) => ({
  schema_version: '1.0',
  frame_id: `fixture-frame-0${180 + index}`,
  model_id: 'aeroguard-offline-fixture-e2',
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

export const offlineRun: RunSummary = {
  run_id: OFFLINE_RUN_ID,
  prediction_source: 'fixture',
  frame_count: offlineFrames.length,
  offline: true,
}

function documentedReport(modelId: string, checkpointId: string, ap50: number, recall: number, detectionAccuracy: number, latency: number): EvaluationReport {
  return {
    schema_version: '1.0',
    artifact_kind: 'development_evaluation',
    benchmark_claim: false,
    partition: 'development',
    protocol_sha256: ZERO,
    final_test_unsealed: false,
    model_id: modelId,
    report_id: modelId,
    checkpoint_id: checkpointId,
    config: { evidence_scope: 'Review 2 documented snapshot' },
    metrics: {
      evaluated_frame_count: 5734,
      pooled: { ap50 },
      latency: { p95_ms: latency },
      operating_point_sweep: {
        best_f1_point: { recall, detection_accuracy: detectionAccuracy },
        selection_partition: 'development',
      },
    },
    limitations: [
      'Development-root evidence only; final-test roots remain sealed.',
      'Detection accuracy means TP/(TP+FP+FN), not image-classification accuracy.',
      'This offline value is a documented snapshot; connect the API for repository-discovered reports.',
    ],
    documentedFallback: true,
  }
}

export const documentedReports: EvaluationReport[] = [
  documentedReport('imagenet-full-pass-e1-rgb-masked', '7cea7f2cd4c6…', 0.1697852101126862, 0.3938837195519469, 0.29061174515719973, 43.9159),
  documentedReport('imagenet-full-pass-e2-paired-film', '4a7b05db34ae…', 0.11455784404639657, 0.28998992473182006, 0.1438398447834906, 44.5629),
]

export const offlineModels: ModelSummary[] = documentedReports.map((report) => ({
  model_id: report.model_id,
  report_id: report.report_id,
  checkpoint_id: report.checkpoint_id,
  partition: report.partition,
  prediction_source: null,
  available_for_inference: false,
  report,
}))

export function fixtureDetections(frame: InferenceRecord, kind: 'e1' | 'e2'): Detection[] {
  if (kind === 'e2') return frame.detections
  return frame.detections.slice(0, Math.max(1, frame.detections.length - 1)).map((item) => ({ ...item, raw_score: Math.max(0, item.raw_score - 0.08) }))
}
