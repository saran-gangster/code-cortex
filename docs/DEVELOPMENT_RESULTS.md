# Development evaluation results

These values come from the single frozen AU-AIR development recording root. They are not final-test results, safety evidence, or a broad benchmark. The final-test roots remain sealed.

All four arms used the same 0.05 AP score floor, 0.30 fixed display threshold, 300-detection cap, protocol hash, and development support of 16,873 labelled objects.

| Run | AP50 | AP50:95 | Recall at 0.30 | Human recall at 0.30 | Latency p50 / p95 |
|---|---:|---:|---:|---:|---:|
| Random E2, image + flight state | **0.0474** | **0.0146** | **0.3571** | **0.0000** | 39.71 / 43.87 ms |
| Random E1, image only | 0.0375 | 0.0117 | 0.3209 | 0.0000 | 40.38 / 44.78 ms |
| ImageNet E2, image + flight state | **0.0692** | **0.0234** | **0.4932** | **0.0612** | 39.47 / 44.68 ms |
| ImageNet E1, image only | 0.0120 | 0.0024 | 0.3333 | 0.0141 | 34.56 / 41.62 ms |

Within the matched ImageNet pair, E2 is the image-plus-state arm and E1 is the image-only arm. The reported values retain their original run ordering while the corrected labels identify the actual experiment inputs.

This supports the flight-state hypothesis on the development recording only. It does not prove generalization. E1 is the leading checkpoint for the next release gate, but fallback and threshold selection remain open because missing-state E1 has not been evaluated and the 0.30 operating point produces many false positives.

The ImageNet E2 detector emitted 3,810 finite zero-area boxes after coordinate conversion. The hardened adapter dropped and counted them before strict metric evaluation. ImageNet E1 completed under the earlier strict path, so any such box would have failed that run; none was observed. Ground-truth validation remained strict for every run.

Machine-readable sources: [random E2](../reports/evaluations/random-control-e2.json), [random E1](../reports/evaluations/random-control-e1.json), [ImageNet E2](../reports/evaluations/imagenet-finetune-e2.json), and [ImageNet E1](../reports/evaluations/imagenet-finetune-e1.json).
