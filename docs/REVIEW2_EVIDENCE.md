# AeroGuard next-review evidence

This page answers the Review 1 questions with machine-generated evidence. Both matched model arms now train for one complete pass over the frozen training split, store every optimizer-step loss, and are compared on the same held-out development flight.

## Dataset split

| Split | Recording groups | Frames | Proportion | Use |
|---|---:|---:|---:|---|
| Train | 5 | 18,523 | 56.43% | Weight updates only |
| Development | 1 | 5,734 | 17.47% | Model comparison and threshold selection |
| Sealed final test | 2 | 8,566 | 26.10% | Untouched until the release decision |
| **Total** | **8** | **32,823** | **100.00%** | |

The split is made by complete flight recording, not by random frames. This prevents almost identical neighboring video frames from appearing in both training and evaluation.

![AU-AIR train, development, and sealed test proportions](assets/review2-data-split.png)

## Full training evidence

| Arm | Steps | Unique training frames | Coverage | First-200 mean loss | Last-200 mean loss | Time |
|---|---:|---:|---:|---:|---:|---:|
| E2: image + flight state | 18,523 | 18,523 | 100.00% | 2.2207 | 1.3254 | 47.6 min |
| E1: image only | 18,523 | 18,523 | 100.00% | 2.2235 | 1.3287 | 47.6 min |

The thin trace below contains every stored loss value. The strong trace is only a 200-step moving average to make the trend readable; no points are hidden from the artifact.

![Complete E2 and E1 training loss](assets/review2-full-training-loss.png)

![Classification, box regression, and centerness loss](assets/review2-loss-components.png)

## Held-out development results

| Arm | AP50 | AP50:95 | Selected threshold | Precision | Recall | F1 | Detection accuracy | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E2: image + flight state | 0.1698 | 0.0665 | 0.45 | 0.5257 | 0.3939 | 0.4503 | 0.2906 | 6,646 | 5,996 | 10,227 |
| E1: image only | 0.1146 | 0.0445 | 0.45 | 0.2220 | 0.2900 | 0.2515 | 0.1438 | 4,893 | 17,144 | 11,980 |

**Detection accuracy** here means `TP / (TP + FP + FN)`. It is an intersection-style object-detection score, not image-classification accuracy. Precision answers how many shown detections were right. Recall answers how many labelled objects were found. F1 balances both.

AP50 measures the precision-recall curve while a predicted box counts as correct at 50% overlap with the labelled box. AP50:95 repeats that calculation at stricter overlap levels and averages the result.

![Development AP and fixed-threshold metrics](assets/review2-development-metrics.png)

![Development threshold sweep](assets/review2-threshold-sweep.png)

![Per-class AP50 and recall](assets/review2-per-class-metrics.png)

## Flight-state robustness

The selected E2 checkpoint is also evaluated with flight state masked and with state shifted by 1,000 frames. Masking measures the image-only fallback. Shifting is a deliberate misalignment stress test and is never used as a deployment mode.

| E2 evaluation mode | AP50 | AP50:95 | Best F1 | Detection accuracy |
|---|---:|---:|---:|---:|
| Correct paired state | 0.1698 | 0.0665 | 0.4503 | 0.2906 |
| State masked | 0.1167 | 0.0452 | 0.2503 | 0.1430 |
| State shifted by 1,000 frames | 0.1138 | 0.0442 | 0.2515 | 0.1438 |

![E2 missing-state and misalignment checks](assets/review2-state-robustness.png)

E2 is the image-plus-state arm and remains the selected full-pass model. Masked and shifted-state checks document how its state path behaves under intervention.

## Follow-up experiment

E3 and E4 start from the stronger E2 checkpoint, freeze every visual-detector weight, and train only the residual FiLM state adapter. E4 also masks state on half of its deterministic training schedule. These runs test controlled variants of the selected E2 image-plus-state model.

## What is now implemented

- Two matched PyTorch Lightning runs execute concurrently on separate Tesla T4 GPUs.
- Each arm sees the same 18,523-frame schedule and starts from the same hashed ImageNet backbone.
- Every training step is persisted for complete loss and component-loss graphs.
- Evaluation reports include AP50, AP50:95, precision, recall, F1, detection accuracy, raw TP/FP/FN, calibration, and latency.
- The confidence threshold is selected on development data only; the final test remains sealed.
- The FastAPI backend exposes health, readiness, capabilities, runs, frames, models, reports, inference, and idempotent human reviews with typed error responses.

## Honest scope

These are held-out development results, not final benchmark or safety-certification results. The final test is still untouched. AeroGuard remains a supervised offline review tool, not an autonomous flight-control system.
