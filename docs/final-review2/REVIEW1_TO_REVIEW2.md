# AeroGuard — Review 1 to Review 2

## One-line summary

Review 1 proved that the idea and training pipeline worked. Review 2 replaced the early pilot result with a complete, controlled evaluation and selected the image-only E1 model honestly.

| Area | Review 1 | Review 2 |
|---|---|---|
| Goal | Setup, architecture, and early feasibility | Full-pass evidence, model selection, working product, and deployment path |
| Dataset use | 5,000-update pilot per arm | Full 18,523-frame training pass per main arm |
| Split reporting | Initial protocol | Exact split: 18,523 train, 5,734 development, 8,566 sealed final test |
| Leakage control | Planned grouped split | Complete flight recordings kept together and verified |
| Training evidence | Pilot summaries | Every optimizer-step loss stored for all 18,523 steps |
| Compute | Matched experiments | E1 and E2 trained in parallel on both Kaggle T4 GPUs with PyTorch Lightning |
| Early conclusion | E2 looked promising: AP50 0.0692 vs E1 0.0120 | Full pass reversed the ranking: E1 0.1698 vs E2 0.1146 |
| Root-cause work | Not yet available | Correct, masked, and shifted-state stress tests exposed visual-weight drift in E2 |
| Corrective experiments | Not yet available | E3/E4 froze and byte-verified E1 visual weights; only the state adapter trained |
| Final development decision | Continue testing state fusion | E4 reached 0.1697 but did not beat E1; keep E1 |
| Accuracy-style evidence | AP50 and recall from pilot | AP50, precision, recall, F1, detection accuracy, TP, FP, and FN |
| Product | Architecture and early interface plan | Working FastAPI + React review console with six real AU-AIR frames and stored predictions |
| Edge inference | Planned | ONNX export, TensorRT build, validation, and inference scripts implemented |
| Final test | Sealed | Still sealed: 8,566 frames will be opened once after freezing the system |

## Review 1 position

- We had a clear problem statement and a matched E1 versus E2 architecture.
- Both arms had completed 5,000 pilot updates.
- E2 appeared stronger in the early development recording.
- We correctly described that result as preliminary.
- The main open questions were full training, exact split proportions, full loss graphs, accuracy-style metrics, and a working end-to-end system.

## Review 2 position

- The complete AU-AIR protocol contains 32,823 frames.
- The split is 56.43% training, 17.47% development, and 26.10% sealed final test.
- E1 and E2 each completed all 18,523 training steps.
- Full loss histories and held-out metrics are saved and graphed.
- E1 is the best development model at AP50 0.1698.
- E2’s weaker result was traced mainly to visual-weight drift during joint training.
- Frozen-visual E4 nearly matched E1 but did not prove a reliable benefit from flight state.
- The UI, backend, real-frame replay, evidence drawer, review workflow, and TensorRT scripts now form an end-to-end engineering path.
- The final test remains untouched, which protects the credibility of the final evaluation.

## The progress story to tell judges

1. **We listened to Review 1:** we added exact train/development/test proportions, full loss histories, accuracy-style measures, and raw error counts.
2. **We scaled the experiment:** from 5,000 pilot updates to all 18,523 training frames per main arm.
3. **We accepted a changed conclusion:** the full-pass evidence showed that E1 is stronger than jointly trained E2.
4. **We diagnosed the failure:** stress tests and frozen-visual adapters isolated visual-weight drift.
5. **We made a conservative decision:** E1 remains selected because E4 did not beat it.
6. **We built the product path:** real-data review UI, backend evidence service, human-review export, and TensorRT tooling.
7. **We protected the final result:** no model has been selected using the sealed 8,566-frame final test.
