# Review 1 submission

## GitHub repository

https://github.com/saran-gangster/code-cortex

## Project description

AeroGuard is a flight-aware aerial object-detection and review system. It tests whether drone context—height, speed, and orientation—can improve detection compared with the same image-only model. The system combines TorchVision FCOS, gated FiLM conditioning, PyTorch Lightning, a FastAPI evidence service, and a React/TypeScript operator console. It keeps every result traceable to its model, protocol, and input source and is designed for supervised offline review rather than autonomous flight control.

## Brief progress description

AeroGuard now has a leakage-safe AU-AIR data pipeline, two matched detector variants, a validated development evaluator, an evidence API, and an operator review console. Four training runs completed 5,000 updates each using PyTorch Lightning on two Tesla T4 GPUs. On the single frozen development recording, the ImageNet-initialized image-plus-flight-state model reached AP50 0.0692 and recall 0.493, compared with AP50 0.0120 and recall 0.333 for the matched image-only baseline. These are non-final development results; the final test remains sealed. Current deliverables include reproducible source code and tests, machine-readable evaluation reports, training/development graphs, architecture and model/data documentation, and a Review 1 presentation package. Remaining work is confidence-threshold selection, missing-state evaluation, release-checkpoint integration, and the planned external evaluation.

## Evidence boundary

The Figma link is optional and is not required to validate the repository. Do not present a local plugin package as a live Figma URL. Development metrics are not final-test, deployment, or autonomous-flight safety claims.
