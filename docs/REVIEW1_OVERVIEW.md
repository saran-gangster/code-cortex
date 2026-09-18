# AeroGuard — Review 1 overview

Flight-aware aerial detection and review. Evidence snapshot: 18 September 2026. Training status comes from saved files, not a live GPU check.

![AeroGuard Review Architecture](assets/aeroguard-review1-architecture.png)

## Problem and solution

Drone images change with height, speed and angle. Engineers and traffic-monitoring operators need to inspect detections and understand the model's inputs. An image-only detector does not explicitly use flight state.

AeroGuard tests whether paired height, speed and attitude help detect objects. We compare the same FCOS detector in two forms: E2 uses RGB images; E1 adds gated FiLM, a small module that adjusts image features using flight state. Missing state switches that adjustment off. The console shows detections, input status and model evidence, and lets an operator save a review.

## Working evidence

- **Data:** 32,823 AU-AIR frames; 131,977 valid boxes after the parser rejected and logged 54 invalid boxes. Eight roots: five train, one development, two sealed final test.
- **Training:** saved random-control summaries show 5,000 updates per arm, 18,523 available training records, separate Tesla T4 GPUs 0/1, and matching start weights and image schedule. Neither arm used development or final-test roots for training.
- **Fine-tuning:** ImageNet summaries record 5,000 completed updates per arm, and the resolved config records the run as complete. Only the backbone uses ImageNet weights; the AU-AIR head starts random and FiLM starts as identity.
- **Development result:** matched ImageNet E1 reached AP50 0.0692 and AP50:95 0.0234, compared with E2 at 0.0120 and 0.0024. E1 is the leading candidate on the single development recording only.
- **Product:** the API contract, fixture replay, input checks, review/export workflow and evaluation software are implemented. Local live inference still needs a release checkpoint/runtime. Fixture scores are not model results.

## Architecture and stack

Validated data → recording-root split → matched E2/E1 training → development evaluation → versioned evidence/API → review console. VisDrone is a planned external RGB-only evaluation.

Python, PyTorch, TorchVision FCOS–ResNet50–FPN and Lightning handle ML; FastAPI/Pydantic handle the service; React/TypeScript/Vite handle the console. Current storage uses local files, JSON/JSONL and checkpoints; no production database is claimed. PyTest checks behavior. Object storage, PostgreSQL and a scaled GPU queue are future extensions.

## Judging alignment and next steps

UI/UX: clear review flow. Novelty/ML: controlled state conditioning. Dataset use: cleaning and grouped splits. Implementation/progress: saved training and an inspectable console. Architecture/stack: clear flow and practical tools. Feasibility/scaling: local limits and a future worker/storage plan. Judges assess the rubric's 40% milestone; we do not assign ourselves a score.

**Limits:** the development comparison supports E1, but final-test metrics are unavailable. One development recording is weak evidence for generalization. The missing-state fallback, release threshold, empty-scene behavior and autonomous-flight safety remain unproven.

**Next:** evaluate missing/corrupted state, choose the fallback and threshold, connect the release runtime, run the planned external test, and rehearse offline. Keep the final test sealed until freeze.

Sources: [data](DATA_CARD.md), [development results](DEVELOPMENT_RESULTS.md), [architecture](ARCHITECTURE.md), [config](../configs/development_training.yaml), [random E2](../reports/development_random_e2_summary.json), [random E1](../reports/development_random_e1_summary.json), [ImageNet E2](../reports/development_imagenet_e2_summary.json), [ImageNet E1](../reports/development_imagenet_e1_summary.json), and [weight origin](../reports/shared_imagenet_warmstart_summary.json). [Script](REVIEW1_PRESENTATION_SCRIPT.md) · [Diagram guide](REVIEW1_ARCHITECTURE.md).
