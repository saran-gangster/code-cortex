# AeroGuard model card

## Status

Research prototype under active implementation. Engineering gates and one frozen development-root comparison have passed; the final test remains sealed.

## Architecture

- Visual detector: TorchVision FCOS with ResNet-50 FPN, nine model labels including background.
- Flight-state encoder: `8 → 64 → 64` MLP.
- Fusion: per-sample gated residual FiLM on five FPN levels.
- State features: altitude metres; signed XYZ velocity m/s; roll and pitch radians; sine/cosine of yaw.
- Missing state: sanitized neutral vector and a zero gate, or a separately selected RGB fallback.

The FiLM output projections initialize to zero, so the untrained transform is exactly the identity. This does not imply a trained conditioned network equals a separately trained RGB baseline.

## Current evidence

- Random-init FCOS one-frame overfit gate reduced loss from 2.87 to 1.28 over 24 updates.
- Lightning E2 and E1 each completed 40 updates concurrently on separate Tesla T4 GPUs.
- Both arms used shared warmstart hash `0159…36cc` and schedule hash `990b…5db1`.
- E1's zero state gate left the FiLM projection norm at zero.
- E2's paired state produced finite gradients and changed the projection L1 norm from zero to 120.735.
- The matched random-control and ImageNet-backbone pairs each completed 5,000 updates per arm on separate T4 GPUs; the saved summaries prove training provenance, not held-out accuracy.

The training results prove wiring and trainability only. The separate development reports show E1 above E2 on one frozen recording root, but do not establish broad generalization or final-test performance.

## Development evidence

- ImageNet E1 image-only: AP50 `0.0120`, AP50:95 `0.0024`, fixed-point recall `0.3333`.
- ImageNet E2 image + flight state: AP50 `0.0692`, AP50:95 `0.0234`, fixed-point recall `0.4932`.
- Both used the same 5,000-step schedule and frozen development partition. E2 is the leading candidate, not a released model.
- The 0.30 display threshold and missing-state E2 path still need selection work. See [development results](DEVELOPMENT_RESULTS.md).

## Intended use

Supervised, offline review of aerial object detections and their paired input provenance. The operator can compare declared model modes, inspect missing/invalid/delayed state, and record a review.

## Prohibited claims

Not a flight controller, collision predictor, safe-landing system, crash-risk score, emergency certifier, or universal OOD detector. Detection scores are not probabilities of safe flight.

## Evaluation plan

Primary evaluation is pooled and per-root AU-AIR AP50:95/AP50 on two sealed recording roots, with per-class support and recall at frozen thresholds. VisDrone is an external RGB/missing-modality test only. Final-test labels must not select hyperparameters or fallbacks.

## Known limitations

- Only one development recording root has been evaluated; final-test and external results are unavailable.
- Eight recording roots provide weak environment diversity.
- AU-AIR lacks genuinely empty annotated scenes before cleaning.
- Paired annotations do not prove measured zero-latency telemetry.
- Data rights are noncommercial and require primary-source review for any pilot.
