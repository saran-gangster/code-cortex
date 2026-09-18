# AeroGuard architecture

Solid nodes are implemented in this repository. Dashed nodes are production-scale extensions and are not claimed as complete.

```mermaid
flowchart LR
  AU[AU-AIR archive] --> V[Validated manifest\nunits, boxes, timestamps]
  V --> S[Recording-root protocol]
  S --> B[RGB FCOS baseline]
  S --> F[Flight-state FiLM detector]
  B --> E[Matched evaluation]
  F --> E
  VD[VisDrone DET] --> X[External RGB-only adapter]
  X --> E
  E --> R[Versioned inference records]
  R --> API[FastAPI service\nbounded worker]
  API --> UI[Review console]
  UI --> REV[Operator reviews + export]

  OBJ[(S3-compatible object storage)]:::future
  PG[(PostgreSQL)]:::future
  Q[Autoscaled GPU queue]:::future
  MON[Latency/drift monitoring]:::future
  V -. production .-> OBJ
  API -. scale .-> Q
  REV -. production .-> PG
  Q -. observe .-> MON

  classDef future stroke-dasharray: 5 5,fill:#f5f5f5,color:#666;
```

## Failure boundaries

- Invalid or non-finite state is sanitized before encoding and visibly marked degraded.
- A missing state gate bypasses the FiLM transform; deployment may instead select the separately validated RGB model.
- Inference errors return errors, never plausible dummy detections.
- Fixture and cached records are visibly labeled and have null inference latency.
- Raw archives remain outside the repository; protocol, normalization, config, hashes, and measured results are versioned.
- The evaluator is implemented independently of Torch and fails closed on non-finite boxes/scores, sealed final-test access, or benchmark claims from engineering reports.
- Development checkpoint evaluation is queued after training; final-test evaluation remains blocked by the protocol seal.
