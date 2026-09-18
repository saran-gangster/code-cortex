# AeroGuard TensorRT edge deployment

This directory is a reproducible deployment path for the repository's actual
`FlightAwareFCOS`: TorchVision FCOS-ResNet50-FPN, nine output classes, and the gated
residual FiLM adapter driven by eight normalized flight-state values plus a validity
mask. It does **not** contain or claim a prebuilt engine. TensorRT engines are tied to
the TensorRT version and usually to the target GPU, so build the engine on the target
class of machine and retain the generated provenance sidecar.

## Why the graph ends before NMS

The ONNX graph exports the backbone, five FiLM-conditioned FPN levels, and FCOS heads.
It emits raw per-level classification logits, non-negative box distances, centerness
logits, and anchors. Resize/normalization/padding and FCOS decode/class-aware NMS remain
explicit deterministic host code. This avoids presenting TorchVision's dynamic list,
resize, and `batched_nms` operations as portable TensorRT support when they are not.

The included NumPy postprocessor follows TorchVision 0.19 FCOS semantics: score is
`sqrt(sigmoid(class) * sigmoid(centerness))`; threshold and top-k are applied per FPN
level; `BoxLinearCoder(normalize_by_size=True)` distances are decoded; boxes are clipped;
then class-aware NMS and the global detection limit are applied.

## Input and output contract

The engine is batch-one only:

| Name | dtype | shape | meaning |
|---|---|---|---|
| `images` | FP32 input | `1x3xHxW` | RGB resized, ImageNet-normalized, bottom/right zero-padded; H/W are profile-bounded multiples of 32 |
| `state` | FP32 input | `1x8` | normalized values in the exact feature order in `deployment.example.json` |
| `state_mask` | FP32 input | `1x1` | exactly 0 or 1; zero disables the FiLM residual |

Each level `p3` through `p7` emits four tensors: `cls_logits_*` (`1xAx9`),
`bbox_regression_*` (`1xAx4`), `bbox_ctrness_*` (`1xAx1`), and `anchors_*`
(`1xAx4`). Host postprocessing returns original-image `box_xyxy`, numeric zero-based
model `label`, and `score`. The padded area is never treated as valid image area.

The default dynamic profile covers padded shapes from `320x320` through `576x576`.
Actual preprocessing preserves aspect ratio, targets a 320-pixel short side without
exceeding a 576-pixel long side, and pads to 32. To make a fixed-shape engine, set the
profile `min`, `opt`, and `max` to the same shape; images whose aspect-ratio-preserving
result does not equal that profile are rejected rather than silently stretched.

## Supported and unsupported

| Capability | Status | Notes |
|---|---|---|
| FP32 | Supported | Build config `precision: fp32` |
| FP16 | Supported when hardware reports fast FP16 | Validate with relaxed tolerances appropriate to the target |
| INT8 | Unsupported | No representative calibration loader/corpus is available; the config validator rejects it |
| Dynamic H/W | Supported contract | Batch remains fixed at one; verify the target TensorRT parser/build |
| Dynamic batch | Unsupported | State/image pairing is intentionally batch-one at the edge |
| In-engine resize or NMS | Unsupported by design | Deterministic host implementation is supplied |
| Windows CPU-only development | Imports, help, config, preprocessing, postprocessing tests | ONNX export needs ML extras; engine build/run needs NVIDIA TensorRT/CUDA |
| x86 NVIDIA Linux | Supported target | NVIDIA driver, CUDA-compatible PyTorch, TensorRT and `trtexec`/Python bindings required |
| Jetson | Supported target workflow | Build on Jetson with the TensorRT shipped by JetPack; memory/workspace may need lowering |

Pillow bilinear resize is deterministic for a pinned Pillow build, but may differ by a
small amount from TorchVision tensor interpolation. Validation compares PyTorch and the
runtime using the exact same deployment-preprocessed tensor, which isolates export and
TensorRT error honestly.

## Dependencies

From the repository environment, install the existing ML extra plus lightweight export
tools as needed:

```bash
python -m pip install -e ".[ml]"
python -m pip install onnx onnxruntime
```

Install TensorRT from NVIDIA for the CUDA/JetPack version on the target. Do not install
an arbitrary PyPI TensorRT build that does not match the system CUDA stack. TensorRT
inference also uses CUDA-enabled PyTorch tensors as device buffers, avoiding an extra
mandatory CUDA Python package.

## Export

The exporter accepts only the trusted project Lightning `.ckpt` convention, uses
`torch.load(..., weights_only=True)`, strips the single Lightning `detector.` prefix,
and performs strict model loading. By default it also requires the adjacent
`run_summary.json` and verifies the checkpoint SHA-256. The escape hatch is explicit
and recorded for trusted local checkpoints that predate summaries.

```bash
python scripts/export_onnx_tensorrt.py \
  --checkpoint runs/E2/checkpoints/final.ckpt \
  --config deployment/tensorrt/deployment.example.json \
  --output build/aeroguard_raw.onnx
```

Successful export runs the ONNX checker and writes `aeroguard_raw.onnx.json`. No
sidecar is emitted for a missing, empty, or checker-invalid graph.

## Build on x86 NVIDIA Linux

```bash
python scripts/tensorrt_build_engine.py \
  --onnx build/aeroguard_raw.onnx \
  --config deployment/tensorrt/deployment.example.json \
  --output build/aeroguard_fp16.engine \
  --backend trtexec
```

Use `--dry-run` to print the exact `trtexec` command, or `--backend python` for the
TensorRT Python builder. FP16 fails if the Python builder does not report fast FP16.
The successful build writes an engine sidecar with hashes, profile, precision, builder,
TensorRT version and the non-portability warning.

## Build on Jetson

Run the same export on x86 or Jetson, copy the ONNX plus config to the Jetson, and build
the engine **on the Jetson** using JetPack's `trtexec`:

```bash
python3 scripts/tensorrt_build_engine.py \
  --onnx build/aeroguard_raw.onnx \
  --config deployment/tensorrt/deployment.example.json \
  --output build/aeroguard_jetson_fp16.engine \
  --backend trtexec \
  --trtexec /usr/src/tensorrt/bin/trtexec
```

If 2048 MiB is inappropriate for the device, copy the example manifest, lower
`workspace_mib`, rebuild, and retain that exact config beside the engine.

## Validate and infer

First compare ONNX Runtime raw outputs with PyTorch on a real image and state. Use an
FP16-appropriate tolerance when validating the engine (for example `--atol 0.01
--rtol 0.01`); acceptance tolerances must be chosen and recorded for the deployment,
not retrofitted after seeing failures.

```bash
python scripts/tensorrt_validate.py \
  --checkpoint runs/E2/checkpoints/final.ckpt \
  --config deployment/tensorrt/deployment.example.json \
  --image sample.jpg \
  --state 24.0 0.1 0.0 0.0 -0.05 0.10 0.44 0.90 \
  --state-valid 1 \
  --backend onnxruntime \
  --artifact build/aeroguard_raw.onnx

python scripts/tensorrt_validate.py \
  --checkpoint runs/E2/checkpoints/final.ckpt \
  --config deployment/tensorrt/deployment.example.json \
  --image sample.jpg \
  --state 24.0 0.1 0.0 0.0 -0.05 0.10 0.44 0.90 \
  --state-valid 1 \
  --backend tensorrt \
  --artifact build/aeroguard_fp16.engine \
  --atol 0.01 --rtol 0.01
```

Inference requires the engine sidecar and verifies both engine and config hashes:

```bash
python scripts/tensorrt_infer.py \
  --engine build/aeroguard_fp16.engine \
  --config deployment/tensorrt/deployment.example.json \
  --image sample.jpg \
  --state 24.0 0.1 0.0 0.0 -0.05 0.10 0.44 0.90 \
  --state-valid 1 \
  --output build/prediction.json
```

Set `--state-valid 0` when telemetry is unavailable. Supply finite placeholder state
values; the exact zero mask disables the learned FiLM residual, preserving the visual
path behavior implemented by `GatedResidualFiLM`.
