# Jetson Orin Nano validation

## Outcome

The selected AeroGuard YOLO26s detector runs live on an 8 GB Jetson Orin Nano at
the trained 960-pixel input size. A native FP16 TensorRT engine reduced measured
batch-one end-to-end latency from 41.64 ms to 31.90 ms and increased throughput
from 24.01 FPS to 31.35 FPS.

| Backend | Precision | Mean end-to-end | Median | P95 | Model inference | FPS | Peak CUDA tensor memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch | FP16 | 41.64 ms | 41.89 ms | 42.56 ms | 32.18 ms | 24.01 | 77.08 MiB |
| TensorRT 10.3 | FP16 | 31.90 ms | 32.07 ms | 33.11 ms | 21.30 ms | 31.35 | 24.60 MiB |

TensorRT delivered 23.39% lower end-to-end latency, 33.82% lower model inference
latency, 30.52% higher throughput, and 68.09% lower measured CUDA tensor memory.

## Test protocol

- Device: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super, 8 GB.
- JetPack userspace: R36.4.7; TensorRT 10.3.0; CUDA driver 12.6.
- PyTorch 2.3.0 with CUDA 12.4; Ultralytics 8.4.155.
- Power profile: maximum available 15 W mode, with `jetson_clocks` applied.
- Model: selected 9,954,056-parameter YOLO26s checkpoint.
- Input: batch one, 960 x 960, confidence threshold 0.25.
- Data: six consecutive genuine AU-AIR drone frames, decoded once and retained in
  memory so storage latency is excluded.
- Timing: 10 warm-up iterations followed by 50 timed iterations, with CUDA
  synchronization around each complete preprocess/inference/postprocess call.

## Output agreement

The parity pass compared class IDs, confidence scores, and decoded boxes on all six
real frames:

- 6 of 6 PyTorch detections were preserved by TensorRT.
- No PyTorch detection was missing.
- Mean matched-box IoU was 0.9411; minimum matched-box IoU was 0.8453.
- TensorRT emitted one additional borderline car detection at 0.270 confidence on
  one frame. This is consistent with a small FP16/NMS boundary change and should be
  considered when choosing the production confidence threshold.
- The remaining five frames had the same detection count and classes.

This is a deployment parity and latency validation, not a replacement for the frozen
1,580-image accuracy evaluation. The AP50, AP50-95, precision, recall, F1, and test
loss in the main final report remain the authoritative quality metrics. A full Jetson
test-set replay is the next step before a safety or certification claim.

## Reproducible artifacts

- `models/AeroGuard_YOLO26s_960_JetsonOrinNano_TRT10.3_FP16.engine`
- `scripts/jetson_benchmark.py`
- `scripts/compare_jetson_backends.py`
- `reports/final/jetson_pytorch_fp16.json`
- `reports/final/jetson_tensorrt_fp16.json`
- `reports/final/jetson_backend_parity.json`

The engine is 22,620,851 bytes with SHA-256
`172e0d31646bafb51f4b153afe02154841e759bbc3e6adc4b333747ad4a34866`.
TensorRT engines are tied to the GPU and TensorRT/JetPack runtime; rebuild from the
selected `.pt` checkpoint for a different board or runtime version.

Example benchmark command on the validated device:

```bash
python3 scripts/jetson_benchmark.py \
  --model models/AeroGuard_YOLO26s_960_JetsonOrinNano_TRT10.3_FP16.engine \
  --images docs/assets/auair-demo \
  --output reports/final/jetson_tensorrt_fp16.json \
  --imgsz 960 --warmup 10 --iterations 50 --half
```
