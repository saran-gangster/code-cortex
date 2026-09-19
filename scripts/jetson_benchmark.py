"""Benchmark an AeroGuard Ultralytics checkpoint or TensorRT engine on real frames."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    args = parse_args()
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    image_paths = sorted(
        path for path in args.images.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"no images found under {args.images}")
    frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in image_paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("at least one image could not be decoded")

    model = YOLO(str(args.model), task="detect")
    predict_args = {
        "imgsz": args.imgsz,
        "device": 0,
        "half": args.half,
        "conf": args.conf,
        "verbose": False,
    }
    for index in range(args.warmup):
        model.predict(frames[index % len(frames)], **predict_args)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    wall_ms: list[float] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    latest_results = {}
    for index in range(args.iterations):
        frame_index = index % len(frames)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = model.predict(frames[frame_index], **predict_args)[0]
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - started) * 1000.0)
        preprocess_ms.append(float(result.speed.get("preprocess", 0.0)))
        inference_ms.append(float(result.speed.get("inference", 0.0)))
        postprocess_ms.append(float(result.speed.get("postprocess", 0.0)))
        latest_results[frame_index] = result

    detections = []
    names = model.names
    for frame_index, path in enumerate(image_paths):
        result = latest_results.get(frame_index)
        if result is None:
            result = model.predict(frames[frame_index], **predict_args)[0]
        classes = []
        if result.boxes is not None:
            for class_id, confidence, xyxy in zip(
                result.boxes.cls.detach().cpu().tolist(),
                result.boxes.conf.detach().cpu().tolist(),
                result.boxes.xyxy.detach().cpu().tolist(),
                strict=True,
            ):
                class_index = int(class_id)
                classes.append(
                    {
                        "class_id": class_index,
                        "class_name": names[class_index],
                        "confidence": round(float(confidence), 6),
                        "box_xyxy": [round(float(value), 3) for value in xyxy],
                    }
                )
        detections.append(
            {"image": path.name, "count": len(classes), "detections": classes}
        )

    mean_ms = statistics.fmean(wall_ms)
    payload = {
        "model": str(args.model.resolve()),
        "backend": args.model.suffix.lower().lstrip("."),
        "model_bytes": args.model.stat().st_size,
        "images": len(image_paths),
        "image_names": [path.name for path in image_paths],
        "imgsz": args.imgsz,
        "precision": "fp16" if args.half else "backend_default",
        "confidence": args.conf,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "latency_ms": {
            "mean_end_to_end": mean_ms,
            "median_end_to_end": statistics.median(wall_ms),
            "p95_end_to_end": percentile(wall_ms, 0.95),
            "minimum_end_to_end": min(wall_ms),
            "mean_preprocess": statistics.fmean(preprocess_ms),
            "mean_inference": statistics.fmean(inference_ms),
            "mean_postprocess": statistics.fmean(postprocess_ms),
        },
        "throughput_fps_batch_1": 1000.0 / mean_ms,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "detections": detections,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
