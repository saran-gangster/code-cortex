#!/usr/bin/env python3
"""Evaluate one trained AeroGuard checkpoint on the frozen development root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from aeroguard.data.provenance import validate_protocol_bundle
from aeroguard.evaluation import (
    build_evaluation_report,
    evaluate_calibration,
    evaluate_detections,
    match_detections,
    normalize_model_prediction,
    write_evaluation_report,
)
from aeroguard.models.fcos import build_flight_aware_fcos
from aeroguard.training import AeroGuardDetectorModule


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load_partition_records(manifest: Path, protocol: dict, partition: str) -> list[dict]:
    if partition != "development":
        raise ValueError("this evaluator is deliberately restricted to the development partition")
    roots = set(protocol["selection"][partition])
    records: list[dict] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["recording_root"] in roots:
                records.append(record)
    records.sort(key=lambda item: (item["recording_root"], item["source_time_iso_no_timezone"]))
    if not records or {item["recording_root"] for item in records} != roots:
        raise RuntimeError("manifest does not fully represent the frozen development roots")
    return records


class EvaluationDataset(Dataset):
    def __init__(self, data_root: Path, records: list[dict], normalizer: dict) -> None:
        self.data_root = data_root
        self.records = records
        self.means = torch.tensor(normalizer["mean"], dtype=torch.float32)
        self.scales = torch.tensor(normalizer["std"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(self.data_root / record["image_relative_path"]) as image:
            pixels = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
        state = (torch.tensor(record["state"], dtype=torch.float32) - self.means) / self.scales
        return pixels, state, record


def collate(samples):
    images, states, records = zip(*samples)
    return list(images), torch.stack(states), list(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("masked", "paired"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/auair/04_AUAIR_multimodal_uav"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/auair.jsonl"))
    parser.add_argument("--protocol", type=Path, default=Path("manifests/protocol.json"))
    parser.add_argument("--normalizer", type=Path, default=Path("manifests/state_normalizer.json"))
    parser.add_argument("--score-floor", type=float, default=0.05)
    parser.add_argument("--display-threshold", type=float, default=0.30)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    normalizer = json.loads(args.normalizer.read_text(encoding="utf-8"))
    partitions = validate_protocol_bundle(protocol, normalizer)
    records = load_partition_records(args.manifest, protocol, "development")
    used_roots = {record["recording_root"] for record in records}
    if used_roots != set(partitions.development):
        raise RuntimeError("evaluation manifest roots do not exactly match development")

    checkpoint_sha = sha256_file(args.checkpoint)
    training_summary_path = args.checkpoint.parent / "run_summary.json"
    if not training_summary_path.is_file():
        raise RuntimeError(f"missing checkpoint provenance summary: {training_summary_path}")
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    expected_arm = "E2_paired_film" if args.mode == "paired" else "E1_rgb_masked"
    provenance_checks = {
        "experiment_arm": expected_arm,
        "checkpoint_sha256": checkpoint_sha,
        "protocol_sha256": protocol["protocol_sha256"],
        "normalizer_sha256": normalizer["normalizer_sha256"],
    }
    for field, expected in provenance_checks.items():
        if training_summary.get(field) != expected:
            raise RuntimeError(
                f"checkpoint provenance mismatch for {field}: "
                f"{training_summary.get(field)!r} != {expected!r}"
            )
    if training_summary.get("final_test_unsealed") is not False:
        raise RuntimeError("checkpoint summary does not preserve the final-test seal")
    if training_summary.get("development_roots_used") or training_summary.get("final_test_roots_used"):
        raise RuntimeError("checkpoint summary reports non-training roots during optimization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = build_flight_aware_fcos(pretrained=False, min_size=320, max_size=576)
    module = AeroGuardDetectorModule(detector)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    model = module.detector.to(device).eval()
    model.detector.score_thresh = args.score_floor
    model.detector.detections_per_img = args.max_detections
    model.detector.topk_candidates = max(args.max_detections, model.detector.topk_candidates)

    loader = DataLoader(
        EvaluationDataset(args.data_root, records, normalizer),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate,
    )
    state_mask_value = 1.0 if args.mode == "paired" else 0.0
    evaluation_records: list[dict] = []
    calibration_scores: list[float] = []
    calibration_correctness: list[bool] = []
    model_times_ms: list[float] = []
    samples: list[dict] = []
    dropped_degenerate_predictions = 0

    with torch.inference_mode():
        for index, (images, state, batch_records) in enumerate(loader):
            images = [image.to(device, non_blocking=True) for image in images]
            state = state.to(device, non_blocking=True)
            state_mask = torch.full((len(images),), state_mask_value, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = model(images, state, state_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if index >= 20:
                model_times_ms.append(elapsed_ms)

            record = batch_records[0]
            output = outputs[0]
            predictions = []
            for box, label, score in zip(
                output["boxes"].cpu().tolist(),
                output["labels"].cpu().tolist(),
                output["scores"].cpu().tolist(),
                strict=True,
            ):
                prediction = normalize_model_prediction(box, label, score)
                if prediction is None:
                    dropped_degenerate_predictions += 1
                else:
                    predictions.append(prediction)
            targets = [
                {"box_xyxy": box, "label": label}
                for box, label in zip(record["boxes_xyxy"], record["model_labels"], strict=True)
            ]
            evaluation_records.append(
                {
                    "image_id": record["frame_id"],
                    "recording_root": record["recording_root"],
                    "predictions": predictions,
                    "targets": targets,
                }
            )
            retained = [item for item in predictions if item["score"] >= args.display_threshold]
            match = match_detections(retained, targets, iou_threshold=0.5)
            calibration_scores.extend(sorted((item["score"] for item in retained), reverse=True))
            calibration_correctness.extend(match.prediction_correctness)
            if len(samples) < 20:
                samples.append(
                    {
                        "frame_id": record["frame_id"],
                        "recording_root": record["recording_root"],
                        "prediction_source": "computed",
                        "input_mode": "paired" if args.mode == "paired" else "masked",
                        "predictions": predictions,
                    }
                )

    result = evaluate_detections(
        evaluation_records,
        score_floor=args.score_floor,
        max_detections_per_image=args.max_detections,
        fixed_score_threshold=args.display_threshold,
        fixed_iou_threshold=0.5,
    )
    metrics = result.as_dict()
    metrics["dropped_degenerate_predictions"] = dropped_degenerate_predictions
    human = result.pooled.per_class.get(1)
    metrics["human_recall_at_fixed_operating_point"] = human.recall if human else None
    metrics["calibration"] = evaluate_calibration(
        calibration_scores,
        calibration_correctness,
        bin_count=10,
    ).to_dict()
    metrics["latency"] = {
        "scope": "model_only_batch_1_after_20_frame_warmup",
        "sample_count": len(model_times_ms),
        "p50_ms": statistics.median(model_times_ms) if model_times_ms else None,
        "p95_ms": percentile(model_times_ms, 0.95),
    }
    report = build_evaluation_report(
        artifact_kind="auair_development_evaluation_not_final_test",
        partition="development",
        protocol_sha256=protocol["protocol_sha256"],
        final_test_unsealed=False,
        model_id=args.model_id,
        checkpoint_id=checkpoint_sha,
        config={
            "state_mode": args.mode,
            "score_floor": args.score_floor,
            "display_threshold": args.display_threshold,
            "max_detections_per_image": args.max_detections,
            "normalizer_sha256": normalizer["normalizer_sha256"],
            "training_schedule_sha256": training_summary["matched_frame_schedule_sha256"],
            "shared_warmstart_sha256": training_summary["shared_warmstart_sha256"],
            "initial_weights_origin": training_summary["initial_weights_origin"],
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "physical_gpu_id": os.getenv("AEROGUARD_PHYSICAL_GPU", "unrecorded"),
            "torch_version": torch.__version__,
        },
        metrics=metrics,
        limitations=[
            "Development-root evidence only; final-test roots remain sealed.",
            "Thresholds are not frozen for release by this report.",
            "Calibration covers emitted detections only and does not measure missed objects.",
            "Finite zero-area detector outputs are dropped and counted before evaluation.",
            "Random-initialized runs are controls unless the model id explicitly states ImageNet.",
        ],
    )
    write_evaluation_report(args.output / "evaluation_report.json", report)
    (args.output / "sample_predictions.json").write_text(
        json.dumps(samples, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
