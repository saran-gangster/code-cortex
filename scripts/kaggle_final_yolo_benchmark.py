"""Train and evaluate the final train-heavy AU-AIR YOLO benchmark on Kaggle.

The script keeps complete recording roots together, records conventional train,
validation, and test losses/metrics, and writes all durable outputs below the
explicit --output directory. It is intended for a single Kaggle T4.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

CLASS_NAMES = ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]
TRAIN_ROOTS = {
    "20190829091111",
    "20190905091750",
    "20190905103112",
    "20190905111947",
    "20190905112522",
    "20190906150731",
}
VALIDATION_ROOTS = {"20190905142119"}
FINAL_TEST_ROOTS = {"20190905143505"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--hours", type=float, default=2.25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--multi-scale", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def split_for_root(root: str) -> str:
    if root in TRAIN_ROOTS:
        return "train"
    if root in VALIDATION_ROOTS:
        return "val"
    if root in FINAL_TEST_ROOTS:
        return "test"
    raise ValueError(f"unassigned recording root: {root}")


def create_dataset(records: list[dict[str, Any]], image_root: Path, output: Path) -> Path:
    dataset = output / "dataset"
    split_frames: Counter[str] = Counter()
    split_boxes: Counter[str] = Counter()
    split_classes: dict[str, Counter[int]] = {name: Counter() for name in ("train", "val", "test")}
    split_roots: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
    missing_images: list[str] = []

    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    for record in records:
        root = str(record["recording_root"])
        split = split_for_root(root)
        source = image_root / record["image_relative_path"]
        if not source.exists():
            missing_images.append(str(source))
            continue
        stem = str(record["frame_id"])
        suffix = source.suffix.lower() or ".jpg"
        image_target = dataset / "images" / split / f"{stem}{suffix}"
        label_target = dataset / "labels" / split / f"{stem}.txt"
        if not image_target.exists():
            image_target.symlink_to(source)

        width = float(record["original_size"]["width"])
        height = float(record["original_size"]["height"])
        lines: list[str] = []
        for box, label in zip(record["boxes_xyxy"], record["model_labels"], strict=True):
            x1, y1, x2, y2 = map(float, box)
            x_center = ((x1 + x2) / 2.0) / width
            y_center = ((y1 + y2) / 2.0) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            class_id = int(label) - 1
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(f"invalid class id {class_id} for {stem}")
            values = (x_center, y_center, box_width, box_height)
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"invalid normalized box for {stem}: {values}")
            lines.append(f"{class_id} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}")
            split_classes[split][class_id] += 1
        label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        split_frames[split] += 1
        split_boxes[split] += len(lines)
        split_roots[split].add(root)

    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} images missing; first={missing_images[0]}")
    if set.union(*split_roots.values()) != TRAIN_ROOTS | VALIDATION_ROOTS | FINAL_TEST_ROOTS:
        raise RuntimeError("recording root coverage mismatch")
    if any(split_roots[a] & split_roots[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("recording root leakage detected")

    data_yaml = output / "auair_trainheavy.yaml"
    payload = {
        "path": str(dataset),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    total_frames = sum(split_frames.values())
    write_json(
        output / "split_manifest.json",
        {
            "schema_version": 1,
            "split_unit": "complete_recording_root",
            "total_frames": total_frames,
            "manifest_sha256": sha256(Path(args.manifest)),
            "splits": {
                split: {
                    "frames": split_frames[split],
                    "percentage": 100.0 * split_frames[split] / total_frames,
                    "boxes": split_boxes[split],
                    "roots": sorted(split_roots[split]),
                    "class_boxes": {
                        CLASS_NAMES[class_id]: split_classes[split][class_id]
                        for class_id in range(len(CLASS_NAMES))
                    },
                }
                for split in ("train", "val", "test")
            },
        },
    )
    return data_yaml


def metric_payload(metrics: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "results_dict": dict(metrics.results_dict),
        "speed_ms": dict(metrics.speed),
    }
    box = metrics.box
    per_class = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        try:
            precision, recall, ap50, ap5095 = box.class_result(class_id)
        except Exception:  # noqa: BLE001,S112 - tolerate absent classes across releases
            continue
        per_class[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "ap50": float(ap50),
            "ap50_95": float(ap5095),
        }
    payload["per_class"] = per_class
    try:
        f1_curve = np.asarray(box.f1_curve, dtype=np.float64)
        px = np.asarray(box.px, dtype=np.float64)
        mean_f1 = f1_curve.mean(axis=0) if f1_curve.ndim == 2 else f1_curve
        index = int(np.nanargmax(mean_f1))
        payload["best_mean_f1"] = float(mean_f1[index])
        payload["best_f1_confidence"] = float(px[index])
    except Exception as exc:  # noqa: BLE001 - metric curve fields vary by release
        payload["curve_error"] = repr(exc)
    return json_safe(payload)


def raw_counts(metrics: Any) -> dict[str, Any]:
    matrix = np.asarray(metrics.confusion_matrix.matrix, dtype=np.float64)
    class_count = len(CLASS_NAMES)
    if matrix.shape != (class_count + 1, class_count + 1):
        return {"matrix": matrix.tolist(), "error": f"unexpected shape {matrix.shape}"}
    true_positives = float(np.trace(matrix[:class_count, :class_count]))
    false_positives = float(matrix[:class_count, class_count].sum())
    false_negatives = float(matrix[class_count, :class_count].sum())
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (
        true_positives / (true_positives + false_positives + false_negatives)
        if true_positives + false_positives + false_negatives
        else 0.0
    )
    return {
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detection_accuracy": accuracy,
        "matrix_predicted_rows_true_columns": matrix.tolist(),
    }


def test_loss_from_trainer(model: Any) -> dict[str, Any]:
    trainer = model.trainer
    if trainer is None:
        return {"error": "trainer unavailable"}
    best = Path(trainer.best)
    if best.exists():
        checkpoint = torch.load(best, map_location=trainer.device, weights_only=False)
        source = checkpoint.get("ema") or checkpoint.get("model")
        if source is not None:
            trainer.model.load_state_dict(source.float().state_dict(), strict=False)
    original_validation = trainer.data.get("val")
    original_loader = trainer.test_loader
    trainer.data["val"] = trainer.data["test"]
    try:
        trainer.test_loader = trainer.get_dataloader(
            trainer.data["test"], batch_size=trainer.batch_size, rank=-1, mode="val"
        )
        validator = trainer.get_validator()
        results = validator(trainer=trainer)
        payload = {
            (key.replace("val/", "test/", 1) if key.startswith("val/") else key): value
            for key, value in dict(results).items()
        }
        payload["partition"] = "test"
        payload["images"] = len(trainer.test_loader.dataset)
        return json_safe(payload)
    except Exception as exc:  # noqa: BLE001 - persist the evaluator failure in the artifact
        return {"error": repr(exc)}
    finally:
        trainer.test_loader = original_loader
        trainer.data["val"] = original_validation


def create_judge_curves(run_dir: Path, output: Path, test_loss: dict[str, Any]) -> None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    def column(name: str) -> np.ndarray:
        return np.asarray([float(row[name].strip()) for row in rows], dtype=np.float64)

    epochs = np.arange(1, len(rows) + 1)
    fieldnames = set(rows[0])
    # Ultralytics detector families expose different regression loss names
    # (for example dfl_loss versus l1_loss).  Discover every recorded loss
    # column so the judge graph remains complete across model families.
    train_loss_names = sorted(
        name for name in fieldnames if name.startswith("train/") and name.endswith("_loss")
    )
    val_loss_names = sorted(
        name for name in fieldnames if name.startswith("val/") and name.endswith("_loss")
    )
    train_loss = sum((column(name) for name in train_loss_names), np.zeros(len(rows)))
    val_loss = sum((column(name) for name in val_loss_names), np.zeros(len(rows)))
    precision = column("metrics/precision(B)")
    recall = column("metrics/recall(B)")
    map50 = column("metrics/mAP50(B)")
    map5095 = column("metrics/mAP50-95(B)")
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    detection_accuracy = precision * recall / np.maximum(
        precision + recall - precision * recall, 1e-12
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].plot(epochs, train_loss, marker="o", label="Training loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Validation loss")
    test_terms = [
        value
        for name, value in test_loss.items()
        if name.startswith("val/") and name.endswith("_loss") and isinstance(value, (int, float))
    ]
    if test_terms:
        axes[0].axhline(sum(test_terms), color="black", linestyle="--", label="Final test loss")
    axes[0].set(title="Train / validation / test loss", xlabel="Epoch", ylabel="Total detector loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for values, label in (
        (precision, "Precision"),
        (recall, "Recall"),
        (f1, "F1"),
        (detection_accuracy, "Detection accuracy"),
        (map50, "mAP50"),
        (map5095, "mAP50-95"),
    ):
        axes[1].plot(epochs, values, marker="o", label=label)
    axes[1].set(title="Conventional validation metrics", xlabel="Epoch", ylabel="Metric", ylim=(0.0, 1.0))
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2)
    figure.suptitle("AeroGuard final pretrained-detector training record", fontweight="bold")
    figure.savefig(output / "judge_training_and_metrics.png", dpi=180)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "run_request.json",
        {
            "model": args.model,
            "imgsz": args.imgsz,
            "hours": args.hours,
            "epochs_ceiling": args.epochs,
            "batch": args.batch,
            "device": args.device,
            "seed": args.seed,
            "started_unix": time.time(),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    )
    records = load_records(args.manifest)
    data_yaml = create_dataset(records, args.image_root, args.output)

    from ultralytics import YOLO

    model = YOLO(args.model)
    run_name = f"{Path(args.model).stem}-{args.imgsz}-trainheavy"
    train_started = time.perf_counter()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        time=args.hours,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        project=str(args.output / "runs"),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-2,
        weight_decay=1e-3,
        warmup_epochs=1.0,
        cos_lr=True,
        amp=True,
        deterministic=False,
        seed=args.seed,
        patience=8,
        close_mosaic=2,
        mosaic=1.0,
        mixup=0.10,
        multi_scale=args.multi_scale,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=3.0,
        translate=0.10,
        scale=0.50,
        fliplr=0.5,
        plots=True,
        save=True,
        verbose=True,
    )
    run_dir = Path(model.trainer.save_dir)
    best = Path(model.trainer.best)
    if not best.exists():
        best = Path(model.trainer.last)
    test_loss = test_loss_from_trainer(model)
    write_json(args.output / "test_loss.json", test_loss)

    selected = YOLO(str(best))
    validation_metrics = selected.val(
        data=str(data_yaml), split="val", imgsz=args.imgsz, batch=args.batch,
        device=0, workers=args.workers, conf=0.001, iou=0.6, plots=True,
        save_json=True, project=str(args.output / "evaluation"), name="validation-ap",
    )
    validation_payload = metric_payload(validation_metrics)
    threshold = float(validation_payload.get("best_f1_confidence", 0.25))
    validation_operating = selected.val(
        data=str(data_yaml), split="val", imgsz=args.imgsz, batch=args.batch,
        device=0, workers=args.workers, conf=threshold, iou=0.6, plots=True,
        project=str(args.output / "evaluation"), name="validation-operating-point",
    )
    validation_payload["selected_confidence"] = threshold
    validation_payload["operating_point_counts"] = raw_counts(validation_operating)
    write_json(args.output / "validation_metrics.json", validation_payload)

    test_metrics = selected.val(
        data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch,
        device=0, workers=args.workers, conf=0.001, iou=0.6, plots=True,
        save_json=True, project=str(args.output / "evaluation"), name="final-test-ap",
    )
    test_payload = metric_payload(test_metrics)
    test_operating = selected.val(
        data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch,
        device=0, workers=args.workers, conf=threshold, iou=0.6, plots=True,
        project=str(args.output / "evaluation"), name="final-test-operating-point",
    )
    test_payload["frozen_validation_confidence"] = threshold
    test_payload["operating_point_counts"] = raw_counts(test_operating)
    test_payload["loss"] = test_loss
    write_json(args.output / "final_test_metrics.json", test_payload)
    create_judge_curves(run_dir, args.output, test_loss)

    write_json(
        args.output / "summary.json",
        {
            "artifact_kind": "aeroguard_final_trainheavy_pretrained_detector",
            "architecture": args.model,
            "best_checkpoint": str(best),
            "best_checkpoint_sha256": sha256(best),
            "data_yaml": str(data_yaml),
            "run_dir": str(run_dir),
            "training_elapsed_seconds": time.perf_counter() - train_started,
            "validation": validation_payload,
            "final_test": test_payload,
            "test_loss": test_loss,
            "finished_unix": time.time(),
        },
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
