"""Failure-safe evaluator and graph generator for the final YOLO benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

CLASS_NAMES = ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def metric_payload(metrics: Any) -> dict[str, Any]:
    box = metrics.box
    payload: dict[str, Any] = {"results_dict": dict(metrics.results_dict), "speed_ms": dict(metrics.speed)}
    class_ids = list(getattr(box, "ap_class_index", range(len(CLASS_NAMES))))
    per_class = {}
    for result_index, class_id in enumerate(class_ids):
        try:
            precision, recall, ap50, ap5095 = box.class_result(result_index)
            class_f1 = 2.0 * float(precision) * float(recall) / max(float(precision) + float(recall), 1e-12)
            supports = np.asarray(getattr(box, "nt_per_class", []), dtype=np.int64)
            per_class[CLASS_NAMES[int(class_id)]] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": class_f1,
                "ap50": float(ap50),
                "ap50_95": float(ap5095),
                "support": int(supports[result_index]) if result_index < len(supports) else None,
            }
        except Exception:  # noqa: BLE001,S112 - tolerate absent classes across releases
            continue
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
    return payload


def raw_counts(metrics: Any) -> dict[str, Any]:
    matrix = np.asarray(metrics.confusion_matrix.matrix, dtype=np.float64)
    nc = len(CLASS_NAMES)
    if matrix.shape != (nc + 1, nc + 1):
        return {"matrix": matrix.tolist(), "error": f"unexpected shape {matrix.shape}"}
    tp = float(np.trace(matrix[:nc, :nc]))
    fp = float(matrix[:nc, nc].sum())
    fn = float(matrix[nc, :nc].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "true_positives": int(tp), "false_positives": int(fp), "false_negatives": int(fn),
        "precision": precision, "recall": recall, "f1": f1, "detection_accuracy": accuracy,
        "matrix_predicted_rows_true_columns": matrix.tolist(),
    }


def find_checkpoint(output: Path) -> tuple[Path, Path]:
    candidates = sorted(output.glob("runs/*/weights/best.pt"))
    if not candidates:
        candidates = sorted(output.glob("runs/*/weights/last.pt"))
    if not candidates:
        raise FileNotFoundError("no trained checkpoint found")
    checkpoint = candidates[-1]
    return checkpoint, checkpoint.parents[1]


def compute_test_loss(
    checkpoint: Path,
    data: Path,
    output: Path,
    imgsz: int,
    batch_size: int,
    device: str,
    workers: int,
) -> dict[str, Any]:
    """Compute detector loss on the true final-test loader, not the cached validation loader."""
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    trainer = DetectionTrainer(
        overrides={
            "model": str(checkpoint),
            "data": str(data),
            "imgsz": imgsz,
            "batch": batch_size,
            "device": device,
            "workers": workers,
            "project": str(output / "loss-evaluation"),
            "name": "final-test-loss",
            "exist_ok": True,
            "plots": False,
            "save": False,
            "val": False,
            "multi_scale": 0.0,
        }
    )
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    trainer.model.eval()
    loader = trainer.get_dataloader(
        trainer.data["test"], batch_size=batch_size, rank=-1, mode="val"
    )
    totals: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for raw_batch in loader:
            prepared = trainer.preprocess_batch(raw_batch)
            with torch.amp.autocast(
                device_type=trainer.device.type,
                enabled=trainer.device.type == "cuda",
            ):
                predictions = trainer.model(prepared["img"])
                _, loss_items = trainer.model.loss(prepared, predictions)
            for name, value in loss_items.items():
                detached = value.detach().float()
                totals[name] = totals.get(name, torch.zeros_like(detached)) + detached
    if not totals or not len(loader):
        raise RuntimeError("final-test loss loader produced no batches")
    result: dict[str, Any] = {
        f"test/{name}": float(value.cpu() / len(loader)) for name, value in totals.items()
    }
    result["test_loss_total"] = sum(
        value for key, value in result.items() if key.startswith("test/") and key.endswith("_loss")
    )
    result.update(
        {
            "partition": "test",
            "batches": len(loader),
            "images": len(loader.dataset),
            "data_yaml": str(data),
            "checkpoint": str(checkpoint),
            "mixed_precision": trainer.device.type == "cuda",
        }
    )
    return result


def curves(run_dir: Path, output: Path, test_loss: dict[str, Any]) -> None:
    csv_path = run_dir / "results.csv"
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("empty training results.csv")

    names = [name.strip() for name in rows[0]]
    normalized_rows = [{key.strip(): value for key, value in row.items()} for row in rows]

    def column(name: str) -> np.ndarray:
        return np.asarray([float(row[name].strip()) for row in normalized_rows], dtype=np.float64)

    train_loss_names = [name for name in names if name.startswith("train/") and name.endswith("_loss")]
    val_loss_names = [name for name in names if name.startswith("val/") and name.endswith("_loss")]
    epochs = np.arange(1, len(rows) + 1)
    train_loss = sum((column(name) for name in train_loss_names), np.zeros(len(rows)))
    val_loss = sum((column(name) for name in val_loss_names), np.zeros(len(rows)))
    precision = column("metrics/precision(B)")
    recall = column("metrics/recall(B)")
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    detection_accuracy = precision * recall / np.maximum(
        precision + recall - precision * recall, 1e-12
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].plot(epochs, train_loss, marker="o", label="Training loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Validation loss")
    test_terms = [
        float(value)
        for key, value in test_loss.items()
        if key.startswith(("test/", "val/"))
        and key.endswith("_loss")
        and isinstance(value, (int, float))
    ]
    if test_terms:
        axes[0].axhline(sum(test_terms), color="black", linestyle="--", label="Final test loss")
    axes[0].set(title="Training, validation and final-test loss", xlabel="Epoch", ylabel="Total detector loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for name, label in (("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall"), ("metrics/mAP50(B)", "AP50"), ("metrics/mAP50-95(B)", "AP50-95")):
        axes[1].plot(epochs, column(name), marker="o", label=label)
    axes[1].plot(epochs, f1, marker="o", label="F1")
    axes[1].plot(epochs, detection_accuracy, marker="o", label="Detection accuracy")
    axes[1].set(title="Validation metrics by epoch", xlabel="Epoch", ylabel="Metric", ylim=(0, 1))
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2)
    figure.suptitle("AeroGuard final pretrained-detector evidence", fontweight="bold")
    figure.savefig(output / "judge_training_and_metrics.png", dpi=180)
    plt.close(figure)

    loss_figure, loss_axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for name in train_loss_names:
        loss_axes[0].plot(epochs, column(name), marker="o", label=name.removeprefix("train/"))
    loss_axes[0].set(title="Training loss components", xlabel="Epoch", ylabel="Loss")
    loss_axes[0].grid(alpha=0.25)
    loss_axes[0].legend()
    for name in val_loss_names:
        component = name.removeprefix("val/")
        loss_axes[1].plot(epochs, column(name), marker="o", label=f"validation {component}")
        test_value = test_loss.get(name.replace("val/", "test/"), test_loss.get(name))
        if isinstance(test_value, (int, float)):
            loss_axes[1].axhline(
                float(test_value), linestyle="--", alpha=0.75, label=f"final test {component}"
            )
    loss_axes[1].set(title="Validation and final-test loss components", xlabel="Epoch", ylabel="Loss")
    loss_axes[1].grid(alpha=0.25)
    loss_axes[1].legend(ncol=2)
    loss_figure.suptitle("AeroGuard detector loss record", fontweight="bold")
    loss_figure.savefig(output / "judge_loss_components.png", dpi=180)
    plt.close(loss_figure)

    write_json(
        output / "epoch_history.json",
        {
            "epochs": epochs.tolist(), "train_loss_columns": train_loss_names,
            "validation_loss_columns": val_loss_names, "training_total_loss": train_loss.tolist(),
            "validation_total_loss": val_loss.tolist(), "precision": precision.tolist(),
            "recall": recall.tolist(), "f1": f1.tolist(),
            "detection_accuracy": detection_accuracy.tolist(),
            "ap50": column("metrics/mAP50(B)").tolist(),
            "ap50_95": column("metrics/mAP50-95(B)").tolist(),
        },
    )
    history_fields = [
        "epoch",
        *train_loss_names,
        *val_loss_names,
        "training_total_loss",
        "validation_total_loss",
        "precision",
        "recall",
        "f1",
        "detection_accuracy",
        "ap50",
        "ap50_95",
    ]
    with (output / "epoch_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()
        ap50 = column("metrics/mAP50(B)")
        ap50_95 = column("metrics/mAP50-95(B)")
        for index, epoch in enumerate(epochs):
            record = {
                "epoch": int(epoch),
                "training_total_loss": float(train_loss[index]),
                "validation_total_loss": float(val_loss[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "detection_accuracy": float(detection_accuracy[index]),
                "ap50": float(ap50[index]),
                "ap50_95": float(ap50_95[index]),
            }
            record.update({name: float(column(name)[index]) for name in train_loss_names})
            record.update({name: float(column(name)[index]) for name in val_loss_names})
            writer.writerow(record)


def write_partition_summary(output: Path, validation: dict[str, Any], final_test: dict[str, Any]) -> None:
    """Write the conventional judge-facing metrics as a two-row CSV table."""
    fields = [
        "partition", "ap50", "ap50_95", "precision", "recall", "f1",
        "detection_accuracy", "true_positives", "false_positives", "false_negatives",
        "operating_confidence",
    ]
    with (output / "partition_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for partition, payload in (("validation", validation), ("final_test", final_test)):
            results = payload.get("results_dict", {})
            counts = payload.get("operating_point_counts", {})
            writer.writerow(
                {
                    "partition": partition,
                    "ap50": results.get("metrics/mAP50(B)"),
                    "ap50_95": results.get("metrics/mAP50-95(B)"),
                    "precision": counts.get("precision", results.get("metrics/precision(B)")),
                    "recall": counts.get("recall", results.get("metrics/recall(B)")),
                    "f1": counts.get("f1", payload.get("best_mean_f1")),
                    "detection_accuracy": counts.get("detection_accuracy"),
                    "true_positives": counts.get("true_positives"),
                    "false_positives": counts.get("false_positives"),
                    "false_negatives": counts.get("false_negatives"),
                    "operating_confidence": payload.get(
                        "selected_confidence", payload.get("frozen_validation_confidence")
                    ),
                }
            )


def main(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    checkpoint, run_dir = find_checkpoint(args.output)
    model = YOLO(str(checkpoint))
    common = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": 0,
        "workers": args.workers,
        "iou": 0.6,
    }
    validation = model.val(
        split="val", conf=0.001, plots=True, save_json=True,
        project=str(args.output / "evaluation"), name="validation-ap-finalizer", **common,
    )
    validation_payload = metric_payload(validation)
    threshold = float(validation_payload.get("best_f1_confidence", 0.25))
    validation_op = model.val(
        split="val", conf=threshold, plots=True,
        project=str(args.output / "evaluation"), name="validation-operating-finalizer", **common,
    )
    validation_payload["selected_confidence"] = threshold
    validation_payload["operating_point_counts"] = raw_counts(validation_op)
    write_json(args.output / "validation_metrics_final.json", validation_payload)

    final_test = model.val(
        split="test", conf=0.001, plots=True, save_json=True,
        project=str(args.output / "evaluation"), name="final-test-ap-finalizer", **common,
    )
    final_payload = metric_payload(final_test)
    final_op = model.val(
        split="test", conf=threshold, plots=True,
        project=str(args.output / "evaluation"), name="final-test-operating-finalizer", **common,
    )
    final_payload["frozen_validation_confidence"] = threshold
    final_payload["operating_point_counts"] = raw_counts(final_op)
    write_json(args.output / "final_test_metrics_final.json", final_payload)

    del model, validation, validation_op, final_test, final_op
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    test_loss = compute_test_loss(
        checkpoint, args.data, args.output, args.imgsz, args.batch, args.device, args.workers
    )
    write_json(args.output / "final_test_loss.json", test_loss)
    curves(run_dir, args.output, test_loss)
    write_partition_summary(args.output, validation_payload, final_payload)
    write_json(
        args.output / "final_summary.json",
        {
            "checkpoint": str(checkpoint), "run_dir": str(run_dir), "validation": validation_payload,
            "final_test": final_payload, "test_loss": test_loss, "finished_unix": time.time(),
        },
    )


if __name__ == "__main__":
    main(parse_args())
