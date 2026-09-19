"""Combine finalized detector candidates into judge-facing JSON, CSV, and graphs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("reports/final/benchmarks"))
    parser.add_argument("--output", type=Path, default=Path("reports/final"))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def number(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def extract(candidate: Path) -> dict[str, Any]:
    summary = read_json(candidate / "final_summary.json")
    request_path = candidate / "run_request.json"
    history_path = candidate / "epoch_history.json"
    request = read_json(request_path) if request_path.exists() else {}
    history = read_json(history_path) if history_path.exists() else {}
    validation = summary["validation"]
    final_test = summary["final_test"]
    validation_results = validation.get("results_dict", {})
    test_results = final_test.get("results_dict", {})
    validation_counts = validation.get("operating_point_counts", {})
    test_counts = final_test.get("operating_point_counts", {})
    test_loss = summary.get("test_loss", {})
    training_loss = history.get("training_total_loss", [])
    validation_loss = history.get("validation_total_loss", [])
    return {
        "candidate": candidate.name,
        "model": request.get("model", candidate.name),
        "image_size": request.get("imgsz"),
        "epochs_completed": len(history.get("epochs", [])),
        "checkpoint": summary.get("checkpoint"),
        "validation": {
            "ap50": number(validation_results, "metrics/mAP50(B)"),
            "ap50_95": number(validation_results, "metrics/mAP50-95(B)"),
            **{key: validation_counts.get(key) for key in (
                "precision", "recall", "f1", "detection_accuracy",
                "true_positives", "false_positives", "false_negatives",
            )},
        },
        "final_test": {
            "ap50": number(test_results, "metrics/mAP50(B)"),
            "ap50_95": number(test_results, "metrics/mAP50-95(B)"),
            **{key: test_counts.get(key) for key in (
                "precision", "recall", "f1", "detection_accuracy",
                "true_positives", "false_positives", "false_negatives",
            )},
            "loss": test_loss,
        },
        "loss_curve": {
            "training_initial": training_loss[0] if training_loss else None,
            "training_final": training_loss[-1] if training_loss else None,
            "validation_initial": validation_loss[0] if validation_loss else None,
            "validation_final": validation_loss[-1] if validation_loss else None,
            "final_test": test_loss.get("test_loss_total"),
        },
    }


def write_csv(path: Path, candidates: list[dict[str, Any]], winner: str) -> None:
    fields = [
        "candidate", "selected", "model", "image_size", "epochs_completed",
        "validation_ap50", "validation_ap50_95", "validation_precision",
        "validation_recall", "validation_f1", "validation_detection_accuracy",
        "test_ap50", "test_ap50_95", "test_precision", "test_recall", "test_f1",
        "test_detection_accuracy", "test_loss_total",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            validation = item["validation"]
            test = item["final_test"]
            writer.writerow(
                {
                    "candidate": item["candidate"],
                    "selected": item["candidate"] == winner,
                    "model": item["model"],
                    "image_size": item["image_size"],
                    "epochs_completed": item["epochs_completed"],
                    **{f"validation_{key}": validation.get(key) for key in (
                        "ap50", "ap50_95", "precision", "recall", "f1", "detection_accuracy",
                    )},
                    **{f"test_{key}": test.get(key) for key in (
                        "ap50", "ap50_95", "precision", "recall", "f1", "detection_accuracy",
                    )},
                    "test_loss_total": test.get("loss", {}).get("test_loss_total"),
                }
            )


def plot(path: Path, candidates: list[dict[str, Any]]) -> None:
    labels = [item["candidate"] for item in candidates]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.18
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for offset, partition, metric, label in (
        (-1.5, "validation", "ap50", "Validation AP50"),
        (-0.5, "validation", "ap50_95", "Validation AP50-95"),
        (0.5, "final_test", "ap50", "Final-test AP50"),
        (1.5, "final_test", "ap50_95", "Final-test AP50-95"),
    ):
        axes[0].bar(
            x + offset * width,
            [item[partition].get(metric) or 0.0 for item in candidates],
            width,
            label=label,
        )
    axes[0].set(title="Validation-selected detector comparison", ylabel="Average precision")
    axes[0].set_xticks(x, labels, rotation=10)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    metric_names = ("precision", "recall", "f1", "detection_accuracy")
    metric_labels = ("Precision", "Recall", "F1", "Detection accuracy")
    metric_x = np.arange(len(metric_names), dtype=np.float64)
    candidate_width = 0.8 / max(len(candidates), 1)
    for index, item in enumerate(candidates):
        axes[1].bar(
            metric_x - 0.4 + candidate_width / 2 + index * candidate_width,
            [item["final_test"].get(name) or 0.0 for name in metric_names],
            candidate_width,
            label=item["candidate"],
        )
    axes[1].set(title="Final-test fixed operating point", ylabel="Metric", ylim=(0, 1))
    axes[1].set_xticks(metric_x, metric_labels, rotation=10)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("AeroGuard final detector benchmark", fontweight="bold")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidate_dirs = sorted(path.parent for path in args.root.glob("*/final_summary.json"))
    if not candidate_dirs:
        raise FileNotFoundError(f"no final_summary.json files below {args.root}")
    candidates = [extract(path) for path in candidate_dirs]
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["validation"].get("ap50_95") or -1.0,
            item["validation"].get("ap50") or -1.0,
        ),
        reverse=True,
    )
    winner = ranked[0]["candidate"]
    payload = {
        "selection_partition": "validation",
        "selection_rule": "highest AP50-95, then AP50",
        "selected_candidate": winner,
        "candidates": candidates,
    }
    (args.output / "candidate_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "candidate_comparison.csv", candidates, winner)
    plot(args.output / "candidate_comparison.png", candidates)
    print(f"selected {winner} from {len(candidates)} candidates")


if __name__ == "__main__":
    main()
