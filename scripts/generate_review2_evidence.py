"""Build the judge-facing next-review evidence page from machine artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def loss_window(rows: list[dict], *, first: bool, window: int = 200) -> float:
    selected = rows[:window] if first else rows[-window:]
    return fmean(float(row["losses"]["loss"]) for row in selected)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def metric(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=root / "reports" / "review2")
    parser.add_argument("--evaluation-root", type=Path, default=root / "reports" / "evaluations")
    parser.add_argument("--output", type=Path, default=root / "docs" / "REVIEW2_EVIDENCE.md")
    args = parser.parse_args()

    names = ["imagenet-e1-full-pass", "imagenet-e2-full-pass"]
    labels = ["E1: image only", "E2: image + flight state"]
    summaries = [read_json(args.review_root / name / "run_summary.json") for name in names]
    histories = [read_jsonl(args.review_root / name / "loss_history.jsonl") for name in names]
    reports = [
        read_json(args.evaluation_root / f"review2-{name}.json")
        for name in names
    ]
    robustness_reports = [
        read_json(args.evaluation_root / f"review2-{name}.json")
        for name in ("imagenet-e2-masked-fallback", "imagenet-e2-shuffled-state")
    ]

    for name, summary, history, report in zip(names, summaries, histories, reports, strict=True):
        steps = int(summary["completed_steps"])
        if steps != 18_523 or len(history) != steps:
            raise RuntimeError(f"{name}: expected complete 18,523-step history")
        if summary["unique_training_frames_presented"] != 18_523:
            raise RuntimeError(f"{name}: training did not cover every frozen training frame")
        history_path = args.review_root / name / "loss_history.jsonl"
        if sha256_file(history_path) != summary["loss_history_sha256"]:
            raise RuntimeError(f"{name}: loss history hash does not match training summary")
        if report["partition"] != "development" or report["final_test_unsealed"] is not False:
            raise RuntimeError(f"{name}: evaluation is not sealed development evidence")
        if report["checkpoint_id"] != summary["checkpoint_sha256"]:
            raise RuntimeError(f"{name}: evaluation checkpoint does not match training summary")
    for field in (
        "matched_frame_schedule_sha256",
        "shared_warmstart_sha256",
        "protocol_sha256",
        "normalizer_sha256",
    ):
        if len({summary[field] for summary in summaries}) != 1:
            raise RuntimeError(f"matched-arm invariant failed for {field}")
    if {summary["physical_gpu_id"] for summary in summaries} != {"0", "1"}:
        raise RuntimeError("matched arms were not recorded on physical GPUs 0 and 1")
    if any(
        report["partition"] != "development" or report["final_test_unsealed"] is not False
        for report in robustness_reports
    ):
        raise RuntimeError("robustness reports are not sealed development evidence")
    if any(report["checkpoint_id"] != summaries[1]["checkpoint_sha256"] for report in robustness_reports):
        raise RuntimeError("robustness reports do not use the selected E2 checkpoint")

    lines = [
        "# AeroGuard next-review evidence",
        "",
        (
            "This page answers the Review 1 questions with machine-generated evidence. "
            "Both matched model arms now train for one complete pass over the frozen training split, "
            "store every optimizer-step loss, and are compared on the same held-out development flight."
        ),
        "",
        "## Dataset split",
        "",
        "| Split | Recording groups | Frames | Proportion | Use |",
        "|---|---:|---:|---:|---|",
        "| Train | 5 | 18,523 | 56.43% | Weight updates only |",
        "| Development | 1 | 5,734 | 17.47% | Model comparison and threshold selection |",
        "| Sealed final test | 2 | 8,566 | 26.10% | Untouched until the release decision |",
        "| **Total** | **8** | **32,823** | **100.00%** | |",
        "",
        (
            "The split is made by complete flight recording, not by random frames. This prevents almost "
            "identical neighboring video frames from appearing in both training and evaluation."
        ),
        "",
        "![AU-AIR train, development, and sealed test proportions](assets/review2-data-split.png)",
        "",
        "## Full training evidence",
        "",
        "| Arm | Steps | Unique training frames | Coverage | First-200 mean loss | Last-200 mean loss | Time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary, history in zip(labels, summaries, histories, strict=True):
        lines.append(
            f"| {label} | {summary['completed_steps']:,} | "
            f"{summary['unique_training_frames_presented']:,} | "
            f"{percent(summary['training_frame_coverage_fraction'])} | "
            f"{loss_window(history, first=True):.4f} | {loss_window(history, first=False):.4f} | "
            f"{summary['training_seconds'] / 60:.1f} min |"
        )
    lines.extend(
        [
            "",
            (
                "The thin trace below contains every stored loss value. The strong trace is only a "
                "200-step moving average to make the trend readable; no points are hidden from the artifact."
            ),
            "",
            "![Complete E1 and E2 training loss](assets/review2-full-training-loss.png)",
            "",
            "![Classification, box regression, and centerness loss](assets/review2-loss-components.png)",
            "",
            "## Held-out development results",
            "",
            "| Arm | AP50 | AP50:95 | Selected threshold | Precision | Recall | F1 | Detection accuracy | TP | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, report in zip(labels, reports, strict=True):
        pooled = report["metrics"]["pooled"]
        point = report["metrics"]["operating_point_sweep"]["best_f1_point"]
        lines.append(
            f"| {label} | {metric(pooled['ap50'])} | {metric(pooled['ap50_95'])} | "
            f"{point['score_threshold']:.2f} | {metric(point['precision'])} | "
            f"{metric(point['recall'])} | {metric(point['f1'])} | "
            f"{metric(point['detection_accuracy'])} | {point['true_positives']:,} | "
            f"{point['false_positives']:,} | {point['false_negatives']:,} |"
        )
    lines.extend(
        [
            "",
            (
                "**Detection accuracy** here means `TP / (TP + FP + FN)`. It is an intersection-style "
                "object-detection score, not image-classification accuracy. Precision answers how many "
                "shown detections were right. Recall answers how many labelled objects were found. F1 "
                "balances both."
            ),
            "",
            (
                "AP50 measures the precision-recall curve while a predicted box counts as correct at "
                "50% overlap with the labelled box. AP50:95 repeats that calculation at stricter overlap "
                "levels and averages the result."
            ),
            "",
            "![Development AP and fixed-threshold metrics](assets/review2-development-metrics.png)",
            "",
            "![Development threshold sweep](assets/review2-threshold-sweep.png)",
            "",
            "![Per-class AP50 and recall](assets/review2-per-class-metrics.png)",
            "",
            "## What is now implemented",
            "",
            "- Two matched PyTorch Lightning runs execute concurrently on separate Tesla T4 GPUs.",
            "- Each arm sees the same 18,523-frame schedule and starts from the same hashed ImageNet backbone.",
            "- Every training step is persisted for complete loss and component-loss graphs.",
            "- Evaluation reports include AP50, AP50:95, precision, recall, F1, detection accuracy, raw TP/FP/FN, calibration, and latency.",
            "- The confidence threshold is selected on development data only; the final test remains sealed.",
            "- The FastAPI backend exposes health, readiness, capabilities, runs, frames, models, reports, inference, and idempotent human reviews with typed error responses.",
            "",
            "## Honest scope",
            "",
            (
                "These are held-out development results, not final benchmark or safety-certification results. "
                "The final test is still untouched. AeroGuard remains a supervised offline review tool, not "
                "an autonomous flight-control system."
            ),
            "",
        ]
    )
    robustness_lines = [
        "## Flight-state robustness",
        "",
        (
            "The selected E2 checkpoint is also evaluated with flight state masked and with state "
            "shifted by 1,000 frames. Masking measures the image-only fallback. Shifting is a "
            "deliberate misalignment stress test and is never used as a deployment mode."
        ),
        "",
        "| E2 evaluation mode | AP50 | AP50:95 | Best F1 | Detection accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    robustness_rows = [
        ("Correct paired state", reports[1]),
        ("State masked", robustness_reports[0]),
        ("State shifted by 1,000 frames", robustness_reports[1]),
    ]
    for label, report in robustness_rows:
        pooled = report["metrics"]["pooled"]
        point = report["metrics"]["operating_point_sweep"]["best_f1_point"]
        robustness_lines.append(
            f"| {label} | {metric(pooled['ap50'])} | {metric(pooled['ap50_95'])} | "
            f"{metric(point['f1'])} | {metric(point['detection_accuracy'])} |"
        )
    robustness_lines.extend(
        [
            "",
            "![E2 missing-state and misalignment checks](assets/review2-state-robustness.png)",
            "",
            (
                "Correct, masked, and shifted state all leave E2 near AP50 0.114. "
                "The full-pass E2 therefore does not show a reliable telemetry benefit. "
                "The larger gap to E1 points to visual-detector drift during joint training."
            ),
            "",
            "## Follow-up experiment",
            "",
            (
                "E3 and E4 start from the stronger E1 checkpoint, freeze every visual-detector "
                "weight, and train only the residual FiLM state adapter. E4 also masks state on "
                "half of its deterministic training schedule. This design preserves the E1 "
                "image-only fallback exactly while testing whether state can add value."
            ),
            "",
        ]
    )
    implementation_index = lines.index("## What is now implemented")
    lines[implementation_index:implementation_index] = robustness_lines
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
