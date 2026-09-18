"""Generate reproducible next-review graphs from frozen AeroGuard artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BG = "#07111F"
PANEL = "#0F1D2D"
TEXT = "#F8FAFC"
MUTED = "#A8B6C8"
GRID = "#29445F"
CYAN = "#22D3EE"
AMBER = "#F59E0B"
GREEN = "#34D399"
PURPLE = "#A78BFA"
RED = "#FB7185"
CLASS_NAMES = ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def moving_average(values: list[float], window: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.size < window:
        return data
    prefix = np.concatenate(([0.0], np.cumsum(data)))
    smooth = (prefix[window:] - prefix[:-window]) / window
    return np.concatenate((np.full(window - 1, np.nan), smooth))


def style_axis(axis: plt.Axes, *, grid_axis: str = "both") -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.title.set_color(TEXT)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.6)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(GRID)


def label_bars(axis: plt.Axes, bars, *, digits: int = 3) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(axis.get_ylim()[1] * 0.018, 0.002),
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )


def generate_split_graph(output: Path) -> None:
    labels = ["Train\n5 recordings", "Development\n1 recording", "Sealed final test\n2 recordings"]
    counts = np.asarray([18_523, 5_734, 8_566])
    percentages = counts / counts.sum() * 100

    figure, axis = plt.subplots(figsize=(12, 7), facecolor=BG)
    bars = axis.bar(labels, percentages, color=[CYAN, AMBER, PURPLE], width=0.62)
    axis.set_title("AU-AIR split by recording, not random frames", fontsize=21, pad=18, fontweight="bold")
    axis.set_ylabel("Percentage of all 32,823 frames")
    axis.set_ylim(0, 66)
    style_axis(axis, grid_axis="y")
    for bar, count, percentage in zip(bars, counts, percentages, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            percentage + 1.2,
            f"{count:,} frames\n{percentage:.2f}%",
            ha="center",
            color=TEXT,
            fontsize=11,
            fontweight="bold",
        )
    figure.text(
        0.5,
        0.035,
        "Frames from the same flight never cross splits, reducing near-duplicate leakage. Final test stays sealed.",
        ha="center",
        color="#FBBF24",
        fontsize=10,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.04, 0.09, 0.98, 0.97))
    figure.savefig(output / "review2-data-split.png", dpi=190, facecolor=BG)
    plt.close(figure)


def _validate_history(run: Path, rows: list[dict]) -> dict:
    summary = read_json(run / "run_summary.json")
    expected = int(summary["completed_steps"])
    steps = [int(row["step"]) for row in rows]
    if len(rows) != expected or steps != list(range(1, expected + 1)):
        raise RuntimeError(f"{run}: loss history is not a complete 1..{expected} sequence")
    if summary["loss_history_rows"] != expected:
        raise RuntimeError(f"{run}: summary does not certify every loss row")
    return summary


def generate_loss_graph(output: Path, run_paths: list[Path]) -> None:
    histories = [read_jsonl(path / "loss_history.jsonl") for path in run_paths]
    summaries = [_validate_history(path, rows) for path, rows in zip(run_paths, histories, strict=True)]
    labels = ["E1 image only", "E2 image + flight state"]
    colors = [CYAN, AMBER]
    window = 200

    figure, axes = plt.subplots(1, 2, figsize=(16, 7.8), facecolor=BG, sharey=True)
    figure.suptitle("Complete full-pass training loss", color=TEXT, fontsize=23, fontweight="bold", y=0.98)
    figure.text(
        0.5,
        0.925,
        "Every optimizer step is stored; the strong line is a 200-step moving average",
        ha="center",
        color=MUTED,
        fontsize=11,
    )
    for axis, rows, summary, label, color in zip(
        axes, histories, summaries, labels, colors, strict=True
    ):
        steps = [int(row["step"]) for row in rows]
        losses = [float(row["losses"]["loss"]) for row in rows]
        axis.plot(steps, losses, color=color, alpha=0.13, linewidth=0.55, label="Every step")
        axis.plot(steps, moving_average(losses, window), color=color, linewidth=2.2, label="200-step mean")
        axis.set_title(label, fontsize=16, pad=14, fontweight="bold")
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Training loss")
        axis.set_xlim(1, int(summary["completed_steps"]))
        style_axis(axis)
        axis.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, loc="upper right")
        axis.text(
            0.03,
            0.05,
            f"{summary['completed_steps']:,} steps | {summary['equivalent_training_epochs']:.2f} pass",
            transform=axis.transAxes,
            color=MUTED,
            fontsize=10,
        )
    figure.text(
        0.5,
        0.025,
        "Training loss shows optimization progress; held-out development metrics are reported separately.",
        ha="center",
        color="#FBBF24",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.035, 0.07, 0.985, 0.89), w_pad=3)
    figure.savefig(output / "review2-full-training-loss.png", dpi=190, facecolor=BG)
    plt.close(figure)


def generate_component_graph(output: Path, run_paths: list[Path]) -> None:
    labels = ["E1 image only", "E2 image + flight state"]
    component_specs = [
        ("classification", "Classification", CYAN),
        ("bbox_regression", "Box regression", AMBER),
        ("bbox_ctrness", "Box centerness", PURPLE),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(16, 7.8), facecolor=BG, sharey=True)
    figure.suptitle("What makes up the training loss", color=TEXT, fontsize=23, fontweight="bold", y=0.97)
    for axis, run, label in zip(axes, run_paths, labels, strict=True):
        rows = read_jsonl(run / "loss_history.jsonl")
        _validate_history(run, rows)
        steps = [int(row["step"]) for row in rows]
        for key, component_label, color in component_specs:
            values = [float(row["losses"][key]) for row in rows]
            axis.plot(steps, moving_average(values, 200), color=color, linewidth=2.0, label=component_label)
        axis.set_title(label, fontsize=16, pad=14, fontweight="bold")
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("200-step mean loss")
        style_axis(axis)
        axis.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    figure.tight_layout(rect=(0.035, 0.06, 0.985, 0.91), w_pad=3)
    figure.savefig(output / "review2-loss-components.png", dpi=190, facecolor=BG)
    plt.close(figure)


def generate_development_graph(output: Path, report_paths: list[Path]) -> None:
    reports = [read_json(path) for path in report_paths]
    for report in reports:
        if report["partition"] != "development" or report["final_test_unsealed"] is not False:
            raise RuntimeError("reports must contain sealed development-only evidence")
    labels = ["E1 image only", "E2 image + state"]
    pooled = [report["metrics"]["pooled"] for report in reports]
    best = [report["metrics"]["operating_point_sweep"]["best_f1_point"] for report in reports]

    figure, axes = plt.subplots(1, 2, figsize=(16, 7.8), facecolor=BG)
    figure.suptitle("Held-out development results", color=TEXT, fontsize=23, fontweight="bold", y=0.98)
    figure.text(
        0.5,
        0.925,
        "One frozen development flight; the 8,566-frame final test remains sealed",
        ha="center",
        color="#FBBF24",
        fontsize=11,
        fontweight="bold",
    )

    x = np.arange(2)
    width = 0.34
    ap50 = [float(item["ap50"]) for item in pooled]
    ap5095 = [float(item["ap50_95"]) for item in pooled]
    bars1 = axes[0].bar(x - width / 2, ap50, width, color=CYAN, label="AP50")
    bars2 = axes[0].bar(x + width / 2, ap5095, width, color=AMBER, label="AP50:95")
    axes[0].set_title("Average precision across confidence levels", fontsize=15, pad=14)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Average precision")
    axes[0].set_ylim(0, max(ap50 + ap5095 + [0.05]) * 1.28)
    style_axis(axes[0], grid_axis="y")
    label_bars(axes[0], bars1, digits=3)
    label_bars(axes[0], bars2, digits=3)
    axes[0].legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    metric_names = ["precision", "recall", "f1", "detection_accuracy"]
    metric_labels = ["Precision", "Recall", "F1", "Detection\naccuracy"]
    metric_colors = [CYAN, AMBER, GREEN, PURPLE]
    group = np.arange(2)
    width = 0.18
    for index, (name, metric_label, color) in enumerate(
        zip(metric_names, metric_labels, metric_colors, strict=True)
    ):
        offset = (index - 1.5) * width
        values = [float(point[name]) for point in best]
        bars = axes[1].bar(group + offset, values, width, color=color, label=metric_label.replace("\n", " "))
        for bar, value in zip(bars, values, strict=True):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.2f}",
                ha="center",
                color=TEXT,
                fontsize=8,
                rotation=90,
            )
    axes[1].set_title("Best development F1 operating point", fontsize=15, pad=14)
    axes[1].set_xticks(group, labels)
    axes[1].set_ylabel("Metric value")
    axes[1].set_ylim(0, 1.08)
    style_axis(axes[1], grid_axis="y")
    axes[1].legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, ncol=2)
    figure.text(
        0.5,
        0.025,
        "Detection accuracy = TP / (TP + FP + FN). It is not image-classification accuracy.",
        ha="center",
        color="#FBBF24",
        fontsize=10,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.035, 0.07, 0.985, 0.89), w_pad=3)
    figure.savefig(output / "review2-development-metrics.png", dpi=190, facecolor=BG)
    plt.close(figure)


def generate_threshold_graph(output: Path, report_paths: list[Path]) -> None:
    labels = ["E1 image only", "E2 image + flight state"]
    colors = [CYAN, AMBER]
    figure, axes = plt.subplots(1, 2, figsize=(16, 7.8), facecolor=BG, sharey=True)
    figure.suptitle("Choosing the confidence threshold on development data", color=TEXT, fontsize=22, fontweight="bold", y=0.98)
    for axis, report_path, label, color in zip(axes, report_paths, labels, colors, strict=True):
        report = read_json(report_path)
        sweep = report["metrics"]["operating_point_sweep"]
        points = sweep["points"]
        thresholds = [float(point["score_threshold"]) for point in points]
        axis.plot(thresholds, [point["precision"] for point in points], color=PURPLE, linewidth=2.0, label="Precision")
        axis.plot(thresholds, [point["recall"] for point in points], color=GREEN, linewidth=2.0, label="Recall")
        axis.plot(thresholds, [point["f1"] for point in points], color=color, linewidth=2.8, label="F1")
        best = sweep["best_f1_point"]
        axis.scatter([best["score_threshold"]], [best["f1"]], color=RED, s=70, zorder=5, label="Selected")
        axis.axvline(float(best["score_threshold"]), color=RED, alpha=0.35, linestyle="--")
        axis.set_title(f"{label}\nselected threshold {best['score_threshold']:.2f}", fontsize=15, pad=12)
        axis.set_xlabel("Confidence threshold")
        axis.set_ylabel("Metric value")
        axis.set_xlim(0.05, 0.95)
        axis.set_ylim(0, 1.02)
        style_axis(axis)
        axis.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, loc="best")
    figure.text(
        0.5,
        0.025,
        "Threshold is selected on development only, then frozen before any final-test evaluation.",
        ha="center",
        color="#FBBF24",
        fontsize=10,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.035, 0.065, 0.985, 0.9), w_pad=3)
    figure.savefig(output / "review2-threshold-sweep.png", dpi=190, facecolor=BG)
    plt.close(figure)


def generate_per_class_graph(output: Path, report_paths: list[Path]) -> None:
    reports = [read_json(path) for path in report_paths]
    per_class = [report["metrics"]["pooled"]["per_class"] for report in reports]
    labels = ["E1 image only", "E2 image + state"]
    colors = [CYAN, AMBER]
    x = np.arange(len(CLASS_NAMES))
    width = 0.36
    supports = [int(per_class[0][str(class_id)]["support"]) for class_id in range(1, 9)]

    figure, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor=BG, sharex=True)
    figure.suptitle("Per-class development results", color=TEXT, fontsize=23, fontweight="bold", y=0.98)
    for axis, metric_name, title in [
        (axes[0], "ap50", "AP50 across confidence levels"),
        (axes[1], "recall", "Recall at the disclosed 0.30 confidence threshold"),
    ]:
        for index, (arm_label, color, values) in enumerate(zip(labels, colors, per_class, strict=True)):
            metric_values = [float(values[str(class_id)][metric_name]) for class_id in range(1, 9)]
            axis.bar(x + (index - 0.5) * width, metric_values, width, color=color, label=arm_label)
        axis.set_title(title, fontsize=15, pad=12)
        axis.set_ylabel(metric_name.upper() if metric_name == "ap50" else "Recall")
        axis.set_ylim(0, 1.02)
        style_axis(axis, grid_axis="y")
        axis.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, ncol=2)
    class_labels = [
        f"{name}\nn={support:,}" for name, support in zip(CLASS_NAMES, supports, strict=True)
    ]
    axes[1].set_xticks(x, class_labels, rotation=20, ha="right")
    figure.text(
        0.5,
        0.025,
        "Rare classes have much less development support; class-level results must be read with their support counts.",
        ha="center",
        color="#FBBF24",
        fontsize=10,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.035, 0.07, 0.985, 0.93), h_pad=2.5)
    figure.savefig(output / "review2-per-class-metrics.png", dpi=190, facecolor=BG)
    plt.close(figure)


def generate_robustness_graph(output: Path, report_paths: list[Path]) -> None:
    reports = [read_json(path) for path in report_paths]
    labels = ["E1\nimage only", "E2\npaired state", "E2\nstate masked", "E2\nstate shifted"]
    colors = [CYAN, AMBER, PURPLE, RED]
    pooled = [report["metrics"]["pooled"] for report in reports]
    best = [report["metrics"]["operating_point_sweep"]["best_f1_point"] for report in reports]

    figure, axes = plt.subplots(1, 2, figsize=(16, 7.8), facecolor=BG)
    figure.suptitle("Flight-state robustness checks", color=TEXT, fontsize=23, fontweight="bold", y=0.98)
    figure.text(
        0.5,
        0.925,
        "Same frozen development flight; E2 uses correct, absent, and deliberately shifted state",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    ap50_values = [float(item["ap50"]) for item in pooled]
    bars = axes[0].bar(labels, ap50_values, color=colors, width=0.65)
    axes[0].set_title("AP50", fontsize=16, pad=14)
    axes[0].set_ylabel("Average precision")
    axes[0].set_ylim(0, max(ap50_values + [0.05]) * 1.28)
    style_axis(axes[0], grid_axis="y")
    label_bars(axes[0], bars, digits=3)

    x = np.arange(len(labels))
    width = 0.34
    f1_bars = axes[1].bar(
        x - width / 2,
        [float(item["f1"]) for item in best],
        width,
        color=GREEN,
        label="Best development F1",
    )
    accuracy_bars = axes[1].bar(
        x + width / 2,
        [float(item["detection_accuracy"]) for item in best],
        width,
        color=PURPLE,
        label="Detection accuracy",
    )
    axes[1].set_title("Selected development operating point", fontsize=16, pad=14)
    axes[1].set_ylabel("Metric value")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1.02)
    style_axis(axes[1], grid_axis="y")
    label_bars(axes[1], f1_bars, digits=3)
    label_bars(axes[1], accuracy_bars, digits=3)
    axes[1].legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    figure.text(
        0.5,
        0.025,
        "Masked state measures the visual fallback. Shifted state is a misalignment stress test, not a deployment mode.",
        ha="center",
        color="#FBBF24",
        fontsize=10,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.035, 0.07, 0.985, 0.89), w_pad=3)
    figure.savefig(output / "review2-state-robustness.png", dpi=190, facecolor=BG)
    plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-run", type=Path, default=root / "reports" / "review2" / "imagenet-e1-full-pass")
    parser.add_argument("--e2-run", type=Path, default=root / "reports" / "review2" / "imagenet-e2-full-pass")
    parser.add_argument("--e1-evaluation", type=Path, default=root / "reports" / "evaluations" / "review2-imagenet-e1-full-pass.json")
    parser.add_argument("--e2-evaluation", type=Path, default=root / "reports" / "evaluations" / "review2-imagenet-e2-full-pass.json")
    parser.add_argument("--e2-masked-evaluation", type=Path, default=root / "reports" / "evaluations" / "review2-imagenet-e2-masked-fallback.json")
    parser.add_argument("--e2-shuffled-evaluation", type=Path, default=root / "reports" / "evaluations" / "review2-imagenet-e2-shuffled-state.json")
    parser.add_argument("--output", type=Path, default=root / "docs" / "assets")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    generate_split_graph(args.output)
    run_paths = [args.e1_run, args.e2_run]
    report_paths = [args.e1_evaluation, args.e2_evaluation]
    generate_loss_graph(args.output, run_paths)
    generate_component_graph(args.output, run_paths)
    generate_development_graph(args.output, report_paths)
    generate_threshold_graph(args.output, report_paths)
    generate_per_class_graph(args.output, report_paths)
    generate_robustness_graph(
        args.output,
        [
            args.e1_evaluation,
            args.e2_evaluation,
            args.e2_masked_evaluation,
            args.e2_shuffled_evaluation,
        ],
    )
    print("Generated Review 2 split, loss, metric, threshold, class, and robustness graphs")


if __name__ == "__main__":
    main()
