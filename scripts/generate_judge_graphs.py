"""Generate judge-facing graphs from committed AeroGuard result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"

RUNS = [
    ("Random E1\nimage only", "development_random_e1_summary.json", "random-control-e1.json"),
    ("Random E2\nimage + state", "development_random_e2_summary.json", "random-control-e2.json"),
    ("ImageNet E1\nimage only", "development_imagenet_e1_summary.json", "imagenet-finetune-e1.json"),
    ("ImageNet E2\nimage + state", "development_imagenet_e2_summary.json", "imagenet-finetune-e2.json"),
]

COLORS = ["#64748B", "#22D3EE", "#94A3B8", "#F59E0B"]
BG = "#07111F"
PANEL = "#0F1D2D"
TEXT = "#F8FAFC"
MUTED = "#9FB0C3"
GRID = "#284158"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs() -> list[dict]:
    rows = []
    for label, summary_name, report_name in RUNS:
        summary = read_json(ROOT / "reports" / summary_name)
        report = read_json(ROOT / "reports" / "evaluations" / report_name)
        assert summary["completed_steps"] == summary["requested_steps"] == 5000
        assert report["partition"] == "development"
        assert report["final_test_unsealed"] is False
        rows.append({"label": label, "summary": summary, "report": report})
    return rows


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def label_bars(ax: plt.Axes, bars, fmt: str) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(ax.get_ylim()[1] * 0.018, 0.002),
            format(value, fmt),
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )


def training_summary(rows: list[dict]) -> None:
    labels = [row["label"] for row in rows]
    steps = [row["summary"]["completed_steps"] for row in rows]
    losses = [row["summary"]["final_logged_train_loss"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG)
    fig.suptitle("AeroGuard training completion", color=TEXT, fontsize=24, fontweight="bold", y=0.96)
    fig.text(
        0.5,
        0.905,
        "Four matched PyTorch Lightning runs • two Tesla T4 GPUs • endpoint evidence, not a learning curve",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    bars = axes[0].bar(labels, steps, color=COLORS, width=0.64)
    axes[0].set_title("Completed training updates", fontsize=15, pad=16)
    axes[0].set_ylabel("Optimizer updates")
    axes[0].set_ylim(0, 5700)
    style_axis(axes[0])
    label_bars(axes[0], bars, ".0f")

    bars = axes[1].bar(labels, losses, color=COLORS, width=0.64)
    axes[1].set_title("Final logged training loss", fontsize=15, pad=16)
    axes[1].set_ylabel("Loss (lower is better; endpoint only)")
    axes[1].set_ylim(0, max(losses) * 1.22)
    style_axis(axes[1])
    label_bars(axes[1], bars, ".3f")

    fig.text(
        0.5,
        0.025,
        "Source: committed training summaries • Training loss is optimization evidence, not held-out accuracy",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout(rect=(0.035, 0.07, 0.98, 0.87), w_pad=4)
    fig.savefig(OUT / "training-summary.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def development_metrics(rows: list[dict]) -> None:
    labels = [row["label"] for row in rows]
    pooled = [row["report"]["metrics"]["pooled"] for row in rows]
    reports = [row["report"]["metrics"] for row in rows]
    ap50 = [item["ap50"] for item in pooled]
    ap5095 = [item["ap50_95"] for item in pooled]
    recall = [item["recall"] for item in pooled]
    human = [item["human_recall_at_fixed_operating_point"] for item in reports]

    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG)
    fig.suptitle("AeroGuard development results", color=TEXT, fontsize=24, fontweight="bold", y=0.96)
    fig.text(
        0.5,
        0.905,
        "Single frozen AU-AIR development recording • non-final evidence • final test remains sealed",
        ha="center",
        color="#FBBF24",
        fontsize=11,
        fontweight="bold",
    )

    b1 = axes[0].bar(x - width / 2, ap50, width, label="AP50", color="#22D3EE")
    b2 = axes[0].bar(x + width / 2, ap5095, width, label="AP50:95", color="#F59E0B")
    axes[0].set_title("Detection average precision", fontsize=15, pad=16)
    axes[0].set_ylabel("Average precision")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 0.085)
    style_axis(axes[0])
    label_bars(axes[0], b1, ".3f")
    label_bars(axes[0], b2, ".3f")
    axes[0].legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    b1 = axes[1].bar(x - width / 2, recall, width, label="Overall recall", color="#22D3EE")
    b2 = axes[1].bar(x + width / 2, human, width, label="Human recall", color="#F59E0B")
    axes[1].set_title("Recall at score threshold 0.30", fontsize=15, pad=16)
    axes[1].set_ylabel("Recall")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 0.58)
    style_axis(axes[1])
    label_bars(axes[1], b1, ".3f")
    label_bars(axes[1], b2, ".3f")
    axes[1].legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    fig.text(
        0.5,
        0.025,
        "E2 leads both matched comparisons; absolute accuracy and false-positive rate still require improvement",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout(rect=(0.035, 0.07, 0.98, 0.87), w_pad=4)
    fig.savefig(OUT / "development-metrics.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def calibration_comparison(rows: list[dict]) -> None:
    selected = rows[2:]
    fig, ax = plt.subplots(figsize=(11, 8), facecolor=BG)
    style_axis(ax)
    ax.set_title("ImageNet pair: emitted-detection calibration", fontsize=18, pad=18)
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1.2, label="Ideal calibration")

    for row, color in zip(selected, ["#94A3B8", "#F59E0B"]):
        bins = row["report"]["metrics"]["calibration"]["bins"]
        points = [(b["mean_confidence"], b["empirical_accuracy"]) for b in bins if b["count"]]
        confidence, accuracy = zip(*points)
        ax.plot(confidence, accuracy, marker="o", linewidth=2.4, color=color, label=row["label"].replace("\n", " "))

    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0.28, 1.0)
    ax.set_ylim(0, 1.0)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, loc="upper left")
    fig.text(
        0.5,
        0.04,
        "Development detections only • missed objects are outside this calibration view • threshold is not frozen",
        ha="center",
        color="#FBBF24",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.05, 0.08, 0.98, 0.93))
    fig.savefig(OUT / "calibration-comparison.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_runs()
    training_summary(rows)
    development_metrics(rows)
    calibration_comparison(rows)
    print("Generated training-summary.png, development-metrics.png, calibration-comparison.png")


if __name__ == "__main__":
    main()
