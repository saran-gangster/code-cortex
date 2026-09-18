"""Generate judge-facing evidence for the frozen-visual state adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean

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


def moving_average(values: list[float], window: int = 200) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    prefix = np.concatenate(([0.0], np.cumsum(data)))
    smooth = (prefix[window:] - prefix[:-window]) / window
    return np.concatenate((np.full(window - 1, np.nan), smooth))


def style_axis(axis: plt.Axes, *, grid_axis: str = "both") -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    axis.title.set_color(TEXT)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.6)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(GRID)


def label_bars(axis: plt.Axes, bars) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    review_root = root / "reports" / "review2"
    evaluation_root = root / "reports" / "evaluations"
    output = root / "docs" / "assets"
    output.mkdir(parents=True, exist_ok=True)

    adapter_names = (
        "imagenet-e3-exact-frozen-visual-film",
        "imagenet-e4-exact-frozen-visual-film-dropout50",
    )
    summaries = [read_json(review_root / name / "run_summary.json") for name in adapter_names]
    histories = [read_jsonl(review_root / name / "loss_history.jsonl") for name in adapter_names]
    adapter_reports = [
        read_json(evaluation_root / f"review2-{name}.json") for name in adapter_names
    ]
    base_names = ("imagenet-e2-full-pass", "imagenet-e1-full-pass")
    base_reports = [
        read_json(evaluation_root / f"review2-{name}.json") for name in base_names
    ]
    base_summary = read_json(review_root / base_names[0] / "run_summary.json")
    warmstart = read_json(review_root / "e2-frozen-visual-warmstart-summary.json")
    integrity = read_json(review_root / "adapter_integrity_summary.json")

    for name, summary, history, report in zip(
        adapter_names, summaries, histories, adapter_reports, strict=True
    ):
        history_path = review_root / name / "loss_history.jsonl"
        if summary["completed_steps"] != 18_523 or len(history) != 18_523:
            raise RuntimeError(f"{name}: incomplete adapter training history")
        if sha256_file(history_path) != summary["loss_history_sha256"]:
            raise RuntimeError(f"{name}: loss history hash mismatch")
        if not summary["visual_detector_frozen"]:
            raise RuntimeError(f"{name}: visual detector was not frozen")
        if report["checkpoint_id"] != summary["checkpoint_sha256"]:
            raise RuntimeError(f"{name}: evaluation checkpoint mismatch")
        if report["partition"] != "development" or report["final_test_unsealed"]:
            raise RuntimeError(f"{name}: report is not sealed development evidence")
    if not integrity["all_visual_tensors_exactly_equal"]:
        raise RuntimeError("adapter visual tensors differ from the E2 parent")
    if warmstart["source_checkpoint_sha256"] != base_summary["checkpoint_sha256"]:
        raise RuntimeError("adapter warmstart does not trace back to the E2 checkpoint")
    for name in adapter_names:
        integrity_run = integrity["runs"][name]
        if not integrity_run["visual_tensors_exactly_equal"]:
            raise RuntimeError(f"{name}: visual tensors changed")
        if integrity_run["changed_film_tensor_count"] <= 0:
            raise RuntimeError(f"{name}: FiLM adapter did not change")
        if (
            integrity_run["visual_state_sha256"]
            != integrity["parent_visual_state_sha256"]
        ):
            raise RuntimeError(f"{name}: visual state hash differs from the parent")

    labels = [
        "E2\nimage + state",
        "E1\nimage only",
        "E3\nexact frozen visual",
        "E4\nexact frozen + dropout",
    ]
    reports = [*base_reports, *adapter_reports]
    colors = [CYAN, AMBER, GREEN, PURPLE]
    ap50 = [float(report["metrics"]["pooled"]["ap50"]) for report in reports]
    best = [report["metrics"]["operating_point_sweep"]["best_f1_point"] for report in reports]

    figure, axes = plt.subplots(1, 2, figsize=(16, 7.8), facecolor=BG)
    figure.suptitle(
        "Frozen-visual state-adapter follow-up",
        color=TEXT,
        fontsize=23,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.925,
        "E3 and E4 preserve every E2 visual tensor exactly and train only the FiLM adapter",
        ha="center",
        color=MUTED,
        fontsize=11,
    )
    bars = axes[0].bar(labels, ap50, color=colors, width=0.65)
    axes[0].set_title("AP50 on the held-out development flight", fontsize=15, pad=14)
    axes[0].set_ylabel("Average precision")
    axes[0].set_ylim(0, max(ap50 + [0.05]) * 1.3)
    style_axis(axes[0], grid_axis="y")
    label_bars(axes[0], bars)

    x = np.arange(len(labels))
    width = 0.34
    f1_bars = axes[1].bar(
        x - width / 2,
        [float(point["f1"]) for point in best],
        width,
        color=GREEN,
        label="Best F1",
    )
    accuracy_bars = axes[1].bar(
        x + width / 2,
        [float(point["detection_accuracy"]) for point in best],
        width,
        color=PURPLE,
        label="Detection accuracy",
    )
    axes[1].set_title("Best development operating point", fontsize=15, pad=14)
    axes[1].set_ylabel("Metric value")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1.02)
    style_axis(axes[1], grid_axis="y")
    label_bars(axes[1], f1_bars)
    label_bars(axes[1], accuracy_bars)
    axes[1].legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    figure.text(
        0.5,
        0.025,
        "Model selection uses development only. The 8,566-frame final test remains sealed.",
        ha="center",
        color="#FBBF24",
        fontsize=10,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.035, 0.07, 0.985, 0.89), w_pad=3)
    figure.savefig(output / "review2-adapter-followup.png", dpi=190, facecolor=BG)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(16, 7.8), facecolor=BG, sharey=True)
    figure.suptitle(
        "Complete adapter training loss",
        color=TEXT,
        fontsize=23,
        fontweight="bold",
        y=0.98,
    )
    adapter_labels = [
        "E3 exact frozen visual",
        "E4 exact frozen visual + 50% state dropout",
    ]
    for axis, rows, label, color in zip(
        axes, histories, adapter_labels, [GREEN, PURPLE], strict=True
    ):
        steps = [int(row["step"]) for row in rows]
        losses = [float(row["losses"]["loss"]) for row in rows]
        axis.plot(steps, losses, color=color, alpha=0.13, linewidth=0.55, label="Every step")
        axis.plot(steps, moving_average(losses), color=color, linewidth=2.2, label="200-step mean")
        axis.set_title(label, fontsize=15, pad=14, fontweight="bold")
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Training loss")
        axis.set_xlim(1, 18_523)
        style_axis(axis)
        axis.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    figure.tight_layout(rect=(0.035, 0.06, 0.985, 0.91), w_pad=3)
    figure.savefig(output / "review2-adapter-loss.png", dpi=190, facecolor=BG)
    plt.close(figure)

    result_rows = []
    for label, report in zip(labels, reports, strict=True):
        pooled = report["metrics"]["pooled"]
        point = report["metrics"]["operating_point_sweep"]["best_f1_point"]
        result_rows.append(
            f"| {label.replace(chr(10), ' ')} | {pooled['ap50']:.4f} | "
            f"{pooled['ap50_95']:.4f} | {point['score_threshold']:.2f} | "
            f"{point['precision']:.4f} | {point['recall']:.4f} | "
            f"{point['f1']:.4f} | {point['detection_accuracy']:.4f} |"
        )
    training_rows = []
    for label, summary, history in zip(adapter_labels, summaries, histories, strict=True):
        first = fmean(float(row["losses"]["loss"]) for row in history[:200])
        last = fmean(float(row["losses"]["loss"]) for row in history[-200:])
        training_rows.append(
            f"| {label} | {summary['completed_steps']:,} | "
            f"{summary['effective_state_mask_mean']:.3f} | {first:.4f} | "
            f"{last:.4f} | {summary['training_seconds'] / 60:.1f} min |"
        )
    best_index = int(np.argmax(ap50))
    conclusion = (
        f"The highest development AP50 remains {labels[best_index].replace(chr(10), ' ')} "
        f"at {ap50[best_index]:.4f}."
    )
    document = [
        "# AeroGuard frozen-visual adapter follow-up",
        "",
        (
            "This experiment tests controlled variants of the selected E2 image-plus-state model. E3 and E4 start "
            "from the stronger E2 checkpoint. Every visual-detector tensor is frozen and "
            "verified byte-for-byte after training; only the residual FiLM adapter changes."
        ),
        "",
        "## Adapter training",
        "",
        "| Arm | Steps | Mean active-state mask | First-200 loss | Last-200 loss | Time |",
        "|---|---:|---:|---:|---:|---:|",
        *training_rows,
        "",
        "![Complete frozen-visual adapter loss](assets/review2-adapter-loss.png)",
        "",
        "## Held-out development comparison",
        "",
        "| Model | AP50 | AP50:95 | Threshold | Precision | Recall | F1 | Detection accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *result_rows,
        "",
        "![Frozen-visual adapter follow-up](assets/review2-adapter-followup.png)",
        "",
        f"**Decision:** {conclusion}",
        "",
        (
            "Detection accuracy is `TP / (TP + FP + FN)`. These are development-only "
            "results. The final test remains sealed."
        ),
        "",
    ]
    (root / "docs" / "REVIEW2_ADAPTER_FOLLOWUP.md").write_text(
        "\n".join(document), encoding="utf-8"
    )
    print("Generated adapter loss, comparison graph, and follow-up evidence page")


if __name__ == "__main__":
    main()
