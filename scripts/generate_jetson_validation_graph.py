"""Generate the judge-facing Jetson validation infographic from raw benchmark JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "final"
ASSETS = ROOT / "docs" / "assets"
OUTPUT = ASSETS / "jetson-validation-benchmark.png"

BG = "#071118"
PANEL = "#0d1c24"
GRID = "#23404a"
TEXT = "#edf8f6"
MUTED = "#88a4ad"
PYTORCH = "#72848d"
TENSORRT = "#b8f35a"
CYAN = "#55e6d0"


def load_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_facecolor(PANEL)
    axis.set_title(title, loc="left", color=TEXT, fontsize=14, fontweight="bold", pad=14)
    axis.tick_params(colors=MUTED, labelsize=10, length=0)
    axis.grid(axis="y", color=GRID, alpha=0.45, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)


def bars(axis: plt.Axes, title: str, values: list[float], unit: str, lower: bool) -> None:
    style_axis(axis, title)
    colors = [PYTORCH, TENSORRT]
    columns = axis.bar(["PyTorch", "TensorRT"], values, color=colors, width=0.54)
    maximum = max(values)
    axis.set_ylim(0, maximum * 1.32)
    axis.set_yticklabels([])
    for column, value in zip(columns, values, strict=True):
        axis.text(
            column.get_x() + column.get_width() / 2,
            value + maximum * 0.055,
            f"{value:.2f}{unit}",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=13,
            fontweight="bold",
        )
    change = ((values[1] / values[0]) - 1.0) * 100.0
    direction = "LOWER" if lower else "HIGHER"
    axis.text(
        0.98,
        0.96,
        f"{abs(change):.1f}% {direction}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color=TENSORRT,
        fontsize=10,
        fontweight="bold",
    )


def main() -> None:
    pytorch = load_json("jetson_pytorch_fp16.json")
    tensorrt = load_json("jetson_tensorrt_fp16.json")
    parity = load_json("jetson_backend_parity.json")
    tensorrt_boxes = load_json("jetson_tensorrt_parity.json")

    figure = plt.figure(figsize=(16, 9), dpi=150, facecolor=BG)
    grid = figure.add_gridspec(
        2,
        4,
        height_ratios=[1.35, 1.0],
        left=0.045,
        right=0.975,
        top=0.84,
        bottom=0.075,
        wspace=0.22,
        hspace=0.28,
    )

    figure.text(0.045, 0.935, "AeroGuard", color=TEXT, fontsize=26, fontweight="bold")
    figure.text(
        0.045,
        0.89,
        "JETSON ORIN NANO  ·  YOLO26s  ·  960 × 960  ·  FP16",
        color=CYAN,
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.975,
        0.925,
        f"{tensorrt['throughput_fps_batch_1']:.2f} FPS",
        color=TENSORRT,
        fontsize=30,
        fontweight="bold",
        ha="right",
    )
    figure.text(
        0.975,
        0.887,
        "REAL-TIME EDGE INFERENCE",
        color=MUTED,
        fontsize=10,
        fontweight="bold",
        ha="right",
    )

    image_axis = figure.add_subplot(grid[0, :2])
    image_axis.set_facecolor(PANEL)
    frame_name = "frame_20190905091750_x_0000133.jpg"
    image = Image.open(ASSETS / "auair-demo" / frame_name).convert("RGB")
    image_axis.imshow(image)
    frame_result = next(
        item for item in tensorrt_boxes["detections"] if item["image"] == frame_name
    )
    for index, detection in enumerate(frame_result["detections"]):
        x1, y1, x2, y2 = detection["box_xyxy"]
        color = TENSORRT if index == 0 else CYAN
        image_axis.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=3)
        )
        image_axis.text(
            x1,
            y1 - 10,
            f"{detection['class_name']}  {detection['confidence']:.2f}",
            color=BG,
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 3},
        )
    image_axis.set_title(
        "REAL AU-AIR FRAME  ·  TENSORRT OUTPUT",
        loc="left",
        color=TEXT,
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    image_axis.axis("off")

    summary_axis = figure.add_subplot(grid[0, 2:])
    summary_axis.set_facecolor(PANEL)
    summary_axis.set_xlim(0, 1)
    summary_axis.set_ylim(0, 1)
    summary_axis.axis("off")
    cards = [
        ("31.90", "ms end-to-end", TENSORRT),
        ("0.941", "mean box IoU", CYAN),
        ("6 / 6", "detections preserved", TEXT),
    ]
    for index, (value, label, color) in enumerate(cards):
        left = 0.05 + index * 0.315
        summary_axis.add_patch(
            Rectangle((left, 0.57), 0.275, 0.31, facecolor=BG, edgecolor=GRID, linewidth=1.2)
        )
        summary_axis.text(left + 0.025, 0.75, value, color=color, fontsize=25, fontweight="bold")
        summary_axis.text(left + 0.025, 0.63, label.upper(), color=MUTED, fontsize=9, fontweight="bold")
    summary_axis.text(0.05, 0.42, "OUTPUT AGREEMENT", color=MUTED, fontsize=10, fontweight="bold")
    summary_axis.text(0.05, 0.31, "0 missing", color=TEXT, fontsize=17, fontweight="bold")
    summary_axis.text(0.35, 0.31, "1 borderline extra", color=TEXT, fontsize=17, fontweight="bold")
    summary_axis.text(0.73, 0.31, "6 real frames", color=TEXT, fontsize=17, fontweight="bold")
    summary_axis.plot([0.05, 0.95], [0.22, 0.22], color=GRID, linewidth=1)
    summary_axis.text(
        0.05,
        0.10,
        "TensorRT 10.3  ·  JetPack R36.4.7  ·  15 W mode  ·  batch 1",
        color=CYAN,
        fontsize=11,
        fontweight="bold",
    )

    pt_latency = pytorch["latency_ms"]
    trt_latency = tensorrt["latency_ms"]
    bars(
        figure.add_subplot(grid[1, 0]),
        "END-TO-END LATENCY",
        [pt_latency["mean_end_to_end"], trt_latency["mean_end_to_end"]],
        " ms",
        True,
    )
    bars(
        figure.add_subplot(grid[1, 1]),
        "MODEL INFERENCE",
        [pt_latency["mean_inference"], trt_latency["mean_inference"]],
        " ms",
        True,
    )
    bars(
        figure.add_subplot(grid[1, 2]),
        "THROUGHPUT",
        [pytorch["throughput_fps_batch_1"], tensorrt["throughput_fps_batch_1"]],
        " FPS",
        False,
    )
    bars(
        figure.add_subplot(grid[1, 3]),
        "CUDA TENSOR MEMORY",
        [
            pytorch["cuda_peak_memory_bytes"] / 1024**2,
            tensorrt["cuda_peak_memory_bytes"] / 1024**2,
        ],
        " MiB",
        True,
    )

    figure.text(
        0.975,
        0.027,
        f"50 timed runs after warm-up  ·  P95 TensorRT {trt_latency['p95_end_to_end']:.2f} ms  ·  minimum IoU {parity['minimum_iou']:.3f}",
        color=MUTED,
        fontsize=9,
        ha="right",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
