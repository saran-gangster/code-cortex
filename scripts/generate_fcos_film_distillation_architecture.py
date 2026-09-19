"""Generate the proposed YOLO26s to FCOS + FiLM distillation architecture."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "yolo26-to-fcos-film-distillation.png"

NAVY = "#10204f"
BLUE = "#159bd3"
BLUE_EDGE = "#0877c9"
BLUE_FILL = "#dff5ff"
GOLD = "#e5a700"
GOLD_FILL = "#fff3bf"
PURPLE = "#7138b2"
PURPLE_FILL = "#eee2ff"
GREEN = "#28a956"
GREEN_FILL = "#e1f7e7"
RED = "#e23b3b"
RED_FILL = "#fff0ec"
STATE = "#16a085"
STATE_FILL = "#dcf8f1"


def box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    lines: list[str],
    edge: str,
    fill: str,
    title_size: float = 13,
    body_size: float = 9.5,
    pill: str | None = None,
    pill_fill: str | None = None,
    pill_size: float = 8.6,
    title_color: str = NAVY,
    body_color: str = NAVY,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.38,rounding_size=0.9",
            facecolor=fill,
            edgecolor=edge,
            linewidth=2.4,
            zorder=3,
        )
    )
    axis.text(
        x + width / 2,
        y + height - 1.45,
        title,
        ha="center",
        va="top",
        color=title_color,
        fontsize=title_size,
        fontweight="bold",
        zorder=4,
    )
    start_y = y + height - 3.7
    if lines:
        axis.text(
            x + width / 2,
            start_y,
            "\n".join(lines),
            ha="center",
            va="top",
            color=body_color,
            fontsize=body_size,
            linespacing=1.28,
            zorder=4,
        )
    if pill:
        pill_height = 1.55
        axis.add_patch(
            FancyBboxPatch(
                (x + 1.0, y + 0.75),
                width - 2.0,
                pill_height,
                boxstyle="round,pad=0.18,rounding_size=0.45",
                facecolor=pill_fill or edge,
                edgecolor="none",
                zorder=4,
            )
        )
        axis.text(
            x + width / 2,
            y + 1.52,
            pill,
            ha="center",
            va="center",
            color=NAVY if pill_fill else "white",
            fontsize=pill_size,
            fontweight="bold",
            zorder=5,
        )


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    width: float = 2.4,
    connection: str = "arc3,rad=0",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=width,
            color=color,
            connectionstyle=connection,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def line(axis: plt.Axes, points: list[tuple[float, float]], color: str, width: float = 2.4) -> None:
    axis.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        color=color,
        linewidth=width,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=1,
    )


def main() -> None:
    figure, axis = plt.subplots(figsize=(20, 11.25), dpi=150)
    figure.patch.set_facecolor("#fbfcfe")
    axis.set_facecolor("#fbfcfe")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 60)
    axis.axis("off")

    axis.text(
        50,
        57.5,
        "AeroGuard YOLO26 → FCOS + FiLM Knowledge Distillation",
        ha="center",
        va="center",
        color=NAVY,
        fontsize=27,
        fontweight="bold",
    )
    axis.text(
        50,
        54.7,
        "PROPOSED CROSS-ARCHITECTURE TRAINING PIPELINE",
        ha="center",
        va="center",
        color=PURPLE,
        fontsize=10.5,
        fontweight="bold",
    )

    box(
        axis,
        1.2,
        39.5,
        14.8,
        9.6,
        title="AU-AIR Dataset",
        lines=["Train: 28,281 images", "Validation: 2,962 images"],
        edge=BLUE_EDGE,
        fill=BLUE_FILL,
        title_size=15,
        body_size=10.5,
    )
    box(
        axis,
        1.2,
        29.4,
        14.8,
        6.2,
        title="Frozen test: 1,580 images",
        lines=["Not used during distillation"],
        edge=RED,
        fill=RED_FILL,
        title_size=11.5,
        body_size=9.3,
        title_color="#961e24",
    )
    box(
        axis,
        1.2,
        16.5,
        14.8,
        9.0,
        title="Flight state",
        lines=["8 normalized values", "altitude · speed · attitude"],
        edge=STATE,
        fill=STATE_FILL,
        title_size=13.5,
        body_size=9.3,
        pill="OPTIONAL · MASKED IF MISSING",
        pill_fill="#8be2ce",
        pill_size=7.0,
    )

    box(
        axis,
        22.0,
        42.0,
        19.0,
        9.8,
        title="Teacher: YOLO26s-960",
        lines=["AeroGuard fine-tuned checkpoint", "9.95M parameters"],
        edge=GOLD,
        fill=GOLD_FILL,
        title_size=15,
        body_size=10,
        pill="FROZEN · EVAL · NO GRADIENTS",
        pill_fill="#f3bd35",
    )
    box(
        axis,
        45.0,
        43.2,
        16.5,
        7.5,
        title="Teacher features",
        lines=["3 neck scales + soft predictions"],
        edge=GOLD,
        fill="#fffaf0",
        title_size=13.2,
        body_size=9.3,
    )

    box(
        axis,
        22.0,
        24.0,
        19.0,
        14.0,
        title="Student: FCOS + gated FiLM",
        lines=[
            "ImageNet ResNet-50 backbone",
            "Feature Pyramid Network P3–P7",
            "Flight-state encoder 8 → 64 → 64",
        ],
        edge=BLUE,
        fill=BLUE_FILL,
        title_size=14.1,
        body_size=9.3,
        pill="TRAINABLE · STATE-AWARE",
    )
    box(
        axis,
        44.0,
        27.2,
        15.5,
        8.4,
        title="Student features",
        lines=["5 FiLM-conditioned FPN levels", "P3 · P4 · P5 · P6 · P7"],
        edge=BLUE,
        fill="#edfaff",
        title_size=12.6,
        body_size=9.1,
    )
    box(
        axis,
        62.0,
        27.2,
        17.5,
        8.4,
        title="Cross-architecture adapters",
        lines=["scale matching + 1×1 conv", "align 5 student levels to teacher"],
        edge=PURPLE,
        fill=PURPLE_FILL,
        title_size=11.8,
        body_size=9,
    )
    box(
        axis,
        66.0,
        42.1,
        15.5,
        7.8,
        title="Distillation loss",
        lines=["score-weighted features", "+ soft class and box targets"],
        edge=PURPLE,
        fill=PURPLE_FILL,
        title_size=13.1,
        body_size=9.1,
    )
    box(
        axis,
        66.0,
        16.8,
        15.5,
        7.6,
        title="Supervised FCOS loss",
        lines=["classification + box regression", "+ centerness"],
        edge=PURPLE,
        fill=PURPLE_FILL,
        title_size=12.3,
        body_size=9,
    )
    box(
        axis,
        85.0,
        29.7,
        13.5,
        10.8,
        title="Total training loss",
        lines=["L = LFCOS + λf Lfeature", "+ λp Lprediction"],
        edge="#56218d",
        fill="#7540aa",
        title_size=12.7,
        body_size=9.6,
        title_color="white",
        body_color="white",
    )
    axis.text(91.75, 32.0, "student + adapters only", ha="center", color="white", fontsize=8.5)

    box(
        axis,
        20.0,
        5.0,
        18.5,
        7.2,
        title="Validation-only selection",
        lines=["Best AP50–95 checkpoint"],
        edge=GREEN,
        fill=GREEN_FILL,
        title_size=12.5,
        body_size=9.5,
    )
    box(
        axis,
        42.0,
        4.4,
        24.0,
        8.4,
        title="AeroGuard FCOS + FiLM student",
        lines=["YOLO knowledge + optional flight state", "No teacher or adapters at inference"],
        edge="#11772d",
        fill="#2f8d43",
        title_size=13.5,
        body_size=9.4,
        title_color="white",
        body_color="white",
    )

    arrow(axis, (16.0, 44.2), (22.0, 47.0), BLUE_EDGE)
    line(axis, [(16.0, 44.2), (18.5, 44.2), (18.5, 31.0)], BLUE_EDGE)
    arrow(axis, (18.5, 31.0), (22.0, 31.0), BLUE_EDGE)
    arrow(axis, (16.0, 21.0), (22.0, 27.5), STATE, connection="arc3,rad=-0.12")

    arrow(axis, (41.0, 47.0), (45.0, 47.0), GOLD)
    line(axis, [(61.5, 47.0), (63.5, 47.0), (63.5, 51.5), (73.8, 51.5)], GOLD)
    arrow(axis, (73.8, 51.5), (73.8, 49.9), GOLD)

    arrow(axis, (41.0, 31.2), (44.0, 31.2), BLUE)
    arrow(axis, (59.5, 31.2), (62.0, 31.2), BLUE)
    arrow(axis, (70.8, 35.6), (70.8, 42.1), PURPLE)
    line(axis, [(41.0, 26.0), (54.0, 26.0), (54.0, 20.6), (66.0, 20.6)], BLUE)
    arrow(axis, (64.0, 20.6), (66.0, 20.6), BLUE)

    line(axis, [(81.5, 46.0), (83.5, 46.0), (83.5, 36.2)], PURPLE)
    arrow(axis, (83.5, 36.2), (85.0, 36.2), PURPLE)
    line(axis, [(81.5, 20.6), (83.5, 20.6), (83.5, 33.2)], PURPLE)
    arrow(axis, (83.5, 33.2), (85.0, 33.2), PURPLE)

    line(axis, [(91.8, 29.7), (91.8, 14.1), (31.5, 14.1)], PURPLE)
    arrow(axis, (31.5, 14.1), (31.5, 24.0), PURPLE)
    arrow(axis, (63.2, 14.1), (63.2, 27.2), PURPLE)
    axis.text(
        61.0,
        14.7,
        "BACKPROPAGATION — TEACHER REMAINS FROZEN",
        ha="center",
        color=PURPLE,
        fontsize=10.2,
        fontweight="bold",
    )

    arrow(axis, (27.8, 24.0), (27.8, 12.2), BLUE_EDGE)
    arrow(axis, (38.5, 8.6), (42.0, 8.6), GREEN)

    axis.text(
        50,
        1.6,
        "Training and model selection use train + validation only  •  frozen test remains outside distillation",
        ha="center",
        va="center",
        color=NAVY,
        fontsize=11,
        fontweight="bold",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
