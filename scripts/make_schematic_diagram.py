#!/usr/bin/env python3
"""
Generate end-to-end schematic figure.
Provides a one-glance workflow diagram that maps directly to the method text.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from biorsp.plotting.style import COLORS, get_column_width, save_figure, set_publication_style

set_publication_style()


def draw_box(ax, x, y, width, height, text, title="", color="skyblue", alpha=0.3):
    """Draws a styled box with a title and description."""
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.25",
        facecolor=color,
        alpha=alpha,
        edgecolor="black",
        lw=1.2,
    )
    ax.add_patch(box)

    if title:
        ax.text(
            x + width / 2,
            y + height - 0.05,
            title,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=10,
        )

    ax.text(
        x + width / 2,
        y + height / 2 - (0.1 if title else 0),
        text,
        ha="center",
        va="center",
        fontsize=9,
        linespacing=1.4,
    )


def draw_arrow(ax, x1, y1, x2, y2, pad=0.12):
    """Draws a styled arrow between boxes with endpoint padding.

    Parameters
    ----------
    ax : Axes
        The axis to draw on.
    x1, y1, x2, y2 : float
        Start and end coordinates in data units.
    pad : float
        Small padding (in data units) to retract the arrow endpoints so heads
        don't overlap box edges.
    """
    dx = x2 - x1
    dy = y2 - y1
    dist = (dx * dx + dy * dy) ** 0.5
    if dist > 1e-8:
        rx = dx * pad / dist
        ry = dy * pad / dist
        start = (x1 + rx, y1 + ry)
        end = (x2 - rx, y2 - ry)
    else:
        start = (x1, y1)
        end = (x2, y2)

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>,head_width=0.25,head_length=0.5",
        color=COLORS["ref_line"],
        lw=1.2,
        mutation_scale=10,
    )
    ax.add_patch(arrow)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(get_column_width("double"), 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    draw_box(
        ax,
        0.5,
        2.5,
        2,
        1.5,
        "Embedding $X$\nFeature vector $g$",
        "1. Input",
        color=COLORS["bg_cells"],
    )
    draw_arrow(ax, 2.5, 3.25, 3.5, 3.25)

    draw_box(
        ax,
        3.5,
        2.5,
        2,
        1.5,
        "$v = \\mathrm{geom\\_median}(X)$\n$(r, \\theta) = \\mathrm{Polar}(X, v)$",
        "2. Vantage &\nTransform",
        color="#FFF2CC",
    )
    draw_arrow(ax, 5.5, 3.25, 6.5, 3.25)

    draw_box(
        ax,
        6.5,
        2.5,
        2,
        1.5,
        "Define $\\theta$ grid\nWindows $W(\\theta)$",
        "3. Sector Scanning",
        color="#D9EAD3",
    )
    draw_arrow(ax, 8.5, 3.25, 9.5, 3.25)

    draw_box(
        ax,
        9.5,
        2.5,
        2,
        1.5,
        "$R_g(\\theta) = s \\cdot \\frac{W_1(P_{\\mathrm{fg}}, P_{\\mathrm{bg}})}{\\mathrm{scale}}$\n$s = \\mathrm{sign}(\\Delta \\mathrm{median})$",
        "4. RSP Calculation",
        color="#F4CCCC",
    )

    draw_arrow(ax, 10.5, 2.5, 10.5, 1.5)

    draw_box(
        ax,
        9.5,
        0.5,
        2,
        1,
        "$A_g$ (RMS magnitude)\nPeak angle & strength",
        "5. Summaries",
        color="#EAD1DC",
    )

    draw_arrow(ax, 9.5, 1.0, 8.5, 1.0)

    draw_box(
        ax,
        6.5,
        0.5,
        2,
        1,
        "Adequacy mask $\\Theta_g$\nPermutation $p$-value",
        "6. Inference & QC",
        color="#CFE2F3",
    )

    ax.text(
        10.5,
        4.8,
        "Sign Interpretation:\n(+) Core: $r_{\\mathrm{fg}} < r_{\\mathrm{bg}}$\n(-) Rim: $r_{\\mathrm{fg}} > r_{\\mathrm{bg}}$",
        ha="center",
        va="center",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor=COLORS["ref_line"], alpha=0.8
        ),
    )

    ax.text(
        4.5,
        1.0,
        "Pooled scale normalization:\n$\\hat{r} = (r - \\mathrm{median}) / \\mathrm{IQR}$",
        ha="center",
        va="center",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor=COLORS["ref_line"], alpha=0.8
        ),
    )

    save_figure(fig, "fig_schematic_diagram", outdir=outdir)


if __name__ == "__main__":
    main()
