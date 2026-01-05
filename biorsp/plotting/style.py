from pathlib import Path

import matplotlib.pyplot as plt

COLORS = {
    "bg_cells": "#D3D3D3",
    "fg_cells": "#E31A1C",
    "ref_line": "#636363",
    "highlight": "#1F78B4",
    "core": "#E31A1C",
    "rim": "#33A02C",
    "text": "#000000",
    "grid": "#F0F0F0",
}


def set_publication_style():
    """Sets matplotlib rcParams for Cell Press Patterns standards."""
    plt.rcParams.update(
        {
            # Font sizes
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            # Font family
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,  # Embed fonts
            "ps.fonttype": 42,
            # Lines and Markers
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "axes.linewidth": 0.8,
            # Layout
            "figure.constrained_layout.use": True,
            "savefig.bbox": "tight",
            "savefig.dpi": 600,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(ax, label, x=-0.15, y=1.15):
    """Adds a bold panel label (A, B, C, D) to an axis with increased padding."""
    ax.text(
        x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="right"
    )


def save_figure(fig, name, outdir="figures"):
    """Saves figure as both high-res PNG and vector PDF."""
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)

    pdf_path = path / f"{name}.pdf"
    png_path = path / f"{name}.png"

    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=600)
    print(f"Saved: {pdf_path} and {png_path}")


def get_column_width(type="single"):
    """Returns standard journal column widths in inches."""
    if type == "single":
        return 3.5
    elif type == "double":
        return 7.2
    return 5.0
