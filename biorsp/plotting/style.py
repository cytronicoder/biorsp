from contextlib import contextmanager
from pathlib import Path
from typing import Optional

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


@contextmanager
def publication_style():
    """Context manager for publication-quality matplotlib styling.

    Temporarily applies Cell Press Patterns standards, then restores
    previous rcParams state. Use this instead of set_publication_style()
    to avoid global state mutations.

    Usage:
        with publication_style():
            fig, ax = plt.subplots()
            # ... plotting code ...
            save_figure(fig, "output.pdf")
    """
    original_params = dict(plt.rcParams)
    try:
        plt.rcParams.update(
            {
                "font.size": 9,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 8,
                "figure.titlesize": 12,
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "lines.linewidth": 1.5,
                "lines.markersize": 4,
                "axes.linewidth": 0.8,
                "figure.constrained_layout.use": True,
                "savefig.bbox": "tight",
                "savefig.dpi": 600,
                "savefig.transparent": False,
                "savefig.facecolor": "white",
            }
        )
        yield
    finally:
        plt.rcParams.update(original_params)


def set_publication_style():
    """Sets matplotlib rcParams for Cell Press Patterns standards.

    DEPRECATED: Use publication_style() context manager instead to avoid
    global state mutations.
    """
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "axes.linewidth": 0.8,
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


def save_figure(
    fig,
    path: str,
    *,
    dpi: int = 300,
    transparent: bool = False,
    bbox: str = "tight",
    pad: float = 0.02,
    close: bool = True,
    formats: Optional[list] = None,
) -> list:
    """Save figure with standardized output settings.

    Ensures consistent high-quality output and avoids clipped labels by using
    bbox_inches='tight' with appropriate padding. Supports multiple formats.

    Args:
        fig: Matplotlib figure object to save.
        path: Output path (with or without extension).
        dpi: Resolution for raster formats (default: 300).
        transparent: Whether to use transparent background (default: False).
        bbox: Bounding box mode - "tight" (default) or "standard".
        pad: Padding around tight bbox in inches (default: 0.02).
        close: Whether to close figure after saving (default: True).
        formats: List of formats to save (default: ["pdf", "png"]).
                 If path has extension, that format is used instead.

    Returns:
        List of saved file paths.

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> save_figure(fig, "output/figure1.pdf")  # Saves PDF only
        >>> save_figure(fig, "output/figure2", formats=["pdf", "png"])  # Both formats
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if formats is None:
        if path_obj.suffix:
            formats = [path_obj.suffix.lstrip(".")]
            base_path = path_obj.with_suffix("")
        else:
            formats = ["pdf", "png"]
            base_path = path_obj
    else:
        base_path = path_obj.with_suffix("") if path_obj.suffix else path_obj

    saved_paths = []
    bbox_kwargs = {"bbox_inches": bbox, "pad_inches": pad} if bbox == "tight" else {}

    for fmt in formats:
        out_path = base_path.with_suffix(f".{fmt}")
        fig.savefig(
            out_path,
            format=fmt,
            dpi=dpi,
            transparent=transparent,
            facecolor=None if transparent else "white",
            **bbox_kwargs,
        )
        saved_paths.append(out_path)
        print(f"Saved: {out_path}")

    if close:
        plt.close(fig)

    return saved_paths


def get_column_width(type="single"):
    """Returns standard journal column widths in inches."""
    if type == "single":
        return 3.5
    elif type == "double":
        return 7.2
    return 5.0
