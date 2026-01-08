"""
DEPRECATED: Thin wrapper around simlib.io for legacy figure saving.

This module provides backward-compatible utilities for figure styling and saving.
New code should import directly from simlib.io and simlib.plotting.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Path bootstrap
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Standard colors (preserved for compatibility)
COLOR_BG = "#e0e0e0"  # Light gray for background
COLOR_FG = "#d62728"  # Red for foreground
COLOR_RSP = "#1f77b4"  # Blue for RSP profile


def setup_style():
    """Set up global matplotlib style for publication-ready figures."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "lines.linewidth": 2,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, output_dir: str, filename: str):
    """Save figure as PNG (wrapper around simlib.io.save_figure)."""
    from simlib import io

    output_path = Path(output_dir)
    io.save_figure(fig, output_path, filename, dpi=300)


def clean_axis(ax: plt.Axes):
    """Remove ticks and set axis equal."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
