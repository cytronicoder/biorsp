"""Shared plotting style settings for deterministic figure outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PlotStyle:
    """Centralized plotting defaults used across pipeline figures."""

    dpi: int = 200
    figsize_umap: tuple[float, float] = (6.0, 5.0)
    figsize_categorical: tuple[float, float] = (8.0, 6.0)
    figsize_qc_split: tuple[float, float] = (7.0, 6.0)
    figsize_centroid: tuple[float, float] = (7.0, 5.0)
    figsize_classification: tuple[float, float] = (7.5, 6.2)
    figsize_class_counts: tuple[float, float] = (8.2, 4.2)
    s_bg: float = 5.0
    s_fg: float = 7.0
    alpha_bg: float = 0.30
    alpha_fg: float = 0.85
    vantage_marker: str = "X"
    vantage_size: float = 110.0
    legend_fontsize: int = 8
    axis_label_fontsize: int = 10
    title_fontsize: int = 11
    colorbar_shrink: float = 0.8
    colorbar_pad: float = 0.02
    categorical_legend_trigger: int = 25
    categorical_legend_top_k: int = 20
    categorical_annotation_max: int = 40
    annotation_min_dist_frac: float = 0.025


DEFAULT_PLOT_STYLE = PlotStyle()


def apply_plot_style(style: PlotStyle = DEFAULT_PLOT_STYLE) -> None:
    """Apply deterministic matplotlib rcParams for pipeline plots."""
    plt.rcParams.update(
        {
            "figure.dpi": style.dpi,
            "savefig.dpi": style.dpi,
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titlesize": style.title_fontsize,
            "axes.labelsize": style.axis_label_fontsize,
            "legend.fontsize": style.legend_fontsize,
            "axes.grid": False,
        }
    )


def plot_style_dict(style: PlotStyle = DEFAULT_PLOT_STYLE) -> dict[str, Any]:
    """Return style + dependency versions for metadata manifests."""
    d = asdict(style)
    d["matplotlib_version"] = str(matplotlib.__version__)
    d["numpy_version"] = str(np.__version__)
    return d
