"""Pair/co-enrichment figure factories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biorsp.plotting.styles import DEFAULT_PLOT_STYLE, PlotStyle
from biorsp.plotting.utils import save_figure


def plot_pair_metrics(
    pair_table: pd.DataFrame,
    out_png: Path,
    *,
    x_col: str = "rmsd",
    y_col: str = "angular_bias",
    q_col: str = "q_value",
    title: str = "Pair Metrics",
    style: PlotStyle = DEFAULT_PLOT_STYLE,
) -> None:
    """Render deterministic pair-metric scatter when tabular pair stats are available."""
    required = {x_col, y_col}
    missing = [col for col in required if col not in pair_table.columns]
    if missing:
        raise ValueError(f"Missing required pair metric columns: {missing}")

    x = pd.to_numeric(pair_table[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pair_table[y_col], errors="coerce").to_numpy(dtype=float)

    if q_col in pair_table.columns:
        q = (
            pd.to_numeric(pair_table[q_col], errors="coerce")
            .fillna(1.0)
            .to_numpy(dtype=float)
        )
        colors = np.clip(-np.log10(np.maximum(q, 1e-300)), 0.0, 20.0)
        cmap = "viridis"
    else:
        colors = "#1f77b4"
        cmap = None

    fig, ax = plt.subplots(figsize=style.figsize_qc_split)
    pts = ax.scatter(
        x,
        y,
        c=colors,
        cmap=cmap,
        s=28,
        alpha=0.85,
        linewidths=0.0,
    )
    if q_col in pair_table.columns:
        cbar = fig.colorbar(pts, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("-log10(q)")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, out_png, style=style)
