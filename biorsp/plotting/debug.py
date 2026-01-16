"""
Debug Plotting Utilities for BioRSP.

Provides standardized debug visualizations to help validate:
- Foreground/background masking
- Sector validity and coverage
- Radar profile components
- Cutoff consistency

These plots are saved under debug/ subdirectory when --debug flag is enabled.
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from biorsp.plotting.spec import PlotSpec
from biorsp.plotting.style import COLORS

logger = logging.getLogger(__name__)


def plot_debug_pointcloud(
    coords: np.ndarray,
    vantage: np.ndarray,
    title: str = "Point Cloud with Vantage",
    figsize: Tuple[float, float] = (6, 6),
) -> Figure:
    """
    Plot raw point cloud with vantage point marked.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (N, 2)
    vantage : np.ndarray
        Vantage point coordinates (2,)
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(coords[:, 0], coords[:, 1], c=COLORS["bg_cells"], s=2, alpha=0.3, label="Cells")
    ax.scatter(
        vantage[0], vantage[1], c="red", marker="x", s=200, linewidths=3, label="Vantage", zorder=10
    )

    ax.set_xlabel("Embedding Dim 1")
    ax.set_ylabel("Embedding Dim 2")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_debug_foreground_mask(
    coords: np.ndarray,
    fg_mask: np.ndarray,
    vantage: np.ndarray,
    title: str = "Foreground Mask",
    figsize: Tuple[float, float] = (6, 6),
) -> Figure:
    """
    Visualize foreground/background assignment on embedding.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (N, 2)
    fg_mask : np.ndarray
        Boolean mask indicating foreground cells
    vantage : np.ndarray
        Vantage point
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    bg_mask = ~fg_mask
    ax.scatter(
        coords[bg_mask, 0],
        coords[bg_mask, 1],
        c=COLORS["bg_cells"],
        s=3,
        alpha=0.4,
        label=f"Background (n={bg_mask.sum()})",
    )
    ax.scatter(
        coords[fg_mask, 0],
        coords[fg_mask, 1],
        c=COLORS["fg_cells"],
        s=10,
        alpha=0.7,
        label=f"Foreground (n={fg_mask.sum()})",
    )
    ax.scatter(vantage[0], vantage[1], c="red", marker="x", s=200, linewidths=3, label="Vantage")

    ax.set_xlabel("Embedding Dim 1")
    ax.set_ylabel("Embedding Dim 2")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_debug_sector_counts(
    theta_centers: np.ndarray,
    counts_fg: np.ndarray,
    counts_bg: np.ndarray,
    valid_mask: np.ndarray,
    title: str = "Sector Counts and Validity",
    figsize: Tuple[float, float] = (10, 8),
) -> Figure:
    """
    Show foreground/background counts per sector and validity mask.

    Parameters
    ----------
    theta_centers : np.ndarray
        Angular centers of sectors (radians)
    counts_fg : np.ndarray
        Foreground cell counts per sector
    counts_bg : np.ndarray
        Background cell counts per sector
    valid_mask : np.ndarray
        Boolean mask indicating valid sectors
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    theta_deg = np.degrees(theta_centers)

    ax = axes[0]
    ax.plot(theta_deg, counts_fg, "o-", color=COLORS["fg_cells"], label="nF (Foreground)")
    ax.fill_between(theta_deg, 0, counts_fg, color=COLORS["fg_cells"], alpha=0.3)
    ax.set_ylabel("nF", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(theta_deg, counts_bg, "o-", color=COLORS["bg_cells"], label="nB (Background)")
    ax.fill_between(theta_deg, 0, counts_bg, color=COLORS["bg_cells"], alpha=0.3)
    ax.set_ylabel("nB", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(theta_deg, valid_mask.astype(int), "r-", drawstyle="steps-mid", linewidth=2)
    ax.fill_between(theta_deg, 0, valid_mask.astype(int), color="red", alpha=0.2, step="mid")
    ax.set_ylabel("Valid (1/0)", fontsize=10)
    ax.set_xlabel("Angle θ (degrees)", fontsize=10)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_debug_cutoff_consistency(
    df,
    spec: PlotSpec,
    title: str = "Cutoff Consistency Check",
    figsize: Tuple[float, float] = (8, 8),
) -> Figure:
    """
    Verify that quadrant colors match classification cutoffs.

    This plot explicitly shows the cutoff lines and archetype regions,
    annotated with the classification logic to ensure no mismatch.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Coverage, Spatial_Bias_Score, Archetype columns
    spec : PlotSpec
        Plot specification with cutoffs
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if spec.archetype_col not in df.columns:
        df = spec.classify_dataframe(df, inplace=False)

    c_cut, s_cut = spec.get_quadrant_bounds()
    c_col = spec.coverage_col
    s_col = spec.spatial_col

    xlim = (0, df[c_col].max() * 1.1)
    ylim = (0, df[s_col].max() * 1.15)

    ax.fill_between([xlim[0], c_cut], ylim[0], s_cut, color=spec.get_color("Basal"), alpha=0.15)

    ax.fill_between(
        [c_cut, xlim[1]], ylim[0], s_cut, color=spec.get_color("Ubiquitous"), alpha=0.15
    )

    ax.fill_between([xlim[0], c_cut], s_cut, ylim[1], color=spec.get_color("Patchy"), alpha=0.15)

    ax.fill_between([c_cut, xlim[1]], s_cut, ylim[1], color=spec.get_color("Gradient"), alpha=0.15)

    ax.axvline(
        c_cut,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"C cutoff = {c_cut:.2f}",
    )
    ax.axhline(
        s_cut,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"S cutoff = {s_cut:.2f}",
    )

    for archetype in spec.get_legend_order():
        mask = df[spec.archetype_col] == archetype
        if not mask.any():
            continue
        color = spec.get_color(archetype)
        ax.scatter(
            df.loc[mask, c_col],
            df.loc[mask, s_col],
            c=color,
            label=archetype,
            s=50,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Coverage (C)", fontsize=11)
    ax.set_ylabel("Spatial Bias Score (S)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout()
    return fig


def save_debug_plots(
    coords: np.ndarray,
    vantage: np.ndarray,
    fg_mask: np.ndarray,
    theta_centers: np.ndarray,
    counts_fg: np.ndarray,
    counts_bg: np.ndarray,
    valid_mask: np.ndarray,
    df,
    spec: PlotSpec,
    outdir: Path,
    case_name: str = "debug",
):
    """
    Generate and save all standard debug plots.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates
    vantage : np.ndarray
        Vantage point
    fg_mask : np.ndarray
        Foreground mask
    theta_centers : np.ndarray
        Sector angular centers
    counts_fg : np.ndarray
        Foreground counts per sector
    counts_bg : np.ndarray
        Background counts per sector
    valid_mask : np.ndarray
        Valid sector mask
    df : pd.DataFrame
        Results DataFrame
    spec : PlotSpec
        Plot specification
    outdir : Path
        Output directory
    case_name : str
        Identifier for this case/gene
    """
    outdir = Path(outdir)
    debug_dir = outdir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating debug plots for {case_name}...")

    fig = plot_debug_pointcloud(coords, vantage, title=f"Point Cloud: {case_name}")
    fig.savefig(debug_dir / f"debug_pointcloud_{case_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_debug_foreground_mask(
        coords, fg_mask, vantage, title=f"Foreground Mask: {case_name}"
    )
    fig.savefig(debug_dir / f"debug_foreground_mask_{case_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_debug_sector_counts(
        theta_centers, counts_fg, counts_bg, valid_mask, title=f"Sector Counts: {case_name}"
    )
    fig.savefig(debug_dir / f"debug_sector_counts_{case_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if df is not None and len(df) > 0:
        fig = plot_debug_cutoff_consistency(df, spec, title=f"Cutoff Consistency: {case_name}")
        fig.savefig(
            debug_dir / f"debug_cutoff_consistency_{case_name}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    logger.info(f"Debug plots saved to {debug_dir}")
