"""
Visualization utilities for Biological Radar Scanning Plot (BioRSP) results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, List, Tuple, Union
from .radar_scan import FeatureResult


def plot_rsp_heatmap(
    result: FeatureResult,
    ax: Optional[Axes] = None,
    cmap: str = "RdBu_r",
    show_peak: bool = True,
    show_colorbar: bool = True,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Axes:
    """
    Plot a single RSP heatmap in polar coordinates.

    Args:
        result: FeatureResult object containing Z_heat data
        ax: Matplotlib polar axes. If None, creates new figure
        cmap: Colormap name
        show_peak: Whether to mark the peak direction
        show_colorbar: Whether to show colorbar
        title: Custom title. If None, uses result name and statistics
        vmin: Minimum value for colormap. If None, uses symmetric range
        vmax: Maximum value for colormap. If None, uses symmetric range

    Returns:
        Matplotlib axes object
    """
    if result.Z_heat is None:
        raise ValueError("FeatureResult has no Z_heat data to plot")

    # Create axes if not provided
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar")

    # Get heatmap dimensions
    J, B = result.Z_heat.shape  # J = number of bands/widths, B = angular bins

    # Create angular and radial coordinates for pcolormesh edges
    # We need B+1 theta values and J+1 radial values for edges
    theta = np.linspace(0, 2 * np.pi, B + 1)  # B+1 for edges
    r = np.arange(J + 1)  # J+1 for pcolormesh edges

    # Create meshgrid
    theta_grid, r_grid = np.meshgrid(theta, r)

    # Set color limits
    if vmin is None or vmax is None:
        max_abs = np.abs(result.Z_heat).max()
        vmin = -max_abs if vmin is None else vmin
        vmax = max_abs if vmax is None else vmax

    # Plot heatmap
    im = ax.pcolormesh(
        theta_grid,
        r_grid,
        result.Z_heat,
        cmap=cmap,
        shading="flat",
        vmin=vmin,
        vmax=vmax,
    )

    # Mark peak direction
    if show_peak and result.phi_star is not None:
        peak_angle = result.phi_star
        ax.plot(
            [peak_angle, peak_angle],
            [0, J],
            "g-",
            linewidth=2,
            alpha=0.7,
            label="Peak direction",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0), fontsize=8)

    # Set title
    if title is None:
        title = f"{result.name or 'Feature'}\n"
        title += f"Z={result.Z_max:.2f}, p={result.p_value:.4f}\n"
        title += f"Peak: {np.degrees(result.phi_star):.1f}°"

    ax.set_title(title, fontsize=10, pad=15)

    # Add colorbar
    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Z-score", pad=0.1, fraction=0.046)

    # Set radial labels to show band numbers
    ax.set_yticks(np.arange(J) + 0.5)
    ax.set_yticklabels([f"Band {i}" for i in range(J)], fontsize=8)

    return ax


def plot_rsp_grid(
    results: List[FeatureResult],
    ncols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "RdBu_r",
    show_peaks: bool = True,
    suptitle: Optional[str] = None,
    sort_by: Optional[str] = None,
    max_plots: Optional[int] = None,
) -> Figure:
    """
    Plot multiple RSP heatmaps in a grid.

    Args:
        results: List of FeatureResult objects
        ncols: Number of columns in grid
        figsize: Figure size. If None, auto-calculated
        cmap: Colormap name
        show_peaks: Whether to mark peak directions
        suptitle: Overall figure title
        sort_by: Sort results by 'p_value', 'Z_max', or None
        max_plots: Maximum number of plots to show

    Returns:
        Matplotlib Figure object
    """
    # Filter results with heatmap data
    results = [r for r in results if r.Z_heat is not None]

    if len(results) == 0:
        raise ValueError("No results with Z_heat data to plot")

    # Sort if requested
    if sort_by == "p_value":
        results = sorted(results, key=lambda x: x.p_value)
    elif sort_by == "Z_max":
        results = sorted(results, key=lambda x: -x.Z_max)

    # Limit number of plots
    if max_plots is not None:
        results = results[:max_plots]

    # Calculate grid dimensions
    nplots = len(results)
    nrows = int(np.ceil(nplots / ncols))

    # Set figure size
    if figsize is None:
        figsize = (5 * ncols, 5 * nrows)

    # Create figure
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, subplot_kw=dict(projection="polar")
    )

    # Flatten axes array
    if nplots == 1:
        axes = np.array([axes])
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    # Plot each result
    for i, result in enumerate(results):
        plot_rsp_heatmap(
            result, ax=axes[i], cmap=cmap, show_peak=show_peaks, show_colorbar=True
        )

    # Hide unused axes
    for i in range(nplots, len(axes)):
        axes[i].axis("off")

    # Add overall title
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.995)

    plt.tight_layout()

    return fig


def plot_rsp_summary(
    result: FeatureResult,
    coords: np.ndarray,
    feature_values: np.ndarray,
    figsize: Tuple[float, float] = (15, 5),
) -> Figure:
    """
    Create a comprehensive summary plot with embedding, feature distribution, and RSP heatmap.

    Args:
        result: FeatureResult object
        coords: Cell coordinates (N, 2)
        feature_values: Feature values for each cell (N,)
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)

    # 1. Embedding colored by feature
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(
        coords[:, 0],
        coords[:, 1],
        c=feature_values,
        cmap="viridis",
        s=1,
        alpha=0.5,
        rasterized=True,
    )
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_title(f'{result.name or "Feature"}\nEmbedding')
    plt.colorbar(scatter, ax=ax1, label="Feature value")
    ax1.set_aspect("equal")

    # 2. Feature distribution
    ax2 = fig.add_subplot(132)
    ax2.hist(feature_values, bins=50, alpha=0.7, edgecolor="black")
    ax2.axvline(
        np.mean(feature_values),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(feature_values):.3f}",
    )
    ax2.set_xlabel("Feature value")
    ax2.set_ylabel("Count")
    ax2.set_title("Feature Distribution")
    ax2.legend()

    # 3. RSP heatmap
    ax3 = fig.add_subplot(133, projection="polar")
    plot_rsp_heatmap(result, ax=ax3, show_colorbar=True)

    plt.tight_layout()

    return fig


def save_top_results(
    results: List[FeatureResult],
    output_dir: str,
    top_n: int = 10,
    sort_by: str = "p_value",
    prefix: str = "rsp",
    dpi: int = 150,
) -> List[str]:
    """
    Save plots for top N results.

    Args:
        results: List of FeatureResult objects
        output_dir: Directory to save plots
        top_n: Number of top results to save
        sort_by: Sort by 'p_value' or 'Z_max'
        prefix: Filename prefix
        dpi: Resolution for saved images

    Returns:
        List of saved file paths
    """
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Sort results
    if sort_by == "p_value":
        results = sorted(results, key=lambda x: x.p_value)
    elif sort_by == "Z_max":
        results = sorted(results, key=lambda x: -x.Z_max)

    # Take top N
    results = results[:top_n]

    saved_paths = []

    for i, result in enumerate(results):
        if result.Z_heat is None:
            continue

        # Create individual plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar")
        plot_rsp_heatmap(result, ax=ax, show_colorbar=True, show_peak=True)

        # Create safe filename
        safe_name = (
            str(result.name or f"feature_{i}").replace("/", "_").replace("=", "_")
        )
        filename = f"{prefix}_{i+1:02d}_{safe_name}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close()

        saved_paths.append(filepath)

    return saved_paths
