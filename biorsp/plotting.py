"""
Plotting module for BioRSP.

Provides visualization functions:
- Radar plots for RSP profiles
- Embedding scatter plots
- Summary visualizations
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .radar import RadarResult
from .summaries import ScalarSummaries


def plot_radar(
    radar: RadarResult,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    color: str = "b",
    alpha: float = 0.5,
    **kwargs,
) -> plt.Axes:
    """
    Plot RSP radar profile.

    Args:
        radar: RadarResult object.
        ax: Matplotlib axes (must be polar). If None, created.
        title: Plot title.
        color: Line/Fill color.
        alpha: Fill transparency.
        **kwargs: Additional arguments for plot.

    Returns:
        ax: The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    theta = radar.centers
    r = radar.rsp

    # Close the loop for plotting
    theta_plot = np.concatenate([theta, [theta[0]]])
    r_plot = np.concatenate([r, [r[0]]])

    # Handle signed values by plotting positive and negative parts separately

    r_pos = np.maximum(0, r_plot)
    r_neg = np.maximum(0, -r_plot)

    # Plot positive lobe
    if np.any(r_pos > 0):
        ax.plot(
            theta_plot,
            r_pos,
            color=color,
            linestyle="-",
            label="Concentrated",
            **kwargs,
        )
        ax.fill(theta_plot, r_pos, color=color, alpha=alpha)

    # Plot negative lobe (dispersed)
    if np.any(r_neg > 0):
        ax.plot(
            theta_plot, r_neg, color="r", linestyle="--", label="Dispersed", **kwargs
        )
        ax.fill(theta_plot, r_neg, color="r", alpha=alpha)

    if title:
        ax.set_title(title)

    return ax


def plot_embedding(
    Z: np.ndarray,
    c: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    s: int = 10,
    **kwargs,
) -> plt.Axes:
    """
    Scatter plot of embedding.

    Args:
        Z: (N, 2) embedding coordinates.
        c: (N,) color values (e.g. expression).
        ax: Matplotlib axes.
        title: Plot title.
        cmap: Colormap.
        s: Marker size.
        **kwargs: Additional arguments for scatter.

    Returns:
        ax: The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots()

    sc = ax.scatter(Z[:, 0], Z[:, 1], c=c, s=s, cmap=cmap, **kwargs)

    if c is not None:
        plt.colorbar(sc, ax=ax)

    if title:
        ax.set_title(title)

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    return ax


def plot_summary(summary: ScalarSummaries, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Display scalar summaries as text.

    Args:
        summary: ScalarSummaries object.
        ax: Matplotlib axes.

    Returns:
        ax: The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.axis("off")

    text = (
        f"Max RSP: {summary.max_rsp:.3f}\n"
        f"Min RSP: {summary.min_rsp:.3f}\n"
        f"Mean Abs RSP: {summary.mean_abs_rsp:.3f}\n"
        f"Integrated RSP: {summary.integrated_rsp:.3f}\n"
        f"Peak Angle: {np.degrees(summary.peak_angle):.1f}°\n"
        f"Trough Angle: {np.degrees(summary.trough_angle):.1f}°"
    )

    ax.text(0.1, 0.5, text, fontsize=12, transform=ax.transAxes, va="center")

    return ax


__all__ = ["plot_radar", "plot_embedding", "plot_summary"]
