"""
Plotting utilities for simulation benchmarks.

Provides matplotlib figure generation for QQ plots, power curves, confusion matrices, PR curves.
"""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_qq(
    expected: np.ndarray,
    observed: np.ndarray,
    title: str = "QQ Plot",
    alpha: float = 0.05,
    figsize: Tuple[float, float] = (5, 5),
) -> Figure:
    """
    Create QQ plot for calibration assessment.

    Parameters
    ----------
    expected : np.ndarray
        Expected quantiles (uniform)
    observed : np.ndarray
        Observed p-value quantiles
    title : str, optional
        Plot title
    alpha : float, optional
        Significance threshold (for visual aid)
    figsize : tuple, optional
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    if len(expected) == 0 or len(observed) == 0:
        logger.warning(
            f"Empty data for QQ plot: expected={len(expected)}, observed={len(observed)}"
        )
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No Data for QQ Plot", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    # Plot diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    # Plot alpha threshold
    ax.axvline(alpha, color="red", linestyle=":", alpha=0.3, label=f"α={alpha}")
    ax.axhline(alpha, color="red", linestyle=":", alpha=0.3)

    # Plot observed vs expected
    ax.scatter(expected, observed, s=10, alpha=0.6)

    ax.set_xlabel("Expected (Uniform)")
    ax.set_ylabel("Observed (P-value)")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_fpr_grid(
    fpr_df: pd.DataFrame,
    row_var: str,
    col_var: str,
    title: str = "False Positive Rate",
    alpha: float = 0.05,
    figsize: Tuple[float, float] = (12, 8),
) -> Figure:
    """
    Create heatmap of FPR across parameter grid.

    Parameters
    ----------
    fpr_df : pd.DataFrame
        DataFrame with FPR values (columns: row_var, col_var, fpr, ci_low, ci_high)
    row_var : str
        Variable for rows
    col_var : str
        Variable for columns
    title : str, optional
        Plot title
    alpha : float, optional
        Expected FPR (for colorbar reference)
    figsize : tuple, optional
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    if fpr_df.empty:
        logger.warning(f"Empty DataFrame for FPR Grid: {title}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No Data for FPR Grid", ha="center", va="center")
        return fig

    pivot = fpr_df.pivot(index=row_var, columns=col_var, values="fpr")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.15)

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.1 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_xlabel(col_var)
    ax.set_ylabel(row_var)
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("FPR")
    cbar.ax.axhline(alpha, color="red", linewidth=2, label=f"α={alpha}")

    plt.tight_layout()
    return fig


def plot_power_curve(
    power_df: pd.DataFrame,
    x_var: str,
    hue_var: Optional[str] = None,
    title: str = "Power Curve",
    figsize: Tuple[float, float] = (8, 6),
) -> Figure:
    """
    Create power curve with confidence intervals.

    Parameters
    ----------
    power_df : pd.DataFrame
        DataFrame with power values (columns: x_var, power, ci_low, ci_high, hue_var)
    x_var : str
        Variable for x-axis
    hue_var : str, optional
        Variable for grouping
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    if power_df.empty:
        logger.warning(f"Empty DataFrame for Power Curve: {title}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No Data for Power Curve", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    if hue_var is None:
        x = power_df[x_var]
        y = power_df["power"]
        ci_low = power_df["ci_low"]
        ci_high = power_df["ci_high"]

        ax.plot(x, y, marker="o", linewidth=2)
        ax.fill_between(x, ci_low, ci_high, alpha=0.3)
    else:
        for group, group_df in power_df.groupby(hue_var):
            x = group_df[x_var]
            y = group_df["power"]
            ci_low = group_df["ci_low"]
            ci_high = group_df["ci_high"]

            ax.plot(x, y, marker="o", linewidth=2, label=str(group))
            ax.fill_between(x, ci_low, ci_high, alpha=0.2)

        ax.legend()

    ax.set_xlabel(x_var)
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="80% power")

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    title: str = "Confusion Matrix",
    figsize: Tuple[float, float] = (8, 6),
    normalize: bool = False,
) -> Figure:
    """
    Plot confusion matrix as heatmap.

    Parameters
    ----------
    cm_df : pd.DataFrame
        Confusion matrix
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    normalize : bool, optional
        Normalize by row (true label)

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    if cm_df.empty:
        logger.warning(f"Empty Confusion Matrix: {title}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Empty Confusion Matrix", ha="center", va="center")
        return fig

    if normalize:
        cm_plot = cm_df.div(cm_df.sum(axis=1), axis=0)
        cmap = "Blues"
    else:
        cm_plot = cm_df
        cmap = "Blues"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, cmap=cmap, aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(cm_plot.columns)))
    ax.set_yticks(np.arange(len(cm_plot.index)))
    ax.set_xticklabels(cm_plot.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm_plot.index)

    # Annotate cells
    for i in range(len(cm_plot.index)):
        for j in range(len(cm_plot.columns)):
            val = cm_plot.iloc[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=10)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    auprc: float,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[float, float] = (6, 6),
) -> Figure:
    """
    Plot precision-recall curve.

    Parameters
    ----------
    precision : np.ndarray
        Precision values
    recall : np.ndarray
        Recall values
    auprc : float
        Area under PR curve
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, linewidth=2, label=f"AUPRC = {auprc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_robustness_delta(
    delta_df: pd.DataFrame,
    x_var: str,
    y_var: str = "median_abs_delta",
    hue_var: Optional[str] = None,
    title: str = "Robustness: Score Stability",
    figsize: Tuple[float, float] = (8, 6),
) -> Figure:
    """
    Plot robustness (median absolute delta vs perturbation strength).

    Parameters
    ----------
    delta_df : pd.DataFrame
        DataFrame with robustness metrics
    x_var : str
        Variable for x-axis (e.g., 'strength')
    y_var : str, optional
        Variable for y-axis
    hue_var : str, optional
        Variable for grouping
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    if delta_df.empty:
        logger.warning(f"Empty DataFrame for Robustness Plot: {title}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No Data for Robustness", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    if hue_var is None:
        x = delta_df[x_var]
        y = delta_df[y_var]
        ax.plot(x, y, marker="o", linewidth=2)
    else:
        for group, group_df in delta_df.groupby(hue_var):
            x = group_df[x_var]
            y = group_df[y_var]
            ax.plot(x, y, marker="o", linewidth=2, label=str(group))

        ax.legend()

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(title)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spatial_embedding(
    coords: np.ndarray,
    values: np.ndarray,
    title: str = "Spatial Embedding",
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (8, 6),
) -> Figure:
    """
    Plot spatial coordinates colored by value.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates (n, 2)
    values : np.ndarray
        Values to color by
    title : str, optional
        Plot title
    cmap : str, optional
        Colormap
    figsize : tuple, optional
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=10, alpha=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal")

    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()
    return fig
