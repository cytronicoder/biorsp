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
    figsize: Tuple[float, float] = (8, 4),
) -> Figure:
    """
    Create QQ plot focused on significant p-value region (0 to α).

    Parameters
    ----------
    expected : np.ndarray
        Expected quantiles (uniform)
    observed : np.ndarray
        Observed p-value quantiles
    title : str, optional
        Plot title
    alpha : float, optional
        Significance threshold (default: 0.05)
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

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[0]
    mask = expected <= alpha
    exp_sig = expected[mask]
    obs_sig = observed[mask]
    ax.plot([0, alpha], [0, alpha], "k--", alpha=0.5, linewidth=2, label="Perfect calibration")
    ax.scatter(
        exp_sig, obs_sig, s=20, alpha=0.7, color="#2196F3", edgecolors="black", linewidth=0.5
    )
    ax.axvline(alpha, color="red", linestyle="--", alpha=0.3, linewidth=1.5, label=f"α={alpha}")
    ax.axhline(alpha, color="red", linestyle="--", alpha=0.3, linewidth=1.5)
    ax.set_xlabel("Expected p-value (Uniform)", fontsize=11)
    ax.set_ylabel("Observed p-value", fontsize=11)
    ax.set_title(f"Significance Region (p ≤ {alpha})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.002, alpha + 0.002)
    ax.set_ylim(-0.002, alpha + 0.002)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=2, label="Perfect calibration")
    ax.scatter(expected, observed, s=10, alpha=0.6, color="#2196F3")
    ax.axvline(alpha, color="red", linestyle=":", alpha=0.3, linewidth=1.5, label=f"α={alpha}")
    ax.axhline(alpha, color="red", linestyle=":", alpha=0.3, linewidth=1.5)
    ax.axvspan(0, alpha, alpha=0.1, color="red", label="Significance region")

    ax.set_xlabel("Expected p-value (Uniform)", fontsize=11)
    ax.set_ylabel("Observed p-value", fontsize=11)
    ax.set_title("Full Range", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
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

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

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

    ax.set_xticks(np.arange(len(cm_plot.columns)))
    ax.set_yticks(np.arange(len(cm_plot.index)))
    ax.set_xticklabels(cm_plot.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm_plot.index)

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


ARCHETYPE_COLORS = {
    "housekeeping": "#4CAF50",
    "regional_program": "#2196F3",
    "sparse_noise": "#9E9E9E",
    "niche_marker": "#FF5722",
    "abstention_stress": "#000000",
}

ARCHETYPE_DESCRIPTIONS = {
    "housekeeping": "Ubiquitous expression\n(high C, low S)",
    "regional_program": "Broad spatial domain\n(high C, high S)",
    "sparse_noise": "Scattered/rare\n(low C, low S)",
    "niche_marker": "Localized marker\n(low C, high S)",
}


def plot_archetype_scatter(
    coverage: np.ndarray,
    spatial_score: np.ndarray,
    true_archetypes: np.ndarray,
    c_cut: float = 0.30,
    s_cut: float = 0.15,
    title: str = "Panel A: Coverage vs Spatial Organization",
    figsize: Tuple[float, float] = (7, 6),
) -> Figure:
    """
    Create scatter plot of C vs S with ground truth coloring and quadrant boundaries.

    This is Panel A of the story figure.

    Parameters
    ----------
    coverage : np.ndarray
        Coverage values (C)
    spatial_score : np.ndarray
        Spatial organization scores (S)
    true_archetypes : np.ndarray
        Ground truth archetype labels
    c_cut : float
        Coverage threshold for quadrant boundary
    s_cut : float
        Spatial score threshold for quadrant boundary
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

    unique_archetypes = np.unique(true_archetypes)
    for archetype in unique_archetypes:
        mask = true_archetypes == archetype
        color = ARCHETYPE_COLORS.get(archetype, "#888888")
        ax.scatter(
            coverage[mask],
            spatial_score[mask],
            c=color,
            label=archetype,
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.axvline(c_cut, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(s_cut, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    annotations = [
        (c_cut / 2, s_cut / 2, "Sparse/Noisy", "#9E9E9E"),
        ((c_cut + x_lim[1]) / 2, s_cut / 2, "Housekeeping", "#4CAF50"),
        (c_cut / 2, (s_cut + y_lim[1]) / 2, "Niche Marker", "#FF5722"),
        ((c_cut + x_lim[1]) / 2, (s_cut + y_lim[1]) / 2, "Regional", "#2196F3"),
    ]

    for x, y, label, color in annotations:
        ax.annotate(
            label,
            (x, y),
            fontsize=9,
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    ax.set_xlabel("Coverage (C)", fontsize=11)
    ax.set_ylabel("Spatial Score (S)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(0, min(1.0, coverage.max() * 1.1))
    ax.set_ylim(0, min(1.0, spatial_score.max() * 1.2))

    plt.tight_layout()
    return fig


def plot_confusion_matrix_styled(
    cm_df: pd.DataFrame,
    title: str = "Panel B: Classification Confusion Matrix",
    accuracy: float = None,
    figsize: Tuple[float, float] = (6, 5),
) -> Figure:
    """
    Plot styled confusion matrix with annotations.

    This is Panel B of the story figure.

    Parameters
    ----------
    cm_df : pd.DataFrame
        Confusion matrix (rows=true, cols=predicted)
    title : str
        Plot title
    accuracy : float, optional
        Overall accuracy to display
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0)

    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    for i in range(len(cm_df.index)):
        for j in range(len(cm_df.columns)):
            count = cm_df.iloc[i, j]
            pct = cm_norm.iloc[i, j]
            text = f"{count}\n({pct:.0%})"
            color = "white" if pct > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_yticks(range(len(cm_df.index)))

    short_labels = {
        "housekeeping": "House.",
        "regional_program": "Regional",
        "sparse_noise": "Sparse",
        "niche_marker": "Niche",
    }
    xlabels = [short_labels.get(c, c) for c in cm_df.columns]
    ylabels = [short_labels.get(r, r) for r in cm_df.index]

    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ylabels, fontsize=9)

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)

    title_text = title
    if accuracy is not None:
        title_text += f"\nOverall Accuracy: {accuracy:.1%}"
    ax.set_title(title_text, fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Recall", shrink=0.8)
    plt.tight_layout()
    return fig


def plot_marker_recovery(
    precision_df: pd.DataFrame,
    title: str = "Panel C: Structured Gene Recovery",
    figsize: Tuple[float, float] = (5, 4),
) -> Figure:
    """
    Plot precision@K curve for marker recovery.

    This is Panel C of the story figure.

    Parameters
    ----------
    precision_df : pd.DataFrame
        Output of precision_at_k_curve with columns: k, precision_at_k
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

    k_vals = precision_df["k"].values
    prec_vals = precision_df["precision_at_k"].values

    bars = ax.bar(range(len(k_vals)), prec_vals, color="#FF5722", alpha=0.8, edgecolor="black")

    for i, (k, p) in enumerate(zip(k_vals, prec_vals)):
        ax.text(i, p + 0.02, f"{p:.0%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([f"Top {k}" for k in k_vals], fontsize=9)
    ax.set_xlabel("Ranked by Spatial Score (S)", fontsize=10)
    ax.set_ylabel("Precision\n(% true structured genes)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_module_recovery(
    module_metrics: dict,
    top_edges_df: pd.DataFrame = None,
    title: str = "Panel D: Gene-Gene Module Recovery",
    figsize: Tuple[float, float] = (5, 4),
) -> Figure:
    """
    Plot module recovery results.

    This is Panel D of the story figure.

    Parameters
    ----------
    module_metrics : dict
        Output of module_recovery_metrics with auprc, auroc, precision values
    top_edges_df : pd.DataFrame, optional
        Top predicted edges with is_true_edge column for visualization
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

    metrics = ["AUPRC", "AUROC", "Prec@10", "Prec@50"]
    values = [
        module_metrics.get("auprc", 0),
        module_metrics.get("auroc", 0),
        module_metrics.get("precision_at_10", 0),
        module_metrics.get("precision_at_50", 0),
    ]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor="black")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def compose_story_onepager(
    panel_a: Figure,
    panel_b: Figure,
    panel_c: Figure,
    panel_d: Figure,
    figsize: Tuple[float, float] = (14, 12),
) -> Figure:
    """
    Compose 4 panels into single one-page figure.

    Parameters
    ----------
    panel_a, panel_b, panel_c, panel_d : Figure
        Individual panel figures
    figsize : tuple
        Overall figure size

    Returns
    -------
    fig : Figure
        Combined figure
    """
    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    def copy_panel(source_fig, target_ax, label):
        source_ax = source_fig.axes[0]

        for collection in source_ax.collections:
            pass

        target_ax.set_title(source_ax.get_title(), fontsize=11, fontweight="bold")
        target_ax.set_xlabel(source_ax.get_xlabel(), fontsize=10)
        target_ax.set_ylabel(source_ax.get_ylabel(), fontsize=10)
        target_ax.text(
            -0.1,
            1.05,
            label,
            transform=target_ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
        )

    # Return the composed figure grid for now
    return fig


def plot_abstention_summary(
    abstention_df: pd.DataFrame,
    title: str = "Abstention Behavior Under Stress",
    figsize: Tuple[float, float] = (6, 4),
) -> Figure:
    """
    Plot abstention rates under various stress conditions.

    Parameters
    ----------
    abstention_df : pd.DataFrame
        DataFrame with columns: condition, abstention_rate, n_samples
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

    conditions = abstention_df["condition"].values
    rates = abstention_df["abstention_rate"].values

    colors = ["#4CAF50" if r < 0.3 else "#FF9800" if r < 0.7 else "#F44336" for r in rates]
    bars = ax.barh(conditions, rates, color=colors, alpha=0.8, edgecolor="black")

    for bar, rate in zip(bars, rates):
        ax.text(
            rate + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.0%}",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Abstention Rate", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_stability_summary(
    stability_metrics: dict,
    title: str = "Cross-Embedding Stability",
    figsize: Tuple[float, float] = (5, 4),
) -> Figure:
    """
    Plot embedding stability metrics.

    Parameters
    ----------
    stability_metrics : dict
        Output of embedding_stability_metrics
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

    metrics = ["Score\nCorrelation", "Label\nAgreement"]
    values = [
        stability_metrics.get("score_correlation", 0),
        stability_metrics.get("label_agreement", 0),
    ]

    colors = ["#2196F3", "#4CAF50"]
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor="black", width=0.5)

    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.9, color="green", linestyle="--", alpha=0.5, label="Target: 0.9")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    return fig


def plot_archetype_examples(
    coords: np.ndarray,
    X: np.ndarray,
    truth_df: pd.DataFrame,
    var_names: list,
    n_examples_per_archetype: int = 3,
    figsize: Tuple[float, float] = (12, 10),
) -> Figure:
    """
    Create visual sanity check: spatial expression maps for example genes from each archetype.

    Shows 4 rows (archetypes) × n_examples columns.
    Purpose: visually confirm that the simulated patterns match expected spatial structure.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n_cells, 2)
    X : np.ndarray
        Expression matrix (n_cells, n_genes)
    truth_df : pd.DataFrame
        Ground truth with 'archetype', 'gene', 'pattern_variant' columns
    var_names : list
        Gene names
    n_examples_per_archetype : int
        Number of example genes to show per archetype
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure (fig_archetype_examples.png)
    """
    archetype_order = ["housekeeping", "regional_program", "sparse_noise", "niche_marker"]
    n_rows = len(archetype_order)
    n_cols = n_examples_per_archetype

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, archetype in enumerate(archetype_order):
        arch_genes = truth_df[truth_df["archetype"] == archetype].head(n_examples_per_archetype)

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]

            if col_idx < len(arch_genes):
                gene_row = arch_genes.iloc[col_idx]
                gene_name = gene_row["gene"]
                gene_idx = (
                    var_names.index(gene_name)
                    if gene_name in var_names
                    else gene_row.get("gene_idx", col_idx)
                )
                pattern_var = gene_row.get("pattern_variant", "?")
                coverage = gene_row.get("achieved_coverage", np.nan)

                expr = X[:, gene_idx]
                binary = (expr > 0).astype(float)

                non_expr_mask = expr == 0
                ax.scatter(
                    coords[non_expr_mask, 0],
                    coords[non_expr_mask, 1],
                    c="lightgray",
                    s=3,
                    alpha=0.3,
                    rasterized=True,
                )
                expr_mask = expr > 0
                sc = ax.scatter(
                    coords[expr_mask, 0],
                    coords[expr_mask, 1],
                    c=expr[expr_mask],
                    cmap="Reds",  # Red shades only
                    s=5,
                    alpha=0.8,
                    vmin=0,
                    vmax=expr.max() if expr.max() > 0 else 1,
                    rasterized=True,
                )

                if expr_mask.sum() > 0:
                    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=6)
                    cbar.set_label("Counts", fontsize=7)

                ax.set_title(f"{pattern_var}\nC={coverage:.0%}", fontsize=9)
                ax.set_aspect("equal")
                ax.axis("off")
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

        # Row label
        axes[row_idx, 0].set_ylabel(archetype.replace("_", "\n"), fontsize=10, fontweight="bold")
        axes[row_idx, 0].yaxis.set_label_position("left")

    fig.suptitle("Archetype Examples: Visual Sanity Check", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_threshold_diagnostics(
    coverage: np.ndarray,
    spatial_score: np.ndarray,
    true_archetypes: np.ndarray,
    c_cut: float = 0.30,
    s_cut: float = 0.15,
    figsize: Tuple[float, float] = (14, 5),
) -> Figure:
    """
    Diagnostic plot showing C and S distributions with threshold cutoffs.

    Shows:
    - Left: Coverage (C) distribution by archetype with c_cut line
    - Center: Spatial Score (S) distribution by archetype with s_cut line
    - Right: Joint density with quadrant boundaries

    Purpose: Verify that thresholds separate archetypes as intended.

    Parameters
    ----------
    coverage : np.ndarray
        Coverage values (C)
    spatial_score : np.ndarray
        Spatial organization scores (S)
    true_archetypes : np.ndarray
        Ground truth archetype labels
    c_cut : float
        Coverage threshold
    s_cut : float
        Spatial score threshold
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure (fig_threshold_diagnostics.png)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    unique_archetypes = sorted(np.unique(true_archetypes))

    ax = axes[0]
    for archetype in unique_archetypes:
        mask = true_archetypes == archetype
        color = ARCHETYPE_COLORS.get(archetype, "#888888")
        ax.hist(
            coverage[mask],
            bins=20,
            alpha=0.5,
            color=color,
            label=archetype,
            density=True,
        )
    ax.axvline(c_cut, color="red", linestyle="--", linewidth=2, label=f"c_cut={c_cut:.2f}")
    ax.set_xlabel("Coverage (C)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Coverage Distribution by Archetype", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for archetype in unique_archetypes:
        mask = true_archetypes == archetype
        color = ARCHETYPE_COLORS.get(archetype, "#888888")
        ax.hist(
            spatial_score[mask],
            bins=20,
            alpha=0.5,
            color=color,
            label=archetype,
            density=True,
        )
    ax.axvline(s_cut, color="red", linestyle="--", linewidth=2, label=f"s_cut={s_cut:.2f}")
    ax.set_xlabel("Spatial Score (S)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Spatial Score Distribution by Archetype", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[2]
    for archetype in unique_archetypes:
        mask = true_archetypes == archetype
        color = ARCHETYPE_COLORS.get(archetype, "#888888")
        ax.scatter(
            coverage[mask],
            spatial_score[mask],
            c=color,
            label=archetype,
            s=30,
            alpha=0.6,
            edgecolors="white",
            linewidth=0.3,
        )
    ax.axvline(c_cut, color="red", linestyle="--", linewidth=2)
    ax.axhline(s_cut, color="red", linestyle="--", linewidth=2)

    ax.text(
        c_cut / 2,
        s_cut / 2,
        "sparse_noise\n(expected)",
        fontsize=8,
        ha="center",
        va="center",
        alpha=0.6,
    )
    ax.text(
        (c_cut + 1) / 2,
        s_cut / 2,
        "housekeeping\n(expected)",
        fontsize=8,
        ha="center",
        va="center",
        alpha=0.6,
    )
    ax.text(
        c_cut / 2,
        (s_cut + spatial_score.max()) / 2,
        "niche_marker\n(expected)",
        fontsize=8,
        ha="center",
        va="center",
        alpha=0.6,
    )
    ax.text(
        (c_cut + 1) / 2,
        (s_cut + spatial_score.max()) / 2,
        "regional_program\n(expected)",
        fontsize=8,
        ha="center",
        va="center",
        alpha=0.6,
    )

    ax.set_xlabel("Coverage (C)", fontsize=11)
    ax.set_ylabel("Spatial Score (S)", fontsize=11)
    ax.set_title("Joint Distribution with Thresholds", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle("Threshold Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_support_diagnostics(
    coverage_fg: np.ndarray,
    coverage_bg: np.ndarray,
    spatial_score: np.ndarray,
    true_archetypes: np.ndarray,
    figsize: Tuple[float, float] = (12, 5),
) -> Figure:
    """
    Diagnostic plot for supporting statistics that affect S-score reliability.

    Shows:
    - Left: S vs foreground cell count (n_fg)
    - Center: S vs background cell count (n_bg)
    - Right: S vs foreground/background ratio

    Purpose: Diagnose whether low support (few cells) causes unreliable S scores.

    Parameters
    ----------
    coverage_fg : np.ndarray
        Foreground (expressing) cell counts or fractions
    coverage_bg : np.ndarray
        Background (non-expressing) cell counts or fractions
    spatial_score : np.ndarray
        Spatial organization scores (S)
    true_archetypes : np.ndarray
        Ground truth archetype labels
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure (fig_support_diagnostics.png)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    unique_archetypes = sorted(np.unique(true_archetypes))

    ax = axes[0]
    for archetype in unique_archetypes:
        mask = true_archetypes == archetype
        color = ARCHETYPE_COLORS.get(archetype, "#888888")
        ax.scatter(
            coverage_fg[mask],
            spatial_score[mask],
            c=color,
            label=archetype,
            s=30,
            alpha=0.6,
        )
    ax.set_xlabel("Foreground Coverage (fraction or count)", fontsize=10)
    ax.set_ylabel("Spatial Score (S)", fontsize=10)
    ax.set_title("S vs Foreground Support", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for archetype in unique_archetypes:
        mask = true_archetypes == archetype
        color = ARCHETYPE_COLORS.get(archetype, "#888888")
        ax.scatter(
            coverage_bg[mask],
            spatial_score[mask],
            c=color,
            label=archetype,
            s=30,
            alpha=0.6,
        )
    ax.set_xlabel("Background Coverage (fraction or count)", fontsize=10)
    ax.set_ylabel("Spatial Score (S)", fontsize=10)
    ax.set_title("S vs Background Support", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ratio = coverage_fg / (coverage_bg + 1e-9)
    for archetype in unique_archetypes:
        mask = true_archetypes == archetype
        color = ARCHETYPE_COLORS.get(archetype, "#888888")
        ax.scatter(
            ratio[mask],
            spatial_score[mask],
            c=color,
            label=archetype,
            s=30,
            alpha=0.6,
        )
    ax.set_xlabel("FG/BG Ratio", fontsize=10)
    ax.set_ylabel("Spatial Score (S)", fontsize=10)
    ax.set_title("S vs FG/BG Ratio", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    fig.suptitle("Support Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_misclassified_scatter(
    coverage: np.ndarray,
    spatial_score: np.ndarray,
    true_archetypes: np.ndarray,
    pred_archetypes: np.ndarray,
    var_names: list,
    c_cut: float = 0.30,
    s_cut: float = 0.15,
    figsize: Tuple[float, float] = (10, 8),
) -> Tuple[Figure, pd.DataFrame]:
    """
    Scatter plot highlighting misclassified genes with detailed audit.

    Shows:
    - Correctly classified genes as small dots
    - Misclassified genes as large X markers with labels
    - Returns DataFrame with misclassification details

    Parameters
    ----------
    coverage : np.ndarray
        Coverage values (C)
    spatial_score : np.ndarray
        Spatial organization scores (S)
    true_archetypes : np.ndarray
        Ground truth archetype labels
    pred_archetypes : np.ndarray
        Predicted archetype labels
    var_names : list
        Gene names
    c_cut : float
        Coverage threshold
    s_cut : float
        Spatial score threshold
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure (fig_misclassified_scatter.png)
    misclass_df : pd.DataFrame
        Misclassification details (misclassified.csv)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Identify misclassified
    correct_mask = np.array(true_archetypes) == np.array(pred_archetypes)
    incorrect_mask = ~correct_mask

    unique_archetypes = sorted(np.unique(true_archetypes))

    for archetype in unique_archetypes:
        mask = (true_archetypes == archetype) & correct_mask
        color = ARCHETYPE_COLORS.get(archetype, "#888888")
        ax.scatter(
            coverage[mask],
            spatial_score[mask],
            c=color,
            s=20,
            alpha=0.3,
            label=f"{archetype} (correct)" if mask.sum() > 0 else None,
        )

    misclass_rows = []
    for i in np.where(incorrect_mask)[0]:
        gene = var_names[i] if i < len(var_names) else f"gene_{i}"
        true_arch = true_archetypes[i]
        pred_arch = pred_archetypes[i]
        c_val = coverage[i]
        s_val = spatial_score[i]

        color = ARCHETYPE_COLORS.get(true_arch, "#888888")
        ax.scatter(
            c_val,
            s_val,
            c=color,
            s=100,
            marker="X",
            edgecolors="black",
            linewidth=1.5,
            alpha=0.9,
        )
        # Annotate
        ax.annotate(
            f"{gene}\n{true_arch}→{pred_arch}",
            (c_val, s_val),
            fontsize=7,
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )

        misclass_rows.append(
            {
                "gene": gene,
                "true_archetype": true_arch,
                "predicted_archetype": pred_arch,
                "coverage": c_val,
                "spatial_score": s_val,
                "error_type": f"{true_arch}→{pred_arch}",
            }
        )

    ax.axvline(c_cut, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.axhline(s_cut, color="red", linestyle="--", linewidth=2, alpha=0.7)

    ax.set_xlabel("Coverage (C)", fontsize=11)
    ax.set_ylabel("Spatial Score (S)", fontsize=11)
    ax.set_title(
        f"Misclassification Audit\n({incorrect_mask.sum()}/{len(true_archetypes)} misclassified = {incorrect_mask.mean():.1%})",
        fontsize=12,
        fontweight="bold",
    )

    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    misclass_df = pd.DataFrame(misclass_rows)
    return fig, misclass_df


def plot_pattern_detectability(
    pattern_results: pd.DataFrame,
    title: str = "Pattern Detectability by S Score",
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """
    Show S score distribution for each pattern type to verify detectability.

    Purpose: Confirm which patterns produce high S (detectable) vs low S (not detectable).

    Parameters
    ----------
    pattern_results : pd.DataFrame
        DataFrame with columns: pattern_variant, spatial_score, archetype
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

    # Group by pattern and compute stats
    pattern_groups = pattern_results.groupby("pattern_variant")["spatial_score"]
    patterns = []
    means = []
    stds = []

    for pattern, scores in pattern_groups:
        patterns.append(pattern)
        means.append(scores.mean())
        stds.append(scores.std())

    sort_idx = np.argsort(means)[::-1]
    patterns = [patterns[i] for i in sort_idx]
    means = [means[i] for i in sort_idx]
    stds = [stds[i] for i in sort_idx]

    # Color by detectability expectation
    detectable_patterns = {"core", "rim", "radial_gradient", "wedge_core", "wedge_rim"}
    colors = ["#4CAF50" if p in detectable_patterns else "#FF5722" for p in patterns]

    bars = ax.barh(patterns, means, xerr=stds, color=colors, alpha=0.7, capsize=3)

    ax.set_xlabel("Spatial Score (S)", fontsize=11)
    ax.set_ylabel("Pattern Type", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axvline(0.15, color="red", linestyle="--", linewidth=2, label="s_cut threshold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    ax.text(
        0.95,
        0.05,
        "Green = Expected detectable\nRed = Expected non-detectable",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    return fig


# Null Calibration and Exemplar Plots (Issue A, B, E fixes)


def plot_null_distribution(
    null_s_values: np.ndarray,
    s_cut: float,
    margin: float = None,
    fpr_target: float = 0.05,
    title: str = "Null S Distribution for Threshold Calibration",
    figsize: Tuple[float, float] = (10, 5),
) -> Figure:
    """
    Plot null S distribution with calibrated threshold and FPR annotation.

    Purpose: Show that S_cut is derived from null and controls FPR at target level.

    Parameters
    ----------
    null_s_values : np.ndarray
        S scores from null (IID) simulations
    s_cut : float
        Calibrated threshold
    margin : float, optional
        Uncertainty margin (for borderline zone)
    fpr_target : float
        Target false positive rate
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure (fig_null_distribution.png)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    s_clean = null_s_values[~np.isnan(null_s_values)]
    empirical_fpr = np.mean(s_clean >= s_cut) if len(s_clean) > 0 else np.nan

    # Left: Histogram with threshold
    ax = axes[0]
    n_bins = min(30, len(s_clean) // 3) if len(s_clean) > 10 else 10
    ax.hist(s_clean, bins=n_bins, color="#2196F3", alpha=0.7, edgecolor="black", density=True)
    ax.axvline(
        s_cut,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"S_cut = {s_cut:.3f}\n(FPR = {empirical_fpr:.1%})",
    )

    if margin is not None:
        ax.axvspan(
            s_cut - margin,
            s_cut + margin,
            alpha=0.2,
            color="orange",
            label=f"Borderline zone (±{margin:.3f})",
        )

    ax.set_xlabel("Spatial Score (S)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Null Distribution", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: Empirical CDF with threshold
    ax = axes[1]
    sorted_s = np.sort(s_clean)
    cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    ax.plot(sorted_s, cdf, color="#2196F3", linewidth=2, label="Empirical CDF")
    ax.axvline(s_cut, color="red", linestyle="--", linewidth=2, label=f"S_cut = {s_cut:.3f}")
    ax.axhline(
        1 - fpr_target,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"1 - FPR_target = {1-fpr_target:.0%}",
    )

    ax.set_xlabel("Spatial Score (S)", fontsize=11)
    ax.set_ylabel("Cumulative Probability", fontsize=11)
    ax.set_title("CDF with Threshold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_exemplar_panel(
    coords: np.ndarray,
    expression: np.ndarray,
    archetype: str,
    coverage: float,
    spatial_score: float,
    s_cut: float,
    c_cut: float,
    margin: float = None,
    confidence: str = None,
    r_theta_profile: np.ndarray = None,
    theta_bins: np.ndarray = None,
    figsize: Tuple[float, float] = (12, 5),
) -> Figure:
    """
    Create exemplar panel for one archetype showing spatial pattern and metrics.

    Purpose: Let biologists visually confirm ground truth matches expectation.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n, 2)
    expression : np.ndarray
        Expression counts (n,)
    archetype : str
        Archetype label
    coverage : float
        Coverage (C) value
    spatial_score : float
        Spatial score (S) value
    s_cut : float
        S threshold
    c_cut : float
        C threshold
    margin : float, optional
        Uncertainty margin
    confidence : str, optional
        Confidence level ('high', 'borderline')
    r_theta_profile : np.ndarray, optional
        R(theta) radial profile for radar plot
    theta_bins : np.ndarray, optional
        Theta bin centers for radar plot
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    if r_theta_profile is not None and theta_bins is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9, 5))
        axes = list(axes) + [None]

    ax = axes[0]
    expr_mask = expression > 0
    non_expr_mask = ~expr_mask

    ax.scatter(
        coords[non_expr_mask, 0],
        coords[non_expr_mask, 1],
        c="lightgray",
        s=8,
        alpha=0.4,
        rasterized=True,
    )
    if expr_mask.sum() > 0:
        sc = ax.scatter(
            coords[expr_mask, 0],
            coords[expr_mask, 1],
            c=expression[expr_mask],
            cmap="Reds",
            s=12,
            alpha=0.8,
            vmin=0,
            vmax=expression.max() if expression.max() > 0 else 1,
            rasterized=True,
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Counts", fontsize=9)

    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_title(f"Spatial Expression\n{archetype.replace('_', ' ').title()}", fontsize=11)
    ax.set_aspect("equal")

    ax = axes[1]
    ax.axis("off")

    metrics_text = [
        f"Archetype: {archetype.replace('_', ' ').title()}",
        "",
        f"Coverage (C): {coverage:.1%}",
        f"Spatial Score (S): {spatial_score:.3f}",
        "",
        "Thresholds:",
        f"  C_cut: {c_cut:.2f}",
        f"  S_cut: {s_cut:.3f}",
    ]

    if margin is not None:
        metrics_text.append(f"  Margin: ±{margin:.3f}")

    if confidence is not None:
        conf_color = "#4CAF50" if confidence == "high" else "#FF9800"
        metrics_text.extend(["", f"Confidence: {confidence.upper()}"])

    # Classification result
    high_c = coverage >= c_cut
    high_s = spatial_score >= s_cut
    if high_c and high_s:
        pred = "regional_program"
    elif high_c and not high_s:
        pred = "housekeeping"
    elif not high_c and high_s:
        pred = "niche_marker"
    else:
        pred = "sparse_noise"

    correct = pred == archetype
    status = "✓ Correct" if correct else f"✗ Predicted: {pred}"
    status_color = "#4CAF50" if correct else "#F44336"

    metrics_text.extend(["", f"Classification: {status}"])

    ax.text(
        0.1,
        0.95,
        "\n".join(metrics_text),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray"),
    )

    if r_theta_profile is not None and theta_bins is not None and axes[2] is not None:
        ax = axes[2]
        ax.plot(theta_bins, r_theta_profile, color="#2196F3", linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(theta_bins, 0, r_theta_profile, alpha=0.3, color="#2196F3")
        ax.set_xlabel("θ (radians)", fontsize=10)
        ax.set_ylabel("R(θ) Radial Shift", fontsize=10)
        ax.set_title("Radial Profile", fontsize=11)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Exemplar: {archetype.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_precision_with_baseline(
    precision_df: pd.DataFrame,
    prevalence_baseline: float,
    title: str = "Precision@K with Baseline",
    figsize: Tuple[float, float] = (8, 5),
) -> Figure:
    """
    Plot precision@K curve with prevalence baseline and fold enrichment.

    Purpose: Show that retrieval performance is meaningful relative to class imbalance.

    Parameters
    ----------
    precision_df : pd.DataFrame
        Output of precision_at_k_curve with columns: k, precision_at_k
    prevalence_baseline : float
        Random baseline = fraction of true positives
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    k_vals = precision_df["k"].values
    prec_vals = precision_df["precision_at_k"].values

    # Left: Precision with baseline
    ax = axes[0]
    bars = ax.bar(range(len(k_vals)), prec_vals, color="#2196F3", alpha=0.8, edgecolor="black")
    ax.axhline(
        prevalence_baseline,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Baseline (prevalence) = {prevalence_baseline:.1%}",
    )

    for i, (k, p) in enumerate(zip(k_vals, prec_vals)):
        ax.text(i, p + 0.02, f"{p:.0%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([f"Top {k}" for k in k_vals], fontsize=9)
    ax.set_xlabel("Ranked by Score", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Precision@K", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Right: Fold enrichment
    ax = axes[1]
    fold_vals = prec_vals / prevalence_baseline if prevalence_baseline > 0 else prec_vals
    bars = ax.bar(range(len(k_vals)), fold_vals, color="#4CAF50", alpha=0.8, edgecolor="black")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=2, label="1× (random)")

    for i, (k, f) in enumerate(zip(k_vals, fold_vals)):
        ax.text(i, f + 0.1, f"{f:.1f}×", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([f"Top {k}" for k in k_vals], fontsize=9)
    ax.set_xlabel("Ranked by Score", fontsize=10)
    ax.set_ylabel("Fold Enrichment", fontsize=10)
    ax.set_title("Fold Enrichment over Random", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_genegene_distributions(
    within_module_scores: np.ndarray,
    between_module_scores: np.ndarray,
    prevalence: float = None,
    title: str = "Gene-Gene Score Distributions",
    figsize: Tuple[float, float] = (10, 5),
) -> Figure:
    """
    Plot within-module vs between-module score distributions.

    Purpose: Visual separation indicates module recovery capability.

    Parameters
    ----------
    within_module_scores : np.ndarray
        Scores for gene pairs in same module
    between_module_scores : np.ndarray
        Scores for gene pairs in different modules
    prevalence : float, optional
        Fraction of within-module pairs (for annotation)
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Overlapping histograms
    ax = axes[0]
    bins = np.linspace(
        min(within_module_scores.min(), between_module_scores.min()),
        max(within_module_scores.max(), between_module_scores.max()),
        30,
    )

    ax.hist(
        between_module_scores,
        bins=bins,
        alpha=0.6,
        color="#9E9E9E",
        label=f"Between-module (n={len(between_module_scores)})",
        density=True,
    )
    ax.hist(
        within_module_scores,
        bins=bins,
        alpha=0.6,
        color="#4CAF50",
        label=f"Within-module (n={len(within_module_scores)})",
        density=True,
    )

    ax.set_xlabel("Similarity Score", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Score Distributions", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: Box/violin plot
    ax = axes[1]
    data = [between_module_scores, within_module_scores]
    positions = [0, 1]
    colors = ["#9E9E9E", "#4CAF50"]

    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Between-module", "Within-module"], fontsize=10)
    ax.set_ylabel("Similarity Score", fontsize=10)
    ax.set_title("Score Comparison", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    from scipy import stats

    statistic, pvalue = stats.mannwhitneyu(
        within_module_scores, between_module_scores, alternative="greater"
    )
    ax.text(
        0.95,
        0.95,
        f"Mann-Whitney p = {pvalue:.2e}",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    if prevalence is not None:
        fig.text(
            0.5,
            0.01,
            f"Prevalence (within-module fraction): {prevalence:.1%}",
            ha="center",
            fontsize=9,
            style="italic",
        )

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_archetype_example_panel(
    coords: np.ndarray,
    counts: np.ndarray,
    archetype: str,
    coverage: float,
    spatial_score: float,
    s_cut: float = None,
    c_cut: float = 0.30,
    title: str = None,
    figsize: Tuple[float, float] = (12, 4),
) -> Figure:
    """
    Create example panel for one archetype showing embedding + R(θ) profile + metrics.

    This function generates biologist-readable visualization of what each archetype
    looks like in terms of spatial distribution and radial profile.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n, 2)
    counts : np.ndarray
        Expression counts (n,)
    archetype : str
        Ground truth archetype label
    coverage : float
        Observed coverage (fraction expressing)
    spatial_score : float
        Computed S score
    s_cut : float, optional
        Calibrated S threshold for reference line
    c_cut : float
        Coverage threshold (default: 0.30)
    title : str, optional
        Panel title (default: archetype name)
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Three-panel figure: scatter, radar profile, metrics box
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    expressing = counts >= 1
    title = title or archetype.replace("_", " ").title()

    ax = axes[0]
    ax.scatter(
        coords[~expressing, 0],
        coords[~expressing, 1],
        c="#CCCCCC",
        s=5,
        alpha=0.4,
        label="Non-expressing",
    )
    ax.scatter(
        coords[expressing, 0],
        coords[expressing, 1],
        c=ARCHETYPE_COLORS.get(archetype, "#2196F3"),
        s=15,
        alpha=0.7,
        label=f"Expressing (n={expressing.sum()})",
    )

    if expressing.sum() > 0:
        fg_center = coords[expressing].mean(axis=0)
        ax.scatter(
            *fg_center, c="red", marker="x", s=100, linewidths=2, label="FG centroid", zorder=10
        )
    bg_center = coords.mean(axis=0)
    ax.scatter(
        *bg_center, c="black", marker="+", s=100, linewidths=2, label="Overall centroid", zorder=10
    )

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Spatial Distribution\nC={coverage:.2f}")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]

    vantage = np.median(coords, axis=0)
    dx = coords[:, 0] - vantage[0]
    dy = coords[:, 1] - vantage[1]
    theta = np.arctan2(dy, dx)
    r = np.sqrt(dx**2 + dy**2)

    # Bin by angle and compute mean radius
    n_bins = 12
    theta_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2

    fg_r_mean = []
    bg_r_mean = []

    for i in range(n_bins):
        mask = (theta >= theta_bins[i]) & (theta < theta_bins[i + 1])
        fg_mask = mask & expressing
        bg_mask = mask & ~expressing

        fg_r_mean.append(r[fg_mask].mean() if fg_mask.sum() > 0 else np.nan)
        bg_r_mean.append(r[bg_mask].mean() if bg_mask.sum() > 0 else np.nan)

    # Close the loop for polar plot
    theta_plot = np.concatenate([theta_centers, [theta_centers[0]]])
    fg_r_plot = np.array(fg_r_mean + [fg_r_mean[0]])
    bg_r_plot = np.array(bg_r_mean + [bg_r_mean[0]])

    ax = plt.subplot(1, 3, 2, polar=True)
    ax.plot(
        theta_plot,
        fg_r_plot,
        "o-",
        color=ARCHETYPE_COLORS.get(archetype, "#2196F3"),
        label="Foreground",
        linewidth=2,
        markersize=4,
    )
    ax.plot(
        theta_plot,
        bg_r_plot,
        "s--",
        color="#666666",
        label="Background",
        linewidth=1.5,
        markersize=3,
    )

    ax.set_title(f"R(θ) Profile\nS={spatial_score:.3f}", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[2]
    ax.axis("off")

    metrics_text = f"""
Archetype: {archetype}

Coverage (C): {coverage:.3f}
Spatial Score (S): {spatial_score:.3f}

Classification:
  C threshold: {c_cut:.2f}
  S threshold: {s_cut:.3f if s_cut else 'N/A'}
  
Derived: {'HIGH' if coverage >= c_cut else 'LOW'} C
         {'HIGH' if (s_cut and spatial_score >= s_cut) else 'LOW'} S

Expressing cells: {expressing.sum()}/{len(counts)}
"""

    ax.text(
        0.1,
        0.9,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_title("Metrics Summary", fontsize=10)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_null_calibration_panel(
    null_s_values: np.ndarray,
    s_cut: float,
    fpr_target: float = 0.05,
    alt_s_values: np.ndarray = None,
    title: str = "Null Calibration",
    figsize: Tuple[float, float] = (10, 4),
) -> Figure:
    """
    Plot S distribution from null with calibrated threshold.

    Shows empirical null distribution, threshold placement, and expected FPR.
    Optionally overlay alternative (structured) distribution for power reference.

    Parameters
    ----------
    null_s_values : np.ndarray
        S scores from null (iid) simulations
    s_cut : float
        Calibrated S threshold
    fpr_target : float
        Target false positive rate (default: 0.05)
    alt_s_values : np.ndarray, optional
        S scores from structured genes for power comparison
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Two-panel figure: histogram + CDF
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    s_clean = null_s_values[~np.isnan(null_s_values)]
    empirical_fpr = np.mean(s_clean >= s_cut) if len(s_clean) > 0 else np.nan

    ax = axes[0]

    bins = np.linspace(s_clean.min(), max(s_clean.max(), s_cut * 1.2), 40)

    ax.hist(
        s_clean,
        bins=bins,
        alpha=0.7,
        color="#9E9E9E",
        label=f"Null (n={len(s_clean)})",
        density=True,
        edgecolor="black",
    )

    if alt_s_values is not None:
        alt_clean = alt_s_values[~np.isnan(alt_s_values)]
        if len(alt_clean) > 0:
            ax.hist(
                alt_clean,
                bins=bins,
                alpha=0.5,
                color="#4CAF50",
                label=f"Structured (n={len(alt_clean)})",
                density=True,
                edgecolor="black",
            )

    ax.axvline(
        s_cut,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"S_cut={s_cut:.3f} (FPR={empirical_fpr:.1%})",
    )

    # Shade rejection region
    ax.axvspan(s_cut, ax.get_xlim()[1], alpha=0.2, color="red", label="Reject region")

    ax.set_xlabel("Spatial Score (S)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("S Distribution", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]

    sorted_s = np.sort(s_clean)
    cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)

    ax.plot(sorted_s, cdf, color="#2196F3", linewidth=2, label="Null ECDF")

    fpr_at_cut = np.mean(s_clean >= s_cut)
    ax.axvline(s_cut, color="red", linestyle="--", linewidth=2)
    ax.axhline(1 - fpr_at_cut, color="red", linestyle=":", alpha=0.5)
    ax.scatter(
        [s_cut],
        [1 - fpr_at_cut],
        c="red",
        s=80,
        zorder=10,
        label=f"S_cut: {1-fpr_at_cut:.1%} null below",
    )

    # Target quantile line
    target_quantile = 1 - fpr_target
    ax.axhline(
        target_quantile,
        color="green",
        linestyle="--",
        alpha=0.5,
        label=f"Target: {target_quantile:.0%} quantile",
    )

    ax.set_xlabel("Spatial Score (S)", fontsize=10)
    ax.set_ylabel("Cumulative Probability", fontsize=10)
    ax.set_title(f"Empirical CDF\nTarget FPR={fpr_target:.0%}", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_genegene_example_panel(
    similarity_matrix: np.ndarray,
    gene_names: list,
    true_modules: np.ndarray,
    predicted_clusters: np.ndarray = None,
    title: str = "Gene-Gene Module Recovery",
    figsize: Tuple[float, float] = (14, 5),
) -> Figure:
    """
    Create example panel for gene-gene module analysis.

    Shows similarity heatmap with module structure and clustering comparison.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Pairwise similarity matrix (n_genes x n_genes)
    gene_names : list
        Gene names for axis labels
    true_modules : np.ndarray
        Ground truth module assignments
    predicted_clusters : np.ndarray, optional
        Predicted cluster assignments for comparison
    title : str
        Panel title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Multi-panel figure showing similarity structure
    """
    n_genes = len(gene_names)
    n_panels = 3 if predicted_clusters is not None else 2

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    ax = axes[0]
    im = ax.imshow(similarity_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_genes))
    ax.set_yticks(range(n_genes))
    ax.set_xticklabels(gene_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(gene_names, fontsize=7)
    ax.set_title("Similarity Matrix", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Similarity")

    ax = axes[1]

    # Sort genes by module
    module_order = np.argsort(true_modules)
    reordered = similarity_matrix[module_order][:, module_order]
    reordered_names = [gene_names[i] for i in module_order]
    reordered_modules = true_modules[module_order]

    im = ax.imshow(reordered, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_genes))
    ax.set_yticks(range(n_genes))
    ax.set_xticklabels(reordered_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(reordered_names, fontsize=7)

    unique_modules = np.unique(reordered_modules)
    boundaries = []
    for i, m in enumerate(unique_modules[:-1]):
        last_idx = np.where(reordered_modules == m)[0][-1]
        boundaries.append(last_idx + 0.5)
        ax.axhline(last_idx + 0.5, color="black", linewidth=1.5)
        ax.axvline(last_idx + 0.5, color="black", linewidth=1.5)

    ax.set_title(f"Ordered by True Modules\n({len(unique_modules)} modules)", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Similarity")

    if predicted_clusters is not None:
        ax = axes[2]

        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

            ari = adjusted_rand_score(true_modules, predicted_clusters)
            nmi = normalized_mutual_info_score(true_modules, predicted_clusters)
        except ImportError:
            ari, nmi = np.nan, np.nan

        # Contingency matrix
        unique_true = np.unique(true_modules)
        unique_pred = np.unique(predicted_clusters)
        contingency = np.zeros((len(unique_true), len(unique_pred)))

        for i, t in enumerate(unique_true):
            for j, p in enumerate(unique_pred):
                contingency[i, j] = np.sum((true_modules == t) & (predicted_clusters == p))

        im = ax.imshow(contingency, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(unique_pred)))
        ax.set_yticks(range(len(unique_true)))
        ax.set_xticklabels([f"C{p}" for p in unique_pred], fontsize=9)
        ax.set_yticklabels([f"M{t}" for t in unique_true], fontsize=9)
        ax.set_xlabel("Predicted Cluster", fontsize=10)
        ax.set_ylabel("True Module", fontsize=10)
        ax.set_title(f"Contingency\nARI={ari:.3f}, NMI={nmi:.3f}", fontsize=10)

        for i in range(len(unique_true)):
            for j in range(len(unique_pred)):
                count = int(contingency[i, j])
                if count > 0:
                    ax.text(
                        j,
                        i,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="white" if count > contingency.max() / 2 else "black",
                    )

        plt.colorbar(im, ax=ax, shrink=0.8, label="Count")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_robustness_example_panel(
    baseline_coords: np.ndarray,
    distorted_coords: np.ndarray,
    baseline_scores: np.ndarray,
    distorted_scores: np.ndarray,
    distortion_type: str,
    gene_names: list = None,
    title: str = None,
    figsize: Tuple[float, float] = (14, 5),
) -> Figure:
    """
    Create example panel for robustness analysis.

    Shows baseline vs distorted comparison for coordinates and scores.

    Parameters
    ----------
    baseline_coords : np.ndarray
        Original cell coordinates (n, 2)
    distorted_coords : np.ndarray
        Distorted cell coordinates (n, 2)
    baseline_scores : np.ndarray
        S scores from baseline (per gene or single)
    distorted_scores : np.ndarray
        S scores from distorted data (matching baseline)
    distortion_type : str
        Name of distortion (e.g., "downsample", "jitter")
    gene_names : list, optional
        Gene names for labeling
    title : str, optional
        Panel title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Three-panel comparison figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    title = title or f"Robustness: {distortion_type}"

    ax = axes[0]

    # Sample subset for visualization if too many points
    n_plot = min(500, len(baseline_coords))
    idx = np.random.choice(len(baseline_coords), n_plot, replace=False)

    ax.scatter(
        baseline_coords[idx, 0],
        baseline_coords[idx, 1],
        c="#2196F3",
        s=10,
        alpha=0.5,
        label="Baseline",
    )
    ax.scatter(
        distorted_coords[idx, 0],
        distorted_coords[idx, 1],
        c="#FF5722",
        s=10,
        alpha=0.5,
        label="Distorted",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Coordinate Comparison\n(n={n_plot} shown)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]

    valid_mask = ~(np.isnan(baseline_scores) | np.isnan(distorted_scores))
    b_valid = baseline_scores[valid_mask]
    d_valid = distorted_scores[valid_mask]

    ax.scatter(b_valid, d_valid, c="#4CAF50", s=30, alpha=0.6, edgecolors="black", linewidth=0.5)

    # Identity line
    lims = [min(b_valid.min(), d_valid.min()), max(b_valid.max(), d_valid.max())]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=2, label="y=x")

    # Correlation
    if len(b_valid) >= 3:
        corr = np.corrcoef(b_valid, d_valid)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("Baseline S", fontsize=10)
    ax.set_ylabel("Distorted S", fontsize=10)
    ax.set_title(f"Score Stability\n(n={len(b_valid)} genes)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    ax = axes[2]

    deltas = d_valid - b_valid

    ax.hist(deltas, bins=30, color="#9C27B0", alpha=0.7, edgecolor="black")
    ax.axvline(0, color="black", linestyle="-", linewidth=2)
    ax.axvline(
        np.median(deltas),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(deltas):.3f}",
    )

    # IQR shading
    q25, q75 = np.percentile(deltas, [25, 75])
    ax.axvspan(q25, q75, alpha=0.2, color="blue", label=f"IQR: [{q25:.3f}, {q75:.3f}]")

    ax.set_xlabel("ΔS (Distorted - Baseline)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Score Changes\nMAD={np.median(np.abs(deltas)):.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def compose_archetype_examples_grid(
    example_data: list,
    s_cut: float,
    c_cut: float = 0.30,
    title: str = "Archetype Examples (2×4 Grid)",
    figsize: Tuple[float, float] = (16, 10),
) -> Figure:
    """
    Compose 2×4 grid showing examples of all 4 archetypes.

    Top row: Spatial scatter plots
    Bottom row: Radial profiles R(θ)

    Parameters
    ----------
    example_data : list
        List of 4 dicts with keys: 'coords', 'counts', 'archetype', 'coverage', 'spatial_score'
    s_cut : float
        Calibrated S threshold
    c_cut : float
        Coverage threshold
    title : str
        Overall figure title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        2×4 grid figure
    """
    assert len(example_data) == 4, "Need exactly 4 archetype examples"

    fig, axes = plt.subplots(2, 4, figsize=figsize)

    archetype_order = ["housekeeping", "regional_program", "sparse_noise", "niche_marker"]

    for col, target_archetype in enumerate(archetype_order):
        # Find matching example
        ex = None
        for d in example_data:
            if d["archetype"] == target_archetype:
                ex = d
                break

        if ex is None:
            continue

        coords = ex["coords"]
        counts = ex["counts"]
        coverage = ex["coverage"]
        spatial_score = ex["spatial_score"]
        expressing = counts >= 1

        ax = axes[0, col]
        ax.scatter(coords[~expressing, 0], coords[~expressing, 1], c="#CCCCCC", s=3, alpha=0.3)
        ax.scatter(
            coords[expressing, 0],
            coords[expressing, 1],
            c=ARCHETYPE_COLORS.get(target_archetype, "#2196F3"),
            s=10,
            alpha=0.6,
        )

        ax.set_aspect("equal")
        ax.set_title(
            f"{target_archetype.replace('_', ' ').title()}\nC={coverage:.2f}, S={spatial_score:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # Color-code border based on classification
        border_color = (
            "green"
            if (
                (coverage >= c_cut and spatial_score < s_cut and target_archetype == "housekeeping")
                or (
                    coverage >= c_cut
                    and spatial_score >= s_cut
                    and target_archetype == "regional_program"
                )
                or (
                    coverage < c_cut
                    and spatial_score < s_cut
                    and target_archetype == "sparse_noise"
                )
                or (
                    coverage < c_cut
                    and spatial_score >= s_cut
                    and target_archetype == "niche_marker"
                )
            )
            else "red"
        )

        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(3)

        ax = axes[1, col]

        vantage = np.median(coords, axis=0)
        dx = coords[:, 0] - vantage[0]
        dy = coords[:, 1] - vantage[1]
        theta = np.arctan2(dy, dx)
        r = np.sqrt(dx**2 + dy**2)

        n_bins = 12
        theta_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2

        fg_r_mean, bg_r_mean = [], []
        for i in range(n_bins):
            mask = (theta >= theta_bins[i]) & (theta < theta_bins[i + 1])
            fg_r_mean.append(
                r[mask & expressing].mean() if (mask & expressing).sum() > 0 else np.nan
            )
            bg_r_mean.append(
                r[mask & ~expressing].mean() if (mask & ~expressing).sum() > 0 else np.nan
            )

        ax.bar(
            range(n_bins),
            fg_r_mean,
            alpha=0.7,
            color=ARCHETYPE_COLORS.get(target_archetype, "#2196F3"),
            label="FG",
        )
        ax.plot(range(n_bins), bg_r_mean, "k--", marker="o", markersize=4, label="BG")

        ax.set_xlabel("θ bin", fontsize=9)
        ax.set_ylabel("Mean radius", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    fig.text(
        0.5,
        0.02,
        f"Thresholds: C_cut={c_cut:.2f}, S_cut={s_cut:.3f}",
        ha="center",
        fontsize=10,
        style="italic",
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
