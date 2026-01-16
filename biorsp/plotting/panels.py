"""
Standardized Panel Generators for BioRSP Plots.

This module provides the canonical panel generators used across all simulation
benchmarks and kidney case studies. Each function produces a specific panel
with consistent styling, axes, labels, and semantics.

Panel naming convention:
- A_archetype_scatter.png: Coverage vs Spatial Score scatter with quadrants
- B_confusion_or_composition.png: Confusion matrix (sim) or composition bar (kidney)
- C_examples_per_archetype.png: Example spatial patterns for each archetype
- D_pairwise_or_module.png: Gene-gene pairs (sim) or modules (kidney)

All panels save with accompanying .txt caption files.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from biorsp.plotting.spec import PlotSpec
from biorsp.plotting.style import (
    get_column_width,
    save_figure,
    set_publication_style,
)

logger = logging.getLogger(__name__)
set_publication_style()


def plot_archetype_scatter(
    df: pd.DataFrame,
    spec: PlotSpec,
    title: str = "Coverage vs Spatial Organization",
    color_by: str = "Archetype",
    highlight_genes: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show_annotations: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """
    Create Panel A: Archetype scatter plot (Coverage vs Spatial Score).

    This is the primary figure showing the (C, S) distribution with quadrant
    boundaries and archetype coloring. The cutoff lines MUST match the
    classification logic in PlotSpec.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Coverage, Spatial_Score, and Archetype columns
    spec : PlotSpec
        Plot specification with cutoffs and colors
    title : str
        Plot title
    color_by : str
        Column to use for coloring ("Archetype" or "true_archetype")
    highlight_genes : list, optional
        Gene names to highlight with larger markers
    figsize : tuple, optional
        Figure size (default: single column width)
    show_annotations : bool
        Whether to show quadrant annotations
    ax : Axes, optional
        Existing axes to plot on (if None, creates new figure)

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    if figsize is None:
        figsize = (get_column_width("single"), get_column_width("single"))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Validate DataFrame
    validation = spec.validate_dataframe(df)
    if validation["status"] == "FAIL":
        logger.error(f"DataFrame validation failed: {validation['issues']}")
        raise ValueError("Invalid DataFrame for plotting")
    if validation["warnings"]:
        for w in validation["warnings"]:
            logger.warning(w)

    # Get column names from spec
    c_col = spec.coverage_col
    s_col = spec.spatial_col

    # Ensure archetype column exists
    if color_by not in df.columns:
        logger.info(f"Column '{color_by}' not found, classifying now...")
        df = spec.classify_dataframe(df, inplace=False)
        color_by = spec.archetype_col

    # Plot each archetype separately for legend control
    archetypes = spec.get_legend_order()
    for archetype in archetypes:
        mask = df[color_by] == archetype
        if not mask.any():
            continue

        color = spec.get_color(archetype)
        ax.scatter(
            df.loc[mask, c_col],
            df.loc[mask, s_col],
            c=color,
            label=archetype,
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            zorder=2,
        )

    # Highlight specific genes if requested
    if highlight_genes and "gene" in df.columns:
        highlight_mask = df["gene"].isin(highlight_genes)
        if highlight_mask.any():
            ax.scatter(
                df.loc[highlight_mask, c_col],
                df.loc[highlight_mask, s_col],
                s=150,
                facecolors="none",
                edgecolors="black",
                linewidths=2,
                zorder=3,
                label="Highlighted",
            )

    # Draw quadrant cutoff lines (CRITICAL: must match classification logic)
    c_cut, s_cut = spec.get_quadrant_bounds()
    ax.axvline(c_cut, color="black", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
    ax.axhline(s_cut, color="black", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)

    # Add quadrant annotations
    if show_annotations:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        annotations = [
            (c_cut * 0.5, s_cut * 0.5, "Basal"),
            ((c_cut + x_lim[1]) * 0.5, s_cut * 0.5, "Ubiquitous"),
            (c_cut * 0.5, (s_cut + y_lim[1]) * 0.5, "Patchy"),
            ((c_cut + x_lim[1]) * 0.5, (s_cut + y_lim[1]) * 0.5, "Gradient"),
        ]

        for x, y, label in annotations:
            color = spec.get_color(label)
            ax.annotate(
                label,
                (x, y),
                fontsize=9,
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
                alpha=0.6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=0,
            )

    ax.set_xlabel("Coverage (C)", fontsize=11)
    ax.set_ylabel("Spatial Score (S)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    ax.legend(loc="upper right", framealpha=0.95, fontsize=9, edgecolor="gray")
    ax.grid(True, alpha=0.3, zorder=0)

    # Set limits with padding
    ax.set_xlim(-0.02, min(1.05, df[c_col].max() * 1.1))
    ax.set_ylim(-0.02, min(1.0, df[s_col].max() * 1.15))

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    df: pd.DataFrame,
    spec: PlotSpec,
    title: str = "Classification Performance",
    normalize: str = "true",
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """
    Create Panel B (simulation): Confusion matrix heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with true_archetype and Archetype columns
    spec : PlotSpec
        Plot specification
    title : str
        Plot title
    normalize : str
        Normalization mode: "true" (by row), "pred" (by col), "all", or None
    figsize : tuple, optional
        Figure size
    ax : Axes, optional
        Existing axes

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    if figsize is None:
        figsize = (6, 5)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Check for required columns
    if "true_archetype" not in df.columns:
        raise ValueError("Confusion matrix requires 'true_archetype' column")

    # Ensure predicted archetype exists
    if spec.archetype_col not in df.columns:
        df = spec.classify_dataframe(df, inplace=False)

    # Build confusion matrix
    archetypes = spec.get_legend_order()
    cm = pd.crosstab(
        df["true_archetype"],
        df[spec.archetype_col],
        rownames=["True"],
        colnames=["Predicted"],
        dropna=False,
    )

    # Reindex to ensure all archetypes appear
    cm = cm.reindex(index=archetypes, columns=archetypes, fill_value=0)

    # Normalize if requested
    if normalize == "true":
        cm = cm.div(cm.sum(axis=1), axis=0).fillna(0)
    elif normalize == "pred":
        cm = cm.div(cm.sum(axis=0), axis=1).fillna(0)
    elif normalize == "all":
        cm = cm / cm.sum().sum()

    # Plot heatmap
    im = ax.imshow(cm.values, cmap="Blues", aspect="auto", vmin=0, vmax=cm.values.max())

    # Add text annotations
    for i in range(len(archetypes)):
        for j in range(len(archetypes)):
            val = cm.iloc[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"

            color = "white" if val > cm.values.max() * 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

    # Set ticks and labels
    ax.set_xticks(range(len(archetypes)))
    ax.set_yticks(range(len(archetypes)))
    ax.set_xticklabels(archetypes, rotation=45, ha="right")
    ax.set_yticklabels(archetypes)

    ax.set_xlabel("Predicted Archetype", fontsize=11)
    ax.set_ylabel("True Archetype", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if normalize:
        cbar.set_label("Fraction", rotation=270, labelpad=15)
    else:
        cbar.set_label("Count", rotation=270, labelpad=15)

    # Don't use tight_layout with colorbar (causes issues)
    return fig


def plot_composition_bar(
    df: pd.DataFrame,
    spec: PlotSpec,
    group_by: str = "condition",
    title: str = "Archetype Composition by Condition",
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """
    Create Panel B (kidney): Stacked bar chart of archetype composition.

    Used when ground truth is not available (e.g., kidney analysis).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Archetype column and grouping column
    spec : PlotSpec
        Plot specification
    group_by : str
        Column name to group by (e.g., "condition", "cluster")
    title : str
        Plot title
    figsize : tuple, optional
        Figure size
    ax : Axes, optional
        Existing axes

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    if figsize is None:
        figsize = (8, 5)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Ensure archetype column exists
    if spec.archetype_col not in df.columns:
        df = spec.classify_dataframe(df, inplace=False)

    # Check grouping column
    if group_by not in df.columns:
        raise ValueError(f"Grouping column '{group_by}' not found in DataFrame")

    # Count archetypes per group
    counts = pd.crosstab(df[group_by], df[spec.archetype_col])

    # Reindex to ensure all archetypes appear in canonical order
    archetypes = spec.get_legend_order()
    counts = counts.reindex(columns=archetypes, fill_value=0)

    # Normalize to fractions
    fractions = counts.div(counts.sum(axis=1), axis=0)

    # Plot stacked bars
    bottom = np.zeros(len(fractions))
    for archetype in archetypes:
        if archetype not in fractions.columns:
            continue
        color = spec.get_color(archetype)
        ax.bar(
            range(len(fractions)),
            fractions[archetype],
            bottom=bottom,
            color=color,
            label=archetype,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += fractions[archetype]

    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels(fractions.index, rotation=45, ha="right")
    ax.set_ylabel("Fraction of Genes", fontsize=11)
    ax.set_xlabel(group_by.capitalize(), fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)

    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def save_panel_with_caption(
    fig: Figure,
    outpath: Path,
    caption: str,
    dpi: int = 300,
):
    """
    Save a panel figure with accompanying caption text file.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save
    outpath : Path
        Output path (e.g., "figures/A_archetype_scatter.png")
    caption : str
        Plain-language caption (1-3 sentences)
    dpi : int
        Resolution for raster formats
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    save_figure(fig, str(outpath), dpi=dpi, formats=["png", "pdf"])

    # Save caption
    caption_path = outpath.with_suffix(".txt")
    with open(caption_path, "w") as f:
        f.write(caption.strip() + "\n")

    logger.info(f"Saved panel: {outpath} (with caption)")


def generate_standard_panels(
    df: pd.DataFrame,
    spec: PlotSpec,
    outdir: Path,
    mode: str = "simulation",
    group_by: Optional[str] = None,
):
    """
    Generate the standard figure set (Panels A and B) for a run.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame with Coverage, Spatial_Score, Archetype columns
    spec : PlotSpec
        Plot specification
    outdir : Path
        Output directory for figures
    mode : str
        "simulation" (with ground truth) or "kidney" (no ground truth)
    group_by : str, optional
        For kidney mode, column to group by in Panel B

    Returns
    -------
    None
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Panel A: Archetype Scatter (always generated)
    logger.info("Generating Panel A: Archetype Scatter...")
    color_by = (
        "true_archetype"
        if mode == "simulation" and "true_archetype" in df.columns
        else spec.archetype_col
    )
    fig_a = plot_archetype_scatter(df, spec, color_by=color_by)

    caption_a = (
        f"Panel A: Coverage vs Spatial Score scatter plot. "
        f"Quadrant boundaries at C={spec.c_cut:.2f}, S={spec.s_cut:.2f}. "
        f"Each point represents one gene or simulation replicate. "
        f"Colors indicate archetype classification."
    )
    save_panel_with_caption(fig_a, outdir / "A_archetype_scatter.png", caption_a)
    plt.close(fig_a)

    # Panel B: Confusion Matrix or Composition
    logger.info("Generating Panel B...")
    if mode == "simulation" and "true_archetype" in df.columns:
        fig_b = plot_confusion_matrix(df, spec)
        caption_b = (
            "Panel B: Confusion matrix showing classification performance. "
            "Rows represent true archetypes, columns represent predicted archetypes. "
            "Values are normalized by true archetype (row sums to 1)."
        )
    elif group_by:
        fig_b = plot_composition_bar(df, spec, group_by=group_by)
        caption_b = (
            f"Panel B: Archetype composition stratified by {group_by}. "
            "Stacked bars show the fraction of genes in each archetype for each group."
        )
    else:
        # Fallback: overall composition pie chart
        logger.warning("No ground truth or grouping column; skipping Panel B")
        return

    save_panel_with_caption(fig_b, outdir / "B_confusion_or_composition.png", caption_b)
    plt.close(fig_b)

    logger.info(f"Standard panels saved to {outdir}")


def plot_examples_panel(
    coords: np.ndarray,
    expression: np.ndarray,
    gene_names: List[str],
    df: pd.DataFrame,
    spec: PlotSpec,
    n_examples_per_archetype: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "C. Representative Genes by Archetype",
) -> Figure:
    """
    Create Panel C: Example spatial patterns for each archetype.

    Shows spatial expression patterns for representative genes in each archetype.

    Parameters
    ----------
    coords : np.ndarray
        Cell spatial coordinates (n_cells, 2)
    expression : np.ndarray
        Expression matrix (n_cells, n_genes)
    gene_names : list
        Gene names corresponding to columns in expression
    df : pd.DataFrame
        Results DataFrame with gene names and archetypes
    spec : PlotSpec
        Plot specification
    n_examples_per_archetype : int
        Number of example genes per archetype
    figsize : tuple, optional
        Figure size
    title : str
        Panel title

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    archetypes = spec.get_legend_order()
    n_archs = len(archetypes)

    if figsize is None:
        figsize = (3 * n_examples_per_archetype, 3 * n_archs)

    fig, axes = plt.subplots(n_archs, n_examples_per_archetype, figsize=figsize, squeeze=False)

    # Ensure archetype column exists
    if spec.archetype_col not in df.columns:
        df = spec.classify_dataframe(df, inplace=False)

    gene_col = "gene" if "gene" in df.columns else "gene_name"
    if gene_col not in df.columns:
        raise ValueError("DataFrame must have 'gene' or 'gene_name' column")

    # Create gene name to index mapping
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    for i, archetype in enumerate(archetypes):
        # Select representative examples (highest spatial score for Patchy/Gradient,
        # highest coverage for Ubiquitous, lowest for Basal)
        if archetype in ["Patchy", "Gradient"]:
            sort_col = spec.spatial_col
            ascending = False
        elif archetype == "Ubiquitous":
            sort_col = spec.coverage_col
            ascending = False
        else:  # Basal
            sort_col = spec.coverage_col
            ascending = True

        examples = (
            df[df[spec.archetype_col] == archetype]
            .sort_values(sort_col, ascending=ascending)
            .head(n_examples_per_archetype)[gene_col]
            .tolist()
        )

        color = spec.get_color(archetype)

        for j in range(n_examples_per_archetype):
            ax = axes[i, j]

            if j < len(examples):
                gene = examples[j]
                if gene in gene_to_idx:
                    idx = gene_to_idx[gene]
                    expr = expression[:, idx]

                    # Normalize for visualization
                    expr_norm = (expr - expr.min()) / (expr.max() - expr.min() + 1e-10)

                    ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        c=expr_norm,
                        cmap="viridis",
                        s=5,
                        alpha=0.7,
                        rasterized=True,
                    )
                    ax.set_title(gene, fontsize=9, color=color, fontweight="bold")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"{gene}\n(not in matrix)",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(gene, fontsize=9, color="gray")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

            # Row label on first column
            if j == 0:
                ax.set_ylabel(archetype, fontsize=10, fontweight="bold", color=color)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_pairwise_panel(
    pairs_df: pd.DataFrame,
    spec: PlotSpec,
    score_col: str = "similarity_profile",
    title: str = "D. Gene-Gene Co-patterning",
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """
    Create Panel D: Gene-gene pairwise similarity analysis.

    Shows distribution of pairwise similarity scores, optionally highlighting
    known co-regulated pairs if ground truth is available.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame with gene pairs and similarity scores
    spec : PlotSpec
        Plot specification
    score_col : str
        Column name for similarity scores
    title : str
        Panel title
    figsize : tuple, optional
        Figure size
    ax : Axes, optional
        Existing axes

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    if figsize is None:
        figsize = (6, 5)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Check for required columns
    if score_col not in pairs_df.columns:
        raise ValueError(f"Missing score column: {score_col}")

    scores = pairs_df[score_col].dropna()

    # Check for ground truth
    has_truth = "is_true_edge" in pairs_df.columns

    if has_truth:
        true_pairs = pairs_df[pairs_df["is_true_edge"] == True][score_col].dropna()  # noqa: E712
        false_pairs = pairs_df[pairs_df["is_true_edge"] == False][score_col].dropna()  # noqa: E712

        # Histogram with two distributions
        bins = np.linspace(scores.min(), scores.max(), 30)
        ax.hist(
            false_pairs,
            bins=bins,
            alpha=0.6,
            color="#9E9E9E",
            label=f"Non-module pairs (n={len(false_pairs):,})",
            density=True,
        )
        ax.hist(
            true_pairs,
            bins=bins,
            alpha=0.8,
            color=spec.get_color("Gradient"),
            label=f"True module pairs (n={len(true_pairs):,})",
            density=True,
        )

        # Add separation statistic
        from scipy.stats import mannwhitneyu

        stat, pval = mannwhitneyu(true_pairs, false_pairs, alternative="greater")
        ax.text(
            0.95,
            0.95,
            f"Mann-Whitney p={pval:.2e}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    else:
        # Simple histogram
        ax.hist(
            scores,
            bins=30,
            alpha=0.7,
            color=spec.get_color("Gradient"),
            label=f"All pairs (n={len(scores):,})",
            density=True,
        )

    ax.set_xlabel("Similarity Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_marker_recovery_panel(
    precision_df: pd.DataFrame,
    title: str = "C. Marker Recovery (Precision@K)",
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """
    Create Panel C alternative: Precision@K bar chart for marker recovery.

    Parameters
    ----------
    precision_df : pd.DataFrame
        DataFrame with k and precision_at_k columns
    title : str
        Panel title
    figsize : tuple, optional
        Figure size
    ax : Axes, optional
        Existing axes

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    if figsize is None:
        figsize = (6, 5)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    k_values = precision_df["k"].values
    precisions = precision_df["precision_at_k"].values

    bars = ax.bar(
        range(len(k_values)),
        precisions,
        color="#FF5722",
        alpha=0.8,
        edgecolor="black",
    )

    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f"Top {int(k)}" for k in k_values], fontsize=9)
    ax.set_xlabel("Genes Ranked by Spatial Score", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)

    # Add baseline reference
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")

    # Add value labels
    for bar, prec in zip(bars, precisions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{prec:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def generate_full_panel_suite(
    df: pd.DataFrame,
    spec: PlotSpec,
    outdir: Path,
    coords: Optional[np.ndarray] = None,
    expression: Optional[np.ndarray] = None,
    gene_names: Optional[List[str]] = None,
    pairs_df: Optional[pd.DataFrame] = None,
    precision_df: Optional[pd.DataFrame] = None,
    mode: str = "simulation",
    group_by: Optional[str] = None,
):
    """
    Generate the complete figure suite (Panels A, B, C, D).

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    spec : PlotSpec
        Plot specification
    outdir : Path
        Output directory
    coords : np.ndarray, optional
        Cell coordinates for Panel C examples
    expression : np.ndarray, optional
        Expression matrix for Panel C examples
    gene_names : list, optional
        Gene names for Panel C examples
    pairs_df : pd.DataFrame, optional
        Pairwise scores for Panel D
    precision_df : pd.DataFrame, optional
        Precision@K results for Panel C alternative
    mode : str
        "simulation" or "kidney"
    group_by : str, optional
        Column for grouping in kidney mode

    Returns
    -------
    None
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Generate standard panels A and B
    generate_standard_panels(df, spec, outdir, mode=mode, group_by=group_by)

    # Panel C: Examples or Marker Recovery
    if coords is not None and expression is not None and gene_names is not None:
        logger.info("Generating Panel C: Spatial Examples...")
        fig_c = plot_examples_panel(
            coords,
            expression,
            gene_names,
            df,
            spec,
            title="C. Representative Genes by Archetype",
        )
        caption_c = (
            "Panel C: Representative spatial expression patterns for each archetype. "
            "Each row shows example genes from one archetype. "
            "Color intensity indicates expression level."
        )
        save_panel_with_caption(fig_c, outdir / "C_examples_per_archetype.png", caption_c)
        plt.close(fig_c)
    elif precision_df is not None:
        logger.info("Generating Panel C: Marker Recovery...")
        fig_c = plot_marker_recovery_panel(precision_df)
        caption_c = (
            "Panel C: Precision@K for structured gene recovery. "
            "Genes are ranked by spatial score; bars show fraction that are truly structured."
        )
        save_panel_with_caption(fig_c, outdir / "C_marker_recovery.png", caption_c)
        plt.close(fig_c)
    else:
        logger.warning("Skipping Panel C: no coords/expression or precision_df provided")

    # Panel D: Pairwise
    if pairs_df is not None:
        logger.info("Generating Panel D: Gene-Gene Co-patterning...")
        fig_d = plot_pairwise_panel(pairs_df, spec)
        caption_d = (
            "Panel D: Distribution of gene-gene similarity scores. "
            "Higher scores indicate genes with similar spatial patterns."
        )
        save_panel_with_caption(fig_d, outdir / "D_pairwise_or_module.png", caption_d)
        plt.close(fig_d)
    else:
        logger.warning("Skipping Panel D: no pairs_df provided")

    logger.info(f"Full panel suite saved to {outdir}")
