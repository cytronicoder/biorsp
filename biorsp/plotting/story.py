"""
One-Page Story Figure Generator for BioRSP.

Generates a unified "one-page story" figure combining Panels A, B, C, D
into a single publication-ready figure. Works for both simulation benchmarks
and kidney case studies, automatically adapting panel content based on
available data.

Key Features:
- Consistent 2x2 or 1x4 layout
- Automatic panel substitution (confusion matrix vs composition bar)
- Caption generation from manifest metadata
- Supports regeneration from standardized outputs

Usage:
    from biorsp.plotting.story import generate_onepager

    generate_onepager(
        runs_csv="outputs/archetypes/runs.csv",
        manifest_json="outputs/archetypes/manifest.json",
        outdir="outputs/archetypes/figures",
    )
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from biorsp.plotting.panels import (
    plot_archetype_scatter,
    plot_composition_bar,
    plot_confusion_matrix,
    plot_pairwise_panel,
)
from biorsp.plotting.spec import PlotSpec, load_spec_from_manifest
from biorsp.plotting.style import add_panel_label, publication_style, save_figure

logger = logging.getLogger(__name__)


def detect_run_type(df: pd.DataFrame, manifest: Optional[Dict] = None) -> str:
    """
    Detect whether this run is simulation or kidney data.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    manifest : dict, optional
        Loaded manifest dictionary

    Returns
    -------
    run_type : str
        "simulation" or "kidney"
    """
    # Check for simulation-specific columns
    if "true_archetype" in df.columns:
        return "simulation"

    if "shape" in df.columns and "N" in df.columns:
        return "simulation"

    # Check manifest for hints
    if manifest:
        benchmark = manifest.get("benchmark", "").lower()
        if any(kw in benchmark for kw in ["archetype", "calibration", "robustness", "genegene"]):
            return "simulation"
        if any(kw in benchmark for kw in ["kpmp", "kidney", "tal", "disease"]):
            return "kidney"

    # Check for kidney-specific columns
    if any(col in df.columns for col in ["condition", "cell_type", "cluster", "donor"]):
        return "kidney"

    logger.warning("Could not detect run type, defaulting to kidney")
    return "kidney"


def generate_onepager(
    runs_csv: Union[str, Path],
    manifest_json: Optional[Union[str, Path]] = None,
    outdir: Optional[Union[str, Path]] = None,
    coords: Optional[np.ndarray] = None,
    expression: Optional[np.ndarray] = None,
    gene_names: Optional[List[str]] = None,
    pairs_df: Optional[pd.DataFrame] = None,
    group_by: Optional[str] = None,
    layout: str = "2x2",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
) -> Tuple[Figure, str]:
    """
    Generate a one-page story figure from standardized outputs.

    Parameters
    ----------
    runs_csv : str or Path
        Path to runs.csv with standardized columns
    manifest_json : str or Path, optional
        Path to manifest.json (for PlotSpec and metadata)
    outdir : str or Path, optional
        Output directory (default: same as runs_csv parent)
    coords : np.ndarray, optional
        Cell coordinates for Panel C (required for spatial examples)
    expression : np.ndarray, optional
        Expression matrix for Panel C (required for spatial examples)
    gene_names : list, optional
        Gene names for Panel C
    pairs_df : pd.DataFrame, optional
        Pairwise similarity scores for Panel D
    group_by : str, optional
        Column for grouping in Panel B (kidney mode)
    layout : str
        Layout mode: "2x2" (grid) or "1x4" (horizontal strip)
    figsize : tuple, optional
        Figure size (default: (12, 10) for 2x2, (16, 4) for 1x4)
    dpi : int
        Resolution for output

    Returns
    -------
    fig : Figure
        Combined figure with Panels A, B, C, D
    caption : str
        Generated caption text
    """
    runs_csv = Path(runs_csv)
    if not runs_csv.exists():
        raise FileNotFoundError(f"runs.csv not found: {runs_csv}")

    df = pd.read_csv(runs_csv)
    logger.info(f"Loaded {len(df)} rows from {runs_csv}")

    # Load manifest and spec
    manifest = {}
    if manifest_json:
        manifest_json = Path(manifest_json)
        if manifest_json.exists():
            with open(manifest_json) as f:
                manifest = json.load(f)
            logger.info(f"Loaded manifest from {manifest_json}")

    # Try to load PlotSpec from manifest, or use defaults
    try:
        if manifest_json and manifest_json.exists():
            spec = load_spec_from_manifest(str(manifest_json))
            logger.info(f"Loaded PlotSpec: c_cut={spec.c_cut}, s_cut={spec.s_cut}")
        else:
            spec = PlotSpec()
    except (KeyError, FileNotFoundError):
        spec = PlotSpec()
        logger.warning("Using default PlotSpec")

    # Detect run type
    run_type = detect_run_type(df, manifest)
    logger.info(f"Detected run type: {run_type}")

    # Classify if needed
    if spec.archetype_col not in df.columns:
        df = spec.classify_dataframe(df, inplace=False)

    # Set output directory
    if outdir is None:
        outdir = runs_csv.parent / "figures"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Set figure size based on layout
    if figsize is None:
        figsize = (12, 10) if layout == "2x2" else (16, 4)

    # Create figure
    with publication_style():
        if layout == "2x2":
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            ax_a, ax_b = axes[0, 0], axes[0, 1]
            ax_c, ax_d = axes[1, 0], axes[1, 1]
        else:  # 1x4
            fig, axes = plt.subplots(1, 4, figsize=figsize)
            ax_a, ax_b, ax_c, ax_d = axes

        # Panel A: Archetype Scatter
        logger.info("Generating Panel A: Archetype Scatter")
        color_by = (
            "true_archetype"
            if run_type == "simulation" and "true_archetype" in df.columns
            else spec.archetype_col
        )
        plot_archetype_scatter(df, spec, color_by=color_by, ax=ax_a, show_annotations=True)
        ax_a.set_title("A. Coverage vs Spatial Bias Score", fontsize=11, fontweight="bold")
        add_panel_label(ax_a, "A")

        # Panel B: Confusion Matrix or Composition
        logger.info("Generating Panel B")
        if run_type == "simulation" and "true_archetype" in df.columns:
            plot_confusion_matrix(df, spec, ax=ax_b)
            ax_b.set_title("B. Classification Confusion Matrix", fontsize=11, fontweight="bold")
        else:
            # Find grouping column
            if group_by is None:
                for candidate in ["condition", "cell_type", "cluster", "donor"]:
                    if candidate in df.columns:
                        group_by = candidate
                        break

            if group_by and group_by in df.columns:
                plot_composition_bar(df, spec, group_by=group_by, ax=ax_b)
                ax_b.set_title(
                    f"B. Archetype Composition by {group_by.title()}",
                    fontsize=11,
                    fontweight="bold",
                )
            else:
                # Fallback: show archetype distribution as pie chart
                counts = df[spec.archetype_col].value_counts()
                colors = [spec.get_color(a) for a in counts.index]
                ax_b.pie(counts.values, labels=counts.index, colors=colors, autopct="%1.0f%%")
                ax_b.set_title("B. Archetype Distribution", fontsize=11, fontweight="bold")

        add_panel_label(ax_b, "B")

        # Panel C: Examples or placeholder
        logger.info("Generating Panel C")
        if coords is not None and expression is not None and gene_names is not None:
            # We can't embed in existing axes, so show summary statistics
            _plot_archetype_summary_panel(df, spec, ax_c)
            ax_c.set_title("C. Archetype Summary Statistics", fontsize=11, fontweight="bold")
        else:
            # Show summary statistics as alternative
            _plot_archetype_summary_panel(df, spec, ax_c)
            ax_c.set_title("C. Archetype Summary Statistics", fontsize=11, fontweight="bold")
        add_panel_label(ax_c, "C")

        # Panel D: Pairwise or placeholder
        logger.info("Generating Panel D")
        if pairs_df is not None and len(pairs_df) > 0:
            plot_pairwise_panel(pairs_df, spec, ax=ax_d)
            ax_d.set_title("D. Gene-Gene Co-patterning", fontsize=11, fontweight="bold")
        else:
            _plot_score_distribution_panel(df, spec, ax_d)
            ax_d.set_title("D. Score Distributions", fontsize=11, fontweight="bold")
        add_panel_label(ax_d, "D")

        plt.tight_layout()

    # Generate caption
    caption = _generate_caption(df, spec, manifest, run_type)

    # Save outputs
    fig_path = outdir / "fig_story_onepager.png"
    save_figure(fig, str(fig_path), dpi=dpi, formats=["png", "pdf"], close=False)

    caption_path = outdir / "fig_story_onepager_caption.txt"
    with open(caption_path, "w") as f:
        f.write(caption)
    logger.info(f"Saved caption to {caption_path}")

    return fig, caption


def _plot_archetype_summary_panel(
    df: pd.DataFrame,
    spec: PlotSpec,
    ax: plt.Axes,
):
    """
    Plot summary statistics per archetype as bar charts.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    spec : PlotSpec
        Plot specification
    ax : Axes
        Matplotlib axes
    """
    archetypes = spec.get_legend_order()

    # Compute mean Coverage and Spatial_Bias_Score per archetype
    summary = df.groupby(spec.archetype_col).agg(
        {
            spec.coverage_col: ["mean", "std", "count"],
            spec.spatial_col: ["mean", "std"],
        }
    )

    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reindex(archetypes)

    x = np.arange(len(archetypes))
    width = 0.35

    # Plot Coverage bars
    coverage_means = summary[f"{spec.coverage_col}_mean"].fillna(0)
    coverage_stds = summary[f"{spec.coverage_col}_std"].fillna(0)

    # Plot Spatial score bars
    spatial_means = summary[f"{spec.spatial_col}_mean"].fillna(0)
    spatial_stds = summary[f"{spec.spatial_col}_std"].fillna(0)

    ax.bar(
        x - width / 2,
        coverage_means,
        width,
        yerr=coverage_stds,
        label="Coverage (C)",
        color="#2196F3",
        alpha=0.8,
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        spatial_means,
        width,
        yerr=spatial_stds,
        label="Spatial Score (S)",
        color="#FF5722",
        alpha=0.8,
        capsize=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(archetypes, rotation=45, ha="right")
    ax.set_ylabel("Score", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # Add counts as annotations
    counts = summary[f"{spec.coverage_col}_count"].fillna(0).astype(int)
    for i, (_arch, count) in enumerate(zip(archetypes, counts)):
        ax.annotate(
            f"n={count}",
            xy=(i, 0),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="gray",
        )


def _plot_score_distribution_panel(
    df: pd.DataFrame,
    spec: PlotSpec,
    ax: plt.Axes,
):
    """
    Plot distribution of Coverage and Spatial_Bias_Score as overlapping histograms.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    spec : PlotSpec
        Plot specification
    ax : Axes
        Matplotlib axes
    """
    coverage = df[spec.coverage_col].dropna()
    spatial = df[spec.spatial_col].dropna()

    ax.hist(coverage, bins=30, alpha=0.6, color="#2196F3", label="Coverage (C)", density=True)
    ax.hist(spatial, bins=30, alpha=0.6, color="#FF5722", label="Spatial Score (S)", density=True)

    c_cut, s_cut = spec.get_quadrant_bounds()
    ax.axvline(
        c_cut, color="#2196F3", linestyle="--", linewidth=2, alpha=0.8, label=f"C cut={c_cut:.2f}"
    )
    ax.axvline(
        s_cut, color="#FF5722", linestyle="--", linewidth=2, alpha=0.8, label=f"S cut={s_cut:.2f}"
    )

    ax.set_xlabel("Score", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def _generate_caption(
    df: pd.DataFrame,
    spec: PlotSpec,
    manifest: Dict,
    run_type: str,
) -> str:
    """
    Generate a caption for the one-page story figure.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    spec : PlotSpec
        Plot specification
    manifest : dict
        Manifest metadata
    run_type : str
        "simulation" or "kidney"

    Returns
    -------
    caption : str
        Multi-paragraph caption text
    """
    n_total = len(df)
    n_per_archetype = df[spec.archetype_col].value_counts().to_dict()
    c_cut, s_cut = spec.get_quadrant_bounds()

    # Get metadata from manifest
    benchmark = manifest.get("benchmark", "unknown")
    timestamp = manifest.get("timestamp", datetime.now().isoformat())
    git_hash = manifest.get("git_commit", "N/A")[:8] if manifest.get("git_commit") else "N/A"

    caption_lines = [
        f"Figure: BioRSP Analysis Summary ({run_type.title()})",
        "",
        f"(A) Coverage vs Spatial Bias Score scatter plot showing {n_total:,} ",
        "genes/replicates classified into four archetypes. Quadrant boundaries at ",
        f"C={c_cut:.2f}, S={s_cut:.2f}. ",
    ]

    if run_type == "simulation":
        caption_lines.extend(
            [
                "(B) Confusion matrix showing classification accuracy against ground truth. ",
                "(C) Summary statistics per archetype with mean ± std. ",
                "(D) Distribution of coverage and spatial scores with threshold markers. ",
            ]
        )
    else:
        caption_lines.extend(
            [
                "(B) Archetype composition across conditions or cell types. ",
                "(C) Summary statistics per archetype with gene counts. ",
                "(D) Score distributions showing the threshold values used for classification. ",
            ]
        )

    caption_lines.extend(
        [
            "",
            f"Archetype counts: {', '.join(f'{k}: {v}' for k, v in sorted(n_per_archetype.items()))}. ",
            f"Generated: {timestamp[:10]}. Git: {git_hash}. Benchmark: {benchmark}.",
        ]
    )

    return "\n".join(caption_lines)


def generate_onepager_from_dir(
    run_dir: Union[str, Path],
    output_name: str = "fig_story_onepager.png",
    **kwargs,
) -> Tuple[Figure, str]:
    """
    Convenience function to generate onepager from a run directory.

    Automatically locates runs.csv and manifest.json in the given directory.

    Parameters
    ----------
    run_dir : str or Path
        Directory containing runs.csv and manifest.json
    output_name : str
        Output filename (default: fig_story_onepager.png)
    **kwargs
        Additional arguments passed to generate_onepager()

    Returns
    -------
    fig : Figure
        Combined figure
    caption : str
        Generated caption
    """
    run_dir = Path(run_dir)

    # Find runs.csv
    runs_candidates = ["runs.csv", "results.csv", "gene_scores.csv", "scores.csv"]
    runs_csv = None
    for candidate in runs_candidates:
        if (run_dir / candidate).exists():
            runs_csv = run_dir / candidate
            break

    if runs_csv is None:
        raise FileNotFoundError(f"Could not find runs CSV in {run_dir}")

    # Find manifest.json
    manifest_json = run_dir / "manifest.json"
    if not manifest_json.exists():
        manifest_json = None

    return generate_onepager(
        runs_csv=runs_csv,
        manifest_json=manifest_json,
        outdir=run_dir / "figures",
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate one-page story figure")
    parser.add_argument("--runs", type=str, required=True, help="Path to runs.csv")
    parser.add_argument("--manifest", type=str, help="Path to manifest.json")
    parser.add_argument("--outdir", type=str, help="Output directory")
    parser.add_argument("--layout", choices=["2x2", "1x4"], default="2x2")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    fig, caption = generate_onepager(
        runs_csv=args.runs,
        manifest_json=args.manifest,
        outdir=args.outdir,
        layout=args.layout,
        dpi=args.dpi,
    )

    print("One-page story figure generated successfully!")
    print(f"\nCaption:\n{caption}")
