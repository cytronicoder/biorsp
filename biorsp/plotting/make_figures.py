"""
CLI entry point for regenerating figures from BioRSP outputs.

This module provides a command-line interface for re-plotting figures from
completed benchmark runs without recomputation. It detects the run type
(simulation vs. kidney) and generates appropriate panels.

Usage:
    python -m biorsp.plotting.make_figures --indir <run_dir>
    python -m biorsp.plotting.make_figures --indir <run_dir> --debug
    python -m biorsp.plotting.make_figures --indir <run_dir> --panels A B C D
    python -m biorsp.plotting.make_figures --indir <run_dir> --format pdf --dpi 600

The tool will:
1. Detect whether the run is simulation or kidney from manifest/data
2. Load PlotSpec from manifest for consistent cutoffs
3. Generate standard panels (A, B, C, D)
4. Optionally generate debug plots

Standard Panels:
- A_archetype_scatter.png: Coverage vs. Spatial Bias Score
- B_confusion_or_composition.png: Confusion matrix (sim) or composition bar (kidney)
- C_examples_per_archetype.png: Representative spatial patterns
- D_pairwise_or_module.png: Gene-gene pairs or modules

Debug Plots (--debug flag):
- debug/debug_cutoff_consistency.png
- debug/debug_score_distributions.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_runs_csv(run_dir: Path) -> Path:
    """Find the runs/results CSV file in a run directory.

    Args:
        run_dir: Run directory to search.

    Returns:
        Path to the runs/results CSV file.

    Raises:
        FileNotFoundError: If no candidate file is found.
    """
    candidates = [
        "runs.csv",
        "results.csv",
        "gene_scores.csv",
        "scores.csv",
        "archetypes.csv",
    ]

    for candidate in candidates:
        path = run_dir / candidate
        if path.exists():
            return path

    # Check in figures subdirectory
    figures_dir = run_dir / "figures"
    if figures_dir.exists():
        for candidate in candidates:
            path = figures_dir / candidate
            if path.exists():
                return path

    raise FileNotFoundError(f"Could not find runs CSV in {run_dir}. Tried: {', '.join(candidates)}")


def find_manifest(run_dir: Path) -> Optional[Path]:
    """Find `manifest.json` in a run directory.

    Args:
        run_dir: Run directory to search.

    Returns:
        Path to `manifest.json` if present, otherwise None.
    """
    candidates = [
        run_dir / "manifest.json",
        run_dir / "figures" / "manifest.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def detect_run_type(df: pd.DataFrame, manifest: dict) -> str:
    """Detect whether this is simulation or kidney data.

    Args:
        df: Results DataFrame.
        manifest: Loaded manifest dictionary.

    Returns:
        `"simulation"` or `"kidney"`.
    """
    # Check for simulation-specific columns
    if "true_archetype" in df.columns:
        return "simulation"

    if "shape" in df.columns and "N" in df.columns:
        return "simulation"

    # Check manifest
    benchmark = manifest.get("benchmark", "").lower()
    if any(kw in benchmark for kw in ["archetype", "calibration", "robustness", "genegene"]):
        return "simulation"
    if any(kw in benchmark for kw in ["kpmp", "kidney", "tal", "disease"]):
        return "kidney"

    # Check for kidney-specific columns
    if any(col in df.columns for col in ["condition", "cell_type", "cluster", "donor"]):
        return "kidney"

    return "kidney"


def generate_panel_a(
    df: pd.DataFrame,
    spec,
    outdir: Path,
    run_type: str,
    dpi: int = 300,
):
    """Generate Panel A: archetype scatter.

    Args:
        df: Results DataFrame.
        spec: PlotSpec instance.
        outdir: Output directory.
        run_type: `simulation` or `kidney`.
        dpi: Resolution in dots per inch.
    """
    from biorsp.plotting.panels import plot_archetype_scatter, save_panel_with_caption

    color_by = (
        "true_archetype"
        if run_type == "simulation" and "true_archetype" in df.columns
        else spec.archetype_col
    )

    fig = plot_archetype_scatter(
        df,
        spec,
        title="Coverage vs. Spatial Bias Score",
        color_by=color_by,
        show_annotations=True,
    )

    c_cut, s_cut = spec.get_quadrant_bounds()
    caption = (
        f"Panel A: Coverage vs. Spatial Bias Score scatter plot. "
        f"Quadrant boundaries at C={c_cut:.2f}, S={s_cut:.2f}. "
        f"Each point represents one gene or simulation replicate. "
        f"Colors indicate archetype classification."
    )

    save_panel_with_caption(fig, outdir / "A_archetype_scatter.png", caption, dpi=dpi)
    plt.close(fig)
    logger.info("Generated Panel A: A_archetype_scatter.png")


def generate_panel_b(
    df: pd.DataFrame,
    spec,
    outdir: Path,
    run_type: str,
    group_by: Optional[str] = None,
    dpi: int = 300,
):
    """Generate Panel B: confusion matrix or composition bar.

    Args:
        df: Results DataFrame.
        spec: PlotSpec instance.
        outdir: Output directory.
        run_type: `simulation` or `kidney`.
        group_by: Optional grouping column for kidney mode.
        dpi: Resolution in dots per inch.
    """
    from biorsp.plotting.panels import (
        plot_composition_bar,
        plot_confusion_matrix,
        save_panel_with_caption,
    )

    if run_type == "simulation" and "true_archetype" in df.columns:
        fig = plot_confusion_matrix(df, spec)
        caption = (
            "Panel B: Confusion matrix showing classification performance. "
            "Rows represent true archetypes, columns represent predicted archetypes. "
            "Values are normalized by true archetype (row sums to 1)."
        )
    else:
        # Find grouping column
        if group_by is None:
            for candidate in ["condition", "cell_type", "cluster", "donor", "disease"]:
                if candidate in df.columns:
                    group_by = candidate
                    break

        if group_by and group_by in df.columns:
            fig = plot_composition_bar(df, spec, group_by=group_by)
            caption = (
                f"Panel B: Archetype composition stratified by {group_by}. "
                "Stacked bars show the fraction of genes in each archetype for each group."
            )
        else:
            # Fallback: archetype distribution pie chart
            fig, ax = plt.subplots(figsize=(6, 5))
            counts = df[spec.archetype_col].value_counts()
            colors = [spec.get_color(a) for a in counts.index]
            ax.pie(counts.values, labels=counts.index, colors=colors, autopct="%1.0f%%")
            ax.set_title("Archetype Distribution")
            caption = "Panel B: Distribution of genes across archetypes."

    save_panel_with_caption(fig, outdir / "B_confusion_or_composition.png", caption, dpi=dpi)
    plt.close(fig)
    logger.info("Generated Panel B: B_confusion_or_composition.png")


def generate_panel_c_summary(
    df: pd.DataFrame,
    spec,
    outdir: Path,
    dpi: int = 300,
):
    """Generate Panel C: summary statistics when spatial data are unavailable.

    Args:
        df: Results DataFrame.
        spec: PlotSpec instance.
        outdir: Output directory.
        dpi: Resolution in dots per inch.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    import numpy as np

    archetypes = spec.get_legend_order()

    # Compute mean Coverage and Spatial_Bias_Score per archetype
    summary = df.groupby(spec.archetype_col).agg(
        {
            spec.coverage_col: ["mean", "std", "count"],
            spec.spatial_col: ["mean", "std"],
        }
    )

    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reindex(archetypes)

    x = np.arange(len(archetypes))
    width = 0.35

    coverage_means = summary[f"{spec.coverage_col}_mean"].fillna(0)
    coverage_stds = summary[f"{spec.coverage_col}_std"].fillna(0)
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
    ax.set_ylabel("Score")
    ax.set_title("C. Archetype Summary Statistics")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # Add counts
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

    plt.tight_layout()

    caption = (
        "Panel C: Summary statistics per archetype. "
        "Bars show mean values with standard deviation error bars. "
        "Counts indicate number of genes/replicates in each archetype."
    )

    from biorsp.plotting.panels import save_panel_with_caption

    save_panel_with_caption(fig, outdir / "C_examples_per_archetype.png", caption, dpi=dpi)
    plt.close(fig)
    logger.info("Generated Panel C: C_examples_per_archetype.png (summary mode)")


def generate_panel_d_distribution(
    df: pd.DataFrame,
    spec,
    outdir: Path,
    dpi: int = 300,
):
    """Generate Panel D: score distributions when pairwise data are unavailable.

    Args:
        df: Results DataFrame.
        spec: PlotSpec instance.
        outdir: Output directory.
        dpi: Resolution in dots per inch.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    coverage = df[spec.coverage_col].dropna()
    spatial = df[spec.spatial_col].dropna()

    ax.hist(coverage, bins=30, alpha=0.6, color="#2196F3", label="Coverage (C)", density=True)
    ax.hist(spatial, bins=30, alpha=0.6, color="#FF5722", label="Spatial Score (S)", density=True)

    c_cut, s_cut = spec.get_quadrant_bounds()
    ax.axvline(c_cut, color="#2196F3", linestyle="--", linewidth=2, alpha=0.8)
    ax.axvline(s_cut, color="#FF5722", linestyle="--", linewidth=2, alpha=0.8)

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("D. Score Distributions")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    caption = (
        f"Panel D: Distribution of coverage and spatial bias scores. "
        f"Vertical lines show classification thresholds (C={c_cut:.2f}, S={s_cut:.2f})."
    )

    from biorsp.plotting.panels import save_panel_with_caption

    save_panel_with_caption(fig, outdir / "D_pairwise_or_module.png", caption, dpi=dpi)
    plt.close(fig)
    logger.info("Generated Panel D: D_pairwise_or_module.png (distribution mode)")


def generate_debug_plots(
    df: pd.DataFrame,
    spec,
    outdir: Path,
    dpi: int = 150,
):
    """Generate debug plots."""
    from biorsp.plotting.debug import plot_debug_cutoff_consistency

    debug_dir = outdir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Cutoff consistency plot
    fig = plot_debug_cutoff_consistency(df, spec, title="Cutoff Consistency Check")
    fig.savefig(debug_dir / "debug_cutoff_consistency.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Generated debug_cutoff_consistency.png")

    # Score distributions with detailed annotations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Coverage distribution
    ax = axes[0]
    coverage = df[spec.coverage_col].dropna()
    ax.hist(coverage, bins=50, color="#2196F3", alpha=0.7, edgecolor="black")
    c_cut, _ = spec.get_quadrant_bounds()
    ax.axvline(c_cut, color="red", linestyle="--", linewidth=2, label=f"Cutoff = {c_cut:.2f}")
    ax.set_xlabel("Coverage (C)")
    ax.set_ylabel("Count")
    ax.set_title(f"Coverage Distribution (n={len(coverage)})")
    ax.legend()

    # Spatial score distribution
    ax = axes[1]
    spatial = df[spec.spatial_col].dropna()
    ax.hist(spatial, bins=50, color="#FF5722", alpha=0.7, edgecolor="black")
    _, s_cut = spec.get_quadrant_bounds()
    ax.axvline(s_cut, color="red", linestyle="--", linewidth=2, label=f"Cutoff = {s_cut:.2f}")
    ax.set_xlabel("Spatial Bias Score (S)")
    ax.set_ylabel("Count")
    ax.set_title(f"Spatial Score Distribution (n={len(spatial)})")
    ax.legend()

    plt.tight_layout()
    fig.savefig(debug_dir / "debug_score_distributions.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Generated debug_score_distributions.png")


def make_figures(
    indir: str,
    outdir: Optional[str] = None,
    panels: Optional[List[str]] = None,
    debug: bool = False,
    format: str = "png",
    dpi: int = 300,
    group_by: Optional[str] = None,
):
    """Generate figures from a completed BioRSP run.

    Args:
        indir: Input directory containing `runs.csv` and optional `manifest.json`.
        outdir: Output directory (default: `indir/figures`).
        panels: List of panels to generate (default: all).
        debug: Whether to generate debug plots.
        format: Output format (`png`, `pdf`, `svg`).
        dpi: Resolution for raster formats.
        group_by: Column for grouping in Panel B (kidney mode).
    """
    from biorsp.plotting.spec import PlotSpec, load_spec_from_manifest

    indir = Path(indir)
    if not indir.exists():
        raise FileNotFoundError(f"Input directory not found: {indir}")

    # Find files
    runs_csv = find_runs_csv(indir)
    manifest_path = find_manifest(indir)

    logger.info(f"Input directory: {indir}")
    logger.info(f"Found runs CSV: {runs_csv}")
    logger.info(f"Found manifest: {manifest_path}")

    # Load data
    df = pd.read_csv(runs_csv)
    logger.info(f"Loaded {len(df)} rows")

    # Load manifest
    manifest = {}
    if manifest_path:
        with open(manifest_path) as f:
            manifest = json.load(f)

    # Load PlotSpec
    try:
        if manifest_path:
            spec = load_spec_from_manifest(str(manifest_path))
            logger.info(f"Loaded PlotSpec: c_cut={spec.c_cut}, s_cut={spec.s_cut}")
        else:
            spec = PlotSpec()
            logger.warning("Using default PlotSpec")
    except (KeyError, FileNotFoundError):
        spec = PlotSpec()
        logger.warning("Using default PlotSpec")

    # Classify if needed
    if spec.archetype_col not in df.columns:
        df = spec.classify_dataframe(df, inplace=False)

    # Detect run type
    run_type = detect_run_type(df, manifest)
    logger.info(f"Detected run type: {run_type}")

    # Set output directory
    output_dir = indir / "figures" if outdir is None else Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Determine which panels to generate
    if panels is None:
        panels = ["A", "B", "C", "D"]
    panels = [p.upper() for p in panels]

    # Generate panels
    if "A" in panels:
        generate_panel_a(df, spec, output_dir, run_type, dpi)

    if "B" in panels:
        generate_panel_b(df, spec, output_dir, run_type, group_by, dpi)

    if "C" in panels:
        generate_panel_c_summary(df, spec, output_dir, dpi)

    if "D" in panels:
        generate_panel_d_distribution(df, spec, output_dir, dpi)

    # Generate debug plots
    if debug:
        generate_debug_plots(df, spec, output_dir, dpi=150)

    # Generate one-pager
    from biorsp.plotting.story import generate_onepager

    try:
        fig, caption = generate_onepager(
            runs_csv=runs_csv,
            manifest_json=manifest_path,
            outdir=output_dir,
            dpi=dpi,
        )
        plt.close(fig)
        logger.info("Generated fig_story_onepager.png")
    except Exception as e:
        logger.warning(f"Could not generate onepager: {e}")

    logger.info(f"All figures saved to {output_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate figures from BioRSP outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m biorsp.plotting.make_figures --indir outputs/archetypes
    python -m biorsp.plotting.make_figures --indir outputs/archetypes --debug
    python -m biorsp.plotting.make_figures --indir outputs/archetypes --panels A B
    python -m biorsp.plotting.make_figures --indir results/kpmp --format pdf
""",
    )

    parser.add_argument(
        "--indir",
        type=str,
        required=True,
        help="Input directory containing runs.csv and manifest.json",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory (default: indir/figures)",
    )
    parser.add_argument(
        "--panels",
        type=str,
        nargs="+",
        default=None,
        help="Panels to generate: A, B, C, D (default: all)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate debug plots",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for raster formats (default: 300)",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default=None,
        help="Column for grouping in Panel B (kidney mode)",
    )

    args = parser.parse_args()

    try:
        make_figures(
            indir=args.indir,
            outdir=args.outdir,
            panels=args.panels,
            debug=args.debug,
            format=args.format,
            dpi=args.dpi,
            group_by=args.group_by,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
