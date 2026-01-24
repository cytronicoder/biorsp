"""
Re-plot panels from manifest without recomputation.

This CLI tool enables reproducible regeneration of figures from a completed
benchmark run using only the manifest.json and runs.csv files.

Key Features:
- Load plot_spec from manifest to ensure identical cutoffs/colors
- Regenerate A_archetype_scatter, B_confusion_or_composition panels
- Override output format (PNG/PDF/SVG)
- Apply style updates without re-running benchmarks

Usage:
    python -m biorsp.plotting.replot --manifest outputs/archetypes/manifest.json
    python -m biorsp.plotting.replot --manifest outputs/story/manifest.json --format pdf
    python -m biorsp.plotting.replot --manifest outputs/story/manifest.json --outdir figures/updated
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from biorsp.plotting.panels import (
    plot_archetype_scatter,
    plot_composition_bar,
    plot_confusion_matrix,
    save_panel_with_caption,
)
from biorsp.plotting.spec import PlotSpec, load_spec_from_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_results_dataframe(manifest_path: Path) -> pd.DataFrame:
    """Load a results DataFrame from the manifest directory.

    Args:
        manifest_path: Path to `manifest.json`.

    Returns:
        Results DataFrame with Coverage and Spatial_Bias_Score columns.

    Raises:
        FileNotFoundError: If no results CSV is found.
    """
    manifest_dir = manifest_path.parent

    candidates = [
        "runs.csv",
        "results.csv",
        "scores.csv",
        "gene_scores.csv",
        "archetypes.csv",
    ]

    for candidate in candidates:
        csv_path = manifest_dir / candidate
        if csv_path.exists():
            logger.info(f"Loading results from {csv_path}")
            return pd.read_csv(csv_path)

    figures_dir = manifest_dir / "figures"
    if figures_dir.exists():
        for candidate in candidates:
            csv_path = figures_dir / candidate
            if csv_path.exists():
                logger.info(f"Loading results from {csv_path}")
                return pd.read_csv(csv_path)

    raise FileNotFoundError(
        f"Could not find results CSV in {manifest_dir}. " f"Tried: {', '.join(candidates)}"
    )


def detect_mode(df: pd.DataFrame, manifest: dict) -> str:
    """Detect whether this is simulation or kidney data.

    Args:
        df: Results DataFrame.
        manifest: Loaded manifest.

    Returns:
        `"simulation"` or `"kidney"`.
    """
    if "true_archetype" in df.columns:
        return "simulation"

    benchmark = manifest.get("benchmark", "")
    if any(kw in benchmark.lower() for kw in ["simulation", "archetype", "calibration", "story"]):
        return "simulation"
    if any(kw in benchmark.lower() for kw in ["kpmp", "kidney", "tal", "disease"]):
        return "kidney"

    if "cell_type" in df.columns or "condition" in df.columns:
        return "kidney"

    logger.warning("Could not detect mode, defaulting to simulation")
    return "simulation"


def replot_from_manifest(
    manifest_path: str,
    outdir: Optional[str] = None,
    format: str = "png",
    dpi: int = 300,
    group_by: Optional[str] = None,
):
    """Regenerate panels from a manifest without recomputation.

    Args:
        manifest_path: Path to `manifest.json`.
        outdir: Output directory (default: manifest directory with `replot` suffix).
        format: Output format (`png`, `pdf`, `svg`).
        dpi: Resolution for raster formats.
        group_by: Grouping column for kidney mode.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    logger.info(f"Loaded manifest: benchmark={manifest.get('benchmark', 'unknown')}")

    try:
        spec = load_spec_from_manifest(str(manifest_path))
        logger.info(f"Loaded PlotSpec from manifest: c_cut={spec.c_cut}, s_cut={spec.s_cut}")
    except (KeyError, FileNotFoundError):
        logger.warning("No plot_spec in manifest, using defaults")
        spec = PlotSpec()

    df = load_results_dataframe(manifest_path)
    logger.info(f"Loaded {len(df)} rows from results CSV")

    column_renames = {}
    if "Spatial_Bias_Score" in df.columns and "Spatial_Bias_Score" not in df.columns:
        column_renames["Spatial_Bias_Score"] = spec.spatial_col
    if column_renames:
        df = df.rename(columns=column_renames)

    mode = detect_mode(df, manifest)
    logger.info(f"Detected mode: {mode}")

    output_dir = Path(outdir) if outdir else manifest_path.parent / "replot"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    if spec.archetype_col not in df.columns:
        logger.info("Classifying genes using PlotSpec...")
        df = spec.classify_dataframe(df, inplace=False)

    if mode == "simulation":
        _replot_simulation_panels(df, spec, output_dir, dpi)
    else:
        _replot_kidney_panels(df, spec, output_dir, dpi, group_by)

    replot_manifest = {
        "source_manifest": str(manifest_path),
        "plot_spec": spec.to_dict(),
        "mode": mode,
        "n_rows": len(df),
        "output_format": format,
        "dpi": dpi,
    }
    with open(output_dir / "replot_manifest.json", "w") as f:
        json.dump(replot_manifest, f, indent=2)

    logger.info(f"Replot complete. Figures saved to {output_dir}")


def _replot_simulation_panels(
    df: pd.DataFrame,
    spec: PlotSpec,
    output_dir: Path,
    dpi: int,
):
    """Generate simulation-specific panels.

    Args:
        df: Results DataFrame.
        spec: PlotSpec instance.
        output_dir: Output directory.
        dpi: Resolution in dots per inch.
    """
    import matplotlib.pyplot as plt

    color_by = "true_archetype" if "true_archetype" in df.columns else spec.archetype_col
    fig_a = plot_archetype_scatter(df, spec, color_by=color_by)

    caption_a = (
        f"Panel A: Coverage vs Spatial Bias Score scatter plot. "
        f"Quadrant boundaries at C={spec.c_cut:.2f}, S={spec.s_cut:.2f}. "
        f"Each point represents one gene or simulation replicate. "
        f"Colors indicate archetype classification."
    )
    save_panel_with_caption(fig_a, output_dir / "A_archetype_scatter.png", caption_a, dpi=dpi)
    plt.close(fig_a)

    if "true_archetype" in df.columns:
        fig_b = plot_confusion_matrix(df, spec)
        caption_b = (
            "Panel B: Confusion matrix showing classification performance. "
            "Rows represent true archetypes, columns represent predicted archetypes."
        )
        save_panel_with_caption(
            fig_b, output_dir / "B_confusion_or_composition.png", caption_b, dpi=dpi
        )
        plt.close(fig_b)

    logger.info("Simulation panels generated")


def _replot_kidney_panels(
    df: pd.DataFrame,
    spec: PlotSpec,
    output_dir: Path,
    dpi: int,
    group_by: Optional[str],
):
    """Generate kidney-specific panels.

    Args:
        df: Results DataFrame.
        spec: PlotSpec instance.
        output_dir: Output directory.
        dpi: Resolution in dots per inch.
        group_by: Optional grouping column.
    """
    import matplotlib.pyplot as plt

    fig_a = plot_archetype_scatter(df, spec)

    caption_a = (
        f"Panel A: Coverage vs Spatial Bias Score scatter plot. "
        f"Quadrant boundaries at C={spec.c_cut:.2f}, S={spec.s_cut:.2f}. "
        f"Colors indicate predicted archetype classification."
    )
    save_panel_with_caption(fig_a, output_dir / "A_archetype_scatter.png", caption_a, dpi=dpi)
    plt.close(fig_a)

    if group_by and group_by in df.columns:
        fig_b = plot_composition_bar(df, spec, group_by=group_by)
        caption_b = (
            f"Panel B: Archetype composition stratified by {group_by}. "
            "Stacked bars show fraction of genes in each archetype per group."
        )
        save_panel_with_caption(
            fig_b, output_dir / "B_confusion_or_composition.png", caption_b, dpi=dpi
        )
        plt.close(fig_b)
    else:
        candidate_cols = ["condition", "disease", "cell_type", "cluster", "sample"]
        for col in candidate_cols:
            if col in df.columns and df[col].nunique() >= 2:
                fig_b = plot_composition_bar(df, spec, group_by=col)
                caption_b = f"Panel B: Archetype composition stratified by {col}."
                save_panel_with_caption(
                    fig_b, output_dir / "B_confusion_or_composition.png", caption_b, dpi=dpi
                )
                plt.close(fig_b)
                break

    logger.info("Kidney panels generated")


def main():
    parser = argparse.ArgumentParser(
        description="Re-plot panels from manifest without recomputation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest.json from a completed benchmark run",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory (default: <manifest_dir>/replot)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for figures",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for raster formats",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default=None,
        help="Column for grouping in Panel B (kidney mode)",
    )

    args = parser.parse_args()

    try:
        replot_from_manifest(
            manifest_path=args.manifest,
            outdir=args.outdir,
            format=args.format,
            dpi=args.dpi,
            group_by=args.group_by,
        )
    except Exception as e:
        logger.error(f"Replot failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
