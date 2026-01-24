"""
Kidney Atlas Plotting Utilities - Thin wrapper around biorsp.plotting.standard.

This module provides adapter functions that delegate to the centralized
biorsp.plotting.standard.make_standard_plot_set to ensure kidney analyses
produce the same canonical figure set as simulations.

Usage:
    from analysis.kidney_atlas.utils.standardized_plotting import generate_kidney_panels

    generate_kidney_panels(
        scores_df=gene_scores_df,
        labels=cell_type_labels,  # Optional, None for composition plots
        output_dir=output_dir,
        plot_config={"show_truth": False},
    )
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from biorsp.plotting.standard import make_standard_plot_set

logger = logging.getLogger(__name__)


def save_kidney_manifest(
    output_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """
    Save metadata manifest for kidney analysis.

    Args:
        output_dir: Directory to save manifest.json
        metadata: Dictionary with analysis metadata (config, genes, thresholds, etc.)
        **kwargs: Additional metadata fields (params, n_genes, n_cells, runtime_seconds, etc.)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge metadata with kwargs
    manifest_data = metadata.copy() if metadata else {}
    manifest_data.update(kwargs)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"✓ Saved manifest to {manifest_path}")


def write_standardized_runs_csv(
    output_dir: Path,
    scores_df: pd.DataFrame,
) -> None:
    """
    Write scores DataFrame to standardized runs.csv format.

    Args:
        output_dir: Directory to save runs.csv
        scores_df: DataFrame with gene scores and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_csv = output_dir / "runs.csv"
    scores_df.to_csv(runs_csv, index=False)

    logger.info(f"✓ Saved runs.csv to {runs_csv}")


def generate_kidney_panels(
    scores_df: pd.DataFrame,
    output_dir: Path,
    labels: Optional[pd.Series] = None,
    plot_config: Optional[Dict[str, Any]] = None,
    c_cut: Optional[float] = None,
    s_cut: Optional[float] = None,
) -> None:
    """
    Generate standardized plot set for kidney analysis.

    This is a thin adapter that delegates to make_standard_plot_set from
    biorsp.plotting.standard, ensuring kidney plots match simulation plots.

    Args:
        scores_df: DataFrame with columns: gene, Spatial_Bias_Score, Coverage, abstain_flag
        output_dir: Directory to save plots
        labels: Optional ground truth labels (None for kidney real data)
        plot_config: Optional plot configuration dict (deprecated, use kwargs instead)
        c_cut: Coverage threshold for archetype classification
        s_cut: Spatial score threshold for archetype classification

    Creates:
        - fig_cs_scatter.png/pdf
        - fig_cs_marginals.png/pdf
        - fig_confusion_or_composition.png/pdf (composition if labels=None)
        - fig_archetype_examples.png/pdf (if gene details available)
        - fig_top_tables.png/pdf
    """
    logger.info(f"Generating standardized kidney plots in {output_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build thresholds dict
    thresholds = {}
    if c_cut is not None:
        thresholds["C_cut"] = c_cut
    if s_cut is not None:
        thresholds["S_cut"] = s_cut

    # Delegate to centralized plotting
    make_standard_plot_set(
        scores_df=scores_df,
        outdir=output_dir,
        thresholds=thresholds if thresholds else None,
        truth_col=None,  # No truth for real kidney data
        debug=False,
    )

    logger.info("✓ Kidney plot set complete")
