"""Adapter utilities for standardized kidney plots.

This module delegates to `biorsp.plotting.standard.make_standard_plot_set` so
that kidney analyses produce the same canonical plot set as simulations.
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
    """Save a metadata manifest for kidney analysis.

    Args:
        output_dir: Directory to save `manifest.json`.
        metadata: Dictionary with analysis metadata (config, genes, thresholds).
        **kwargs: Additional metadata fields (e.g., params, n_genes, runtime_seconds).
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
    """Write a scores DataFrame to `runs.csv`.

    Args:
        output_dir: Directory to save `runs.csv`.
        scores_df: DataFrame with gene scores and metadata.
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
    """Generate the standardized plot set for kidney analysis.

    Args:
        scores_df: DataFrame with columns such as `gene`, `Spatial_Bias_Score`,
            `Coverage`, and optional abstention metadata.
        output_dir: Directory to save plots.
        labels: Optional ground-truth labels (unused for KPMP real data).
        plot_config: Optional plot configuration dictionary (deprecated).
        c_cut: Coverage threshold for archetype classification.
        s_cut: Spatial score threshold for archetype classification.

    Notes:
        The function delegates to `make_standard_plot_set` and writes the
        canonical figure names (`fig_cs_scatter.png`, `fig_cs_marginals.png`,
        `fig_confusion_or_composition.png`, `fig_archetype_examples.png`,
        `fig_top_tables.png`).
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
