"""Canonical entry point for BioRSP."""

import time
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from biorsp.core.adequacy import assess_adequacy
from biorsp.core.engine import compute_rsp_radar
from biorsp.core.inference import compute_p_value
from biorsp.core.results import FeatureResult, RunSummary, assign_feature_types
from biorsp.core.summaries import compute_scalar_summaries
from biorsp.io.manifest import create_manifest, save_manifest
from biorsp.preprocess.foreground import define_foreground
from biorsp.preprocess.geometry import compute_vantage, polar_coordinates
from biorsp.preprocess.normalization import normalize_radii
from biorsp.utils.config import BioRSPConfig
from biorsp.utils.logging import get_logger
from biorsp.utils.validation import validate_inputs

logger = get_logger("main")


def run(
    coords: np.ndarray,
    expression: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    umi_counts: Optional[np.ndarray] = None,
    config: Optional[BioRSPConfig] = None,
    outdir: Optional[str] = None,
) -> RunSummary:
    """
    Run the full BioRSP pipeline on a dataset.

    Parameters
    ----------
    coords : np.ndarray
        (N, 2) array of spatial coordinates.
    expression : Union[pd.DataFrame, np.ndarray]
        (N, G) expression matrix.
    feature_names : Optional[List[str]]
        List of G feature names. If expression is a DataFrame, these are inferred.
    umi_counts : Optional[np.ndarray]
        (N,) array of total UMI counts per cell for stratified inference.
    config : Optional[BioRSPConfig]
        Configuration object. If None, defaults are used.
    outdir : Optional[str]
        Directory to save results and manifest.

    Returns
    -------
    RunSummary
        Object containing per-feature results and metadata.
    """
    start_time = time.time()
    config = config or BioRSPConfig()

    # 1. Validation
    validate_inputs(coords, expression, umi_counts)

    if isinstance(expression, pd.DataFrame):
        feature_names = expression.columns.tolist()
        x_mat = expression.values
    else:
        x_mat = expression
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(x_mat.shape[1])]

    n_cells, n_features = x_mat.shape
    logger.info(f"Starting BioRSP run on {n_cells} cells and {n_features} features")

    # 2. Geometry
    logger.info("Computing vantage point and polar coordinates")
    vantage = compute_vantage(coords, method=config.vantage)
    r, theta = polar_coordinates(coords, vantage)
    r_norm, norm_stats = normalize_radii(r)

    # 3. Process features
    feature_results = {}
    rng = np.random.default_rng(config.seed)

    abstention_counts = 0

    for i, name in enumerate(feature_names):
        x = x_mat[:, i]

        # Define foreground
        y, fg_info = define_foreground(
            x,
            mode=config.foreground_mode,
            q=config.foreground_quantile,
            abs_threshold=config.foreground_threshold,
            min_fg=config.min_fg_total,
            rng=rng,
        )

        if y is None:
            logger.warning(
                f"Feature '{name}' is underpowered (status: {fg_info.get('status')}). Skipping."
            )
            abstention_counts += 1
            continue

        # Compute RSP
        adequacy = assess_adequacy(r_norm, theta, y, config=config)
        if not adequacy.is_adequate:
            logger.warning(f"Feature '{name}' is inadequate (reason: {adequacy.reason}). Skipping.")
            abstention_counts += 1
            continue

        radar = compute_rsp_radar(
            r_norm,
            theta,
            y,
            config=config,
            sector_indices=adequacy.sector_indices,
            frozen_mask=adequacy.sector_mask,
        )

        # Inference
        inference = compute_p_value(
            r_norm, theta, y, config=config, umi_counts=umi_counts, adequacy=adequacy, rng=rng
        )

        # Summaries
        summaries = compute_scalar_summaries(radar)
        feature_results[name] = FeatureResult(
            feature=name,
            threshold_quantile=config.foreground_quantile,
            coverage_quantile=fg_info.get("q", 0.0),
            coverage_prevalence=fg_info.get("target_frac", 0.0),
            adequacy=adequacy,
            summaries=summaries,
            foreground_info=fg_info,
            radar=radar,
            p_value=inference.p_value,
            perm_mode=inference.perm_mode,
            K_eff=inference.K_eff,
            empty_sector_count=inference.empty_sector_count,
            sector_weight_mode=config.sector_weight_mode,
            sector_weight_k=config.sector_weight_k,
        )

    # 4. Typing
    logger.info("Assigning feature types")
    feature_results, typing_thresholds = assign_feature_types(feature_results)

    end_time = time.time()
    duration = end_time - start_time

    summary = RunSummary(
        feature_results=feature_results,
        config=config,
        metadata={
            "duration_seconds": duration,
            "n_cells": n_cells,
            "n_features": n_features,
            "abstention_count": abstention_counts,
        },
        typing_thresholds=typing_thresholds,
    )

    # 5. Manifest and Output
    if outdir:
        import os

        os.makedirs(outdir, exist_ok=True)

        manifest = create_manifest(
            parameters=config.to_dict(),
            seed=config.seed,
            dataset_summary={
                "n_cells": n_cells,
                "n_features": n_features,
                "n_foreground_avg": np.mean(
                    [np.sum(feature_results[n].radar.counts_fg) for n in feature_names]
                ),
            },
            timings={"total_duration": duration},
            extra_metadata={"abstention_count": abstention_counts},
        )
        save_manifest(manifest, os.path.join(outdir, "run_metadata.json"))
        logger.info(f"Results and manifest saved to {outdir}")

    return summary
