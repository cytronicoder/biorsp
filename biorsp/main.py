"""Canonical entry point for BioRSP."""

import time
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

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


def _process_feature(
    name: str,
    x: np.ndarray,
    r_norm: np.ndarray,
    theta: np.ndarray,
    umi_counts: Optional[np.ndarray],
    config: BioRSPConfig,
    seed: int,
) -> Optional[FeatureResult]:
    """Process a single feature."""
    rng = np.random.default_rng(seed)

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
        return None

    # Compute RSP
    adequacy = assess_adequacy(r_norm, theta, y, config=config, x=x)
    if not adequacy.is_adequate:
        logger.warning(f"Feature '{name}' is inadequate (reason: {adequacy.reason}). Skipping.")
        return None

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
    return FeatureResult(
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


def run(
    coords: np.ndarray,
    expression: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    umi_counts: Optional[np.ndarray] = None,
    config: Optional[BioRSPConfig] = None,
    outdir: Optional[str] = None,
    n_workers: int = 1,
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
    n_workers : int
        Number of workers for parallel processing of features.

    Returns
    -------
    RunSummary
        Object containing per-feature results and metadata.
    """
    start_time = time.time()
    config = config or BioRSPConfig()

    # Validation
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

    # Geometry
    logger.info("Computing vantage point and polar coordinates")
    vantage = compute_vantage(
        coords,
        method=config.vantage,
        tol=config.geom_median_tol,
        max_iter=config.geom_median_max_iter,
        knn_k=config.center_knn_k,
        density_percentile=config.center_density_percentile,
        seed=config.seed,
    )
    r, theta = polar_coordinates(coords, vantage)
    r_norm, norm_stats = normalize_radii(r)

    # Process features
    feature_results = {}
    rng = np.random.default_rng(config.seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_features)

    abstention_reasons = {}

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        logger.info(f"Processing features in parallel with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _process_feature,
                    name,
                    x_mat[:, i],
                    r_norm,
                    theta,
                    umi_counts,
                    config,
                    seeds[i],
                ): name
                for i, name in enumerate(feature_names)
            }

            for future in tqdm(futures, desc="BioRSP Features"):
                name = futures[future]
                try:
                    res = future.result()
                    if res is not None:
                        feature_results[name] = res
                    else:
                        abstention_reasons["unknown_or_underpowered"] = (
                            abstention_reasons.get("unknown_or_underpowered", 0) + 1
                        )
                except Exception as e:
                    logger.error(f"Error processing feature '{name}': {e}")
                    abstention_reasons["error"] = abstention_reasons.get("error", 0) + 1
    else:
        for i, name in enumerate(tqdm(feature_names, desc="BioRSP Features")):
            # Define foreground to get reason if it fails there
            y, fg_info = define_foreground(
                x_mat[:, i],
                mode=config.foreground_mode,
                q=config.foreground_quantile,
                abs_threshold=config.foreground_threshold,
                min_fg=config.min_fg_total,
                rng=np.random.default_rng(seeds[i]),
            )
            if y is None:
                reason = fg_info.get("status", "underpowered")
                abstention_reasons[reason] = abstention_reasons.get(reason, 0) + 1
                continue

            adequacy = assess_adequacy(r_norm, theta, y, config=config, x=x_mat[:, i])
            if not adequacy.is_adequate:
                reason = adequacy.reason
                abstention_reasons[reason] = abstention_reasons.get(reason, 0) + 1
                continue

            res = _process_feature(name, x_mat[:, i], r_norm, theta, umi_counts, config, seeds[i])
            if res is not None:
                feature_results[name] = res
            else:
                abstention_reasons["unknown"] = abstention_reasons.get("unknown", 0) + 1

    # Typing
    logger.info("Assigning feature types")
    feature_results, typing_thresholds = assign_feature_types(feature_results)

    end_time = time.time()
    duration = end_time - start_time

    total_abstained = sum(abstention_reasons.values())
    summary = RunSummary(
        feature_results=feature_results,
        config=config,
        metadata={
            "duration_seconds": duration,
            "n_cells": n_cells,
            "n_features": n_features,
            "abstention_count": total_abstained,
            "abstention_reasons": abstention_reasons,
        },
        typing_thresholds=typing_thresholds,
    )

    # Manifest and Output
    if outdir:
        import os

        os.makedirs(outdir, exist_ok=True)

        # Calculate coverage distribution
        coverages = [res.adequacy.adequacy_fraction for res in feature_results.values()]
        coverage_dist = {
            "mean": float(np.mean(coverages)) if coverages else 0.0,
            "median": float(np.median(coverages)) if coverages else 0.0,
            "min": float(np.min(coverages)) if coverages else 0.0,
            "max": float(np.max(coverages)) if coverages else 0.0,
        }

        manifest = create_manifest(
            parameters=config.to_dict(),
            seed=config.seed,
            dataset_summary={
                "n_cells": n_cells,
                "n_features": n_features,
                "n_adequate": len(feature_results),
                "abstention_rate": total_abstained / n_features if n_features > 0 else 0.0,
                "top_abstention_reasons": dict(
                    sorted(abstention_reasons.items(), key=lambda x: x[1], reverse=True)[:3]
                ),
                "coverage_distribution": coverage_dist,
            },
            timings={"total_duration": duration},
            extra_metadata={"abstention_reasons": abstention_reasons},
        )
        save_manifest(manifest, os.path.join(outdir, "run_metadata.json"))
        logger.info(f"Results and manifest saved to {outdir}")

    return summary
