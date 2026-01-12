"""Statistical inference for BioRSP.

This module implements permutation testing to assess the significance of
observed spatial organization score S_g, using geometry-aware stratification
and finite-permutation corrected p-values.

Key fix: geometry support mask and weights are FIXED from observed data.
This avoids selection bias where geometry would change per permutation.
"""

import logging
import multiprocessing as mp
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from biorsp.core.adequacy import assess_adequacy
from biorsp.core.engine import compute_anisotropy, compute_rsp_radar
from biorsp.core.typing import AdequacyReport, InferenceResult
from biorsp.preprocess.stratification import get_strata_indices
from biorsp.utils.config import BioRSPConfig

logger = logging.getLogger(__name__)


def _permutation_worker(
    seed: int,
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    strata_indices: List[np.ndarray],
    config: BioRSPConfig,
    valid_mask: np.ndarray,
    sector_indices: List[np.ndarray],
    sector_weights: Optional[np.ndarray] = None,
) -> Tuple[float, int, bool]:
    """Worker function for parallel permutation testing.

    Key fix: uses FIXED valid_mask and sector_weights from observed data.

    Parameters
    ----------
    seed : int
        Random seed for this permutation.
    r : np.ndarray
        (N,) array of radial distances.
    theta : np.ndarray
        (N,) array of angles.
    y : np.ndarray
        (N,) foreground weights.
    strata_indices : List[np.ndarray]
        Indices for each stratum.
    config : BioRSPConfig
        Configuration object.
    valid_mask : np.ndarray
        (B,) FIXED boolean mask of valid sectors from observed data.
    sector_indices : List[np.ndarray]
        Precomputed sector indices.
    sector_weights : np.ndarray, optional
        FIXED precomputed sector weights from observed data.

    Returns
    -------
    Tuple[float, int, bool]
        (null_S_g, empty_sector_count, is_valid)

    """
    rng = np.random.default_rng(seed)

    y_perm = y.copy()
    for idx in strata_indices:
        if len(idx) > 1:
            shuffled_idx = rng.permutation(idx)
            y_perm[idx] = y_perm[shuffled_idx]

    radar_perm = compute_rsp_radar(
        r,
        theta,
        y_perm,
        config=config,
        sector_indices=sector_indices,
        frozen_mask=valid_mask,
        sector_weights=sector_weights,
    )

    is_valid = not np.any(valid_mask & ((radar_perm.counts_fg == 0) | (radar_perm.counts_bg == 0)))
    empty_count = np.sum(valid_mask & ((radar_perm.counts_fg == 0) | (radar_perm.counts_bg == 0)))

    # Compute S_g using fixed mask and weights
    return (
        compute_anisotropy(radar_perm.rsp, valid_mask, sector_weights),
        int(empty_count),
        bool(is_valid),
    )


def compute_p_value(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    config: Optional[BioRSPConfig] = None,
    n_perm: int = 1000,
    umi_counts: Optional[np.ndarray] = None,
    donor_labels: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = 42,
    n_workers: int = 1,
    show_progress: bool = True,
    adequacy: Optional[AdequacyReport] = None,
) -> InferenceResult:
    r"""Compute p-value for the observed spatial organization score S_g.

    Key fix: geometry support mask and weights are FIXED from observed data.
    This avoids selection bias where geometry would change per permutation.

    The p-value is computed using the finite-permutation correction:
    $$p = \frac{1 + \sum_{k=1}^K I(S_k \geq S_{obs})}{K + 1}$$
    where $S_{obs}$ is the observed spatial score and $S_k$ are the null scores.

    Parameters
    ----------
    r : np.ndarray
        (N,) array of normalized radial distances.
    theta : np.ndarray
        (N,) array of angles in radians.
    y : np.ndarray
        (N,) foreground weights or binary indicators.
    config : BioRSPConfig, optional
        Configuration object, by default None (creates default config).
    n_perm : int, optional
        Number of permutations, by default 1000.
    umi_counts : np.ndarray, optional
        (N,) array of UMI counts for stratification, by default None.
    donor_labels : np.ndarray, optional
        (N,) array of donor labels for mandatory stratification, by default None.
    rng : np.random.Generator, optional
        Random generator, by default None.
    seed : int, optional
        Seed for RNG if rng is None, by default 42.
    n_workers : int, optional
        Number of workers for multiprocessing, by default 1.
    show_progress : bool, optional
        Whether to show a progress bar, by default True.
    adequacy : AdequacyReport, optional
        Precomputed adequacy report, by default None.

    Returns
    -------
    InferenceResult
        The result of the permutation test.

    """
    if config is None:
        config = BioRSPConfig()
    if rng is None:
        rng = np.random.default_rng(seed)

    strata_indices = get_strata_indices(
        r=r,
        theta=theta,
        umi_counts=umi_counts,
        donor_labels=donor_labels if config.donor_stratify else None,
        n_r_bins=config.n_r_bins,
        n_theta_bins=config.n_theta_bins,
        n_umi_bins=config.umi_bins,
        min_stratum_size=config.min_stratum_size,
        mode=config.perm_mode,
    )

    if adequacy is None:
        adequacy = assess_adequacy(r, theta, y, config=config)

    radar_obs = compute_rsp_radar(
        r, theta, y, config=config, sector_indices=adequacy.sector_indices
    )

    # Use geom_supported_mask as the valid mask (fixed for all permutations)
    valid_mask = (
        radar_obs.geom_supported_mask
        if radar_obs.geom_supported_mask is not None
        else adequacy.sector_mask
    )

    if config.sector_weight_mode != "none":
        logger.info(
            f"Using sector weighting mode: {config.sector_weight_mode} (k={config.sector_weight_k})"
        )
        logger.info("Observed sector weights will be reused for all permutations.")

    if not np.any(valid_mask):
        return InferenceResult(
            p_value=np.nan,
            observed_stat=np.nan,
            null_stats=np.full(n_perm, np.nan),
            valid_mask=valid_mask,
            perm_mode=config.perm_mode,
            K_eff=0,
            rejection_count=0,
            empty_sector_count=0,
        )

    # Compute observed S_g using fixed mask and weights
    observed_stat = compute_anisotropy(radar_obs.rsp, valid_mask, radar_obs.sector_weights)

    null_stats = []
    all_seeds = []
    rejection_count = 0
    empty_sector_sum = 0
    total_attempts = 0
    max_total_attempts = n_perm * config.rejection_max_retries

    if n_workers > 1:
        worker_func = partial(
            _permutation_worker,
            r=r,
            theta=theta,
            y=y,
            strata_indices=strata_indices,
            config=config,
            valid_mask=valid_mask,
            sector_indices=adequacy.sector_indices,
            sector_weights=radar_obs.sector_weights,
        )

        import os

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        with mp.Pool(processes=n_workers) as pool:
            pbar = tqdm(
                total=n_perm, disable=not show_progress, desc="Permutations (Rejection-Resampling)"
            )
            while len(null_stats) < n_perm and total_attempts < max_total_attempts:
                batch_size = n_perm - len(null_stats)
                seeds = rng.integers(0, 2**31 - 1, size=batch_size)
                results = pool.map(worker_func, seeds)
                for i, (stat, empty_count, is_valid) in enumerate(results):
                    total_attempts += 1
                    if is_valid:
                        null_stats.append(stat)
                        all_seeds.append(seeds[i])
                        empty_sector_sum += empty_count
                        pbar.update(1)
                    else:
                        rejection_count += 1
                    if len(null_stats) >= n_perm:
                        break
            pbar.close()
    else:
        pbar = tqdm(
            total=n_perm, disable=not show_progress, desc="Permutations (Rejection-Resampling)"
        )
        while len(null_stats) < n_perm and total_attempts < max_total_attempts:
            total_attempts += 1
            seed_k = int(rng.integers(0, 2**31 - 1))
            stat, empty_count, is_valid = _permutation_worker(
                seed_k,
                r,
                theta,
                y,
                strata_indices,
                config,
                valid_mask,
                adequacy.sector_indices,
                radar_obs.sector_weights,
            )
            if is_valid:
                null_stats.append(stat)
                all_seeds.append(seed_k)
                empty_sector_sum += empty_count
                pbar.update(1)
            else:
                rejection_count += 1
        pbar.close()

    if len(null_stats) < n_perm:
        logger.warning(
            f"Failed to collect {n_perm} valid permutations after {max_total_attempts} attempts. "
            f"Collected {len(null_stats)}. Returning NaN p-value."
        )
        return InferenceResult(
            p_value=np.nan,
            observed_stat=observed_stat,
            null_stats=np.array(null_stats),
            valid_mask=valid_mask,
            seeds=np.array(all_seeds) if all_seeds else None,
            perm_mode=config.perm_mode,
            K_eff=len(null_stats),
            rejection_count=rejection_count,
            empty_sector_count=int(empty_sector_sum),
        )

    null_stats = np.array(null_stats)
    count_ge = np.sum(null_stats >= observed_stat)
    p_value = (count_ge + 1) / (n_perm + 1)

    return InferenceResult(
        p_value=float(p_value),
        observed_stat=observed_stat,
        null_stats=null_stats,
        valid_mask=valid_mask,
        seeds=np.array(all_seeds),
        perm_mode=config.perm_mode,
        K_eff=n_perm,
        rejection_count=rejection_count,
        empty_sector_count=int(empty_sector_sum),
    )


def compute_diagnostic_null(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    config: Optional[BioRSPConfig] = None,
    n_perm: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Compute diagnostic null by rotating embedding (shifting theta).

    This preserves the radial distribution but scrambles angular structure.
    Used to detect embedding-induced artifacts.

    Parameters
    ----------
    r : np.ndarray
        (N,) array of radial distances.
    theta : np.ndarray
        (N,) array of angles in radians.
    y : np.ndarray
        (N,) foreground weights.
    config : BioRSPConfig, optional
        Configuration object, by default None (creates default config).
    config : BioRSPConfig
        Configuration object.
    n_perm : int
        Number of rotations.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        (n_perm,) array of null statistics.

    """
    rng = np.random.default_rng(seed)
    null_stats = np.zeros(n_perm)

    adequacy = assess_adequacy(r, theta, y, config=config)
    valid_mask = adequacy.sector_mask

    if not np.any(valid_mask):
        return np.full(n_perm, np.nan)

    for i in range(n_perm):
        shift = rng.uniform(0, 2 * np.pi)
        theta_shifted = (theta + shift) % (2 * np.pi)

        radar_null = compute_rsp_radar(r, theta_shifted, y, config=config, frozen_mask=valid_mask)
        null_stats[i] = compute_anisotropy(radar_null.rsp, valid_mask)

    return null_stats
