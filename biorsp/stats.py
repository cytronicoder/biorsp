"""
Statistical inference module for BioRSP.

Implements permutation testing for statistical significance:
- Stratified permutation of labels.
- Finite-permutation corrected p-values.
- FDR correction.
"""

import multiprocessing as mp
from functools import partial
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from .core import compute_anisotropy, compute_rsp_radar
from .typing import AdequacyReport, BioRSPConfig, InferenceResult
from .utils import bh_fdr


def _permutation_worker(
    seed: int,
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    strata_indices: List[np.ndarray],
    config: BioRSPConfig,
    valid_mask: np.ndarray,
    adequacy: AdequacyReport,
) -> float:
    """Worker function for parallel permutation testing."""
    rng = np.random.default_rng(seed)

    # 1. Shuffle labels within strata
    y_perm = y.copy()
    for idx in strata_indices:
        shuffled_idx = rng.permutation(idx)
        y_perm[idx] = y_perm[shuffled_idx]

    # 2. Compute RSP
    radar_perm = compute_rsp_radar(r, theta, y_perm, config=config, adequacy=adequacy)

    # 3. Compute anisotropy
    return compute_anisotropy(radar_perm.rsp, valid_mask)


def compute_p_value(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    config: BioRSPConfig = BioRSPConfig(),
    n_perm: int = 1000,
    umi_counts: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = 42,
    n_workers: int = 1,
    show_progress: bool = True,
    adequacy: Optional[AdequacyReport] = None,
) -> InferenceResult:
    """
    Compute p-value for the observed anisotropy using a permutation test.

    Parameters
    ----------
    r : np.ndarray
        (N,) array of normalized radial distances.
    theta : np.ndarray
        (N,) array of angles in radians.
    y : np.ndarray
        (N,) foreground weights or binary indicators.
    config : BioRSPConfig, optional
        Configuration object, by default BioRSPConfig().
    n_perm : int, optional
        Number of permutations, by default 1000.
    umi_counts : np.ndarray, optional
        (N,) array of UMI counts for stratification, by default None.
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
    if rng is None:
        rng = np.random.default_rng(seed)

    # 1. Prepare stratification
    n_cells = len(y)
    if umi_counts is not None:
        # Create deterministic strata using ranks
        if n_cells < config.umi_bins:
            strata = np.zeros(n_cells, dtype=int)
        else:
            ranks = np.argsort(np.argsort(umi_counts))
            strata = (ranks * config.umi_bins) // n_cells
    else:
        strata = np.zeros(n_cells, dtype=int)

    unique_strata = np.unique(strata)
    strata_indices = [np.where(strata == s)[0] for s in unique_strata]

    # 2. Observed statistic
    if adequacy is None:
        from .core import assess_adequacy

        adequacy = assess_adequacy(r, theta, y, config=config)

    radar_obs = compute_rsp_radar(r, theta, y, config=config, adequacy=adequacy)
    valid_mask = ~np.isnan(radar_obs.rsp)

    if not np.any(valid_mask):
        return InferenceResult(
            p_value=np.nan,
            observed_stat=np.nan,
            null_stats=np.full(n_perm, np.nan),
            valid_mask=valid_mask,
        )

    observed_stat = compute_anisotropy(radar_obs.rsp, valid_mask)

    # 3. Permutations
    seeds = rng.integers(0, 2**31 - 1, size=n_perm)

    if n_workers > 1:
        # Use multiprocessing
        # Note: We pass large arrays via global-ish or shared memory if possible,
        # but for now we rely on fork (on Unix) or pickling.
        worker_func = partial(
            _permutation_worker,
            r=r,
            theta=theta,
            y=y,
            strata_indices=strata_indices,
            config=config,
            valid_mask=valid_mask,
            adequacy=adequacy,
        )

        with mp.Pool(processes=n_workers) as pool:
            null_stats_list = list(
                tqdm(
                    pool.imap(worker_func, seeds),
                    total=n_perm,
                    disable=not show_progress,
                    desc="Permutations",
                )
            )
        null_stats = np.array(null_stats_list)
    else:
        # Serial execution
        null_stats = np.full(n_perm, np.nan)
        for i in tqdm(range(n_perm), disable=not show_progress, desc="Permutations"):
            null_stats[i] = _permutation_worker(
                seeds[i], r, theta, y, strata_indices, config, valid_mask, adequacy
            )

    # 4. P-value calculation (finite-permutation correction)
    # p = (count(null >= obs) + 1) / (n_perm + 1)
    count_ge = np.sum(null_stats >= observed_stat)
    p_value = (count_ge + 1) / (n_perm + 1)

    return InferenceResult(
        p_value=float(p_value),
        observed_stat=observed_stat,
        null_stats=null_stats,
        valid_mask=valid_mask,
        seeds=seeds,
    )


__all__ = [
    "compute_p_value",
    "bh_fdr",
]
