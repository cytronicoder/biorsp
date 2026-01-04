"""
Statistical inference module for BioRSP.

Implements permutation testing for statistical significance:
- Stratified permutation of labels.
- Finite-permutation corrected p-values.
- FDR correction.
"""

import multiprocessing as mp
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .core import compute_anisotropy, compute_rsp_radar
from .stratification import get_strata_indices
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
) -> Tuple[float, int]:
    """Worker function for parallel permutation testing."""
    rng = np.random.default_rng(seed)

    # 1. Shuffle labels within strata
    y_perm = y.copy()
    for idx in strata_indices:
        if len(idx) > 1:
            shuffled_idx = rng.permutation(idx)
            y_perm[idx] = y_perm[shuffled_idx]

    # 2. Compute RSP with frozen mask
    radar_perm = compute_rsp_radar(
        r, theta, y_perm, config=config, adequacy=adequacy, frozen_mask=valid_mask
    )

    # 3. Count empty sectors (sectors in valid_mask that became empty under permutation)
    empty_count = np.sum(valid_mask & ((radar_perm.counts_fg == 0) | (radar_perm.counts_bg == 0)))

    # 4. Compute anisotropy
    return compute_anisotropy(radar_perm.rsp, valid_mask), int(empty_count)


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
    strata_indices = get_strata_indices(
        r=r,
        theta=theta,
        umi_counts=umi_counts,
        n_r_bins=config.n_r_bins,
        n_theta_bins=config.n_theta_bins,
        n_umi_bins=config.umi_bins,
        min_stratum_size=config.min_stratum_size,
        mode=config.perm_mode,
    )

    # 2. Observed statistic
    if adequacy is None:
        from .core import assess_adequacy

        adequacy = assess_adequacy(r, theta, y, config=config)

    radar_obs = compute_rsp_radar(r, theta, y, config=config, adequacy=adequacy)
    valid_mask = adequacy.sector_mask

    if not np.any(valid_mask):
        return InferenceResult(
            p_value=np.nan,
            observed_stat=np.nan,
            null_stats=np.full(n_perm, np.nan),
            valid_mask=valid_mask,
            perm_mode=config.perm_mode,
            K_eff=0,
            empty_sector_count=0,
        )

    observed_stat = compute_anisotropy(radar_obs.rsp, valid_mask)

    # 3. Permutations
    seeds = rng.integers(0, 2**31 - 1, size=n_perm)

    if n_workers > 1:
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
            results = list(
                tqdm(
                    pool.imap(worker_func, seeds),
                    total=n_perm,
                    disable=not show_progress,
                    desc="Permutations",
                )
            )
        null_stats = np.array([r[0] for r in results])
        empty_counts = np.array([r[1] for r in results])
    else:
        # Serial execution
        null_stats = np.full(n_perm, np.nan)
        empty_counts = np.zeros(n_perm, dtype=int)
        for i in tqdm(range(n_perm), disable=not show_progress, desc="Permutations"):
            null_stats[i], empty_counts[i] = _permutation_worker(
                seeds[i], r, theta, y, strata_indices, config, valid_mask, adequacy
            )

    # 4. P-value calculation (finite-permutation correction)
    # Filter out NaNs if any (though there shouldn't be any with the new logic)
    valid_nulls = null_stats[np.isfinite(null_stats)]
    K_eff = len(valid_nulls)

    if K_eff > 0:
        count_ge = np.sum(valid_nulls >= observed_stat)
        p_value = (count_ge + 1) / (K_eff + 1)
    else:
        p_value = np.nan

    return InferenceResult(
        p_value=float(p_value),
        observed_stat=observed_stat,
        null_stats=null_stats,
        valid_mask=valid_mask,
        seeds=seeds,
        perm_mode=config.perm_mode,
        K_eff=K_eff,
        empty_sector_count=int(np.sum(empty_counts)),
    )


def compute_diagnostic_null(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    config: BioRSPConfig = BioRSPConfig(),
    n_perm: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute diagnostic null by rotating embedding (shifting theta).

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

    # 1. Observed adequacy
    from .core import assess_adequacy

    adequacy = assess_adequacy(r, theta, y, config=config)
    valid_mask = adequacy.sector_mask

    if not np.any(valid_mask):
        return np.full(n_perm, np.nan)

    # 2. Rotations
    for i in range(n_perm):
        # Random rotation
        shift = rng.uniform(0, 2 * np.pi)
        theta_shifted = (theta + shift) % (2 * np.pi)

        # Recompute RSP with shifted theta
        # Note: sector_indices will be recomputed internally for theta_shifted
        radar_null = compute_rsp_radar(r, theta_shifted, y, config=config, frozen_mask=valid_mask)
        null_stats[i] = compute_anisotropy(radar_null.rsp, valid_mask)

    return null_stats


__all__ = [
    "compute_p_value",
    "compute_diagnostic_null",
    "bh_fdr",
]
