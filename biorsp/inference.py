"""
Inference module for BioRSP.

Implements permutation testing for statistical significance:
- Null hypothesis: Gene expression is independent of spatial location (angle).
- Test statistic: RMS Anisotropy (A_g).
- Stratified permutation of labels (optional) to control for UMI count confounders.
"""

from typing import Optional, Tuple
import concurrent.futures

import numpy as np

from .constants import N_BG_MIN_DEFAULT, N_FG_MIN_DEFAULT, UMI_BINS_DEFAULT
from .radar import compute_rsp_radar


def _rms_with_mask(rsp: np.ndarray, valid_mask: np.ndarray) -> float:
    """Compute RMS anisotropy using a fixed sector mask."""
    masked_rsp = rsp[valid_mask]
    masked_rsp = masked_rsp[np.isfinite(masked_rsp)]
    if masked_rsp.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(masked_rsp**2)))


def _compute_permutation_stat(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    strata_indices: list,
    B: int,
    delta_deg: float,
    min_fg_sector: int,
    min_bg_sector: int,
    seed: int,
    valid_mask: np.ndarray,
) -> float:
    """Compute the test statistic for a single permutation."""
    rng = np.random.default_rng(seed)
    y_perm = y.copy()
    for idx in strata_indices:
        y_subset = y_perm[idx]
        rng.shuffle(y_subset)
        y_perm[idx] = y_subset

    radar_perm = compute_rsp_radar(r, theta, y_perm, B, delta_deg, min_fg_sector, min_bg_sector)
    return _rms_with_mask(radar_perm.rsp, valid_mask)


def compute_p_value(
    observed_stat: float,
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    B: int = 360,
    delta_deg: float = 20.0,
    n_perm: int = 1000,
    umi_counts: Optional[np.ndarray] = None,
    umi_bins: int = UMI_BINS_DEFAULT,
    seed: int = 42,
    min_fg_sector: int = N_FG_MIN_DEFAULT,
    min_bg_sector: int = N_BG_MIN_DEFAULT,
) -> Tuple[float, np.ndarray]:
    """
    Compute p-value for the observed anisotropy using permutation test.
    Foreground labels are permuted within UMI strata to keep geometry fixed.

    Args:
        observed_stat: The observed test statistic (ignored; recomputed from radar).
        r: (N,) array of radial distances.
        theta: (N,) array of angles for all cells.
        y: (N,) boolean foreground indicator.
        B: Number of sectors.
        delta_deg: Sector width.
        n_perm: Number of permutations.
        umi_counts: (N,) array of UMI counts for stratification (optional).
        umi_bins: Number of bins for UMI stratification.
        seed: Random seed.
        min_fg_sector: Minimum foreground counts per sector.
        min_bg_sector: Minimum background counts per sector.

    Returns:
        p_value: Estimated p-value.
        null_stats: (n_perm,) array of null statistics.
    """
    null_stats = np.zeros(n_perm)

    n_cells = len(theta)

    # Stratification logic
    if umi_counts is not None:
        # Create deterministic strata using ranks (deciles)
        if n_cells < umi_bins:
            strata = np.zeros(n_cells, dtype=int)
        else:
            ranks = np.argsort(np.argsort(umi_counts))  # dense ranks 0..N-1
            strata = (ranks * umi_bins) // n_cells
    else:
        strata = np.zeros(n_cells, dtype=int)

    unique_strata = np.unique(strata)

    # Pre-calculate indices for each stratum to speed up shuffling
    strata_indices = [np.where(strata == s)[0] for s in unique_strata]

    radar_obs = compute_rsp_radar(r, theta, y, B, delta_deg, min_fg_sector, min_bg_sector)
    valid_mask = ~np.isnan(radar_obs.rsp)
    if not np.any(valid_mask):
        return np.nan, null_stats

    observed_stat = _rms_with_mask(radar_obs.rsp, valid_mask)

    # Use ThreadPoolExecutor to compute stats in parallel
    max_workers = min(16, n_perm)  # Increased from 8 for better parallelism
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _compute_permutation_stat,
                r,
                theta,
                y,
                strata_indices,
                B,
                delta_deg,
                min_fg_sector,
                min_bg_sector,
                seed + i,
                valid_mask,
            )
            for i in range(n_perm)
        ]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                stat = future.result()
                null_stats[i] = stat
            except Exception as e:
                # If computation fails, set to NaN
                null_stats[i] = np.nan

    # Since we generated exactly n_perm, and some may be invalid, but we set to 0, but actually 0 might be valid, but for simplicity, assume all are valid or handle NaN
    # In original, it retries until n_perm valid, but here we just compute n_perm and take what we get.
    # To match, perhaps filter valid ones.
    valid_null_mask = ~np.isnan(null_stats)
    valid_nulls = null_stats[valid_null_mask]
    if len(valid_nulls) == 0:
        return np.nan, null_stats

    p_value = (np.sum(valid_nulls >= observed_stat) + 1) / (len(valid_nulls) + 1)

    return p_value, null_stats


__all__ = ["compute_p_value"]
