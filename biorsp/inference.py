"""
Inference module for BioRSP.

Implements permutation testing for statistical significance:
- Null hypothesis: Gene expression is independent of spatial location (angle).
- Test statistic: RMS Anisotropy (A_g).
- Stratified permutation (optional) to control for UMI count confounders.
"""

from typing import Optional, Tuple

import numpy as np

from .constants import N_BG_MIN_DEFAULT, N_FG_MIN_DEFAULT
from .radar import compute_rsp_radar
from .summaries import compute_scalar_summaries


def compute_p_value(
    observed_stat: float,
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    B: int = 360,
    delta_deg: float = 20.0,
    n_perm: int = 1000,
    umi_counts: Optional[np.ndarray] = None,
    seed: int = 42,
    min_fg_sector: int = N_FG_MIN_DEFAULT,
    min_bg_sector: int = N_BG_MIN_DEFAULT,
) -> Tuple[float, np.ndarray]:
    """
    Compute p-value for the observed anisotropy using permutation test.

    Args:
        observed_stat: The observed test statistic (e.g. rms_anisotropy).
        r: (N,) array of radial distances.
        theta: (N,) array of angles for all cells.
        y: (N,) boolean foreground indicator.
        B: Number of sectors.
        delta_deg: Sector width.
        n_perm: Number of permutations.
        umi_counts: (N,) array of UMI counts for stratification (optional).
        seed: Random seed.
        min_fg_sector: Minimum foreground counts per sector.
        min_bg_sector: Minimum background counts per sector.

    Returns:
        p_value: Estimated p-value.
        null_stats: (n_perm,) array of null statistics.
    """
    rng = np.random.default_rng(seed)
    null_stats = np.zeros(n_perm)

    n_cells = len(theta)

    # Stratification logic
    if umi_counts is not None:
        # Create deterministic strata using ranks (deciles)
        n_bins = 10
        if n_cells < n_bins:
            strata = np.zeros(n_cells, dtype=int)
        else:
            ranks = np.argsort(np.argsort(umi_counts))  # dense ranks 0..N-1
            strata = (ranks * n_bins) // n_cells
    else:
        strata = np.zeros(n_cells, dtype=int)

    unique_strata = np.unique(strata)

    # Pre-calculate indices for each stratum to speed up shuffling
    strata_indices = [np.where(strata == s)[0] for s in unique_strata]

    valid_count = 0
    max_attempts = n_perm * 2  # Safety limit to prevent infinite loops
    attempts = 0

    while valid_count < n_perm and attempts < max_attempts:
        attempts += 1

        # Permute y within strata (theta and r fixed)
        y_perm = y.copy()

        for idx in strata_indices:
            # Shuffle y values within stratum
            # We extract the y values, shuffle them, and put them back
            y_subset = y_perm[idx]
            rng.shuffle(y_subset)
            y_perm[idx] = y_subset

        # Compute statistic on permuted data
        radar_perm = compute_rsp_radar(r, theta, y_perm, B, delta_deg, min_fg_sector, min_bg_sector)

        # Compute Summary
        summary_perm = compute_scalar_summaries(radar_perm)
        stat = summary_perm.rms_anisotropy

        if not np.isnan(stat):
            null_stats[valid_count] = stat
            valid_count += 1

    if valid_count < n_perm:
        # Not enough valid permutations: fill the remainder with NaNs and compute p-value
        # using the valid permutations only.
        null_stats[valid_count:] = np.nan
        valid_nulls = null_stats[:valid_count]
        if valid_count == 0:
            return np.nan, null_stats

        p_value = (np.sum(valid_nulls >= observed_stat) + 1) / (valid_count + 1)
    else:
        # Compute p-value
        # P = (sum(null >= obs) + 1) / (n_perm + 1)
        p_value = (np.sum(null_stats >= observed_stat) + 1) / (n_perm + 1)

    return p_value, null_stats


__all__ = ["compute_p_value"]
