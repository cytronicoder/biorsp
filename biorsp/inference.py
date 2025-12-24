"""
Inference module for BioRSP.

Implements permutation testing for statistical significance:
- Null hypothesis: Gene expression is independent of spatial location (angle).
- Test statistic: Mean Absolute RSP (Anisotropy).
- Stratified permutation (optional) to control for UMI count confounders.
"""

from typing import Optional, Tuple

import numpy as np

from .radar import compute_rsp_radar
from .summaries import compute_scalar_summaries


def compute_p_value(
    observed_stat: float,
    theta: np.ndarray,
    y: np.ndarray,
    B: int = 360,
    delta_deg: float = 20.0,
    n_perm: int = 1000,
    umi_counts: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[float, np.ndarray]:
    """
    Compute p-value for the observed anisotropy using permutation test.

    Args:
        observed_stat: The observed test statistic (e.g. mean_abs_rsp).
        theta: (N,) array of angles for all cells.
        y: (N,) boolean foreground indicator.
        B: Number of sectors.
        delta_deg: Sector width.
        n_perm: Number of permutations.
        umi_counts: (N,) array of UMI counts for stratification (optional).
        seed: Random seed.

    Returns:
        p_value: Estimated p-value.
        null_stats: (n_perm,) array of null statistics.

    Notes:
        Permuted datasets may yield inadequate radar profiles (e.g., empty sectors).
        `compute_rsp_radar` handles such cases, and the test statistic (mean_abs_rsp)
        is robust to occasional inadequate windows.
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

    for i in range(n_perm):
        # Permute theta within strata
        theta_perm = np.zeros_like(theta)

        for s in unique_strata:
            mask = strata == s
            # Shuffle theta within the stratum (y labels remain fixed)
            subset_theta = theta[mask]
            rng.shuffle(subset_theta)
            theta_perm[mask] = subset_theta

        # Compute statistic on permuted data
        # 1. Extract foreground angles
        theta_fg_perm = theta_perm[y]

        # 2. Compute Radar (permuted profiles may be inadequate; compute_rsp_radar
        # handles such cases)

        radar_perm = compute_rsp_radar(theta_fg_perm, B, delta_deg)

        # 3. Compute Summary
        summary_perm = compute_scalar_summaries(radar_perm)
        null_stats[i] = summary_perm.mean_abs_rsp

    # Compute p-value
    # P = (sum(null >= obs) + 1) / (n_perm + 1)
    p_value = (np.sum(null_stats >= observed_stat) + 1) / (n_perm + 1)

    return p_value, null_stats


__all__ = ["compute_p_value"]
