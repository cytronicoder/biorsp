"""
Inference module for BioRSP.

Implements permutation testing for statistical significance:
- Null hypothesis: Gene expression is independent of spatial location (angle).
- Test statistic: RMS Anisotropy (A_g).
- Stratified permutation of labels (optional) to control for UMI count confounders.
"""

from typing import Optional, Tuple

import numpy as np

from .constants import N_BG_MIN_DEFAULT, N_FG_MIN_DEFAULT, UMI_BINS_DEFAULT
from .radar import compute_rsp_radar


def _rms_with_mask(rsp: np.ndarray, valid_mask: np.ndarray, missing_as_zero: bool = False) -> float:
    """Compute RMS anisotropy using a fixed sector mask.

    Args:
        rsp: RSP values (may include NaN).
        valid_mask: Boolean mask specifying observed valid sectors.
        missing_as_zero: If True, treat NaNs in valid sectors as zeros.
    """
    masked_rsp = rsp[valid_mask]

    if masked_rsp.size == 0:
        return np.nan

    if missing_as_zero and np.isnan(masked_rsp).any():
        masked_rsp = np.nan_to_num(masked_rsp, nan=0.0)

    if np.isnan(masked_rsp).any():
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
    rng: np.random.Generator,
    valid_mask: np.ndarray,
) -> float:
    """Compute the test statistic for a single permutation."""
    y_perm = y.copy()
    for idx in strata_indices:
        shuffled = rng.permutation(idx)
        y_perm[idx] = y_perm[shuffled]

    radar_perm = compute_rsp_radar(r, theta, y_perm, B, delta_deg, min_fg_sector, min_bg_sector)
    return _rms_with_mask(radar_perm.rsp, valid_mask, missing_as_zero=True)


def compute_p_value(
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
) -> Tuple[float, np.ndarray, float, np.ndarray]:
    """
    Compute p-value for the observed anisotropy using permutation test.
    Foreground labels are permuted within UMI strata to keep geometry fixed.

    Args:
        r: (N,) array of radial distances.
        theta: (N,) array of angles for all cells.
        y: (N,) boolean foreground indicator.
        B: Number of sectors.
        delta_deg: Sector width.
        n_perm: Number of permutations (number of *valid* nulls desired).
        umi_counts: (N,) array of UMI counts for stratification (optional).
        umi_bins: Number of bins for UMI stratification.
        seed: Random seed; different seeds are consumed per trial.
        min_fg_sector: Minimum foreground counts per sector.
        min_bg_sector: Minimum background counts per sector.

    Returns:
        p_value: Estimated p-value (NaN if observed mask empty).
        null_stats: (n_perm,) array of null statistics.
        observed_stat: Observed statistic recomputed from data.
        valid_mask: Boolean mask of observed valid sectors.
    """
    # Prepare output buffer
    null_stats = np.full(n_perm, np.nan)

    n_cells = len(theta)

    # Stratification logic
    if umi_counts is not None:
        # Create deterministic strata using ranks
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

    # Observed radar and valid mask
    radar_obs = compute_rsp_radar(r, theta, y, B, delta_deg, min_fg_sector, min_bg_sector)
    valid_mask = ~np.isnan(radar_obs.rsp)
    if not np.any(valid_mask):
        return np.nan, null_stats, np.nan, valid_mask

    observed_stat = _rms_with_mask(radar_obs.rsp, valid_mask)

    rng = np.random.default_rng(seed)

    for k in range(n_perm):
        null_stats[k] = _compute_permutation_stat(
            r,
            theta,
            y,
            strata_indices,
            B,
            delta_deg,
            min_fg_sector,
            min_bg_sector,
            rng,
            valid_mask,
        )

    p_value = (np.sum(null_stats >= observed_stat) + 1) / (len(null_stats) + 1)

    return p_value, null_stats, observed_stat, valid_mask


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of p-values (NaNs allowed).

    Returns:
        Array of q-values with NaNs preserved.
    """
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan, dtype=float)

    finite_mask = np.isfinite(p_values)
    if not np.any(finite_mask):
        return q_values

    pvals = p_values[finite_mask]
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = len(ranked)
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    q_values[finite_mask] = q[np.argsort(order)]
    return q_values


__all__ = ["compute_p_value", "bh_fdr"]
