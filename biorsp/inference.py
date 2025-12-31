"""
Inference module for BioRSP.

Implements permutation testing for statistical significance:
- Null hypothesis: Gene expression is independent of spatial location (angle).
- Test statistic: Studentized RMS anisotropy (A_g / SE[A_g]).
- Stratified permutation of labels within donor (when available) and UMI bins.
"""

from typing import Optional, Tuple

import numpy as np

from .constants import N_BG_MIN_DEFAULT, N_FG_MIN_DEFAULT, UMI_BINS_DEFAULT
from .radar import compute_rsp_radar


def _rms_with_mask(rsp: np.ndarray, valid_mask: np.ndarray) -> float:
    """Compute RMS anisotropy using a fixed sector mask.

    Args:
        rsp: RSP values (may include NaN).
        valid_mask: Boolean mask specifying observed valid sectors.
    """
    masked_rsp = rsp[valid_mask]

    if masked_rsp.size == 0:
        return np.nan

    if np.isnan(masked_rsp).any():
        return np.nan

    return float(np.sqrt(np.mean(masked_rsp**2)))


def _build_strata_indices(
    n_cells: int,
    umi_counts: Optional[np.ndarray],
    umi_bins: int,
    donor_ids: Optional[np.ndarray],
) -> list:
    """Build permutation strata indices, enforcing donor-aware shuffling when available."""
    if donor_ids is None:
        donor_codes = np.zeros(n_cells, dtype=int)
    else:
        donor_ids = np.asarray(donor_ids)
        if len(donor_ids) != n_cells:
            raise ValueError("donor_ids length must match number of cells.")
        _, donor_codes = np.unique(donor_ids, return_inverse=True)

    umi_counts_arr = None
    if umi_counts is not None:
        umi_counts_arr = np.asarray(umi_counts)
        if len(umi_counts_arr) != n_cells:
            raise ValueError("umi_counts length must match number of cells.")

    strata_indices = []
    for donor in np.unique(donor_codes):
        donor_idx = np.where(donor_codes == donor)[0]
        if umi_counts_arr is None:
            strata_indices.append(donor_idx)
            continue

        n_donor = len(donor_idx)
        if n_donor < umi_bins:
            bins = np.zeros(n_donor, dtype=int)
        else:
            ranks = np.argsort(np.argsort(umi_counts_arr[donor_idx]))
            bins = (ranks * umi_bins) // n_donor

        for b in np.unique(bins):
            strata_indices.append(donor_idx[bins == b])

    return strata_indices


def _estimate_anisotropy_se(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    B: int,
    delta_deg: float,
    min_fg_sector: int,
    min_bg_sector: int,
    rng: np.random.Generator,
    valid_mask: np.ndarray,
    n_subsample: int = 20,
    subsample_frac: float = 0.8,
) -> float:
    """Estimate SE of anisotropy via subsampling on fixed foreground labels."""
    n_cells = len(r)
    n_keep = max(1, int(n_cells * subsample_frac))

    anisotropies = []
    for _ in range(n_subsample):
        indices = rng.choice(n_cells, size=n_keep, replace=False)
        radar_sub = compute_rsp_radar(
            r[indices],
            theta[indices],
            y[indices],
            B,
            delta_deg,
            min_fg_sector,
            min_bg_sector,
        )
        stat_sub = _rms_with_mask(radar_sub.rsp, valid_mask)
        anisotropies.append(stat_sub)

    anisotropies = np.asarray(anisotropies, dtype=float)
    if np.sum(np.isfinite(anisotropies)) < 2:
        return np.nan

    return float(np.nanstd(anisotropies, ddof=1))


def compute_p_value(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    B: int = 360,
    delta_deg: float = 20.0,
    n_perm: int = 1000,
    umi_counts: Optional[np.ndarray] = None,
    umi_bins: int = UMI_BINS_DEFAULT,
    donor_ids: Optional[np.ndarray] = None,
    seed: int = 42,
    min_fg_sector: int = N_FG_MIN_DEFAULT,
    min_bg_sector: int = N_BG_MIN_DEFAULT,
) -> Tuple[float, np.ndarray, float, np.ndarray, int]:
    """
    Compute p-value for the observed anisotropy using permutation test.
    Foreground labels are permuted within donor strata (if available) and UMI bins to keep geometry fixed.

    Args:
        r: (N,) array of radial distances.
        theta: (N,) array of angles for all cells.
        y: (N,) boolean foreground indicator.
        B: Number of sectors.
        delta_deg: Sector width.
        n_perm: Number of permutations (number of *valid* nulls desired).
        umi_counts: (N,) array of UMI counts for stratification (optional).
        umi_bins: Number of bins for UMI stratification.
        donor_ids: Optional donor/sample labels for donor-aware stratification.
        seed: Random seed; different seeds are consumed per trial.
        min_fg_sector: Minimum foreground counts per sector.
        min_bg_sector: Minimum background counts per sector.

    Returns:
        p_value: Estimated p-value (NaN if observed mask empty).
        null_stats: (n_perm,) array of null statistics.
        observed_stat: Observed studentized statistic recomputed from data.
        valid_mask: Boolean mask of observed valid sectors.
        rejected_permutations: Number of permutations rejected due to missing sectors or invalid SE.
    """
    # Prepare output buffer
    null_stats = np.full(n_perm, np.nan)

    n_cells = len(theta)

    strata_indices = _build_strata_indices(n_cells, umi_counts, umi_bins, donor_ids)

    # Observed radar and valid mask
    radar_obs = compute_rsp_radar(r, theta, y, B, delta_deg, min_fg_sector, min_bg_sector)
    valid_mask = ~np.isnan(radar_obs.rsp)
    if not np.any(valid_mask):
        return np.nan, null_stats, np.nan, valid_mask, 0

    observed_stat = _rms_with_mask(radar_obs.rsp, valid_mask)

    rng = np.random.default_rng(seed)

    observed_se = _estimate_anisotropy_se(
        r,
        theta,
        y,
        B,
        delta_deg,
        min_fg_sector,
        min_bg_sector,
        rng,
        valid_mask,
    )
    if not np.isfinite(observed_stat) or not np.isfinite(observed_se) or observed_se <= 0:
        return np.nan, null_stats, np.nan, valid_mask, 0

    observed_stat = observed_stat / observed_se

    rejected = 0
    k = 0
    while k < n_perm:
        y_perm = y.copy()
        for idx in strata_indices:
            shuffled = rng.permutation(idx)
            y_perm[idx] = y_perm[shuffled]

        radar_perm = compute_rsp_radar(
            r, theta, y_perm, B, delta_deg, min_fg_sector, min_bg_sector
        )
        stat_perm = _rms_with_mask(radar_perm.rsp, valid_mask)
        if not np.isfinite(stat_perm):
            rejected += 1
            continue

        se_perm = _estimate_anisotropy_se(
            r,
            theta,
            y_perm,
            B,
            delta_deg,
            min_fg_sector,
            min_bg_sector,
            rng,
            valid_mask,
        )
        if not np.isfinite(se_perm) or se_perm <= 0:
            rejected += 1
            continue

        null_stats[k] = stat_perm / se_perm
        k += 1

    p_value = (np.sum(null_stats >= observed_stat) + 1) / (len(null_stats) + 1)

    return p_value, null_stats, observed_stat, valid_mask, rejected


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
