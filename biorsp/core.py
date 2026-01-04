"""
Core BioRSP algorithms.

Implements the radar radius function R_g(theta) and anisotropy A_g.
Optimized for performance via precomputed indices and vectorized stats.
"""

from typing import List, Optional

import numpy as np

from .typing import AdequacyReport, BioRSPConfig, RadarResult
from .utils import weighted_quantile, weighted_quantile_sorted


def compute_anisotropy(rsp: np.ndarray, valid_mask: np.ndarray) -> float:
    r"""
    Compute RMS anisotropy $A_g$ using a fixed sector mask.

    The anisotropy is defined as the root-mean-square of the RSP values
    across all valid sectors:
    $$A_g = \sqrt{\frac{1}{|B_{valid}|} \sum_{b \in B_{valid}} R_g(\theta_b)^2}$$

    Parameters
    ----------
    rsp : np.ndarray
        (B,) array of RSP values $R_g(\theta_b)$.
    valid_mask : np.ndarray
        (B,) boolean mask of valid sectors $B_{valid}$.

    Returns
    -------
    float
        RMS anisotropy value $A_g$. Returns NaN if no valid sectors exist.
    """
    masked_rsp = rsp[valid_mask]
    if masked_rsp.size == 0:
        return np.nan

    # Treat NaNs in valid sectors as zero (missing data in a valid window)
    clean_rsp = np.nan_to_num(masked_rsp, nan=0.0)
    return float(np.sqrt(np.mean(clean_rsp**2)))


def compute_rsp_radar(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    config: Optional[BioRSPConfig] = None,
    sector_indices: Optional[List[np.ndarray]] = None,
    adequacy: Optional[AdequacyReport] = None,
    normalization_stats: Optional[dict] = None,
    **kwargs,
) -> RadarResult:
    r"""
    Compute the signed RSP radar function $R_g(\theta)$.

    For each sector $b$ with center $\phi_b$, the radar function is defined as:
    $$R_g(\phi_b) = \text{sign}(\text{median}(r_{bg}) - \text{median}(r_{fg})) \cdot \frac{W_1(P_{fg}, P_{bg})}{\max(\text{IQR}(r_{bg}), \tau_{IQR})}$$
    where $W_1$ is the Wasserstein-1 distance between foreground and background
    radial distributions in the angular window $[\phi_b - \delta/2, \phi_b + \delta/2]$.

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
    sector_indices : List[np.ndarray], optional
        Precomputed per-sector cell indices, by default None.
    adequacy : AdequacyReport, optional
        Precomputed adequacy report to reuse indices/mask, by default None.
    normalization_stats : dict, optional
        Radial normalization metadata, by default None.

    Returns
    -------
    RadarResult
        The computed radar function and metadata.
    """
    if config is None:
        config = BioRSPConfig(**kwargs)
    elif kwargs:
        from dataclasses import replace

        config = replace(config, **kwargs)

    B = config.B
    if sector_indices is None and adequacy is not None:
        sector_indices = adequacy.sector_indices

    if sector_indices is None:
        from .geometry import get_sector_indices

        sector_indices = get_sector_indices(theta, B, config.delta_deg)

    rsp_values = np.full(B, np.nan)
    counts_fg = np.zeros(B)
    counts_bg = np.zeros(B)
    iqr_floor_hits = np.zeros(B, dtype=bool)

    # 1. Global background IQR for stability floor
    y_bg_global = 1.0 - y
    global_sort_idx = np.argsort(r)
    r_global_sorted = r[global_sort_idx]
    y_bg_global_sorted = y_bg_global[global_sort_idx]

    global_iqr = weighted_quantile_sorted(
        r_global_sorted, y_bg_global_sorted, 0.75
    ) - weighted_quantile_sorted(r_global_sorted, y_bg_global_sorted, 0.25)
    if not np.isfinite(global_iqr) or global_iqr <= 0:
        global_iqr = 0.0

    iqr_floor = max(config.iqr_floor_pct * global_iqr, 1e-8)

    # 2. Per-sector computation
    n_points = r.size
    for b in range(B):
        idx = sector_indices[b]
        if idx.size == 0:
            continue

        # Efficiently get sorted values for this sector using global sort
        mask = np.zeros(n_points, dtype=bool)
        mask[idx] = True
        sector_sort_idx = global_sort_idx[mask[global_sort_idx]]

        r_sorted = r[sector_sort_idx]
        w_fg_sorted = y[sector_sort_idx]
        w_bg_sorted = (1.0 - y)[sector_sort_idx]

        n_fg = np.sum(w_fg_sorted)
        n_bg = np.sum(w_bg_sorted)
        counts_fg[b] = n_fg
        counts_bg[b] = n_bg

        if n_fg < config.min_fg_sector or n_bg < config.min_bg_sector:
            continue

        # CDFs
        cdf_fg = np.cumsum(w_fg_sorted) / n_fg
        cdf_bg = np.cumsum(w_bg_sorted) / n_bg

        # Integral |CDF_fg - CDF_bg| dr
        dr = np.diff(r_sorted)
        w1 = np.sum(np.abs(cdf_fg[:-1] - cdf_bg[:-1]) * dr)

        # IQR normalization
        q75 = weighted_quantile_sorted(r_sorted, w_bg_sorted, 0.75)
        q25 = weighted_quantile_sorted(r_sorted, w_bg_sorted, 0.25)
        iqr_b = q75 - q25

        denom = max(iqr_b, iqr_floor)
        if iqr_b < iqr_floor:
            iqr_floor_hits[b] = True

        # Sign: positive if foreground is proximal (smaller radii)
        # Sign = sign(median_bg - median_fg)
        med_fg = weighted_quantile(r_sorted, w_fg_sorted, 0.5)
        med_bg = weighted_quantile(r_sorted, w_bg_sorted, 0.5)

        diff = med_bg - med_fg
        if diff == 0:
            # Tie-breaker: mean
            mean_fg = np.sum(r_sorted * w_fg_sorted) / n_fg
            mean_bg = np.sum(r_sorted * w_bg_sorted) / n_bg
            diff = mean_bg - mean_fg

        sign = 1.0 if diff >= 0 else -1.0
        rsp_values[b] = sign * (w1 / denom)

    from .geometry import angle_grid

    return RadarResult(
        rsp=rsp_values,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        centers=angle_grid(B),
        iqr_floor=iqr_floor,
        iqr_floor_hits=iqr_floor_hits,
        normalization_stats=normalization_stats or {},
    )


def assess_adequacy(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    config: Optional[BioRSPConfig] = None,
    **kwargs,
) -> AdequacyReport:
    r"""
    Assess gene adequacy for BioRSP.

    Parameters
    ----------
    r : np.ndarray
        (N,) array of normalized radial distances (unused, for API consistency).
    theta : np.ndarray
        (N,) array of angles in radians.
    y : np.ndarray
        (N,) foreground weights or binary indicators.
    config : BioRSPConfig, optional
        Configuration object, by default BioRSPConfig().

    AdequacyReport
        Report on gene adequacy.
    """
    if config is None:
        # Handle n_sectors alias for B
        if "n_sectors" in kwargs:
            kwargs["B"] = kwargs.pop("n_sectors")
        config = BioRSPConfig(**kwargs)
    elif kwargs:
        from dataclasses import replace

        if "n_sectors" in kwargs:
            kwargs["B"] = kwargs.pop("n_sectors")
        config = replace(config, **kwargs)

    from .constants import (
        REASON_GENE_UNDERPOWERED,
        REASON_OK,
        REASON_SECTOR_BG_TOO_SMALL,
        REASON_SECTOR_FG_TOO_SMALL,
        REASON_SECTOR_MIXED_TOO_SMALL,
    )
    from .geometry import get_sector_indices

    B = config.B
    sector_indices = get_sector_indices(theta, B, config.delta_deg)

    counts_fg = np.zeros(B)
    counts_bg = np.zeros(B)

    for b in range(B):
        idx = sector_indices[b]
        if idx.size > 0:
            counts_fg[b] = np.sum(y[idx])
            counts_bg[b] = np.sum(1.0 - y[idx])

    fg_mask = counts_fg >= config.min_fg_sector
    bg_mask = counts_bg >= config.min_bg_sector
    sector_mask = fg_mask & bg_mask

    n_fg_total = np.sum(y)
    n_bg_total = np.sum(1.0 - y)
    adequacy_fraction = np.mean(sector_mask)

    is_adequate = (
        n_fg_total >= config.min_fg_total and adequacy_fraction >= config.min_adequacy_fraction
    )

    if n_fg_total < config.min_fg_total:
        reason = REASON_GENE_UNDERPOWERED
    elif is_adequate:
        reason = REASON_OK
    elif not np.any(fg_mask):
        reason = REASON_SECTOR_FG_TOO_SMALL
    elif not np.any(bg_mask):
        reason = REASON_SECTOR_BG_TOO_SMALL
    else:
        reason = REASON_SECTOR_MIXED_TOO_SMALL

    return AdequacyReport(
        is_adequate=is_adequate,
        reason=reason,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        sector_mask=sector_mask,
        n_foreground=float(n_fg_total),
        n_background=float(n_bg_total),
        adequacy_fraction=float(adequacy_fraction),
        sector_indices=sector_indices,
    )


__all__ = [
    "compute_anisotropy",
    "compute_rsp_radar",
    "assess_adequacy",
]
