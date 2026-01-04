"""
Core BioRSP algorithms.

Implements the radar radius function R_g(theta) and anisotropy A_g.
Optimized for performance via precomputed indices and vectorized stats.
"""

from typing import List, Optional

import numpy as np

from .typing import AdequacyReport, BioRSPConfig, RadarResult
from .utils import (
    weighted_quantile,
    weighted_quantile_sorted,
    weighted_wasserstein_1d,
)


def sector_signed_stat(
    r: np.ndarray,
    y: np.ndarray,
    in_sector_idx: np.ndarray,
    *,
    eps: float = 1e-12,
    sign_tol: float = 0.0,
    scale_mode: str = "pooled_iqr",
    min_scale: float = 0.0,
) -> dict:
    """
    Compute the signed per-sector radar statistic R_g(theta).

    Parameters
    ----------
    r : np.ndarray
        (N,) array of radial distances.
    y : np.ndarray
        (N,) foreground weights or binary indicators.
    in_sector_idx : np.ndarray
        Indices of cells in the current sector.
    eps : float, optional
        Small value to avoid division by zero, by default 1e-12.
    sign_tol : float, optional
        Tolerance for median difference to be considered zero, by default 0.0.
    scale_mode : str, optional
        How to compute the robust scale denominator, by default "pooled_iqr".
        Options: "pooled_iqr", "bg_iqr", "fg_iqr", "pooled_mad".
    min_scale : float, optional
        Minimum scale value to avoid degeneracy, by default 0.0.

    Returns
    -------
    dict
        Dictionary containing the statistic and metadata.
    """
    if in_sector_idx.size == 0:
        return {
            "stat": np.nan,
            "sign": 0,
            "w1": np.nan,
            "denom": np.nan,
            "medF": np.nan,
            "medB": np.nan,
            "nF": 0.0,
            "nB": 0.0,
            "status": "empty_sector",
        }

    r_s = r[in_sector_idx]
    y_s = y[in_sector_idx]
    w_fg = y_s
    w_bg = 1.0 - y_s

    nF = float(np.sum(w_fg))
    nB = float(np.sum(w_bg))

    if nF <= 0:
        return {
            "stat": np.nan,
            "sign": 0,
            "w1": np.nan,
            "denom": np.nan,
            "medF": np.nan,
            "medB": np.nan,
            "nF": nF,
            "nB": nB,
            "status": "empty_fg",
        }
    if nB <= 0:
        return {
            "stat": np.nan,
            "sign": 0,
            "w1": np.nan,
            "denom": np.nan,
            "medF": np.nan,
            "medB": np.nan,
            "nF": nF,
            "nB": nB,
            "status": "empty_bg",
        }

    # Medians
    medF = weighted_quantile(r_s, w_fg, 0.5)
    medB = weighted_quantile(r_s, w_bg, 0.5)

    # Sign: positive if foreground is proximal (smaller radii)
    diff = medB - medF
    if np.abs(diff) <= sign_tol:
        s = 0
    else:
        s = 1 if diff > 0 else -1

    # Wasserstein-1 (unsigned)
    w1 = weighted_wasserstein_1d(r_s, w_fg, r_s, w_bg)

    # Scale
    if scale_mode == "pooled_iqr":
        # IQR(concat(R_F, R_B)) -> for binary this is just IQR(r_s)
        # For weighted, we use unweighted IQR of the sector points
        q75 = np.percentile(r_s, 75)
        q25 = np.percentile(r_s, 25)
        denom = q75 - q25
    elif scale_mode == "bg_iqr":
        q75 = weighted_quantile(r_s, w_bg, 0.75)
        q25 = weighted_quantile(r_s, w_bg, 0.25)
        denom = q75 - q25
    elif scale_mode == "fg_iqr":
        q75 = weighted_quantile(r_s, w_fg, 0.75)
        q25 = weighted_quantile(r_s, w_fg, 0.25)
        denom = q75 - q25
    elif scale_mode == "pooled_mad":
        # MAD(concat(R_F, R_B)) -> unweighted MAD of sector points
        med = np.median(r_s)
        denom = 1.4826 * np.median(np.abs(r_s - med))
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    # Degeneracy guard
    if denom < min_scale:
        status = "degenerate_scale"
        stat = 0.0  # Set to 0 as per requirement (a)
    else:
        status = "ok"
        stat = s * (w1 / (denom + eps))

    return {
        "stat": stat,
        "sign": s,
        "w1": w1,
        "denom": denom,
        "medF": medF,
        "medB": medB,
        "nF": nF,
        "nB": nB,
        "status": status,
    }


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
    frozen_mask: Optional[np.ndarray] = None,
    **kwargs,
) -> RadarResult:
    r"""
    Compute the signed RSP radar function $R_g(\theta)$.

    For each sector $b$ with center $\phi_b$, the radar function is defined as:
    $$R_g(\phi_b) = s \cdot \frac{W_1(P_{fg}, P_{bg})}{\text{scale}}$$
    where $s = \text{sign}(\text{median}(r_{bg}) - \text{median}(r_{fg}))$ is the robust sign,
    and $W_1$ is the Wasserstein-1 distance between foreground and background
    radial distributions in the angular window $[\phi_b - \delta/2, \phi_b + \delta/2]$.
    The scale is either the pooled IQR or background IQR plus a stability floor.
    We normalize by the pooled robust scale within each sector by default to improve
    stability under heterogeneous density and to prevent inflation when background
    dispersion is small. Degenerate sectors (scale < min_scale) are assigned a
    statistic of 0.0 and may be excluded from the adequacy mask.

    Positive $R$ means core/proximal enrichment (foreground radii smaller than background).
    Negative $R$ means rim/distal enrichment (foreground radii larger than background).

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
    for b in range(B):
        idx = sector_indices[b]
        if idx.size == 0:
            continue

        # Use the new robust signed stat function
        # We pass iqr_floor as the eps to maintain the stability floor
        res = sector_signed_stat(
            r,
            y,
            idx,
            eps=iqr_floor,
            sign_tol=config.sign_tol,
            scale_mode=config.scale_mode,
            min_scale=config.min_scale,
        )

        counts_fg[b] = res["nF"]
        counts_bg[b] = res["nB"]

        if frozen_mask is not None:
            if not frozen_mask[b]:
                continue
            if res["nF"] == 0 or res["nB"] == 0:
                # Safe behavior for empty sectors in permutations
                rsp_values[b] = 0.0
                continue
        elif res["nF"] < config.min_fg_sector or res["nB"] < config.min_bg_sector:
            continue

        rsp_values[b] = res["stat"]
        if res["status"] == "degenerate_scale":
            iqr_floor_hits[b] = True
            # If degenerate, we set to 0 as per sector_signed_stat
            # but we might also want to mark it as invalid in some contexts.
            # For now, we follow the requirement to set stat=0.

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
        (N,) array of normalized radial distances.
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
    scale_mask = np.ones(B, dtype=bool)

    for b in range(B):
        idx = sector_indices[b]
        if idx.size > 0:
            counts_fg[b] = np.sum(y[idx])
            counts_bg[b] = np.sum(1.0 - y[idx])

            # Scale guard
            if config.min_scale > 0:
                r_s = r[idx]
                if config.scale_mode == "pooled_iqr":
                    scale = np.percentile(r_s, 75) - np.percentile(r_s, 25)
                elif config.scale_mode == "bg_iqr":
                    w_bg = 1.0 - y[idx]
                    if np.sum(w_bg) > 0:
                        scale = weighted_quantile(r_s, w_bg, 0.75) - weighted_quantile(
                            r_s, w_bg, 0.25
                        )
                    else:
                        scale = 0.0
                elif config.scale_mode == "fg_iqr":
                    w_fg = y[idx]
                    if np.sum(w_fg) > 0:
                        scale = weighted_quantile(r_s, w_fg, 0.75) - weighted_quantile(
                            r_s, w_fg, 0.25
                        )
                    else:
                        scale = 0.0
                elif config.scale_mode == "pooled_mad":
                    med = np.median(r_s)
                    scale = 1.4826 * np.median(np.abs(r_s - med))
                else:
                    scale = 0.0

                if scale < config.min_scale:
                    scale_mask[b] = False

    fg_mask = counts_fg >= config.min_fg_sector
    bg_mask = counts_bg >= config.min_bg_sector
    sector_mask = fg_mask & bg_mask & scale_mask

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
