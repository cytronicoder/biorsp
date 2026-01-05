"""
Core BioRSP algorithms.

Implements the radar radius function R_g(theta) and anisotropy A_g.
Optimized for performance via precomputed indices and vectorized stats.
"""

from typing import List, Optional

import numpy as np

from biorsp.core.qc import compute_sector_qc
from biorsp.core.typing import RadarResult
from biorsp.utils.config import BioRSPConfig
from biorsp.utils.constants import REASON_OK
from biorsp.utils.helpers import (
    compute_sector_weight,
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
    weight_mode: str = "none",
    weight_k: float = 5.0,
    config: Optional[BioRSPConfig] = None,
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
    weight_mode : str, optional
        Weighting mode for support-based downweighting, by default "none".
    weight_k : float, optional
        Tunable parameter for weighting, by default 5.0.
    config : BioRSPConfig, optional
        Configuration object for principled QC.

    Returns
    -------
    dict
        Dictionary containing the statistic and metadata.
    """
    if in_sector_idx.size == 0:
        return {
            "stat": np.nan,
            "stat_raw": np.nan,
            "support_weight": 0.0,
            "sign": 0,
            "w1": np.nan,
            "denom": np.nan,
            "medF": np.nan,
            "medB": np.nan,
            "nF": 0.0,
            "nB": 0.0,
            "status": "empty_sector",
            "valid": False,
        }

    r_s = r[in_sector_idx]
    y_s = y[in_sector_idx]
    w_fg = y_s
    w_bg = 1.0 - y_s

    nF = float(np.sum(w_fg))
    nB = float(np.sum(w_bg))

    if nF <= 0 or nB <= 0:
        return {
            "stat": np.nan,
            "stat_raw": np.nan,
            "support_weight": 0.0,
            "sign": 0,
            "w1": np.nan,
            "denom": np.nan,
            "medF": np.nan,
            "medB": np.nan,
            "nF": nF,
            "nB": nB,
            "status": "empty_fg_or_bg",
            "valid": False,
        }

    sort_idx = np.argsort(r_s)
    r_sorted = r_s[sort_idx]
    w_fg_sorted = w_fg[sort_idx]
    w_bg_sorted = w_bg[sort_idx]

    medF = weighted_quantile_sorted(r_sorted, w_fg_sorted, 0.5)
    medB = weighted_quantile_sorted(r_sorted, w_bg_sorted, 0.5)

    diff = medB - medF
    if np.abs(diff) <= sign_tol:
        s = 0
    else:
        s = 1 if diff > 0 else -1

    w1 = weighted_wasserstein_1d(r_sorted, w_fg_sorted, r_sorted, w_bg_sorted)

    if scale_mode == "pooled_iqr":
        q75 = np.percentile(r_s, 75)
        q25 = np.percentile(r_s, 25)
        denom = q75 - q25
    elif scale_mode == "bg_iqr":
        q75 = weighted_quantile_sorted(r_sorted, w_bg_sorted, 0.75)
        q25 = weighted_quantile_sorted(r_sorted, w_bg_sorted, 0.25)
        denom = q75 - q25
    elif scale_mode == "fg_iqr":
        q75 = weighted_quantile_sorted(r_sorted, w_fg_sorted, 0.75)
        q25 = weighted_quantile_sorted(r_sorted, w_fg_sorted, 0.25)
        denom = q75 - q25
    elif scale_mode == "pooled_mad":
        med = np.median(r_s)
        denom = 1.4826 * np.median(np.abs(r_s - med))
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    if config is not None and config.qc_mode == "principled":
        valid, status, _ = compute_sector_qc(y_s, denom, config)
    else:
        valid = (
            (nF >= (config.min_fg_sector if config else 0))
            and (nB >= (config.min_bg_sector if config else 0))
            and (denom >= min_scale)
        )
        status = REASON_OK if valid else "low_support_or_scale"

    support_weight = compute_sector_weight(nF, nB, mode=weight_mode, k=weight_k)

    if not valid:
        stat_raw = 0.0
    else:
        stat_raw = s * (w1 / (denom + eps))

    stat = support_weight * stat_raw

    return {
        "stat": stat,
        "stat_raw": stat_raw,
        "support_weight": support_weight,
        "sign": s,
        "w1": w1,
        "denom": denom,
        "medF": medF,
        "medB": medB,
        "nF": nF,
        "nB": nB,
        "status": status,
        "valid": valid,
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
    frozen_mask: Optional[np.ndarray] = None,
    normalization_stats: Optional[dict] = None,
    sector_weights: Optional[np.ndarray] = None,
    **kwargs,
) -> RadarResult:
    r"""
    Compute the signed RSP radar function $R_g(\theta)$.

    For each sector $b$ with center $\phi_b$, the radar function is defined as:
    $$R_g(\phi_b) = s \cdot \frac{W_1(P_{fg}, P_{bg})}{\text{scale}}$$
    where $s = \text{sign}(\text{median}(r_{bg}) - \text{median}(r_{fg}))$ is the robust sign,
    and $W_1$ is the Wasserstein-1 distance between foreground and background
    radial distributions in the angular window $[\phi_b - \delta/2, \phi_b + \delta/2]$.

    We optionally apply a smooth support-based downweighting of sector statistics
    to reduce variance from low-support sectors while preserving the sign and
    overall enrichment morphology.

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
    frozen_mask : np.ndarray, optional
        Fixed boolean mask of valid sectors to compute, by default None.
    normalization_stats : dict, optional
        Radial normalization metadata, by default None.
    sector_weights : np.ndarray, optional
        Precomputed sector weights to reuse (e.g. for permutations), by default None.
    **kwargs
        Overrides for config parameters.

    Returns
    -------
    RadarResult
        The computed radar function and metadata.
    """
    if config is None:
        config = BioRSPConfig(**kwargs)
    elif kwargs:
        from dataclasses import fields, replace

        config_fields = {f.name for f in fields(BioRSPConfig)}
        config_overrides = {k: v for k, v in kwargs.items() if k in config_fields}
        if config_overrides:
            config = replace(config, **config_overrides)

    B = config.B
    if sector_indices is None:
        from biorsp.preprocess.geometry import get_sector_indices

        sector_indices = get_sector_indices(theta, B, config.delta_deg)

    rsp_values = np.full(B, np.nan)
    counts_fg = np.zeros(B)
    counts_bg = np.zeros(B)
    iqr_floor_hits = np.zeros(B, dtype=bool)
    computed_weights = np.ones(B)

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

    for b in range(B):
        if frozen_mask is not None and not frozen_mask[b]:
            continue

        idx = sector_indices[b]
        if idx.size == 0:
            if frozen_mask is not None:
                rsp_values[b] = 0.0
            continue

        res = sector_signed_stat(
            r,
            y,
            idx,
            eps=iqr_floor,
            sign_tol=config.sign_tol,
            scale_mode=config.scale_mode,
            min_scale=config.min_scale,
            weight_mode=config.sector_weight_mode,
            weight_k=config.sector_weight_k,
            config=config,
        )

        counts_fg[b] = res["nF"]
        counts_bg[b] = res["nB"]

        if not res["valid"]:
            if frozen_mask is not None:
                rsp_values[b] = 0.0
            continue

        if sector_weights is not None:
            rsp_values[b] = sector_weights[b] * res["stat_raw"]
            computed_weights[b] = sector_weights[b]
        else:
            rsp_values[b] = res["stat"]
            computed_weights[b] = res["support_weight"]

        if res["status"] == "degenerate_scale":
            iqr_floor_hits[b] = True

    from biorsp.preprocess.geometry import angle_grid

    return RadarResult(
        rsp=rsp_values,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        centers=angle_grid(B),
        iqr_floor=iqr_floor,
        iqr_floor_hits=iqr_floor_hits,
        sector_weights=computed_weights,
        normalization_stats=normalization_stats or {},
    )


__all__ = [
    "compute_anisotropy",
    "compute_rsp_radar",
    "sector_signed_stat",
]
