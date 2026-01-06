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
        denom = np.nan
        if nB > 0 and scale_mode in ["pooled_iqr", "bg_iqr", "pooled_mad"]:
            if scale_mode == "pooled_mad":
                med = np.median(r_s)
                denom = 1.4826 * np.median(np.abs(r_s - med))
            else:
                q75 = np.percentile(r_s, 75)
                q25 = np.percentile(r_s, 25)
                denom = q75 - q25

        return {
            "stat": np.nan,
            "stat_raw": np.nan,
            "support_weight": 0.0,
            "sign": 0,
            "w1": np.nan,
            "denom": denom,
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
    s = 0 if np.abs(diff) <= sign_tol else 1 if diff > 0 else -1

    r_bg_only = r_s[w_bg > 0]

    if scale_mode == "u_space":
        # Transform to U-space using background CDF to ensure scale-neutrality.
        if r_bg_only.size > 0:
            r_bg_sorted_local = np.sort(r_bg_only)
            n_bg_points = len(r_bg_sorted_local)
            # Use average of searchsorted to handle ties and stay strictly in (0, 1)
            u_s = (
                np.searchsorted(r_bg_sorted_local, r_s, side="left")
                + np.searchsorted(r_bg_sorted_local, r_s, side="right")
            ) / (2.0 * n_bg_points)
            u_sorted = u_s[sort_idx]
            w1 = weighted_wasserstein_1d(u_sorted, w_fg_sorted, u_sorted, w_bg_sorted)
            denom = 1.0
        else:
            w1 = np.nan
            denom = np.nan
    else:
        # Compute Wasserstein distance and robust scale on raw radial distances.
        w1 = weighted_wasserstein_1d(r_sorted, w_fg_sorted, r_sorted, w_bg_sorted)
        if scale_mode == "bg_iqr":
            denom = weighted_quantile_sorted(
                r_sorted, w_bg_sorted, 0.75
            ) - weighted_quantile_sorted(r_sorted, w_bg_sorted, 0.25)
        elif scale_mode == "fg_iqr":
            denom = weighted_quantile_sorted(
                r_sorted, w_fg_sorted, 0.75
            ) - weighted_quantile_sorted(r_sorted, w_fg_sorted, 0.25)
        elif scale_mode == "pooled_mad":
            med = weighted_quantile_sorted(r_sorted, np.ones_like(r_sorted), 0.5)
            abs_diff = np.abs(r_sorted - med)
            abs_diff_sorted = np.sort(abs_diff)
            denom = 1.4826 * weighted_quantile_sorted(
                abs_diff_sorted, np.ones_like(abs_diff_sorted), 0.5
            )
        else:
            denom = weighted_quantile_sorted(
                r_sorted, np.ones_like(r_sorted), 0.75
            ) - weighted_quantile_sorted(r_sorted, np.ones_like(r_sorted), 0.25)

    if config is not None and config.qc_mode == "principled":
        valid, status, _ = compute_sector_qc(y_s, denom, config)
    else:
        nF_min = config.min_fg_sector if config else 0
        nB_min = config.min_bg_sector if config else 0
        support_ok = (nF >= nF_min) and (nB >= nB_min)
        scale_ok = denom >= min_scale
        valid = support_ok and scale_ok
        if valid:
            status = REASON_OK
        elif not support_ok:
            status = "low_support"
        else:
            status = "degenerate_scale"

    support_weight = compute_sector_weight(nF, nB, mode=weight_mode, k=weight_k)

    stat_raw = 0.0 if not valid else s * (w1 / (denom + eps))

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
    debug: bool = False,
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
    debug : bool, optional
        If True, print per-sector debug info to stdout, by default False.
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

    if config.delta_deg >= 90:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"Wide angular window (delta={config.delta_deg} deg) detected. "
            "Directional localization is not identifiable for wide delta; "
            "interpretation will be limited to global/localized shift via coverage."
        )

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

    if debug:
        print(
            f"{'Theta':>6} | {'nF':>6} | {'nB':>6} | {'Valid':>5} | {'Status':>15} | {'Raw':>8} | {'RSP':>8}"
        )

    for b in range(B):
        if frozen_mask is not None and not frozen_mask[b]:
            if debug:
                theta_rad = -np.pi + b * (2 * np.pi / B)
                theta_deg = np.degrees(theta_rad)
                print(
                    f"{theta_deg:6.1f} | {'-':>6} | {'-':>6} | {'False':>5} | {'frozen_skip':>15} | {'-':>8} | {0.0:8.3f}"
                )
            continue

        idx = sector_indices[b]

        # Define support by total cells in sector and IQR of all radii in sector.
        n_total_sector = idx.size
        if n_total_sector > 0:
            r_s_all = r[idx]
            q75_all = np.percentile(r_s_all, 75)
            q25_all = np.percentile(r_s_all, 25)
            denom_all = q75_all - q25_all
        else:
            denom_all = 0.0

        bg_supp = (n_total_sector >= config.min_bg_sector) and (denom_all >= config.min_scale)

        if idx.size == 0:
            if frozen_mask is not None:
                rsp_values[b] = 0.0
            if debug:
                theta_rad = -np.pi + b * (2 * np.pi / B)
                theta_deg = np.degrees(theta_rad)
                print(
                    f"{theta_deg:6.1f} | {0:6.1f} | {0:6.1f} | {'False':>5} | {'empty_idx':>15} | {'-':>8} | {rsp_values[b]:8.3f}"
                )
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

        if res["status"] == "degenerate_scale":
            iqr_floor_hits[b] = True

        # Enforce background support mask strictly.
        if not bg_supp:
            rsp_values[b] = np.nan
            if debug:
                print(
                    f"DEBUG: Sector {b} NOT bg_supported (n_total={n_total_sector}, denom={denom_all:.4f})"
                )
            continue

        if not res["valid"]:
            is_empty_fg = res["nF"] == 0
            # Since bg_supp is true, we check empty_fg_policy
            if is_empty_fg and config.empty_fg_policy == "zero":
                rsp_values[b] = 0.0
                if debug:
                    print(f"DEBUG: Zero-filling sector {b} (nF={res['nF']}, nB={res['nB']})")
            else:
                rsp_values[b] = np.nan

            if debug:
                theta_rad = -np.pi + b * (2 * np.pi / B)
                theta_deg = np.degrees(theta_rad)
                val_str = f"{rsp_values[b]:8.3f}"
                print(
                    f"{theta_deg:6.1f} | {res['nF']:6.1f} | {res['nB']:6.1f} | {str(res['valid']):>5} | {res['status']:>15} | {res.get('stat_raw', np.nan):8.3f} | {val_str}"
                )
            continue

        if sector_weights is not None:
            rsp_values[b] = sector_weights[b] * res["stat_raw"]
            computed_weights[b] = sector_weights[b]
        else:
            rsp_values[b] = res["stat"]
            computed_weights[b] = res["support_weight"]

        if debug:
            theta_rad = -np.pi + b * (2 * np.pi / B)
            theta_deg = np.degrees(theta_rad)
            print(
                f"{theta_deg:6.1f} | {res['nF']:6.1f} | {res['nB']:6.1f} | {str(res['valid']):>5} | {res['status']:>15} | {res.get('stat_raw', np.nan):8.3f} | {rsp_values[b]:8.3f}"
            )

    from biorsp.preprocess.geometry import angle_grid

    # Determine background-supported sectors (Option A: Label-free support)
    bg_supported_mask = np.zeros(B, dtype=bool)
    denom_scales = np.zeros(B)
    for b in range(B):
        idx = sector_indices[b]
        # Supported if total cells >= min_bg_sector AND radial scale > 0
        if idx.size >= config.min_bg_sector:
            # We check the scale of the background in this sector
            # (which is the same fixed global background distribution)
            s_idx = idx
            r_s = r[s_idx]
            if len(r_s) > 0 and (np.max(r_s) - np.min(r_s) > 1e-9):
                bg_supported_mask[b] = True
                denom_scales[b] = np.std(r_s)  # example proxy for scale

    # Final RSP values: if bg is supported but no FG, set to 0 (if policy is zero)
    # The loop already filled rsp_values with 0.0 or NaN.
    # We update NaN to 0.0 if supported.
    if config.empty_fg_policy == "zero":
        nan_mask = np.isnan(rsp_values)
        rsp_values[nan_mask & bg_supported_mask] = 0.0

    return RadarResult(
        rsp=rsp_values,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        centers=angle_grid(B),
        iqr_floor=iqr_floor,
        iqr_floor_hits=iqr_floor_hits,
        sector_weights=computed_weights,
        normalization_stats=normalization_stats or {},
        n_fg_per_sector=counts_fg,  # For non-weighted, mass=count
        n_bg_per_sector=counts_bg,
        denom_scale_per_sector=denom_scales,
        bg_supported_mask=bg_supported_mask,
    )


__all__ = [
    "compute_anisotropy",
    "compute_rsp_radar",
    "sector_signed_stat",
]
