"""Core BioRSP algorithms.

Implements the radar radius function R_g(theta) and spatial organization score S_g.
Optimized for performance via precomputed indices and vectorized stats.

RadarResult contract:
- rsp is RAW (unweighted statistic). Weights are NEVER applied inside the engine.
- Weights are stored in sector_weights for aggregation in scoring/summaries.
- geom_supported_mask: sector has sufficient total density AND valid scale.
- contrast_supported_mask: sector has sufficient FG and BG for FG/BG contrast.
- forced_zero_mask: geometry-supported but n_fg==0 and empty_fg_policy="zero".
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np

from biorsp.core.qc import compute_sector_qc
from biorsp.core.typing import RadarResult
from biorsp.utils.config import BioRSPConfig
from biorsp.utils.constants import (
    REASON_OK,
    REASON_SECTOR_BG_TOO_SMALL,
    REASON_SECTOR_EMPTY_FG_FORCED_ZERO,
    REASON_SECTOR_FG_TOO_SMALL,
    REASON_SECTOR_LOW_SCALE,
    REASON_SECTOR_LOW_TOTAL,
)
from biorsp.utils.helpers import (
    compute_sector_weight,
    weighted_quantile_sorted,
    weighted_wasserstein_1d,
)


@dataclass
class SectorIndex:
    """Precomputed sector assignments for reuse across features.

    Attributes:
        sector_id: (N,) array mapping each cell to a combined (angle, radius) bin.
        angle_bin: (N,) array of angular bin indices.
        radial_bin: (N,) array of radial bin indices.
        n_sectors: Total number of combined sectors (B * n_radial).
        angle_edges: Bin edges for angle (radians).
        radial_edges: Bin edges for normalized radius in [0, 1].
        angle_centers: Centers for angular bins (radians).
        radial_centers: Centers for radial bins (normalized).
        sector_indices: Optional list of indices per angular window (delta-based).
    """

    sector_id: np.ndarray
    angle_bin: np.ndarray
    radial_bin: np.ndarray
    n_sectors: int
    angle_edges: np.ndarray
    radial_edges: np.ndarray
    angle_centers: np.ndarray
    radial_centers: np.ndarray
    sector_indices: Optional[List[np.ndarray]] = None


def assign_sectors(
    theta: np.ndarray,
    r_norm: np.ndarray,
    *,
    B: int,
    n_radial: int,
    radial_rule: Literal["equal", "quantile"],
    seed: int,
) -> SectorIndex:
    """Assign each cell to a combined angular/radial sector.

    Args:
        theta: (N,) array of angles in radians.
        r_norm: (N,) array of normalized radii in [0, 1].
        B: Number of angular bins.
        n_radial: Number of radial bins.
        radial_rule: Radial binning rule ("equal" or "quantile").
        seed: Random seed (reserved for deterministic tie-breaking).

    Returns:
        SectorIndex with per-cell bin assignments and bin edges.
    """
    if B <= 0 or n_radial <= 0:
        raise ValueError("B and n_radial must be positive.")

    theta = np.asarray(theta)
    r_norm = np.asarray(r_norm)
    if theta.shape != r_norm.shape:
        raise ValueError("theta and r_norm must have the same shape.")

    two_pi = 2 * np.pi
    theta_mod = (theta + two_pi) % two_pi
    angle_edges = np.linspace(0.0, two_pi, B + 1)
    angle_bin = np.digitize(theta_mod, angle_edges, right=False) - 1
    angle_bin = np.clip(angle_bin, 0, B - 1)
    angle_centers = angle_edges[:-1] + (two_pi / B) / 2.0
    angle_centers = angle_centers - np.pi

    r_norm = np.clip(r_norm, 0.0, 1.0)
    if n_radial == 1:
        radial_edges = np.array([0.0, 1.0])
    elif radial_rule == "equal":
        radial_edges = np.linspace(0.0, 1.0, n_radial + 1)
    elif radial_rule == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_radial + 1)
        radial_edges = np.quantile(r_norm, quantiles)
        radial_edges[0] = 0.0
        radial_edges[-1] = 1.0
        for i in range(1, len(radial_edges)):
            if radial_edges[i] <= radial_edges[i - 1]:
                radial_edges[i] = min(1.0, radial_edges[i - 1] + 1e-6)
    else:
        raise ValueError(f"Unknown radial_rule: {radial_rule}")

    radial_bin = np.digitize(r_norm, radial_edges, right=False) - 1
    radial_bin = np.clip(radial_bin, 0, n_radial - 1)
    radial_centers = (radial_edges[:-1] + radial_edges[1:]) / 2.0

    sector_id = angle_bin + B * radial_bin
    n_sectors = B * n_radial

    return SectorIndex(
        sector_id=sector_id,
        angle_bin=angle_bin,
        radial_bin=radial_bin,
        n_sectors=n_sectors,
        angle_edges=angle_edges,
        radial_edges=radial_edges,
        angle_centers=angle_centers,
        radial_centers=radial_centers,
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
    sort_idx: Optional[np.ndarray] = None,
) -> dict:
    """Compute the signed per-sector radar statistic R_g(theta).

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
    sort_idx : np.ndarray, optional
        Precomputed sorting indices for r[in_sector_idx], by default None.

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

    n_fg = float(np.sum(w_fg))
    n_bg = float(np.sum(w_bg))

    if n_fg <= 0 or n_bg <= 0:
        denom = np.nan
        if n_bg > 0 and scale_mode in ["pooled_iqr", "bg_iqr", "pooled_mad"]:
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
            "nF": n_fg,
            "nB": n_bg,
            "status": "empty_fg_or_bg",
            "valid": False,
        }

    if sort_idx is None:
        sort_idx = np.argsort(r_s)

    r_sorted = r_s[sort_idx]
    w_fg_sorted = w_fg[sort_idx]
    w_bg_sorted = w_bg[sort_idx]

    med_fg = weighted_quantile_sorted(r_sorted, w_fg_sorted, 0.5)
    med_bg = weighted_quantile_sorted(r_sorted, w_bg_sorted, 0.5)

    diff = med_bg - med_fg
    s = 0 if np.abs(diff) <= sign_tol else 1 if diff > 0 else -1

    r_bg_only = r_s[w_bg > 0]

    if scale_mode == "u_space":
        if r_bg_only.size > 0:
            r_bg_sorted_local = np.sort(r_bg_only)
            n_bg_points = len(r_bg_sorted_local)

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
            med = np.median(r_sorted)
            denom = 1.4826 * np.median(np.abs(r_sorted - med))
        else:
            denom = np.percentile(r_sorted, 75) - np.percentile(r_sorted, 25)

    if config is not None:
        valid, status, _ = compute_sector_qc(y_s, denom, config)
    else:
        min_nf = 0
        min_nb = 0
        support_ok = (n_fg >= min_nf) and (n_bg >= min_nb)
        scale_ok = denom >= min_scale
        valid = support_ok and scale_ok
        if valid:
            status = REASON_OK
        elif not support_ok:
            status = "low_support"
        else:
            status = "degenerate_scale"

    support_weight = compute_sector_weight(n_fg, n_bg, mode=weight_mode, k=weight_k)

    stat_raw = 0.0 if not valid else s * (w1 / (denom + eps))

    stat = support_weight * stat_raw

    return {
        "stat": stat,
        "stat_raw": stat_raw,
        "support_weight": support_weight,
        "sign": s,
        "w1": w1,
        "denom": denom,
        "medF": med_fg,
        "medB": med_bg,
        "nF": n_fg,
        "nB": n_bg,
        "status": status,
        "valid": valid,
    }


def compute_anisotropy(
    rsp: np.ndarray, valid_mask: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    r"""Compute weighted RMS spatial organization score $S_g$.

    The score is defined as the weighted root-mean-square of the RSP values:
    $$S_g = \sqrt{\frac{\sum_{b \in B_{valid}} w_b \cdot R_g(\theta_b)^2}{\sum_{b \in B_{valid}} w_b}}$$

    Parameters
    ----------
    rsp : np.ndarray
        (B,) array of RAW RSP values $R_g(\theta_b)$.
    valid_mask : np.ndarray
        (B,) boolean mask of valid sectors $B_{valid}$.
    weights : np.ndarray, optional
        (B,) array of sector weights. If None, uniform weights are used.

    Returns
    -------
    float
        Weighted RMS score $S_g$. Returns NaN if no valid sectors exist.

    """
    masked_rsp = rsp[valid_mask]
    if masked_rsp.size == 0:
        return np.nan

    clean_rsp = np.nan_to_num(masked_rsp, nan=0.0)

    if weights is not None:
        w = weights[valid_mask]
        w = np.nan_to_num(w, nan=0.0)
        sum_w = np.sum(w)
        if sum_w <= 0:
            return float(np.sqrt(np.mean(clean_rsp**2)))
        return float(np.sqrt(np.sum(w * clean_rsp**2) / sum_w))
    else:
        return float(np.sqrt(np.mean(clean_rsp**2)))


def compute_rsp_radar(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    config: Optional[BioRSPConfig] = None,
    sector_indices: Optional[List[np.ndarray]] = None,
    sector_sort_indices: Optional[List[np.ndarray]] = None,
    frozen_mask: Optional[np.ndarray] = None,
    normalization_stats: Optional[dict] = None,
    sector_weights: Optional[np.ndarray] = None,
    sector_index: Optional[SectorIndex] = None,
    debug: bool = False,
    **kwargs,
) -> RadarResult:
    r"""Compute the signed RSP radar function $R_g(\theta)$.

    RadarResult contract:
    - rsp is RAW (unweighted). Weights are applied ONCE in scoring/summaries.
    - geom_supported_mask indicates geometry validity (total support + scale).
    - contrast_supported_mask indicates FG/BG contrast validity.
    - forced_zero_mask indicates sectors where rsp=0 due to empty_fg_policy="zero".

    For each sector $b$ with center $\phi_b$, the radar function is defined as:
    $$R_g(\phi_b) = s \cdot \frac{W_1(P_{fg}, P_{bg})}{\text{scale}}$$
    where $s = \text{sign}(\text{median}(r_{bg}) - \text{median}(r_{fg}))$ is the robust sign,
    and $W_1$ is the Wasserstein-1 distance between foreground and background
    radial distributions in the angular window $[\phi_b - \delta/2, \phi_b + \delta/2]$.

    Parameters
    ----------
    r : np.ndarray
        (N,) array of normalized radial distances in [0, 1].
    theta : np.ndarray
        (N,) array of angles in radians.
    y : np.ndarray
        (N,) foreground weights or binary indicators.
    config : BioRSPConfig, optional
        Configuration object, by default BioRSPConfig().
    sector_indices : List[np.ndarray], optional
        Precomputed per-sector cell indices, by default None.
    sector_sort_indices : List[np.ndarray], optional
        Precomputed per-sector sorting indices for radii, by default None.
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
        The computed radar function and metadata. rsp is RAW (unweighted).

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

    from biorsp.core.geometry import angle_grid, get_sector_indices

    n_sectors = config.B
    if sector_indices is None:
        if sector_index is not None and sector_index.sector_indices is not None:
            sector_indices = sector_index.sector_indices
        else:
            sector_indices = get_sector_indices(theta, n_sectors, config.delta_deg)

    rsp_values = np.full(n_sectors, np.nan)
    counts_fg = np.zeros(n_sectors)
    counts_bg = np.zeros(n_sectors)
    counts_total = np.zeros(n_sectors)
    iqr_floor_hits = np.zeros(n_sectors, dtype=bool)
    denom_scales = np.zeros(n_sectors)
    invalid_reasons = [REASON_OK] * n_sectors

    geom_supported_mask = np.zeros(n_sectors, dtype=bool)
    contrast_supported_mask = np.zeros(n_sectors, dtype=bool)
    forced_zero_mask = np.zeros(n_sectors, dtype=bool)

    min_total = getattr(config, "min_total_per_sector", config.min_bg_sector)

    for b in range(n_sectors):
        idx = sector_indices[b]
        n_total = idx.size
        counts_total[b] = n_total

        if n_total < min_total:
            invalid_reasons[b] = REASON_SECTOR_LOW_TOTAL
            continue

        r_s = r[idx]
        iqr = np.percentile(r_s, 75) - np.percentile(r_s, 25)
        denom_scales[b] = iqr

        if iqr < config.min_scale:
            invalid_reasons[b] = REASON_SECTOR_LOW_SCALE
            continue

        geom_supported_mask[b] = True
        invalid_reasons[b] = REASON_OK

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
        header = f"{'Theta':>6} | {'nF':>6} | {'nB':>6} | {'nTot':>6} | {'Geom':>5} | {'Reason':>20} | {'Raw':>8}"
        print(header)

    for b in range(n_sectors):
        if frozen_mask is not None and not frozen_mask[b]:
            if debug:
                theta_rad = -np.pi + b * (2 * np.pi / n_sectors)
                theta_deg = np.degrees(theta_rad)
                print(
                    f"{theta_deg:6.1f} | {'-':>6} | {'-':>6} | {'-':>6} | {'False':>5} | "
                    f"{'frozen_skip':>20} | {0.0:8.3f}"
                )
            continue

        idx = sector_indices[b]

        if idx.size == 0:
            invalid_reasons[b] = REASON_SECTOR_LOW_TOTAL
            if frozen_mask is not None:
                rsp_values[b] = 0.0
            if debug:
                theta_rad = -np.pi + b * (2 * np.pi / n_sectors)
                theta_deg = np.degrees(theta_rad)
                print(
                    f"{theta_deg:6.1f} | {0:6.1f} | {0:6.1f} | {0:6.1f} | {'False':>5} | "
                    f"{'empty_idx':>20} | {rsp_values[b]:8.3f}"
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
            weight_mode="none",
            weight_k=config.sector_weight_k,
            config=config,
            sort_idx=sector_sort_indices[b] if sector_sort_indices else None,
        )

        counts_fg[b] = res["nF"]
        counts_bg[b] = res["nB"]

        if res["status"] == "degenerate_scale":
            iqr_floor_hits[b] = True

        if not geom_supported_mask[b]:
            rsp_values[b] = np.nan
            if debug:
                theta_rad = -np.pi + b * (2 * np.pi / n_sectors)
                theta_deg = np.degrees(theta_rad)
                print(
                    f"{theta_deg:6.1f} | {res['nF']:6.1f} | {res['nB']:6.1f} | {counts_total[b]:6.1f} | "
                    f"{'False':>5} | {invalid_reasons[b]:>20} | {np.nan:8.3f}"
                )
            continue

        n_fg = res["nF"]
        n_bg = res["nB"]
        has_fg = n_fg >= config.min_fg_sector
        has_bg = n_bg >= config.min_bg_sector

        if has_fg and has_bg:
            contrast_supported_mask[b] = True
            rsp_values[b] = res["stat_raw"]
            invalid_reasons[b] = REASON_OK
        elif n_fg == 0 and has_bg and config.empty_fg_policy == "zero":
            forced_zero_mask[b] = True
            contrast_supported_mask[b] = True
            rsp_values[b] = 0.0
            invalid_reasons[b] = REASON_SECTOR_EMPTY_FG_FORCED_ZERO
        elif not has_fg:
            if config.empty_fg_policy == "zero" and geom_supported_mask[b]:
                forced_zero_mask[b] = True
                rsp_values[b] = 0.0
                invalid_reasons[b] = REASON_SECTOR_FG_TOO_SMALL
            else:
                rsp_values[b] = np.nan
                invalid_reasons[b] = REASON_SECTOR_FG_TOO_SMALL
        else:
            if config.empty_fg_policy == "zero" and geom_supported_mask[b]:
                forced_zero_mask[b] = True
                rsp_values[b] = 0.0
                invalid_reasons[b] = REASON_SECTOR_BG_TOO_SMALL
            else:
                rsp_values[b] = np.nan
                invalid_reasons[b] = REASON_SECTOR_BG_TOO_SMALL

        if debug:
            theta_rad = -np.pi + b * (2 * np.pi / n_sectors)
            theta_deg = np.degrees(theta_rad)
            print(
                f"{theta_deg:6.1f} | {res['nF']:6.1f} | {res['nB']:6.1f} | {counts_total[b]:6.1f} | "
                f"{'True':>5} | {invalid_reasons[b]:>20} | {rsp_values[b]:8.3f}"
            )

    if sector_weights is not None:
        computed_weights = sector_weights.copy()
    else:
        max_total = np.max(counts_total)
        if max_total > 0:
            computed_weights = np.clip(counts_total / max_total, 0.0, 1.0)
        else:
            computed_weights = np.ones(n_sectors)

        if config.sector_weight_mode != "none":
            for b in range(n_sectors):
                support_weight = compute_sector_weight(
                    counts_fg[b],
                    counts_bg[b],
                    mode=config.sector_weight_mode,
                    k=config.sector_weight_k,
                )
                computed_weights[b] *= support_weight

    computed_weights[~geom_supported_mask] = 0.0

    return RadarResult(
        rsp=rsp_values,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        counts_total=counts_total,
        centers=angle_grid(n_sectors),
        iqr_floor=iqr_floor,
        iqr_floor_hits=iqr_floor_hits,
        sector_weights=computed_weights,
        normalization_stats=normalization_stats or {},
        denom_scale=denom_scales,
        geom_supported_mask=geom_supported_mask,
        contrast_supported_mask=contrast_supported_mask,
        forced_zero_mask=forced_zero_mask,
        invalid_reason=invalid_reasons,
    )


__all__ = [
    "SectorIndex",
    "assign_sectors",
    "compute_anisotropy",
    "compute_rsp_radar",
    "sector_signed_stat",
]
