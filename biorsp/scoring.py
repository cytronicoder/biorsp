"""Deterministic gene-level scoring and classification utilities for BioRSP."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from biorsp.smoothing import circular_moving_average

DEFAULT_QC_THRESH = 0.35


def _as_1d_float(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def compute_T(E_phi: np.ndarray) -> float:
    """Compute the two-sided anisotropy statistic ``T = max(abs(E_phi))``."""
    arr = _as_1d_float("E_phi", E_phi)
    return float(np.max(np.abs(arr)))


def robust_z(x_obs: float, x_null: np.ndarray, eps: float = 1e-12) -> float:
    """Compute robust Z-score against a null distribution using median and MAD."""
    if not np.isfinite(float(x_obs)):
        raise ValueError("x_obs must be finite.")
    if float(eps) <= 0.0:
        raise ValueError("eps must be > 0.")

    null = _as_1d_float("x_null", x_null)
    med = float(np.median(null))
    mad = float(np.median(np.abs(null - med)))
    scale = 1.4826 * mad + float(eps)
    return float((float(x_obs) - med) / scale)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (pure NumPy, stable ordering)."""
    arr = np.asarray(pvals, dtype=float)
    flat = arr.ravel()
    q_flat = np.ones_like(flat, dtype=float)

    finite_mask = np.isfinite(flat)
    if np.any((flat[finite_mask] < 0.0) | (flat[finite_mask] > 1.0)):
        raise ValueError("p-values must be within [0, 1] (or NaN).")

    if np.any(finite_mask):
        p = flat[finite_mask]
        m = int(p.size)
        order = np.argsort(p, kind="mergesort")
        ranked = p[order]
        ranks = np.arange(1, m + 1, dtype=float)
        adjusted = ranked * (float(m) / ranks)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)

        q_valid = np.empty_like(ranked)
        q_valid[order] = adjusted
        q_flat[finite_mask] = q_valid

    return q_flat.reshape(arr.shape)


def donor_effective_counts(
    donor_ids: np.ndarray,
    f: np.ndarray,
    min_fg_per_donor: int = 10,
    min_bg_per_donor: int = 10,
) -> dict[str, Any]:
    """Summarize donor-level foreground/background support and D_eff."""
    f_bool = np.asarray(f, dtype=bool).ravel()
    donor_arr = np.asarray(donor_ids)
    if donor_arr.size != f_bool.size:
        raise ValueError("donor_ids and f must have the same length.")
    if int(min_fg_per_donor) < 0 or int(min_bg_per_donor) < 0:
        raise ValueError("Per-donor minimum counts must be non-negative.")

    unique_donors, inv = np.unique(donor_arr, return_inverse=True)
    n_total = np.bincount(inv, minlength=unique_donors.size).astype(int)
    n_fg = np.bincount(
        inv, weights=f_bool.astype(int), minlength=unique_donors.size
    ).astype(int)
    n_bg = n_total - n_fg
    informative = (n_fg >= int(min_fg_per_donor)) & (n_bg >= int(min_bg_per_donor))

    donor_stats = {
        str(unique_donors[i]): {"n_fg": int(n_fg[i]), "n_bg": int(n_bg[i])}
        for i in range(unique_donors.size)
    }

    return {
        "donor_stats": donor_stats,
        "donor_labels": unique_donors.astype(str),
        "n_fg_per_donor": n_fg,
        "n_bg_per_donor": n_bg,
        "D_total": int(unique_donors.size),
        "D_eff": int(np.sum(informative)),
        "informative_mask": informative.astype(bool),
        "n_fg_total": int(np.sum(f_bool)),
        "n_bg_total": int(f_bool.size - np.sum(f_bool)),
    }


def evaluate_underpowered(
    *,
    donor_ids: np.ndarray,
    f: np.ndarray,
    n_perm: int,
    p_min: float = 0.005,
    min_fg_total: int = 50,
    min_fg_per_donor: int = 10,
    min_bg_per_donor: int = 10,
    d_eff_min: int = 2,
    min_perm: int = 200,
) -> dict[str, Any]:
    """Apply donor-effective underpowered gating and return detailed diagnostics."""
    f_bool = np.asarray(f, dtype=bool).ravel()
    if f_bool.size == 0:
        raise ValueError("f must contain at least one value.")

    prevalence = float(np.mean(f_bool))
    donor_info = donor_effective_counts(
        donor_ids=donor_ids,
        f=f_bool,
        min_fg_per_donor=int(min_fg_per_donor),
        min_bg_per_donor=int(min_bg_per_donor),
    )

    cond_prev = bool(prevalence < float(p_min))
    cond_fg = bool(int(donor_info["n_fg_total"]) < int(min_fg_total))
    cond_deff = bool(int(donor_info["D_eff"]) < int(d_eff_min))
    cond_perm = bool(int(n_perm) < int(min_perm))
    underpowered = bool(cond_prev or cond_fg or cond_deff or cond_perm)

    donor_fg = np.asarray(donor_info["n_fg_per_donor"], dtype=float)
    donor_bg = np.asarray(donor_info["n_bg_per_donor"], dtype=float)
    cond_fg_per_donor = bool(np.any(donor_fg < int(min_fg_per_donor)))
    cond_bg_per_donor = bool(np.any(donor_bg < int(min_bg_per_donor)))

    abstain_reasons: list[str] = []
    if underpowered:
        if cond_prev:
            abstain_reasons.append("prev_obs_below_floor")
        if cond_fg:
            abstain_reasons.append("fg_total_below_min")
        if cond_deff:
            if cond_fg_per_donor or cond_bg_per_donor:
                abstain_reasons.append("fg_per_donor_below_min")
            abstain_reasons.append("D_eff_below_min")
        if cond_perm:
            abstain_reasons.append("n_perm_below_min")

    return {
        **donor_info,
        "prev": prevalence,
        "underpowered": underpowered,
        "underpowered_reasons": {
            "prev_lt_p_min": cond_prev,
            "n_fg_total_lt_min_fg_total": cond_fg,
            "d_eff_lt_d_eff_min": cond_deff,
            "n_perm_lt_min_perm": cond_perm,
        },
        "abstain_reasons": abstain_reasons,
        "donor_fg_min": float(np.min(donor_fg)) if donor_fg.size else float("nan"),
        "donor_fg_med": float(np.median(donor_fg)) if donor_fg.size else float("nan"),
        "donor_fg_median": (
            float(np.median(donor_fg)) if donor_fg.size else float("nan")
        ),
        "donor_fg_max": float(np.max(donor_fg)) if donor_fg.size else float("nan"),
        "donor_bg_min": float(np.min(donor_bg)) if donor_bg.size else float("nan"),
        "donor_bg_med": float(np.median(donor_bg)) if donor_bg.size else float("nan"),
        "donor_bg_median": (
            float(np.median(donor_bg)) if donor_bg.size else float("nan")
        ),
        "donor_bg_max": float(np.max(donor_bg)) if donor_bg.size else float("nan"),
    }


def circular_smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Compatibility wrapper around canonical circular moving average."""
    return circular_moving_average(x, w)


def coverage_from_null(E_obs: np.ndarray, null_E: np.ndarray, q: float = 0.95) -> float:
    """Compute coverage score: fraction of bins with ``E_obs > q-quantile(null_E)``."""
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError("q must be in [0, 1].")
    obs = _as_1d_float("E_obs", E_obs)
    null = np.asarray(null_E, dtype=float)
    if null.ndim != 2:
        raise ValueError("null_E must have shape (n_perm, n_bins).")
    if null.shape[0] == 0:
        raise ValueError("null_E must include at least one permutation.")
    if null.shape[1] != obs.size:
        raise ValueError(
            f"E_obs length ({obs.size}) must match null_E n_bins ({null.shape[1]})."
        )
    if not np.all(np.isfinite(null)):
        raise ValueError("null_E must be finite.")

    tau = np.quantile(null, float(q), axis=0)
    return float(np.mean(obs > tau))


def peak_count(
    E_obs: np.ndarray,
    null_E: np.ndarray,
    smooth_w: int = 3,
    q_prom: float = 0.95,
) -> int:
    """Count significant circular peaks in ``E_obs`` using null-calibrated prominence."""
    if not (0.0 <= float(q_prom) <= 1.0):
        raise ValueError("q_prom must be in [0, 1].")

    obs = _as_1d_float("E_obs", E_obs)
    if obs.size < 3:
        return 0

    null = np.asarray(null_E, dtype=float)
    if null.ndim != 2:
        raise ValueError("null_E must have shape (n_perm, n_bins).")
    if null.shape[0] == 0:
        raise ValueError("null_E must include at least one permutation.")
    if null.shape[1] != obs.size:
        raise ValueError(
            f"E_obs length ({obs.size}) must match null_E n_bins ({null.shape[1]})."
        )
    if not np.all(np.isfinite(null)):
        raise ValueError("null_E must be finite.")

    obs_smooth = circular_smooth(obs, smooth_w)
    perm_prom = np.max(null, axis=1) - np.median(null, axis=1)
    prom_tau = float(np.quantile(perm_prom, float(q_prom)))

    pad = max(1, int(smooth_w))
    pad = min(pad, obs_smooth.size - 1)
    extended = np.concatenate([obs_smooth[-pad:], obs_smooth, obs_smooth[:pad]])

    peaks, _ = find_peaks(extended, prominence=prom_tau)
    mapped = peaks - pad
    keep = (mapped >= 0) & (mapped < obs_smooth.size)
    if not np.any(keep):
        return 0
    return int(np.unique(mapped[keep]).size)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x_sub = x[mask]
    y_sub = y[mask]
    if np.allclose(x_sub, x_sub[0]) or np.allclose(y_sub, y_sub[0]):
        return float("nan")
    rho = spearmanr(x_sub, y_sub, nan_policy="omit").correlation
    if rho is None or not np.isfinite(float(rho)):
        return float("nan")
    return float(rho)


def detect_qc_columns(
    adata_obs: pd.DataFrame,
    covariate_candidates: dict[str, list[str]],
) -> dict[str, str | None]:
    """Resolve available QC covariate columns from candidate keys."""
    if not isinstance(adata_obs, pd.DataFrame):
        raise TypeError("adata_obs must be a pandas DataFrame.")
    if not isinstance(covariate_candidates, dict) or len(covariate_candidates) == 0:
        raise ValueError("covariate_candidates must be a non-empty dictionary.")

    def _resolve(candidates: list[str]) -> str | None:
        for key in candidates:
            if key in adata_obs.columns:
                return key
        return None

    return {
        "depth_key": _resolve(covariate_candidates.get("total_counts", [])),
        "mt_key": _resolve(covariate_candidates.get("pct_counts_mt", [])),
        "ribo_key": _resolve(covariate_candidates.get("pct_counts_ribo", [])),
    }


def qc_risk_from_covariates(
    expr_or_f: np.ndarray,
    adata_obs: pd.DataFrame,
    qc_columns: dict[str, str | None],
) -> dict[str, float]:
    """Compute QC correlations and absolute-risk score from resolved columns."""
    if not isinstance(adata_obs, pd.DataFrame):
        raise TypeError("adata_obs must be a pandas DataFrame.")
    x = _as_1d_float("expr_or_f", expr_or_f)
    if x.size != int(adata_obs.shape[0]):
        raise ValueError(
            f"expr_or_f length ({x.size}) must match adata_obs rows ({adata_obs.shape[0]})."
        )
    if not isinstance(qc_columns, dict):
        raise TypeError("qc_columns must be a dictionary.")

    depth_key = qc_columns.get("depth_key")
    mt_key = qc_columns.get("mt_key")
    ribo_key = qc_columns.get("ribo_key")

    rho_depth = float("nan")
    rho_mt = float("nan")
    rho_ribo = float("nan")

    if depth_key is not None:
        depth = pd.to_numeric(adata_obs[depth_key], errors="coerce").to_numpy(
            dtype=float
        )
        rho_depth = _safe_spearman(x, depth)
    if mt_key is not None:
        mt = pd.to_numeric(adata_obs[mt_key], errors="coerce").to_numpy(dtype=float)
        rho_mt = _safe_spearman(x, mt)
    if ribo_key is not None:
        ribo = pd.to_numeric(adata_obs[ribo_key], errors="coerce").to_numpy(dtype=float)
        rho_ribo = _safe_spearman(x, ribo)

    rho_values = np.array([rho_depth, rho_mt, rho_ribo], dtype=float)
    finite = rho_values[np.isfinite(rho_values)]
    qc_risk = float(np.max(np.abs(finite))) if finite.size > 0 else 0.0

    return {
        "rho_depth": rho_depth,
        "rho_mt": rho_mt,
        "rho_ribo": rho_ribo,
        "qc_risk": qc_risk,
    }


def qc_metrics(
    expr_or_f: np.ndarray,
    adata_obs: pd.DataFrame,
    covariate_candidates: dict[str, list[str]],
) -> dict[str, Any]:
    """Compute QC correlation metrics and a q-value-independent QC risk summary."""
    qc_columns = detect_qc_columns(adata_obs, covariate_candidates)
    qc_stats = qc_risk_from_covariates(expr_or_f, adata_obs, qc_columns)

    return {
        **qc_columns,
        **qc_stats,
        "qc_driven": bool(qc_stats["qc_risk"] >= DEFAULT_QC_THRESH),
    }


def classify_row(row: dict[str, Any] | pd.Series, thresholds: dict[str, Any]) -> str:
    """Assign deterministic gene classification label for one scored row."""
    if hasattr(row, "get"):
        getter = row.get  # type: ignore[assignment]
    else:
        raise TypeError("row must be a mapping-like object (dict or pandas Series).")
    if not isinstance(thresholds, dict):
        raise TypeError("thresholds must be a dictionary.")

    q_sig = float(thresholds.get("q_sig", 0.05))
    high_prev = float(thresholds.get("high_prev", 0.6))
    qc_thresh = float(thresholds.get("qc_thresh", DEFAULT_QC_THRESH))

    underpowered = bool(getter("underpowered", False))
    if underpowered:
        return "Underpowered"

    q_t_raw = getter("q_T", 1.0)
    q_t = float(q_t_raw) if np.isfinite(float(q_t_raw)) else 1.0
    prev_raw = getter("prev", 0.0)
    prev = float(prev_raw) if np.isfinite(float(prev_raw)) else 0.0
    peaks_raw = getter("peaks_K", 0)
    peaks = int(peaks_raw) if np.isfinite(float(peaks_raw)) else 0

    qc_driven_val = getter("qc_driven", None)
    if qc_driven_val is None or (
        isinstance(qc_driven_val, float) and not np.isfinite(qc_driven_val)
    ):
        qc_risk_raw = getter("qc_risk", float("nan"))
        qc_risk = (
            float(qc_risk_raw) if np.isfinite(float(qc_risk_raw)) else float("nan")
        )
        qc_driven = bool(
            np.isfinite(qc_risk) and abs(qc_risk) >= qc_thresh and q_t <= q_sig
        )
    else:
        qc_driven = bool(qc_driven_val)

    if q_t > q_sig and prev >= high_prev:
        return "Ubiquitous (non-localized)"
    if q_t <= q_sig and peaks == 1 and not qc_driven:
        return "Localized–unimodal"
    if q_t <= q_sig and peaks >= 2 and not qc_driven:
        return "Localized–multimodal"
    if qc_driven:
        return "QC-driven"
    return "Uncertain"
