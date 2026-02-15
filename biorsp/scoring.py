"""Deterministic gene-level scoring and classification utilities for BioRSP."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import spearmanr

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


def circular_smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Circular moving-average smoothing with wrap-around padding."""
    arr = _as_1d_float("x", x)
    if int(w) != w or int(w) <= 0:
        raise ValueError("w must be a positive integer.")
    w_int = int(w)
    if w_int % 2 == 0:
        raise ValueError("w must be odd for centered smoothing.")
    if arr.size <= 1 or w_int == 1:
        return arr.copy()

    if w_int > arr.size:
        w_int = arr.size if arr.size % 2 == 1 else arr.size - 1
        if w_int <= 1:
            return arr.copy()

    half = w_int // 2
    padded = np.concatenate([arr[-half:], arr, arr[:half]])
    kernel = np.ones(w_int, dtype=float) / float(w_int)
    smoothed = np.convolve(padded, kernel, mode="valid")
    if smoothed.size != arr.size:
        raise AssertionError("Circular smoothing produced unexpected output length.")
    return smoothed


def coverage_from_null(
    E_obs: np.ndarray, null_E: np.ndarray, q: float = 0.95
) -> float:
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


def qc_metrics(
    expr_or_f: np.ndarray,
    adata_obs: pd.DataFrame,
    covariate_candidates: dict[str, list[str]],
) -> dict[str, Any]:
    """Compute QC correlation metrics and a q-value-independent QC risk summary."""
    if not isinstance(adata_obs, pd.DataFrame):
        raise TypeError("adata_obs must be a pandas DataFrame.")
    x = _as_1d_float("expr_or_f", expr_or_f)
    if x.size != int(adata_obs.shape[0]):
        raise ValueError(
            f"expr_or_f length ({x.size}) must match adata_obs rows ({adata_obs.shape[0]})."
        )

    if not isinstance(covariate_candidates, dict) or len(covariate_candidates) == 0:
        raise ValueError("covariate_candidates must be a non-empty dictionary.")

    def _resolve(candidates: list[str]) -> str | None:
        for key in candidates:
            if key in adata_obs.columns:
                return key
        return None

    depth_key = _resolve(covariate_candidates.get("total_counts", []))
    mt_key = _resolve(covariate_candidates.get("pct_counts_mt", []))
    ribo_key = _resolve(covariate_candidates.get("pct_counts_ribo", []))

    rho_depth = float("nan")
    rho_mt = float("nan")
    rho_ribo = float("nan")

    if depth_key is not None:
        depth = pd.to_numeric(adata_obs[depth_key], errors="coerce").to_numpy(dtype=float)
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
        "depth_key": depth_key,
        "mt_key": mt_key,
        "ribo_key": ribo_key,
        "rho_depth": rho_depth,
        "rho_mt": rho_mt,
        "rho_ribo": rho_ribo,
        "qc_risk": qc_risk,
        "qc_driven": bool(qc_risk >= DEFAULT_QC_THRESH),
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
        qc_driven = bool(np.isfinite(qc_risk) and abs(qc_risk) >= qc_thresh and q_t <= q_sig)
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

