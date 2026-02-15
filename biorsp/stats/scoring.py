"""Scoring utilities and higher-level gene diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from biorsp.core.compute import compute_rsp
from biorsp.core.types import RSPConfig

DEFAULT_QC_THRESH = 0.35


def _as_1d_float(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def compute_T(E_theta: np.ndarray) -> float:
    arr = _as_1d_float("E_theta", E_theta)
    return float(np.max(np.abs(arr)))


def robust_z(x_obs: float, x_null: np.ndarray, eps: float = 1e-12) -> float:
    if not np.isfinite(float(x_obs)):
        raise ValueError("x_obs must be finite.")
    null = _as_1d_float("x_null", x_null)
    med = float(np.median(null))
    mad = float(np.median(np.abs(null - med)))
    scale = 1.4826 * mad + float(eps)
    return float((float(x_obs) - med) / scale)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    arr = np.asarray(pvals, dtype=float)
    flat = arr.ravel()
    q = np.ones_like(flat)
    finite = np.isfinite(flat)
    if np.any((flat[finite] < 0.0) | (flat[finite] > 1.0)):
        raise ValueError("p-values must be in [0,1] or NaN.")
    if np.any(finite):
        p = flat[finite]
        m = int(p.size)
        order = np.argsort(p, kind="mergesort")
        ranked = p[order]
        ranks = np.arange(1, m + 1, dtype=float)
        adj = ranked * (float(m) / ranks)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        adj = np.clip(adj, 0.0, 1.0)
        q_valid = np.empty_like(adj)
        q_valid[order] = adj
        q[finite] = q_valid
    return q.reshape(arr.shape)


def circular_smooth(x: np.ndarray, w: int) -> np.ndarray:
    arr = _as_1d_float("x", x)
    w_i = int(w)
    if w_i <= 0:
        raise ValueError("w must be positive.")
    if w_i % 2 == 0:
        raise ValueError("w must be odd.")
    if arr.size <= 1 or w_i == 1:
        return arr.copy()
    if w_i > arr.size:
        w_i = arr.size if arr.size % 2 == 1 else arr.size - 1
        if w_i <= 1:
            return arr.copy()
    half = w_i // 2
    padded = np.concatenate([arr[-half:], arr, arr[:half]])
    kernel = np.ones(w_i, dtype=float) / float(w_i)
    return np.convolve(padded, kernel, mode="valid")


def coverage_from_null(E_obs: np.ndarray, null_E: np.ndarray, q: float = 0.95) -> float:
    obs = _as_1d_float("E_obs", E_obs)
    null = np.asarray(null_E, dtype=float)
    if null.ndim != 2 or null.shape[1] != obs.size:
        raise ValueError("null_E must have shape (n_perm, n_bins).")
    tau = np.quantile(null, float(q), axis=0)
    return float(np.mean(obs > tau))


def peak_count(E_obs: np.ndarray, null_E: np.ndarray, smooth_w: int = 3, q_prom: float = 0.95) -> int:
    obs = _as_1d_float("E_obs", E_obs)
    null = np.asarray(null_E, dtype=float)
    if null.ndim != 2 or null.shape[1] != obs.size:
        raise ValueError("null_E must have shape (n_perm, n_bins).")
    obs_s = circular_smooth(obs, smooth_w)
    perm_prom = np.max(null, axis=1) - np.median(null, axis=1)
    prom_tau = float(np.quantile(perm_prom, float(q_prom)))
    pad = min(max(1, int(smooth_w)), obs_s.size - 1)
    ext = np.concatenate([obs_s[-pad:], obs_s, obs_s[:pad]])
    peaks, _ = find_peaks(ext, prominence=prom_tau)
    mapped = peaks - pad
    keep = (mapped >= 0) & (mapped < obs_s.size)
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
    x = _as_1d_float("expr_or_f", expr_or_f)
    if x.size != int(adata_obs.shape[0]):
        raise ValueError("expr_or_f length must match adata_obs rows.")

    def _resolve(keys: list[str]) -> str | None:
        for key in keys:
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

    vals = np.array([rho_depth, rho_mt, rho_ribo], dtype=float)
    finite = vals[np.isfinite(vals)]
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
    q_sig = float(thresholds.get("q_sig", 0.05))
    high_prev = float(thresholds.get("high_prev", 0.6))
    qc_thresh = float(thresholds.get("qc_thresh", DEFAULT_QC_THRESH))

    underpowered = bool(row.get("underpowered", False))
    if underpowered:
        return "Underpowered"

    q_t = float(row.get("q_T", 1.0)) if np.isfinite(float(row.get("q_T", 1.0))) else 1.0
    prev = float(row.get("prev", row.get("prevalence", 0.0)))
    peaks = int(row.get("peaks_K", 0)) if np.isfinite(float(row.get("peaks_K", 0))) else 0

    qc_driven = bool(row.get("qc_driven", False))
    if not qc_driven:
        qc_risk = float(row.get("qc_risk", 0.0))
        qc_driven = bool(np.isfinite(qc_risk) and abs(qc_risk) >= qc_thresh and q_t <= q_sig)

    if q_t <= q_sig:
        if qc_driven:
            return "QC-driven"
        if peaks >= 2:
            return "Localized-multimodal"
        return "Localized-unimodal"

    if prev >= high_prev:
        return "Ubiquitous"
    return "Uncertain"


@dataclass(frozen=True)
class ScoreGeneConfig:
    basis: str = "X_umap"
    bins: int = 72
    threshold: float = 0.0
    center_method: str = "median"


def make_foreground_mask(x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    arr = _as_1d_float("x", x)
    return arr > float(threshold)


def score_gene(
    *,
    expr: np.ndarray,
    embedding_xy: np.ndarray,
    feature_label: str,
    config: ScoreGeneConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ScoreGeneConfig()
    out = compute_rsp(
        expr=np.asarray(expr, dtype=float),
        embedding_xy=np.asarray(embedding_xy, dtype=float),
        config=RSPConfig(
            basis=cfg.basis,
            bins=int(cfg.bins),
            center_method=cfg.center_method,
            threshold=float(cfg.threshold),
            feature_label=feature_label,
        ),
        feature_label=feature_label,
    )
    return {
        "gene": feature_label,
        "anisotropy": float(out.anisotropy),
        "peak_direction": float(out.peak_direction),
        "breadth": float(out.breadth),
        "coverage": float(out.coverage),
        "E_max": float(out.E_max),
    }


def diagnose_random_gene_scores(*args, **kwargs) -> dict[str, Any]:
    raise NotImplementedError(
        "diagnose_random_gene_scores is removed from the public scoring module in this refactor."
    )
