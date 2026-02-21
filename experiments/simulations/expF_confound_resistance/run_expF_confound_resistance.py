#!/usr/bin/env python3
"""Simulation Experiment F: confound resistance under QC artifacts and embedding distortions.

This experiment stress-tests individual-gene BioRSP scoring with donor-stratified
permutation inference, explicit QC confound checks, and final labeling:
- UNDERPOWERED
- NOT_SIGNIFICANT
- QC_DRIVEN
- LOCALIZED_PASS
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from scipy.stats import chi2, spearmanr
from scipy.stats import t as student_t

# Safe non-interactive plotting setup.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-biorsp-expF")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache-biorsp-expF")
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from biorsp.moran import morans_i
from biorsp.permutation import perm_null_T
from biorsp.power import evaluate_underpowered
from biorsp.rsp import compute_rsp_profile_from_boolean
from biorsp.scoring import bh_fdr, robust_z
from experiments.simulations._shared.cli import add_common_args
from experiments.simulations._shared.donors import (
    assign_donors,
    donor_effect_vector,
    sample_donor_effects,
)
from experiments.simulations._shared.geometry import sample_geometry
from experiments.simulations._shared.io import (
    atomic_write_csv,
    ensure_dir,
    git_commit_hash,
    timestamped_run_id,
    write_config,
)
from experiments.simulations._shared.plots import (
    p_hist,
    plot_embedding_with_foreground,
    plot_rsp_polar,
    qq_plot,
    savefig,
    wilson_ci,
)
from experiments.simulations._shared.runner import (
    finalize_legacy_run,
    prepare_legacy_run,
)
from experiments.simulations._shared.seeding import rng_from_seed, stable_seed
from experiments.simulations._sim_testmode import apply_testmode_overrides

DEFAULT_GEOMETRIES = ["disk_gaussian", "density_gradient_disk"]
DEFAULT_D_GRID = [5, 10]
DEFAULT_PI_GRID = [0.05, 0.2, 0.6]
DEFAULT_G_QC_GRID = [0.0, 0.5, 1.0]
DEFAULT_GAMMA_GRID = [0.5, 1.0, 1.5]
DEFAULT_BETA_GRID = [0.75, 1.25]
DEFAULT_MORAN_K_GRID = [10, 20, 30]

LABEL_ORDER = ["UNDERPOWERED", "NOT_SIGNIFICANT", "QC_DRIVEN", "LOCALIZED_PASS"]


@dataclass(frozen=True)
class DatasetContext:
    seed_run: int
    geometry: str
    N: int
    D: int
    sigma_eta: float
    g_qc: float
    X: np.ndarray
    theta: np.ndarray
    donor_ids: np.ndarray
    donor_to_idx: dict[str, np.ndarray]
    eta_d: np.ndarray
    eta_cell: np.ndarray
    log_library_size: np.ndarray
    pct_mt: np.ndarray
    ribo_score: np.ndarray
    z_log_library: np.ndarray
    z_pct_mt: np.ndarray
    z_ribo: np.ndarray
    bin_id: np.ndarray
    bin_counts_total: np.ndarray


def _fmt(x: float | int) -> str:
    return f"{float(x):.3f}".rstrip("0").rstrip(".")


def _clip_pi(pi_target: float) -> float:
    return float(np.clip(float(pi_target), 1e-9, 1.0 - 1e-9))


def _compute_bin_cache(theta: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    wrapped = np.mod(np.asarray(theta, dtype=float).ravel(), 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, int(n_bins) + 1, endpoint=True)
    bin_id = np.digitize(wrapped, edges, right=False) - 1
    bin_id = np.where(bin_id == int(n_bins), int(n_bins) - 1, bin_id).astype(np.int32)
    bin_counts_total = np.bincount(bin_id, minlength=int(n_bins)).astype(np.int64)
    return bin_id, bin_counts_total


def _zscore(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float).ravel()
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def _donor_to_idx_map(donor_ids: np.ndarray) -> dict[str, np.ndarray]:
    donor_arr = np.asarray(donor_ids)
    labels = np.unique(donor_arr)
    return {str(d): np.flatnonzero(donor_arr == d).astype(int) for d in labels}


def _qc_geometry_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.size != y_arr.size or x_arr.size < 3:
        return float("nan")
    sx = float(np.std(x_arr))
    sy = float(np.std(y_arr))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _build_knn_weights(X: np.ndarray, k: int) -> sp.csr_matrix:
    n = int(X.shape[0])
    if int(k) <= 0:
        raise ValueError("k must be positive for kNN graph.")
    k_eff = int(min(int(k), max(1, n - 1)))
    tree = cKDTree(np.asarray(X, dtype=float))
    _, idx = tree.query(X, k=k_eff + 1)
    rows = np.repeat(np.arange(n, dtype=np.int32), k_eff)
    cols = np.asarray(idx[:, 1:], dtype=np.int32).reshape(-1)
    data = np.ones(rows.size, dtype=float)
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
    W = W.maximum(W.T).tocsr()
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def _fisher_combine_pvalues(pvals: list[float]) -> tuple[float, float]:
    vals = [float(p) for p in pvals if np.isfinite(float(p)) and float(p) > 0.0]
    if not vals:
        return 0.0, 1.0
    stat = float(
        -2.0 * np.sum(np.log(np.clip(np.asarray(vals, dtype=float), 1e-300, 1.0)))
    )
    p_comb = float(chi2.sf(stat, 2 * len(vals)))
    return stat, p_comb


def _within_donor_spearman(
    f: np.ndarray,
    covariate: np.ndarray,
    donor_to_idx: dict[str, np.ndarray],
) -> tuple[float, float, float, int]:
    f_arr = np.asarray(f, dtype=float).ravel()
    c_arr = np.asarray(covariate, dtype=float).ravel()
    if f_arr.size != c_arr.size:
        raise ValueError("f and covariate lengths must match.")

    rhos: list[float] = []
    weights: list[float] = []

    for idx in donor_to_idx.values():
        idx_arr = np.asarray(idx, dtype=int)
        if idx_arr.size < 6:
            continue
        f_sub = f_arr[idx_arr]
        c_sub = c_arr[idx_arr]
        if np.allclose(f_sub, f_sub[0]) or np.allclose(c_sub, c_sub[0]):
            continue
        rho, p = spearmanr(f_sub, c_sub, nan_policy="omit")
        if rho is None or p is None:
            continue
        rho_f = float(rho)
        if p is None:
            continue
        p_f = float(p)
        if not (np.isfinite(rho_f) and np.isfinite(p_f)):
            continue
        rhos.append(rho_f)
        weights.append(float(max(idx_arr.size - 3, 1)))

    if not rhos:
        return 0.0, 1.0, 0.0, 0

    w_arr = np.asarray(weights, dtype=float)
    r_arr = np.asarray(rhos, dtype=float)
    pooled_rho = float(np.sum(w_arr * r_arr) / np.sum(w_arr))

    if r_arr.size >= 2:
        z = np.arctanh(np.clip(r_arr, -0.999999, 0.999999))
        z_mean = float(np.mean(z))
        z_sd = float(np.std(z, ddof=1))
        if z_sd <= 1e-12:
            t_stat = float(np.sign(z_mean) * np.inf) if abs(z_mean) > 1e-12 else 0.0
            p_meta = 0.0 if np.isfinite(t_stat) and abs(t_stat) > 0 else 1.0
        else:
            t_stat = float(z_mean / (z_sd / math.sqrt(float(z.size))))
            p_meta = float(2.0 * student_t.sf(abs(t_stat), df=int(z.size - 1)))
        stat = float(t_stat * t_stat) if np.isfinite(t_stat) else float("inf")
    else:
        # One donor does not support a stable donor-level meta p-value.
        p_meta = 1.0
        stat = 0.0

    return pooled_rho, float(np.clip(p_meta, 0.0, 1.0)), stat, len(rhos)


def qc_assoc_test_spearman(
    *,
    f: np.ndarray,
    context: DatasetContext,
) -> dict[str, float]:
    rho_l, p_l, stat_l, _ = _within_donor_spearman(
        f, context.log_library_size, context.donor_to_idx
    )
    rho_m, p_m, stat_m, _ = _within_donor_spearman(
        f, context.pct_mt, context.donor_to_idx
    )
    rho_r, p_r, stat_r, _ = _within_donor_spearman(
        f, context.ribo_score, context.donor_to_idx
    )

    block_stat, block_p = _fisher_combine_pvalues([p_l, p_m, p_r])

    rho_vec = np.asarray([rho_l, rho_m, rho_r], dtype=float)
    p_vec = np.asarray([p_l, p_m, p_r], dtype=float)
    idx = int(np.nanargmax(np.abs(rho_vec))) if rho_vec.size else 0
    max_abs = float(np.max(np.abs(rho_vec))) if rho_vec.size else 0.0
    max_abs_p = float(p_vec[idx]) if p_vec.size else 1.0

    return {
        "qc_assoc_stat": float(block_stat + stat_l + stat_m + stat_r),
        "qc_assoc_p": float(block_p),
        "max_abs_qc_corr": float(max_abs),
        "max_abs_qc_corr_p": float(max_abs_p),
        "rho_log_library": float(rho_l),
        "rho_pct_mt": float(rho_m),
        "rho_ribo": float(rho_r),
        "p_log_library": float(p_l),
        "p_pct_mt": float(p_m),
        "p_ribo": float(p_r),
    }


def qc_assoc_test_logistic(
    *,
    f: np.ndarray,
    context: DatasetContext,
) -> dict[str, float]:
    """Optional logistic QC-block test; falls back to Spearman if unavailable."""
    try:
        import statsmodels.api as sm
    except Exception:
        warnings.warn(
            "statsmodels unavailable; falling back to --qc_test spearman.",
            RuntimeWarning,
            stacklevel=2,
        )
        return qc_assoc_test_spearman(f=f, context=context)

    y = np.asarray(f, dtype=int).ravel()
    donor = pd.Categorical(context.donor_ids.astype(str))
    donor_df = pd.get_dummies(donor, drop_first=True, dtype=float)

    x0 = donor_df.copy()
    x0.insert(0, "intercept", 1.0)

    x1 = x0.copy()
    x1["z_log_library"] = _zscore(context.log_library_size)
    x1["z_pct_mt"] = _zscore(context.pct_mt)
    x1["z_ribo"] = _zscore(context.ribo_score)

    try:
        fit0 = sm.Logit(y, x0).fit(disp=False, method="lbfgs", maxiter=200)
        fit1 = sm.Logit(y, x1).fit(disp=False, method="lbfgs", maxiter=200)
        ll0 = float(fit0.llf)
        ll1 = float(fit1.llf)
        lr_stat = max(0.0, 2.0 * (ll1 - ll0))
        qc_p = float(chi2.sf(lr_stat, 3))
    except Exception:
        warnings.warn(
            "Logistic QC test failed; falling back to Spearman QC test.",
            RuntimeWarning,
            stacklevel=2,
        )
        return qc_assoc_test_spearman(f=f, context=context)

    spearman_part = qc_assoc_test_spearman(f=f, context=context)
    return {
        **spearman_part,
        "qc_assoc_stat": float(lr_stat),
        "qc_assoc_p": float(qc_p),
    }


def _simulate_gene_foreground(
    *,
    context: DatasetContext,
    regime: str,
    pi_target: float,
    gamma: float,
    beta: float,
    theta0: float,
    seed_gene: int,
) -> np.ndarray:
    rng = rng_from_seed(int(seed_gene))
    alpha = float(logit(_clip_pi(float(pi_target))))

    regime_s = str(regime)
    if regime_s in {"R0", "R1"}:
        logits = alpha + context.eta_cell
    elif regime_s == "R2":
        logits = (
            alpha
            + context.eta_cell
            + float(gamma) * context.z_log_library
            + float(gamma) * context.z_pct_mt
            + float(gamma) * context.z_ribo
        )
    elif regime_s == "R3":
        logits = (
            alpha
            + context.eta_cell
            + float(beta) * np.cos(context.theta - float(theta0))
            + 0.3 * context.z_log_library
        )
    else:
        raise ValueError(f"Unknown regime '{regime_s}'.")

    p = expit(logits)
    return (rng.random(p.size) < p).astype(bool)


def _dataset_key(context: DatasetContext) -> tuple[int, str, int, float, float, int]:
    return (
        int(context.seed_run),
        str(context.geometry),
        int(context.D),
        float(context.sigma_eta),
        float(context.g_qc),
        int(context.N),
    )


def _context_from_key(
    contexts: dict[tuple[int, str, int, float, float, int], DatasetContext],
    row: pd.Series,
) -> DatasetContext:
    key = (
        int(row["seed_run"]),
        str(row["geometry"]),
        int(row["D"]),
        float(row["sigma_eta"]),
        float(row["g_qc"]),
        int(row["N"]),
    )
    return contexts[key]


def _simulate_dataset_context(
    *,
    master_seed: int,
    seed_run: int,
    geometry: str,
    N: int,
    D: int,
    sigma_eta: float,
    g_qc: float,
    n_bins: int,
    density_k: float,
    mu0_library: float,
    sigma_library: float,
    mt_mean: float,
    mt_concentration: float,
) -> tuple[DatasetContext, dict[str, float]]:
    seed_dataset = stable_seed(
        int(master_seed),
        "expF",
        "dataset",
        int(seed_run),
        str(geometry),
        int(N),
        int(D),
        float(sigma_eta),
        float(g_qc),
    )
    rng = rng_from_seed(int(seed_dataset))

    kwargs: dict[str, Any] = {}
    if str(geometry) == "density_gradient_disk":
        kwargs["k"] = float(density_k)
    X, _ = sample_geometry(str(geometry), int(N), rng, **kwargs)
    theta = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2.0 * np.pi).astype(float)

    donor_ids = assign_donors(int(N), int(D), rng)
    donor_to_idx = _donor_to_idx_map(donor_ids)

    eta_d = sample_donor_effects(int(D), float(sigma_eta), rng)
    eta_cell = donor_effect_vector(donor_ids, eta_d)

    mu_shift_d = rng.normal(loc=0.0, scale=0.4, size=int(D)).astype(float)
    mu_L_d = float(mu0_library) + mu_shift_d
    log_library_size = rng.normal(
        loc=mu_L_d[donor_ids],
        scale=float(sigma_library),
        size=int(N),
    ).astype(float)

    mt_shift_d = rng.uniform(low=-0.02, high=0.02, size=int(D)).astype(float)
    mt_mean_d = np.clip(float(mt_mean) + mt_shift_d, 0.005, 0.35)
    a_d = np.clip(mt_mean_d * float(mt_concentration), 0.1, None)
    b_d = np.clip((1.0 - mt_mean_d) * float(mt_concentration), 0.1, None)
    pct_mt = rng.beta(a=a_d[donor_ids], b=b_d[donor_ids]).astype(float)

    if float(g_qc) > 0.0:
        x = np.asarray(X[:, 0], dtype=float)
        log_library_size = log_library_size + float(g_qc) * x
        mt_logit = logit(np.clip(pct_mt, 1e-4, 1.0 - 1e-4))
        mt_logit = mt_logit + (0.35 * float(g_qc)) * x
        pct_mt = expit(mt_logit)

    ribo_score = 0.5 * log_library_size + rng.normal(loc=0.0, scale=0.3, size=int(N))

    z_log_library = _zscore(log_library_size)
    z_pct_mt = _zscore(pct_mt)
    z_ribo = _zscore(ribo_score)

    bin_id, bin_counts_total = _compute_bin_cache(theta, int(n_bins))

    qc_corr = {
        "corr_log_library_x": _qc_geometry_corr(log_library_size, X[:, 0]),
        "corr_log_library_y": _qc_geometry_corr(log_library_size, X[:, 1]),
        "corr_pct_mt_x": _qc_geometry_corr(pct_mt, X[:, 0]),
        "corr_pct_mt_y": _qc_geometry_corr(pct_mt, X[:, 1]),
        "corr_ribo_x": _qc_geometry_corr(ribo_score, X[:, 0]),
        "corr_ribo_y": _qc_geometry_corr(ribo_score, X[:, 1]),
    }
    finite = np.asarray([v for v in qc_corr.values() if np.isfinite(v)], dtype=float)
    qc_corr["max_abs_corr"] = (
        float(np.max(np.abs(finite))) if finite.size else float("nan")
    )

    context = DatasetContext(
        seed_run=int(seed_run),
        geometry=str(geometry),
        N=int(N),
        D=int(D),
        sigma_eta=float(sigma_eta),
        g_qc=float(g_qc),
        X=X,
        theta=theta,
        donor_ids=donor_ids,
        donor_to_idx=donor_to_idx,
        eta_d=eta_d,
        eta_cell=eta_cell,
        log_library_size=log_library_size,
        pct_mt=pct_mt,
        ribo_score=ribo_score,
        z_log_library=z_log_library,
        z_pct_mt=z_pct_mt,
        z_ribo=z_ribo,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )
    return context, qc_corr


def _regime_plan_for_gqc(g_qc: float) -> list[tuple[str, float, float]]:
    """Return (regime, gamma, beta) tuples for this dataset g_qc value."""
    plan: list[tuple[str, float, float]] = [("R1", 0.0, 0.0)]
    if abs(float(g_qc)) < 1e-12:
        plan.insert(0, ("R0", 0.0, 0.0))
    return plan


def _attach_q_values(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    out["qc_assoc_q"] = np.nan

    group_cols = [
        "seed_run",
        "geometry",
        "N",
        "D",
        "sigma_eta",
        "regime",
        "pi_target",
        "beta",
        "gamma",
        "g_qc",
    ]
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        pvals = pd.to_numeric(out.loc[idx_arr, "qc_assoc_p"], errors="coerce").to_numpy(
            dtype=float
        )
        qvals = bh_fdr(np.where(np.isfinite(pvals), pvals, 1.0))
        out.loc[idx_arr, "qc_assoc_q"] = qvals

    out["qc_assoc_q"] = pd.to_numeric(out["qc_assoc_q"], errors="coerce").fillna(1.0)
    return out


def _assign_labels(
    metrics: pd.DataFrame,
    *,
    qc_q_thresh: float,
    qc_corr_thresh: float,
    qc_corr_p_thresh: float,
) -> pd.DataFrame:
    out = metrics.copy()

    q_sig = pd.to_numeric(out["qc_assoc_q"], errors="coerce") <= float(qc_q_thresh)
    corr_sig = (
        pd.to_numeric(out["max_abs_qc_corr"], errors="coerce") >= float(qc_corr_thresh)
    ) & (
        pd.to_numeric(out["max_abs_qc_corr_p"], errors="coerce")
        <= float(qc_corr_p_thresh)
    )
    out["qc_flag"] = (q_sig | corr_sig).astype(bool)

    under = out["underpowered"].astype(bool)
    bio_sig = pd.to_numeric(out["bio_p"], errors="coerce") <= 0.05

    labels = np.full(out.shape[0], "NOT_SIGNIFICANT", dtype=object)
    labels[under.to_numpy()] = "UNDERPOWERED"
    sig_idx = (~under & bio_sig).to_numpy()

    qc_idx = sig_idx & out["qc_flag"].to_numpy(dtype=bool)
    labels[qc_idx] = "QC_DRIVEN"

    pass_idx = sig_idx & (~out["qc_flag"].to_numpy(dtype=bool))
    labels[pass_idx] = "LOCALIZED_PASS"

    out["final_label"] = labels
    return out


def _prop_ci(k: int, n: int) -> tuple[float, float, float]:
    if int(n) <= 0:
        return float("nan"), float("nan"), float("nan")
    p = float(k) / float(n)
    lo, hi = wilson_ci(int(k), int(n))
    return p, float(lo), float(hi)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    group_cols = [
        "geometry",
        "N",
        "D",
        "sigma_eta",
        "regime",
        "pi_target",
        "beta",
        "gamma",
        "g_qc",
    ]
    for keys, grp in metrics.groupby(group_cols, sort=True):
        n_genes = int(grp.shape[0])
        non_under = grp.loc[~grp["underpowered"].astype(bool)]
        n_non = int(non_under.shape[0])

        sig = non_under.loc[pd.to_numeric(non_under["bio_p"], errors="coerce") <= 0.05]
        n_sig = int(sig.shape[0])

        n_qc_sig = int(sig["qc_flag"].astype(bool).sum())
        n_localized_pass = int((non_under["final_label"] == "LOCALIZED_PASS").sum())

        frac_sig, frac_sig_lo, frac_sig_hi = _prop_ci(n_sig, n_non)
        frac_qc_sig, frac_qc_sig_lo, frac_qc_sig_hi = _prop_ci(n_qc_sig, n_sig)
        frac_local, frac_local_lo, frac_local_hi = _prop_ci(n_localized_pass, n_non)

        true_qc_det = float("nan")
        true_qc_det_lo = float("nan")
        true_qc_det_hi = float("nan")
        true_spatial_rec = float("nan")
        true_spatial_rec_lo = float("nan")
        true_spatial_rec_hi = float("nan")

        regime = str(keys[4])
        if regime == "R2":
            k = int((non_under["final_label"] == "QC_DRIVEN").sum())
            true_qc_det, true_qc_det_lo, true_qc_det_hi = _prop_ci(k, n_non)
        if regime == "R3":
            k = int((non_under["final_label"] == "LOCALIZED_PASS").sum())
            true_spatial_rec, true_spatial_rec_lo, true_spatial_rec_hi = _prop_ci(
                k, n_non
            )

        rows.append(
            {
                "geometry": str(keys[0]),
                "N": int(keys[1]),
                "D": int(keys[2]),
                "sigma_eta": float(keys[3]),
                "regime": regime,
                "pi_target": float(keys[5]),
                "beta": float(keys[6]),
                "gamma": float(keys[7]),
                "g_qc": float(keys[8]),
                "n_genes": int(n_genes),
                "n_non_underpowered": int(n_non),
                "frac_significant": float(frac_sig),
                "frac_significant_ci_low": float(frac_sig_lo),
                "frac_significant_ci_high": float(frac_sig_hi),
                "frac_qc_flag_among_sig": float(frac_qc_sig),
                "frac_qc_flag_among_sig_ci_low": float(frac_qc_sig_lo),
                "frac_qc_flag_among_sig_ci_high": float(frac_qc_sig_hi),
                "frac_localized_pass": float(frac_local),
                "frac_localized_pass_ci_low": float(frac_local_lo),
                "frac_localized_pass_ci_high": float(frac_local_hi),
                "true_qc_detection_rate": float(true_qc_det),
                "true_qc_detection_rate_ci_low": float(true_qc_det_lo),
                "true_qc_detection_rate_ci_high": float(true_qc_det_hi),
                "true_spatial_recall": float(true_spatial_rec),
                "true_spatial_recall_ci_low": float(true_spatial_rec_lo),
                "true_spatial_recall_ci_high": float(true_spatial_rec_hi),
            }
        )

    return pd.DataFrame(rows)


def plot_calibration_r0(
    *,
    metrics: pd.DataFrame,
    outdir: Path,
    n_perm: int,
    bins: int,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    r0 = metrics.loc[
        (metrics["regime"] == "R0") & (~metrics["underpowered"].astype(bool))
    ].copy()
    geometries = sorted(r0["geometry"].astype(str).unique().tolist())
    if not geometries:
        return

    fig_h, axes_h = plt.subplots(
        1,
        len(geometries),
        figsize=(5.0 * len(geometries), 3.6),
        constrained_layout=False,
    )
    axes_h = np.atleast_1d(axes_h)
    for ax, geom in zip(axes_h, geometries):
        pvals = pd.to_numeric(
            r0.loc[r0["geometry"] == geom, "bio_p"], errors="coerce"
        ).to_numpy(dtype=float)
        p_hist(ax, pvals, bins=20, density=True)
        ax.set_title(f"R0 p-hist | geom={geom} | n_perm={n_perm} B={bins}")
        ax.set_xlabel("bio_p")
        ax.set_ylabel("density")
    fig_h.tight_layout()
    savefig(fig_h, plots_dir / "calib_R0_p_hist.png")
    plt.close(fig_h)

    fig_q, axes_q = plt.subplots(
        1,
        len(geometries),
        figsize=(5.0 * len(geometries), 3.6),
        constrained_layout=False,
    )
    axes_q = np.atleast_1d(axes_q)
    for ax, geom in zip(axes_q, geometries):
        pvals = pd.to_numeric(
            r0.loc[r0["geometry"] == geom, "bio_p"], errors="coerce"
        ).to_numpy(dtype=float)
        qq_plot(ax, pvals)
        ax.set_title(f"R0 QQ | geom={geom} | n_perm={n_perm} B={bins}")
        ax.set_xlabel("Expected U(0,1)")
        ax.set_ylabel("Observed bio_p")
    fig_q.tight_layout()
    savefig(fig_q, plots_dir / "calib_R0_qq.png")
    plt.close(fig_q)


def plot_confound_scatter(
    *,
    metrics: pd.DataFrame,
    outdir: Path,
    n_perm: int,
    bins: int,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    df = metrics.loc[
        (~metrics["underpowered"].astype(bool))
        & (metrics["regime"].isin(["R1", "R2", "R3"]))
    ].copy()
    if df.empty:
        return

    eps = 1e-12
    df["x_bio"] = -np.log10(
        np.clip(pd.to_numeric(df["bio_p"], errors="coerce"), eps, 1.0)
    )
    df["y_qc"] = -np.log10(
        np.clip(pd.to_numeric(df["qc_assoc_q"], errors="coerce"), eps, 1.0)
    )

    color_map = {"R1": "#7f7f7f", "R2": "#d62728", "R3": "#1f77b4"}
    marker_map = {0.05: "o", 0.2: "s", 0.6: "^"}

    fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=False)
    for (regime, pi), grp in df.groupby(["regime", "pi_target"], sort=True):
        ax.scatter(
            grp["x_bio"],
            grp["y_qc"],
            s=12,
            alpha=0.38,
            color=color_map.get(str(regime), "#444444"),
            marker=marker_map.get(float(pi), "o"),
            label=f"{regime}, pi={_fmt(float(pi))}",
            linewidths=0.0,
        )

    thr = -math.log10(0.05)
    ax.axvline(thr, color="#222222", linestyle="--", linewidth=0.9)
    ax.axhline(thr, color="#222222", linestyle="--", linewidth=0.9)
    ax.set_xlabel("-log10(bio_p)")
    ax.set_ylabel("-log10(qc_assoc_q)")
    ax.set_title(f"BioRSP vs QC confound signal | n_perm={n_perm} B={bins}")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            by_label[label] = handle
        ax.legend(
            by_label.values(), by_label.keys(), loc="upper left", ncol=2, frameon=False
        )

    fig.tight_layout()
    savefig(fig, plots_dir / "bio_vs_qc_scatter.png")
    plt.close(fig)


def plot_label_rates(
    *,
    metrics: pd.DataFrame,
    outdir: Path,
    n_perm: int,
    bins: int,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    df = metrics.copy()
    df["strength_label"] = "none"
    is_r2 = df["regime"] == "R2"
    is_r3 = df["regime"] == "R3"
    df.loc[is_r2, "strength_label"] = "gamma=" + df.loc[is_r2, "gamma"].map(
        lambda x: _fmt(float(x))
    )
    df.loc[is_r3, "strength_label"] = "beta=" + df.loc[is_r3, "beta"].map(
        lambda x: _fmt(float(x))
    )

    geometries = sorted(df["geometry"].astype(str).unique().tolist())
    if not geometries:
        return

    fig, axes = plt.subplots(
        len(geometries),
        1,
        figsize=(12.0, 3.7 * len(geometries)),
        constrained_layout=False,
        squeeze=False,
    )

    color_map = {
        "UNDERPOWERED": "#b0b0b0",
        "NOT_SIGNIFICANT": "#c7e9c0",
        "QC_DRIVEN": "#fb6a4a",
        "LOCALIZED_PASS": "#3182bd",
    }

    for i, geom in enumerate(geometries):
        ax = axes[i, 0]
        sub = df.loc[df["geometry"] == geom].copy()
        if sub.empty:
            ax.set_axis_off()
            continue

        grp = (
            sub.groupby(["regime", "g_qc", "strength_label", "final_label"], sort=True)
            .size()
            .rename("n")
            .reset_index()
        )
        den = (
            sub.groupby(["regime", "g_qc", "strength_label"], sort=True)
            .size()
            .rename("n_total")
            .reset_index()
        )
        merged = grp.merge(den, on=["regime", "g_qc", "strength_label"], how="left")
        merged["frac"] = merged["n"] / merged["n_total"]

        x_tbl = (
            den.assign(
                x_label=lambda d: d.apply(
                    lambda r: f"{r['regime']}|g={_fmt(float(r['g_qc']))}|{r['strength_label']}",
                    axis=1,
                )
            )
            .sort_values(["regime", "g_qc", "strength_label"])
            .reset_index(drop=True)
        )

        x_labels = x_tbl["x_label"].tolist()
        x_pos = np.arange(len(x_labels), dtype=float)
        bottom = np.zeros(len(x_labels), dtype=float)

        for label in LABEL_ORDER:
            vals = np.zeros(len(x_labels), dtype=float)
            for j, row in x_tbl.iterrows():
                mask = (
                    (merged["regime"] == row["regime"])
                    & (merged["g_qc"] == row["g_qc"])
                    & (merged["strength_label"] == row["strength_label"])
                    & (merged["final_label"] == label)
                )
                if mask.any():
                    vals[j] = float(merged.loc[mask, "frac"].iloc[0])
            ax.bar(
                x_pos,
                vals,
                bottom=bottom,
                color=color_map[label],
                edgecolor="white",
                linewidth=0.2,
                label=label if i == 0 else None,
            )
            bottom += vals

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=40, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Fraction")
        ax.set_title(f"Label rates | geom={geom} | n_perm={n_perm} B={bins}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    savefig(fig, plots_dir / "label_rates_by_regime.png")
    plt.close(fig)


def plot_detection_and_recall(
    *,
    metrics: pd.DataFrame,
    outdir: Path,
    n_perm: int,
    bins: int,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")

    # R2 QC-driven detection vs gamma
    r2 = metrics.loc[
        (metrics["regime"] == "R2") & (~metrics["underpowered"].astype(bool))
    ].copy()
    if not r2.empty:
        grp = (
            r2.groupby(["geometry", "g_qc", "gamma"], sort=True)["final_label"]
            .apply(lambda s: float(np.mean(s == "QC_DRIVEN")))
            .rename("rate")
            .reset_index()
        )
        geoms = sorted(grp["geometry"].astype(str).unique().tolist())
        fig, axes = plt.subplots(
            1, len(geoms), figsize=(5.2 * len(geoms), 3.8), constrained_layout=False
        )
        axes = np.atleast_1d(axes)
        for ax, geom in zip(axes, geoms):
            sub = grp.loc[grp["geometry"] == geom].copy()
            for g_qc, gsub in sub.groupby("g_qc", sort=True):
                gsub = gsub.sort_values("gamma")
                ax.plot(
                    gsub["gamma"],
                    gsub["rate"],
                    marker="o",
                    linewidth=1.5,
                    label=f"g_qc={_fmt(float(g_qc))}",
                )
            ax.axhline(0.8, color="#333333", linestyle="--", linewidth=0.8)
            ax.set_ylim(0.0, 1.02)
            ax.set_xlabel("gamma")
            ax.set_ylabel("QC_DRIVEN detection rate")
            ax.set_title(f"R2 QC detection | geom={geom} | n_perm={n_perm} B={bins}")
            ax.legend(frameon=False)
        fig.tight_layout()
        savefig(fig, plots_dir / "qc_detection_vs_gamma.png")
        plt.close(fig)

    # R3 spatial recall vs beta
    r3 = metrics.loc[
        (metrics["regime"] == "R3") & (~metrics["underpowered"].astype(bool))
    ].copy()
    if not r3.empty:
        grp = (
            r3.groupby(["geometry", "g_qc", "beta"], sort=True)["final_label"]
            .apply(lambda s: float(np.mean(s == "LOCALIZED_PASS")))
            .rename("rate")
            .reset_index()
        )
        geoms = sorted(grp["geometry"].astype(str).unique().tolist())
        fig, axes = plt.subplots(
            1, len(geoms), figsize=(5.2 * len(geoms), 3.8), constrained_layout=False
        )
        axes = np.atleast_1d(axes)
        for ax, geom in zip(axes, geoms):
            sub = grp.loc[grp["geometry"] == geom].copy()
            for g_qc, gsub in sub.groupby("g_qc", sort=True):
                gsub = gsub.sort_values("beta")
                ax.plot(
                    gsub["beta"],
                    gsub["rate"],
                    marker="o",
                    linewidth=1.5,
                    label=f"g_qc={_fmt(float(g_qc))}",
                )
            ax.axhline(0.7, color="#333333", linestyle="--", linewidth=0.8)
            ax.set_ylim(0.0, 1.02)
            ax.set_xlabel("beta")
            ax.set_ylabel("LOCALIZED_PASS recall")
            ax.set_title(f"R3 spatial recall | geom={geom} | n_perm={n_perm} B={bins}")
            ax.legend(frameon=False)
        fig.tight_layout()
        savefig(fig, plots_dir / "spatial_recall_vs_beta.png")
        plt.close(fig)


def _rotate_coords(X: np.ndarray, deg: float) -> np.ndarray:
    rad = float(np.deg2rad(float(deg)))
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    R = np.array([[c, -s], [s, c]], dtype=float)
    return np.asarray(X, dtype=float) @ R.T


def _scale_coords(X: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = np.asarray(X, dtype=float).copy()
    out[:, 0] *= float(sx)
    out[:, 1] *= float(sy)
    return out


def _warp_coords(X: np.ndarray, alpha: float) -> np.ndarray:
    out = np.asarray(X, dtype=float).copy()
    out[:, 0] = out[:, 0] + float(alpha) * np.tanh(out[:, 1])
    return out


def run_distortion_analysis(
    *,
    metrics: pd.DataFrame,
    contexts: dict[tuple[int, str, int, float, float, int], DatasetContext],
    outdir: Path,
    master_seed: int,
    bins: int,
    n_perm_distortion: int,
    subset_per_regime: int,
    moran_k_grid: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    plots_dir = ensure_dir(outdir / "plots")
    results_dir = ensure_dir(outdir / "results")

    warnings_out: list[str] = []

    # Representative condition target from spec.
    target_geometry = "density_gradient_disk"
    target_D = 10
    target_pi = 0.2
    target_g_qc = 1.0

    # Fallbacks if exact target unavailable.
    available = metrics[
        ["geometry", "D", "pi_target", "g_qc", "seed_run", "N", "sigma_eta"]
    ].drop_duplicates()
    if available.empty:
        return pd.DataFrame(), pd.DataFrame(), warnings_out

    def _closest(val: float, arr: np.ndarray) -> float:
        return float(arr[np.argmin(np.abs(arr - float(val)))])

    geoms = available["geometry"].astype(str).unique().tolist()
    geom_use = target_geometry if target_geometry in geoms else geoms[0]
    d_vals = np.sort(
        available.loc[available["geometry"] == geom_use, "D"].astype(float).unique()
    )
    d_use = int(_closest(float(target_D), d_vals))
    pi_vals = np.sort(
        available.loc[
            (available["geometry"] == geom_use) & (available["D"] == d_use), "pi_target"
        ]
        .astype(float)
        .unique()
    )
    pi_use = float(_closest(float(target_pi), pi_vals))
    g_vals = np.sort(
        available.loc[
            (available["geometry"] == geom_use)
            & (available["D"] == d_use)
            & (available["pi_target"] == pi_use),
            "g_qc",
        ]
        .astype(float)
        .unique()
    )
    g_use = float(_closest(float(target_g_qc), g_vals))

    sub = metrics.loc[
        (metrics["geometry"] == geom_use)
        & (metrics["D"] == d_use)
        & (np.isclose(metrics["pi_target"], pi_use))
        & (np.isclose(metrics["g_qc"], g_use))
        & (metrics["regime"].isin(["R2", "R3"]))
        & (~metrics["underpowered"].astype(bool))
    ].copy()
    if sub.empty:
        warnings_out.append("Distortion analysis skipped: representative subset empty.")
        return pd.DataFrame(), pd.DataFrame(), warnings_out

    # Pick one seed_run/N/sigma slice deterministically.
    sub = sub.sort_values(["seed_run", "N", "sigma_eta", "gene_index"]).copy()
    first_slice = sub[["seed_run", "N", "sigma_eta"]].drop_duplicates().iloc[0]
    sub = sub.loc[
        (sub["seed_run"] == int(first_slice["seed_run"]))
        & (sub["N"] == int(first_slice["N"]))
        & (np.isclose(sub["sigma_eta"], float(first_slice["sigma_eta"])))
    ].copy()

    rng_pick = rng_from_seed(stable_seed(int(master_seed), "expF", "distortion_subset"))
    picks: list[pd.DataFrame] = []
    for regime in ["R2", "R3"]:
        rsub = sub.loc[sub["regime"] == regime].copy()
        if rsub.empty:
            continue
        n_take = int(min(int(subset_per_regime), rsub.shape[0]))
        idx = rng_pick.choice(
            rsub.index.to_numpy(dtype=int), size=n_take, replace=False
        )
        picks.append(rsub.loc[idx].copy())
    if not picks:
        warnings_out.append(
            "Distortion analysis skipped: no genes available after regime sampling."
        )
        return pd.DataFrame(), pd.DataFrame(), warnings_out

    picked = pd.concat(picks, axis=0, ignore_index=True)

    key0 = (
        int(first_slice["seed_run"]),
        str(geom_use),
        int(d_use),
        float(first_slice["sigma_eta"]),
        float(g_use),
        int(first_slice["N"]),
    )
    context = contexts[key0]

    distortions: dict[str, np.ndarray] = {}
    x_rot = _rotate_coords(context.X, 37.0)
    x_scl = _scale_coords(context.X, sx=1.5, sy=0.8)
    x_wrp = _warp_coords(context.X, alpha=0.3)
    distortions["ROTATE"] = np.mod(np.arctan2(x_rot[:, 1], x_rot[:, 0]), 2.0 * np.pi)
    distortions["SCALE"] = np.mod(np.arctan2(x_scl[:, 1], x_scl[:, 0]), 2.0 * np.pi)
    distortions["WARP"] = np.mod(np.arctan2(x_wrp[:, 1], x_wrp[:, 0]), 2.0 * np.pi)

    distortion_bin_cache = {
        name: _compute_bin_cache(theta, int(bins))
        for name, theta in distortions.items()
    }

    dist_rows: list[dict[str, Any]] = []
    moran_rows: list[dict[str, Any]] = []

    # Pre-build kNN graphs for optional Moran stability on representative context.
    moran_graphs: dict[int, sp.csr_matrix] = {}
    for k in sorted(set(int(kv) for kv in moran_k_grid if int(kv) > 0)):
        moran_graphs[int(k)] = _build_knn_weights(context.X, int(k))

    for _, row in picked.iterrows():
        regime = str(row["regime"])
        seed_gene = int(row["seed_gene"])
        pi_target = float(row["pi_target"])
        gamma = float(row["gamma"])
        beta = float(row["beta"])
        theta0 = float(row["theta0"])

        f = _simulate_gene_foreground(
            context=context,
            regime=regime,
            pi_target=pi_target,
            gamma=gamma,
            beta=beta,
            theta0=theta0,
            seed_gene=seed_gene,
        )

        # Moran graph-choice stability (baseline sensitivity check).
        for k, W in moran_graphs.items():
            try:
                moran_val = float(morans_i(f.astype(float), W, row_standardize=True))
            except Exception:
                moran_val = float("nan")
            moran_rows.append(
                {
                    "gene_index": int(row["gene_index"]),
                    "regime": regime,
                    "k": int(k),
                    "moran_I": float(moran_val),
                    "geometry": str(geom_use),
                    "D": int(d_use),
                    "pi_target": float(pi_use),
                    "g_qc": float(g_use),
                }
            )

        for dist_name, theta_dist in distortions.items():
            bin_id, bin_counts = distortion_bin_cache[dist_name]
            seed_perm = stable_seed(
                int(master_seed),
                "expF",
                "distortion",
                int(row["gene_index"]),
                str(dist_name),
            )
            perm = perm_null_T(
                f=f,
                angles=theta_dist,
                donor_ids=context.donor_ids,
                n_bins=int(bins),
                n_perm=int(n_perm_distortion),
                seed=int(seed_perm),
                mode="raw",
                smooth_w=1,
                donor_stratified=True,
                return_null_T=True,
                return_obs_profile=False,
                bin_id=bin_id,
                bin_counts_total=bin_counts,
            )
            z_dist = float(
                robust_z(float(perm["T_obs"]), np.asarray(perm["null_T"], dtype=float))
            )

            dist_rows.append(
                {
                    "gene_index": int(row["gene_index"]),
                    "regime": regime,
                    "distortion": str(dist_name),
                    "bio_Z_orig": float(row["bio_Z"]),
                    "bio_Z_dist": float(z_dist),
                    "qc_flag_orig": bool(row["qc_flag"]),
                    "qc_flag_dist": bool(row["qc_flag"]),
                    "qc_flag_agree": True,
                    "geometry": str(geom_use),
                    "D": int(d_use),
                    "pi_target": float(pi_use),
                    "g_qc": float(g_use),
                    "n_perm_distortion": int(n_perm_distortion),
                }
            )

    distortion_df = pd.DataFrame(dist_rows)
    moran_df = pd.DataFrame(moran_rows)

    if not distortion_df.empty:
        atomic_write_csv(results_dir / "distortion_metrics.csv", distortion_df)

        # C5a bio_Z stability scatter.
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=False)
        for ax, dist_name in zip(axes, ["ROTATE", "SCALE", "WARP"]):
            subd = distortion_df.loc[distortion_df["distortion"] == dist_name].copy()
            if subd.empty:
                ax.set_axis_off()
                continue
            for regime, gsub in subd.groupby("regime", sort=True):
                ax.scatter(
                    gsub["bio_Z_orig"],
                    gsub["bio_Z_dist"],
                    s=14,
                    alpha=0.45,
                    label=str(regime),
                )
            xy = pd.concat([subd["bio_Z_orig"], subd["bio_Z_dist"]], axis=0)
            finite = xy[np.isfinite(xy.to_numpy(dtype=float))]
            if finite.size > 0:
                lo = float(np.min(finite))
                hi = float(np.max(finite))
                ax.plot(
                    [lo, hi], [lo, hi], linestyle="--", color="#222222", linewidth=0.9
                )
            ax.set_title(f"{dist_name} | n_perm={n_perm_distortion}")
            ax.set_xlabel("bio_Z original")
            ax.set_ylabel("bio_Z distorted")
            ax.legend(frameon=False)
        fig.suptitle(
            (
                "Embedding distortion stability of BioRSP Z "
                f"| geom={geom_use}, D={d_use}, pi={_fmt(pi_use)}, g_qc={_fmt(g_use)}"
            )
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        savefig(fig, plots_dir / "distortion_stability_bioZ.png")
        plt.close(fig)

        # C5b qc_flag agreement rate.
        agree = (
            distortion_df.groupby(["regime", "distortion"], sort=True)["qc_flag_agree"]
            .mean()
            .reset_index(name="agreement")
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=False)
        regimes = ["R2", "R3"]
        dist_names = ["ROTATE", "SCALE", "WARP"]
        x = np.arange(len(dist_names), dtype=float)
        width = 0.35
        for i, regime in enumerate(regimes):
            vals = []
            for dname in dist_names:
                subp = agree.loc[
                    (agree["regime"] == regime) & (agree["distortion"] == dname),
                    "agreement",
                ]
                vals.append(float(subp.iloc[0]) if not subp.empty else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=regime)
        ax.axhline(0.9, color="#222222", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(dist_names)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("QC flag agreement")
        ax.set_title(
            (
                "QC-flag stability across embedding distortions "
                f"| geom={geom_use}, D={d_use}, pi={_fmt(pi_use)}, g_qc={_fmt(g_use)}"
            )
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        savefig(fig, plots_dir / "distortion_stability_qcflag.png")
        plt.close(fig)

    if not moran_df.empty:
        # Save Moran graph-choice sensitivity summary.
        base_k = min(moran_graphs.keys())
        piv = moran_df.pivot_table(
            index=["gene_index", "regime"],
            columns="k",
            values="moran_I",
            aggfunc="first",
        )
        stab_rows: list[dict[str, Any]] = []
        for regime, grp in piv.groupby(level=1, sort=True):
            if base_k not in grp.columns:
                continue
            base = grp[base_k].to_numpy(dtype=float)
            for k in sorted(moran_graphs.keys()):
                if k == base_k or k not in grp.columns:
                    continue
                arr = grp[k].to_numpy(dtype=float)
                mask = np.isfinite(base) & np.isfinite(arr)
                if int(np.sum(mask)) < 3:
                    rho = float("nan")
                else:
                    rr, _ = spearmanr(base[mask], arr[mask], nan_policy="omit")
                    rho = float(rr) if rr is not None else float("nan")
                stab_rows.append(
                    {
                        "regime": str(regime),
                        "k_base": int(base_k),
                        "k_compare": int(k),
                        "spearman_rho": float(rho),
                        "n": int(np.sum(mask)),
                    }
                )
        moran_stab = pd.DataFrame(stab_rows)
        atomic_write_csv(results_dir / "moran_k_metrics.csv", moran_df)
        atomic_write_csv(results_dir / "moran_k_stability.csv", moran_stab)

    return distortion_df, moran_df, warnings_out


def _binned_detection_vs_cov(
    cov: np.ndarray, f: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    cov_arr = np.asarray(cov, dtype=float).ravel()
    f_arr = np.asarray(f, dtype=bool).ravel().astype(float)
    if cov_arr.size != f_arr.size or cov_arr.size < 5:
        return np.array([], dtype=float), np.array([], dtype=float)

    edges = np.quantile(cov_arr, np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return np.array([], dtype=float), np.array([], dtype=float)

    x_out: list[float] = []
    y_out: list[float] = []
    for i in range(edges.size - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i < edges.size - 2:
            mask = (cov_arr >= lo) & (cov_arr < hi)
        else:
            mask = (cov_arr >= lo) & (cov_arr <= hi)
        if int(np.sum(mask)) < 10:
            continue
        x_out.append(float(i + 1))
        y_out.append(float(np.mean(f_arr[mask])))
    return np.asarray(x_out, dtype=float), np.asarray(y_out, dtype=float)


def _plot_example_gene_panel(
    *,
    row: pd.Series,
    context: DatasetContext,
    outpath: Path,
    bins: int,
    n_perm: int,
) -> None:
    regime = str(row["regime"])
    f = _simulate_gene_foreground(
        context=context,
        regime=regime,
        pi_target=float(row["pi_target"]),
        gamma=float(row["gamma"]),
        beta=float(row["beta"]),
        theta0=float(row["theta0"]),
        seed_gene=int(row["seed_gene"]),
    )

    E_phi, _, _ = compute_rsp_profile_from_boolean(
        f,
        context.theta,
        int(bins),
        bin_id=context.bin_id,
        bin_counts_total=context.bin_counts_total,
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), constrained_layout=False)

    # Embedding + foreground map.
    ax0 = axes[0]
    plot_embedding_with_foreground(
        context.X,
        f,
        ax=ax0,
        title="Embedding foreground",
        s=5.0,
        alpha_bg=0.35,
        alpha_fg=0.7,
    )

    # QC covariate binned trend.
    ax1 = axes[1]
    covs = {
        "log_library": context.log_library_size,
        "pct_mt": context.pct_mt,
        "ribo": context.ribo_score,
    }
    color_cov = {"log_library": "#2ca02c", "pct_mt": "#ff7f0e", "ribo": "#9467bd"}
    for name, cov in covs.items():
        xb, yb = _binned_detection_vs_cov(cov, f, n_bins=10)
        if xb.size > 0:
            ax1.plot(
                xb, yb, marker="o", linewidth=1.2, label=name, color=color_cov[name]
            )
    ax1.set_xlabel("Covariate quantile bin")
    ax1.set_ylabel("Pr(f=1)")
    ax1.set_title("QC covariate association")
    ax1.legend(frameon=False)

    # E(theta) profile.
    gs = axes[2].get_gridspec()
    axes[2].remove()
    ax2 = fig.add_subplot(gs[0, 2], projection="polar")
    centers = np.linspace(0.0, 2.0 * np.pi, int(bins), endpoint=False)
    plot_rsp_polar(
        centers,
        E_phi,
        ax=ax2,
        color="#111111",
        linewidth=1.3,
        title="BioRSP profile (polar)",
    )

    fig.suptitle(
        (
            f"{regime} example | N={context.N}, D={context.D}, sigma_eta={_fmt(context.sigma_eta)}, "
            f"n_perm={n_perm}, pi={_fmt(float(row['pi_target']))}, g_qc={_fmt(float(row['g_qc']))}"
        )
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    savefig(fig, outpath)
    plt.close(fig)


def plot_example_panels(
    *,
    metrics: pd.DataFrame,
    contexts: dict[tuple[int, str, int, float, float, int], DatasetContext],
    outdir: Path,
    master_seed: int,
    bins: int,
    n_perm: int,
) -> list[str]:
    plots_dir = ensure_dir(outdir / "plots")
    warnings_out: list[str] = []

    rng = rng_from_seed(stable_seed(int(master_seed), "expF", "example_panels"))

    qc_candidates = metrics.loc[
        (metrics["regime"] == "R2")
        & (metrics["final_label"] == "QC_DRIVEN")
        & (~metrics["underpowered"].astype(bool))
    ].copy()
    spatial_candidates = metrics.loc[
        (metrics["regime"] == "R3")
        & (metrics["final_label"] == "LOCALIZED_PASS")
        & (~metrics["underpowered"].astype(bool))
    ].copy()

    def _pick(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if df.empty:
            return df
        n_take = int(min(int(n), df.shape[0]))
        idx = rng.choice(df.index.to_numpy(dtype=int), size=n_take, replace=False)
        return df.loc[idx].copy()

    pick_qc = _pick(qc_candidates, 2)
    pick_sp = _pick(spatial_candidates, 2)

    if pick_qc.shape[0] < 2:
        warnings_out.append(
            "Fewer than 2 QC-driven genes available for example panels."
        )
    if pick_sp.shape[0] < 2:
        warnings_out.append(
            "Fewer than 2 true spatial genes available for example panels."
        )

    for i, (_, row) in enumerate(pick_qc.reset_index(drop=True).iterrows(), start=1):
        context = _context_from_key(contexts, row)
        _plot_example_gene_panel(
            row=row,
            context=context,
            outpath=plots_dir / f"example_panels_qc_driven_{i}.png",
            bins=int(bins),
            n_perm=int(n_perm),
        )

    for i, (_, row) in enumerate(pick_sp.reset_index(drop=True).iterrows(), start=1):
        context = _context_from_key(contexts, row)
        _plot_example_gene_panel(
            row=row,
            context=context,
            outpath=plots_dir / f"example_panels_true_spatial_{i}.png",
            bins=int(bins),
            n_perm=int(n_perm),
        )

    return warnings_out


def run_validations(
    *,
    metrics: pd.DataFrame,
    distortion_df: pd.DataFrame,
    moran_df: pd.DataFrame,
    outdir: Path,
) -> list[str]:
    results_dir = ensure_dir(outdir / "results")
    warnings_out: list[str] = []
    report_lines: list[str] = []

    valid = metrics.loc[~metrics["underpowered"].astype(bool)].copy()
    if valid.empty:
        msg = "No non-underpowered genes available; validation checks skipped."
        report_lines.append(msg)
        warnings_out.append(msg)
        dbg_path = results_dir / "validation_debug_report.txt"
        dbg_path.write_text("\n".join(report_lines), encoding="utf-8")
        return warnings_out

    # R0 calibration and low qc-flag rate.
    r0 = valid.loc[valid["regime"] == "R0"].copy()
    if r0.empty:
        msg = "R0 validation skipped: no R0 genes found (likely g_qc grid did not include 0)."
        report_lines.append(msg)
        warnings_out.append(msg)
    else:
        for geom, grp in r0.groupby("geometry", sort=True):
            n = int(grp.shape[0])
            sig = float(np.mean(pd.to_numeric(grp["bio_p"], errors="coerce") <= 0.05))
            qc_rate = float(np.mean(grp["qc_flag"].astype(bool)))
            report_lines.append(
                f"R0 geom={geom}: n={n}, frac_significant={sig:.3f}, frac_qc_flag={qc_rate:.3f}"
            )
            if abs(sig - 0.05) > 0.03:
                warnings_out.append(
                    f"R0 calibration warning geom={geom}: significant rate {sig:.3f} deviates from 0.05."
                )
            if qc_rate > 0.10:
                warnings_out.append(
                    f"R0 qc-flag warning geom={geom}: qc_flag rate {qc_rate:.3f} > 0.10."
                )

    # R2 confound detection threshold.
    r2 = valid.loc[
        (valid["regime"] == "R2") & (valid["g_qc"] >= 0.5) & (valid["gamma"] >= 1.0)
    ].copy()
    if not r2.empty:
        for keys, grp in r2.groupby(["geometry", "g_qc", "gamma"], sort=True):
            det = float(np.mean(grp["final_label"] == "QC_DRIVEN"))
            report_lines.append(
                f"R2 detection geom={keys[0]}, g_qc={_fmt(float(keys[1]))}, gamma={_fmt(float(keys[2]))}: {det:.3f}"
            )
            if det < 0.8:
                warnings_out.append(
                    (
                        "R2 detection warning "
                        f"geom={keys[0]}, g_qc={_fmt(float(keys[1]))}, gamma={_fmt(float(keys[2]))}: "
                        f"QC_DRIVEN={det:.3f} (<0.8)."
                    )
                )

    # R3 recall threshold.
    r3 = valid.loc[
        (valid["regime"] == "R3") & (valid["g_qc"] >= 0.5) & (valid["beta"] >= 1.0)
    ].copy()
    if not r3.empty:
        for keys, grp in r3.groupby(["geometry", "g_qc", "beta"], sort=True):
            rec = float(np.mean(grp["final_label"] == "LOCALIZED_PASS"))
            report_lines.append(
                f"R3 recall geom={keys[0]}, g_qc={_fmt(float(keys[1]))}, beta={_fmt(float(keys[2]))}: {rec:.3f}"
            )
            if rec < 0.7:
                warnings_out.append(
                    (
                        "R3 recall warning "
                        f"geom={keys[0]}, g_qc={_fmt(float(keys[1]))}, beta={_fmt(float(keys[2]))}: "
                        f"LOCALIZED_PASS={rec:.3f} (<0.7)."
                    )
                )

    # Distortion stability thresholds.
    if distortion_df.empty:
        msg = "Distortion validation skipped: distortion_metrics.csv is empty."
        report_lines.append(msg)
        warnings_out.append(msg)
    else:
        for keys, grp in distortion_df.groupby(["regime", "distortion"], sort=True):
            x = pd.to_numeric(grp["bio_Z_orig"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(grp["bio_Z_dist"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            corr = (
                float(np.corrcoef(x[mask], y[mask])[0, 1])
                if int(np.sum(mask)) >= 3
                else float("nan")
            )
            agree = float(np.mean(grp["qc_flag_agree"].astype(bool)))
            report_lines.append(
                f"Distortion regime={keys[0]}, type={keys[1]}: corr_Z={corr:.3f}, qc_agree={agree:.3f}, n={int(grp.shape[0])}"
            )
            if keys[0] == "R3" and np.isfinite(corr) and corr < 0.9:
                warnings_out.append(
                    f"Distortion warning: R3 corr(bio_Z orig vs {keys[1]})={corr:.3f} < 0.9."
                )
            if agree < 0.9:
                warnings_out.append(
                    f"Distortion warning: qc_flag agreement for {keys[0]} {keys[1]} is {agree:.3f} < 0.9."
                )

    # Moran kNN graph-choice stability (if baseline computed).
    if not moran_df.empty:
        piv = moran_df.pivot_table(
            index=["gene_index", "regime"],
            columns="k",
            values="moran_I",
            aggfunc="first",
        )
        ks = sorted([int(k) for k in piv.columns.tolist()])
        if len(ks) >= 2:
            base_k = ks[0]
            for regime, grp in piv.groupby(level=1, sort=True):
                base = grp[base_k].to_numpy(dtype=float)
                for k in ks[1:]:
                    arr = grp[k].to_numpy(dtype=float)
                    mask = np.isfinite(base) & np.isfinite(arr)
                    rho = float("nan")
                    if int(np.sum(mask)) >= 3:
                        rr, _ = spearmanr(base[mask], arr[mask], nan_policy="omit")
                        rho = float(rr) if rr is not None else float("nan")
                    report_lines.append(
                        f"Moran stability regime={regime}: k={base_k} vs k={k}, rho={rho:.3f}, n={int(np.sum(mask))}"
                    )

    if warnings_out:
        report_lines.append("")
        report_lines.append("Validation warnings:")
        report_lines.extend([f"- {w}" for w in warnings_out])
    else:
        report_lines.append("All built-in validation checks passed.")

    dbg_path = results_dir / "validation_debug_report.txt"
    dbg_path.write_text("\n".join(report_lines), encoding="utf-8")

    if warnings_out:
        print(f"Validation warnings detected. See {dbg_path}", flush=True)
    else:
        print("Validation checks passed without warnings.", flush=True)

    return warnings_out


def run_experiment(args: argparse.Namespace) -> None:
    outdir = ensure_dir(args.outdir)
    results_dir = ensure_dir(outdir / "results")
    ensure_dir(outdir / "plots")

    run_id = timestamped_run_id(prefix="expF")
    n_perm_eff = int(args.n_perm)
    min_perm_eff = int(args.min_perm)
    if n_perm_eff < min_perm_eff:
        warnings.warn(
            f"n_perm ({n_perm_eff}) < min_perm ({min_perm_eff}); lowering min_perm gate to {n_perm_eff}.",
            RuntimeWarning,
            stacklevel=2,
        )
        min_perm_eff = int(n_perm_eff)

    cfg = {
        "experiment": "expF_confound_resistance",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit_hash(cwd=REPO_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "master_seed": int(args.master_seed),
        "seed_run": int(args.master_seed),
        "N": int(args.N),
        "geometries": [str(x) for x in args.geometries],
        "D_grid": [int(x) for x in args.D_grid],
        "sigma_eta_grid": [float(x) for x in args.sigma_eta_grid],
        "pi_grid": [float(x) for x in args.pi_grid],
        "g_qc_grid": [float(x) for x in args.g_qc_grid],
        "gamma_grid": [float(x) for x in args.gamma_grid],
        "beta_grid": [float(x) for x in args.beta_grid],
        "genes_per_condition": int(args.genes_per_condition),
        "n_perm": int(args.n_perm),
        "bins": int(args.bins),
        "qc_test": str(args.qc_test),
        "underpowered": {
            "p_min": float(args.p_min),
            "min_fg_total": int(args.min_fg_total),
            "min_fg_per_donor": int(args.min_fg_per_donor),
            "min_bg_per_donor": int(args.min_bg_per_donor),
            "d_eff_min": int(args.d_eff_min),
            "min_perm": int(min_perm_eff),
        },
        "qc_flag_thresholds": {
            "qc_assoc_q": float(args.qc_q_thresh),
            "max_abs_qc_corr": float(args.qc_corr_thresh),
            "max_abs_qc_corr_p": float(args.qc_corr_p_thresh),
        },
        "distortion": {
            "n_perm": int(args.distortion_n_perm),
            "subset_per_regime": int(args.distortion_subset_per_regime),
            "rotate_degrees": 37.0,
            "scale_x": 1.5,
            "scale_y": 0.8,
            "warp_alpha": 0.3,
        },
        "moran_k_grid": [int(x) for x in args.moran_k_grid],
    }
    write_config(outdir, cfg)

    print(
        (
            "Running ExpF with "
            f"N={int(args.N)}, geometries={list(args.geometries)}, D_grid={list(args.D_grid)}, "
            f"g_qc_grid={list(args.g_qc_grid)}, genes_per_condition={int(args.genes_per_condition)}, "
            f"n_perm={int(args.n_perm)}, bins={int(args.bins)}, qc_test={str(args.qc_test)}"
        ),
        flush=True,
    )

    contexts: dict[tuple[int, str, int, float, float, int], DatasetContext] = {}
    dataset_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    seed_run = int(args.master_seed)
    start_t = time.time()
    gene_counter = 0

    # Estimate total genes (for progress only).
    n_dataset = (
        len(args.geometries)
        * len(args.D_grid)
        * len(args.sigma_eta_grid)
        * len(args.g_qc_grid)
    )
    n_r0 = len(args.pi_grid)
    n_r1 = len(args.pi_grid)
    n_r2 = len(args.pi_grid) * len(args.gamma_grid)
    n_r3 = len(args.pi_grid) * len(args.beta_grid)
    genes_per_dataset_avg = n_r1 + n_r2 + n_r3 + (n_r0 / max(1, len(args.g_qc_grid)))
    total_genes_est = int(
        n_dataset * genes_per_dataset_avg * int(args.genes_per_condition)
    )

    for geometry in args.geometries:
        for D in args.D_grid:
            for sigma_eta in args.sigma_eta_grid:
                for g_qc in args.g_qc_grid:
                    context, qc_corr = _simulate_dataset_context(
                        master_seed=int(args.master_seed),
                        seed_run=int(seed_run),
                        geometry=str(geometry),
                        N=int(args.N),
                        D=int(D),
                        sigma_eta=float(sigma_eta),
                        g_qc=float(g_qc),
                        n_bins=int(args.bins),
                        density_k=float(args.density_k),
                        mu0_library=float(args.mu0_library),
                        sigma_library=float(args.sigma_library),
                        mt_mean=float(args.mt_mean),
                        mt_concentration=float(args.mt_concentration),
                    )
                    contexts[_dataset_key(context)] = context
                    dataset_rows.append(
                        {
                            "seed_run": int(seed_run),
                            "geometry": str(geometry),
                            "N": int(args.N),
                            "D": int(D),
                            "sigma_eta": float(sigma_eta),
                            "g_qc": float(g_qc),
                            **{k: float(v) for k, v in qc_corr.items()},
                        }
                    )

                    print(
                        (
                            "Dataset "
                            f"geom={geometry}, D={int(D)}, sigma_eta={_fmt(float(sigma_eta))}, g_qc={_fmt(float(g_qc))}, "
                            f"max|corr(QC,x/y)|={_fmt(float(qc_corr['max_abs_corr']))}"
                        ),
                        flush=True,
                    )

                    # R0 and R1 plans depend on g_qc.
                    base_plans = _regime_plan_for_gqc(float(g_qc))

                    # Run R0/R1 (no QC- or spatial-driven genes).
                    for regime, gamma_base, beta_base in base_plans:
                        for pi_target in args.pi_grid:
                            for rep in range(int(args.genes_per_condition)):
                                seed_gene = stable_seed(
                                    int(args.master_seed),
                                    "expF",
                                    "gene",
                                    int(seed_run),
                                    str(geometry),
                                    int(args.N),
                                    int(D),
                                    float(sigma_eta),
                                    float(g_qc),
                                    str(regime),
                                    float(pi_target),
                                    float(gamma_base),
                                    float(beta_base),
                                    int(rep),
                                )
                                theta0 = float(
                                    rng_from_seed(
                                        stable_seed(int(seed_gene), "expF", "theta0")
                                    ).uniform(0.0, 2.0 * np.pi)
                                )
                                f = _simulate_gene_foreground(
                                    context=context,
                                    regime=str(regime),
                                    pi_target=float(pi_target),
                                    gamma=float(gamma_base),
                                    beta=float(beta_base),
                                    theta0=float(theta0),
                                    seed_gene=int(seed_gene),
                                )

                                power = evaluate_underpowered(
                                    donor_ids=context.donor_ids,
                                    f=f,
                                    n_perm=int(n_perm_eff),
                                    p_min=float(args.p_min),
                                    min_fg_total=int(args.min_fg_total),
                                    min_fg_per_donor=int(args.min_fg_per_donor),
                                    min_bg_per_donor=int(args.min_bg_per_donor),
                                    d_eff_min=int(args.d_eff_min),
                                    min_perm=int(min_perm_eff),
                                )
                                underpowered = bool(power["underpowered"])

                                bio_T = float("nan")
                                bio_p = float("nan")
                                bio_Z = float("nan")
                                if not underpowered:
                                    seed_perm = stable_seed(
                                        int(args.master_seed),
                                        "expF",
                                        "perm",
                                        int(seed_gene),
                                    )
                                    perm = perm_null_T(
                                        f=f,
                                        angles=context.theta,
                                        donor_ids=context.donor_ids,
                                        n_bins=int(args.bins),
                                        n_perm=int(n_perm_eff),
                                        seed=int(seed_perm),
                                        mode="raw",
                                        smooth_w=1,
                                        donor_stratified=True,
                                        return_null_T=True,
                                        return_obs_profile=False,
                                        bin_id=context.bin_id,
                                        bin_counts_total=context.bin_counts_total,
                                    )
                                    bio_T = float(perm["T_obs"])
                                    bio_p = float(perm["p_T"])
                                    bio_Z = float(
                                        robust_z(
                                            float(bio_T),
                                            np.asarray(perm["null_T"], dtype=float),
                                        )
                                    )

                                if str(args.qc_test) == "logistic":
                                    qc = qc_assoc_test_logistic(f=f, context=context)
                                else:
                                    qc = qc_assoc_test_spearman(f=f, context=context)

                                metric_rows.append(
                                    {
                                        "run_id": run_id,
                                        "seed": int(seed_gene),
                                        "seed_run": int(seed_run),
                                        "seed_gene": int(seed_gene),
                                        "gene_index": int(gene_counter),
                                        "geometry": str(geometry),
                                        "N": int(args.N),
                                        "D": int(D),
                                        "sigma_eta": float(sigma_eta),
                                        "regime": str(regime),
                                        "pi_target": float(pi_target),
                                        "beta": float(beta_base),
                                        "gamma": float(gamma_base),
                                        "g_qc": float(g_qc),
                                        "theta0": float(theta0),
                                        "prev_obs": float(np.mean(f)),
                                        "n_fg_total": int(np.sum(f)),
                                        "D_eff": int(power["D_eff"]),
                                        "underpowered": bool(underpowered),
                                        "bio_T": float(bio_T),
                                        "bio_p": float(bio_p),
                                        "bio_Z": float(bio_Z),
                                        "qc_assoc_stat": float(qc["qc_assoc_stat"]),
                                        "qc_assoc_p": float(qc["qc_assoc_p"]),
                                        "max_abs_qc_corr": float(qc["max_abs_qc_corr"]),
                                        "max_abs_qc_corr_p": float(
                                            qc["max_abs_qc_corr_p"]
                                        ),
                                        "rho_log_library": float(qc["rho_log_library"]),
                                        "rho_pct_mt": float(qc["rho_pct_mt"]),
                                        "rho_ribo": float(qc["rho_ribo"]),
                                    }
                                )
                                gene_counter += 1
                                if gene_counter % int(args.progress_every) == 0:
                                    elapsed = time.time() - start_t
                                    rate = (
                                        gene_counter / elapsed
                                        if elapsed > 0
                                        else float("nan")
                                    )
                                    print(
                                        f"  progress: genes={gene_counter} (~{rate:.2f} genes/s), est_total~{total_genes_est}",
                                        flush=True,
                                    )

                    # R2 QC-driven confound genes.
                    for pi_target in args.pi_grid:
                        for gamma in args.gamma_grid:
                            for rep in range(int(args.genes_per_condition)):
                                seed_gene = stable_seed(
                                    int(args.master_seed),
                                    "expF",
                                    "gene",
                                    int(seed_run),
                                    str(geometry),
                                    int(args.N),
                                    int(D),
                                    float(sigma_eta),
                                    float(g_qc),
                                    "R2",
                                    float(pi_target),
                                    float(gamma),
                                    int(rep),
                                )
                                theta0 = float(
                                    rng_from_seed(
                                        stable_seed(int(seed_gene), "expF", "theta0")
                                    ).uniform(0.0, 2.0 * np.pi)
                                )
                                f = _simulate_gene_foreground(
                                    context=context,
                                    regime="R2",
                                    pi_target=float(pi_target),
                                    gamma=float(gamma),
                                    beta=0.0,
                                    theta0=float(theta0),
                                    seed_gene=int(seed_gene),
                                )

                                power = evaluate_underpowered(
                                    donor_ids=context.donor_ids,
                                    f=f,
                                    n_perm=int(n_perm_eff),
                                    p_min=float(args.p_min),
                                    min_fg_total=int(args.min_fg_total),
                                    min_fg_per_donor=int(args.min_fg_per_donor),
                                    min_bg_per_donor=int(args.min_bg_per_donor),
                                    d_eff_min=int(args.d_eff_min),
                                    min_perm=int(min_perm_eff),
                                )
                                underpowered = bool(power["underpowered"])

                                bio_T = float("nan")
                                bio_p = float("nan")
                                bio_Z = float("nan")
                                if not underpowered:
                                    seed_perm = stable_seed(
                                        int(args.master_seed),
                                        "expF",
                                        "perm",
                                        int(seed_gene),
                                    )
                                    perm = perm_null_T(
                                        f=f,
                                        angles=context.theta,
                                        donor_ids=context.donor_ids,
                                        n_bins=int(args.bins),
                                        n_perm=int(n_perm_eff),
                                        seed=int(seed_perm),
                                        mode="raw",
                                        smooth_w=1,
                                        donor_stratified=True,
                                        return_null_T=True,
                                        return_obs_profile=False,
                                        bin_id=context.bin_id,
                                        bin_counts_total=context.bin_counts_total,
                                    )
                                    bio_T = float(perm["T_obs"])
                                    bio_p = float(perm["p_T"])
                                    bio_Z = float(
                                        robust_z(
                                            float(bio_T),
                                            np.asarray(perm["null_T"], dtype=float),
                                        )
                                    )

                                if str(args.qc_test) == "logistic":
                                    qc = qc_assoc_test_logistic(f=f, context=context)
                                else:
                                    qc = qc_assoc_test_spearman(f=f, context=context)

                                metric_rows.append(
                                    {
                                        "run_id": run_id,
                                        "seed": int(seed_gene),
                                        "seed_run": int(seed_run),
                                        "seed_gene": int(seed_gene),
                                        "gene_index": int(gene_counter),
                                        "geometry": str(geometry),
                                        "N": int(args.N),
                                        "D": int(D),
                                        "sigma_eta": float(sigma_eta),
                                        "regime": "R2",
                                        "pi_target": float(pi_target),
                                        "beta": 0.0,
                                        "gamma": float(gamma),
                                        "g_qc": float(g_qc),
                                        "theta0": float(theta0),
                                        "prev_obs": float(np.mean(f)),
                                        "n_fg_total": int(np.sum(f)),
                                        "D_eff": int(power["D_eff"]),
                                        "underpowered": bool(underpowered),
                                        "bio_T": float(bio_T),
                                        "bio_p": float(bio_p),
                                        "bio_Z": float(bio_Z),
                                        "qc_assoc_stat": float(qc["qc_assoc_stat"]),
                                        "qc_assoc_p": float(qc["qc_assoc_p"]),
                                        "max_abs_qc_corr": float(qc["max_abs_qc_corr"]),
                                        "max_abs_qc_corr_p": float(
                                            qc["max_abs_qc_corr_p"]
                                        ),
                                        "rho_log_library": float(qc["rho_log_library"]),
                                        "rho_pct_mt": float(qc["rho_pct_mt"]),
                                        "rho_ribo": float(qc["rho_ribo"]),
                                    }
                                )
                                gene_counter += 1
                                if gene_counter % int(args.progress_every) == 0:
                                    elapsed = time.time() - start_t
                                    rate = (
                                        gene_counter / elapsed
                                        if elapsed > 0
                                        else float("nan")
                                    )
                                    print(
                                        f"  progress: genes={gene_counter} (~{rate:.2f} genes/s), est_total~{total_genes_est}",
                                        flush=True,
                                    )

                    # R3 true spatial genes with mild QC covariate dependence.
                    for pi_target in args.pi_grid:
                        for beta in args.beta_grid:
                            for rep in range(int(args.genes_per_condition)):
                                seed_gene = stable_seed(
                                    int(args.master_seed),
                                    "expF",
                                    "gene",
                                    int(seed_run),
                                    str(geometry),
                                    int(args.N),
                                    int(D),
                                    float(sigma_eta),
                                    float(g_qc),
                                    "R3",
                                    float(pi_target),
                                    float(beta),
                                    int(rep),
                                )
                                theta0 = float(
                                    rng_from_seed(
                                        stable_seed(int(seed_gene), "expF", "theta0")
                                    ).uniform(0.0, 2.0 * np.pi)
                                )
                                f = _simulate_gene_foreground(
                                    context=context,
                                    regime="R3",
                                    pi_target=float(pi_target),
                                    gamma=0.0,
                                    beta=float(beta),
                                    theta0=float(theta0),
                                    seed_gene=int(seed_gene),
                                )

                                power = evaluate_underpowered(
                                    donor_ids=context.donor_ids,
                                    f=f,
                                    n_perm=int(n_perm_eff),
                                    p_min=float(args.p_min),
                                    min_fg_total=int(args.min_fg_total),
                                    min_fg_per_donor=int(args.min_fg_per_donor),
                                    min_bg_per_donor=int(args.min_bg_per_donor),
                                    d_eff_min=int(args.d_eff_min),
                                    min_perm=int(min_perm_eff),
                                )
                                underpowered = bool(power["underpowered"])

                                bio_T = float("nan")
                                bio_p = float("nan")
                                bio_Z = float("nan")
                                if not underpowered:
                                    seed_perm = stable_seed(
                                        int(args.master_seed),
                                        "expF",
                                        "perm",
                                        int(seed_gene),
                                    )
                                    perm = perm_null_T(
                                        f=f,
                                        angles=context.theta,
                                        donor_ids=context.donor_ids,
                                        n_bins=int(args.bins),
                                        n_perm=int(n_perm_eff),
                                        seed=int(seed_perm),
                                        mode="raw",
                                        smooth_w=1,
                                        donor_stratified=True,
                                        return_null_T=True,
                                        return_obs_profile=False,
                                        bin_id=context.bin_id,
                                        bin_counts_total=context.bin_counts_total,
                                    )
                                    bio_T = float(perm["T_obs"])
                                    bio_p = float(perm["p_T"])
                                    bio_Z = float(
                                        robust_z(
                                            float(bio_T),
                                            np.asarray(perm["null_T"], dtype=float),
                                        )
                                    )

                                if str(args.qc_test) == "logistic":
                                    qc = qc_assoc_test_logistic(f=f, context=context)
                                else:
                                    qc = qc_assoc_test_spearman(f=f, context=context)

                                metric_rows.append(
                                    {
                                        "run_id": run_id,
                                        "seed": int(seed_gene),
                                        "seed_run": int(seed_run),
                                        "seed_gene": int(seed_gene),
                                        "gene_index": int(gene_counter),
                                        "geometry": str(geometry),
                                        "N": int(args.N),
                                        "D": int(D),
                                        "sigma_eta": float(sigma_eta),
                                        "regime": "R3",
                                        "pi_target": float(pi_target),
                                        "beta": float(beta),
                                        "gamma": 0.0,
                                        "g_qc": float(g_qc),
                                        "theta0": float(theta0),
                                        "prev_obs": float(np.mean(f)),
                                        "n_fg_total": int(np.sum(f)),
                                        "D_eff": int(power["D_eff"]),
                                        "underpowered": bool(underpowered),
                                        "bio_T": float(bio_T),
                                        "bio_p": float(bio_p),
                                        "bio_Z": float(bio_Z),
                                        "qc_assoc_stat": float(qc["qc_assoc_stat"]),
                                        "qc_assoc_p": float(qc["qc_assoc_p"]),
                                        "max_abs_qc_corr": float(qc["max_abs_qc_corr"]),
                                        "max_abs_qc_corr_p": float(
                                            qc["max_abs_qc_corr_p"]
                                        ),
                                        "rho_log_library": float(qc["rho_log_library"]),
                                        "rho_pct_mt": float(qc["rho_pct_mt"]),
                                        "rho_ribo": float(qc["rho_ribo"]),
                                    }
                                )
                                gene_counter += 1
                                if gene_counter % int(args.progress_every) == 0:
                                    elapsed = time.time() - start_t
                                    rate = (
                                        gene_counter / elapsed
                                        if elapsed > 0
                                        else float("nan")
                                    )
                                    print(
                                        f"  progress: genes={gene_counter} (~{rate:.2f} genes/s), est_total~{total_genes_est}",
                                        flush=True,
                                    )

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise RuntimeError("No metric rows generated.")

    metrics = _attach_q_values(metrics)
    metrics = _assign_labels(
        metrics,
        qc_q_thresh=float(args.qc_q_thresh),
        qc_corr_thresh=float(args.qc_corr_thresh),
        qc_corr_p_thresh=float(args.qc_corr_p_thresh),
    )

    summary = summarize_metrics(metrics)
    dataset_meta = pd.DataFrame(dataset_rows)

    atomic_write_csv(results_dir / "metrics_long.csv", metrics)
    atomic_write_csv(results_dir / "summary.csv", summary)
    atomic_write_csv(results_dir / "dataset_qc_geometry_correlations.csv", dataset_meta)

    _set_plot_style()
    plot_calibration_r0(
        metrics=metrics, outdir=outdir, n_perm=int(args.n_perm), bins=int(args.bins)
    )
    plot_confound_scatter(
        metrics=metrics, outdir=outdir, n_perm=int(args.n_perm), bins=int(args.bins)
    )
    plot_label_rates(
        metrics=metrics, outdir=outdir, n_perm=int(args.n_perm), bins=int(args.bins)
    )
    plot_detection_and_recall(
        metrics=metrics, outdir=outdir, n_perm=int(args.n_perm), bins=int(args.bins)
    )

    distortion_df, moran_df, distortion_warnings = run_distortion_analysis(
        metrics=metrics,
        contexts=contexts,
        outdir=outdir,
        master_seed=int(args.master_seed),
        bins=int(args.bins),
        n_perm_distortion=int(args.distortion_n_perm),
        subset_per_regime=int(args.distortion_subset_per_regime),
        moran_k_grid=[int(x) for x in args.moran_k_grid],
    )

    panel_warnings = plot_example_panels(
        metrics=metrics,
        contexts=contexts,
        outdir=outdir,
        master_seed=int(args.master_seed),
        bins=int(args.bins),
        n_perm=int(args.n_perm),
    )

    validation_warnings = run_validations(
        metrics=metrics,
        distortion_df=distortion_df,
        moran_df=moran_df,
        outdir=outdir,
    )

    # Append additional warnings from non-validation stages into debug report.
    extra_warnings = distortion_warnings + panel_warnings
    if extra_warnings:
        dbg = results_dir / "validation_debug_report.txt"
        prior = dbg.read_text(encoding="utf-8") if dbg.exists() else ""
        lines = [prior.strip(), "", "Additional warnings:"]
        lines.extend([f"- {w}" for w in extra_warnings])
        dbg.write_text("\n".join([x for x in lines if x]), encoding="utf-8")

    elapsed = time.time() - start_t
    n_non = int((~metrics["underpowered"].astype(bool)).sum())
    print(
        (
            f"Completed ExpF in {elapsed/60.0:.2f} min. rows={metrics.shape[0]}, "
            f"non_underpowered={n_non}, warnings={len(validation_warnings) + len(extra_warnings)}"
        ),
        flush=True,
    )
    print(f"Outputs written to: {outdir}", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Simulation Experiment F (confound resistance) for individual-gene BioRSP scoring "
            "with QC confound checks, distortion robustness, and reviewer-facing summaries/plots."
        )
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/simulations/expF_confound_resistance",
        help="Output directory.",
    )
    parser.add_argument("--master_seed", type=int, default=123, help="Master seed.")

    parser.add_argument("--N", type=int, default=20000, help="Cells per dataset.")
    parser.add_argument(
        "--geometries",
        type=str,
        nargs="+",
        default=DEFAULT_GEOMETRIES,
        help="Geometry names.",
    )
    parser.add_argument(
        "--D_grid",
        type=int,
        nargs="+",
        default=DEFAULT_D_GRID,
        help="Donor count grid.",
    )
    parser.add_argument(
        "--sigma_eta_grid",
        type=float,
        nargs="+",
        default=[0.4],
        help="Donor random effect sigma grid.",
    )

    parser.add_argument(
        "--pi_grid",
        type=float,
        nargs="+",
        default=DEFAULT_PI_GRID,
        help="Target prevalence grid.",
    )
    parser.add_argument(
        "--g_qc_grid",
        type=float,
        nargs="+",
        default=DEFAULT_G_QC_GRID,
        help="QC gradient strength grid.",
    )
    parser.add_argument(
        "--gamma_grid",
        type=float,
        nargs="+",
        default=DEFAULT_GAMMA_GRID,
        help="QC confound strength grid for R2.",
    )
    parser.add_argument(
        "--beta_grid",
        type=float,
        nargs="+",
        default=DEFAULT_BETA_GRID,
        help="True spatial strength grid for R3.",
    )

    parser.add_argument(
        "--genes_per_condition", type=int, default=200, help="Genes per condition."
    )
    parser.add_argument("--n_perm", type=int, default=500, help="BioRSP permutations.")
    parser.add_argument("--bins", type=int, default=36, help="BioRSP angular bins.")
    parser.add_argument(
        "--qc_test",
        type=str,
        choices=["spearman", "logistic"],
        default="spearman",
        help="QC association test mode.",
    )

    parser.add_argument(
        "--p_min", type=float, default=0.005, help="Underpowered prevalence floor."
    )
    parser.add_argument(
        "--min_fg_total",
        type=int,
        default=50,
        help="Underpowered min total foreground.",
    )
    parser.add_argument(
        "--min_fg_per_donor",
        type=int,
        default=10,
        help="Underpowered min fg per donor.",
    )
    parser.add_argument(
        "--min_bg_per_donor",
        type=int,
        default=10,
        help="Underpowered min bg per donor.",
    )
    parser.add_argument(
        "--d_eff_min", type=int, default=2, help="Underpowered min informative donors."
    )
    parser.add_argument(
        "--min_perm", type=int, default=200, help="Underpowered min permutation count."
    )

    parser.add_argument(
        "--qc_q_thresh", type=float, default=0.05, help="QC block BH-q threshold."
    )
    parser.add_argument(
        "--qc_corr_thresh",
        type=float,
        default=0.2,
        help="Absolute QC correlation threshold.",
    )
    parser.add_argument(
        "--qc_corr_p_thresh",
        type=float,
        default=0.01,
        help="P-value threshold paired with qc_corr_thresh.",
    )

    parser.add_argument(
        "--density_k",
        type=float,
        default=1.5,
        help="Acceptance slope k for density_gradient_disk geometry.",
    )

    parser.add_argument(
        "--mu0_library",
        type=float,
        default=10.0,
        help="Baseline donor library-size mean.",
    )
    parser.add_argument(
        "--sigma_library", type=float, default=0.6, help="Library-size sigma."
    )
    parser.add_argument(
        "--mt_mean", type=float, default=0.07, help="Baseline pct_mt mean."
    )
    parser.add_argument(
        "--mt_concentration",
        type=float,
        default=180.0,
        help="Beta concentration for pct_mt.",
    )

    parser.add_argument(
        "--distortion_n_perm",
        type=int,
        default=200,
        help="Permutations for distortion re-scoring subset.",
    )
    parser.add_argument(
        "--distortion_subset_per_regime",
        type=int,
        default=100,
        help="Subset size for distortion analysis per regime (R2/R3).",
    )
    parser.add_argument(
        "--moran_k_grid",
        type=int,
        nargs="+",
        default=DEFAULT_MORAN_K_GRID,
        help="kNN graph sizes for optional Moran baseline stability checks.",
    )

    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Progress print frequency (genes).",
    )
    add_common_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    warnings.simplefilter("default", RuntimeWarning)
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if bool(getattr(args, "dry_run", False)):
        print("dry_run=True: skipping execution for legacy runner.", flush=True)
        return 0
    if bool(getattr(args, "test_mode", False)):
        args = apply_testmode_overrides(args, exp_name="expF_confound_resistance")

    if int(args.N) <= 0:
        raise ValueError("N must be positive.")
    if int(args.n_perm) <= 0 or int(args.bins) <= 0:
        raise ValueError("n_perm and bins must be positive.")
    if int(args.genes_per_condition) <= 0:
        raise ValueError("genes_per_condition must be positive.")

    run_ctx = prepare_legacy_run(args, "expF_confound_resistance", __file__)
    run_experiment(args)
    finalize_legacy_run(run_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
