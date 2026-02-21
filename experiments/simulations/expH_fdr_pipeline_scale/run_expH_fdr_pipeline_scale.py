#!/usr/bin/env python3
"""Simulation Experiment H: multiple testing + triage/classification reliability at scale."""

from __future__ import annotations

import argparse
import json
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
from scipy.signal import find_peaks
from scipy.special import expit
from scipy.stats import chi2, kstest, spearmanr

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-biorsp-expH")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache-biorsp-expH")
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from biorsp.permutation import mode_max_stat_from_profiles, perm_null_T
from biorsp.power import evaluate_underpowered
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

TRUTH_CATEGORIES = [
    "NULL",
    "TRUE_UNIMODAL",
    "TRUE_BIMODAL",
    "TRUE_PATCHY",
    "DONOR_SPECIFIC",
    "QC_DRIVEN",
]

TRUE_SPATIAL = {"TRUE_UNIMODAL", "TRUE_BIMODAL", "TRUE_PATCHY"}

PRED_SHAPE_CLASSES = [
    "LOCALIZED_UNIMODAL",
    "LOCALIZED_BIMODAL",
    "LOCALIZED_MULTIMODAL",
    "UNCERTAIN_SHAPE",
]


@dataclass(frozen=True)
class DatasetContext:
    run_idx: int
    geometry: str
    N: int
    D: int
    donor_ids: np.ndarray
    donor_to_idx: dict[str, np.ndarray]
    theta: np.ndarray
    X: np.ndarray
    eta_d: np.ndarray
    eta_cell: np.ndarray
    log_library: np.ndarray
    pct_mt: np.ndarray
    ribo: np.ndarray
    z_log_library: np.ndarray
    z_pct_mt: np.ndarray
    z_ribo: np.ndarray
    bin_id: np.ndarray
    bin_counts: np.ndarray
    g_qc: float


@dataclass(frozen=True)
class GeneSimResult:
    f: np.ndarray
    pi_target: float
    beta: float
    gamma: float
    dropout_noise: float
    theta0: float
    patchy_width: float


def _fmt(x: float | int) -> str:
    return f"{float(x):.3f}".rstrip("0").rstrip(".")


def _tok(x: float | int | str) -> str:
    return str(x).replace(".", "p")


def _clip_pi(x: float) -> float:
    return float(np.clip(float(x), 0.005, 0.95))


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


def _compute_bin_cache(theta: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    wrapped = np.mod(np.asarray(theta, dtype=float).ravel(), 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, int(bins) + 1, endpoint=True)
    bin_id = np.digitize(wrapped, edges, right=False) - 1
    bin_id = np.where(bin_id == int(bins), int(bins) - 1, bin_id).astype(np.int32)
    counts = np.bincount(bin_id, minlength=int(bins)).astype(np.int64)
    return bin_id, counts


def _zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sd


def _donor_to_idx_map(donor_ids: np.ndarray) -> dict[str, np.ndarray]:
    donor_arr = np.asarray(donor_ids)
    labels = np.unique(donor_arr)
    return {str(d): np.flatnonzero(donor_arr == d).astype(int) for d in labels}


def _sample_pi_target(rng: np.random.Generator) -> float:
    u = float(rng.random())
    if u < 0.4:
        out = float(rng.uniform(0.01, 0.05))
    elif u < 0.8:
        out = float(rng.uniform(0.05, 0.2))
    else:
        out = float(rng.uniform(0.2, 0.8))
    return _clip_pi(out)


def _sample_dropout(rng: np.random.Generator) -> float:
    vals = np.asarray([0.0, 0.1, 0.2], dtype=float)
    probs = np.asarray([0.5, 0.3, 0.2], dtype=float)
    i = int(rng.choice(np.arange(vals.size), p=probs))
    return float(vals[i])


def _circular_distance_vec(theta: np.ndarray, center: float) -> np.ndarray:
    d = np.abs(np.asarray(theta, dtype=float) - float(center))
    d = np.mod(d, 2.0 * np.pi)
    return np.minimum(d, 2.0 * np.pi - d)


def _patchy_signal(theta: np.ndarray, theta0: float, width: float) -> np.ndarray:
    centers = np.asarray([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=float) + float(
        theta0
    )
    th = np.asarray(theta, dtype=float).ravel()
    mats = []
    for c in centers:
        d = _circular_distance_vec(th, float(c))
        mats.append(np.exp(-0.5 * (d / max(float(width), 1e-3)) ** 2))
    s = np.mean(np.vstack(mats), axis=0)
    s = _zscore(s)
    return s


def _simulate_dataset_context(
    *,
    master_seed: int,
    run_idx: int,
    geometry: str,
    N: int,
    D: int,
    sigma_eta: float,
    bins: int,
    density_k: float,
) -> DatasetContext:
    seed_ctx = stable_seed(
        int(master_seed), "expH", "context", int(run_idx), str(geometry), int(N), int(D)
    )
    rng = rng_from_seed(int(seed_ctx))

    kwargs: dict[str, Any] = {}
    if str(geometry) == "density_gradient_disk":
        kwargs["k"] = float(density_k)
    X, _ = sample_geometry(str(geometry), int(N), rng, **kwargs)
    theta = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2.0 * np.pi).astype(float)
    donor_ids = assign_donors(int(N), int(D), rng)
    donor_to_idx = _donor_to_idx_map(donor_ids)

    eta_d = sample_donor_effects(int(D), float(sigma_eta), rng)
    eta_cell = donor_effect_vector(donor_ids, eta_d)

    # QC covariates with donor shifts; gradient only in adversarial geometry.
    mu0 = 10.0
    sigma_l = 0.6
    mu_shift_d = rng.normal(loc=0.0, scale=0.4, size=int(D)).astype(float)
    mu_d = float(mu0) + mu_shift_d
    log_library = rng.normal(
        loc=mu_d[donor_ids], scale=float(sigma_l), size=int(N)
    ).astype(float)

    mt_shift_d = rng.uniform(low=-0.02, high=0.02, size=int(D)).astype(float)
    mt_mean_d = np.clip(0.07 + mt_shift_d, 0.005, 0.35)
    conc = 180.0
    a_d = np.clip(mt_mean_d * conc, 0.1, None)
    b_d = np.clip((1.0 - mt_mean_d) * conc, 0.1, None)
    pct_mt = rng.beta(a=a_d[donor_ids], b=b_d[donor_ids]).astype(float)

    g_qc = 1.0 if str(geometry) == "density_gradient_disk" else 0.0
    if g_qc > 0.0:
        x = np.asarray(X[:, 0], dtype=float)
        log_library = log_library + float(g_qc) * x

    ribo = 0.5 * log_library + rng.normal(loc=0.0, scale=0.3, size=int(N))

    z_log_library = _zscore(log_library)
    z_pct_mt = _zscore(pct_mt)
    z_ribo = _zscore(ribo)

    bin_id, bin_counts = _compute_bin_cache(theta, int(bins))

    return DatasetContext(
        run_idx=int(run_idx),
        geometry=str(geometry),
        N=int(N),
        D=int(D),
        donor_ids=donor_ids,
        donor_to_idx=donor_to_idx,
        theta=theta,
        X=X,
        eta_d=eta_d,
        eta_cell=eta_cell,
        log_library=log_library,
        pct_mt=pct_mt,
        ribo=ribo,
        z_log_library=z_log_library,
        z_pct_mt=z_pct_mt,
        z_ribo=z_ribo,
        bin_id=bin_id,
        bin_counts=bin_counts,
        g_qc=float(g_qc),
    )


def _simulate_gene(
    *,
    category: str,
    context: DatasetContext,
    seed_gene: int,
) -> GeneSimResult:
    rng = rng_from_seed(int(seed_gene))

    pi_target = _sample_pi_target(rng)
    dropout = _sample_dropout(rng)
    theta0 = float(rng.uniform(0.0, 2.0 * np.pi))
    patchy_width = float(rng.uniform(0.28, 0.42))

    beta = 0.0
    gamma = 0.0

    if category == "TRUE_UNIMODAL":
        beta = float(rng.uniform(0.5, 1.25))
    elif category == "TRUE_BIMODAL":
        beta = float(rng.uniform(0.5, 1.25))
    elif category == "TRUE_PATCHY":
        beta = float(rng.uniform(0.6, 1.4))
    elif category == "DONOR_SPECIFIC":
        beta = float(rng.uniform(0.6, 1.4))
    elif category == "QC_DRIVEN":
        gamma = float(rng.uniform(0.8, 1.5))

    alpha = math.log(_clip_pi(pi_target) / (1.0 - _clip_pi(pi_target)))

    if category == "NULL":
        logits = alpha + context.eta_cell
    elif category == "TRUE_UNIMODAL":
        logits = (
            alpha
            + context.eta_cell
            + float(beta) * np.cos(context.theta - float(theta0))
        )
    elif category == "TRUE_BIMODAL":
        logits = (
            alpha
            + context.eta_cell
            + float(beta) * np.cos(2.0 * (context.theta - float(theta0)))
        )
    elif category == "TRUE_PATCHY":
        sig = _patchy_signal(context.theta, float(theta0), float(patchy_width))
        logits = alpha + context.eta_cell + float(beta) * sig
    elif category == "DONOR_SPECIFIC":
        # Donor-specific directions break cross-donor replication.
        donor_theta0 = rng.uniform(0.0, 2.0 * np.pi, size=int(context.D)).astype(float)
        theta0_cell = donor_theta0[context.donor_ids]
        logits = (
            alpha + context.eta_cell + float(beta) * np.cos(context.theta - theta0_cell)
        )
    elif category == "QC_DRIVEN":
        logits = alpha + context.eta_cell + float(gamma) * context.z_log_library
    else:
        raise ValueError(f"Unsupported category '{category}'.")

    p = expit(logits)
    f = (rng.random(p.size) < p).astype(bool)

    # Dropout: flip a fraction of 1s to 0.
    if float(dropout) > 0.0:
        ones = np.flatnonzero(f)
        if ones.size > 0:
            drop_mask = rng.random(ones.size) < float(dropout)
            f[ones[drop_mask]] = False

    return GeneSimResult(
        f=f,
        pi_target=float(pi_target),
        beta=float(beta),
        gamma=float(gamma),
        dropout_noise=float(dropout),
        theta0=float(theta0),
        patchy_width=float(patchy_width),
    )


def _max_peak_prominence_positive(E: np.ndarray) -> float:
    arr = np.asarray(E, dtype=float).ravel()
    if arr.size < 3:
        return 0.0
    peaks, props = find_peaks(arr)
    if peaks.size == 0:
        return 0.0
    prom = np.asarray(props.get("prominences", np.zeros(peaks.size)), dtype=float)
    vals = arr[peaks]
    mask = vals > 0.0
    if not np.any(mask):
        return 0.0
    return float(np.max(prom[mask]))


def _count_peaks_null_calibrated(
    E_obs: np.ndarray, null_E: np.ndarray, q: float = 0.95
) -> tuple[int, float]:
    obs = np.asarray(E_obs, dtype=float).ravel()
    null = np.asarray(null_E, dtype=float)
    if null.ndim != 2 or null.shape[1] != obs.size or null.shape[0] < 1:
        return 0, float("nan")

    max_prom = np.array(
        [_max_peak_prominence_positive(row) for row in null], dtype=float
    )
    prom_thr = float(np.quantile(max_prom, float(q)))

    peaks, props = find_peaks(obs, prominence=max(float(prom_thr), 1e-12))
    if peaks.size == 0:
        return 0, float(prom_thr)
    vals = obs[peaks]
    k = int(np.sum(vals > 0.0))
    return k, float(prom_thr)


def _shape_call_from_k(k_hat: float) -> str:
    if not np.isfinite(float(k_hat)):
        return "UNCERTAIN_SHAPE"
    k = int(round(float(k_hat)))
    if k <= 0:
        return "UNCERTAIN_SHAPE"
    if k == 1:
        return "LOCALIZED_UNIMODAL"
    if k == 2:
        return "LOCALIZED_BIMODAL"
    return "LOCALIZED_MULTIMODAL"


def _truth_shape_class(category: str) -> str:
    if category == "TRUE_UNIMODAL":
        return "TRUE_UNIMODAL"
    if category == "TRUE_BIMODAL":
        return "TRUE_BIMODAL"
    if category == "TRUE_PATCHY":
        return "TRUE_PATCHY"
    return "NONSPATIAL"


def _fisher_combine_pvalues(pvals: list[float]) -> tuple[float, float]:
    vals = [float(p) for p in pvals if np.isfinite(float(p)) and float(p) > 0.0]
    if not vals:
        return 0.0, 1.0
    stat = float(
        -2.0 * np.sum(np.log(np.clip(np.asarray(vals, dtype=float), 1e-300, 1.0)))
    )
    p = float(chi2.sf(stat, 2 * len(vals)))
    return stat, p


def _within_donor_spearman(
    f: np.ndarray, cov: np.ndarray, donor_to_idx: dict[str, np.ndarray]
) -> tuple[float, float]:
    f_arr = np.asarray(f, dtype=float).ravel()
    c_arr = np.asarray(cov, dtype=float).ravel()
    rhos: list[float] = []
    pvals: list[float] = []
    ws: list[float] = []

    for idx in donor_to_idx.values():
        ii = np.asarray(idx, dtype=int)
        if ii.size < 6:
            continue
        fs = f_arr[ii]
        cs = c_arr[ii]
        if np.allclose(fs, fs[0]) or np.allclose(cs, cs[0]):
            continue
        rho, p = spearmanr(fs, cs, nan_policy="omit")
        if rho is None or p is None:
            continue
        rf = float(rho)
        pf = float(p)
        if not (np.isfinite(rf) and np.isfinite(pf)):
            continue
        rhos.append(rf)
        pvals.append(max(pf, 1e-300))
        ws.append(float(max(ii.size - 3, 1)))

    if not rhos:
        return 0.0, 1.0

    w = np.asarray(ws, dtype=float)
    r = np.asarray(rhos, dtype=float)
    pooled = float(np.sum(w * r) / np.sum(w))
    _, p_meta = _fisher_combine_pvalues(pvals)
    return pooled, float(p_meta)


def _qc_assoc_metrics(f: np.ndarray, context: DatasetContext) -> dict[str, float]:
    rho_l, p_l = _within_donor_spearman(f, context.log_library, context.donor_to_idx)
    rho_m, p_m = _within_donor_spearman(f, context.pct_mt, context.donor_to_idx)
    rho_r, p_r = _within_donor_spearman(f, context.ribo, context.donor_to_idx)

    stat, p_block = _fisher_combine_pvalues([p_l, p_m, p_r])
    rho_vec = np.asarray([rho_l, rho_m, rho_r], dtype=float)
    p_vec = np.asarray([p_l, p_m, p_r], dtype=float)
    j = int(np.nanargmax(np.abs(rho_vec))) if rho_vec.size else 0

    return {
        "qc_assoc_stat": float(stat),
        "qc_assoc_p": float(p_block),
        "max_abs_qc_corr": float(np.max(np.abs(rho_vec))) if rho_vec.size else 0.0,
        "max_abs_qc_corr_p": float(p_vec[j]) if p_vec.size else 1.0,
    }


def _score_gene(
    *,
    f: np.ndarray,
    context: DatasetContext,
    bins: int,
    mode: str,
    smooth_w: int,
    n_perm: int,
    seed_perm: int,
    power_cfg: dict[str, Any],
    min_perm_eff: int,
) -> dict[str, Any]:
    f_arr = np.asarray(f, dtype=bool).ravel()
    prev_obs = float(np.mean(f_arr))
    n_fg_total = int(np.sum(f_arr))

    power = evaluate_underpowered(
        donor_ids=context.donor_ids,
        f=f_arr,
        n_perm=int(n_perm),
        p_min=float(power_cfg["p_min"]),
        min_fg_total=int(power_cfg["min_fg_total"]),
        min_fg_per_donor=int(power_cfg["min_fg_per_donor"]),
        min_bg_per_donor=int(power_cfg["min_bg_per_donor"]),
        d_eff_min=int(power_cfg["d_eff_min"]),
        min_perm=int(min_perm_eff),
    )
    underpowered = bool(power["underpowered"])

    base = {
        "prev_obs": float(prev_obs),
        "n_fg_total": int(n_fg_total),
        "D_eff": int(power["D_eff"]),
        "underpowered": bool(underpowered),
        "bio_T_raw": float("nan"),
        "bio_p_raw": float("nan"),
        "bio_Z_raw": float("nan"),
        "bio_T": float("nan"),
        "bio_p": float("nan"),
        "bio_Z": float("nan"),
        "K_hat": float("nan"),
        "C_hat": float("nan"),
        "prominence_thr": float("nan"),
        "S1": float("nan"),
        "S2": float("nan"),
        "S3": float("nan"),
        "S4": float("nan"),
        "shape_call_prelim": "UNCERTAIN_SHAPE",
        "E_obs": np.full(int(bins), np.nan, dtype=float),
    }

    if underpowered or n_fg_total == 0 or n_fg_total == int(f_arr.size):
        return base

    perm_raw = perm_null_T(
        f=f_arr,
        angles=context.theta,
        donor_ids=context.donor_ids,
        n_bins=int(bins),
        n_perm=int(n_perm),
        seed=int(seed_perm),
        mode="raw",
        smooth_w=1,
        donor_stratified=True,
        return_null_T=True,
        return_obs_profile=True,
        return_null_profiles=True,
        bin_id=context.bin_id,
        bin_counts_total=context.bin_counts,
    )

    E_raw = np.asarray(perm_raw["E_phi_obs"], dtype=float)
    null_E_raw = np.asarray(perm_raw["null_E_phi"], dtype=float)
    T_raw = float(perm_raw["T_obs"])
    p_raw = float(perm_raw["p_T"])
    null_T_raw = np.asarray(perm_raw["null_T"], dtype=float)
    z_raw = float(robust_z(T_raw, null_T_raw))

    if str(mode) == "raw":
        E_use = E_raw
        null_E_use = null_E_raw
        T_use = T_raw
        p_use = p_raw
        null_T_use = null_T_raw
    else:
        mode_res = mode_max_stat_from_profiles(
            E_raw, null_E_raw, mode=str(mode), smooth_w=int(smooth_w)
        )
        E_use = np.asarray(mode_res["E_phi_obs"], dtype=float)
        null_E_use = np.asarray(mode_res["null_E_phi"], dtype=float)
        T_use = float(mode_res["T_obs"])
        p_use = float(mode_res["p_T"])
        null_T_use = np.asarray(mode_res["null_T"], dtype=float)

    z_use = float(robust_z(T_use, null_T_use))

    tau = np.quantile(null_E_use, 0.95, axis=0)
    C_hat = float(np.mean(E_use > tau))
    K_hat, prom_thr = _count_peaks_null_calibrated(E_use, null_E_use, q=0.95)

    # FFT features (harmonic ratios)
    centered = np.asarray(E_use, dtype=float) - float(np.mean(E_use))
    fft = np.abs(np.fft.rfft(centered))
    denom = float(np.sum(fft[1:])) if fft.size > 1 else 0.0

    def _s(k: int) -> float:
        if denom <= 1e-12:
            return float("nan")
        if k < fft.size:
            return float(fft[k] / denom)
        return float(0.0)

    shape_pre = _shape_call_from_k(float(K_hat))

    return {
        **base,
        "bio_T_raw": float(T_raw),
        "bio_p_raw": float(p_raw),
        "bio_Z_raw": float(z_raw),
        "bio_T": float(T_use),
        "bio_p": float(p_use),
        "bio_Z": float(z_use),
        "K_hat": float(K_hat),
        "C_hat": float(C_hat),
        "prominence_thr": float(prom_thr),
        "S1": _s(1),
        "S2": _s(2),
        "S3": _s(3),
        "S4": _s(4),
        "shape_call_prelim": str(shape_pre),
        "E_obs": E_use,
    }


def _apply_bh_and_calls(
    gene_table: pd.DataFrame,
    *,
    enable_qc_flag: bool,
    q_thresh: float,
) -> pd.DataFrame:
    out = gene_table.copy()
    out["bio_q"] = np.nan
    out["qc_q"] = np.nan

    for _, idx in out.groupby(["run_id", "geometry"], sort=False).groups.items():
        ii = np.asarray(list(idx), dtype=int)
        tested = (~out.loc[ii, "underpowered"].astype(bool)).to_numpy(dtype=bool)
        if np.any(tested):
            pvals = pd.to_numeric(
                out.loc[ii[tested], "bio_p"], errors="coerce"
            ).to_numpy(dtype=float)
            q = bh_fdr(np.where(np.isfinite(pvals), pvals, 1.0))
            out.loc[ii[tested], "bio_q"] = q

            if bool(enable_qc_flag):
                qp = pd.to_numeric(
                    out.loc[ii[tested], "qc_assoc_p"], errors="coerce"
                ).to_numpy(dtype=float)
                q_qc = bh_fdr(np.where(np.isfinite(qp), qp, 1.0))
                out.loc[ii[tested], "qc_q"] = q_qc

    out["bio_q"] = pd.to_numeric(out["bio_q"], errors="coerce")
    out["called_localized"] = (
        (~out["underpowered"].astype(bool))
        & np.isfinite(pd.to_numeric(out["bio_q"], errors="coerce"))
        & (pd.to_numeric(out["bio_q"], errors="coerce") <= float(q_thresh))
    )

    # Called class only among discoveries.
    called_class = np.full(out.shape[0], "NOT_CALLED", dtype=object)
    mask_called = out["called_localized"].to_numpy(dtype=bool)
    called_class[mask_called] = (
        out.loc[mask_called, "shape_call_prelim"].astype(str).to_numpy()
    )
    out["called_class"] = called_class

    if bool(enable_qc_flag):
        q_sig = pd.to_numeric(out["qc_q"], errors="coerce") <= 0.05
        corr_sig = (pd.to_numeric(out["max_abs_qc_corr"], errors="coerce") >= 0.2) & (
            pd.to_numeric(out["max_abs_qc_corr_p"], errors="coerce") <= 0.01
        )
        out["qc_flag"] = (q_sig | corr_sig).astype(bool)
    else:
        out["qc_flag"] = False

    final_call = np.full(out.shape[0], "not_called", dtype=object)
    call_mask = out["called_localized"].to_numpy(dtype=bool)
    if bool(enable_qc_flag):
        qmask = call_mask & out["qc_flag"].to_numpy(dtype=bool)
        pmask = call_mask & (~out["qc_flag"].to_numpy(dtype=bool))
        final_call[qmask] = "qc_quarantined"
        final_call[pmask] = "localized_pass"
    else:
        final_call[call_mask] = "localized_pass"

    out["final_call"] = final_call
    out["called_localized_pass"] = out["final_call"] == "localized_pass"
    return out


def _prop_ci(k: int, n: int) -> tuple[float, float, float]:
    if int(n) <= 0:
        return float("nan"), float("nan"), float("nan")
    p = float(k) / float(n)
    lo, hi = wilson_ci(int(k), int(n))
    return p, float(lo), float(hi)


def summarize_runlevel(gene_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for keys, grp in gene_table.groupby(["run_id", "seed_run", "geometry"], sort=True):
        n_total = int(grp.shape[0])
        under = grp["underpowered"].astype(bool)
        n_under = int(np.sum(under))
        tested = grp.loc[~under].copy()
        n_tested = int(tested.shape[0])

        called = tested.loc[tested["called_localized"].astype(bool)].copy()
        pass_called = tested.loc[tested["final_call"] == "localized_pass"].copy()
        quarantined = tested.loc[tested["final_call"] == "qc_quarantined"].copy()

        n_called = int(called.shape[0])
        n_pass = int(pass_called.shape[0])
        n_quarantine = int(quarantined.shape[0])

        truth_spatial_tested = tested["truth_category"].isin(TRUE_SPATIAL)
        n_true_spatial_tested = int(np.sum(truth_spatial_tested))

        tp = int(np.sum(pass_called["truth_category"].isin(TRUE_SPATIAL)))
        fp = int(np.sum(~pass_called["truth_category"].isin(TRUE_SPATIAL)))

        fdr = float(fp / max(1, n_pass))
        power = float(tp / max(1, n_true_spatial_tested))
        fdr_ci = _prop_ci(fp, max(1, n_pass))
        power_ci = _prop_ci(tp, max(1, n_true_spatial_tested))

        ds_tested = tested.loc[tested["truth_category"] == "DONOR_SPECIFIC"]
        ds_leak = (
            float(np.mean(ds_tested["final_call"] == "localized_pass"))
            if not ds_tested.empty
            else float("nan")
        )

        # Shape confusion among TP discoveries.
        tp_rows = pass_called.loc[
            pass_called["truth_category"].isin(TRUE_SPATIAL)
        ].copy()
        shape_truth = ["TRUE_UNIMODAL", "TRUE_BIMODAL", "TRUE_PATCHY"]
        shape_pred = PRED_SHAPE_CLASSES
        cm: dict[str, int] = {}
        for t in shape_truth:
            for p in shape_pred:
                cm[f"cm_{t}_pred_{p}"] = int(
                    np.sum(
                        (tp_rows["truth_category"] == t)
                        & (tp_rows["called_class"] == p)
                    )
                )

        rows.append(
            {
                "run_id": str(keys[0]),
                "seed_run": int(keys[1]),
                "geometry": str(keys[2]),
                "n_genes_total": int(n_total),
                "n_underpowered": int(n_under),
                "n_tested": int(n_tested),
                "n_called_localized": int(n_called),
                "n_called_localized_pass": int(n_pass),
                "n_qc_quarantined": int(n_quarantine),
                "empirical_FDR": float(fdr),
                "empirical_FDR_ci_low": float(fdr_ci[1]),
                "empirical_FDR_ci_high": float(fdr_ci[2]),
                "empirical_power": float(power),
                "empirical_power_ci_low": float(power_ci[1]),
                "empirical_power_ci_high": float(power_ci[2]),
                "donor_specific_leakage_rate": float(ds_leak),
                **cm,
            }
        )

    return pd.DataFrame(rows)


def summarize_by_truth(gene_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, grp in gene_table.groupby(["truth_category", "geometry"], sort=True):
        n = int(grp.shape[0])
        tested = (~grp["underpowered"].astype(bool)).to_numpy(dtype=bool)
        called = grp["called_localized"].astype(bool).to_numpy(dtype=bool)
        passed = (grp["final_call"].astype(str) == "localized_pass").to_numpy(
            dtype=bool
        )

        called_q = pd.to_numeric(grp.loc[called, "bio_q"], errors="coerce").to_numpy(
            dtype=float
        )
        called_q = called_q[np.isfinite(called_q)]

        rows.append(
            {
                "truth_category": str(keys[0]),
                "geometry": str(keys[1]),
                "n_genes": int(n),
                "tested_rate": float(np.mean(tested)) if n > 0 else float("nan"),
                "call_rate": float(np.mean(called)) if n > 0 else float("nan"),
                "pass_rate": float(np.mean(passed)) if n > 0 else float("nan"),
                "mean_q_among_called": (
                    float(np.mean(called_q)) if called_q.size else float("nan")
                ),
                "qc_quarantine_rate_among_called": (
                    float(np.mean(grp.loc[called, "final_call"] == "qc_quarantined"))
                    if int(np.sum(called)) > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_fdr_power_by_run(
    summary_runlevel: pd.DataFrame, outdir: Path, meta: str
) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    geoms = sorted(summary_runlevel["geometry"].astype(str).unique().tolist())
    fig, axes = plt.subplots(
        1, len(geoms), figsize=(5.5 * len(geoms), 4.0), constrained_layout=False
    )
    axes = np.atleast_1d(axes)

    for ax, geom in zip(axes, geoms):
        sub = summary_runlevel.loc[summary_runlevel["geometry"] == geom].sort_values(
            "seed_run"
        )
        x = np.arange(sub.shape[0], dtype=float)
        w = 0.35
        ax.bar(x - w / 2, sub["empirical_FDR"], width=w, label="FDR")
        ax.bar(x + w / 2, sub["empirical_power"], width=w, label="Power")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["seed_run"].astype(int).astype(str).tolist())
        ax.set_xlabel("seed_run")
        ax.set_ylabel("Rate")
        ax.set_title(f"geometry={geom}")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"FDR and power by run\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    savefig(fig, plots_dir / "fdr_power_by_run.png")
    plt.close(fig)


def plot_call_composition(gene_table: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    geoms = sorted(gene_table["geometry"].astype(str).unique().tolist())
    categories = TRUTH_CATEGORIES

    rows = []
    for geom in geoms:
        sub = gene_table.loc[gene_table["geometry"] == geom].copy()
        called = sub.loc[sub["called_localized"].astype(bool)]
        passed = sub.loc[sub["final_call"] == "localized_pass"]

        for stage, df in [("called_localized", called), ("localized_pass", passed)]:
            n = int(df.shape[0])
            for cat in categories:
                frac = float(np.mean(df["truth_category"] == cat)) if n > 0 else 0.0
                rows.append(
                    {
                        "geometry": geom,
                        "stage": stage,
                        "truth_category": cat,
                        "frac": frac,
                    }
                )

    comp = pd.DataFrame(rows)
    fig, axes = plt.subplots(
        1, len(geoms), figsize=(5.6 * len(geoms), 4.1), constrained_layout=False
    )
    axes = np.atleast_1d(axes)

    palette = {
        "NULL": "#9ecae1",
        "TRUE_UNIMODAL": "#1f77b4",
        "TRUE_BIMODAL": "#ff7f0e",
        "TRUE_PATCHY": "#2ca02c",
        "DONOR_SPECIFIC": "#d62728",
        "QC_DRIVEN": "#9467bd",
    }

    for ax, geom in zip(axes, geoms):
        sub = comp.loc[comp["geometry"] == geom].copy()
        stages = ["called_localized", "localized_pass"]
        x = np.arange(len(stages), dtype=float)
        bottom = np.zeros(len(stages), dtype=float)
        for cat in categories:
            vals = []
            for st in stages:
                v = sub.loc[
                    (sub["stage"] == st) & (sub["truth_category"] == cat), "frac"
                ]
                vals.append(float(v.iloc[0]) if not v.empty else 0.0)
            ax.bar(x, vals, bottom=bottom, color=palette[cat], width=0.6, label=cat)
            bottom += np.asarray(vals, dtype=float)
        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=20)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"geometry={geom}")
        ax.set_ylabel("Fraction")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        by_label = {}
        for handle, label in zip(handles, labels):
            by_label[label] = handle
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            ncol=3,
            frameon=False,
        )
    fig.suptitle(f"Call composition by truth category\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    savefig(fig, plots_dir / "call_composition.png")
    plt.close(fig)


def plot_null_calibration(gene_table: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    null_df = gene_table.loc[
        (gene_table["truth_category"] == "NULL")
        & (~gene_table["underpowered"].astype(bool))
    ].copy()
    geoms = sorted(null_df["geometry"].astype(str).unique().tolist())
    if not geoms:
        return

    fig_h, axes_h = plt.subplots(
        1, len(geoms), figsize=(5.0 * len(geoms), 3.8), constrained_layout=False
    )
    axes_h = np.atleast_1d(axes_h)
    for ax, geom in zip(axes_h, geoms):
        p = pd.to_numeric(
            null_df.loc[null_df["geometry"] == geom, "bio_p"], errors="coerce"
        ).to_numpy(dtype=float)
        p_hist(ax, p, bins=20, density=True)
        ax.set_title(f"NULL p-hist | {geom}")
        ax.set_xlabel("bio_p")
        ax.set_ylabel("density")
    fig_h.suptitle(f"NULL p-value calibration histogram\n{meta}")
    fig_h.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    savefig(fig_h, plots_dir / "null_calibration_p_hist.png")
    plt.close(fig_h)

    fig_q, axes_q = plt.subplots(
        1, len(geoms), figsize=(5.0 * len(geoms), 3.8), constrained_layout=False
    )
    axes_q = np.atleast_1d(axes_q)
    for ax, geom in zip(axes_q, geoms):
        p = pd.to_numeric(
            null_df.loc[null_df["geometry"] == geom, "bio_p"], errors="coerce"
        ).to_numpy(dtype=float)
        qq_plot(ax, p)
        ax.set_title(f"NULL QQ | {geom}")
        ax.set_xlabel("Expected")
        ax.set_ylabel("Observed")
    fig_q.suptitle(f"NULL p-value QQ\n{meta}")
    fig_q.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    savefig(fig_q, plots_dir / "null_calibration_qq.png")
    plt.close(fig_q)

    fig_v, axes_v = plt.subplots(
        1, len(geoms), figsize=(5.0 * len(geoms), 3.8), constrained_layout=False
    )
    axes_v = np.atleast_1d(axes_v)
    for ax, geom in zip(axes_v, geoms):
        qvals = pd.to_numeric(
            null_df.loc[null_df["geometry"] == geom, "bio_q"], errors="coerce"
        ).to_numpy(dtype=float)
        qvals = qvals[np.isfinite(qvals)]
        if qvals.size:
            ax.hist(
                qvals,
                bins=np.linspace(0.0, 1.0, 21),
                color="#4c78a8",
                alpha=0.85,
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"NULL q-hist | {geom}")
        ax.set_xlabel("bio_q")
        ax.set_ylabel("count")
    fig_v.suptitle(f"NULL q-value histogram\n{meta}")
    fig_v.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    savefig(fig_v, plots_dir / "null_qvalue_hist.png")
    plt.close(fig_v)


def plot_fdr_power_curves(
    gene_table: pd.DataFrame, outdir: Path, meta: str, enable_qc_flag: bool
) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    df = gene_table.loc[~gene_table["underpowered"].astype(bool)].copy()
    if df.empty:
        return

    thresholds = np.linspace(0.01, 0.2, 40)
    geoms = sorted(df["geometry"].astype(str).unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=False)
    ax_fdr, ax_pow = axes

    for geom in geoms:
        sub = df.loc[df["geometry"] == geom].copy()
        fdr_vals: list[float] = []
        pow_vals: list[float] = []

        truth_spatial = sub["truth_category"].isin(TRUE_SPATIAL).to_numpy(dtype=bool)
        for t in thresholds:
            called = pd.to_numeric(sub["bio_q"], errors="coerce").to_numpy(
                dtype=float
            ) <= float(t)
            if bool(enable_qc_flag):
                called = called & (~sub["qc_flag"].to_numpy(dtype=bool))

            tp = int(np.sum(called & truth_spatial))
            fp = int(np.sum(called & (~truth_spatial)))
            fdr = float(fp / max(1, tp + fp))
            power = float(tp / max(1, int(np.sum(truth_spatial))))
            fdr_vals.append(fdr)
            pow_vals.append(power)

        ax_fdr.plot(thresholds, fdr_vals, linewidth=1.5, label=geom)
        ax_pow.plot(thresholds, pow_vals, linewidth=1.5, label=geom)

    ax_fdr.set_xlabel("q-threshold")
    ax_fdr.set_ylabel("Empirical FDR")
    ax_fdr.set_ylim(0.0, 1.0)
    ax_fdr.set_title("FDR vs q-threshold")

    ax_pow.set_xlabel("q-threshold")
    ax_pow.set_ylabel("Empirical power")
    ax_pow.set_ylim(0.0, 1.0)
    ax_pow.set_title("Power vs q-threshold")

    ax_fdr.legend(frameon=False)
    ax_pow.legend(frameon=False)

    fig.suptitle(f"FDR/power operating curves\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    savefig(fig, plots_dir / "fdr_power_curves.png")
    plt.close(fig)


def plot_shape_confusion(gene_table: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots_dir = ensure_dir(outdir / "plots")

    truth_labels = ["TRUE_UNIMODAL", "TRUE_BIMODAL", "TRUE_PATCHY"]
    pred_labels = PRED_SHAPE_CLASSES

    sub = gene_table.loc[
        gene_table["truth_category"].isin(TRUE_SPATIAL)
        & (gene_table["final_call"] == "localized_pass")
    ].copy()

    mat = np.zeros((len(truth_labels), len(pred_labels)), dtype=float)
    for i, t in enumerate(truth_labels):
        for j, p in enumerate(pred_labels):
            mat[i, j] = float(
                np.sum((sub["truth_category"] == t) & (sub["called_class"] == p))
            )

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=False)
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_xticklabels(pred_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(truth_labels)))
    ax.set_yticklabels(truth_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center", fontsize=8)
    ax.set_title("Shape confusion among true-spatial localized_pass")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Shape confusion matrix\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    savefig(fig, plots_dir / "shape_confusion_matrix.png")
    plt.close(fig)

    # Precision/recall per predicted shape class.
    precision: list[float] = []
    recall: list[float] = []
    for p in ["LOCALIZED_UNIMODAL", "LOCALIZED_BIMODAL", "LOCALIZED_MULTIMODAL"]:
        pred_mask = sub["called_class"] == p
        if p == "LOCALIZED_UNIMODAL":
            truth_target = sub["truth_category"] == "TRUE_UNIMODAL"
        elif p == "LOCALIZED_BIMODAL":
            truth_target = sub["truth_category"] == "TRUE_BIMODAL"
        else:
            truth_target = sub["truth_category"] == "TRUE_PATCHY"

        tp = int(np.sum(pred_mask & truth_target))
        fp = int(np.sum(pred_mask & (~truth_target)))
        fn = int(np.sum((~pred_mask) & truth_target))

        precision.append(float(tp / max(1, tp + fp)))
        recall.append(float(tp / max(1, tp + fn)))

    cls = ["UNIMODAL", "BIMODAL", "MULTIMODAL"]
    x = np.arange(len(cls), dtype=float)
    w = 0.35

    fig2, ax2 = plt.subplots(figsize=(6.0, 4.0), constrained_layout=False)
    ax2.bar(x - w / 2, precision, width=w, label="Precision")
    ax2.bar(x + w / 2, recall, width=w, label="Recall")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cls)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Score")
    ax2.set_title("Shape precision/recall")
    ax2.legend(frameon=False)
    fig2.suptitle(f"Shape precision and recall\n{meta}")
    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    savefig(fig2, plots_dir / "shape_precision_recall.png")
    plt.close(fig2)


def plot_score_scatter(gene_table: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    df = gene_table.loc[~gene_table["underpowered"].astype(bool)].copy()
    if df.empty:
        return

    x = pd.to_numeric(df["bio_Z"], errors="coerce").to_numpy(dtype=float)
    y = -np.log10(
        np.clip(
            pd.to_numeric(df["bio_q"], errors="coerce").to_numpy(dtype=float),
            1e-12,
            1.0,
        )
    )
    truth = df["truth_category"].astype(str).to_numpy()

    colors = {
        "NULL": "#9ecae1",
        "TRUE_UNIMODAL": "#1f77b4",
        "TRUE_BIMODAL": "#ff7f0e",
        "TRUE_PATCHY": "#2ca02c",
        "DONOR_SPECIFIC": "#d62728",
        "QC_DRIVEN": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=False)
    for cat in TRUTH_CATEGORIES:
        mask = truth == cat
        if not np.any(mask):
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=10,
            alpha=0.35,
            label=cat,
            color=colors[cat],
            linewidths=0.0,
        )
    ax.set_xlabel("bio_Z")
    ax.set_ylabel("-log10(bio_q)")
    ax.set_title("Score scatter by truth category")
    ax.legend(frameon=False, ncol=2)
    fig.suptitle(f"Score diagnostics\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    savefig(fig, plots_dir / "score_scatter_by_truth.png")
    plt.close(fig)


def plot_underpowered_impact(gene_table: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    df = gene_table.copy()

    bins = [0.0, 0.05, 0.2, 1.0]
    labels = ["[0,0.05)", "[0.05,0.2)", "[0.2,1]"]
    df["pi_bin"] = pd.cut(
        pd.to_numeric(df["pi_target"], errors="coerce"),
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    grp = (
        df.groupby(["truth_category", "pi_bin"], sort=True)["underpowered"]
        .mean()
        .reset_index(name="under_rate")
    )

    truth_order = TRUTH_CATEGORIES
    fig, axes = plt.subplots(
        1, len(labels), figsize=(4.5 * len(labels), 4.0), constrained_layout=False
    )
    axes = np.atleast_1d(axes)

    for ax, pi_label in zip(axes, labels):
        sub = grp.loc[grp["pi_bin"].astype(str) == pi_label].copy()
        vals = []
        for t in truth_order:
            row = sub.loc[sub["truth_category"] == t, "under_rate"]
            vals.append(float(row.iloc[0]) if not row.empty else np.nan)
        x = np.arange(len(truth_order), dtype=float)
        ax.bar(x, vals, color="#6baed6")
        ax.set_xticks(x)
        ax.set_xticklabels(truth_order, rotation=35, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"pi bin {pi_label}")
        ax.set_ylabel("underpowered rate")

    fig.suptitle(f"Underpowered impact by truth and prevalence\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    savefig(fig, plots_dir / "underpowered_by_truth_and_pi.png")
    plt.close(fig)


def plot_example_panels(
    gene_table: pd.DataFrame,
    contexts: dict[tuple[int, str], DatasetContext],
    outdir: Path,
    bins: int,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")

    # Candidate sets.
    good_true = gene_table.loc[
        gene_table["truth_category"].isin(TRUE_SPATIAL)
        & (gene_table["final_call"] == "localized_pass")
    ].copy()
    null_fp = gene_table.loc[
        (gene_table["truth_category"] == "NULL")
        & (gene_table["final_call"] == "localized_pass")
    ].copy()
    qc_quar = gene_table.loc[
        (gene_table["truth_category"] == "QC_DRIVEN")
        & (gene_table["final_call"] == "qc_quarantined")
    ].copy()

    selected_rows: list[pd.Series] = []

    def _pick(df: pd.DataFrame, n: int) -> list[pd.Series]:
        if df.empty:
            return []
        dd = df.sort_values("bio_q")
        return [dd.iloc[i] for i in range(min(int(n), dd.shape[0]))]

    selected_rows.extend(_pick(good_true, 2))
    selected_rows.extend(_pick(null_fp, 2))
    selected_rows.extend(_pick(qc_quar, 2))

    if not selected_rows:
        # Fallback: choose lowest-q genes across all truth categories.
        fallback = gene_table.copy()
        fallback["q_rank"] = pd.to_numeric(fallback["bio_q"], errors="coerce").fillna(
            1.0
        )
        fallback = fallback.sort_values("q_rank").head(6)
        selected_rows.extend([fallback.iloc[i] for i in range(fallback.shape[0])])

    if not selected_rows:
        fig, ax = plt.subplots(figsize=(7.2, 2.2), constrained_layout=False)
        ax.axis("off")
        ax.text(
            0.5, 0.5, "No genes available for example panels.", ha="center", va="center"
        )
        fig.tight_layout()
        savefig(fig, plots_dir / "example_panels.png")
        plt.close(fig)
        return

    centers = np.linspace(0.0, 2.0 * np.pi, int(bins), endpoint=False)
    fig, axes = plt.subplots(
        len(selected_rows),
        2,
        figsize=(10.5, 3.0 * len(selected_rows)),
        constrained_layout=False,
    )
    if len(selected_rows) == 1:
        axes = np.asarray([axes])

    for i, row in enumerate(selected_rows):
        run_idx = int(row["seed_run"])
        geom = str(row["geometry"])
        context = contexts[(run_idx, geom)]

        sim = _simulate_gene(
            category=str(row["truth_category"]),
            context=context,
            seed_gene=int(row["seed"]),
        )
        f = sim.f

        ax0 = axes[i, 0]
        plot_embedding_with_foreground(
            context.X,
            f,
            ax=ax0,
            title=(
                f"{row['truth_category']} | {row['final_call']}\n"
                f"q={_fmt(float(row['bio_q']))}, class={row['called_class']}"
            ),
            s=4.0,
            alpha_bg=0.30,
            alpha_fg=0.6,
        )
        ax0.set_title(
            f"{row['truth_category']} | {row['final_call']}\nq={_fmt(float(row['bio_q']))}, class={row['called_class']}"
        )

        gs = axes[i, 1].get_gridspec()
        axes[i, 1].remove()
        ax1 = fig.add_subplot(gs[i, 1], projection="polar")
        E_obs = np.asarray(json.loads(row["E_obs_json"]), dtype=float)
        plot_rsp_polar(
            centers,
            E_obs,
            ax=ax1,
            color="#111111",
            linewidth=1.1,
            title=f"K_hat={_fmt(float(row['K_hat']))}, C_hat={_fmt(float(row['C_hat']))}",
        )

    fig.suptitle("Representative examples (true calls, null FPs, QC quarantined)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    savefig(fig, plots_dir / "example_panels.png")
    plt.close(fig)


def run_validations(
    gene_table: pd.DataFrame,
    summary_runlevel: pd.DataFrame,
    outdir: Path,
    enable_qc_flag: bool,
) -> list[str]:
    results_dir = ensure_dir(outdir / "results")
    warnings_out: list[str] = []
    lines: list[str] = []

    # 1) NULL p-value KS in disk_gaussian.
    disk_null = gene_table.loc[
        (gene_table["geometry"] == "disk_gaussian")
        & (gene_table["truth_category"] == "NULL")
        & (~gene_table["underpowered"].astype(bool))
    ].copy()
    pvals = pd.to_numeric(disk_null["bio_p"], errors="coerce").to_numpy(dtype=float)
    pvals = pvals[np.isfinite(pvals)]
    if pvals.size >= 100:
        ks = kstest(pvals, "uniform")
        ks_p = float(ks.pvalue)
        lines.append(f"Disk NULL KS p-value: {ks_p:.4g} (n={pvals.size})")
        if ks_p <= 1e-3:
            warnings_out.append(f"Disk NULL KS p-value is low ({ks_p:.4g} <= 1e-3).")
    else:
        warnings_out.append("Insufficient disk NULL tested genes for KS validation.")

    # 2) Empirical FDR target on disk.
    disk_sum = summary_runlevel.loc[
        summary_runlevel["geometry"] == "disk_gaussian"
    ].copy()
    if not disk_sum.empty:
        mean_fdr = float(
            np.mean(pd.to_numeric(disk_sum["empirical_FDR"], errors="coerce"))
        )
        lines.append(f"Mean empirical FDR on disk: {mean_fdr:.3f}")
        if mean_fdr > 0.08:
            warnings_out.append(
                f"Empirical FDR on disk is high ({mean_fdr:.3f} > 0.08)."
            )

    # 3) Power should increase with beta within true spatial.
    ts = gene_table.loc[
        gene_table["truth_category"].isin(TRUE_SPATIAL)
        & (~gene_table["underpowered"].astype(bool))
    ].copy()
    if not ts.empty:
        agg = (
            ts.groupby("beta", sort=True)["called_localized_pass"].mean().reset_index()
        )
        if agg.shape[0] >= 3:
            rho, _ = spearmanr(
                agg["beta"], agg["called_localized_pass"], nan_policy="omit"
            )
            r = float(rho) if rho is not None else float("nan")
            lines.append(f"Power-vs-beta Spearman (true spatial): {r:.3f}")
            if not np.isfinite(r) or r < 0.3:
                warnings_out.append(
                    f"Power does not clearly increase with beta (rho={r:.3f})."
                )

    # 4) QC quarantine in adversary geometry.
    if bool(enable_qc_flag):
        adv_qc = gene_table.loc[
            (gene_table["geometry"] == "density_gradient_disk")
            & (gene_table["truth_category"] == "QC_DRIVEN")
            & (gene_table["called_localized"].astype(bool))
        ].copy()
        if not adv_qc.empty:
            quar_rate = float(np.mean(adv_qc["final_call"] == "qc_quarantined"))
            lines.append(
                f"QC quarantine rate for adversary QC_DRIVEN among called: {quar_rate:.3f}"
            )
            if quar_rate < 0.6:
                warnings_out.append(
                    f"QC quarantine rate is low in adversary geometry ({quar_rate:.3f} < 0.6)."
                )
        else:
            lines.append(
                "No adversary QC_DRIVEN called genes; quarantine rate not computed."
            )

    # 5) Donor-specific leakage reporting.
    ds = gene_table.loc[
        (gene_table["truth_category"] == "DONOR_SPECIFIC")
        & (~gene_table["underpowered"].astype(bool))
    ].copy()
    if not ds.empty:
        leakage = float(np.mean(ds["final_call"] == "localized_pass"))
        lines.append(f"DONOR_SPECIFIC leakage rate (localized_pass): {leakage:.3f}")
        if leakage > 0.15:
            warnings_out.append(
                f"DONOR_SPECIFIC leakage is non-trivial ({leakage:.3f}); consider applying Exp G replication filter."
            )

    # Top false positives for debugging.
    fps = gene_table.loc[
        (~gene_table["truth_category"].isin(TRUE_SPATIAL))
        & (gene_table["final_call"] == "localized_pass")
    ].copy()
    fps = fps.sort_values("bio_q").head(20)

    if not fps.empty:
        lines.append("")
        lines.append("Top false positives (non-spatial localized_pass):")
        cols = [
            "run_id",
            "geometry",
            "truth_category",
            "pi_target",
            "beta",
            "bio_p",
            "bio_q",
            "called_class",
        ]
        for _, row in fps.iterrows():
            lines.append(" | ".join([f"{c}={row[c]}" for c in cols]))

    if warnings_out:
        lines.append("")
        lines.append("Validation warnings:")
        lines.extend([f"- {w}" for w in warnings_out])
    else:
        lines.append("")
        lines.append("All built-in validation checks passed.")

    dbg_path = results_dir / "validation_debug_report.txt"
    dbg_path.write_text("\n".join(lines), encoding="utf-8")

    if warnings_out:
        print(f"Validation warnings detected. See {dbg_path}", flush=True)
    else:
        print("Validation checks passed without warnings.", flush=True)

    return warnings_out


def run_experiment(args: argparse.Namespace) -> None:
    outdir = ensure_dir(args.outdir)
    results_dir = ensure_dir(outdir / "results")
    ensure_dir(outdir / "plots")

    run_id = timestamped_run_id(prefix="expH")

    # Normalize/validate mixture proportions.
    mix = np.asarray(
        [
            float(args.mixture_null),
            float(args.mixture_unimodal),
            float(args.mixture_bimodal),
            float(args.mixture_patchy),
            float(args.mixture_donor_specific),
            float(args.mixture_qc),
        ],
        dtype=float,
    )
    if np.any(mix < 0):
        raise ValueError("Mixture proportions must be non-negative.")
    if float(np.sum(mix)) <= 0:
        raise ValueError("Mixture proportions sum to zero.")
    mix = mix / float(np.sum(mix))

    min_perm_eff = int(args.min_perm)
    if int(args.n_perm) < int(min_perm_eff):
        warnings.warn(
            f"n_perm ({int(args.n_perm)}) < min_perm ({int(min_perm_eff)}); lowering gate to n_perm.",
            RuntimeWarning,
            stacklevel=2,
        )
        min_perm_eff = int(args.n_perm)

    cfg = {
        "experiment": "expH_fdr_pipeline_scale",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit_hash(cwd=REPO_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "master_seed": int(args.master_seed),
        "runs": int(args.runs),
        "N": int(args.N),
        "D": int(args.D),
        "G_total": int(args.G_total),
        "bins": int(args.bins),
        "mode": str(args.mode),
        "w": int(args.w),
        "n_perm": int(args.n_perm),
        "geometry": [str(g) for g in args.geometry],
        "sigma_eta": float(args.sigma_eta),
        "enable_qc_flag": bool(args.enable_qc_flag),
        "mixture": {
            "NULL": float(mix[0]),
            "TRUE_UNIMODAL": float(mix[1]),
            "TRUE_BIMODAL": float(mix[2]),
            "TRUE_PATCHY": float(mix[3]),
            "DONOR_SPECIFIC": float(mix[4]),
            "QC_DRIVEN": float(mix[5]),
        },
        "underpowered": {
            "p_min": float(args.p_min),
            "min_fg_total": int(args.min_fg_total),
            "min_fg_per_donor": int(args.min_fg_per_donor),
            "min_bg_per_donor": int(args.min_bg_per_donor),
            "d_eff_min": int(args.d_eff_min),
            "min_perm": int(min_perm_eff),
        },
    }
    write_config(outdir, cfg)

    print(
        (
            "Running ExpH with "
            f"runs={int(args.runs)}, geometries={list(args.geometry)}, N={int(args.N)}, D={int(args.D)}, "
            f"G_total={int(args.G_total)}, n_perm={int(args.n_perm)}, mode={str(args.mode)}, w={int(args.w)}"
        ),
        flush=True,
    )

    contexts: dict[tuple[int, str], DatasetContext] = {}
    rows: list[dict[str, Any]] = []

    total_cells = int(args.runs) * len(args.geometry) * int(args.G_total)
    counter = 0
    t0 = time.time()

    power_cfg = {
        "p_min": float(args.p_min),
        "min_fg_total": int(args.min_fg_total),
        "min_fg_per_donor": int(args.min_fg_per_donor),
        "min_bg_per_donor": int(args.min_bg_per_donor),
        "d_eff_min": int(args.d_eff_min),
    }

    for run_idx in range(int(args.runs)):
        seed_run = int(args.master_seed) + int(run_idx)
        for geom in args.geometry:
            context = _simulate_dataset_context(
                master_seed=int(args.master_seed),
                run_idx=int(run_idx),
                geometry=str(geom),
                N=int(args.N),
                D=int(args.D),
                sigma_eta=float(args.sigma_eta),
                bins=int(args.bins),
                density_k=float(args.density_k),
            )
            contexts[(int(seed_run), str(geom))] = context

            rng_cat = rng_from_seed(
                stable_seed(
                    int(args.master_seed), "expH", "categories", int(run_idx), str(geom)
                )
            )
            cat_idx = rng_cat.choice(
                np.arange(len(TRUTH_CATEGORIES), dtype=int),
                size=int(args.G_total),
                p=mix,
            )
            cat_labels = [TRUTH_CATEGORIES[int(i)] for i in cat_idx]

            print(
                f"Dataset run={run_idx}, seed={seed_run}, geometry={geom}", flush=True
            )

            for g_idx in range(int(args.G_total)):
                category = str(cat_labels[g_idx])
                seed_gene = stable_seed(
                    int(args.master_seed),
                    "expH",
                    "gene",
                    int(run_idx),
                    str(geom),
                    int(g_idx),
                    str(category),
                )

                sim = _simulate_gene(
                    category=category, context=context, seed_gene=int(seed_gene)
                )
                f = sim.f

                seed_perm = stable_seed(
                    int(args.master_seed),
                    "expH",
                    "perm",
                    int(run_idx),
                    str(geom),
                    int(g_idx),
                    str(category),
                )
                score = _score_gene(
                    f=f,
                    context=context,
                    bins=int(args.bins),
                    mode=str(args.mode),
                    smooth_w=int(args.w),
                    n_perm=int(args.n_perm),
                    seed_perm=int(seed_perm),
                    power_cfg=power_cfg,
                    min_perm_eff=int(min_perm_eff),
                )

                qc = _qc_assoc_metrics(f, context)

                rows.append(
                    {
                        "run_id": run_id,
                        "seed_run": int(seed_run),
                        "run_index": int(run_idx),
                        "seed": int(seed_gene),
                        "geometry": str(geom),
                        "truth_category": category,
                        "truth_shape_class": _truth_shape_class(category),
                        "pi_target": float(sim.pi_target),
                        "beta": float(sim.beta),
                        "gamma": float(sim.gamma),
                        "dropout_noise": float(sim.dropout_noise),
                        "theta0": float(sim.theta0),
                        "patchy_width": float(sim.patchy_width),
                        **{k: v for k, v in score.items() if k != "E_obs"},
                        **qc,
                        "E_obs_json": json.dumps(
                            np.asarray(score["E_obs"], dtype=float).tolist(),
                            separators=(",", ":"),
                        ),
                    }
                )

                counter += 1
                if counter % int(args.progress_every) == 0 or counter == int(
                    total_cells
                ):
                    elapsed = time.time() - t0
                    rate = counter / elapsed if elapsed > 0 else float("nan")
                    print(
                        f"  progress: genes={counter}/{total_cells} ({rate:.2f} genes/s)",
                        flush=True,
                    )

    gene_table = pd.DataFrame(rows)
    if gene_table.empty:
        raise RuntimeError("No genes simulated.")

    gene_table = _apply_bh_and_calls(
        gene_table,
        enable_qc_flag=bool(args.enable_qc_flag),
        q_thresh=float(args.q_thresh),
    )

    summary_runlevel = summarize_runlevel(gene_table)
    summary_by_truth = summarize_by_truth(gene_table)

    atomic_write_csv(results_dir / "gene_table.csv", gene_table)
    atomic_write_csv(results_dir / "summary_runlevel.csv", summary_runlevel)
    atomic_write_csv(results_dir / "summary_by_truth.csv", summary_by_truth)

    meta = (
        f"N={int(args.N)}, D={int(args.D)}, B={int(args.bins)}, n_perm={int(args.n_perm)}, "
        f"mode={str(args.mode)}, w={int(args.w)}"
    )

    _set_plot_style()
    plot_fdr_power_by_run(summary_runlevel, outdir, meta)
    plot_call_composition(gene_table, outdir, meta)
    plot_null_calibration(gene_table, outdir, meta)
    plot_fdr_power_curves(gene_table, outdir, meta, bool(args.enable_qc_flag))
    plot_shape_confusion(gene_table, outdir, meta)
    plot_score_scatter(gene_table, outdir, meta)
    plot_underpowered_impact(gene_table, outdir, meta)
    plot_example_panels(gene_table, contexts, outdir, int(args.bins))

    warnings_out = run_validations(
        gene_table, summary_runlevel, outdir, bool(args.enable_qc_flag)
    )

    elapsed = time.time() - t0
    print(
        (
            f"Completed ExpH in {elapsed/60.0:.2f} min. "
            f"rows={gene_table.shape[0]}, warnings={len(warnings_out)}"
        ),
        flush=True,
    )
    print(f"Outputs written to: {outdir}", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Simulation Experiment H: FDR + triage/classification pipeline reliability."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/simulations/expH_fdr_pipeline_scale",
        help="Output directory.",
    )
    parser.add_argument("--master_seed", type=int, default=123)
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument("--N", type=int, default=20000)
    parser.add_argument("--D", type=int, default=10)
    parser.add_argument("--G_total", type=int, default=5000)

    parser.add_argument("--bins", type=int, default=36)
    parser.add_argument(
        "--mode", type=str, choices=["raw", "smoothed"], default="smoothed"
    )
    parser.add_argument("--w", type=int, default=3)
    parser.add_argument("--n_perm", type=int, default=500)

    parser.add_argument(
        "--geometry",
        type=str,
        nargs="+",
        default=["disk_gaussian", "density_gradient_disk"],
    )
    parser.add_argument("--sigma_eta", type=float, default=0.4)
    parser.add_argument("--density_k", type=float, default=1.5)

    parser.add_argument("--enable_qc_flag", action="store_true")
    parser.add_argument("--q_thresh", type=float, default=0.05)

    parser.add_argument("--mixture_null", type=float, default=0.85)
    parser.add_argument("--mixture_unimodal", type=float, default=0.06)
    parser.add_argument("--mixture_bimodal", type=float, default=0.03)
    parser.add_argument("--mixture_patchy", type=float, default=0.03)
    parser.add_argument("--mixture_donor_specific", type=float, default=0.02)
    parser.add_argument("--mixture_qc", type=float, default=0.01)

    parser.add_argument("--p_min", type=float, default=0.005)
    parser.add_argument("--min_fg_total", type=int, default=50)
    parser.add_argument("--min_fg_per_donor", type=int, default=10)
    parser.add_argument("--min_bg_per_donor", type=int, default=10)
    parser.add_argument("--d_eff_min", type=int, default=2)
    parser.add_argument("--min_perm", type=int, default=200)

    parser.add_argument("--progress_every", type=int, default=200)

    add_common_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    warnings.simplefilter("default", RuntimeWarning)
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if bool(getattr(args, "dry_run", False)):
        print("dry_run=True: skipping execution for legacy runner.", flush=True)
        return 0
    if bool(getattr(args, "test_mode", False)):
        args = apply_testmode_overrides(args, exp_name="expH_fdr_pipeline_scale")

    if int(args.N) <= 0 or int(args.D) <= 0 or int(args.G_total) <= 0:
        raise ValueError("N, D, G_total must be positive.")
    if int(args.n_perm) <= 0 or int(args.bins) <= 0:
        raise ValueError("n_perm and bins must be positive.")
    if str(args.mode) == "smoothed" and (int(args.w) < 1 or int(args.w) % 2 == 0):
        raise ValueError("For smoothed mode, w must be odd and >=1.")

    run_ctx = prepare_legacy_run(args, "expH_fdr_pipeline_scale", __file__)
    run_experiment(args)
    finalize_legacy_run(run_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
