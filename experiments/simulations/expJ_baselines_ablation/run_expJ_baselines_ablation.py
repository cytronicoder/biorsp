#!/usr/bin/env python3
"""Simulation Experiment J: baselines + ablation study for BioRSP."""

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
from scipy import sparse
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.special import expit
from scipy.stats import kstest, norm, rankdata, spearmanr

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-biorsp-expJ")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache-biorsp-expJ")
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from biorsp.permutation import perm_null_T
from biorsp.power import evaluate_underpowered
from biorsp.rsp import compute_rsp_profile_from_boolean
from biorsp.scoring import bh_fdr, robust_z
from biorsp.smoothing import circular_moving_average
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
    "QC_DRIVEN",
    "DONOR_SPECIFIC",
]

TRUE_SPATIAL = {"TRUE_UNIMODAL", "TRUE_BIMODAL", "TRUE_PATCHY"}

TRUTH_TO_SHAPE = {
    "TRUE_UNIMODAL": "LOCALIZED_UNIMODAL",
    "TRUE_BIMODAL": "LOCALIZED_BIMODAL",
    "TRUE_PATCHY": "LOCALIZED_MULTIMODAL",
}

SHAPE_CLASSES = [
    "LOCALIZED_UNIMODAL",
    "LOCALIZED_BIMODAL",
    "LOCALIZED_MULTIMODAL",
    "UNCERTAIN_SHAPE",
]


@dataclass(frozen=True)
class DatasetContext:
    run_idx: int
    run_id: str
    geometry: str
    X: np.ndarray
    theta: np.ndarray
    donor_ids: np.ndarray
    donor_to_idx: dict[str, np.ndarray]
    eta_d: np.ndarray
    eta_cell: np.ndarray
    log_library: np.ndarray
    z_log_library: np.ndarray
    bin_cache: dict[int, tuple[np.ndarray, np.ndarray]]
    W: sparse.csr_matrix | None
    W_sum: float | None


@dataclass(frozen=True)
class GeneMeta:
    gene_id: int
    truth_category: str
    pi_target: float
    beta: float
    gamma: float
    dropout_noise: float
    theta0: float
    patchy_width: float


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    bins: int
    mode: str
    w: int
    donor_stratified: bool
    underpowered_gate: bool
    stat_kind: str
    use_permutation: bool
    shape_enabled: bool
    shape_ablate: bool


def _fmt(x: float | int) -> str:
    return f"{float(x):.3f}".rstrip("0").rstrip(".")


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


def _zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sd


def _compute_bin_cache(theta: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    wrapped = np.mod(np.asarray(theta, dtype=float).ravel(), 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, int(bins) + 1, endpoint=True)
    bin_id = np.digitize(wrapped, edges, right=False) - 1
    bin_id = np.where(bin_id == int(bins), int(bins) - 1, bin_id).astype(np.int32)
    counts = np.bincount(bin_id, minlength=int(bins)).astype(np.int64)
    return bin_id, counts


def _donor_to_idx_map(donor_ids: np.ndarray) -> dict[str, np.ndarray]:
    d = np.asarray(donor_ids)
    labels = np.unique(d)
    return {str(k): np.flatnonzero(d == k).astype(int) for k in labels}


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
    idx = int(rng.choice(np.arange(vals.size), p=probs))
    return float(vals[idx])


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
    return _zscore(s)


def _build_knn_graph(X: np.ndarray, k: int) -> tuple[sparse.csr_matrix, float]:
    n = int(X.shape[0])
    tree = cKDTree(np.asarray(X, dtype=float))
    _, nn = tree.query(np.asarray(X, dtype=float), k=int(k) + 1)
    rows = np.repeat(np.arange(n, dtype=int), int(k))
    cols = np.asarray(nn[:, 1:], dtype=int).reshape(-1)
    data = np.ones(rows.size, dtype=float)

    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
    W = A.maximum(A.T).tocsr()

    rs = np.asarray(W.sum(axis=1)).ravel().astype(float)
    rs_safe = np.where(rs > 0.0, rs, 1.0)
    W = sparse.diags(1.0 / rs_safe, dtype=float).dot(W).tocsr()
    W_sum = float(np.asarray(W.sum()).ravel()[0])
    return W, W_sum


def _moran_I(f: np.ndarray, W: sparse.csr_matrix, W_sum: float) -> float:
    x = np.asarray(f, dtype=float).ravel()
    n = int(x.size)
    x_c = x - float(np.mean(x))
    den = float(np.dot(x_c, x_c))
    if den <= 1e-12 or n <= 1 or W_sum <= 1e-12:
        return 0.0
    wx = W.dot(x_c)
    num = float(np.dot(x_c, wx))
    return float((n / W_sum) * (num / den))


def _rayleigh_stat(f: np.ndarray, theta: np.ndarray) -> float:
    mask = np.asarray(f, dtype=bool).ravel()
    n_fg = int(np.sum(mask))
    if n_fg <= 1:
        return 0.0
    th = np.asarray(theta, dtype=float).ravel()[mask]
    c = float(np.sum(np.cos(th)))
    s = float(np.sum(np.sin(th)))
    R = math.sqrt(c * c + s * s) / float(n_fg)
    return float(float(n_fg) * (R * R))


def _chisq_stat(
    f: np.ndarray, bin_id: np.ndarray, bin_counts_total: np.ndarray
) -> float:
    ff = np.asarray(f, dtype=bool).ravel()
    b = np.asarray(bin_id, dtype=int).ravel()
    total = np.asarray(bin_counts_total, dtype=float).ravel()
    n = int(ff.size)
    n_fg = int(np.sum(ff))
    n_bg = n - n_fg
    if n_fg <= 0 or n_bg <= 0:
        return 0.0

    fg = np.bincount(b[ff], minlength=total.size).astype(float)
    bg = total - fg
    obs = np.vstack([fg, bg])

    row_tot = np.asarray([n_fg, n_bg], dtype=float)
    col_tot = total.astype(float)
    exp = np.outer(row_tot, col_tot) / float(n)
    mask = exp > 1e-12
    stat = np.sum(((obs - exp) ** 2)[mask] / exp[mask])
    return float(stat)


def _profile_with_mode(
    f: np.ndarray,
    theta: np.ndarray,
    bins: int,
    mode: str,
    w: int,
    bin_id: np.ndarray | None,
    bin_counts: np.ndarray | None,
) -> np.ndarray:
    E_raw, _, _ = compute_rsp_profile_from_boolean(
        np.asarray(f, dtype=bool),
        np.asarray(theta, dtype=float),
        int(bins),
        bin_id=bin_id,
        bin_counts_total=bin_counts,
    )
    if str(mode) == "smoothed":
        return circular_moving_average(np.asarray(E_raw, dtype=float), int(w))
    return np.asarray(E_raw, dtype=float)


def _max_peak_prominence_positive(E: np.ndarray) -> float:
    arr = np.asarray(E, dtype=float).ravel()
    if arr.size < 3:
        return 0.0
    peaks, props = find_peaks(arr, prominence=0.0)
    if peaks.size == 0:
        return 0.0
    vals = arr[peaks]
    if "prominences" not in props:
        return 0.0
    prom = np.asarray(props["prominences"], dtype=float)
    if prom.size == 0:
        return 0.0
    mask = vals > 0.0
    if not np.any(mask):
        return 0.0
    return float(np.max(prom[mask]))


def _count_peaks_null_calibrated(
    E_obs: np.ndarray, null_E: np.ndarray, q: float = 0.95
) -> tuple[int, float]:
    obs = np.asarray(E_obs, dtype=float).ravel()
    null = np.asarray(null_E, dtype=float)
    if null.ndim != 2 or null.shape[0] < 1 or null.shape[1] != obs.size:
        return 0, float("nan")

    max_prom = np.array(
        [_max_peak_prominence_positive(row) for row in null], dtype=float
    )
    thr = float(np.quantile(max_prom, float(q)))
    peaks, _ = find_peaks(obs, prominence=max(float(thr), 1e-12))
    if peaks.size == 0:
        return 0, float(thr)
    vals = obs[peaks]
    k = int(np.sum(vals > 0.0))
    return k, float(thr)


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


def _shuffle_boolean(
    f: np.ndarray,
    rng: np.random.Generator,
    donor_to_idx: dict[str, np.ndarray],
    donor_stratified: bool,
) -> np.ndarray:
    arr = np.asarray(f, dtype=bool).ravel()
    if not donor_stratified:
        return arr[rng.permutation(arr.size)]

    out = arr.copy()
    for idx in donor_to_idx.values():
        ii = np.asarray(idx, dtype=int)
        if ii.size <= 1:
            continue
        out[ii] = arr[ii[rng.permutation(ii.size)]]
    return out


def _sample_truth_category(
    rng: np.random.Generator,
    geometry: str,
    enable_qc_in_geo2: bool,
    probs: dict[str, float],
) -> str:
    p = {k: float(v) for k, v in probs.items()}
    if bool(enable_qc_in_geo2) and str(geometry) != "density_gradient_disk":
        p["NULL"] = float(p.get("NULL", 0.0) + p.get("QC_DRIVEN", 0.0))
        p["QC_DRIVEN"] = 0.0

    keys = TRUTH_CATEGORIES.copy()
    vals = np.asarray([max(0.0, p.get(k, 0.0)) for k in keys], dtype=float)
    vals = vals / float(np.sum(vals))
    i = int(rng.choice(np.arange(len(keys), dtype=int), p=vals))
    return str(keys[i])


def _simulate_dataset_context(
    *,
    master_seed: int,
    run_idx: int,
    geometry: str,
    N: int,
    D: int,
    sigma_eta: float,
    bin_sizes: list[int],
    k_nn: int,
    include_moran: bool,
    enable_qc_in_geo2: bool,
    density_k: float,
) -> DatasetContext:
    seed_ctx = stable_seed(
        int(master_seed), "expJ", "context", int(run_idx), str(geometry), int(N), int(D)
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

    mu0 = 10.0
    mu_shift = rng.normal(loc=0.0, scale=0.4, size=int(D)).astype(float)
    mu_d = float(mu0) + mu_shift
    log_library = rng.normal(loc=mu_d[donor_ids], scale=0.6, size=int(N)).astype(float)

    if bool(enable_qc_in_geo2) and str(geometry) == "density_gradient_disk":
        log_library = log_library + 1.0 * np.asarray(X[:, 0], dtype=float)
    z_log_library = _zscore(log_library)

    bin_cache = {}
    for b in sorted(set(int(x) for x in bin_sizes)):
        bin_cache[int(b)] = _compute_bin_cache(theta, int(b))

    W = None
    W_sum = None
    if bool(include_moran):
        W, W_sum = _build_knn_graph(X, int(k_nn))

    return DatasetContext(
        run_idx=int(run_idx),
        run_id=f"run{int(run_idx):02d}",
        geometry=str(geometry),
        X=X,
        theta=theta,
        donor_ids=donor_ids,
        donor_to_idx=donor_to_idx,
        eta_d=eta_d,
        eta_cell=eta_cell,
        log_library=log_library,
        z_log_library=z_log_library,
        bin_cache=bin_cache,
        W=W,
        W_sum=W_sum,
    )


def _simulate_gene(
    *,
    context: DatasetContext,
    gene_id: int,
    seed_gene: int,
    probs: dict[str, float],
    enable_qc_in_geo2: bool,
) -> tuple[GeneMeta, np.ndarray]:
    rng = rng_from_seed(int(seed_gene))
    cat = _sample_truth_category(rng, context.geometry, bool(enable_qc_in_geo2), probs)

    pi_target = _sample_pi_target(rng)
    dropout = _sample_dropout(rng)
    theta0 = float(rng.uniform(0.0, 2.0 * np.pi))
    patchy_width = float(rng.uniform(0.28, 0.42))

    beta = 0.0
    gamma = 0.0

    if cat in {"TRUE_UNIMODAL", "TRUE_BIMODAL"}:
        beta = float(rng.uniform(0.5, 1.25))
    elif cat in {"TRUE_PATCHY", "DONOR_SPECIFIC"}:
        beta = float(rng.uniform(0.6, 1.4))
    elif cat == "QC_DRIVEN":
        gamma = float(rng.uniform(0.8, 1.5))

    alpha = math.log(_clip_pi(pi_target) / (1.0 - _clip_pi(pi_target)))

    if cat == "NULL":
        logits = alpha + context.eta_cell
    elif cat == "TRUE_UNIMODAL":
        logits = (
            alpha
            + context.eta_cell
            + float(beta) * np.cos(context.theta - float(theta0))
        )
    elif cat == "TRUE_BIMODAL":
        logits = (
            alpha
            + context.eta_cell
            + float(beta) * np.cos(2.0 * (context.theta - float(theta0)))
        )
    elif cat == "TRUE_PATCHY":
        sig = _patchy_signal(context.theta, float(theta0), float(patchy_width))
        logits = alpha + context.eta_cell + float(beta) * sig
    elif cat == "DONOR_SPECIFIC":
        theta0_d = rng.uniform(0.0, 2.0 * np.pi, size=context.eta_d.size).astype(float)
        logits = (
            alpha
            + context.eta_cell
            + float(beta) * np.cos(context.theta - theta0_d[context.donor_ids])
        )
    elif cat == "QC_DRIVEN":
        if bool(enable_qc_in_geo2) and context.geometry == "density_gradient_disk":
            logits = alpha + context.eta_cell + float(gamma) * context.z_log_library
        else:
            logits = alpha + context.eta_cell
    else:
        raise ValueError(cat)

    p = expit(logits)
    f = (rng.random(p.size) < p).astype(bool)

    if float(dropout) > 0.0:
        ones = np.flatnonzero(f)
        if ones.size:
            dm = rng.random(ones.size) < float(dropout)
            f[ones[dm]] = False

    meta = GeneMeta(
        gene_id=int(gene_id),
        truth_category=str(cat),
        pi_target=float(pi_target),
        beta=float(beta),
        gamma=float(gamma),
        dropout_noise=float(dropout),
        theta0=float(theta0),
        patchy_width=float(patchy_width),
    )
    return meta, f


def _build_methods(args: argparse.Namespace) -> list[MethodSpec]:
    methods: list[MethodSpec] = [
        MethodSpec(
            name="BioRSP_FULL",
            family="bio",
            bins=int(args.bins),
            mode=str(args.mode),
            w=int(args.w),
            donor_stratified=True,
            underpowered_gate=True,
            stat_kind="max",
            use_permutation=True,
            shape_enabled=True,
            shape_ablate=False,
        )
    ]

    ablations = set(str(x) for x in args.include_ablations)

    if "A1_raw" in ablations:
        methods.append(
            MethodSpec(
                "BioRSP_A1_raw",
                "bio",
                int(args.bins),
                "raw",
                1,
                True,
                True,
                "max",
                True,
                True,
                False,
            )
        )
    if "A2_nostrat" in ablations:
        methods.append(
            MethodSpec(
                "BioRSP_A2_nostrat",
                "bio",
                int(args.bins),
                str(args.mode),
                int(args.w),
                False,
                True,
                "max",
                True,
                True,
                False,
            )
        )
    if "A3_nogate" in ablations:
        methods.append(
            MethodSpec(
                "BioRSP_A3_nogate",
                "bio",
                int(args.bins),
                str(args.mode),
                int(args.w),
                True,
                False,
                "max",
                True,
                True,
                False,
            )
        )
    if "A4_nonmax" in ablations:
        methods.append(
            MethodSpec(
                "BioRSP_A4_nonmax",
                "bio",
                int(args.bins),
                str(args.mode),
                int(args.w),
                True,
                True,
                "nonmax",
                True,
                True,
                False,
            )
        )
    if "A5_noperm" in ablations:
        methods.append(
            MethodSpec(
                "BioRSP_A5_noperm",
                "bio",
                int(args.bins),
                str(args.mode),
                int(args.w),
                True,
                True,
                "parametric",
                False,
                False,
                False,
            )
        )
    if "A6_B24" in ablations:
        methods.append(
            MethodSpec(
                "BioRSP_A6_B24",
                "bio",
                24,
                str(args.mode),
                int(args.w),
                True,
                True,
                "max",
                True,
                True,
                False,
            )
        )
    if "A7_B72w5" in ablations:
        methods.append(
            MethodSpec(
                "BioRSP_A7_B72w5",
                "bio",
                72,
                "smoothed",
                5,
                True,
                True,
                "max",
                True,
                True,
                False,
            )
        )
    if "A8_shapeablate" in ablations:
        methods.append(
            MethodSpec(
                "BioRSP_A8_shapeablate",
                "bio",
                int(args.bins),
                str(args.mode),
                int(args.w),
                True,
                True,
                "max",
                True,
                True,
                True,
            )
        )

    bases = set(str(x).lower() for x in args.include_baselines)
    if "moran" in bases:
        methods.append(
            MethodSpec(
                "Moran",
                "moran",
                int(args.bins),
                "raw",
                1,
                True,
                True,
                "moran",
                True,
                False,
                False,
            )
        )
    if "rayleigh" in bases:
        methods.append(
            MethodSpec(
                "Rayleigh",
                "rayleigh",
                int(args.bins),
                "raw",
                1,
                True,
                True,
                "rayleigh",
                True,
                False,
                False,
            )
        )
    if "chisq" in bases:
        methods.append(
            MethodSpec(
                "ChiSq",
                "chisq",
                int(args.bins),
                "raw",
                1,
                True,
                True,
                "chisq",
                True,
                False,
                False,
            )
        )

    return methods


def _score_bio_method(
    *,
    f: np.ndarray,
    context: DatasetContext,
    spec: MethodSpec,
    n_perm: int,
    power_cfg: dict[str, Any],
    seed_perm: int,
) -> dict[str, Any]:
    f_bool = np.asarray(f, dtype=bool).ravel()

    power = evaluate_underpowered(
        donor_ids=context.donor_ids,
        f=f_bool,
        n_perm=int(n_perm),
        p_min=float(power_cfg["p_min"]),
        min_fg_total=int(power_cfg["min_fg_total"]),
        min_fg_per_donor=int(power_cfg["min_fg_per_donor"]),
        min_bg_per_donor=int(power_cfg["min_bg_per_donor"]),
        d_eff_min=int(power_cfg["d_eff_min"]),
        min_perm=int(power_cfg["min_perm"]),
    )

    underpowered = (
        bool(power["underpowered"]) if bool(spec.underpowered_gate) else False
    )
    if underpowered:
        return {
            "underpowered_flag": True,
            "score": float("nan"),
            "p_value": float("nan"),
            "pred_class": "NA",
        }

    bin_id, bin_counts = context.bin_cache[int(spec.bins)]

    if not bool(spec.use_permutation):
        E = _profile_with_mode(
            f=f_bool,
            theta=context.theta,
            bins=int(spec.bins),
            mode=str(spec.mode),
            w=int(spec.w),
            bin_id=bin_id,
            bin_counts=bin_counts,
        )
        sd = float(np.std(E))
        z_param = float(np.max(np.abs(E)) / max(sd, 1e-12))
        p = float(2.0 * norm.sf(abs(z_param)))
        pred = "UNCERTAIN_SHAPE"
        return {
            "underpowered_flag": False,
            "score": float(z_param),
            "p_value": float(np.clip(p, 1e-300, 1.0)),
            "pred_class": pred,
        }

    need_null_profiles = bool(spec.shape_enabled) or str(spec.stat_kind) == "nonmax"
    perm = perm_null_T(
        f=f_bool,
        angles=context.theta,
        donor_ids=context.donor_ids,
        n_bins=int(spec.bins),
        n_perm=int(n_perm),
        seed=int(seed_perm),
        mode=str(spec.mode),
        smooth_w=int(spec.w),
        donor_stratified=bool(spec.donor_stratified),
        return_null_T=True,
        return_obs_profile=True,
        return_null_profiles=bool(need_null_profiles),
        bin_id=bin_id,
        bin_counts_total=bin_counts,
    )

    E_obs = np.asarray(perm["E_phi_obs"], dtype=float)
    if str(spec.stat_kind) == "max":
        obs_stat = float(perm["T_obs"])
        null_stat = np.asarray(perm["null_T"], dtype=float)
        p = float(perm["p_T"])
    elif str(spec.stat_kind) == "nonmax":
        null_E = np.asarray(perm["null_E_phi"], dtype=float)
        obs_stat = float(np.mean(np.abs(E_obs)))
        null_stat = (
            np.mean(np.abs(null_E), axis=1)
            if null_E.ndim == 2 and null_E.shape[0] > 0
            else np.zeros(0, dtype=float)
        )
        p = (
            float((1.0 + np.sum(null_stat >= obs_stat)) / (1.0 + null_stat.size))
            if null_stat.size > 0
            else 1.0
        )
    else:
        raise ValueError(spec.stat_kind)

    z = (
        float(robust_z(obs_stat, np.asarray(null_stat, dtype=float)))
        if np.asarray(null_stat).size > 0
        else 0.0
    )

    pred = "UNCERTAIN_SHAPE"
    if bool(spec.shape_enabled) and need_null_profiles:
        null_E = np.asarray(
            perm.get("null_E_phi", np.zeros((0, E_obs.size), dtype=float)), dtype=float
        )
        k_hat, _ = _count_peaks_null_calibrated(E_obs, null_E, q=0.95)
        pred = _shape_call_from_k(float(k_hat))

    if bool(spec.shape_ablate):
        pred = "LOCALIZED_UNIMODAL"

    return {
        "underpowered_flag": False,
        "score": float(z),
        "p_value": float(np.clip(p, 1e-300, 1.0)),
        "pred_class": str(pred),
    }


def _score_baseline_method(
    *,
    f: np.ndarray,
    context: DatasetContext,
    spec: MethodSpec,
    n_perm: int,
    power_cfg: dict[str, Any],
    seed_perm: int,
) -> dict[str, Any]:
    f_bool = np.asarray(f, dtype=bool).ravel()

    power = evaluate_underpowered(
        donor_ids=context.donor_ids,
        f=f_bool,
        n_perm=int(n_perm),
        p_min=float(power_cfg["p_min"]),
        min_fg_total=int(power_cfg["min_fg_total"]),
        min_fg_per_donor=int(power_cfg["min_fg_per_donor"]),
        min_bg_per_donor=int(power_cfg["min_bg_per_donor"]),
        d_eff_min=int(power_cfg["d_eff_min"]),
        min_perm=int(power_cfg["min_perm"]),
    )
    underpowered = (
        bool(power["underpowered"]) if bool(spec.underpowered_gate) else False
    )
    if underpowered:
        return {
            "underpowered_flag": True,
            "score": float("nan"),
            "p_value": float("nan"),
            "pred_class": "NA",
        }

    if spec.family == "moran":
        if context.W is None or context.W_sum is None:
            return {
                "underpowered_flag": False,
                "score": float("nan"),
                "p_value": 1.0,
                "pred_class": "NA",
            }
        obs = _moran_I(f_bool, context.W, float(context.W_sum))

        def stat_fn(x: np.ndarray) -> float:
            return _moran_I(x, context.W, float(context.W_sum))

    elif spec.family == "rayleigh":
        obs = _rayleigh_stat(f_bool, context.theta)

        def stat_fn(x: np.ndarray) -> float:
            return _rayleigh_stat(x, context.theta)

    elif spec.family == "chisq":
        b, c = context.bin_cache[int(spec.bins)]
        obs = _chisq_stat(f_bool, b, c)

        def stat_fn(x: np.ndarray) -> float:
            return _chisq_stat(x, b, c)

    else:
        raise ValueError(spec.family)

    rng = rng_from_seed(int(seed_perm))
    null = np.zeros(int(n_perm), dtype=float)
    for j in range(int(n_perm)):
        fp = _shuffle_boolean(
            f_bool, rng, context.donor_to_idx, bool(spec.donor_stratified)
        )
        null[j] = stat_fn(fp)

    p = float((1.0 + np.sum(null >= obs)) / (1.0 + null.size))
    score = float(-np.log10(max(p, 1e-300)))

    return {
        "underpowered_flag": False,
        "score": float(score),
        "p_value": float(np.clip(p, 1e-300, 1.0)),
        "pred_class": "NA",
    }


def _apply_bh(gene_scores: pd.DataFrame, q_thresh: float = 0.05) -> pd.DataFrame:
    out = gene_scores.copy()
    out["q_value"] = np.nan

    for _, idx in out.groupby(
        ["geometry", "run_id", "method_name"], sort=False
    ).groups.items():
        ii = np.asarray(list(idx), dtype=int)
        tested = (~out.loc[ii, "underpowered_flag"].astype(bool)).to_numpy(dtype=bool)
        p = pd.to_numeric(out.loc[ii[tested], "p_value"], errors="coerce").to_numpy(
            dtype=float
        )
        if p.size > 0:
            q = bh_fdr(np.where(np.isfinite(p), p, 1.0))
            out.loc[ii[tested], "q_value"] = q

    out["called"] = (
        (~out["underpowered_flag"].astype(bool))
        & np.isfinite(pd.to_numeric(out["q_value"], errors="coerce"))
        & (pd.to_numeric(out["q_value"], errors="coerce") <= float(q_thresh))
    )

    mask_a8 = out["method_name"].astype(str) == "BioRSP_A8_shapeablate"
    out.loc[mask_a8 & out["called"].astype(bool), "pred_class"] = "LOCALIZED_UNIMODAL"
    return out


def _auc_from_scores(score: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(score, dtype=float)
    y = np.asarray(labels, dtype=bool)
    m = np.isfinite(s)
    s = s[m]
    y = y[m]
    n_pos = int(np.sum(y))
    n_neg = int(np.sum(~y))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    sum_pos = float(np.sum(ranks[y]))
    auc = (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def _prop_ci(k: int, n: int) -> tuple[float, float, float]:
    if int(n) <= 0:
        return float("nan"), float("nan"), float("nan")
    p = float(k) / float(n)
    lo, hi = wilson_ci(int(k), int(n))
    return p, float(lo), float(hi)


def _build_metrics_summary(gene_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (geometry, method_name), grp in gene_scores.groupby(
        ["geometry", "method_name"], sort=True
    ):
        tested = ~grp["underpowered_flag"].astype(bool)
        called = grp["called"].astype(bool)
        truth = grp["truth_category"].astype(str)

        n_genes = int(grp.shape[0])
        n_tested = int(np.sum(tested))
        n_called = int(np.sum(called & tested))

        tp = int(np.sum(called & tested & truth.isin(TRUE_SPATIAL)))
        fp = int(np.sum(called & tested & (truth == "NULL")))
        n_true = int(np.sum(tested & truth.isin(TRUE_SPATIAL)))

        fdr, fdr_lo, fdr_hi = _prop_ci(fp, max(1, n_called))
        tpr, tpr_lo, tpr_hi = _prop_ci(tp, max(1, n_true))

        auc_mask = tested & truth.isin(list(TRUE_SPATIAL) + ["NULL"])
        auc_labels = truth[auc_mask].isin(TRUE_SPATIAL).to_numpy(dtype=bool)
        auc_scores = pd.to_numeric(
            grp.loc[auc_mask, "score"], errors="coerce"
        ).to_numpy(dtype=float)
        auc = _auc_from_scores(auc_scores, auc_labels)

        beta_mask = tested & truth.isin(TRUE_SPATIAL)
        beta = pd.to_numeric(grp.loc[beta_mask, "beta"], errors="coerce").to_numpy(
            dtype=float
        )
        scr = pd.to_numeric(grp.loc[beta_mask, "score"], errors="coerce").to_numpy(
            dtype=float
        )
        ok = np.isfinite(beta) & np.isfinite(scr)
        if int(np.sum(ok)) >= 3:
            rho, _ = spearmanr(scr[ok], beta[ok], nan_policy="omit")
            rho_v = float(rho) if rho is not None else float("nan")
        else:
            rho_v = float("nan")

        qc_mask = tested & (truth == "QC_DRIVEN")
        donor_mask = tested & (truth == "DONOR_SPECIFIC")
        qc_leak = (
            float(np.mean((called & qc_mask)[qc_mask]))
            if int(np.sum(qc_mask)) > 0
            else float("nan")
        )
        donor_leak = (
            float(np.mean((called & donor_mask)[donor_mask]))
            if int(np.sum(donor_mask)) > 0
            else float("nan")
        )

        shape_mask = called & tested & truth.isin(TRUE_SPATIAL)
        pred = grp.loc[shape_mask, "pred_class"].astype(str)
        truth_shape = (
            grp.loc[shape_mask, "truth_category"]
            .map(TRUTH_TO_SHAPE)
            .fillna("UNCERTAIN_SHAPE")
        )
        valid = pred != "NA"
        shape_acc = (
            float(np.mean(pred[valid].to_numpy() == truth_shape[valid].to_numpy()))
            if int(np.sum(valid)) > 0
            else float("nan")
        )

        rows.append(
            {
                "geometry": str(geometry),
                "method_name": str(method_name),
                "n_genes": int(n_genes),
                "n_tested": int(n_tested),
                "called": int(n_called),
                "empirical_FDR": float(fdr),
                "empirical_FDR_ci_low": float(fdr_lo),
                "empirical_FDR_ci_high": float(fdr_hi),
                "TPR": float(tpr),
                "TPR_ci_low": float(tpr_lo),
                "TPR_ci_high": float(tpr_hi),
                "AUC": float(auc),
                "spearman_score_beta": float(rho_v),
                "qc_leakage_rate": float(qc_leak),
                "donor_specific_leakage_rate": float(donor_leak),
                "shape_accuracy": float(shape_acc),
            }
        )

    return pd.DataFrame(rows)


def _geo_tag(geometry: str) -> str:
    return "GEO1" if str(geometry) == "disk_gaussian" else "GEO2"


def plot_fdr_power(metrics: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots = ensure_dir(outdir / "plots")
    for geo, gdf in metrics.groupby("geometry", sort=True):
        fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=False)
        x = pd.to_numeric(gdf["empirical_FDR"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(gdf["TPR"], errors="coerce").to_numpy(dtype=float)
        names = gdf["method_name"].astype(str).tolist()
        ax.scatter(x, y, s=38, color="#1f77b4", alpha=0.85)
        for i, nm in enumerate(names):
            if np.isfinite(x[i]) and np.isfinite(y[i]):
                ax.text(x[i] + 0.004, y[i] + 0.004, nm, fontsize=7)
        ax.axvline(0.05, linestyle="--", linewidth=0.8, color="#444444")
        ax.set_xlabel("Empirical FDR")
        ax.set_ylabel("TPR")
        ax.set_xlim(0.0, max(0.2, np.nanmax(x) + 0.03 if x.size else 0.2))
        ax.set_ylim(0.0, min(1.0, np.nanmax(y) + 0.08 if y.size else 1.0))
        ax.set_title(f"FDR vs Power ({_geo_tag(str(geo))})\n{meta}")
        savefig(fig, plots / f"fdr_vs_power_{_geo_tag(str(geo))}.png")
        plt.close(fig)


def plot_auc_spearman(metrics: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots = ensure_dir(outdir / "plots")
    geos = sorted(metrics["geometry"].astype(str).unique().tolist())
    fig, axes = plt.subplots(
        nrows=max(1, len(geos)),
        ncols=2,
        figsize=(12, 3.2 * max(1, len(geos))),
        constrained_layout=False,
    )
    if len(geos) == 1:
        axes = np.array([axes], dtype=object)

    for r, geo in enumerate(geos):
        gdf = metrics.loc[metrics["geometry"] == geo].sort_values("method_name")
        names = gdf["method_name"].astype(str).tolist()
        auc = pd.to_numeric(gdf["AUC"], errors="coerce").to_numpy(dtype=float)
        spr = pd.to_numeric(gdf["spearman_score_beta"], errors="coerce").to_numpy(
            dtype=float
        )

        ax0 = axes[r, 0]
        ax1 = axes[r, 1]

        ax0.bar(np.arange(len(names)), auc, color="#4C78A8", alpha=0.9)
        ax0.set_title(f"AUC ({_geo_tag(geo)})")
        ax0.set_xticks(np.arange(len(names)))
        ax0.set_xticklabels(names, rotation=55, ha="right")
        ax0.set_ylim(0.0, 1.02)

        ax1.bar(np.arange(len(names)), spr, color="#E45756", alpha=0.9)
        ax1.axhline(0.0, linestyle="--", linewidth=0.7, color="#444444")
        ax1.set_title(f"Spearman(score,beta) ({_geo_tag(geo)})")
        ax1.set_xticks(np.arange(len(names)))
        ax1.set_xticklabels(names, rotation=55, ha="right")
        ax1.set_ylim(-0.2, 1.02)

    fig.suptitle(f"AUC and Effect-Size Rank Correlation\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    savefig(fig, plots / "auc_spearman_bars.png")
    plt.close(fig)


def plot_ablation_delta(metrics: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots = ensure_dir(outdir / "plots")
    for geo, gdf in metrics.groupby("geometry", sort=True):
        full = gdf.loc[gdf["method_name"] == "BioRSP_FULL"]
        if full.empty:
            continue
        fdr0 = float(full["empirical_FDR"].iloc[0])
        tpr0 = float(full["TPR"].iloc[0])

        ab = gdf.loc[gdf["method_name"].astype(str).str.startswith("BioRSP_A")].copy()
        ab = ab.sort_values("method_name")
        if ab.empty:
            fig, ax = plt.subplots(figsize=(7.0, 3.0))
            ax.axis("off")
            ax.text(0.5, 0.5, "No ablations selected.", ha="center", va="center")
            ax.set_title(f"Ablation deltas ({_geo_tag(str(geo))})\n{meta}")
            savefig(fig, plots / f"ablation_delta_metrics_{_geo_tag(str(geo))}.png")
            plt.close(fig)
            continue

        names = ab["method_name"].astype(str).tolist()
        d_tpr = pd.to_numeric(ab["TPR"], errors="coerce").to_numpy(dtype=float) - tpr0
        d_fdr = (
            pd.to_numeric(ab["empirical_FDR"], errors="coerce").to_numpy(dtype=float)
            - fdr0
        )

        x = np.arange(len(names), dtype=float)
        w = 0.38

        fig, ax = plt.subplots(figsize=(10.0, 4.4), constrained_layout=False)
        ax.bar(x - w / 2.0, d_tpr, width=w, color="#4C78A8", label="ΔTPR")
        ax.bar(x + w / 2.0, d_fdr, width=w, color="#E45756", label="ΔFDR")
        ax.axhline(0.0, linestyle="--", linewidth=0.8, color="#444444")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Delta vs BioRSP_FULL")
        ax.legend(frameon=False)
        ax.set_title(f"Ablation Impact ({_geo_tag(str(geo))})\n{meta}")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        savefig(fig, plots / f"ablation_delta_metrics_{_geo_tag(str(geo))}.png")
        plt.close(fig)


def plot_null_calibration(gene_scores: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots = ensure_dir(outdir / "plots")
    methods = ["BioRSP_FULL", "Moran", "Rayleigh", "ChiSq"]

    for geo, sdf in gene_scores.groupby("geometry", sort=True):
        present = [
            m for m in methods if m in set(sdf["method_name"].astype(str).unique())
        ]
        if not present:
            continue
        fig, axes = plt.subplots(
            nrows=2,
            ncols=len(present),
            figsize=(3.1 * len(present), 5.2),
            constrained_layout=False,
        )
        if len(present) == 1:
            axes = np.array([[axes[0]], [axes[1]]], dtype=object)

        for j, method in enumerate(present):
            sub = sdf.loc[
                (sdf["method_name"] == method)
                & (sdf["truth_category"] == "NULL")
                & (~sdf["underpowered_flag"].astype(bool))
            ].copy()
            p = pd.to_numeric(sub["p_value"], errors="coerce").to_numpy(dtype=float)
            p = p[np.isfinite(p)]

            axh = axes[0, j]
            axq = axes[1, j]
            p_hist(axh, p, bins=20, density=True)
            qq_plot(axq, p)
            axh.set_title(f"{method} (n={p.size})")
            axh.set_xlabel("p")
            axh.set_ylabel("density")
            axq.set_xlabel("expected")
            axq.set_ylabel("observed")

        fig.suptitle(f"NULL Calibration Small Multiples ({_geo_tag(str(geo))})\n{meta}")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        savefig(
            fig, plots / f"null_calibration_small_multiples_{_geo_tag(str(geo))}.png"
        )
        plt.close(fig)


def plot_shape_confusion(gene_scores: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots = ensure_dir(outdir / "plots")
    sub = gene_scores.loc[
        (gene_scores["method_name"] == "BioRSP_FULL")
        & (gene_scores["called"].astype(bool))
        & (gene_scores["truth_category"].isin(TRUE_SPATIAL))
    ].copy()

    truth_labels = ["LOCALIZED_UNIMODAL", "LOCALIZED_BIMODAL", "LOCALIZED_MULTIMODAL"]
    pred_labels = [
        "LOCALIZED_UNIMODAL",
        "LOCALIZED_BIMODAL",
        "LOCALIZED_MULTIMODAL",
        "UNCERTAIN_SHAPE",
    ]
    mat = np.zeros((len(truth_labels), len(pred_labels)), dtype=int)

    if not sub.empty:
        truth = (
            sub["truth_category"]
            .map(TRUTH_TO_SHAPE)
            .fillna("UNCERTAIN_SHAPE")
            .astype(str)
        )
        pred = sub["pred_class"].fillna("UNCERTAIN_SHAPE").astype(str)
        for i, t in enumerate(truth_labels):
            for j, p in enumerate(pred_labels):
                mat[i, j] = int(np.sum((truth == t) & (pred == p)))

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=False)
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_xticklabels(pred_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(truth_labels)))
    ax.set_yticklabels(truth_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Truth")
    ax.set_title(f"Shape Confusion: BioRSP_FULL\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    savefig(fig, plots / "shape_confusion_BioRSP_FULL.png")
    plt.close(fig)


def plot_call_composition_geo2(
    gene_scores: pd.DataFrame, outdir: Path, meta: str
) -> None:
    plots = ensure_dir(outdir / "plots")
    geo2 = gene_scores.loc[gene_scores["geometry"] == "density_gradient_disk"].copy()
    methods = sorted(geo2["method_name"].astype(str).unique().tolist())
    cats = TRUTH_CATEGORIES

    if not methods:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No GEO2 rows available.", ha="center", va="center")
        ax.set_title(f"Call Composition by Method (GEO2)\n{meta}")
        savefig(fig, plots / "call_composition_by_method_GEO2.png")
        plt.close(fig)
        return

    frac = np.zeros((len(cats), len(methods)), dtype=float)
    for j, m in enumerate(methods):
        sub = geo2.loc[(geo2["method_name"] == m) & (geo2["called"].astype(bool))]
        denom = max(1, int(sub.shape[0]))
        for i, c in enumerate(cats):
            frac[i, j] = float(np.sum(sub["truth_category"] == c) / denom)

    colors = {
        "NULL": "#BBBBBB",
        "TRUE_UNIMODAL": "#4C78A8",
        "TRUE_BIMODAL": "#F58518",
        "TRUE_PATCHY": "#54A24B",
        "QC_DRIVEN": "#E45756",
        "DONOR_SPECIFIC": "#B279A2",
    }

    fig, ax = plt.subplots(
        figsize=(max(8.0, 0.7 * len(methods) + 2.5), 4.8), constrained_layout=False
    )
    bottom = np.zeros(len(methods), dtype=float)
    x = np.arange(len(methods), dtype=float)
    for i, c in enumerate(cats):
        ax.bar(x, frac[i, :], bottom=bottom, color=colors.get(c, None), label=c)
        bottom += frac[i, :]

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction among called")
    ax.set_title(f"Call Composition by Method (GEO2)\n{meta}")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    savefig(fig, plots / "call_composition_by_method_GEO2.png")
    plt.close(fig)


def _score_lookup(
    gene_scores: pd.DataFrame, run_id: str, geometry: str, gene_id: int
) -> dict[str, dict[str, Any]]:
    sub = gene_scores.loc[
        (gene_scores["run_id"] == run_id)
        & (gene_scores["geometry"] == geometry)
        & (gene_scores["gene_id"] == int(gene_id))
    ].copy()
    out: dict[str, dict[str, Any]] = {}
    for _, r in sub.iterrows():
        out[str(r["method_name"])] = {
            "score": r.get("score", np.nan),
            "p": r.get("p_value", np.nan),
            "q": r.get("q_value", np.nan),
            "called": bool(r.get("called", False)),
        }
    return out


def plot_example_panels(
    *,
    gene_scores: pd.DataFrame,
    outdir: Path,
    context_geo2_run0: DatasetContext | None,
    f_cache_geo2_run0: dict[int, np.ndarray],
    meta_cache_geo2_run0: dict[int, GeneMeta],
    bins: int,
    mode: str,
    w: int,
    meta: str,
) -> None:
    plots = ensure_dir(outdir / "plots")

    if context_geo2_run0 is None or not f_cache_geo2_run0:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "Example cache unavailable.", ha="center", va="center")
        ax.set_title(f"Example Panels\n{meta}")
        savefig(fig, plots / "example_panels.png")
        plt.close(fig)
        return

    run_id = context_geo2_run0.run_id
    geo = context_geo2_run0.geometry
    sub = gene_scores.loc[
        (gene_scores["run_id"] == run_id) & (gene_scores["geometry"] == geo)
    ].copy()

    def _called(gid: int, method: str) -> bool:
        s = sub.loc[
            (sub["gene_id"] == int(gid)) & (sub["method_name"] == str(method)), "called"
        ]
        return bool(s.iloc[0]) if not s.empty else False

    cand_patchy = (
        sub.loc[
            (sub["truth_category"] == "TRUE_PATCHY")
            & (sub["method_name"] == "BioRSP_FULL")
            & (sub["called"].astype(bool))
        ]["gene_id"]
        .astype(int)
        .tolist()
    )
    patchy_id = None
    for gid in cand_patchy:
        if not _called(int(gid), "Rayleigh"):
            patchy_id = int(gid)
            break
    if patchy_id is None and cand_patchy:
        patchy_id = int(cand_patchy[0])

    qc_id = None
    cand_qc = sorted(
        set(
            sub.loc[sub["truth_category"] == "QC_DRIVEN", "gene_id"]
            .astype(int)
            .tolist()
        )
    )
    for gid in cand_qc:
        if (not _called(gid, "BioRSP_FULL")) and (
            _called(gid, "BioRSP_A2_nostrat") or _called(gid, "BioRSP_A3_nogate")
        ):
            qc_id = gid
            break
    if qc_id is None and cand_qc:
        qc_id = int(cand_qc[0])

    null_id = None
    cand_null = sorted(
        set(sub.loc[sub["truth_category"] == "NULL", "gene_id"].astype(int).tolist())
    )
    for gid in cand_null:
        if any(
            _called(gid, m)
            for m in [
                "BioRSP_FULL",
                "Moran",
                "Rayleigh",
                "ChiSq",
                "BioRSP_A2_nostrat",
                "BioRSP_A3_nogate",
            ]
        ):
            null_id = gid
            break
    if null_id is None and cand_null:
        null_id = int(cand_null[0])

    picks = [x for x in [patchy_id, qc_id, null_id] if x is not None]
    if not picks:
        picks = list(f_cache_geo2_run0.keys())[:3]

    nrows = len(picks)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=3,
        figsize=(12, max(3.2, 3.2 * nrows)),
        constrained_layout=False,
    )
    if nrows == 1:
        axes = np.array([axes], dtype=object)

    centers = np.linspace(0.0, 2.0 * np.pi, int(bins), endpoint=False)
    for i, gid in enumerate(picks):
        f = np.asarray(f_cache_geo2_run0[int(gid)], dtype=bool)
        meta_g = meta_cache_geo2_run0.get(int(gid))

        ax0, ax1, ax2 = axes[i, 0], axes[i, 1], axes[i, 2]

        plot_embedding_with_foreground(
            context_geo2_run0.X,
            f,
            ax=ax0,
            title=f"gene={gid} embedding",
            s=4.0,
            alpha_bg=0.35,
            alpha_fg=0.75,
        )
        ax0.set_title(f"gene={gid} embedding")

        gs = ax1.get_gridspec()
        ax1.remove()
        ax1 = fig.add_subplot(gs[i, 1], projection="polar")
        b, c = context_geo2_run0.bin_cache[int(bins)]
        E = _profile_with_mode(
            f, context_geo2_run0.theta, int(bins), str(mode), int(w), b, c
        )
        plot_rsp_polar(
            centers,
            E,
            ax=ax1,
            color="#1f77b4",
            linewidth=1.1,
            title="BioRSP profile (polar)",
        )

        lookup = _score_lookup(gene_scores, run_id, geo, int(gid))
        lines = [
            f"truth={meta_g.truth_category if meta_g else 'NA'}",
            f"pi={_fmt(meta_g.pi_target) if meta_g else 'NA'}, beta={_fmt(meta_g.beta) if meta_g else 'NA'}",
        ]
        for m in [
            "BioRSP_FULL",
            "Rayleigh",
            "ChiSq",
            "Moran",
            "BioRSP_A2_nostrat",
            "BioRSP_A3_nogate",
        ]:
            if m in lookup:
                d = lookup[m]
                lines.append(
                    f"{m}: s={_fmt(float(d['score'])) if np.isfinite(float(d['score'])) else 'NA'}, "
                    f"p={_fmt(float(d['p'])) if np.isfinite(float(d['p'])) else 'NA'}, "
                    f"q={_fmt(float(d['q'])) if np.isfinite(float(d['q'])) else 'NA'}, "
                    f"call={int(bool(d['called']))}"
                )
        ax2.axis("off")
        ax2.text(
            0.0,
            1.0,
            "\n".join(lines),
            ha="left",
            va="top",
            family="monospace",
            fontsize=7.3,
        )
        ax2.set_title("Method stats")

    fig.suptitle(f"Example Panels (GEO2)\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    savefig(fig, plots / "example_panels.png")
    plt.close(fig)


def run_validations(
    gene_scores: pd.DataFrame, metrics: pd.DataFrame, outdir: Path
) -> list[str]:
    results = ensure_dir(outdir / "results")
    lines: list[str] = []
    warns: list[str] = []

    def _m(geometry: str, method: str, col: str) -> float:
        s = metrics.loc[
            (metrics["geometry"] == geometry) & (metrics["method_name"] == method), col
        ]
        return float(s.iloc[0]) if not s.empty else float("nan")

    geo1 = "disk_gaussian"
    geo2 = "density_gradient_disk"

    fdr_full_g1 = _m(geo1, "BioRSP_FULL", "empirical_FDR")
    tpr_full_g1 = _m(geo1, "BioRSP_FULL", "TPR")
    lines.append(f"GEO1 BioRSP_FULL: FDR={fdr_full_g1:.3f}, TPR={tpr_full_g1:.3f}")
    if np.isfinite(fdr_full_g1) and fdr_full_g1 > 0.10:
        warns.append("GEO1 BioRSP_FULL FDR above expected nominal range.")

    fdr_full_g2 = _m(geo2, "BioRSP_FULL", "empirical_FDR")
    fdr_a2_g2 = _m(geo2, "BioRSP_A2_nostrat", "empirical_FDR")
    fdr_a5_g2 = _m(geo2, "BioRSP_A5_noperm", "empirical_FDR")
    lines.append(
        f"GEO2 FDR: FULL={fdr_full_g2:.3f}, A2={fdr_a2_g2:.3f}, A5={fdr_a5_g2:.3f}"
    )
    if (
        np.isfinite(fdr_a2_g2)
        and np.isfinite(fdr_full_g2)
        and fdr_a2_g2 <= fdr_full_g2 + 0.01
    ):
        warns.append(
            "A2_nostrat did not show expected calibration degradation in GEO2."
        )
    if (
        np.isfinite(fdr_a5_g2)
        and np.isfinite(fdr_full_g2)
        and fdr_a5_g2 <= fdr_full_g2 + 0.01
    ):
        warns.append("A5_noperm did not show expected degradation in GEO2.")

    for geo in [geo1, geo2]:
        sub = gene_scores.loc[
            (gene_scores["geometry"] == geo)
            & (gene_scores["method_name"] == "Rayleigh")
        ]
        if sub.empty:
            continue
        call_rate_uni = float(
            np.mean(
                sub.loc[sub["truth_category"] == "TRUE_UNIMODAL", "called"].astype(bool)
            )
        )
        call_rate_bi = float(
            np.mean(
                sub.loc[sub["truth_category"] == "TRUE_BIMODAL", "called"].astype(bool)
            )
        )
        call_rate_patch = float(
            np.mean(
                sub.loc[sub["truth_category"] == "TRUE_PATCHY", "called"].astype(bool)
            )
        )
        lines.append(
            f"Rayleigh call rates ({_geo_tag(geo)}): uni={call_rate_uni:.3f}, bi={call_rate_bi:.3f}, patch={call_rate_patch:.3f}"
        )
        if (
            np.isfinite(call_rate_uni)
            and np.isfinite(call_rate_bi)
            and call_rate_uni < call_rate_bi
        ):
            warns.append(
                f"Rayleigh unexpectedly stronger on bimodal than unimodal in {_geo_tag(geo)}."
            )

    fdr_moran_g1 = _m(geo1, "Moran", "empirical_FDR")
    fdr_moran_g2 = _m(geo2, "Moran", "empirical_FDR")
    lines.append(f"Moran FDR GEO1={fdr_moran_g1:.3f}, GEO2={fdr_moran_g2:.3f}")

    for geo in [geo1]:
        for method in ["BioRSP_FULL", "Moran", "Rayleigh", "ChiSq"]:
            sub = gene_scores.loc[
                (gene_scores["geometry"] == geo)
                & (gene_scores["method_name"] == method)
                & (gene_scores["truth_category"] == "NULL")
                & (~gene_scores["underpowered_flag"].astype(bool))
            ]
            p = pd.to_numeric(sub["p_value"], errors="coerce").to_numpy(dtype=float)
            p = p[np.isfinite(p)]
            if p.size >= 100:
                ksp = float(kstest(p, "uniform").pvalue)
                lines.append(
                    f"NULL KS p ({_geo_tag(geo)}, {method}) = {ksp:.4g} (n={p.size})"
                )
                if ksp < 1e-3:
                    warns.append(
                        f"Potential anti-conservative null p-values for {method} in {_geo_tag(geo)}."
                    )

    if warns:
        lines.append("")
        lines.append("Validation warnings:")
        lines.extend([f"- {w}" for w in warns])
    else:
        lines.append("")
        lines.append("All built-in validation checks passed.")

    dbg_path = results / "validation_debug_report.txt"
    dbg_path.write_text("\n".join(lines), encoding="utf-8")
    if warns:
        print(f"Validation warnings detected. See {dbg_path}", flush=True)
    else:
        print("Validation checks passed without warnings.", flush=True)
    return warns


def run_experiment(args: argparse.Namespace) -> None:
    outdir = ensure_dir(args.outdir)
    results = ensure_dir(outdir / "results")
    ensure_dir(outdir / "plots")

    run_id = timestamped_run_id(prefix="expJ")
    methods = _build_methods(args)

    bin_sizes = [int(args.bins)]
    for m in methods:
        if m.family in {"bio", "chisq"}:
            bin_sizes.append(int(m.bins))

    cfg = {
        "experiment": "expJ_baselines_ablation",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit_hash(cwd=REPO_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "master_seed": int(args.master_seed),
        "runs": int(args.runs),
        "N": int(args.N),
        "D": int(args.D),
        "G": int(args.G),
        "geometries": [str(g) for g in args.geometries],
        "bins": int(args.bins),
        "mode": str(args.mode),
        "w": int(args.w),
        "n_perm": int(args.n_perm),
        "k_nn": int(args.k_nn),
        "enable_qc_in_geo2": bool(args.enable_qc_in_geo2),
        "mixture": {
            "NULL": float(args.mixture_null),
            "TRUE_UNIMODAL": float(args.mixture_unimodal),
            "TRUE_BIMODAL": float(args.mixture_bimodal),
            "TRUE_PATCHY": float(args.mixture_patchy),
            "QC_DRIVEN": float(args.mixture_qc),
            "DONOR_SPECIFIC": float(args.mixture_donor_specific),
        },
        "methods": [m.__dict__ for m in methods],
        "underpowered": {
            "p_min": float(args.p_min),
            "min_fg_total": int(args.min_fg_total),
            "min_fg_per_donor": int(args.min_fg_per_donor),
            "min_bg_per_donor": int(args.min_bg_per_donor),
            "d_eff_min": int(args.d_eff_min),
            "min_perm": int(args.min_perm),
        },
    }
    write_config(outdir, cfg)

    power_cfg = {
        "p_min": float(args.p_min),
        "min_fg_total": int(args.min_fg_total),
        "min_fg_per_donor": int(args.min_fg_per_donor),
        "min_bg_per_donor": int(args.min_bg_per_donor),
        "d_eff_min": int(args.d_eff_min),
        "min_perm": int(args.min_perm),
    }

    probs = {
        "NULL": float(args.mixture_null),
        "TRUE_UNIMODAL": float(args.mixture_unimodal),
        "TRUE_BIMODAL": float(args.mixture_bimodal),
        "TRUE_PATCHY": float(args.mixture_patchy),
        "QC_DRIVEN": float(args.mixture_qc),
        "DONOR_SPECIFIC": float(args.mixture_donor_specific),
    }

    vals = np.asarray([max(0.0, probs[k]) for k in TRUTH_CATEGORIES], dtype=float)
    if float(np.sum(vals)) <= 0.0:
        raise ValueError("Mixture proportions sum to zero.")

    include_moran = any(m.family == "moran" for m in methods)

    rows: list[dict[str, Any]] = []
    cache_context_geo2_run0: DatasetContext | None = None
    cache_f_geo2_run0: dict[int, np.ndarray] = {}
    cache_meta_geo2_run0: dict[int, GeneMeta] = {}

    t0 = time.time()
    total_tasks = int(args.runs) * len(args.geometries) * int(args.G)
    done = 0

    print(
        (
            "Running ExpJ with "
            f"runs={int(args.runs)}, geometries={list(args.geometries)}, N={int(args.N)}, D={int(args.D)}, G={int(args.G)}, "
            f"methods={len(methods)}, n_perm={int(args.n_perm)}"
        ),
        flush=True,
    )

    for run_idx in range(int(args.runs)):
        run_tag = f"run{int(run_idx):02d}"
        for geometry in [str(g) for g in args.geometries]:
            print(f"Context {run_tag} / {geometry}", flush=True)
            ctx = _simulate_dataset_context(
                master_seed=int(args.master_seed),
                run_idx=int(run_idx),
                geometry=str(geometry),
                N=int(args.N),
                D=int(args.D),
                sigma_eta=float(args.sigma_eta),
                bin_sizes=bin_sizes,
                k_nn=int(args.k_nn),
                include_moran=bool(include_moran),
                enable_qc_in_geo2=bool(args.enable_qc_in_geo2),
                density_k=float(args.density_k),
            )

            if run_idx == 0 and str(geometry) == "density_gradient_disk":
                cache_context_geo2_run0 = ctx

            for gene_id in range(int(args.G)):
                seed_gene = stable_seed(
                    int(args.master_seed),
                    "expJ",
                    int(run_idx),
                    str(geometry),
                    "gene",
                    int(gene_id),
                )
                meta_g, f = _simulate_gene(
                    context=ctx,
                    gene_id=int(gene_id),
                    seed_gene=int(seed_gene),
                    probs=probs,
                    enable_qc_in_geo2=bool(args.enable_qc_in_geo2),
                )

                if run_idx == 0 and str(geometry) == "density_gradient_disk":
                    cache_f_geo2_run0[int(gene_id)] = np.asarray(f, dtype=bool)
                    cache_meta_geo2_run0[int(gene_id)] = meta_g

                for method in methods:
                    seed_perm = stable_seed(
                        int(args.master_seed),
                        "expJ",
                        "perm",
                        int(run_idx),
                        str(geometry),
                        int(gene_id),
                        str(method.name),
                    )

                    if method.family == "bio":
                        out = _score_bio_method(
                            f=f,
                            context=ctx,
                            spec=method,
                            n_perm=int(args.n_perm),
                            power_cfg=power_cfg,
                            seed_perm=int(seed_perm),
                        )
                    else:
                        out = _score_baseline_method(
                            f=f,
                            context=ctx,
                            spec=method,
                            n_perm=int(args.n_perm),
                            power_cfg=power_cfg,
                            seed_perm=int(seed_perm),
                        )

                    rows.append(
                        {
                            "geometry": str(geometry),
                            "run_id": str(run_tag),
                            "seed": int(args.master_seed),
                            "gene_id": int(gene_id),
                            "truth_category": meta_g.truth_category,
                            "pi_target": float(meta_g.pi_target),
                            "beta": float(meta_g.beta),
                            "gamma": float(meta_g.gamma),
                            "dropout_noise": float(meta_g.dropout_noise),
                            "method_name": method.name,
                            "underpowered_flag": bool(out["underpowered_flag"]),
                            "score": float(out["score"]),
                            "p_value": float(out["p_value"]),
                            "q_value": float("nan"),
                            "called": False,
                            "pred_class": str(out["pred_class"]),
                        }
                    )

                done += 1
                if done % int(args.progress_every) == 0 or done == total_tasks:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else float("nan")
                    print(
                        f"  progress: genes={done}/{total_tasks} ({rate:.2f} genes/s)",
                        flush=True,
                    )

    gene_scores = pd.DataFrame(rows)
    if gene_scores.empty:
        raise RuntimeError("No rows generated.")

    gene_scores = _apply_bh(gene_scores, q_thresh=float(args.q_thresh))
    metrics = _build_metrics_summary(gene_scores)

    atomic_write_csv(results / "gene_scores.csv", gene_scores)
    atomic_write_csv(results / "metrics_summary.csv", metrics)

    mix_str = (
        f"mix[N={_fmt(args.mixture_null)},U={_fmt(args.mixture_unimodal)},B={_fmt(args.mixture_bimodal)},"
        f"P={_fmt(args.mixture_patchy)},Q={_fmt(args.mixture_qc)},Dspec={_fmt(args.mixture_donor_specific)}]"
    )
    meta = (
        f"N={int(args.N)}, D={int(args.D)}, B={int(args.bins)}, w={int(args.w)}, "
        f"n_perm={int(args.n_perm)}, mode={str(args.mode)}, {mix_str}"
    )

    _set_plot_style()
    plot_fdr_power(metrics, outdir, meta)
    plot_auc_spearman(metrics, outdir, meta)
    plot_ablation_delta(metrics, outdir, meta)
    plot_null_calibration(gene_scores, outdir, meta)
    plot_shape_confusion(gene_scores, outdir, meta)
    plot_call_composition_geo2(gene_scores, outdir, meta)
    plot_example_panels(
        gene_scores=gene_scores,
        outdir=outdir,
        context_geo2_run0=cache_context_geo2_run0,
        f_cache_geo2_run0=cache_f_geo2_run0,
        meta_cache_geo2_run0=cache_meta_geo2_run0,
        bins=int(args.bins),
        mode=str(args.mode),
        w=int(args.w),
        meta=meta,
    )

    warns = run_validations(gene_scores, metrics, outdir)

    elapsed = time.time() - t0
    print(
        f"Completed ExpJ in {elapsed/60.0:.2f} min. rows={gene_scores.shape[0]}, methods={len(methods)}, warnings={len(warns)}",
        flush=True,
    )
    print(f"Outputs written to: {outdir}", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Simulation Experiment J: comparative baselines + ablation study."
    )
    p.add_argument(
        "--outdir", type=str, default="experiments/simulations/expJ_baselines_ablation"
    )
    p.add_argument("--master_seed", type=int, default=123)
    p.add_argument("--runs", type=int, default=2)

    p.add_argument("--N", type=int, default=20000)
    p.add_argument("--D", type=int, default=10)
    p.add_argument("--G", type=int, default=3000)
    p.add_argument(
        "--geometries",
        type=str,
        nargs="+",
        default=["disk_gaussian", "density_gradient_disk"],
    )

    p.add_argument("--bins", type=int, default=36)
    p.add_argument("--mode", type=str, choices=["raw", "smoothed"], default="smoothed")
    p.add_argument("--w", type=int, default=3)
    p.add_argument("--n_perm", type=int, default=500)
    p.add_argument("--q_thresh", type=float, default=0.05)
    p.add_argument("--k_nn", type=int, default=15)
    p.add_argument("--sigma_eta", type=float, default=0.4)
    p.add_argument("--density_k", type=float, default=1.5)

    p.add_argument("--enable_qc_in_geo2", action="store_true")

    p.add_argument(
        "--include_baselines",
        type=str,
        nargs="+",
        default=["moran", "rayleigh", "chisq"],
    )
    p.add_argument(
        "--include_ablations",
        type=str,
        nargs="+",
        default=[
            "A1_raw",
            "A2_nostrat",
            "A3_nogate",
            "A4_nonmax",
            "A6_B24",
            "A7_B72w5",
        ],
    )

    p.add_argument("--mixture_null", type=float, default=0.80)
    p.add_argument("--mixture_unimodal", type=float, default=0.08)
    p.add_argument("--mixture_bimodal", type=float, default=0.04)
    p.add_argument("--mixture_patchy", type=float, default=0.04)
    p.add_argument("--mixture_qc", type=float, default=0.02)
    p.add_argument("--mixture_donor_specific", type=float, default=0.02)

    p.add_argument("--p_min", type=float, default=0.005)
    p.add_argument("--min_fg_total", type=int, default=50)
    p.add_argument("--min_fg_per_donor", type=int, default=10)
    p.add_argument("--min_bg_per_donor", type=int, default=10)
    p.add_argument("--d_eff_min", type=int, default=2)
    p.add_argument("--min_perm", type=int, default=200)

    p.add_argument("--progress_every", type=int, default=100)
    add_common_args(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    warnings.simplefilter("default", RuntimeWarning)
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if bool(getattr(args, "dry_run", False)):
        print("dry_run=True: skipping execution for legacy runner.", flush=True)
        return 0
    if bool(getattr(args, "test_mode", False)):
        args = apply_testmode_overrides(args, exp_name="expJ_baselines_ablation")

    if int(args.N) <= 0 or int(args.D) <= 0 or int(args.G) <= 0:
        raise ValueError("N, D, G must be positive.")
    if int(args.bins) <= 0 or int(args.n_perm) <= 0:
        raise ValueError("bins and n_perm must be positive.")
    if str(args.mode) == "smoothed" and (int(args.w) < 1 or int(args.w) % 2 == 0):
        raise ValueError("smoothed mode requires odd w >= 1.")

    run_ctx = prepare_legacy_run(args, "expJ_baselines_ablation", __file__)
    run_experiment(args)
    finalize_legacy_run(run_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
