#!/usr/bin/env python3
"""Simulation Experiment I: embedding-method robustness for BioRSP."""

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
from scipy.stats import kstest, spearmanr

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-biorsp-expI")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache-biorsp-expI")
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from biorsp.permutation import perm_null_T
from biorsp.power import evaluate_underpowered
from biorsp.scoring import bh_fdr, robust_z
from experiments.simulations._shared.cli import add_common_args
from experiments.simulations._shared.donors import (
    assign_donors,
    donor_effect_vector,
    sample_donor_effects,
)
from experiments.simulations._shared.io import (
    atomic_write_csv,
    ensure_dir,
    git_commit_hash,
    timestamped_run_id,
    write_config,
)
from experiments.simulations._shared.plots import (
    plot_embedding_with_foreground,
    plot_rsp_polar,
    savefig,
    wilson_ci,
)
from experiments.simulations._shared.runner import (
    finalize_legacy_run,
    prepare_legacy_run,
)
from experiments.simulations._shared.seeding import rng_from_seed, stable_seed
from experiments.simulations._sim_testmode import apply_testmode_overrides

TRUTH_CATEGORIES = ["NULL", "TRUE_UNIMODAL", "TRUE_BIMODAL", "TRUE_PATCHY"]
TRUE_SPATIAL = {"TRUE_UNIMODAL", "TRUE_BIMODAL", "TRUE_PATCHY"}
SHAPE_CALLED = [
    "LOCALIZED_UNIMODAL",
    "LOCALIZED_BIMODAL",
    "LOCALIZED_MULTIMODAL",
    "UNCERTAIN_SHAPE",
]

DEFAULT_VARIANTS = ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
DEFAULT_ORIGINS = ["O1", "O2", "O3"]


@dataclass(frozen=True)
class BaseContext:
    X_true: np.ndarray
    theta_true: np.ndarray
    donor_ids: np.ndarray
    donor_to_idx: dict[str, np.ndarray]
    eta_d: np.ndarray
    eta_cell: np.ndarray


@dataclass(frozen=True)
class GeneTruth:
    gene_id: int
    truth_category: str
    pi_target: float
    beta: float
    dropout_noise: float
    theta0: float
    patchy_width: float
    f: np.ndarray


@dataclass(frozen=True)
class VariantData:
    name: str
    X: np.ndarray
    origin_O1: tuple[float, float]


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
        return np.zeros_like(arr)
    return (arr - mu) / sd


def _donor_to_idx_map(donor_ids: np.ndarray) -> dict[str, np.ndarray]:
    donor_arr = np.asarray(donor_ids)
    labels = np.unique(donor_arr)
    return {str(d): np.flatnonzero(donor_arr == d).astype(int) for d in labels}


def _sample_pi_target(rng: np.random.Generator) -> float:
    u = float(rng.random())
    if u < 0.5:
        out = float(rng.uniform(0.02, 0.08))
    elif u < 0.8:
        out = float(rng.uniform(0.08, 0.2))
    else:
        out = float(rng.uniform(0.2, 0.7))
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
) -> int:
    obs = np.asarray(E_obs, dtype=float).ravel()
    null = np.asarray(null_E, dtype=float)
    if null.ndim != 2 or null.shape[1] != obs.size or null.shape[0] < 1:
        return 0
    max_prom = np.array(
        [_max_peak_prominence_positive(row) for row in null], dtype=float
    )
    thr = float(np.quantile(max_prom, float(q)))
    peaks, _ = find_peaks(obs, prominence=max(thr, 1e-12))
    if peaks.size == 0:
        return 0
    vals = obs[peaks]
    return int(np.sum(vals > 0.0))


def _shape_call(k_hat: float) -> str:
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


def _generate_base_context(
    *, seed: int, N: int, D: int, sigma_eta: float
) -> BaseContext:
    rng = rng_from_seed(int(seed))
    X = rng.normal(loc=0.0, scale=1.0, size=(int(N), 2)).astype(float)
    theta = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2.0 * np.pi)
    donor_ids = assign_donors(int(N), int(D), rng)
    donor_to_idx = _donor_to_idx_map(donor_ids)
    eta_d = sample_donor_effects(int(D), float(sigma_eta), rng)
    eta_cell = donor_effect_vector(donor_ids, eta_d)
    return BaseContext(
        X_true=X,
        theta_true=theta,
        donor_ids=donor_ids,
        donor_to_idx=donor_to_idx,
        eta_d=eta_d,
        eta_cell=eta_cell,
    )


def _sample_truth_category(rng: np.random.Generator, mixture: np.ndarray) -> str:
    i = int(rng.choice(np.arange(len(TRUTH_CATEGORIES), dtype=int), p=mixture))
    return str(TRUTH_CATEGORIES[i])


def _simulate_gene_truth(
    *, base: BaseContext, gene_id: int, seed_gene: int, mixture: np.ndarray
) -> GeneTruth:
    rng = rng_from_seed(int(seed_gene))
    category = _sample_truth_category(rng, mixture)
    pi_target = _sample_pi_target(rng)
    dropout = _sample_dropout(rng)
    theta0 = float(rng.uniform(0.0, 2.0 * np.pi))
    patchy_width = float(rng.uniform(0.28, 0.42))

    beta = 0.0
    if category == "TRUE_UNIMODAL":
        beta = float(rng.uniform(0.6, 1.25))
    elif category == "TRUE_BIMODAL":
        beta = float(rng.uniform(0.6, 1.25))
    elif category == "TRUE_PATCHY":
        beta = float(rng.uniform(0.7, 1.4))

    alpha = math.log(_clip_pi(pi_target) / (1.0 - _clip_pi(pi_target)))

    if category == "NULL":
        logits = alpha + base.eta_cell
    elif category == "TRUE_UNIMODAL":
        logits = (
            alpha
            + base.eta_cell
            + float(beta) * np.cos(base.theta_true - float(theta0))
        )
    elif category == "TRUE_BIMODAL":
        logits = (
            alpha
            + base.eta_cell
            + float(beta) * np.cos(2.0 * (base.theta_true - float(theta0)))
        )
    elif category == "TRUE_PATCHY":
        sig = _patchy_signal(base.theta_true, float(theta0), float(patchy_width))
        logits = alpha + base.eta_cell + float(beta) * sig
    else:
        raise ValueError(category)

    p = expit(logits)
    f = (rng.random(p.size) < p).astype(bool)

    if float(dropout) > 0.0:
        ones = np.flatnonzero(f)
        if ones.size:
            dm = rng.random(ones.size) < float(dropout)
            f[ones[dm]] = False

    return GeneTruth(
        gene_id=int(gene_id),
        truth_category=category,
        pi_target=float(pi_target),
        beta=float(beta),
        dropout_noise=float(dropout),
        theta0=float(theta0),
        patchy_width=float(patchy_width),
        f=f,
    )


def _transform_variants(X_true: np.ndarray, seed: int) -> dict[str, VariantData]:
    rng = rng_from_seed(int(seed))
    X = np.asarray(X_true, dtype=float)

    def rot(x: np.ndarray, deg: float) -> np.ndarray:
        rad = float(np.deg2rad(float(deg)))
        c = float(np.cos(rad))
        s = float(np.sin(rad))
        R = np.array([[c, -s], [s, c]], dtype=float)
        return x @ R.T

    def affine_random(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = rng.uniform(-1.0, 1.0, size=(2, 2))
        U, _, Vt = np.linalg.svd(u)
        s1 = float(rng.uniform(0.7, 1.7))
        s2 = float(rng.uniform(max(0.35, s1 / 3.0), s1))
        S = np.diag([s1, s2])
        A = U @ S @ Vt
        t = rng.uniform(low=-0.6, high=0.6, size=2).astype(float)
        y = x @ A.T + t
        return y, A, t

    out: dict[str, VariantData] = {}

    out["V0"] = VariantData(name="V0", X=X.copy(), origin_O1=(0.0, 0.0))

    X1 = rot(X, 37.0)
    o1_v1 = rot(np.array([[0.0, 0.0]], dtype=float), 37.0)[0]
    out["V1"] = VariantData(
        name="V1", X=X1, origin_O1=(float(o1_v1[0]), float(o1_v1[1]))
    )

    t2 = np.array([1.5, -0.7], dtype=float)
    X2 = X + t2
    out["V2"] = VariantData(name="V2", X=X2, origin_O1=(float(t2[0]), float(t2[1])))

    X3 = X.copy()
    X3[:, 0] *= 1.6
    X3[:, 1] *= 0.7
    out["V3"] = VariantData(name="V3", X=X3, origin_O1=(0.0, 0.0))

    X4, A4, t4 = affine_random(X)
    o1_v4 = (np.array([[0.0, 0.0]], dtype=float) @ A4.T + t4)[0]
    out["V4"] = VariantData(
        name="V4", X=X4, origin_O1=(float(o1_v4[0]), float(o1_v4[1]))
    )

    X5 = X.copy()
    X5[:, 0] = X5[:, 0] + 0.25 * np.tanh(X5[:, 1])
    out["V5"] = VariantData(name="V5", X=X5, origin_O1=(0.0, 0.0))

    X6 = X.copy()
    X6[:, 0] = X6[:, 0] + 0.6 * np.tanh(2.0 * X6[:, 1])
    X6[:, 1] = X6[:, 1] + 0.2 * np.tanh(2.0 * X6[:, 0])
    out["V6"] = VariantData(name="V6", X=X6, origin_O1=(0.0, 0.0))

    X7 = X.copy()
    X7[:, 1] = X7[:, 1] * (0.5 + expit(X7[:, 0]))
    out["V7"] = VariantData(name="V7", X=X7, origin_O1=(0.0, 0.0))

    X8 = X.copy()
    mask = X8[:, 0] > 0.0
    X8[mask, 0] = X8[mask, 0] + 4.0
    out["V8"] = VariantData(name="V8", X=X8, origin_O1=(0.0, 0.0))

    return out


def _origin_for_mode(variant: VariantData, mode: str) -> tuple[float, float]:
    if mode == "O1":
        return (float(variant.origin_O1[0]), float(variant.origin_O1[1]))
    if mode == "O2":
        x = np.asarray(variant.X[:, 0], dtype=float)
        y = np.asarray(variant.X[:, 1], dtype=float)
        return (float(np.median(x)), float(np.median(y)))
    if mode == "O3":
        return (0.0, 0.0)
    raise ValueError(mode)


def _score_gene_variant(
    *,
    f: np.ndarray,
    theta: np.ndarray,
    donor_ids: np.ndarray,
    bin_id: np.ndarray,
    bin_counts: np.ndarray,
    bins: int,
    mode: str,
    w: int,
    n_perm: int,
    seed_perm: int,
    min_perm_eff: int,
    power_cfg: dict[str, Any],
) -> dict[str, Any]:
    arr = np.asarray(f, dtype=bool).ravel()
    prev = float(np.mean(arr))
    n_fg = int(np.sum(arr))

    power = evaluate_underpowered(
        donor_ids=donor_ids,
        f=arr,
        n_perm=int(n_perm),
        p_min=float(power_cfg["p_min"]),
        min_fg_total=int(power_cfg["min_fg_total"]),
        min_fg_per_donor=int(power_cfg["min_fg_per_donor"]),
        min_bg_per_donor=int(power_cfg["min_bg_per_donor"]),
        d_eff_min=int(power_cfg["d_eff_min"]),
        min_perm=int(min_perm_eff),
    )

    under = bool(power["underpowered"])

    out = {
        "prev_obs": float(prev),
        "n_fg_total": int(n_fg),
        "D_eff": int(power["D_eff"]),
        "underpowered": bool(under),
        "T_obs": float("nan"),
        "p_T": float("nan"),
        "Z_T": float("nan"),
        "K_hat": float("nan"),
        "called_class_prelim": "UNCERTAIN_SHAPE",
        "E_obs_json": "[]",
    }

    if under or n_fg == 0 or n_fg == int(arr.size):
        return out

    perm = perm_null_T(
        f=arr,
        angles=theta,
        donor_ids=donor_ids,
        n_bins=int(bins),
        n_perm=int(n_perm),
        seed=int(seed_perm),
        mode=str(mode),
        smooth_w=int(w),
        donor_stratified=True,
        return_null_T=True,
        return_obs_profile=True,
        return_null_profiles=True,
        bin_id=bin_id,
        bin_counts_total=bin_counts,
    )

    T_obs = float(perm["T_obs"])
    p_T = float(perm["p_T"])
    null_T = np.asarray(perm["null_T"], dtype=float)
    Z = float(robust_z(T_obs, null_T))

    E_obs = np.asarray(perm["E_phi_obs"], dtype=float)
    null_E = np.asarray(perm["null_E_phi"], dtype=float)
    K_hat = int(_count_peaks_null_calibrated(E_obs, null_E, q=0.95))

    out.update(
        {
            "T_obs": float(T_obs),
            "p_T": float(p_T),
            "Z_T": float(Z),
            "K_hat": float(K_hat),
            "called_class_prelim": _shape_call(float(K_hat)),
            "E_obs_json": json.dumps(E_obs.tolist(), separators=(",", ":")),
        }
    )
    return out


def _apply_bh(gene_variant_table: pd.DataFrame, q_thresh: float) -> pd.DataFrame:
    out = gene_variant_table.copy()
    out["q_T"] = np.nan

    for _, idx in out.groupby(["variant", "origin_mode"], sort=False).groups.items():
        ii = np.asarray(list(idx), dtype=int)
        tested = (~out.loc[ii, "underpowered"].astype(bool)).to_numpy(dtype=bool)
        if np.any(tested):
            p = pd.to_numeric(out.loc[ii[tested], "p_T"], errors="coerce").to_numpy(
                dtype=float
            )
            q = bh_fdr(np.where(np.isfinite(p), p, 1.0))
            out.loc[ii[tested], "q_T"] = q

    out["q_T"] = pd.to_numeric(out["q_T"], errors="coerce")
    out["called_localized"] = (
        (~out["underpowered"].astype(bool))
        & np.isfinite(pd.to_numeric(out["q_T"], errors="coerce"))
        & (pd.to_numeric(out["q_T"], errors="coerce") <= float(q_thresh))
    )

    call_class = np.full(out.shape[0], "NOT_CALLED", dtype=object)
    m = out["called_localized"].to_numpy(dtype=bool)
    call_class[m] = out.loc[m, "called_class_prelim"].astype(str).to_numpy()
    out["called_class"] = call_class
    return out


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return float(inter / max(1, union))


def _prop_ci(k: int, n: int) -> tuple[float, float, float]:
    if int(n) <= 0:
        return float("nan"), float("nan"), float("nan")
    p = float(k) / float(n)
    lo, hi = wilson_ci(int(k), int(n))
    return p, float(lo), float(hi)


def build_concordance_summary(
    gene_variant_table: pd.DataFrame,
    baseline_variant: str = "V0",
    baseline_origin: str = "O1",
) -> pd.DataFrame:
    baseline = gene_variant_table.loc[
        (gene_variant_table["variant"] == str(baseline_variant))
        & (gene_variant_table["origin_mode"] == str(baseline_origin))
    ].copy()

    rows: list[dict[str, Any]] = []

    for keys, grp in gene_variant_table.groupby(["variant", "origin_mode"], sort=True):
        merged = baseline[["gene_id", "Z_T", "called_localized", "called_class"]].merge(
            grp[
                [
                    "gene_id",
                    "Z_T",
                    "called_localized",
                    "called_class",
                    "truth_category",
                    "q_T",
                    "underpowered",
                ]
            ],
            on="gene_id",
            suffixes=("_base", "_var"),
            how="inner",
        )
        if merged.empty:
            continue

        z0 = pd.to_numeric(merged["Z_T_base"], errors="coerce").to_numpy(dtype=float)
        z1 = pd.to_numeric(merged["Z_T_var"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(z0) & np.isfinite(z1)
        if int(np.sum(mask)) >= 3:
            rho, _ = spearmanr(z0[mask], z1[mask], nan_policy="omit")
            spearman_z = float(rho) if rho is not None else float("nan")
        else:
            spearman_z = float("nan")

        ids = merged["gene_id"].to_numpy(dtype=int)
        order0 = np.argsort(-np.where(np.isfinite(z0), z0, -np.inf))
        order1 = np.argsort(-np.where(np.isfinite(z1), z1, -np.inf))
        top_sets = {}
        for K in [50, 100, 200]:
            a = set(ids[order0[: min(K, ids.size)]].tolist())
            b = set(ids[order1[: min(K, ids.size)]].tolist())
            top_sets[K] = _jaccard(a, b)

        base_disc = set(
            merged.loc[merged["called_localized_base"].astype(bool), "gene_id"]
            .astype(int)
            .tolist()
        )
        var_disc = set(
            merged.loc[merged["called_localized_var"].astype(bool), "gene_id"]
            .astype(int)
            .tolist()
        )

        discovery_j = _jaccard(base_disc, var_disc)
        rec_base = float(len(base_disc.intersection(var_disc)) / max(1, len(base_disc)))
        novelty = float(len(var_disc.difference(base_disc)) / max(1, len(var_disc)))

        both_disc = merged.loc[
            merged["called_localized_base"].astype(bool)
            & merged["called_localized_var"].astype(bool)
        ].copy()
        if both_disc.empty:
            class_agree = float("nan")
            cm = {}
        else:
            class_agree = float(
                np.mean(
                    both_disc["called_class_base"].astype(str)
                    == both_disc["called_class_var"].astype(str)
                )
            )
            cm = {}
            for cb in SHAPE_CALLED:
                for cv in SHAPE_CALLED:
                    cm[f"cm_base_{cb}_var_{cv}"] = int(
                        np.sum(
                            (both_disc["called_class_base"] == cb)
                            & (both_disc["called_class_var"] == cv)
                        )
                    )

        tested = (~merged["underpowered"].astype(bool)).to_numpy(dtype=bool)
        called = merged["called_localized_var"].astype(bool).to_numpy(dtype=bool)
        truth_spatial = merged["truth_category"].isin(TRUE_SPATIAL).to_numpy(dtype=bool)

        tp = int(np.sum(called & truth_spatial & tested))
        fp = int(np.sum(called & (~truth_spatial) & tested))
        n_true = int(np.sum(truth_spatial & tested))
        n_called = int(np.sum(called & tested))

        tpr = float(tp / max(1, n_true))
        fdr = float(fp / max(1, n_called))
        tpr_ci = _prop_ci(tp, max(1, n_true))
        fdr_ci = _prop_ci(fp, max(1, n_called))

        rows.append(
            {
                "variant": str(keys[0]),
                "origin_mode": str(keys[1]),
                "spearman_Z_vs_baseline": float(spearman_z),
                "top50_jaccard": float(top_sets[50]),
                "top100_jaccard": float(top_sets[100]),
                "top200_jaccard": float(top_sets[200]),
                "discovery_jaccard": float(discovery_j),
                "discovery_recall_baseline": float(rec_base),
                "discovery_novelty_rate": float(novelty),
                "class_agreement_rate": float(class_agree),
                "truth_TPR": float(tpr),
                "truth_TPR_ci_low": float(tpr_ci[1]),
                "truth_TPR_ci_high": float(tpr_ci[2]),
                "truth_FDR": float(fdr),
                "truth_FDR_ci_low": float(fdr_ci[1]),
                "truth_FDR_ci_high": float(fdr_ci[2]),
                **cm,
            }
        )

    return pd.DataFrame(rows)


def _pivot_heatmap(
    conc: pd.DataFrame, value_col: str
) -> tuple[np.ndarray, list[str], list[str]]:
    pv = conc.pivot(index="variant", columns="origin_mode", values=value_col)
    pv = pv.reindex(index=sorted(pv.index), columns=sorted(pv.columns))
    return (
        pv.to_numpy(dtype=float),
        pv.index.astype(str).tolist(),
        pv.columns.astype(str).tolist(),
    )


def plot_spearman_heatmap(conc: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots = ensure_dir(outdir / "plots")
    mat, rows, cols = _pivot_heatmap(conc, "spearman_Z_vs_baseline")
    fig, ax = plt.subplots(
        figsize=(1.1 * max(4, len(cols)) + 1.5, 0.48 * max(4, len(rows)) + 1.6),
        constrained_layout=False,
    )
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = mat[i, j]
            txt = "NA" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    ax.set_title("Spearman(Z) vs baseline")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Rank concordance heatmap\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    savefig(fig, plots / "spearman_heatmap.png")
    plt.close(fig)


def plot_topk_curves(conc: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots = ensure_dir(outdir / "plots")
    origins = sorted(conc["origin_mode"].astype(str).unique().tolist())
    fig, axes = plt.subplots(
        1, len(origins), figsize=(4.8 * len(origins), 3.9), constrained_layout=False
    )
    axes = np.atleast_1d(axes)

    for ax, origin in zip(axes, origins):
        sub = conc.loc[conc["origin_mode"] == origin].copy()
        for _, row in sub.iterrows():
            k = np.array([50, 100, 200], dtype=float)
            y = np.array(
                [row["top50_jaccard"], row["top100_jaccard"], row["top200_jaccard"]],
                dtype=float,
            )
            ax.plot(k, y, marker="o", linewidth=1.2, label=str(row["variant"]))
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("K")
        ax.set_ylabel("Top-K Jaccard")
        ax.set_title(f"origin={origin}")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        by_label = {}
        for handle, label in zip(handles, labels):
            by_label[label] = handle
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            ncol=6,
            frameon=False,
        )
    fig.suptitle(f"Top-K overlap curves\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    savefig(fig, plots / "topk_jaccard_curves.png")
    plt.close(fig)


def plot_discovery_jaccard_heatmap(conc: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots = ensure_dir(outdir / "plots")
    mat, rows, cols = _pivot_heatmap(conc, "discovery_jaccard")
    fig, ax = plt.subplots(
        figsize=(1.1 * max(4, len(cols)) + 1.5, 0.48 * max(4, len(rows)) + 1.6),
        constrained_layout=False,
    )
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = mat[i, j]
            txt = "NA" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    ax.set_title("Discovery-set Jaccard vs baseline")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Discovery stability heatmap\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    savefig(fig, plots / "discovery_jaccard_heatmap.png")
    plt.close(fig)


def plot_translate_origin_failure(table: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots = ensure_dir(outdir / "plots")
    baseline = table.loc[
        (table["variant"] == "V0") & (table["origin_mode"] == "O1"), ["gene_id", "Z_T"]
    ].rename(columns={"Z_T": "Z_base"})
    v2o1 = table.loc[
        (table["variant"] == "V2") & (table["origin_mode"] == "O1"), ["gene_id", "Z_T"]
    ].rename(columns={"Z_T": "Z_v2_o1"})
    v2o3 = table.loc[
        (table["variant"] == "V2") & (table["origin_mode"] == "O3"), ["gene_id", "Z_T"]
    ].rename(columns={"Z_T": "Z_v2_o3"})

    m1 = baseline.merge(v2o1, on="gene_id", how="inner")
    m2 = baseline.merge(v2o3, on="gene_id", how="inner")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=False)
    for ax, df, ycol, ttl in [
        (axes[0], m1, "Z_v2_o1", "Translated with correct origin (O1)"),
        (axes[1], m2, "Z_v2_o3", "Translated with wrong origin (O3)"),
    ]:
        x = pd.to_numeric(df["Z_base"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=10, alpha=0.30, linewidths=0.0)
        if int(np.sum(mask)) >= 3:
            rr, _ = spearmanr(x[mask], y[mask], nan_policy="omit")
            rho = float(rr) if rr is not None else float("nan")
        else:
            rho = float("nan")
        finite = (
            np.concatenate([x[mask], y[mask]]) if int(np.sum(mask)) else np.array([0.0])
        )
        lo, hi = float(np.min(finite)), float(np.max(finite))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#333333", linewidth=0.8)
        ax.set_xlabel("Z baseline V0,O1")
        ax.set_ylabel("Z translated")
        ax.set_title(f"{ttl}\nSpearman={_fmt(rho)}")

    fig.suptitle(f"Translation-origin failure demonstration\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    savefig(fig, plots / "translate_origin_failure.png")
    plt.close(fig)


def plot_truth_tpr_fdr_curves(table: pd.DataFrame, outdir: Path, meta: str) -> None:
    plots = ensure_dir(outdir / "plots")
    df = table.loc[
        (table["origin_mode"] == "O1") & (~table["underpowered"].astype(bool))
    ].copy()
    if df.empty:
        return
    thresholds = np.linspace(0.01, 0.2, 40)
    variants = sorted(df["variant"].astype(str).unique().tolist())

    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=False)
    for variant in variants:
        sub = df.loc[df["variant"] == variant].copy()
        tpr_vals: list[float] = []
        fdr_vals: list[float] = []
        truth_spatial = sub["truth_category"].isin(TRUE_SPATIAL).to_numpy(dtype=bool)

        for t in thresholds:
            called = pd.to_numeric(sub["q_T"], errors="coerce").to_numpy(
                dtype=float
            ) <= float(t)
            tp = int(np.sum(called & truth_spatial))
            fp = int(np.sum(called & (~truth_spatial)))
            tpr = float(tp / max(1, int(np.sum(truth_spatial))))
            fdr = float(fp / max(1, int(np.sum(called))))
            tpr_vals.append(tpr)
            fdr_vals.append(fdr)

        ax.plot(fdr_vals, tpr_vals, linewidth=1.4, label=variant)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("FDR")
    ax.set_ylabel("TPR")
    ax.set_title("Truth-aware TPR vs FDR curves (O1)")
    ax.legend(frameon=False, ncol=3)
    fig.suptitle(f"Truth-aware performance across variants\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    savefig(fig, plots / "truth_tpr_fdr_curves.png")
    plt.close(fig)


def _class_confusion_mat(base: pd.Series, var: pd.Series) -> np.ndarray:
    labels = SHAPE_CALLED
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for i, b in enumerate(labels):
        for j, v in enumerate(labels):
            mat[i, j] = float(np.sum((base == b) & (var == v)))
    return mat


def plot_class_confusion_key_variants(
    table: pd.DataFrame, outdir: Path, meta: str
) -> None:
    plots = ensure_dir(outdir / "plots")

    base = table.loc[
        (table["variant"] == "V0")
        & (table["origin_mode"] == "O1")
        & (table["called_localized"].astype(bool)),
        ["gene_id", "called_class"],
    ].rename(columns={"called_class": "class_base"})

    keys = ["V5", "V8"]
    fig, axes = plt.subplots(
        1, len(keys), figsize=(5.6 * len(keys), 4.5), constrained_layout=False
    )
    axes = np.atleast_1d(axes)

    for ax, v in zip(axes, keys):
        cur = table.loc[
            (table["variant"] == v)
            & (table["origin_mode"] == "O1")
            & (table["called_localized"].astype(bool)),
            ["gene_id", "called_class"],
        ].rename(columns={"called_class": "class_var"})

        m = base.merge(cur, on="gene_id", how="inner")
        if m.empty:
            mat = np.zeros((len(SHAPE_CALLED), len(SHAPE_CALLED)), dtype=float)
        else:
            mat = _class_confusion_mat(
                m["class_base"].astype(str), m["class_var"].astype(str)
            )
        im = ax.imshow(mat, aspect="auto", cmap="Blues")
        ax.set_xticks(np.arange(len(SHAPE_CALLED)))
        ax.set_xticklabels(SHAPE_CALLED, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(SHAPE_CALLED)))
        ax.set_yticklabels(SHAPE_CALLED)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center", fontsize=7)
        ax.set_title(f"V0,O1 vs {v},O1")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Class stability confusion (key variants)\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    savefig(fig, plots / "class_confusion_key_variants.png")
    plt.close(fig)


def plot_example_profiles(
    table: pd.DataFrame, outdir: Path, bins: int, meta: str
) -> None:
    plots = ensure_dir(outdir / "plots")

    wanted = []

    c1 = table.loc[
        (table["truth_category"] == "TRUE_UNIMODAL")
        & (table["variant"] == "V0")
        & (table["origin_mode"] == "O1")
    ].copy()
    if not c1.empty:
        wanted.append(int(c1.sort_values("q_T").iloc[0]["gene_id"]))

    c2 = table.loc[
        (table["truth_category"] == "TRUE_PATCHY")
        & (table["variant"] == "V0")
        & (table["origin_mode"] == "O1")
    ].copy()
    if not c2.empty:
        wanted.append(int(c2.sort_values("q_T").iloc[0]["gene_id"]))

    c3 = table.loc[
        (table["truth_category"] == "NULL")
        & (table["variant"] == "V0")
        & (table["origin_mode"] == "O1")
    ].copy()
    if not c3.empty:
        wanted.append(int(c3.sort_values("q_T").iloc[0]["gene_id"]))

    wanted = wanted[:3]
    if not wanted:
        return

    settings = [("V0", "O1"), ("V6", "O1"), ("V2", "O3")]
    theta = np.linspace(0.0, 2.0 * np.pi, int(bins), endpoint=False)

    fig, axes = plt.subplots(
        len(wanted),
        1,
        figsize=(6.2, 4.6 * len(wanted)),
        subplot_kw={"projection": "polar"},
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)

    for ax, gid in zip(axes, wanted):
        plotted = 0
        for v, o in settings:
            row = table.loc[
                (table["gene_id"] == int(gid))
                & (table["variant"] == v)
                & (table["origin_mode"] == o)
            ]
            if row.empty:
                continue
            rr = row.iloc[0]
            E = np.asarray(json.loads(rr["E_obs_json"]), dtype=float)
            if E.size != int(bins):
                continue
            plot_rsp_polar(
                theta,
                E,
                ax=ax,
                linewidth=1.2,
                label=f"{v},{o} (q={_fmt(float(rr['q_T']))})",
                title=None,
            )
            plotted += 1
        info = table.loc[
            (table["gene_id"] == int(gid))
            & (table["variant"] == "V0")
            & (table["origin_mode"] == "O1")
        ].iloc[0]
        ax.set_title(
            f"gene={gid}, truth={info['truth_category']}, beta={_fmt(float(info['beta']))}"
        )
        if plotted > 0:
            ax.legend(frameon=False, loc="upper right")

    fig.suptitle(f"Example profiles across variants\n{meta}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    savefig(fig, plots / "example_profiles_across_variants.png")
    plt.close(fig)


def run_validations(table: pd.DataFrame, conc: pd.DataFrame, outdir: Path) -> list[str]:
    results = ensure_dir(outdir / "results")
    lines: list[str] = []
    warns: list[str] = []

    def _get(v: str, o: str, col: str) -> float:
        row = conc.loc[(conc["variant"] == v) & (conc["origin_mode"] == o), col]
        return float(row.iloc[0]) if not row.empty else float("nan")

    v1 = _get("V1", "O1", "spearman_Z_vs_baseline")
    j1 = _get("V1", "O1", "top100_jaccard")
    lines.append(f"V1,O1 spearman={v1:.3f}, top100_jaccard={j1:.3f}")
    if not np.isfinite(v1) or v1 < 0.95 or not np.isfinite(j1) or j1 < 0.8:
        warns.append("Benign rotation invariance check failed (V1,O1).")

    v3 = _get("V3", "O1", "spearman_Z_vs_baseline")
    j3 = _get("V3", "O1", "top100_jaccard")
    lines.append(f"V3,O1 spearman={v3:.3f}, top100_jaccard={j3:.3f}")
    if not np.isfinite(v3) or v3 < 0.95 or not np.isfinite(j3) or j3 < 0.8:
        warns.append("Benign scaling invariance check failed (V3,O1).")

    t_o1 = _get("V2", "O1", "spearman_Z_vs_baseline")
    t_o2 = _get("V2", "O2", "spearman_Z_vs_baseline")
    t_o3 = _get("V2", "O3", "spearman_Z_vs_baseline")
    lines.append(f"V2 spearman: O1={t_o1:.3f}, O2={t_o2:.3f}, O3={t_o3:.3f}")
    if np.isfinite(t_o1) and np.isfinite(t_o2) and (t_o1 < 0.8 or t_o2 < 0.8):
        warns.append("Translation with correct/data-driven origin has low concordance.")
    if np.isfinite(t_o1) and np.isfinite(t_o3) and not (t_o3 < t_o1 - 0.1):
        warns.append(
            "Wrong-origin translation did not degrade concordance as expected."
        )

    v5 = _get("V5", "O1", "spearman_Z_vs_baseline")
    lines.append(f"V5,O1 spearman={v5:.3f}")
    if not np.isfinite(v5) or v5 < 0.8:
        warns.append("Mild warp concordance below expectation (V5,O1 < 0.8).")

    for v in ["V6", "V8"]:
        s = _get(v, "O1", "spearman_Z_vs_baseline")
        fdr = _get(v, "O1", "truth_FDR")
        lines.append(f"Stress variant {v},O1: spearman={s:.3f}, truth_FDR={fdr:.3f}")

    for v, o in [("V0", "O1"), ("V1", "O1"), ("V3", "O1")]:
        sub = table.loc[
            (table["variant"] == v)
            & (table["origin_mode"] == o)
            & (table["truth_category"] == "NULL")
            & (~table["underpowered"].astype(bool))
        ].copy()
        p = pd.to_numeric(sub["p_T"], errors="coerce").to_numpy(dtype=float)
        p = p[np.isfinite(p)]
        if p.size >= 100:
            ks = kstest(p, "uniform")
            ksp = float(ks.pvalue)
            lines.append(f"NULL KS p for {v},{o}: {ksp:.4g} (n={p.size})")
            if ksp <= 1e-3 and v in {"V1", "V3"}:
                warns.append(
                    f"Benign null calibration failed at {v},{o} (KS p={ksp:.3g})."
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

    run_id = timestamped_run_id(prefix="expI")

    min_perm_eff = int(args.min_perm)
    if int(args.n_perm) < int(min_perm_eff):
        warnings.warn(
            f"n_perm ({int(args.n_perm)}) < min_perm ({int(min_perm_eff)}); lowering gate to n_perm.",
            RuntimeWarning,
            stacklevel=2,
        )
        min_perm_eff = int(args.n_perm)

    mix = np.asarray(
        [
            float(args.mixture_null),
            float(args.mixture_unimodal),
            float(args.mixture_bimodal),
            float(args.mixture_patchy),
        ],
        dtype=float,
    )
    if np.any(mix < 0) or float(np.sum(mix)) <= 0:
        raise ValueError("Invalid mixture proportions.")
    mix = mix / float(np.sum(mix))

    variants = [str(v) for v in args.variants]
    origins = [str(o) for o in args.origins]

    pairs: list[tuple[str, str]] = []
    if bool(args.fast_mode):
        for v in variants:
            for o in origins:
                if o in {"O1", "O2"}:
                    pairs.append((v, o))
                elif o == "O3" and v in {"V0", "V2", "V4"}:
                    pairs.append((v, o))
    else:
        for v in variants:
            for o in origins:
                pairs.append((v, o))

    if ("V0", "O1") not in pairs:
        pairs.append(("V0", "O1"))
    if "V2" in variants and "O3" in origins and ("V2", "O3") not in pairs:
        pairs.append(("V2", "O3"))

    cfg = {
        "experiment": "expI_embedding_robustness",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit_hash(cwd=REPO_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "master_seed": int(args.master_seed),
        "N": int(args.N),
        "D": int(args.D),
        "G": int(args.G),
        "bins": int(args.bins),
        "mode": str(args.mode),
        "w": int(args.w),
        "n_perm": int(args.n_perm),
        "variants": variants,
        "origins": origins,
        "fast_mode": bool(args.fast_mode),
        "evaluated_pairs": [f"{v}:{o}" for v, o in pairs],
        "sigma_eta": float(args.sigma_eta),
        "mixture": {
            "NULL": float(mix[0]),
            "TRUE_UNIMODAL": float(mix[1]),
            "TRUE_BIMODAL": float(mix[2]),
            "TRUE_PATCHY": float(mix[3]),
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
            "Running ExpI with "
            f"N={int(args.N)}, D={int(args.D)}, G={int(args.G)}, pairs={len(pairs)}, "
            f"n_perm={int(args.n_perm)}, mode={str(args.mode)}, w={int(args.w)}"
        ),
        flush=True,
    )

    seed_base = stable_seed(int(args.master_seed), "expI", "base")
    base = _generate_base_context(
        seed=int(seed_base),
        N=int(args.N),
        D=int(args.D),
        sigma_eta=float(args.sigma_eta),
    )

    truths: list[GeneTruth] = []
    for gid in range(int(args.G)):
        seed_gene = stable_seed(int(args.master_seed), "expI", "gene", int(gid))
        truths.append(
            _simulate_gene_truth(
                base=base, gene_id=int(gid), seed_gene=int(seed_gene), mixture=mix
            )
        )

    var_seed = stable_seed(int(args.master_seed), "expI", "variants")
    variants_data = _transform_variants(base.X_true, int(var_seed))

    rows: list[dict[str, Any]] = []
    total = len(pairs) * int(args.G)
    done = 0
    t0 = time.time()

    power_cfg = {
        "p_min": float(args.p_min),
        "min_fg_total": int(args.min_fg_total),
        "min_fg_per_donor": int(args.min_fg_per_donor),
        "min_bg_per_donor": int(args.min_bg_per_donor),
        "d_eff_min": int(args.d_eff_min),
    }

    for variant_name, origin_mode in pairs:
        if variant_name not in variants_data:
            continue
        vd = variants_data[variant_name]
        origin = _origin_for_mode(vd, origin_mode)
        theta = np.mod(
            np.arctan2(vd.X[:, 1] - float(origin[1]), vd.X[:, 0] - float(origin[0])),
            2.0 * np.pi,
        )
        bin_id, bin_counts = _compute_bin_cache(theta, int(args.bins))

        print(f"Setting variant={variant_name}, origin={origin_mode}", flush=True)

        for truth in truths:
            # Keep permutation streams matched across embedding settings for each gene
            # so cross-variant concordance reflects embedding effects, not Monte Carlo jitter.
            seed_perm = stable_seed(
                int(args.master_seed), "expI", "perm", int(truth.gene_id)
            )
            score = _score_gene_variant(
                f=truth.f,
                theta=theta,
                donor_ids=base.donor_ids,
                bin_id=bin_id,
                bin_counts=bin_counts,
                bins=int(args.bins),
                mode=str(args.mode),
                w=int(args.w),
                n_perm=int(args.n_perm),
                seed_perm=int(seed_perm),
                min_perm_eff=int(min_perm_eff),
                power_cfg=power_cfg,
            )

            rows.append(
                {
                    "run_id": run_id,
                    "seed": int(args.master_seed),
                    "gene_id": int(truth.gene_id),
                    "truth_category": truth.truth_category,
                    "pi_target": float(truth.pi_target),
                    "beta": float(truth.beta),
                    "dropout_noise": float(truth.dropout_noise),
                    "variant": str(variant_name),
                    "origin_mode": str(origin_mode),
                    "bins_B": int(args.bins),
                    "mode": str(args.mode),
                    "w": int(args.w),
                    "n_perm": int(args.n_perm),
                    **score,
                }
            )

            done += 1
            if done % int(args.progress_every) == 0 or done == int(total):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else float("nan")
                print(f"  progress: {done}/{total} ({rate:.2f} genes/s)", flush=True)

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No rows produced.")

    table = _apply_bh(table, q_thresh=float(args.q_thresh))
    conc = build_concordance_summary(table)

    atomic_write_csv(results / "gene_variant_table.csv", table)
    atomic_write_csv(results / "concordance_summary.csv", conc)

    meta = f"N={int(args.N)}, D={int(args.D)}, B={int(args.bins)}, n_perm={int(args.n_perm)}, mode={str(args.mode)}, w={int(args.w)}"

    _set_plot_style()
    plot_spearman_heatmap(conc, outdir, meta)
    plot_topk_curves(conc, outdir, meta)
    plot_discovery_jaccard_heatmap(conc, outdir, meta)
    plot_translate_origin_failure(table, outdir, meta)
    plot_truth_tpr_fdr_curves(table, outdir, meta)
    plot_class_confusion_key_variants(table, outdir, meta)
    plot_example_profiles(table, outdir, int(args.bins), meta)
    if truths:
        fig_emb, ax_emb = plt.subplots(figsize=(5.2, 4.4), constrained_layout=False)
        plot_embedding_with_foreground(
            base.X_true,
            truths[0].f,
            ax=ax_emb,
            title="Representative embedding foreground (ExpI)",
            s=5.0,
            alpha_bg=0.30,
            alpha_fg=0.75,
        )
        savefig(fig_emb, ensure_dir(outdir / "plots") / "embedding_example.png")
        plt.close(fig_emb)
    ref = table.loc[(table["variant"] == "V0") & (table["origin_mode"] == "O1")].copy()
    if not ref.empty:
        row0 = ref.sort_values(["q_T", "gene_id"]).iloc[0]
        E0 = np.asarray(json.loads(row0["E_obs_json"]), dtype=float)
        theta0 = np.linspace(0.0, 2.0 * np.pi, int(args.bins), endpoint=False)
        fig_pol, ax_pol = plt.subplots(
            figsize=(5.2, 4.8),
            subplot_kw={"projection": "polar"},
            constrained_layout=False,
        )
        plot_rsp_polar(
            theta0,
            E0,
            ax=ax_pol,
            title=f"Representative RSP profile (ExpI)\ngene={int(row0['gene_id'])}, truth={row0['truth_category']}",
        )
        savefig(fig_pol, ensure_dir(outdir / "plots") / "polar_rsp_example.png")
        plt.close(fig_pol)

    warns = run_validations(table, conc, outdir)

    elapsed = time.time() - t0
    print(
        f"Completed ExpI in {elapsed/60.0:.2f} min. rows={table.shape[0]}, settings={len(pairs)}, warnings={len(warns)}",
        flush=True,
    )
    print(f"Outputs written to: {outdir}", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Simulation Experiment I: embedding robustness."
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="experiments/simulations/expI_embedding_robustness",
    )
    p.add_argument("--master_seed", type=int, default=123)
    p.add_argument("--N", type=int, default=20000)
    p.add_argument("--D", type=int, default=10)
    p.add_argument("--G", type=int, default=2000)

    p.add_argument("--bins", type=int, default=36)
    p.add_argument("--mode", type=str, choices=["raw", "smoothed"], default="raw")
    p.add_argument("--w", type=int, default=1)
    p.add_argument("--n_perm", type=int, default=300)
    p.add_argument("--q_thresh", type=float, default=0.05)

    p.add_argument("--variants", type=str, nargs="+", default=DEFAULT_VARIANTS)
    p.add_argument("--origins", type=str, nargs="+", default=DEFAULT_ORIGINS)
    p.add_argument("--fast_mode", action="store_true")

    p.add_argument("--sigma_eta", type=float, default=0.4)

    p.add_argument("--mixture_null", type=float, default=0.8)
    p.add_argument("--mixture_unimodal", type=float, default=0.1)
    p.add_argument("--mixture_bimodal", type=float, default=0.05)
    p.add_argument("--mixture_patchy", type=float, default=0.05)

    p.add_argument("--p_min", type=float, default=0.005)
    p.add_argument("--min_fg_total", type=int, default=50)
    p.add_argument("--min_fg_per_donor", type=int, default=10)
    p.add_argument("--min_bg_per_donor", type=int, default=10)
    p.add_argument("--d_eff_min", type=int, default=2)
    p.add_argument("--min_perm", type=int, default=200)

    p.add_argument("--progress_every", type=int, default=200)
    add_common_args(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    warnings.simplefilter("default", RuntimeWarning)
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if bool(getattr(args, "dry_run", False)):
        print("dry_run=True: skipping execution for legacy runner.", flush=True)
        return 0
    if bool(getattr(args, "test_mode", False)):
        args = apply_testmode_overrides(args, exp_name="expI_embedding_robustness")

    if int(args.N) <= 0 or int(args.D) <= 0 or int(args.G) <= 0:
        raise ValueError("N, D, G must be positive.")
    if int(args.bins) <= 0 or int(args.n_perm) <= 0:
        raise ValueError("bins and n_perm must be positive.")
    if str(args.mode) == "smoothed" and (int(args.w) < 1 or int(args.w) % 2 == 0):
        raise ValueError("smoothed mode requires odd w >= 1.")

    run_ctx = prepare_legacy_run(args, "expI_embedding_robustness", __file__)
    run_experiment(args)
    finalize_legacy_run(run_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
