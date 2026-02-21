#!/usr/bin/env python3
"""Simulation Experiment G: donor replication and cross-donor generalization.

This experiment evaluates whether pooled BioRSP localization calls are replicable
across donors via direction/shape concordance metrics.
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
from scipy.signal import find_peaks
from scipy.stats import spearmanr

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-biorsp-expG")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache-biorsp-expG")
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from biorsp.permutation import perm_null_T
from biorsp.power import evaluate_underpowered
from biorsp.rsp import compute_rsp_profile_from_boolean
from experiments.simulations._shared.cli import add_common_args
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

DEFAULT_D_GRID = [5, 10, 15]
DEFAULT_SIGMA_GRID = [0.0, 0.4]
DEFAULT_PI_GRID = [0.05, 0.2, 0.6]
DEFAULT_BETA_GRID = [0.0, 0.5, 1.0, 1.25]
DEFAULT_REGIMES = ["R_true", "R_null", "R_donor_specific"]


@dataclass(frozen=True)
class DonorGeometryContext:
    D: int
    N_total: int
    donor_sizes: np.ndarray
    theta_by_donor: list[np.ndarray]
    X_by_donor: list[np.ndarray]
    bin_id_by_donor: list[np.ndarray]
    bin_counts_by_donor: list[np.ndarray]
    pooled_theta: np.ndarray
    pooled_X: np.ndarray
    pooled_bin_id: np.ndarray
    pooled_bin_counts: np.ndarray
    pooled_donor_ids: np.ndarray


def _fmt(x: float | int) -> str:
    return f"{float(x):.3f}".rstrip("0").rstrip(".")


def _tok(x: float | int | str) -> str:
    return str(x).replace(".", "p")


def _clip_pi(pi: float) -> float:
    return float(np.clip(float(pi), 1e-9, 1.0 - 1e-9))


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


def _theta_centers(bins: int) -> np.ndarray:
    edges = np.linspace(0.0, 2.0 * np.pi, int(bins) + 1, endpoint=True)
    return (edges[:-1] + edges[1:]) / 2.0


def _theta_hat_from_profile(E: np.ndarray, bins: int) -> float:
    arr = np.asarray(E, dtype=float).ravel()
    if arr.size != int(bins):
        return float("nan")
    idx = int(np.argmax(arr))
    centers = _theta_centers(int(bins))
    return float(np.mod(centers[idx], 2.0 * np.pi))


def _circular_distance(a: float, b: float) -> float:
    if not (np.isfinite(float(a)) and np.isfinite(float(b))):
        return float("nan")
    d = abs(float(a) - float(b)) % (2.0 * np.pi)
    return float(min(d, 2.0 * np.pi - d))


def _count_peaks(E: np.ndarray, min_prom_frac: float = 0.15) -> int:
    arr = np.asarray(E, dtype=float).ravel()
    if arr.size < 3:
        return 0
    baseline = float(np.max(arr) - np.min(arr))
    if baseline <= 1e-12:
        return 0
    prom = max(1e-6, float(min_prom_frac) * baseline)

    # Circular treatment by tiling; evaluate peaks in the center copy only.
    tiled = np.concatenate([arr, arr, arr])
    peaks, props = find_peaks(tiled, prominence=prom)
    if peaks.size == 0:
        return 0
    n = arr.size
    center_mask = (peaks >= n) & (peaks < 2 * n)
    peaks_center = peaks[center_mask] - n
    if peaks_center.size == 0:
        return 0

    # Count only positive peaks to align with localization peak interpretation.
    vals = arr[peaks_center]
    return int(np.sum(vals > 0.0))


def _even_donor_sizes(N_total: int, D: int) -> np.ndarray:
    n = int(N_total)
    d = int(D)
    base = n // d
    rem = n % d
    sizes = np.full(d, base, dtype=int)
    sizes[:rem] += 1
    return sizes


def build_donor_geometry_context(
    *,
    N_total: int,
    D: int,
    bins: int,
    seed: int,
) -> DonorGeometryContext:
    rng = rng_from_seed(int(seed))
    sizes = _even_donor_sizes(int(N_total), int(D))

    theta_by_donor: list[np.ndarray] = []
    X_by_donor: list[np.ndarray] = []
    bin_id_by_donor: list[np.ndarray] = []
    bin_counts_by_donor: list[np.ndarray] = []

    pooled_theta_parts: list[np.ndarray] = []
    pooled_x_parts: list[np.ndarray] = []
    pooled_donor_parts: list[np.ndarray] = []

    for d_idx, n_d in enumerate(sizes.tolist()):
        X = rng.normal(loc=0.0, scale=1.0, size=(int(n_d), 2)).astype(float)
        theta = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2.0 * np.pi).astype(float)
        b_id, b_counts = _compute_bin_cache(theta, int(bins))

        X_by_donor.append(X)
        theta_by_donor.append(theta)
        bin_id_by_donor.append(b_id)
        bin_counts_by_donor.append(b_counts)

        pooled_x_parts.append(X)
        pooled_theta_parts.append(theta)
        pooled_donor_parts.append(np.full(int(n_d), int(d_idx), dtype=np.int16))

    pooled_X = np.vstack(pooled_x_parts).astype(float)
    pooled_theta = np.concatenate(pooled_theta_parts).astype(float)
    pooled_donor_ids = np.concatenate(pooled_donor_parts).astype(np.int16)
    pooled_bin_id, pooled_bin_counts = _compute_bin_cache(pooled_theta, int(bins))

    return DonorGeometryContext(
        D=int(D),
        N_total=int(N_total),
        donor_sizes=sizes,
        theta_by_donor=theta_by_donor,
        X_by_donor=X_by_donor,
        bin_id_by_donor=bin_id_by_donor,
        bin_counts_by_donor=bin_counts_by_donor,
        pooled_theta=pooled_theta,
        pooled_X=pooled_X,
        pooled_bin_id=pooled_bin_id,
        pooled_bin_counts=pooled_bin_counts,
        pooled_donor_ids=pooled_donor_ids,
    )


def simulate_gene_detections(
    *,
    context: DonorGeometryContext,
    regime: str,
    pi_target: float,
    beta: float,
    sigma_eta: float,
    seed_gene: int,
) -> tuple[list[np.ndarray], np.ndarray, float, np.ndarray, np.ndarray]:
    """Simulate boolean detections per donor and pooled.

    Returns:
      f_by_donor, f_pool, theta0_global, theta0_by_donor, eta_d
    """
    rng = rng_from_seed(int(seed_gene))

    eta_d = rng.normal(loc=0.0, scale=float(sigma_eta), size=int(context.D)).astype(
        float
    )
    theta0_global = float(rng.uniform(0.0, 2.0 * np.pi))
    theta0_donor = rng.uniform(0.0, 2.0 * np.pi, size=int(context.D)).astype(float)

    alpha = float(
        math.log(_clip_pi(float(pi_target)) / (1.0 - _clip_pi(float(pi_target))))
    )

    f_by_donor: list[np.ndarray] = []
    for d_idx in range(int(context.D)):
        theta = context.theta_by_donor[d_idx]
        eta = float(eta_d[d_idx])

        if regime == "R_true":
            logits = alpha + float(beta) * np.cos(theta - float(theta0_global)) + eta
        elif regime == "R_null":
            logits = alpha + eta
        elif regime == "R_donor_specific":
            logits = (
                alpha + float(beta) * np.cos(theta - float(theta0_donor[d_idx])) + eta
            )
        else:
            raise ValueError(f"Unsupported regime '{regime}'.")

        p = 1.0 / (1.0 + np.exp(-logits))
        f = (rng.random(theta.size) < p).astype(bool)
        f_by_donor.append(f)

    f_pool = np.concatenate(f_by_donor).astype(bool)
    return f_by_donor, f_pool, theta0_global, theta0_donor, eta_d


def score_single_donor(
    *,
    f_d: np.ndarray,
    theta_d: np.ndarray,
    bin_id_d: np.ndarray,
    bin_counts_d: np.ndarray,
    bins: int,
    n_perm_donor: int,
    seed_perm: int,
    donor_prev_floor: float,
    donor_min_fg: int,
) -> dict[str, Any]:
    f = np.asarray(f_d, dtype=bool).ravel()
    n = int(f.size)
    n_fg = int(np.sum(f))
    prev = float(np.mean(f)) if n > 0 else float("nan")

    under = bool(
        (prev < float(donor_prev_floor))
        or (n_fg < int(donor_min_fg))
        or (n_fg == 0)
        or (n_fg == n)
    )

    if under:
        return {
            "N_d": int(n),
            "prev_d": float(prev),
            "n_fg_d": int(n_fg),
            "underpowered_d": True,
            "T_d": float("nan"),
            "p_d": float("nan"),
            "theta_hat_d": float("nan"),
            "K_hat_d": float("nan"),
            "E_obs_d": np.full(int(bins), np.nan, dtype=float),
        }

    perm = perm_null_T(
        f=f,
        angles=theta_d,
        donor_ids=None,
        n_bins=int(bins),
        n_perm=int(n_perm_donor),
        seed=int(seed_perm),
        mode="raw",
        smooth_w=1,
        donor_stratified=False,
        return_null_T=True,
        return_obs_profile=True,
        bin_id=bin_id_d,
        bin_counts_total=bin_counts_d,
    )

    E_obs = np.asarray(perm["E_phi_obs"], dtype=float)
    theta_hat = _theta_hat_from_profile(E_obs, int(bins))
    K_hat = int(_count_peaks(E_obs))

    return {
        "N_d": int(n),
        "prev_d": float(prev),
        "n_fg_d": int(n_fg),
        "underpowered_d": False,
        "T_d": float(perm["T_obs"]),
        "p_d": float(perm["p_T"]),
        "theta_hat_d": float(theta_hat),
        "K_hat_d": float(K_hat),
        "E_obs_d": E_obs,
    }


def score_pooled(
    *,
    f_pool: np.ndarray,
    context: DonorGeometryContext,
    bins: int,
    n_perm_pool: int,
    seed_perm: int,
    p_min: float,
    min_fg_total: int,
    min_fg_per_donor: int,
    min_bg_per_donor: int,
    d_eff_min: int,
    min_perm: int,
) -> dict[str, Any]:
    f = np.asarray(f_pool, dtype=bool).ravel()

    power = evaluate_underpowered(
        donor_ids=context.pooled_donor_ids,
        f=f,
        n_perm=int(n_perm_pool),
        p_min=float(p_min),
        min_fg_total=int(min_fg_total),
        min_fg_per_donor=int(min_fg_per_donor),
        min_bg_per_donor=int(min_bg_per_donor),
        d_eff_min=int(d_eff_min),
        min_perm=int(min_perm),
    )

    pooled_prev = float(np.mean(f))
    pooled_n_fg = int(np.sum(f))
    under = bool(power["underpowered"])

    if under or pooled_n_fg == 0 or pooled_n_fg == int(f.size):
        return {
            "pooled_prev": float(pooled_prev),
            "pooled_n_fg": int(pooled_n_fg),
            "pooled_D_eff": int(power["D_eff"]),
            "pooled_underpowered": True,
            "T_pool": float("nan"),
            "p_pool": float("nan"),
            "theta_hat_pool": float("nan"),
            "K_hat_pool": float("nan"),
            "E_obs_pool": np.full(int(bins), np.nan, dtype=float),
        }

    perm = perm_null_T(
        f=f,
        angles=context.pooled_theta,
        donor_ids=context.pooled_donor_ids,
        n_bins=int(bins),
        n_perm=int(n_perm_pool),
        seed=int(seed_perm),
        mode="raw",
        smooth_w=1,
        donor_stratified=True,
        return_null_T=True,
        return_obs_profile=True,
        bin_id=context.pooled_bin_id,
        bin_counts_total=context.pooled_bin_counts,
    )

    E_obs = np.asarray(perm["E_phi_obs"], dtype=float)
    theta_hat = _theta_hat_from_profile(E_obs, int(bins))
    K_hat = int(_count_peaks(E_obs))

    return {
        "pooled_prev": float(pooled_prev),
        "pooled_n_fg": int(pooled_n_fg),
        "pooled_D_eff": int(power["D_eff"]),
        "pooled_underpowered": False,
        "T_pool": float(perm["T_obs"]),
        "p_pool": float(perm["p_T"]),
        "theta_hat_pool": float(theta_hat),
        "K_hat_pool": float(K_hat),
        "E_obs_pool": E_obs,
    }


def _mode_int(values: np.ndarray) -> tuple[int, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1, float("nan")
    ints = np.round(arr).astype(int)
    uniq, counts = np.unique(ints, return_counts=True)
    i = int(np.argmax(counts))
    k_mode = int(uniq[i])
    frac = float(counts[i] / max(1, ints.size))
    return k_mode, frac


def assign_replication_label(
    *,
    pooled_underpowered: bool,
    p_pool: float,
    D: int,
    D_eff_info: int,
    median_ang_err: float,
    frac_within_30deg: float,
    K_consistency: float,
    p_thresh: float,
    min_info_abs: int,
    min_info_frac: float,
    ang_thresh_rad: float,
    frac_within_thresh: float,
    use_k_consistency: bool,
    k_consistency_thresh: float,
) -> str:
    if bool(pooled_underpowered):
        return "UNDERPOWERED"
    if not np.isfinite(float(p_pool)) or float(p_pool) > float(p_thresh):
        return "NOT_LOCALIZED"

    info_min = int(max(int(min_info_abs), math.ceil(float(min_info_frac) * int(D))))
    cond_info = int(D_eff_info) >= int(info_min)
    cond_dir = (
        np.isfinite(float(median_ang_err))
        and np.isfinite(float(frac_within_30deg))
        and (float(median_ang_err) <= float(ang_thresh_rad))
        and (float(frac_within_30deg) >= float(frac_within_thresh))
    )
    cond_k = (not bool(use_k_consistency)) or (
        np.isfinite(float(K_consistency))
        and (float(K_consistency) >= float(k_consistency_thresh))
    )

    if cond_info and cond_dir and cond_k:
        return "REPLICABLE_LOCALIZED"
    return "LOCALIZED_NONREPLICABLE"


def _prop_ci(k: int, n: int) -> tuple[float, float, float]:
    if int(n) <= 0:
        return float("nan"), float("nan"), float("nan")
    p = float(k) / float(n)
    lo, hi = wilson_ci(int(k), int(n))
    return p, float(lo), float(hi)


def summarize_gene_metrics(gene_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grp_cols = ["regime", "D", "sigma_eta", "pi_target", "beta"]

    for keys, grp in gene_metrics.groupby(grp_cols, sort=True):
        n = int(grp.shape[0])
        under = grp["pooled_underpowered"].astype(bool)
        n_under = int(np.sum(under))
        n_non = int(n - n_under)

        non = grp.loc[~under].copy()
        pooled_loc = non["p_pool"].le(0.05)
        repl = non["final_replication_label"].eq("REPLICABLE_LOCALIZED")
        nonrepl = non["final_replication_label"].eq("LOCALIZED_NONREPLICABLE")

        frac_under, frac_under_lo, frac_under_hi = _prop_ci(n_under, n)
        frac_pool, frac_pool_lo, frac_pool_hi = _prop_ci(int(np.sum(pooled_loc)), n_non)
        frac_repl, frac_repl_lo, frac_repl_hi = _prop_ci(int(np.sum(repl)), n_non)

        recall_repl = float("nan")
        recall_repl_lo = float("nan")
        recall_repl_hi = float("nan")
        false_repl = float("nan")
        false_repl_lo = float("nan")
        false_repl_hi = float("nan")
        nonrepl_rate = float("nan")
        nonrepl_rate_lo = float("nan")
        nonrepl_rate_hi = float("nan")

        regime = str(keys[0])
        if regime == "R_true":
            recall_repl, recall_repl_lo, recall_repl_hi = _prop_ci(
                int(np.sum(repl)), n_non
            )
        if regime == "R_null":
            false_repl, false_repl_lo, false_repl_hi = _prop_ci(
                int(np.sum(repl)), n_non
            )
        if regime == "R_donor_specific":
            nonrepl_rate, nonrepl_rate_lo, nonrepl_rate_hi = _prop_ci(
                int(np.sum(nonrepl)), n_non
            )

        rows.append(
            {
                "regime": regime,
                "D": int(keys[1]),
                "sigma_eta": float(keys[2]),
                "pi_target": float(keys[3]),
                "beta": float(keys[4]),
                "n_genes": int(n),
                "n_non_underpowered": int(n_non),
                "frac_underpowered": float(frac_under),
                "frac_underpowered_ci_low": float(frac_under_lo),
                "frac_underpowered_ci_high": float(frac_under_hi),
                "frac_pooled_localized": float(frac_pool),
                "frac_pooled_localized_ci_low": float(frac_pool_lo),
                "frac_pooled_localized_ci_high": float(frac_pool_hi),
                "frac_replicable_localized": float(frac_repl),
                "frac_replicable_localized_ci_low": float(frac_repl_lo),
                "frac_replicable_localized_ci_high": float(frac_repl_hi),
                "recall_replicable": float(recall_repl),
                "recall_replicable_ci_low": float(recall_repl_lo),
                "recall_replicable_ci_high": float(recall_repl_hi),
                "false_replicable_rate": float(false_repl),
                "false_replicable_rate_ci_low": float(false_repl_lo),
                "false_replicable_rate_ci_high": float(false_repl_hi),
                "nonreplicable_rate": float(nonrepl_rate),
                "nonreplicable_rate_ci_low": float(nonrepl_rate_lo),
                "nonreplicable_rate_ci_high": float(nonrepl_rate_hi),
            }
        )

    return pd.DataFrame(rows)


def plot_replicable_rate_curves(
    *,
    summary: pd.DataFrame,
    outdir: Path,
    N_total: int,
    n_perm_pool: int,
    n_perm_donor: int,
    bins: int,
    ang_thresh_deg: float,
    frac_within_thresh: float,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    for D in sorted(summary["D"].astype(int).unique().tolist()):
        for sigma in sorted(summary["sigma_eta"].astype(float).unique().tolist()):
            sub = summary.loc[
                (summary["D"] == int(D))
                & np.isclose(summary["sigma_eta"], float(sigma))
            ].copy()
            if sub.empty:
                continue

            # Average over prevalence for a compact operating curve.
            agg = (
                sub.groupby(["regime", "beta"], sort=True)
                .agg(
                    frac_repl=("frac_replicable_localized", "mean"),
                    n_non=("n_non_underpowered", "sum"),
                )
                .reset_index()
            )

            fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=False)
            for regime in ["R_true", "R_null", "R_donor_specific"]:
                g = agg.loc[agg["regime"] == regime].sort_values("beta")
                if g.empty:
                    continue
                ax.plot(
                    g["beta"], g["frac_repl"], marker="o", linewidth=1.7, label=regime
                )

            ax.set_ylim(0.0, 1.02)
            ax.set_xlabel("beta")
            ax.set_ylabel("Frac REPLICABLE_LOCALIZED")
            ax.set_title(
                (
                    f"Replicable rate vs beta | D={D}, sigma_eta={_fmt(sigma)}\n"
                    f"N_total={N_total}, n_perm_pool={n_perm_pool}, n_perm_donor={n_perm_donor}, B={bins}, "
                    f"ang<={_fmt(ang_thresh_deg)}°, frac>={_fmt(frac_within_thresh)}"
                )
            )
            ax.legend(frameon=False)
            fig.tight_layout()
            savefig(
                fig,
                plots_dir / f"replicable_rate_vs_beta_D{int(D)}_sigma{_tok(sigma)}.png",
            )
            plt.close(fig)


def plot_angle_error_hist(
    *,
    donor_metrics: pd.DataFrame,
    outdir: Path,
    N_total: int,
    n_perm_pool: int,
    n_perm_donor: int,
    bins: int,
    pi_ref: float,
    beta_ref: float,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    sub = donor_metrics.loc[
        np.isclose(donor_metrics["pi_target"], float(pi_ref))
        & np.isclose(donor_metrics["beta"], float(beta_ref))
        & donor_metrics["informative_d"].astype(bool)
        & np.isfinite(pd.to_numeric(donor_metrics["ang_err_d"], errors="coerce"))
    ].copy()
    if sub.empty:
        return

    D_vals = sorted(sub["D"].astype(int).unique().tolist())
    fig, axes = plt.subplots(
        1, len(D_vals), figsize=(4.6 * len(D_vals), 3.8), constrained_layout=False
    )
    axes = np.atleast_1d(axes)

    color = {"R_true": "#1f77b4", "R_null": "#7f7f7f", "R_donor_specific": "#d62728"}
    bins_hist = np.linspace(0.0, math.pi, 25)
    for ax, D in zip(axes, D_vals):
        dsub = sub.loc[sub["D"] == int(D)]
        for regime in ["R_true", "R_null", "R_donor_specific"]:
            r = dsub.loc[dsub["regime"] == regime]
            vals = pd.to_numeric(r["ang_err_d"], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            ax.hist(
                vals,
                bins=bins_hist,
                alpha=0.38,
                density=True,
                label=regime,
                color=color[regime],
            )
        ax.set_xlim(0.0, math.pi)
        ax.set_xlabel("Angular error (rad)")
        ax.set_ylabel("Density")
        ax.set_title(f"D={D}")

    fig.suptitle(
        (
            f"Angle-error distributions | pi={_fmt(pi_ref)}, beta={_fmt(beta_ref)}\n"
            f"N_total={N_total}, n_perm_pool={n_perm_pool}, n_perm_donor={n_perm_donor}, B={bins}"
        )
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    savefig(
        fig,
        plots_dir
        / f"angle_error_hist_pi{float(pi_ref):.1f}_beta{float(beta_ref):.1f}.png",
    )
    plt.close(fig)


def plot_pooled_vs_concordance(
    *,
    gene_metrics: pd.DataFrame,
    outdir: Path,
    N_total: int,
    n_perm_pool: int,
    n_perm_donor: int,
    bins: int,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    sub = gene_metrics.loc[~gene_metrics["pooled_underpowered"].astype(bool)].copy()
    if sub.empty:
        return

    x = -np.log10(
        np.clip(
            pd.to_numeric(sub["p_pool"], errors="coerce").to_numpy(dtype=float),
            1e-12,
            1.0,
        )
    )
    y = pd.to_numeric(sub["frac_within_30deg"], errors="coerce").to_numpy(dtype=float)
    c = sub["regime"].astype(str).to_numpy()

    cmap = {"R_true": "#1f77b4", "R_null": "#7f7f7f", "R_donor_specific": "#d62728"}

    fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=False)
    for regime in ["R_true", "R_null", "R_donor_specific"]:
        mask = c == regime
        if not np.any(mask):
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=14,
            alpha=0.35,
            color=cmap[regime],
            label=regime,
            linewidths=0.0,
        )
    ax.axvline(-math.log10(0.05), linestyle="--", color="#333333", linewidth=0.9)
    ax.axhline(0.6, linestyle="--", color="#333333", linewidth=0.9)
    ax.set_xlabel("-log10(p_pool)")
    ax.set_ylabel("frac_within_30deg")
    ax.set_title(
        (
            "Pooled significance vs donor concordance\n"
            f"N_total={N_total}, n_perm_pool={n_perm_pool}, n_perm_donor={n_perm_donor}, B={bins}"
        )
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, plots_dir / "pooled_significance_vs_concordance.png")
    plt.close(fig)


def plot_replication_pr(
    *,
    gene_metrics: pd.DataFrame,
    outdir: Path,
    N_total: int,
    n_perm_pool: int,
    n_perm_donor: int,
    bins: int,
    min_info_abs: int,
    min_info_frac: float,
    ang_thresh_rad: float,
    frac_within_thresh: float,
    use_k_consistency: bool,
    k_consistency_thresh: float,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    sub = gene_metrics.loc[~gene_metrics["pooled_underpowered"].astype(bool)].copy()
    if sub.empty:
        return

    D_vals = sorted(sub["D"].astype(int).unique().tolist())
    fig, axes = plt.subplots(
        1, len(D_vals), figsize=(4.8 * len(D_vals), 3.9), constrained_layout=False
    )
    axes = np.atleast_1d(axes)

    thresholds = np.linspace(0.001, 0.2, 50)

    for ax, D in zip(axes, D_vals):
        dsub = sub.loc[sub["D"] == int(D)].copy()
        truth_pos = (
            (dsub["regime"] == "R_true")
            & (pd.to_numeric(dsub["beta"], errors="coerce") > 0.0)
        ).to_numpy(dtype=bool)

        pr_x: list[float] = []
        pr_y: list[float] = []
        for t in thresholds:
            info_min = int(
                max(int(min_info_abs), math.ceil(float(min_info_frac) * int(D)))
            )
            cond_info = pd.to_numeric(dsub["D_eff_info"], errors="coerce").to_numpy(
                dtype=float
            ) >= float(info_min)
            cond_dir = (
                pd.to_numeric(dsub["median_ang_err"], errors="coerce").to_numpy(
                    dtype=float
                )
                <= float(ang_thresh_rad)
            ) & (
                pd.to_numeric(dsub["frac_within_30deg"], errors="coerce").to_numpy(
                    dtype=float
                )
                >= float(frac_within_thresh)
            )
            if bool(use_k_consistency):
                cond_k = pd.to_numeric(dsub["K_consistency"], errors="coerce").to_numpy(
                    dtype=float
                ) >= float(k_consistency_thresh)
            else:
                cond_k = np.ones(dsub.shape[0], dtype=bool)

            pred = (
                (
                    pd.to_numeric(dsub["p_pool"], errors="coerce").to_numpy(dtype=float)
                    <= float(t)
                )
                & cond_info
                & cond_dir
                & cond_k
            )

            tp = int(np.sum(pred & truth_pos))
            fp = int(np.sum(pred & (~truth_pos)))
            fn = int(np.sum((~pred) & truth_pos))

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            pr_x.append(rec)
            pr_y.append(prec)

        ax.plot(pr_x, pr_y, linewidth=1.5)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"D={D}")

    fig.suptitle(
        (
            "Replication precision-recall (truth positive: R_true, beta>0)\n"
            f"N_total={N_total}, n_perm_pool={n_perm_pool}, n_perm_donor={n_perm_donor}, B={bins}"
        )
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    savefig(fig, plots_dir / "replication_PR_or_ROC.png")
    plt.close(fig)


def plot_example_replication_panels(
    *,
    gene_metrics: pd.DataFrame,
    contexts: dict[int, DonorGeometryContext],
    outdir: Path,
    bins: int,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")

    chosen: list[pd.Series] = []
    target_p = 0.03
    for regime in ["R_true", "R_donor_specific", "R_null"]:
        sub = gene_metrics.loc[
            (gene_metrics["regime"] == regime)
            & (~gene_metrics["pooled_underpowered"].astype(bool))
            & np.isfinite(pd.to_numeric(gene_metrics["p_pool"], errors="coerce"))
        ].copy()
        if sub.empty:
            continue
        sub["p_dist"] = np.abs(
            pd.to_numeric(sub["p_pool"], errors="coerce") - float(target_p)
        )
        chosen.append(sub.sort_values("p_dist").iloc[0])

    if (not chosen) and (not gene_metrics.empty):
        chosen = [gene_metrics.iloc[0]]
    if not chosen:
        return

    fig, axes = plt.subplots(
        1,
        len(chosen),
        figsize=(4.8 * len(chosen), 4.6),
        subplot_kw={"projection": "polar"},
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)

    centers = _theta_centers(int(bins))

    for i, (ax, row) in enumerate(zip(axes, chosen), start=1):
        D = int(row["D"])
        context = contexts[int(D)]

        seed_gene = int(row["seed"])  # stable gene seed used for simulation
        regime = str(row["regime"])
        pi_target = float(row["pi_target"])
        beta = float(row["beta"])
        sigma_eta = float(row["sigma_eta"])

        f_by_donor, f_pool, _, _, _ = simulate_gene_detections(
            context=context,
            regime=regime,
            pi_target=float(pi_target),
            beta=float(beta),
            sigma_eta=float(sigma_eta),
            seed_gene=int(seed_gene),
        )

        if i == 1:
            fig_emb, ax_emb = plt.subplots(figsize=(5.2, 4.4), constrained_layout=False)
            plot_embedding_with_foreground(
                context.pooled_X,
                f_pool,
                ax=ax_emb,
                title=(
                    "Representative embedding foreground (ExpG)\n"
                    f"{regime}, D={D}, pi={_fmt(pi_target)}, beta={_fmt(beta)}"
                ),
                s=5.0,
                alpha_bg=0.30,
                alpha_fg=0.75,
            )
            savefig(fig_emb, plots_dir / "embedding_example.png")
            plt.close(fig_emb)

        for d_idx in range(int(context.D)):
            f = f_by_donor[d_idx]
            if int(np.sum(f)) == 0 or int(np.sum(~f)) == 0:
                continue
            E, _, _ = compute_rsp_profile_from_boolean(
                f,
                context.theta_by_donor[d_idx],
                int(bins),
                bin_id=context.bin_id_by_donor[d_idx],
                bin_counts_total=context.bin_counts_by_donor[d_idx],
            )
            plot_rsp_polar(
                centers,
                E,
                ax=ax,
                color="#999999",
                linewidth=0.8,
                title=None,
            )

        pooled_theta = float(row["theta_hat_pool"])
        if np.isfinite(pooled_theta):
            rmin, rmax = ax.get_ylim()
            ax.plot(
                [pooled_theta, pooled_theta],
                [rmin, rmax],
                color="#111111",
                linestyle="--",
                linewidth=1.1,
            )

        ax.set_title(
            (
                f"{regime}\nD={D}, pi={_fmt(pi_target)}, beta={_fmt(beta)}, "
                f"p_pool={_fmt(float(row['p_pool']))}"
            )
        )
    fig.suptitle("Example donor-replication panels (donor E(theta) overlays)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    savefig(fig, plots_dir / "example_gene_replication_panels.png")
    plt.close(fig)


def run_validations(
    *,
    gene_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    outdir: Path,
) -> list[str]:
    results_dir = ensure_dir(outdir / "results")
    report_lines: list[str] = []
    warnings_out: list[str] = []

    non = gene_metrics.loc[~gene_metrics["pooled_underpowered"].astype(bool)].copy()
    if non.empty:
        msg = "No non-underpowered genes available; validation checks skipped."
        report_lines.append(msg)
        warnings_out.append(msg)
        (results_dir / "validation_debug_report.txt").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )
        return warnings_out

    # R_true: replicable fraction should increase with beta and D.
    true_sub = summary.loc[
        (summary["regime"] == "R_true") & (summary["beta"] > 0)
    ].copy()
    if true_sub.empty:
        warnings_out.append("R_true validation skipped: no R_true beta>0 cells.")
    else:
        # beta monotonicity within each D.
        for D, grp in true_sub.groupby("D", sort=True):
            agg = (
                grp.groupby("beta", sort=True)["frac_replicable_localized"]
                .mean()
                .reset_index()
            )
            if agg.shape[0] >= 2:
                rho, _ = spearmanr(
                    agg["beta"], agg["frac_replicable_localized"], nan_policy="omit"
                )
                r = float(rho) if rho is not None else float("nan")
                report_lines.append(f"R_true beta trend D={int(D)}: spearman={r:.3f}")
                if not np.isfinite(r) or r < 0.4:
                    warnings_out.append(
                        f"R_true replicable-vs-beta trend weak at D={int(D)} (rho={r:.3f})."
                    )

        # D monotonicity within each beta.
        for beta, grp in true_sub.groupby("beta", sort=True):
            agg = (
                grp.groupby("D", sort=True)["frac_replicable_localized"]
                .mean()
                .reset_index()
            )
            if agg.shape[0] >= 2:
                rho, _ = spearmanr(
                    agg["D"], agg["frac_replicable_localized"], nan_policy="omit"
                )
                r = float(rho) if rho is not None else float("nan")
                report_lines.append(
                    f"R_true D trend beta={_fmt(float(beta))}: spearman={r:.3f}"
                )
                if not np.isfinite(r) or r < 0.2:
                    warnings_out.append(
                        f"R_true replicable-vs-D trend weak at beta={_fmt(float(beta))} (rho={r:.3f})."
                    )

    # R_null false replicable rate low.
    null_sub = summary.loc[(summary["regime"] == "R_null")].copy()
    if not null_sub.empty:
        high = null_sub.loc[null_sub["frac_replicable_localized"] > 0.10]
        report_lines.append(
            "R_null false replicable rates: "
            + ", ".join(
                [
                    f"{x:.3f}"
                    for x in null_sub["frac_replicable_localized"].head(8).tolist()
                ]
            )
        )
        if not high.empty:
            for _, row in high.iterrows():
                warnings_out.append(
                    (
                        "High R_null false replicable rate at "
                        f"D={int(row['D'])}, sigma={_fmt(float(row['sigma_eta']))}, "
                        f"pi={_fmt(float(row['pi_target']))}, beta={_fmt(float(row['beta']))}: "
                        f"{float(row['frac_replicable_localized']):.3f}"
                    )
                )

    # donor_specific should mostly be nonreplicable when localized.
    ds_sub = non.loc[non["regime"] == "R_donor_specific"].copy()
    if not ds_sub.empty:
        pooled_sig = ds_sub["p_pool"].le(0.05)
        if int(np.sum(pooled_sig)) > 0:
            nonrepl_rate = float(
                np.mean(
                    ds_sub.loc[pooled_sig, "final_replication_label"]
                    == "LOCALIZED_NONREPLICABLE"
                )
            )
            report_lines.append(
                f"R_donor_specific nonreplicable among pooled-significant: {nonrepl_rate:.3f}"
            )
            if nonrepl_rate < 0.7:
                warnings_out.append(
                    f"R_donor_specific nonreplicable rate among pooled-significant is low ({nonrepl_rate:.3f} < 0.7)."
                )

    # Angle errors: R_true should be smaller than donor_specific (beta>=1, pi>=0.2).
    ang_true = non.loc[
        (non["regime"] == "R_true")
        & (pd.to_numeric(non["beta"], errors="coerce") >= 1.0)
        & (pd.to_numeric(non["pi_target"], errors="coerce") >= 0.2),
        "median_ang_err",
    ]
    ang_ds = non.loc[
        (non["regime"] == "R_donor_specific")
        & (pd.to_numeric(non["beta"], errors="coerce") >= 1.0)
        & (pd.to_numeric(non["pi_target"], errors="coerce") >= 0.2),
        "median_ang_err",
    ]
    arr_t = pd.to_numeric(ang_true, errors="coerce").to_numpy(dtype=float)
    arr_s = pd.to_numeric(ang_ds, errors="coerce").to_numpy(dtype=float)
    arr_t = arr_t[np.isfinite(arr_t)]
    arr_s = arr_s[np.isfinite(arr_s)]
    if arr_t.size > 0 and arr_s.size > 0:
        med_t = float(np.median(arr_t))
        med_s = float(np.median(arr_s))
        report_lines.append(
            f"Median angle error R_true={med_t:.3f}, R_donor_specific={med_s:.3f}"
        )
        if med_t >= med_s:
            warnings_out.append(
                f"Angle-error separation failed: median R_true ({med_t:.3f}) >= R_donor_specific ({med_s:.3f})."
            )

    if not report_lines:
        report_lines.append("No validation statistics were computed.")

    if warnings_out:
        report_lines.append("")
        report_lines.append("Validation warnings:")
        report_lines.extend([f"- {w}" for w in warnings_out])
    else:
        report_lines.append("")
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

    run_id = timestamped_run_id(prefix="expG")

    min_perm_pool_eff = int(args.min_perm_pool)
    if int(args.n_perm_pool) < int(min_perm_pool_eff):
        warnings.warn(
            f"n_perm_pool ({int(args.n_perm_pool)}) < min_perm_pool ({int(min_perm_pool_eff)}); lowering gate.",
            RuntimeWarning,
            stacklevel=2,
        )
        min_perm_pool_eff = int(args.n_perm_pool)

    cfg = {
        "experiment": "expG_donor_replication",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit_hash(cwd=REPO_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "master_seed": int(args.master_seed),
        "N_total": int(args.N_total),
        "D_grid": [int(x) for x in args.D_grid],
        "sigma_eta_grid": [float(x) for x in args.sigma_eta_grid],
        "pi_grid": [float(x) for x in args.pi_grid],
        "beta_grid": [float(x) for x in args.beta_grid],
        "regimes": list(DEFAULT_REGIMES),
        "genes_per_condition": int(args.genes_per_condition),
        "n_perm_pool": int(args.n_perm_pool),
        "n_perm_donor": int(args.n_perm_donor),
        "bins": int(args.bins),
        "within_donor_power": {
            "donor_prev_floor": float(args.donor_prev_floor),
            "donor_min_fg": int(args.donor_min_fg),
        },
        "pooled_power": {
            "p_min": float(args.p_min),
            "min_fg_total": int(args.min_fg_total),
            "min_fg_per_donor": int(args.min_fg_per_donor),
            "min_bg_per_donor": int(args.min_bg_per_donor),
            "d_eff_min": int(args.d_eff_min),
            "min_perm_pool": int(min_perm_pool_eff),
        },
        "replication_rule": {
            "p_thresh": float(args.replication_p_thresh),
            "min_info_abs": int(args.replication_min_info_abs),
            "min_info_frac": float(args.replication_min_info_frac),
            "ang_thresh_deg": float(args.replication_ang_thresh_deg),
            "frac_within_thresh": float(args.replication_frac_within_thresh),
            "informative_requires_p": bool(args.informative_requires_p),
            "informative_p_thresh": float(args.informative_p_thresh),
            "use_k_consistency": bool(args.use_k_consistency),
            "k_consistency_thresh": float(args.k_consistency_thresh),
        },
    }
    write_config(outdir, cfg)

    print(
        (
            "Running ExpG with "
            f"N_total={int(args.N_total)}, D_grid={list(args.D_grid)}, sigma_grid={list(args.sigma_eta_grid)}, "
            f"pi_grid={list(args.pi_grid)}, beta_grid={list(args.beta_grid)}, "
            f"genes_per_condition={int(args.genes_per_condition)}, n_perm_pool={int(args.n_perm_pool)}, "
            f"n_perm_donor={int(args.n_perm_donor)}, bins={int(args.bins)}"
        ),
        flush=True,
    )

    contexts: dict[int, DonorGeometryContext] = {}
    for D in args.D_grid:
        seed_ctx = stable_seed(
            int(args.master_seed),
            "expG",
            "context",
            int(D),
            int(args.N_total),
            int(args.bins),
        )
        contexts[int(D)] = build_donor_geometry_context(
            N_total=int(args.N_total),
            D=int(D),
            bins=int(args.bins),
            seed=int(seed_ctx),
        )

    donor_rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []

    start_t = time.time()
    total_genes = (
        len(DEFAULT_REGIMES)
        * len(args.D_grid)
        * len(args.sigma_eta_grid)
        * len(args.pi_grid)
        * len(args.beta_grid)
        * int(args.genes_per_condition)
    )
    gene_counter = 0

    ang_thresh_rad = float(np.deg2rad(float(args.replication_ang_thresh_deg)))

    for regime in DEFAULT_REGIMES:
        for D in args.D_grid:
            context = contexts[int(D)]
            for sigma_eta in args.sigma_eta_grid:
                for pi_target in args.pi_grid:
                    for beta in args.beta_grid:
                        print(
                            (
                                f"Condition regime={regime}, D={int(D)}, sigma={_fmt(float(sigma_eta))}, "
                                f"pi={_fmt(float(pi_target))}, beta={_fmt(float(beta))}"
                            ),
                            flush=True,
                        )
                        for gene_idx in range(int(args.genes_per_condition)):
                            seed_gene = stable_seed(
                                int(args.master_seed),
                                "expG",
                                "gene",
                                str(regime),
                                int(D),
                                float(sigma_eta),
                                float(pi_target),
                                float(beta),
                                int(gene_idx),
                            )

                            f_by_donor, f_pool, theta0_global, theta0_donor, eta_d = (
                                simulate_gene_detections(
                                    context=context,
                                    regime=str(regime),
                                    pi_target=float(pi_target),
                                    beta=float(beta),
                                    sigma_eta=float(sigma_eta),
                                    seed_gene=int(seed_gene),
                                )
                            )

                            # Per-donor scoring.
                            donor_stats: list[dict[str, Any]] = []
                            for d_idx in range(int(context.D)):
                                seed_perm_d = stable_seed(
                                    int(args.master_seed),
                                    "expG",
                                    "donor_perm",
                                    int(seed_gene),
                                    int(d_idx),
                                )
                                dstat = score_single_donor(
                                    f_d=f_by_donor[d_idx],
                                    theta_d=context.theta_by_donor[d_idx],
                                    bin_id_d=context.bin_id_by_donor[d_idx],
                                    bin_counts_d=context.bin_counts_by_donor[d_idx],
                                    bins=int(args.bins),
                                    n_perm_donor=int(args.n_perm_donor),
                                    seed_perm=int(seed_perm_d),
                                    donor_prev_floor=float(args.donor_prev_floor),
                                    donor_min_fg=int(args.donor_min_fg),
                                )
                                donor_stats.append(dstat)

                            # Pooled scoring.
                            seed_perm_pool = stable_seed(
                                int(args.master_seed),
                                "expG",
                                "pool_perm",
                                int(seed_gene),
                            )
                            pstat = score_pooled(
                                f_pool=f_pool,
                                context=context,
                                bins=int(args.bins),
                                n_perm_pool=int(args.n_perm_pool),
                                seed_perm=int(seed_perm_pool),
                                p_min=float(args.p_min),
                                min_fg_total=int(args.min_fg_total),
                                min_fg_per_donor=int(args.min_fg_per_donor),
                                min_bg_per_donor=int(args.min_bg_per_donor),
                                d_eff_min=int(args.d_eff_min),
                                min_perm=int(min_perm_pool_eff),
                            )

                            # Informative donor mask for replication metrics.
                            informative_mask = np.zeros(int(context.D), dtype=bool)
                            theta_hat_vec = np.full(int(context.D), np.nan, dtype=float)
                            k_hat_vec = np.full(int(context.D), np.nan, dtype=float)
                            p_d_vec = np.full(int(context.D), np.nan, dtype=float)
                            ang_err_vec = np.full(int(context.D), np.nan, dtype=float)

                            for d_idx, dstat in enumerate(donor_stats):
                                theta_hat_vec[d_idx] = float(dstat["theta_hat_d"])
                                k_hat_vec[d_idx] = float(dstat["K_hat_d"])
                                p_d_vec[d_idx] = float(dstat["p_d"])

                                informative = not bool(dstat["underpowered_d"])
                                if bool(args.informative_requires_p):
                                    informative = (
                                        informative
                                        and np.isfinite(float(dstat["p_d"]))
                                        and (
                                            float(dstat["p_d"])
                                            <= float(args.informative_p_thresh)
                                        )
                                    )
                                informative_mask[d_idx] = bool(informative)

                                if (
                                    informative
                                    and np.isfinite(float(pstat["theta_hat_pool"]))
                                    and np.isfinite(float(dstat["theta_hat_d"]))
                                ):
                                    ang_err_vec[d_idx] = _circular_distance(
                                        float(dstat["theta_hat_d"]),
                                        float(pstat["theta_hat_pool"]),
                                    )

                            info_err = ang_err_vec[informative_mask]
                            info_err = info_err[np.isfinite(info_err)]
                            median_ang_err = (
                                float(np.median(info_err))
                                if info_err.size
                                else float("nan")
                            )
                            frac_within = (
                                float(np.mean(info_err <= ang_thresh_rad))
                                if info_err.size
                                else float("nan")
                            )
                            D_eff_info = int(
                                np.sum(informative_mask & np.isfinite(theta_hat_vec))
                            )

                            k_mode, k_consistency = _mode_int(
                                k_hat_vec[informative_mask]
                            )

                            label = assign_replication_label(
                                pooled_underpowered=bool(pstat["pooled_underpowered"]),
                                p_pool=float(pstat["p_pool"]),
                                D=int(D),
                                D_eff_info=int(D_eff_info),
                                median_ang_err=float(median_ang_err),
                                frac_within_30deg=float(frac_within),
                                K_consistency=float(k_consistency),
                                p_thresh=float(args.replication_p_thresh),
                                min_info_abs=int(args.replication_min_info_abs),
                                min_info_frac=float(args.replication_min_info_frac),
                                ang_thresh_rad=float(ang_thresh_rad),
                                frac_within_thresh=float(
                                    args.replication_frac_within_thresh
                                ),
                                use_k_consistency=bool(args.use_k_consistency),
                                k_consistency_thresh=float(args.k_consistency_thresh),
                            )

                            # Store donor rows.
                            for d_idx, dstat in enumerate(donor_stats):
                                donor_rows.append(
                                    {
                                        "run_id": run_id,
                                        "seed": int(seed_gene),
                                        "gene_index": int(gene_counter),
                                        "regime": str(regime),
                                        "D": int(D),
                                        "donor_id": int(d_idx),
                                        "sigma_eta": float(sigma_eta),
                                        "pi_target": float(pi_target),
                                        "beta": float(beta),
                                        "N_d": int(dstat["N_d"]),
                                        "prev_d": float(dstat["prev_d"]),
                                        "n_fg_d": int(dstat["n_fg_d"]),
                                        "underpowered_d": bool(dstat["underpowered_d"]),
                                        "T_d": float(dstat["T_d"]),
                                        "p_d": float(dstat["p_d"]),
                                        "theta_hat_d": float(dstat["theta_hat_d"]),
                                        "K_hat_d": float(dstat["K_hat_d"]),
                                        "informative_d": bool(informative_mask[d_idx]),
                                        "ang_err_d": float(ang_err_vec[d_idx]),
                                    }
                                )

                            # Store per-gene row.
                            gene_rows.append(
                                {
                                    "run_id": run_id,
                                    "seed": int(seed_gene),
                                    "gene_index": int(gene_counter),
                                    "regime": str(regime),
                                    "D": int(D),
                                    "sigma_eta": float(sigma_eta),
                                    "pi_target": float(pi_target),
                                    "beta": float(beta),
                                    "pooled_prev": float(pstat["pooled_prev"]),
                                    "pooled_n_fg": int(pstat["pooled_n_fg"]),
                                    "pooled_D_eff": int(pstat["pooled_D_eff"]),
                                    "pooled_underpowered": bool(
                                        pstat["pooled_underpowered"]
                                    ),
                                    "T_pool": float(pstat["T_pool"]),
                                    "p_pool": float(pstat["p_pool"]),
                                    "theta_hat_pool": float(pstat["theta_hat_pool"]),
                                    "K_hat_pool": float(pstat["K_hat_pool"]),
                                    "D_eff_info": int(D_eff_info),
                                    "median_ang_err": float(median_ang_err),
                                    "frac_within_30deg": float(frac_within),
                                    "K_mode": int(k_mode),
                                    "K_consistency": float(k_consistency),
                                    "theta0_global": float(theta0_global),
                                    "final_replication_label": str(label),
                                }
                            )

                            gene_counter += 1
                            if gene_counter % int(
                                args.progress_every
                            ) == 0 or gene_counter == int(total_genes):
                                elapsed = time.time() - start_t
                                rate = (
                                    gene_counter / elapsed
                                    if elapsed > 0
                                    else float("nan")
                                )
                                print(
                                    f"  progress: genes={gene_counter}/{total_genes} ({rate:.2f} genes/s)",
                                    flush=True,
                                )

    donor_metrics = pd.DataFrame(donor_rows)
    gene_metrics = pd.DataFrame(gene_rows)
    if donor_metrics.empty or gene_metrics.empty:
        raise RuntimeError("No output rows produced.")

    summary = summarize_gene_metrics(gene_metrics)

    atomic_write_csv(results_dir / "donor_metrics_long.csv", donor_metrics)
    atomic_write_csv(results_dir / "gene_metrics.csv", gene_metrics)
    atomic_write_csv(results_dir / "summary.csv", summary)

    _set_plot_style()
    plot_replicable_rate_curves(
        summary=summary,
        outdir=outdir,
        N_total=int(args.N_total),
        n_perm_pool=int(args.n_perm_pool),
        n_perm_donor=int(args.n_perm_donor),
        bins=int(args.bins),
        ang_thresh_deg=float(args.replication_ang_thresh_deg),
        frac_within_thresh=float(args.replication_frac_within_thresh),
    )
    plot_angle_error_hist(
        donor_metrics=donor_metrics,
        outdir=outdir,
        N_total=int(args.N_total),
        n_perm_pool=int(args.n_perm_pool),
        n_perm_donor=int(args.n_perm_donor),
        bins=int(args.bins),
        pi_ref=float(args.pi_ref_plot),
        beta_ref=float(args.beta_ref_plot),
    )
    plot_pooled_vs_concordance(
        gene_metrics=gene_metrics,
        outdir=outdir,
        N_total=int(args.N_total),
        n_perm_pool=int(args.n_perm_pool),
        n_perm_donor=int(args.n_perm_donor),
        bins=int(args.bins),
    )
    plot_replication_pr(
        gene_metrics=gene_metrics,
        outdir=outdir,
        N_total=int(args.N_total),
        n_perm_pool=int(args.n_perm_pool),
        n_perm_donor=int(args.n_perm_donor),
        bins=int(args.bins),
        min_info_abs=int(args.replication_min_info_abs),
        min_info_frac=float(args.replication_min_info_frac),
        ang_thresh_rad=float(np.deg2rad(float(args.replication_ang_thresh_deg))),
        frac_within_thresh=float(args.replication_frac_within_thresh),
        use_k_consistency=bool(args.use_k_consistency),
        k_consistency_thresh=float(args.k_consistency_thresh),
    )
    plot_example_replication_panels(
        gene_metrics=gene_metrics,
        contexts=contexts,
        outdir=outdir,
        bins=int(args.bins),
    )

    warnings_out = run_validations(
        gene_metrics=gene_metrics, summary=summary, outdir=outdir
    )

    elapsed = time.time() - start_t
    print(
        (
            f"Completed ExpG in {elapsed/60.0:.2f} min. "
            f"genes={gene_metrics.shape[0]}, donor_rows={donor_metrics.shape[0]}, "
            f"warnings={len(warnings_out)}"
        ),
        flush=True,
    )
    print(f"Outputs written to: {outdir}", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Simulation Experiment G: donor replication and cross-donor generalization."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/simulations/expG_donor_replication",
        help="Output directory.",
    )
    parser.add_argument("--master_seed", type=int, default=123, help="Master seed.")

    parser.add_argument(
        "--N_total", type=int, default=20000, help="Total cells across donors."
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
        default=DEFAULT_SIGMA_GRID,
        help="Donor offset sigma grid.",
    )
    parser.add_argument(
        "--pi_grid",
        type=float,
        nargs="+",
        default=DEFAULT_PI_GRID,
        help="Target prevalence grid.",
    )
    parser.add_argument(
        "--beta_grid",
        type=float,
        nargs="+",
        default=DEFAULT_BETA_GRID,
        help="Effect-size beta grid.",
    )

    parser.add_argument(
        "--genes_per_condition",
        type=int,
        default=200,
        help="Genes per (regime,D,sigma,pi,beta).",
    )
    parser.add_argument(
        "--n_perm_pool",
        type=int,
        default=300,
        help="Permutation count for pooled scoring.",
    )
    parser.add_argument(
        "--n_perm_donor",
        type=int,
        default=150,
        help="Permutation count for per-donor scoring.",
    )
    parser.add_argument("--bins", type=int, default=36, help="Angular bins.")

    parser.add_argument(
        "--donor_prev_floor",
        type=float,
        default=0.005,
        help="Within-donor prevalence floor.",
    )
    parser.add_argument(
        "--donor_min_fg",
        type=int,
        default=20,
        help="Within-donor minimum foreground count.",
    )

    parser.add_argument(
        "--p_min",
        type=float,
        default=0.005,
        help="Pooled underpowered prevalence floor.",
    )
    parser.add_argument(
        "--min_fg_total",
        type=int,
        default=50,
        help="Pooled underpowered min total foreground.",
    )
    parser.add_argument(
        "--min_fg_per_donor",
        type=int,
        default=10,
        help="Pooled underpowered min fg per donor.",
    )
    parser.add_argument(
        "--min_bg_per_donor",
        type=int,
        default=10,
        help="Pooled underpowered min bg per donor.",
    )
    parser.add_argument(
        "--d_eff_min",
        type=int,
        default=2,
        help="Pooled underpowered min informative donors.",
    )
    parser.add_argument(
        "--min_perm_pool",
        type=int,
        default=100,
        help="Pooled underpowered min permutations.",
    )

    parser.add_argument(
        "--replication_p_thresh",
        type=float,
        default=0.05,
        help="Pooled p-value threshold for localization.",
    )
    parser.add_argument(
        "--replication_min_info_abs",
        type=int,
        default=3,
        help="Minimum informative donors (absolute).",
    )
    parser.add_argument(
        "--replication_min_info_frac",
        type=float,
        default=1.0 / 3.0,
        help="Minimum informative donors as fraction of D.",
    )
    parser.add_argument(
        "--replication_ang_thresh_deg",
        type=float,
        default=30.0,
        help="Median angle-error threshold in degrees.",
    )
    parser.add_argument(
        "--replication_frac_within_thresh",
        type=float,
        default=0.6,
        help="Fraction-within-30deg threshold.",
    )

    parser.add_argument(
        "--informative_requires_p",
        action="store_true",
        help="Require per-donor p_d <= informative_p_thresh for informative donors.",
    )
    parser.add_argument(
        "--informative_p_thresh",
        type=float,
        default=0.1,
        help="Per-donor p-threshold for informative donors when enabled.",
    )

    parser.add_argument(
        "--use_k_consistency",
        action="store_true",
        help="Include K-consistency threshold in replication rule.",
    )
    parser.add_argument(
        "--k_consistency_thresh",
        type=float,
        default=0.6,
        help="Minimum K-consistency if enabled.",
    )

    parser.add_argument(
        "--pi_ref_plot",
        type=float,
        default=0.2,
        help="Representative pi for angle-error histogram.",
    )
    parser.add_argument(
        "--beta_ref_plot",
        type=float,
        default=1.0,
        help="Representative beta for angle-error histogram.",
    )

    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Progress print frequency in genes.",
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
        args = apply_testmode_overrides(args, exp_name="expG_donor_replication")

    if int(args.N_total) <= 0:
        raise ValueError("N_total must be positive.")
    if int(args.bins) <= 0:
        raise ValueError("bins must be positive.")
    if int(args.n_perm_pool) <= 0 or int(args.n_perm_donor) <= 0:
        raise ValueError("Permutation counts must be positive.")
    if int(args.genes_per_condition) <= 0:
        raise ValueError("genes_per_condition must be positive.")

    run_ctx = prepare_legacy_run(args, "expG_donor_replication", __file__)
    run_experiment(args)
    finalize_legacy_run(run_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
