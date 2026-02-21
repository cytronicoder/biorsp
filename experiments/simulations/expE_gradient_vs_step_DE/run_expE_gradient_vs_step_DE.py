#!/usr/bin/env python3
"""Simulation Experiment E: continuous gradient vs cluster-separable step change.

This experiment compares BioRSP max-statistic scoring against Moran's I and a
cluster-DE proxy across step-change and continuous-gradient scenarios.
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
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from scipy.stats import kstest, rankdata, spearmanr

# Safe non-interactive plotting setup.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-biorsp-expE")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache-biorsp-expE")
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from biorsp.moran import morans_i
from biorsp.permutation import perm_null_T
from biorsp.power import evaluate_underpowered
from biorsp.rsp import compute_rsp_profile_from_boolean
from experiments.simulations._shared.cli import add_common_args
from experiments.simulations._shared.donors import (
    assign_donors,
    donor_effect_vector,
    sample_donor_effects,
)
from experiments.simulations._shared.geometry import sample_disk_gaussian
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
)
from experiments.simulations._shared.runner import (
    finalize_legacy_run,
    prepare_legacy_run,
)
from experiments.simulations._shared.seeding import rng_from_seed, stable_seed
from experiments.simulations._sim_testmode import banner as testmode_banner
from experiments.simulations._sim_testmode import get_testmode_config, resolve_outdir

SCENARIOS = ["S1_step", "S2_gradient"]
DEFAULT_D_GRID = [5, 10]
DEFAULT_SIGMA_ETA_GRID = [0.0, 0.4]
DEFAULT_PI_GRID = [0.01, 0.05, 0.2, 0.6]
DEFAULT_BETA_GRID = [0.0, 0.5, 1.0, 1.25]
DEFAULT_K_GRID = [2, 4, 8, 16]


@dataclass(frozen=True)
class DatasetContext:
    scenario: str
    N: int
    D: int
    sigma_eta: float
    seed_run: int
    X: np.ndarray
    theta: np.ndarray
    donor_ids: np.ndarray
    donor_eta_cell: np.ndarray
    step_labels: np.ndarray | None
    bin_id: np.ndarray
    bin_counts_total: np.ndarray


def _fmt(x: float | int) -> str:
    return f"{float(x):.3f}".rstrip("0").rstrip(".")


def _tok(x: float | int | str) -> str:
    return str(x).replace(".", "p")


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


def _clip_pi(pi_target: float) -> float:
    return float(np.clip(float(pi_target), 1e-12, 1.0 - 1e-12))


def _compute_bin_cache(theta: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    wrapped = np.mod(np.asarray(theta, dtype=float).ravel(), 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, int(n_bins) + 1, endpoint=True)
    bin_id = np.digitize(wrapped, edges, right=False) - 1
    bin_id = np.where(bin_id == int(n_bins), int(n_bins) - 1, bin_id).astype(np.int32)
    bin_counts_total = np.bincount(bin_id, minlength=int(n_bins)).astype(np.int64)
    return bin_id, bin_counts_total


def _sample_two_islands(
    N: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = int(N)
    n1 = n // 2
    n0 = n - n1
    x0 = rng.normal(loc=(-2.0, 0.0), scale=1.0, size=(n0, 2)).astype(float)
    x1 = rng.normal(loc=(2.0, 0.0), scale=1.0, size=(n1, 2)).astype(float)
    X = np.vstack([x0, x1])
    z = np.concatenate([np.zeros(n0, dtype=np.int8), np.ones(n1, dtype=np.int8)])
    order = rng.permutation(n)
    return X[order], z[order]


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


def _run_kmeans_labels(X: np.ndarray, K: int, seed: int) -> np.ndarray:
    n = int(X.shape[0])
    k = int(min(int(K), max(2, n)))
    rng = rng_from_seed(int(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, labels = kmeans2(
            np.asarray(X, dtype=float),
            k,
            iter=40,
            minit="points",
            missing="warn",
            rng=rng,
        )
    out = np.asarray(labels, dtype=np.int16).ravel()
    if out.size != n:
        raise RuntimeError(
            f"kmeans2 returned labels with unexpected length: {out.size} != {n}"
        )
    return out


def _de_score_range(f: np.ndarray, labels: np.ndarray, K: int) -> float:
    arr = np.asarray(f, dtype=bool).ravel()
    lab = np.asarray(labels, dtype=int).ravel()
    k = int(K)
    if arr.size != lab.size:
        raise ValueError("f and labels must have same length.")

    counts = np.bincount(lab, minlength=k).astype(float)
    fg = np.bincount(lab, weights=arr.astype(float), minlength=k).astype(float)
    valid = counts > 0
    if int(np.sum(valid)) < 2:
        return 0.0
    prev = np.divide(fg, counts, out=np.zeros_like(fg), where=valid)
    return float(np.max(prev[valid]) - np.min(prev[valid]))


def _robust_z(obs: float, null_values: np.ndarray) -> float:
    null_arr = np.asarray(null_values, dtype=float).ravel()
    if null_arr.size == 0:
        return float("nan")
    med = float(np.median(null_arr))
    mad = float(np.median(np.abs(null_arr - med)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        if abs(float(obs) - med) <= 1e-12:
            return 0.0
        return float(np.sign(float(obs) - med) * np.inf)
    return float((float(obs) - med) / scale)


def _mean_ci(values: np.ndarray) -> tuple[float, float, float, int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(arr))
    if n == 1:
        return mean, mean, mean, n
    se = float(np.std(arr, ddof=1) / math.sqrt(n))
    half = 1.96 * se
    return mean, mean - half, mean + half, n


def _bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_boot: int = 200,
) -> tuple[float, float, float, int]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_use = x_arr[mask]
    y_use = y_arr[mask]
    n = int(x_use.size)
    if n < 3 or np.allclose(x_use, x_use[0]) or np.allclose(y_use, y_use[0]):
        return float("nan"), float("nan"), float("nan"), n

    rho, _ = spearmanr(x_use, y_use, nan_policy="omit")
    rho_f = float(rho) if rho is not None else float("nan")

    rng = rng_from_seed(int(seed))
    boot = np.full(int(n_boot), np.nan, dtype=float)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        xb = x_use[idx]
        yb = y_use[idx]
        if np.allclose(xb, xb[0]) or np.allclose(yb, yb[0]):
            continue
        rb, _ = spearmanr(xb, yb, nan_policy="omit")
        if rb is not None and np.isfinite(float(rb)):
            boot[i] = float(rb)

    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return rho_f, float("nan"), float("nan"), n
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return rho_f, float(lo), float(hi), n


def _simulate_gene(
    *,
    scenario: str,
    pi_target: float,
    beta: float,
    theta0: float,
    theta: np.ndarray,
    donor_eta_cell: np.ndarray,
    step_labels: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    alpha = float(logit(_clip_pi(float(pi_target))))
    if str(scenario) == "S1_step":
        if step_labels is None:
            raise ValueError("S1_step requires step_labels.")
        s = np.where(np.asarray(step_labels, dtype=int) > 0, 1.0, -1.0)
    elif str(scenario) == "S2_gradient":
        s = np.cos(np.asarray(theta, dtype=float) - float(theta0))
    else:
        raise ValueError(f"Unknown scenario '{scenario}'.")

    logits = alpha + float(beta) * s + np.asarray(donor_eta_cell, dtype=float)
    p = expit(logits)
    return (rng.random(p.size) < p).astype(bool)


def _build_dataset(
    *,
    scenario: str,
    N: int,
    D: int,
    sigma_eta: float,
    seed_run: int,
    master_seed: int,
    n_bins: int,
    k_nn: int,
    k_values: list[int],
) -> tuple[DatasetContext, sp.csr_matrix, dict[int, np.ndarray]]:
    seed_dataset = stable_seed(
        int(master_seed),
        "expE",
        "dataset",
        int(seed_run),
        str(scenario),
        int(N),
        int(D),
        float(sigma_eta),
    )
    rng = rng_from_seed(int(seed_dataset))

    if str(scenario) == "S1_step":
        X, step_labels = _sample_two_islands(int(N), rng)
    elif str(scenario) == "S2_gradient":
        X, _ = sample_disk_gaussian(int(N), rng)
        step_labels = None
    else:
        raise ValueError(f"Unknown scenario '{scenario}'.")

    # Keep canonical arctan2 direction but wrap to [0, 2pi) for RSP utilities.
    theta = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2.0 * np.pi).astype(float)
    donor_ids = assign_donors(int(N), int(D), rng)
    eta_d = sample_donor_effects(int(D), float(sigma_eta), rng)
    eta_cell = donor_effect_vector(donor_ids, eta_d)

    W = _build_knn_weights(X, int(k_nn))

    cluster_labels: dict[int, np.ndarray] = {}
    for K in k_values:
        seed_k = stable_seed(
            int(master_seed),
            "expE",
            "kmeans",
            int(seed_run),
            str(scenario),
            int(N),
            int(D),
            float(sigma_eta),
            int(K),
        )
        cluster_labels[int(K)] = _run_kmeans_labels(X, int(K), int(seed_k))

    bin_id, bin_counts = _compute_bin_cache(theta, int(n_bins))
    context = DatasetContext(
        scenario=str(scenario),
        N=int(N),
        D=int(D),
        sigma_eta=float(sigma_eta),
        seed_run=int(seed_run),
        X=np.asarray(X, dtype=float),
        theta=np.asarray(theta, dtype=float),
        donor_ids=np.asarray(donor_ids, dtype=np.int16),
        donor_eta_cell=np.asarray(eta_cell, dtype=float),
        step_labels=step_labels,
        bin_id=bin_id,
        bin_counts_total=bin_counts,
    )
    return context, W, cluster_labels


def summarize(metrics: pd.DataFrame, corr_boot: int, master_seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_group = ["scenario", "N", "D", "sigma_eta", "pi_target", "beta"]
    for keys, grp in metrics.groupby(base_group, sort=True):
        scenario, N, D, sigma_eta, pi_target, beta = keys
        n_genes = int(grp.shape[0])
        valid = grp.loc[~grp["underpowered"].astype(bool)]
        n_valid = int(valid.shape[0])
        frac_underpowered = float(grp["underpowered"].mean())

        rows.append(
            {
                "scenario": str(scenario),
                "N": int(N),
                "D": int(D),
                "sigma_eta": float(sigma_eta),
                "pi_target": float(pi_target),
                "beta": float(beta),
                "n_genes": n_genes,
                "n_non_underpowered": n_valid,
                "frac_underpowered": frac_underpowered,
                "mean_bio_Z": float(
                    pd.to_numeric(valid["bio_Z"], errors="coerce").mean()
                ),
                "mean_moran_I": float(
                    pd.to_numeric(valid["moran_I"], errors="coerce").mean()
                ),
                "mean_de_score_best": float(
                    pd.to_numeric(valid["de_score_best"], errors="coerce").mean()
                ),
            }
        )
    summary = pd.DataFrame(rows)

    corr_rows: list[dict[str, Any]] = []
    corr_group = ["scenario", "N", "D", "sigma_eta", "pi_target"]
    for keys, grp in metrics.groupby(corr_group, sort=True):
        scenario, N, D, sigma_eta, pi_target = keys
        valid = grp.loc[
            (~grp["underpowered"].astype(bool))
            & (pd.to_numeric(grp["beta"], errors="coerce") > 0)
        ]
        beta_vals = pd.to_numeric(valid["beta"], errors="coerce").to_numpy(dtype=float)

        rho_bio, bio_lo, bio_hi, n_corr = _bootstrap_spearman(
            beta_vals,
            pd.to_numeric(valid["bio_Z"], errors="coerce").to_numpy(dtype=float),
            seed=stable_seed(
                int(master_seed),
                "expE",
                "corr",
                str(scenario),
                int(N),
                int(D),
                float(sigma_eta),
                float(pi_target),
                "bio",
            ),
            n_boot=int(corr_boot),
        )
        rho_moran, mor_lo, mor_hi, _ = _bootstrap_spearman(
            beta_vals,
            pd.to_numeric(valid["moran_I"], errors="coerce").to_numpy(dtype=float),
            seed=stable_seed(
                int(master_seed),
                "expE",
                "corr",
                str(scenario),
                int(N),
                int(D),
                float(sigma_eta),
                float(pi_target),
                "moran",
            ),
            n_boot=int(corr_boot),
        )
        rho_de, de_lo, de_hi, _ = _bootstrap_spearman(
            beta_vals,
            pd.to_numeric(valid["de_score_best"], errors="coerce").to_numpy(
                dtype=float
            ),
            seed=stable_seed(
                int(master_seed),
                "expE",
                "corr",
                str(scenario),
                int(N),
                int(D),
                float(sigma_eta),
                float(pi_target),
                "de",
            ),
            n_boot=int(corr_boot),
        )

        corr_rows.append(
            {
                "scenario": str(scenario),
                "N": int(N),
                "D": int(D),
                "sigma_eta": float(sigma_eta),
                "pi_target": float(pi_target),
                "n_corr_genes": int(n_corr),
                "spearman_bioZ_vs_beta": rho_bio,
                "spearman_bioZ_ci_low": bio_lo,
                "spearman_bioZ_ci_high": bio_hi,
                "spearman_moran_vs_beta": rho_moran,
                "spearman_moran_ci_low": mor_lo,
                "spearman_moran_ci_high": mor_hi,
                "spearman_debest_vs_beta": rho_de,
                "spearman_debest_ci_low": de_lo,
                "spearman_debest_ci_high": de_hi,
            }
        )

    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty and not summary.empty:
        summary = summary.merge(corr_df, on=corr_group, how="left")
    return summary


def _select_refs(values: list[float | int], target: float | int) -> float | int:
    arr = np.asarray(values, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(target))))
    return values[idx]


def plot_score_vs_beta(
    metrics: pd.DataFrame,
    *,
    outdir: Path,
    scenarios: list[str],
    beta_grid: list[float],
    pi_grid: list[float],
    N_ref: int,
    D_ref: int,
    sigma_ref: float,
    n_perm: int,
    bins: int,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    methods = [
        ("bio_Z", "BioRSP Z_T", "#1f77b4"),
        ("moran_I", "Moran's I", "#ff7f0e"),
        ("de_score_best", "DE score (best K)", "#2ca02c"),
    ]
    colors_pi = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]

    for scenario in scenarios:
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), constrained_layout=False)
        ax_arr = np.atleast_1d(axes)
        for m_idx, (col, label, _) in enumerate(methods):
            ax = ax_arr[m_idx]
            for p_idx, pi in enumerate(pi_grid):
                means: list[float] = []
                lows: list[float] = []
                highs: list[float] = []
                ns: list[int] = []
                for beta in beta_grid:
                    sub = metrics.loc[
                        (metrics["scenario"] == str(scenario))
                        & (metrics["N"] == int(N_ref))
                        & (metrics["D"] == int(D_ref))
                        & (metrics["sigma_eta"] == float(sigma_ref))
                        & (metrics["pi_target"] == float(pi))
                        & (metrics["beta"] == float(beta))
                        & (~metrics["underpowered"].astype(bool))
                    ]
                    mean, lo, hi, n = _mean_ci(
                        pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
                    )
                    means.append(mean)
                    lows.append(lo)
                    highs.append(hi)
                    ns.append(int(n))

                x = np.asarray(beta_grid, dtype=float)
                y = np.asarray(means, dtype=float)
                ylo = np.asarray(lows, dtype=float)
                yhi = np.asarray(highs, dtype=float)
                color = colors_pi[p_idx % len(colors_pi)]
                if np.isfinite(y).any():
                    yerr = np.vstack([y - ylo, yhi - y])
                    yerr = np.where(np.isfinite(yerr), np.maximum(yerr, 0.0), np.nan)
                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        marker="o",
                        linewidth=1.2,
                        markersize=3.5,
                        color=color,
                        capsize=2,
                        label=f"pi={_fmt(pi)}",
                    )
                    last_valid = np.where(np.isfinite(y))[0]
                    if last_valid.size > 0:
                        ii = int(last_valid[-1])
                        ax.text(x[ii], y[ii], f" n={ns[ii]}", fontsize=6.0, va="center")

            ax.set_xlabel("beta")
            ax.set_ylabel(label)
            ax.grid(True, axis="y", linestyle=":", alpha=0.35)
            ax.set_title(label)
            if m_idx == 0:
                ax.legend(frameon=False, ncol=1)

        label_s = "S1" if scenario == "S1_step" else "S2"
        fig.suptitle(
            (
                f"Experiment E {label_s}: method score vs beta (mean ±95% CI)\n"
                f"N={N_ref}, D={D_ref}, sigma={_fmt(sigma_ref)}, bins={bins}, n_perm={n_perm}, boolean foreground simulation"
            ),
            y=0.995,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
        savefig(fig, plots_dir / f"score_vs_beta_{label_s}.png")
        plt.close(fig)


def plot_spearman_summary(
    metrics: pd.DataFrame,
    *,
    outdir: Path,
    master_seed: int,
    corr_boot: int,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")
    methods = [
        ("bio_Z", "BioRSP Z_T", "#1f77b4"),
        ("moran_I", "Moran's I", "#ff7f0e"),
        ("de_score_best", "DE best", "#2ca02c"),
    ]
    scenarios = ["S1_step", "S2_gradient"]

    table: dict[tuple[str, str], tuple[float, float, float]] = {}
    for scenario in scenarios:
        sub = metrics.loc[
            (metrics["scenario"] == scenario)
            & (~metrics["underpowered"].astype(bool))
            & (pd.to_numeric(metrics["beta"], errors="coerce") > 0)
        ]
        beta_vals = pd.to_numeric(sub["beta"], errors="coerce").to_numpy(dtype=float)
        for col, _, _ in methods:
            rho, lo, hi, _ = _bootstrap_spearman(
                beta_vals,
                pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float),
                seed=stable_seed(
                    int(master_seed), "expE", "spearman_summary", scenario, col
                ),
                n_boot=int(corr_boot),
            )
            table[(scenario, col)] = (rho, lo, hi)

    x = np.arange(len(scenarios), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.9), constrained_layout=False)
    for idx, (col, label, color) in enumerate(methods):
        vals = []
        lo_err = []
        hi_err = []
        for scenario in scenarios:
            rho, lo, hi = table.get((scenario, col), (np.nan, np.nan, np.nan))
            vals.append(rho)
            lo_err.append(
                max(0.0, rho - lo) if np.isfinite(rho) and np.isfinite(lo) else np.nan
            )
            hi_err.append(
                max(0.0, hi - rho) if np.isfinite(rho) and np.isfinite(hi) else np.nan
            )
        xpos = x + (idx - 1) * width
        ax.bar(xpos, vals, width=width, color=color, alpha=0.9, label=label)
        ax.errorbar(
            xpos,
            vals,
            yerr=[lo_err, hi_err],
            fmt="none",
            ecolor="#333333",
            capsize=2,
            linewidth=0.8,
        )

    ax.axhline(0.0, color="#444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["S1 step-change", "S2 gradient"])
    ax.set_ylabel("Spearman(score, beta)")
    ax.set_title("Experiment E: rank correlation with truth beta (bootstrapped 95% CI)")
    ax.legend(frameon=False, ncol=3)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    savefig(fig, plots_dir / "spearman_summary.png")
    plt.close(fig)


def _rank_std_over_k(sub: pd.DataFrame) -> np.ndarray:
    cols = ["de_score_K2", "de_score_K4", "de_score_K8", "de_score_K16"]
    mat = np.column_stack(
        [pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float) for c in cols]
    )
    mask = np.all(np.isfinite(mat), axis=1)
    if not np.any(mask):
        return np.zeros(0, dtype=float)
    m = mat[mask]
    ranks = np.zeros_like(m, dtype=float)
    for j in range(m.shape[1]):
        ranks[:, j] = rankdata(-m[:, j], method="average")
    return np.std(ranks, axis=1)


def plot_de_instability(
    metrics: pd.DataFrame,
    *,
    outdir: Path,
    N_ref: int,
    D_ref: int,
    sigma_ref: float,
    pi_ref: float,
    n_perm: int,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")

    sub = metrics.loc[
        (metrics["scenario"] == "S2_gradient")
        & (metrics["N"] == int(N_ref))
        & (metrics["D"] == int(D_ref))
        & (metrics["sigma_eta"] == float(sigma_ref))
        & (metrics["pi_target"] == float(pi_ref))
        & (~metrics["underpowered"].astype(bool))
    ].copy()

    rank_std = _rank_std_over_k(sub)
    fig, ax = plt.subplots(figsize=(6.8, 3.7), constrained_layout=False)
    if rank_std.size > 0:
        bins = np.linspace(float(np.min(rank_std)), float(np.max(rank_std)) + 1e-9, 24)
        ax.hist(
            rank_std,
            bins=bins,
            color="#4C78A8",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axvline(
            float(np.mean(rank_std)),
            color="#d62728",
            linestyle="--",
            linewidth=1.0,
            label=f"mean={np.mean(rank_std):.2f}",
        )
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5,
            0.5,
            "No non-underpowered genes",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Std. dev. of DE rank across K={2,4,8,16}")
    ax.set_ylabel("Gene count")
    ax.set_title(
        (
            "Experiment E S2: DE rank instability distribution\n"
            f"N={N_ref}, D={D_ref}, sigma={_fmt(sigma_ref)}, pi={_fmt(pi_ref)}, n_perm={n_perm}, boolean foreground simulation"
        )
    )
    fig.tight_layout()
    savefig(fig, plots_dir / "de_instability_distribution_S2.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.9), constrained_layout=False)
    x = pd.to_numeric(sub["de_instability"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sub["bio_Z"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(sub["beta"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if np.any(m):
        sc = ax.scatter(x[m], y[m], c=c[m], cmap="viridis", s=18, alpha=0.75)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("beta")
    else:
        ax.text(
            0.5,
            0.5,
            "No finite points",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("DE instability (std of DE_score_K)")
    ax.set_ylabel("BioRSP Z_T")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_title(
        (
            "Experiment E S2: DE instability vs BioRSP signal\n"
            f"N={N_ref}, D={D_ref}, sigma={_fmt(sigma_ref)}, pi={_fmt(pi_ref)}"
        )
    )
    fig.tight_layout()
    savefig(fig, plots_dir / "de_instability_vs_bioZ_S2.png")
    plt.close(fig)


def plot_de_stability_compare(
    metrics: pd.DataFrame,
    *,
    outdir: Path,
    N_ref: int,
    D_ref: int,
    sigma_ref: float,
    pi_ref: float,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=False)

    # Panel A: DE score K=2 vs beta in S1/S2.
    ax = axes[0]
    for scenario, color in [("S1_step", "#1f77b4"), ("S2_gradient", "#d62728")]:
        sub = metrics.loc[
            (metrics["scenario"] == scenario)
            & (metrics["N"] == int(N_ref))
            & (metrics["D"] == int(D_ref))
            & (metrics["sigma_eta"] == float(sigma_ref))
            & (metrics["pi_target"] == float(pi_ref))
            & (~metrics["underpowered"].astype(bool))
        ].copy()

        betas = np.sort(pd.to_numeric(sub["beta"], errors="coerce").dropna().unique())
        ys = []
        for beta in betas:
            vals = pd.to_numeric(
                sub.loc[sub["beta"] == float(beta), "de_score_K2"], errors="coerce"
            ).to_numpy(dtype=float)
            ys.append(float(np.nanmean(vals)) if vals.size > 0 else np.nan)
        ax.plot(betas, ys, marker="o", color=color, linewidth=1.4, label=scenario)

        rho, _ = spearmanr(
            pd.to_numeric(sub.loc[sub["beta"] > 0, "beta"], errors="coerce").to_numpy(
                dtype=float
            ),
            pd.to_numeric(
                sub.loc[sub["beta"] > 0, "de_score_K2"], errors="coerce"
            ).to_numpy(dtype=float),
            nan_policy="omit",
        )
        if rho is not None and np.isfinite(float(rho)):
            ax.text(
                0.02,
                0.9 if scenario == "S1_step" else 0.82,
                f"{scenario} rho={float(rho):.2f}",
                transform=ax.transAxes,
                fontsize=7,
            )

    ax.set_xlabel("beta")
    ax.set_ylabel("mean DE_score_K2")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    ax.set_title("DE_K2 tracks beta in S1")

    # Panel B: instability compare.
    ax = axes[1]
    data = []
    labels = []
    for scenario in ["S1_step", "S2_gradient"]:
        sub = metrics.loc[
            (metrics["scenario"] == scenario)
            & (metrics["N"] == int(N_ref))
            & (metrics["D"] == int(D_ref))
            & (metrics["sigma_eta"] == float(sigma_ref))
            & (metrics["pi_target"] == float(pi_ref))
            & (~metrics["underpowered"].astype(bool))
            & (pd.to_numeric(metrics["beta"], errors="coerce") > 0)
        ]
        vals = pd.to_numeric(sub["de_instability"], errors="coerce").to_numpy(
            dtype=float
        )
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            data.append(vals)
            labels.append("S1 step" if scenario == "S1_step" else "S2 gradient")

    if data:
        ax.boxplot(data, tick_labels=labels, widths=0.6, patch_artist=True)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("DE instability (std score across K)")
    ax.set_title("Instability higher in gradient scenario")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    fig.suptitle(
        (
            "Experiment E: DE stability sanity (S1 vs S2)\n"
            f"N={N_ref}, D={D_ref}, sigma={_fmt(sigma_ref)}, pi={_fmt(pi_ref)}"
        ),
        y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    savefig(fig, plots_dir / "de_stability_compare_S1_vs_S2.png")
    plt.close(fig)


def plot_moran_vs_bio(
    metrics: pd.DataFrame,
    *,
    outdir: Path,
    N_ref: int,
    D_ref: int,
    sigma_ref: float,
    pi_ref: float,
    n_perm: int,
) -> None:
    plots_dir = ensure_dir(outdir / "plots")

    for scenario in ["S1_step", "S2_gradient"]:
        sub = metrics.loc[
            (metrics["scenario"] == scenario)
            & (metrics["N"] == int(N_ref))
            & (metrics["D"] == int(D_ref))
            & (metrics["sigma_eta"] == float(sigma_ref))
            & (metrics["pi_target"] == float(pi_ref))
            & (~metrics["underpowered"].astype(bool))
        ]
        x = pd.to_numeric(sub["moran_I"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["bio_Z"], errors="coerce").to_numpy(dtype=float)
        c = pd.to_numeric(sub["beta"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)

        fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=False)
        if np.any(m):
            sc = ax.scatter(x[m], y[m], c=c[m], cmap="viridis", s=20, alpha=0.78)
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label("beta")
        else:
            ax.text(
                0.5,
                0.5,
                "No finite points",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xlabel("Moran's I")
        ax.set_ylabel("BioRSP Z_T")
        ax.grid(True, linestyle=":", alpha=0.35)
        lbl = "S1" if scenario == "S1_step" else "S2"
        ax.set_title(
            (
                f"Experiment E {lbl}: Moran vs BioRSP\n"
                f"N={N_ref}, D={D_ref}, sigma={_fmt(sigma_ref)}, pi={_fmt(pi_ref)}, n_perm={n_perm}, boolean foreground simulation"
            )
        )
        fig.tight_layout()
        savefig(fig, plots_dir / f"moran_vs_bioZ_{lbl}.png")
        plt.close(fig)


def _rebuild_gene_from_row(row: pd.Series, context: DatasetContext) -> np.ndarray:
    seed_gene = int(row["seed"])
    rng = rng_from_seed(seed_gene)
    theta0 = float(rng.uniform(0.0, 2.0 * np.pi))

    return _simulate_gene(
        scenario=str(row["scenario"]),
        pi_target=float(row["pi_target"]),
        beta=float(row["beta"]),
        theta0=theta0,
        theta=np.asarray(context.theta, dtype=float),
        donor_eta_cell=np.asarray(context.donor_eta_cell, dtype=float),
        step_labels=context.step_labels,
        rng=rng,
    )


def plot_examples(
    metrics: pd.DataFrame,
    *,
    outdir: Path,
    bins: int,
    contexts: dict[tuple[str, int], DatasetContext],
    pi_ref: float,
    beta_grid: list[float],
) -> None:
    plots_dir = ensure_dir(outdir / "plots")

    beta_pos = sorted([float(b) for b in beta_grid if float(b) > 0.0])
    if not beta_pos:
        beta_targets = [0.0]
    elif len(beta_pos) == 1:
        beta_targets = [0.0, beta_pos[0], beta_pos[0]]
    elif len(beta_pos) == 2:
        beta_targets = [0.0, beta_pos[0], beta_pos[1]]
    else:
        beta_targets = [0.0, beta_pos[len(beta_pos) // 2], beta_pos[-1]]

    for scenario in ["S1_step", "S2_gradient"]:
        # Choose the first available context for this scenario.
        ctx_candidates = [c for (sc, _), c in contexts.items() if sc == scenario]
        if not ctx_candidates:
            continue
        ctx = ctx_candidates[0]

        sub = metrics.loc[
            (metrics["scenario"] == scenario)
            & (metrics["seed_run"] == int(ctx.seed_run))
            & (metrics["N"] == int(ctx.N))
            & (metrics["D"] == int(ctx.D))
            & (metrics["sigma_eta"] == float(ctx.sigma_eta))
            & (metrics["pi_target"] == float(pi_ref))
        ].copy()
        if sub.empty:
            continue

        chosen: list[pd.Series] = []
        for target in beta_targets:
            cand = sub.loc[
                np.isclose(pd.to_numeric(sub["beta"], errors="coerce"), float(target))
            ].copy()
            if cand.empty:
                continue
            cand = cand.sort_values(
                by=["underpowered", "bio_Z", "seed"], ascending=[True, False, True]
            )
            pick = cand.iloc[0]
            chosen.append(pick)

        # Guarantee at most 3 and unique seeds.
        dedup: dict[int, pd.Series] = {}
        for row in chosen:
            dedup[int(row["seed"])] = row
        chosen_rows = list(dedup.values())

        if len(chosen_rows) < 3:
            extras = sub.sort_values(
                by=["underpowered", "bio_Z", "seed"], ascending=[True, False, True]
            )
            for _, row in extras.iterrows():
                sid = int(row["seed"])
                if sid in dedup:
                    continue
                dedup[sid] = row
                chosen_rows.append(row)
                if len(chosen_rows) >= 3:
                    break

        chosen_rows = chosen_rows[:3]

        label = "S1" if scenario == "S1_step" else "S2"
        for i, row in enumerate(chosen_rows, start=1):
            f = _rebuild_gene_from_row(row, ctx)
            E_obs, _, _ = compute_rsp_profile_from_boolean(
                f=f,
                angles=np.asarray(ctx.theta, dtype=float),
                n_bins=int(bins),
                bin_id=np.asarray(ctx.bin_id, dtype=int),
                bin_counts_total=np.asarray(ctx.bin_counts_total, dtype=int),
            )
            E_obs = np.asarray(E_obs, dtype=float)
            peak_idx = int(np.argmax(np.abs(E_obs)))

            fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=False)
            ax0, ax1 = axes

            plot_embedding_with_foreground(
                ctx.X,
                f,
                ax=ax0,
                title="Embedding foreground",
                s=6.0,
                alpha_bg=0.35,
                alpha_fg=0.8,
            )

            gs = ax1.get_gridspec()
            ax1.remove()
            ax1 = fig.add_subplot(gs[0, 1], projection="polar")
            theta = np.linspace(0.0, 2.0 * np.pi, int(bins), endpoint=False)
            plot_rsp_polar(
                theta,
                E_obs,
                ax=ax1,
                color="#1f77b4",
                linewidth=1.6,
                title="BioRSP profile (polar)",
            )
            ax1.plot(
                [theta[peak_idx], theta[peak_idx]],
                [ax1.get_ylim()[0], ax1.get_ylim()[1]],
                color="#d62728",
                linestyle="--",
                linewidth=1.0,
                label="|E| max",
            )
            if scenario == "S2_gradient":
                theta0 = float(row["theta0"]) % (2.0 * np.pi)
                ax1.plot(
                    [theta0, theta0],
                    [ax1.get_ylim()[0], ax1.get_ylim()[1]],
                    color="#2ca02c",
                    linestyle=":",
                    linewidth=1.0,
                    label="theta0",
                )
            ax1.legend(frameon=False, fontsize=6.5)

            fig.suptitle(
                (
                    f"Experiment E {label} example gene #{i}: beta={_fmt(float(row['beta']))}, pi={_fmt(float(row['pi_target']))}, "
                    f"p={_fmt(float(row['bio_p'])) if np.isfinite(float(row['bio_p'])) else 'nan'}"
                ),
                y=0.99,
            )
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
            savefig(fig, plots_dir / f"examples_{label}_gene{i}.png")
            plt.close(fig)


def make_plots(
    metrics: pd.DataFrame,
    *,
    outdir: Path,
    scenarios: list[str],
    beta_grid: list[float],
    pi_grid: list[float],
    N_ref: int,
    D_ref: int,
    sigma_ref: float,
    pi_ref: float,
    n_perm: int,
    bins: int,
    master_seed: int,
    corr_boot: int,
    contexts_for_examples: dict[tuple[str, int], DatasetContext],
) -> None:
    plot_score_vs_beta(
        metrics,
        outdir=outdir,
        scenarios=scenarios,
        beta_grid=beta_grid,
        pi_grid=pi_grid,
        N_ref=int(N_ref),
        D_ref=int(D_ref),
        sigma_ref=float(sigma_ref),
        n_perm=int(n_perm),
        bins=int(bins),
    )
    plot_spearman_summary(
        metrics,
        outdir=outdir,
        master_seed=int(master_seed),
        corr_boot=int(corr_boot),
    )
    plot_de_instability(
        metrics,
        outdir=outdir,
        N_ref=int(N_ref),
        D_ref=int(D_ref),
        sigma_ref=float(sigma_ref),
        pi_ref=float(pi_ref),
        n_perm=int(n_perm),
    )
    plot_de_stability_compare(
        metrics,
        outdir=outdir,
        N_ref=int(N_ref),
        D_ref=int(D_ref),
        sigma_ref=float(sigma_ref),
        pi_ref=float(pi_ref),
    )
    plot_moran_vs_bio(
        metrics,
        outdir=outdir,
        N_ref=int(N_ref),
        D_ref=int(D_ref),
        sigma_ref=float(sigma_ref),
        pi_ref=float(pi_ref),
        n_perm=int(n_perm),
    )
    plot_examples(
        metrics,
        outdir=outdir,
        bins=int(bins),
        contexts=contexts_for_examples,
        pi_ref=float(pi_ref),
        beta_grid=beta_grid,
    )


def run_validations(metrics: pd.DataFrame, outdir: Path) -> None:
    report_lines: list[str] = []
    has_warning = False

    valid = metrics.loc[~metrics["underpowered"].astype(bool)].copy()
    if valid.empty:
        report_lines.append(
            "No non-underpowered genes available; skipped all validations."
        )
        has_warning = True
    else:
        # S1: DE_score_K2 should correlate strongly with beta for pi>=0.05.
        s1 = valid.loc[
            (valid["scenario"] == "S1_step")
            & (pd.to_numeric(valid["pi_target"], errors="coerce") >= 0.05)
        ]
        for keys, grp in s1.groupby(["N", "D", "sigma_eta", "pi_target"], sort=True):
            sub = grp.loc[pd.to_numeric(grp["beta"], errors="coerce") > 0]
            beta = pd.to_numeric(sub["beta"], errors="coerce").to_numpy(dtype=float)
            score = pd.to_numeric(sub["de_score_K2"], errors="coerce").to_numpy(
                dtype=float
            )
            mask = np.isfinite(beta) & np.isfinite(score)
            if int(np.sum(mask)) < 20:
                continue
            rho, _ = spearmanr(beta[mask], score[mask], nan_policy="omit")
            r = float(rho) if rho is not None else float("nan")
            if not np.isfinite(r) or r < 0.7:
                has_warning = True
                report_lines.append(
                    "S1 validation fail: DE_score_K2 Spearman(beta) < 0.7 "
                    f"at N={keys[0]}, D={keys[1]}, sigma={_fmt(keys[2])}, pi={_fmt(keys[3])}; rho={r:.3f}"
                )

        # S2: BioRSP should correlate with beta at least as well as DE-best.
        s2 = valid.loc[
            (valid["scenario"] == "S2_gradient")
            & (pd.to_numeric(valid["pi_target"], errors="coerce") >= 0.05)
        ]
        for keys, grp in s2.groupby(["N", "D", "sigma_eta", "pi_target"], sort=True):
            sub = grp.loc[pd.to_numeric(grp["beta"], errors="coerce") > 0]
            beta = pd.to_numeric(sub["beta"], errors="coerce").to_numpy(dtype=float)
            zb = pd.to_numeric(sub["bio_Z"], errors="coerce").to_numpy(dtype=float)
            de = pd.to_numeric(sub["de_score_best"], errors="coerce").to_numpy(
                dtype=float
            )
            m1 = np.isfinite(beta) & np.isfinite(zb)
            m2 = np.isfinite(beta) & np.isfinite(de)
            if int(np.sum(m1)) < 20 or int(np.sum(m2)) < 20:
                continue
            rho_b, _ = spearmanr(beta[m1], zb[m1], nan_policy="omit")
            rho_d, _ = spearmanr(beta[m2], de[m2], nan_policy="omit")
            rb = float(rho_b) if rho_b is not None else float("nan")
            rd = float(rho_d) if rho_d is not None else float("nan")
            if np.isfinite(rb) and np.isfinite(rd) and rb + 0.02 < rd:
                has_warning = True
                report_lines.append(
                    "S2 validation fail: BioRSP correlation lower than DE-best "
                    f"at N={keys[0]}, D={keys[1]}, sigma={_fmt(keys[2])}, pi={_fmt(keys[3])}; "
                    f"rho_bio={rb:.3f}, rho_de={rd:.3f}"
                )

        # S2 instability should exceed S1 for matched slices.
        for keys, grp_s1 in valid.loc[
            (valid["scenario"] == "S1_step")
            & (pd.to_numeric(valid["beta"], errors="coerce") > 0)
            & (pd.to_numeric(valid["pi_target"], errors="coerce") >= 0.05)
        ].groupby(["N", "D", "sigma_eta", "pi_target"], sort=True):
            grp_s2 = valid.loc[
                (valid["scenario"] == "S2_gradient")
                & (valid["N"] == keys[0])
                & (valid["D"] == keys[1])
                & (valid["sigma_eta"] == keys[2])
                & (valid["pi_target"] == keys[3])
                & (pd.to_numeric(valid["beta"], errors="coerce") > 0)
            ]
            if grp_s2.empty:
                continue
            m1 = float(pd.to_numeric(grp_s1["de_instability"], errors="coerce").mean())
            m2 = float(pd.to_numeric(grp_s2["de_instability"], errors="coerce").mean())
            if np.isfinite(m1) and np.isfinite(m2) and m2 <= m1 + 0.01:
                has_warning = True
                report_lines.append(
                    "Instability validation warning: S2 instability not higher than S1 "
                    f"at N={keys[0]}, D={keys[1]}, sigma={_fmt(keys[2])}, pi={_fmt(keys[3])}; "
                    f"mean_S1={m1:.3f}, mean_S2={m2:.3f}"
                )

        # beta=0 p-values should be approximately uniform in well-powered regimes.
        for keys, grp in valid.loc[
            (pd.to_numeric(valid["beta"], errors="coerce") == 0.0)
            & (pd.to_numeric(valid["pi_target"], errors="coerce") >= 0.05)
            & (pd.to_numeric(valid["D"], errors="coerce") >= 5)
        ].groupby(["scenario", "N", "D", "sigma_eta", "pi_target"], sort=True):
            pvals = pd.to_numeric(grp["bio_p"], errors="coerce").to_numpy(dtype=float)
            pvals = pvals[np.isfinite(pvals)]
            if pvals.size < 100:
                continue
            ks = kstest(pvals, "uniform")
            if float(ks.pvalue) < 1e-3:
                has_warning = True
                report_lines.append(
                    "Null calibration warning (beta=0): KS p < 1e-3 "
                    f"for scenario={keys[0]}, N={keys[1]}, D={keys[2]}, sigma={_fmt(keys[3])}, pi={_fmt(keys[4])}; "
                    f"ks_p={float(ks.pvalue):.3g}, n={pvals.size}"
                )

    if not report_lines:
        report_lines.append("All soft validation checks passed.")

    dbg = outdir / "results" / "validation_debug_report.txt"
    dbg.write_text("\n".join(report_lines), encoding="utf-8")

    if has_warning:
        print(f"Validation warnings detected. See {dbg}", flush=True)
    else:
        print("Validation checks passed without warnings.", flush=True)


def run_experiment(args: argparse.Namespace) -> None:
    test_cfg = (
        get_testmode_config(int(args.master_seed)) if bool(args.test_mode) else None
    )
    testmode_banner(bool(args.test_mode), test_cfg)

    outdir = resolve_outdir(args.outdir, bool(args.test_mode))
    outdir = ensure_dir(outdir)
    results_dir = ensure_dir(outdir / "results")
    ensure_dir(outdir / "plots")

    if bool(args.test_mode):
        N_grid = [int(test_cfg.N)]
        D_grid = [5, 10]
        sigma_eta_grid = [0.0, 0.4]
        pi_grid = [0.05, 0.2]
        beta_grid = [0.0, 0.5, 1.0]
        genes_per_condition = max(
            10, int(test_cfg.G) // (len(pi_grid) * len(beta_grid) * 2)
        )
        n_perm = int(test_cfg.n_perm)
        n_master_seeds = 1
    else:
        N_grid = [int(x) for x in (args.N_grid if args.N_grid else [args.N])]
        D_grid = [int(x) for x in args.D_grid]
        sigma_eta_grid = [float(x) for x in args.sigma_eta_grid]
        pi_grid = [float(x) for x in args.pi_grid]
        beta_grid = [float(x) for x in args.beta_grid]
        genes_per_condition = int(args.genes_per_condition)
        n_perm = int(args.n_perm)
        n_master_seeds = int(args.n_master_seeds)

    k_values = sorted(set(int(x) for x in args.k_values))
    if not {2, 4, 8, 16}.issubset(set(k_values)):
        warnings.warn(
            "k_values does not include all of {2,4,8,16}; DE score columns for missing K values will be NaN.",
            RuntimeWarning,
            stacklevel=2,
        )

    min_perm_eff = int(args.min_perm)
    if int(n_perm) < min_perm_eff:
        warnings.warn(
            f"n_perm ({n_perm}) < min_perm ({min_perm_eff}); lowering min_perm gate to {n_perm} for this run.",
            RuntimeWarning,
            stacklevel=2,
        )
        min_perm_eff = int(n_perm)

    seed_values = [int(args.master_seed) + i for i in range(int(n_master_seeds))]
    run_id = timestamped_run_id(prefix="expE")

    total_genes = (
        len(seed_values)
        * len(SCENARIOS)
        * len(N_grid)
        * len(D_grid)
        * len(sigma_eta_grid)
        * len(pi_grid)
        * len(beta_grid)
        * int(genes_per_condition)
    )

    cfg = {
        "experiment": "expE_gradient_vs_step_DE",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit_hash(cwd=REPO_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "master_seed": int(args.master_seed),
        "seed_values": seed_values,
        "N_grid": [int(x) for x in N_grid],
        "D_grid": [int(x) for x in D_grid],
        "sigma_eta_grid": [float(x) for x in sigma_eta_grid],
        "pi_grid": [float(x) for x in pi_grid],
        "beta_grid": [float(x) for x in beta_grid],
        "scenarios": list(SCENARIOS),
        "genes_per_condition": int(genes_per_condition),
        "n_perm": int(n_perm),
        "bins": int(args.bins),
        "k_nn": int(args.k_nn),
        "k_values": [int(x) for x in k_values],
        "n_master_seeds": int(n_master_seeds),
        "corr_boot": int(args.corr_boot),
        "underpowered": {
            "p_min": 0.005,
            "min_fg_total": int(args.min_fg_total),
            "min_fg_per_donor": int(args.min_fg_per_donor),
            "min_bg_per_donor": int(args.min_bg_per_donor),
            "d_eff_min": int(args.d_eff_min),
            "min_perm": int(min_perm_eff),
        },
        "test_mode": bool(args.test_mode),
        "total_gene_instances": int(total_genes),
    }
    write_config(outdir, cfg)

    print(
        (
            "Running ExpE with "
            f"{total_genes} genes, n_perm={n_perm}, bins={int(args.bins)}, "
            f"N_grid={N_grid}, D_grid={D_grid}, sigma_grid={sigma_eta_grid}"
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    contexts_for_examples: dict[tuple[str, int], DatasetContext] = {}

    total_start = time.time()
    gene_counter = 0

    N_ref = int(_select_refs([int(x) for x in N_grid], 20000))
    D_ref = int(_select_refs([int(x) for x in D_grid], 10))
    sigma_ref = float(_select_refs([float(x) for x in sigma_eta_grid], 0.4))
    pi_ref = float(_select_refs([float(x) for x in pi_grid], 0.2))

    for seed_run in seed_values:
        for scenario in SCENARIOS:
            for N in N_grid:
                for D in D_grid:
                    for sigma_eta in sigma_eta_grid:
                        print(
                            f"Dataset: seed={seed_run}, scenario={scenario}, N={N}, D={D}, sigma={_fmt(sigma_eta)}",
                            flush=True,
                        )
                        context, W_knn, cluster_labels = _build_dataset(
                            scenario=str(scenario),
                            N=int(N),
                            D=int(D),
                            sigma_eta=float(sigma_eta),
                            seed_run=int(seed_run),
                            master_seed=int(args.master_seed),
                            n_bins=int(args.bins),
                            k_nn=int(args.k_nn),
                            k_values=k_values,
                        )

                        if (
                            int(seed_run) == int(seed_values[0])
                            and int(N) == int(N_ref)
                            and int(D) == int(D_ref)
                            and abs(float(sigma_eta) - float(sigma_ref)) < 1e-12
                        ):
                            contexts_for_examples[(str(scenario), int(seed_run))] = (
                                context
                            )

                        for pi_target in pi_grid:
                            for beta in beta_grid:
                                for rep in range(int(genes_per_condition)):
                                    seed_gene = stable_seed(
                                        int(args.master_seed),
                                        "expE",
                                        "gene",
                                        int(seed_run),
                                        str(scenario),
                                        int(N),
                                        int(D),
                                        float(sigma_eta),
                                        float(pi_target),
                                        float(beta),
                                        int(rep),
                                    )
                                    rng_gene = rng_from_seed(int(seed_gene))
                                    theta0 = float(rng_gene.uniform(0.0, 2.0 * np.pi))

                                    f = _simulate_gene(
                                        scenario=str(scenario),
                                        pi_target=float(pi_target),
                                        beta=float(beta),
                                        theta0=theta0,
                                        theta=context.theta,
                                        donor_eta_cell=context.donor_eta_cell,
                                        step_labels=context.step_labels,
                                        rng=rng_gene,
                                    )

                                    prev_obs = float(np.mean(f))
                                    n_fg_total = int(np.sum(f))

                                    power = evaluate_underpowered(
                                        donor_ids=context.donor_ids,
                                        f=f,
                                        n_perm=int(n_perm),
                                        p_min=0.005,
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
                                    used_donor_stratified = False

                                    if not underpowered:
                                        seed_perm = stable_seed(
                                            int(args.master_seed),
                                            "expE",
                                            "perm",
                                            int(seed_run),
                                            str(scenario),
                                            int(N),
                                            int(D),
                                            float(sigma_eta),
                                            float(pi_target),
                                            float(beta),
                                            int(rep),
                                        )
                                        perm = perm_null_T(
                                            f=f,
                                            angles=context.theta,
                                            donor_ids=context.donor_ids,
                                            n_bins=int(args.bins),
                                            n_perm=int(n_perm),
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
                                        used_donor_stratified = bool(
                                            perm.get("used_donor_stratified", False)
                                        )
                                        bio_Z = _robust_z(
                                            float(bio_T),
                                            np.asarray(perm["null_T"], dtype=float),
                                        )

                                    try:
                                        moran_I = float(
                                            morans_i(
                                                f.astype(float),
                                                W_knn,
                                                row_standardize=True,
                                            )
                                        )
                                    except Exception:
                                        moran_I = float("nan")

                                    de_scores: dict[int, float] = {}
                                    for K in k_values:
                                        de_scores[int(K)] = _de_score_range(
                                            f, cluster_labels[int(K)], int(K)
                                        )

                                    ordered_scores = [
                                        de_scores[k] for k in k_values if k in de_scores
                                    ]
                                    de_best = (
                                        float(np.max(ordered_scores))
                                        if ordered_scores
                                        else float("nan")
                                    )
                                    de_instability = (
                                        float(np.std(ordered_scores))
                                        if ordered_scores
                                        else float("nan")
                                    )

                                    rows.append(
                                        {
                                            "run_id": run_id,
                                            "seed": int(seed_gene),
                                            "seed_run": int(seed_run),
                                            "scenario": str(scenario),
                                            "N": int(N),
                                            "D": int(D),
                                            "sigma_eta": float(sigma_eta),
                                            "pi_target": float(pi_target),
                                            "beta": float(beta),
                                            "theta0": float(theta0),
                                            "prev_obs": float(prev_obs),
                                            "n_fg_total": int(n_fg_total),
                                            "D_eff": int(power["D_eff"]),
                                            "underpowered": bool(underpowered),
                                            "bio_T": float(bio_T),
                                            "bio_p": float(bio_p),
                                            "bio_Z": float(bio_Z),
                                            "moran_I": float(moran_I),
                                            "de_score_K2": float(
                                                de_scores.get(2, np.nan)
                                            ),
                                            "de_score_K4": float(
                                                de_scores.get(4, np.nan)
                                            ),
                                            "de_score_K8": float(
                                                de_scores.get(8, np.nan)
                                            ),
                                            "de_score_K16": float(
                                                de_scores.get(16, np.nan)
                                            ),
                                            "de_score_best": float(de_best),
                                            "de_instability": float(de_instability),
                                            "used_donor_stratified": bool(
                                                used_donor_stratified
                                            ),
                                        }
                                    )

                                    gene_counter += 1
                                    if (
                                        gene_counter % int(args.progress_every) == 0
                                        or gene_counter == total_genes
                                    ):
                                        elapsed = time.time() - total_start
                                        rate = (
                                            gene_counter / elapsed
                                            if elapsed > 0
                                            else float("nan")
                                        )
                                        print(
                                            f"  progress: {gene_counter}/{total_genes} genes ({rate:.2f} genes/s)",
                                            flush=True,
                                        )

    metrics = pd.DataFrame(rows)
    summary = summarize(
        metrics, corr_boot=int(args.corr_boot), master_seed=int(args.master_seed)
    )

    atomic_write_csv(results_dir / "metrics_long.csv", metrics)
    atomic_write_csv(results_dir / "summary.csv", summary)

    _set_plot_style()
    make_plots(
        metrics,
        outdir=outdir,
        scenarios=list(SCENARIOS),
        beta_grid=[float(x) for x in beta_grid],
        pi_grid=[float(x) for x in pi_grid],
        N_ref=int(N_ref),
        D_ref=int(D_ref),
        sigma_ref=float(sigma_ref),
        pi_ref=float(pi_ref),
        n_perm=int(n_perm),
        bins=int(args.bins),
        master_seed=int(args.master_seed),
        corr_boot=int(args.corr_boot),
        contexts_for_examples=contexts_for_examples,
    )

    run_validations(metrics, outdir=outdir)

    elapsed = time.time() - total_start
    print(
        (
            f"Completed ExpE in {elapsed/60.0:.2f} min. "
            f"rows={metrics.shape[0]}, non_underpowered={int((~metrics['underpowered']).sum())}"
        ),
        flush=True,
    )
    print(f"Outputs written to: {outdir}", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Simulation Experiment E: continuous gradient vs cluster-separable step change, "
            "comparing BioRSP vs Moran's I vs cluster-DE proxy."
        )
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/simulations/expE_gradient_vs_step_DE",
        help="Output directory.",
    )
    parser.add_argument("--master_seed", type=int, default=123, help="Master seed.")
    parser.add_argument(
        "--n_master_seeds",
        type=int,
        default=1,
        help="Number of consecutive seeds to pool.",
    )

    parser.add_argument(
        "--N", type=int, default=20000, help="Default N when --N_grid is not provided."
    )
    parser.add_argument(
        "--N_grid",
        type=int,
        nargs="*",
        default=None,
        help="Optional N grid; overrides --N if provided.",
    )
    parser.add_argument(
        "--D_grid", type=int, nargs="+", default=DEFAULT_D_GRID, help="Donor grid."
    )
    parser.add_argument(
        "--sigma_eta_grid",
        type=float,
        nargs="+",
        default=DEFAULT_SIGMA_ETA_GRID,
        help="Donor random-effect sigma grid.",
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
        help="Genes per (scenario,N,D,sigma,pi,beta).",
    )
    parser.add_argument(
        "--n_perm", type=int, default=500, help="Permutation count for BioRSP p-values."
    )
    parser.add_argument(
        "--bins", type=int, default=36, help="Angular bin count for BioRSP profile."
    )
    parser.add_argument(
        "--k_nn", type=int, default=15, help="k for kNN graph used in Moran's I."
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=DEFAULT_K_GRID,
        help="k-means K values for DE resolution sweep.",
    )

    parser.add_argument(
        "--corr_boot",
        type=int,
        default=200,
        help="Bootstrap count for Spearman CI summaries.",
    )

    parser.add_argument(
        "--min_fg_total",
        type=int,
        default=50,
        help="Underpowered min total foreground count.",
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
        "--min_perm",
        type=int,
        default=200,
        help="Underpowered min permutation threshold.",
    )

    parser.add_argument(
        "--progress_every",
        type=int,
        default=200,
        help="Progress logging frequency in genes.",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run a tiny deterministic subset for CI/local sanity checks.",
    )
    add_common_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    warnings.simplefilter("default", RuntimeWarning)
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if bool(getattr(args, "dry_run", False)):
        print("dry_run=True: skipping execution for legacy runner.", flush=True)
        return 0
    if args.n_perm <= 0 or args.bins <= 0 or args.genes_per_condition <= 0:
        raise ValueError("n_perm, bins, and genes_per_condition must be positive.")
    run_ctx = prepare_legacy_run(args, "expE_gradient_vs_step_DE", __file__)
    run_experiment(args)
    finalize_legacy_run(run_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
