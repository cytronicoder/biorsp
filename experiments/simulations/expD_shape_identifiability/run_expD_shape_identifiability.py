#!/usr/bin/env python3
"""Simulation Experiment D: shape identifiability under noise and bin/smoothing sensitivity."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr

# Safe non-interactive plotting setup.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-biorsp-expD")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache-biorsp-expD")
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from biorsp.peaks import find_circular_peaks
from biorsp.permutation import perm_null_T
from biorsp.power import evaluate_underpowered
from biorsp.rsp import compute_rsp_profile_from_boolean
from biorsp.smoothing import circular_moving_average
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
    write_config,
)
from experiments.simulations._shared.models import apply_dropout_noise
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

TRUTH_CLASSES = ["UNIMODAL", "BIMODAL", "TRIMODAL", "PATCHY_4LOBE"]
TRUTH_K = {
    "UNIMODAL": 1,
    "BIMODAL": 2,
    "TRIMODAL": 3,
    "PATCHY_4LOBE": 4,
}
TRUTH_BUCKET = {
    "UNIMODAL": "UNIMODAL",
    "BIMODAL": "BIMODAL",
    "TRIMODAL": "MULTIMODAL_PATCHY",
    "PATCHY_4LOBE": "MULTIMODAL_PATCHY",
}
STRUCTURAL_BUCKETS = ["UNIMODAL", "BIMODAL", "MULTIMODAL_PATCHY"]
PRED_CLASSES = [
    "UNIMODAL",
    "BIMODAL",
    "MULTIMODAL_PATCHY",
    "UBIQUITOUS_OR_NULL",
    "UNCERTAIN_SHAPE",
    "UNCERTAIN_UNDERPOWERED",
]


def _fmt(x: float) -> str:
    return f"{float(x):.3f}".rstrip("0").rstrip(".")


def _tok(x: float | int | str) -> str:
    return str(x).replace(".", "p")


def _circular_distance(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (np.asarray(a, dtype=float) - float(b)))))


def _normalize_signal(g: np.ndarray) -> np.ndarray:
    arr = np.asarray(g, dtype=float)
    arr = arr - float(np.mean(arr))
    sd = float(np.std(arr))
    if sd > 0:
        arr = arr / sd
    return arr


def shape_signal(
    theta: np.ndarray, theta0: float, truth_class: str, patchy_width: float
) -> np.ndarray:
    th = np.asarray(theta, dtype=float)
    cls = str(truth_class)
    t0 = float(theta0)
    if cls == "UNIMODAL":
        return _normalize_signal(np.cos(th - t0))
    if cls == "BIMODAL":
        return _normalize_signal(np.cos(2.0 * (th - t0)))
    if cls == "TRIMODAL":
        return _normalize_signal(np.cos(3.0 * (th - t0)))
    if cls == "PATCHY_4LOBE":
        mus = t0 + np.arange(4, dtype=float) * (2.0 * np.pi / 4.0)
        lobes = []
        s = max(1e-3, float(patchy_width))
        for mu in mus:
            d = _circular_distance(th, float(mu))
            lobes.append(np.exp(-0.5 * (d**2) / (s**2)))
        g = np.max(np.vstack(lobes), axis=0)
        return _normalize_signal(g)
    raise ValueError(f"Unknown truth_class '{truth_class}'.")


def _apply_mode_profile(E: np.ndarray, mode: str, smooth_w: int) -> np.ndarray:
    arr = np.asarray(E, dtype=float)
    if str(mode) == "smoothed":
        return circular_moving_average(arr, int(smooth_w))
    return arr.copy()


def _k_bucket(k_hat: int) -> int:
    k = int(k_hat)
    if k == 1:
        return 1
    if k == 2:
        return 2
    if k >= 3:
        return 3
    return 0


def _pred_bucket(pred_class: str) -> str:
    p = str(pred_class)
    if p in STRUCTURAL_BUCKETS:
        return p
    return "OTHER"


def _peak_and_coverage_features(
    E_obs: np.ndarray, null_E: np.ndarray
) -> tuple[int, float, float, np.ndarray, float]:
    e_obs = np.asarray(E_obs, dtype=float).ravel()
    nE = np.asarray(null_E, dtype=float)
    if nE.ndim != 2 or nE.shape[1] != e_obs.size:
        raise ValueError("null_E must have shape (n_perm, n_bins) matching E_obs.")

    tau = float(np.quantile(nE, 0.95))
    c_hat = float(np.mean(e_obs > tau))

    max_prom = np.zeros(nE.shape[0], dtype=float)
    for i in range(nE.shape[0]):
        peaks_i = find_circular_peaks(nE[i, :], prominence_threshold=0.0)
        prom_i = np.asarray(peaks_i["prominences"], dtype=float)
        max_prom[i] = float(np.max(prom_i)) if prom_i.size > 0 else 0.0
    prom_thresh = float(np.quantile(max_prom, 0.95))

    peaks_obs = find_circular_peaks(e_obs, prominence_threshold=prom_thresh)
    idx_obs = np.asarray(peaks_obs["indices"], dtype=int)
    k_hat = int(idx_obs.size)
    return k_hat, c_hat, prom_thresh, idx_obs, tau


def _fourier_ratios(E_obs: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(E_obs, dtype=float).ravel()
    arr = arr - float(np.mean(arr))
    fft = np.fft.rfft(arr)
    power = np.abs(fft) ** 2
    if power.size <= 1:
        return float("nan"), float("nan"), float("nan")
    denom = float(np.sum(power[1:]))
    if denom <= 0.0:
        return 0.0, 0.0, 0.0

    def _safe(idx: int) -> float:
        return float(power[idx] / denom) if idx < power.size else 0.0

    return _safe(1), _safe(2), _safe(3)


def _bootstrap_stability(
    *,
    f: np.ndarray,
    theta: np.ndarray,
    bin_id: np.ndarray,
    donor_groups: list[np.ndarray],
    n_bins: int,
    mode: str,
    smooth_w: int,
    prom_thresh: float,
    k_obs: int,
    n_boot: int,
    seed: int,
) -> float:
    if int(n_boot) <= 0:
        return float("nan")

    rng = rng_from_seed(int(seed))
    obs_bucket = _k_bucket(int(k_obs))
    buckets: list[int] = []

    for _ in range(int(n_boot)):
        parts = []
        for grp in donor_groups:
            if grp.size == 0:
                continue
            parts.append(grp[rng.integers(0, grp.size, size=grp.size)])
        if not parts:
            continue
        idx = np.concatenate(parts).astype(int)
        f_boot = np.asarray(f, dtype=bool)[idx]
        if int(np.sum(f_boot)) == 0 or int(np.sum(~f_boot)) == 0:
            continue

        bid = np.asarray(bin_id, dtype=int)[idx]
        counts = np.bincount(bid, minlength=int(n_bins)).astype(np.int64)
        E_boot, _, _ = compute_rsp_profile_from_boolean(
            f=f_boot,
            angles=np.asarray(theta, dtype=float)[idx],
            n_bins=int(n_bins),
            bin_id=bid,
            bin_counts_total=counts,
        )
        E_boot = _apply_mode_profile(E_boot, mode=mode, smooth_w=int(smooth_w))
        peaks_b = find_circular_peaks(E_boot, prominence_threshold=float(prom_thresh))
        k_boot = int(np.asarray(peaks_b["indices"], dtype=int).size)
        buckets.append(_k_bucket(k_boot))

    if not buckets:
        return float("nan")
    b = np.asarray(buckets, dtype=int)
    return float(np.mean(b == int(obs_bucket)))


def _simulate_gene(
    *,
    theta: np.ndarray,
    donor_eta: np.ndarray,
    pi_target: float,
    beta: float,
    theta0: float,
    truth_class: str,
    patchy_width: float,
    rng: np.random.Generator,
) -> np.ndarray:
    alpha = float(logit(np.clip(float(pi_target), 1e-12, 1.0 - 1e-12)))
    g = shape_signal(
        theta,
        theta0=float(theta0),
        truth_class=truth_class,
        patchy_width=float(patchy_width),
    )
    logits = alpha + float(beta) * g + np.asarray(donor_eta, dtype=float)
    p = expit(logits)
    return (rng.random(p.size) < p).astype(bool)


def _select_rep(value: float, grid: list[float]) -> float:
    arr = np.asarray(grid, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(value))))
    return float(arr[idx])


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


def summarize(metrics: pd.DataFrame, alpha_sig: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "N",
        "D",
        "sigma_eta",
        "truth_class",
        "pi_target",
        "beta",
        "dropout_noise",
        "bins_B",
        "mode",
        "smooth_w",
    ]

    for keys, grp in metrics.groupby(group_cols, sort=True):
        N, D, sigma, truth_class, pi, beta, drop, bins, mode, w = keys
        n_genes = int(grp.shape[0])
        valid = grp.loc[~grp["underpowered"].astype(bool)]
        n_valid = int(valid.shape[0])
        frac_under = float(grp["underpowered"].mean())

        true_bucket = TRUTH_BUCKET[str(truth_class)]
        pred_bucket = valid["pred_bucket"].astype(str)
        acc_overall = (
            float(np.mean(pred_bucket == true_bucket)) if n_valid > 0 else float("nan")
        )

        sig = valid.loc[
            pd.to_numeric(valid["p_T"], errors="coerce") <= float(alpha_sig)
        ]
        n_sig = int(sig.shape[0])
        acc_sig = (
            float(np.mean(sig["pred_bucket"].astype(str) == true_bucket))
            if n_sig > 0
            else float("nan")
        )

        rows.append(
            {
                "N": int(N),
                "D": int(D),
                "sigma_eta": float(sigma),
                "truth_class": str(truth_class),
                "truth_bucket": str(true_bucket),
                "pi_target": float(pi),
                "beta": float(beta),
                "dropout_noise": float(drop),
                "bins_B": int(bins),
                "mode": str(mode),
                "smooth_w": int(w),
                "n_genes": n_genes,
                "n_non_underpowered": n_valid,
                "frac_underpowered": frac_under,
                "accuracy_overall": acc_overall,
                "accuracy_conditional_on_significant": acc_sig,
                "mean_prev_obs": float(
                    pd.to_numeric(grp["prev_obs"], errors="coerce").mean()
                ),
                "mean_D_eff": float(
                    pd.to_numeric(grp["D_eff"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def write_confusion_csvs(metrics: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    group_cols = [
        "N",
        "D",
        "sigma_eta",
        "pi_target",
        "beta",
        "dropout_noise",
        "bins_B",
        "mode",
        "smooth_w",
    ]

    for keys, grp in metrics.groupby(group_cols, sort=True):
        N, D, sigma, pi, beta, drop, bins, mode, w = keys
        valid = grp.loc[~grp["underpowered"].astype(bool)].copy()
        if valid.empty:
            continue
        ctab = pd.crosstab(
            valid["truth_bucket"].astype(str),
            valid["pred_class"].astype(str),
            dropna=False,
        ).reindex(index=STRUCTURAL_BUCKETS, columns=PRED_CLASSES, fill_value=0)

        rows = []
        for t in STRUCTURAL_BUCKETS:
            row_sum = int(ctab.loc[t, :].sum())
            for p in PRED_CLASSES:
                c = int(ctab.loc[t, p])
                rows.append(
                    {
                        "N": int(N),
                        "D": int(D),
                        "sigma_eta": float(sigma),
                        "pi_target": float(pi),
                        "beta": float(beta),
                        "dropout_noise": float(drop),
                        "bins_B": int(bins),
                        "mode": str(mode),
                        "smooth_w": int(w),
                        "true_class": t,
                        "pred_class": p,
                        "count": c,
                        "row_frac": float(c / row_sum) if row_sum > 0 else float("nan"),
                    }
                )
        df = pd.DataFrame(rows)
        fname = (
            f"confusion_N{int(N)}_D{int(D)}_sigma{_tok(sigma)}_pi{_tok(pi)}_"
            f"beta{_tok(beta)}_drop{_tok(drop)}_B{int(bins)}_{mode}_w{int(w)}.csv"
        )
        atomic_write_csv(out_dir / fname, df)


def _plot_confusion_matrix(
    ax: plt.Axes,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    title: str,
) -> None:
    ctab = pd.crosstab(pd.Series(true_labels), pd.Series(pred_labels), dropna=False)
    ctab = ctab.reindex(index=STRUCTURAL_BUCKETS, columns=PRED_CLASSES, fill_value=0)
    counts = ctab.to_numpy(dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_frac = np.divide(
        counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0
    )

    im = ax.imshow(row_frac, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(PRED_CLASSES)))
    ax.set_xticklabels(PRED_CLASSES, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(STRUCTURAL_BUCKETS)))
    ax.set_yticklabels(STRUCTURAL_BUCKETS)
    ax.set_title(title)
    for i in range(row_frac.shape[0]):
        for j in range(row_frac.shape[1]):
            txt = f"{row_frac[i,j]:.2f}\n({int(counts[i,j])})"
            color = "white" if row_frac[i, j] > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=color)
    return im


def make_plots(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    outdir: Path,
    alpha_sig: float,
    n_perm: int,
    N_ref: int,
    D_ref: int,
    sigma_ref: float,
    bins_ref: int,
    mode_ref: str,
    w_ref: int,
    pi_grid: list[float],
    beta_grid: list[float],
    dropout_grid: list[float],
    bins_grid: list[int],
    w_grid: list[int],
    modes: list[str],
    example_profiles: dict[str, list[dict[str, Any]]],
) -> None:
    plots_dir = ensure_dir(outdir / "plots")

    # C1) Confusion matrices for representative B/mode/w across pi/drop/beta slices.
    beta_focus = [
        b
        for b in [0.5, 1.0, 1.25]
        if any(abs(float(b) - float(x)) < 1e-9 for x in beta_grid)
    ]
    if not beta_focus and beta_grid:
        beta_focus = [float(beta_grid[min(len(beta_grid) - 1, 0)])]

    for pi in pi_grid:
        for drop in dropout_grid:
            for beta in beta_focus:
                sub = metrics.loc[
                    (metrics["N"] == int(N_ref))
                    & (metrics["D"] == int(D_ref))
                    & (metrics["sigma_eta"] == float(sigma_ref))
                    & (metrics["pi_target"] == float(pi))
                    & (metrics["beta"] == float(beta))
                    & (metrics["dropout_noise"] == float(drop))
                    & (metrics["bins_B"] == int(bins_ref))
                    & (metrics["mode"] == str(mode_ref))
                    & (metrics["smooth_w"] == int(w_ref))
                    & (~metrics["underpowered"].astype(bool))
                ]
                if sub.empty:
                    continue
                fig, ax = plt.subplots(figsize=(8.2, 3.9), constrained_layout=False)
                im = _plot_confusion_matrix(
                    ax,
                    sub["truth_bucket"].astype(str).to_numpy(),
                    sub["pred_class"].astype(str).to_numpy(),
                    (
                        "ExpD confusion (excluding underpowered)\n"
                        f"N={N_ref}, D={D_ref}, sigma={_fmt(sigma_ref)}, pi={_fmt(pi)}, beta={_fmt(beta)}, "
                        f"drop={_fmt(drop)}, B={bins_ref}, mode={mode_ref}, w={w_ref}, n_perm={n_perm}"
                    ),
                )
                cbar = fig.colorbar(im, ax=ax, pad=0.02)
                cbar.set_label("row-normalized fraction")
                fig.tight_layout()
                savefig(
                    fig,
                    plots_dir
                    / f"confusion_pi{_tok(pi)}_drop{_tok(drop)}_beta{_tok(beta)}.png",
                )
                plt.close(fig)

    # C2) Phase diagrams for class recall vs beta/dropout at representative condition, pi=0.2.
    pi_phase = _select_rep(0.2, pi_grid)
    for cls in STRUCTURAL_BUCKETS:
        mat = np.full((len(dropout_grid), len(beta_grid)), np.nan, dtype=float)
        nmat = np.full((len(dropout_grid), len(beta_grid)), np.nan, dtype=float)
        for i, drop in enumerate(dropout_grid):
            for j, beta in enumerate(beta_grid):
                sub = metrics.loc[
                    (metrics["N"] == int(N_ref))
                    & (metrics["D"] == int(D_ref))
                    & (metrics["sigma_eta"] == float(sigma_ref))
                    & (metrics["pi_target"] == float(pi_phase))
                    & (metrics["beta"] == float(beta))
                    & (metrics["dropout_noise"] == float(drop))
                    & (metrics["bins_B"] == int(bins_ref))
                    & (metrics["mode"] == str(mode_ref))
                    & (metrics["smooth_w"] == int(w_ref))
                    & (~metrics["underpowered"].astype(bool))
                    & (metrics["truth_bucket"] == cls)
                ]
                n = int(sub.shape[0])
                nmat[i, j] = float(n)
                if n == 0:
                    continue
                mat[i, j] = float(np.mean(sub["pred_bucket"].astype(str) == cls))

        fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=False)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(beta_grid)))
        ax.set_xticklabels([_fmt(b) for b in beta_grid])
        ax.set_yticks(np.arange(len(dropout_grid)))
        ax.set_yticklabels([_fmt(d) for d in dropout_grid])
        ax.set_xlabel("beta")
        ax.set_ylabel("dropout")
        ax.set_title(
            (
                "ExpD phase accuracy map\n"
                f"class={cls}, pi={_fmt(pi_phase)}, N={N_ref}, D={D_ref}, sigma={_fmt(sigma_ref)}, "
                f"B={bins_ref}, {mode_ref}, w={w_ref}"
            )
        )
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                n = nmat[i, j]
                txt = "NA" if not np.isfinite(v) else f"{v:.2f}\nn={int(n)}"
                color = "white" if np.isfinite(v) and v > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=color)
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("class recall")
        fig.tight_layout()
        savefig(
            fig,
            plots_dir
            / f"phase_accuracy_class{cls}_pi{_tok(pi_phase)}_B{int(bins_ref)}_{mode_ref}_w{int(w_ref)}.png",
        )
        plt.close(fig)

    # C3) Feature separation diagnostics on representative condition.
    beta_feat = _select_rep(1.0, beta_grid)
    drop_feat = _select_rep(0.2, dropout_grid)
    sub_feat = metrics.loc[
        (metrics["N"] == int(N_ref))
        & (metrics["D"] == int(D_ref))
        & (metrics["sigma_eta"] == float(sigma_ref))
        & (metrics["pi_target"] == float(pi_phase))
        & (metrics["beta"] == float(beta_feat))
        & (metrics["dropout_noise"] == float(drop_feat))
        & (metrics["bins_B"] == int(bins_ref))
        & (metrics["mode"] == str(mode_ref))
        & (metrics["smooth_w"] == int(w_ref))
        & (~metrics["underpowered"].astype(bool))
    ].copy()

    if not sub_feat.empty:
        colors = {
            "UNIMODAL": "#1f77b4",
            "BIMODAL": "#ff7f0e",
            "TRIMODAL": "#2ca02c",
            "PATCHY_4LOBE": "#d62728",
        }

        fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=False)
        for cls in TRUTH_CLASSES:
            d = sub_feat.loc[sub_feat["truth_class"] == cls]
            if d.empty:
                continue
            ax.scatter(
                pd.to_numeric(d["K_hat"], errors="coerce"),
                pd.to_numeric(d["C_hat"], errors="coerce"),
                s=15,
                alpha=0.75,
                color=colors.get(cls, "#444444"),
                label=cls,
            )
        ax.set_xlabel("K_hat")
        ax.set_ylabel("C_hat")
        ax.set_title(
            (
                "ExpD feature scatter K_hat vs C_hat\n"
                f"pi={_fmt(pi_phase)}, beta={_fmt(beta_feat)}, drop={_fmt(drop_feat)}, B={bins_ref}, {mode_ref}, w={w_ref}"
            )
        )
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        savefig(fig, plots_dir / "feature_scatter_KC.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=False)
        for cls in TRUTH_CLASSES:
            d = sub_feat.loc[sub_feat["truth_class"] == cls]
            if d.empty:
                continue
            ax.scatter(
                pd.to_numeric(d["S1"], errors="coerce"),
                pd.to_numeric(d["S2"], errors="coerce"),
                s=15,
                alpha=0.75,
                color=colors.get(cls, "#444444"),
                label=cls,
            )
        ax.set_xlabel("S1")
        ax.set_ylabel("S2")
        ax.set_title(
            (
                "ExpD feature scatter S1 vs S2\n"
                f"pi={_fmt(pi_phase)}, beta={_fmt(beta_feat)}, drop={_fmt(drop_feat)}, B={bins_ref}, {mode_ref}, w={w_ref}"
            )
        )
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        savefig(fig, plots_dir / "feature_scatter_S1S2.png")
        plt.close(fig)

        # K_hat distribution by truth class.
        kh_bins = [0, 1, 2, 3]
        labels = ["0", "1", "2", "3+"]
        fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=False)
        x = np.arange(len(TRUTH_CLASSES))
        bottom = np.zeros(len(TRUTH_CLASSES), dtype=float)
        for k_idx, k in enumerate(kh_bins):
            heights = []
            for cls in TRUTH_CLASSES:
                d = sub_feat.loc[sub_feat["truth_class"] == cls, "K_hat"]
                vals = pd.to_numeric(d, errors="coerce").to_numpy(dtype=float)
                if vals.size == 0:
                    heights.append(0.0)
                    continue
                if k < 3:
                    heights.append(float(np.mean(vals == float(k))))
                else:
                    heights.append(float(np.mean(vals >= 3.0)))
            h = np.asarray(heights, dtype=float)
            ax.bar(x, h, bottom=bottom, label=labels[k_idx], alpha=0.85)
            bottom += h
        ax.set_xticks(x)
        ax.set_xticklabels(TRUTH_CLASSES, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("fraction")
        ax.set_title("ExpD K_hat distribution by truth class")
        ax.legend(title="K_hat", frameon=False, ncol=4)
        fig.tight_layout()
        savefig(fig, plots_dir / "Khat_distribution.png")
        plt.close(fig)

    # C4) Sensitivity to bins/smoothing.
    beta_sens = _select_rep(1.0, beta_grid)
    drop_sens = _select_rep(0.2, dropout_grid)
    fig, ax = plt.subplots(figsize=(7.2, 4.1), constrained_layout=False)
    for mode in modes:
        for w in w_grid:
            ys = []
            ns = []
            for b in bins_grid:
                sub = metrics.loc[
                    (metrics["N"] == int(N_ref))
                    & (metrics["D"] == int(D_ref))
                    & (metrics["sigma_eta"] == float(sigma_ref))
                    & (metrics["pi_target"] == float(pi_phase))
                    & (metrics["beta"] == float(beta_sens))
                    & (metrics["dropout_noise"] == float(drop_sens))
                    & (metrics["bins_B"] == int(b))
                    & (metrics["mode"] == str(mode))
                    & (metrics["smooth_w"] == int(w))
                    & (~metrics["underpowered"].astype(bool))
                ]
                n = int(sub.shape[0])
                ns.append(n)
                if n == 0:
                    ys.append(np.nan)
                else:
                    ys.append(
                        float(
                            np.mean(
                                sub["pred_bucket"].astype(str)
                                == sub["truth_bucket"].astype(str)
                            )
                        )
                    )
            ax.plot(
                bins_grid, ys, marker="o", linewidth=1.4, label=f"{mode}, w={int(w)}"
            )
            for xx, yy, nn in zip(bins_grid, ys, ns):
                if np.isfinite(yy):
                    ax.text(
                        float(xx),
                        float(yy) + 0.02,
                        f"n={int(nn)}",
                        fontsize=6,
                        ha="center",
                    )
    ax.set_xlabel("bins_B")
    ax.set_ylabel("overall accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    ax.set_title(
        (
            "ExpD accuracy vs bins/smoothing\n"
            f"N={N_ref}, D={D_ref}, sigma={_fmt(sigma_ref)}, pi={_fmt(pi_phase)}, beta={_fmt(beta_sens)}, drop={_fmt(drop_sens)}"
        )
    )
    fig.tight_layout()
    savefig(fig, plots_dir / "accuracy_vs_bins_smoothing.png")
    plt.close(fig)

    # C5) Example profiles by truth class.
    embedding_written = False
    for truth_class, items in example_profiles.items():
        for i, ex in enumerate(items, start=1):
            if (not embedding_written) and ("X" in ex) and ("f" in ex):
                fig_emb, ax_emb = plt.subplots(
                    figsize=(5.2, 4.4), constrained_layout=False
                )
                plot_embedding_with_foreground(
                    np.asarray(ex["X"], dtype=float),
                    np.asarray(ex["f"], dtype=bool),
                    ax=ax_emb,
                    title=f"Representative embedding (ExpD {truth_class})",
                    s=5.0,
                    alpha_bg=0.30,
                    alpha_fg=0.75,
                )
                savefig(fig_emb, plots_dir / "embedding_example.png")
                plt.close(fig_emb)
                embedding_written = True
            E = np.asarray(ex["E_obs"], dtype=float)
            peak_idx = np.asarray(ex["peak_indices"], dtype=int)
            theta = np.linspace(0.0, 2.0 * np.pi, E.size, endpoint=False)
            fig, ax = plt.subplots(
                figsize=(5.2, 4.8),
                subplot_kw={"projection": "polar"},
                constrained_layout=False,
            )
            plot_rsp_polar(
                theta,
                E,
                ax=ax,
                color="#1f77b4",
                label="E(theta)",
                title=(
                    f"ExpD example profile: {truth_class} #{i}\n"
                    f"pred={ex['pred_class']}, K_hat={int(ex['K_hat'])}, p_T={float(ex['p_T']):.3f}, stability={ex['stability']}"
                ),
            )
            if peak_idx.size > 0:
                peak_theta = theta[np.clip(peak_idx, 0, max(0, E.size - 1))]
                ax.scatter(
                    peak_theta,
                    E[np.clip(peak_idx, 0, max(0, E.size - 1))],
                    color="#d62728",
                    s=28,
                    label="Detected peaks",
                )
            tau = float(ex["tau"])
            if np.isfinite(tau):
                ax.plot(
                    np.linspace(0.0, 2.0 * np.pi, 361),
                    np.full(361, tau),
                    linestyle="--",
                    color="#2ca02c",
                    linewidth=1.0,
                    label="tau (95% null E)",
                )
            ax.legend(frameon=False)
            fig.tight_layout()
            savefig(fig, plots_dir / f"example_profiles_{truth_class}_{int(i)}.png")
            plt.close(fig)


def run_validations(metrics: pd.DataFrame, outdir: Path, alpha_sig: float) -> None:
    report_lines: list[str] = []

    valid = metrics.loc[~metrics["underpowered"].astype(bool)].copy()
    if valid.empty:
        report_lines.append("No non-underpowered rows; validations skipped.")
        (outdir / "results" / "validation_debug_report.txt").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )
        print("Validation warning: no non-underpowered rows.", flush=True)
        return

    # Overall accuracy table at condition level.
    cond_cols = [
        "N",
        "D",
        "sigma_eta",
        "pi_target",
        "beta",
        "dropout_noise",
        "bins_B",
        "mode",
        "smooth_w",
    ]
    cond_acc = valid.groupby(cond_cols, as_index=False).agg(
        n=("pred_bucket", "size"),
        accuracy=(
            "pred_bucket",
            lambda s: float(
                np.mean(
                    s.to_numpy(dtype=str)
                    == valid.loc[s.index, "truth_bucket"].to_numpy(dtype=str)
                )
            ),
        ),
        sig_rate=(
            "p_T",
            lambda s: float(
                np.mean(pd.to_numeric(s, errors="coerce") <= float(alpha_sig))
            ),
        ),
    )

    # Beta monotonicity (soft).
    beta_off = []
    beta_grp_cols = [
        "N",
        "D",
        "sigma_eta",
        "pi_target",
        "dropout_noise",
        "bins_B",
        "mode",
        "smooth_w",
    ]
    for keys, grp in cond_acc.groupby(beta_grp_cols):
        x = pd.to_numeric(grp["beta"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(grp["accuracy"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(np.sum(mask)) < 3:
            continue
        if np.allclose(x[mask], x[mask][0]) or np.allclose(y[mask], y[mask][0]):
            rho = float("nan")
        else:
            rho, _ = spearmanr(x[mask], y[mask], nan_policy="omit")
        if rho is None or not np.isfinite(float(rho)) or float(rho) < 0.4:
            beta_off.append(
                {
                    "N": keys[0],
                    "D": keys[1],
                    "sigma_eta": keys[2],
                    "pi_target": keys[3],
                    "dropout_noise": keys[4],
                    "bins_B": keys[5],
                    "mode": keys[6],
                    "smooth_w": keys[7],
                    "rho_beta_accuracy": (
                        float(rho) if rho is not None else float("nan")
                    ),
                }
            )

    # Dropout monotonicity (soft, negative).
    drop_off = []
    drop_grp_cols = [
        "N",
        "D",
        "sigma_eta",
        "pi_target",
        "beta",
        "bins_B",
        "mode",
        "smooth_w",
    ]
    for keys, grp in cond_acc.groupby(drop_grp_cols):
        x = pd.to_numeric(grp["dropout_noise"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(grp["accuracy"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(np.sum(mask)) < 3:
            continue
        if np.allclose(x[mask], x[mask][0]) or np.allclose(y[mask], y[mask][0]):
            rho = float("nan")
        else:
            rho, _ = spearmanr(x[mask], y[mask], nan_policy="omit")
        if rho is None or not np.isfinite(float(rho)) or float(rho) > -0.2:
            drop_off.append(
                {
                    "N": keys[0],
                    "D": keys[1],
                    "sigma_eta": keys[2],
                    "pi_target": keys[3],
                    "beta": keys[4],
                    "bins_B": keys[5],
                    "mode": keys[6],
                    "smooth_w": keys[7],
                    "rho_dropout_accuracy": (
                        float(rho) if rho is not None else float("nan")
                    ),
                }
            )

    # Null check if beta=0 present.
    null_rows = valid.loc[pd.to_numeric(valid["beta"], errors="coerce") == 0.0]
    null_off = []
    if not null_rows.empty:
        null_grp = null_rows.groupby(
            ["N", "D", "sigma_eta", "pi_target", "bins_B", "mode", "smooth_w"],
            as_index=False,
        ).agg(
            n=("p_T", "size"),
            sig_rate=(
                "p_T",
                lambda s: float(
                    np.mean(pd.to_numeric(s, errors="coerce") <= float(alpha_sig))
                ),
            ),
        )
        for _, r in null_grp.iterrows():
            if int(r["n"]) < 20:
                continue
            if not (0.03 <= float(r["sig_rate"]) <= 0.07):
                null_off.append(r.to_dict())

    # High-beta/low-drop diagonal sanity.
    hi_beta = float(pd.to_numeric(valid["beta"], errors="coerce").max())
    lo_drop = float(pd.to_numeric(valid["dropout_noise"], errors="coerce").min())
    hi_sub = valid.loc[(valid["beta"] == hi_beta) & (valid["dropout_noise"] == lo_drop)]
    hi_acc = (
        float(
            np.mean(
                hi_sub["pred_bucket"].astype(str) == hi_sub["truth_bucket"].astype(str)
            )
        )
        if not hi_sub.empty
        else float("nan")
    )

    if beta_off:
        report_lines.append("Beta monotonicity offenders (rho < 0.4):")
        report_lines.append(pd.DataFrame(beta_off).to_string(index=False))
    else:
        report_lines.append("Beta monotonicity soft-check passed.")

    if drop_off:
        report_lines.append("Dropout monotonicity offenders (rho > -0.2):")
        report_lines.append(pd.DataFrame(drop_off).to_string(index=False))
    else:
        report_lines.append("Dropout monotonicity soft-check passed.")

    if null_rows.empty:
        report_lines.append("Null calibration check skipped (beta=0 not in grid).")
    elif null_off:
        report_lines.append(
            "Null calibration offenders (sig_rate outside [0.03,0.07]):"
        )
        report_lines.append(pd.DataFrame(null_off).to_string(index=False))
    else:
        report_lines.append("Null calibration check passed where n>=20.")

    if np.isfinite(hi_acc) and hi_acc < 0.7:
        report_lines.append(
            f"High-beta/low-drop diagonal warning: accuracy={hi_acc:.3f} < 0.7"
        )
    else:
        report_lines.append(
            f"High-beta/low-drop diagonal sanity: accuracy={hi_acc:.3f}"
        )

    dbg_path = outdir / "results" / "validation_debug_report.txt"
    dbg_path.write_text("\n\n".join(report_lines), encoding="utf-8")

    # Console summary.
    if beta_off or drop_off or null_off or (np.isfinite(hi_acc) and hi_acc < 0.7):
        print(f"Validation warnings detected. See {dbg_path}", flush=True)
    else:
        print("Validation checks passed without warnings.", flush=True)


def run_experiment(args: argparse.Namespace) -> None:
    test_cfg = (
        get_testmode_config(int(args.master_seed)) if bool(args.test_mode) else None
    )
    testmode_banner(bool(args.test_mode), test_cfg)

    outdir = resolve_outdir(args.outdir, bool(args.test_mode))
    results_dir = ensure_dir(outdir / "results")
    ensure_dir(outdir / "plots")
    confusion_dir = ensure_dir(outdir / "confusion_matrices")

    if bool(args.test_mode):
        N_grid = [3000]
        D_grid = [10]
        sigma_eta_grid = [0.4]
        pi_grid = [0.05, 0.2]
        beta_grid = [0.0, 0.5, 1.0]
        dropout_grid = [0.0, 0.2]
        bins_grid = [24, 36]
        w_grid = [1, 3]
        modes = ["raw", "smoothed"]
        genes_per_class = 8
        n_perm = 120
        n_master_seeds = 1
        n_boot = 5
    else:
        N_grid = [int(x) for x in (args.N_grid if args.N_grid else [args.N])]
        D_grid = [int(x) for x in (args.D_grid if args.D_grid else [args.D])]
        sigma_eta_grid = [float(x) for x in args.sigma_eta_grid]
        pi_grid = [float(x) for x in args.pi_grid]
        beta_grid = [float(x) for x in args.beta_grid]
        dropout_grid = [float(x) for x in args.dropout_grid]
        bins_grid = [int(x) for x in args.bins_grid]
        w_grid = [int(x) for x in args.w_grid]
        modes = [str(x) for x in args.modes]
        genes_per_class = int(args.genes_per_class)
        n_perm = int(args.n_perm)
        n_master_seeds = int(args.n_master_seeds)
        n_boot = int(args.n_boot)

    if bool(args.add_null_beta) and all(abs(float(b)) > 1e-12 for b in beta_grid):
        beta_grid = [0.0] + beta_grid

    truth_classes = [str(x) for x in args.truth_classes]
    min_perm_eff = min(int(args.min_perm), int(n_perm))
    seed_values = [int(args.master_seed) + i for i in range(int(n_master_seeds))]

    rep_N = int(N_grid[np.argmin(np.abs(np.asarray(N_grid) - 20000))])
    rep_D = int(D_grid[np.argmin(np.abs(np.asarray(D_grid) - 10))])
    rep_sigma = _select_rep(0.4, sigma_eta_grid)
    rep_bins = int(bins_grid[np.argmin(np.abs(np.asarray(bins_grid) - 36))])
    rep_mode = "smoothed" if "smoothed" in modes else modes[0]
    rep_w = int(w_grid[np.argmin(np.abs(np.asarray(w_grid) - 3))])
    rep_pi = _select_rep(0.2, pi_grid)
    rep_beta = _select_rep(1.0, beta_grid)
    rep_drop = _select_rep(0.1, dropout_grid)

    cfg = {
        "experiment": "Simulation Experiment D: shape identifiability under noise + bin/smoothing sensitivity",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit_short": git_commit_hash(cwd=REPO_ROOT),
        "master_seed": int(args.master_seed),
        "seed_values": [int(x) for x in seed_values],
        "N_grid": N_grid,
        "D_grid": D_grid,
        "sigma_eta_grid": sigma_eta_grid,
        "pi_grid": pi_grid,
        "beta_grid": beta_grid,
        "dropout_grid": dropout_grid,
        "bins_grid": bins_grid,
        "w_grid": w_grid,
        "modes": modes,
        "truth_classes": truth_classes,
        "genes_per_class": int(genes_per_class),
        "n_perm": int(n_perm),
        "n_boot": int(n_boot),
        "alpha_sig": float(args.alpha_sig),
        "dropout_symmetric": bool(args.dropout_symmetric),
        "dropout_p01_factor": float(args.dropout_p01_factor),
        "patchy_width": float(args.patchy_width),
        "underpowered": {
            "p_min": 0.005,
            "min_fg_total": int(args.min_fg_total),
            "min_fg_per_donor": int(args.min_fg_per_donor),
            "min_bg_per_donor": int(args.min_bg_per_donor),
            "d_eff_min": int(args.d_eff_min),
            "min_perm_effective": int(min_perm_eff),
        },
        "representative_condition": {
            "N": int(rep_N),
            "D": int(rep_D),
            "sigma_eta": float(rep_sigma),
            "pi_target": float(rep_pi),
            "beta": float(rep_beta),
            "dropout_noise": float(rep_drop),
            "bins_B": int(rep_bins),
            "mode": str(rep_mode),
            "smooth_w": int(rep_w),
        },
        "test_mode": bool(args.test_mode),
    }
    write_config(outdir, cfg)

    dataset_grid = [
        (int(seed_run), int(N), int(D), float(sigma))
        for seed_run in seed_values
        for N in N_grid
        for D in D_grid
        for sigma in sigma_eta_grid
    ]

    total_genes = (
        len(dataset_grid)
        * len(truth_classes)
        * len(pi_grid)
        * len(beta_grid)
        * len(dropout_grid)
        * len(bins_grid)
        * len(modes)
        * len(w_grid)
        * int(genes_per_class)
    )
    print(
        (
            f"ExpD work plan: datasets={len(dataset_grid)}, total_genes={total_genes}, n_perm={n_perm}, "
            f"truth_classes={truth_classes}"
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    example_profiles: dict[str, list[dict[str, Any]]] = {
        cls: [] for cls in truth_classes
    }

    start = time.time()
    counter = 0

    for ds_i, (seed_run, N, D, sigma_eta) in enumerate(dataset_grid, start=1):
        seed_dataset = stable_seed(
            int(args.master_seed),
            "expD_dataset",
            int(seed_run),
            int(N),
            int(D),
            float(sigma_eta),
        )
        rng_ds = rng_from_seed(seed_dataset)

        X, _ = sample_disk_gaussian(int(N), rng_ds)
        theta = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2.0 * np.pi)
        donor_ids = assign_donors(int(N), int(D), rng_ds)
        eta_d = sample_donor_effects(int(D), float(sigma_eta), rng_ds)
        eta_cell = donor_effect_vector(donor_ids, eta_d)
        donor_groups = [
            np.flatnonzero(donor_ids == d).astype(int) for d in np.unique(donor_ids)
        ]

        bin_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for b in bins_grid:
            edges = np.linspace(0.0, 2.0 * np.pi, int(b) + 1, endpoint=True)
            bid = np.digitize(theta, edges, right=False) - 1
            bid = np.where(bid == int(b), int(b) - 1, bid).astype(np.int32)
            btot = np.bincount(bid, minlength=int(b)).astype(np.int64)
            bin_cache[int(b)] = (bid, btot)

        run_id_base = f"N{int(N)}_D{int(D)}_sigma{_fmt(sigma_eta)}_seed{int(seed_run)}"
        print(f"[{ds_i}/{len(dataset_grid)}] {run_id_base}", flush=True)

        for truth_class in truth_classes:
            for pi_target in pi_grid:
                for beta in beta_grid:
                    for drop in dropout_grid:
                        for bins_B in bins_grid:
                            bin_id, bin_counts_total = bin_cache[int(bins_B)]
                            for mode in modes:
                                for smooth_w in w_grid:
                                    for rep in range(int(genes_per_class)):
                                        counter += 1
                                        seed_gene = stable_seed(
                                            int(args.master_seed),
                                            "expD_gene",
                                            int(seed_run),
                                            int(N),
                                            int(D),
                                            float(sigma_eta),
                                            str(truth_class),
                                            float(pi_target),
                                            float(beta),
                                            float(drop),
                                            int(bins_B),
                                            str(mode),
                                            int(smooth_w),
                                            int(rep),
                                        )
                                        rng_gene = rng_from_seed(seed_gene)
                                        theta0 = float(
                                            rng_gene.uniform(0.0, 2.0 * np.pi)
                                        )

                                        f = _simulate_gene(
                                            theta=theta,
                                            donor_eta=eta_cell,
                                            pi_target=float(pi_target),
                                            beta=float(beta),
                                            theta0=theta0,
                                            truth_class=str(truth_class),
                                            patchy_width=float(args.patchy_width),
                                            rng=rng_gene,
                                        )
                                        f = apply_dropout_noise(
                                            f=f,
                                            dropout_rate=float(drop),
                                            rng=rng_gene,
                                            symmetric=bool(args.dropout_symmetric),
                                            p01_factor=float(args.dropout_p01_factor),
                                        )

                                        prev_obs = float(np.mean(f))
                                        n_fg_total = int(np.sum(f))
                                        truth_bucket = TRUTH_BUCKET[str(truth_class)]

                                        power = evaluate_underpowered(
                                            donor_ids=donor_ids,
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

                                        T_obs = float("nan")
                                        p_T = float("nan")
                                        K_hat = float("nan")
                                        C_hat = float("nan")
                                        S1, S2, S3 = (
                                            float("nan"),
                                            float("nan"),
                                            float("nan"),
                                        )
                                        stability = float("nan")
                                        pred_class = "UNCERTAIN_UNDERPOWERED"

                                        peak_indices = np.zeros(0, dtype=int)
                                        tau = float("nan")
                                        prom_thresh = float("nan")
                                        E_obs = None

                                        if not underpowered:
                                            seed_perm = stable_seed(
                                                int(args.master_seed),
                                                "expD_perm",
                                                int(seed_run),
                                                int(N),
                                                int(D),
                                                float(sigma_eta),
                                                str(truth_class),
                                                float(pi_target),
                                                float(beta),
                                                float(drop),
                                                int(bins_B),
                                                str(mode),
                                                int(smooth_w),
                                                int(rep),
                                            )

                                            perm = perm_null_T(
                                                f=f,
                                                angles=theta,
                                                donor_ids=donor_ids,
                                                n_bins=int(bins_B),
                                                n_perm=int(n_perm),
                                                seed=int(seed_perm),
                                                mode=str(mode),
                                                smooth_w=int(smooth_w),
                                                donor_stratified=True,
                                                return_null_T=True,
                                                return_obs_profile=True,
                                                return_null_profiles=True,
                                                bin_id=bin_id,
                                                bin_counts_total=bin_counts_total,
                                            )
                                            T_obs = float(perm["T_obs"])
                                            p_T = float(perm["p_T"])
                                            E_obs = np.asarray(
                                                perm["E_phi_obs"], dtype=float
                                            )
                                            null_E = np.asarray(
                                                perm["null_E_phi"], dtype=float
                                            )

                                            (
                                                K_hat_int,
                                                C_hat_val,
                                                prom_thresh,
                                                peak_indices,
                                                tau,
                                            ) = _peak_and_coverage_features(
                                                E_obs, null_E
                                            )
                                            K_hat = int(K_hat_int)
                                            C_hat = float(C_hat_val)
                                            S1, S2, S3 = _fourier_ratios(E_obs)

                                            if int(n_boot) > 0:
                                                seed_boot = stable_seed(
                                                    int(args.master_seed),
                                                    "expD_boot",
                                                    int(seed_run),
                                                    int(N),
                                                    int(D),
                                                    float(sigma_eta),
                                                    str(truth_class),
                                                    float(pi_target),
                                                    float(beta),
                                                    float(drop),
                                                    int(bins_B),
                                                    str(mode),
                                                    int(smooth_w),
                                                    int(rep),
                                                )
                                                stability = _bootstrap_stability(
                                                    f=f,
                                                    theta=theta,
                                                    bin_id=bin_id,
                                                    donor_groups=donor_groups,
                                                    n_bins=int(bins_B),
                                                    mode=str(mode),
                                                    smooth_w=int(smooth_w),
                                                    prom_thresh=float(prom_thresh),
                                                    k_obs=int(K_hat_int),
                                                    n_boot=int(n_boot),
                                                    seed=int(seed_boot),
                                                )

                                            if float(p_T) > float(args.alpha_sig):
                                                pred_class = "UBIQUITOUS_OR_NULL"
                                            else:
                                                if np.isfinite(stability) and float(
                                                    stability
                                                ) < float(args.stability_thresh):
                                                    pred_class = "UNCERTAIN_SHAPE"
                                                else:
                                                    if int(K_hat_int) == 1:
                                                        pred_class = "UNIMODAL"
                                                    elif int(K_hat_int) == 2:
                                                        pred_class = "BIMODAL"
                                                    elif int(K_hat_int) >= 3:
                                                        pred_class = "MULTIMODAL_PATCHY"
                                                    else:
                                                        pred_class = "UNCERTAIN_SHAPE"

                                        pred_bucket = _pred_bucket(pred_class)
                                        rows.append(
                                            {
                                                "run_id": run_id_base,
                                                "seed": int(seed_gene),
                                                "N": int(N),
                                                "D": int(D),
                                                "sigma_eta": float(sigma_eta),
                                                "truth_class": str(truth_class),
                                                "truth_bucket": str(truth_bucket),
                                                "truth_k": int(
                                                    TRUTH_K[str(truth_class)]
                                                ),
                                                "truth_theta0": float(theta0),
                                                "pi_target": float(pi_target),
                                                "beta": float(beta),
                                                "dropout_noise": float(drop),
                                                "bins_B": int(bins_B),
                                                "mode": str(mode),
                                                "smooth_w": int(smooth_w),
                                                "prev_obs": float(prev_obs),
                                                "n_fg_total": int(n_fg_total),
                                                "D_eff": int(power["D_eff"]),
                                                "underpowered": bool(underpowered),
                                                "T_obs": float(T_obs),
                                                "p_T": float(p_T),
                                                "K_hat": float(K_hat),
                                                "C_hat": float(C_hat),
                                                "S1": float(S1),
                                                "S2": float(S2),
                                                "S3": float(S3),
                                                "stability": float(stability),
                                                "pred_class": str(pred_class),
                                                "pred_bucket": str(pred_bucket),
                                            }
                                        )

                                        # Store representative examples.
                                        if (
                                            E_obs is not None
                                            and (not underpowered)
                                            and (float(p_T) <= float(args.alpha_sig))
                                            and (
                                                (not np.isfinite(stability))
                                                or (
                                                    float(stability)
                                                    >= float(args.stability_thresh)
                                                )
                                            )
                                            and len(example_profiles[str(truth_class)])
                                            < 3
                                        ):
                                            example_profiles[str(truth_class)].append(
                                                {
                                                    "E_obs": np.asarray(
                                                        E_obs, dtype=float
                                                    ).copy(),
                                                    "X": np.asarray(
                                                        X, dtype=float
                                                    ).copy(),
                                                    "f": np.asarray(
                                                        f, dtype=bool
                                                    ).copy(),
                                                    "peak_indices": np.asarray(
                                                        peak_indices, dtype=int
                                                    ).copy(),
                                                    "tau": float(tau),
                                                    "prom_thresh": float(prom_thresh),
                                                    "pred_class": str(pred_class),
                                                    "K_hat": (
                                                        int(K_hat)
                                                        if np.isfinite(K_hat)
                                                        else -1
                                                    ),
                                                    "p_T": float(p_T),
                                                    "stability": (
                                                        float(stability)
                                                        if np.isfinite(stability)
                                                        else "NA"
                                                    ),
                                                }
                                            )

                                        if (
                                            counter % int(args.progress_every) == 0
                                            or counter == total_genes
                                        ):
                                            elapsed = time.time() - start
                                            rate = (
                                                counter / elapsed
                                                if elapsed > 0
                                                else float("nan")
                                            )
                                            print(
                                                f"  progress: {counter}/{total_genes} genes ({rate:.2f} genes/s)",
                                                flush=True,
                                            )

    metrics = pd.DataFrame(rows)
    summary = summarize(metrics, alpha_sig=float(args.alpha_sig))

    atomic_write_csv(results_dir / "metrics_long.csv", metrics)
    atomic_write_csv(results_dir / "summary.csv", summary)
    write_confusion_csvs(metrics, confusion_dir)

    _set_plot_style()
    make_plots(
        metrics,
        summary,
        outdir=outdir,
        alpha_sig=float(args.alpha_sig),
        n_perm=int(n_perm),
        N_ref=int(rep_N),
        D_ref=int(rep_D),
        sigma_ref=float(rep_sigma),
        bins_ref=int(rep_bins),
        mode_ref=str(rep_mode),
        w_ref=int(rep_w),
        pi_grid=pi_grid,
        beta_grid=beta_grid,
        dropout_grid=dropout_grid,
        bins_grid=bins_grid,
        w_grid=w_grid,
        modes=modes,
        example_profiles=example_profiles,
    )

    run_validations(metrics, outdir=outdir, alpha_sig=float(args.alpha_sig))

    elapsed = time.time() - start
    print(
        (
            f"Completed ExpD in {elapsed/60.0:.2f} min. rows={metrics.shape[0]}, "
            f"non_underpowered={int((~metrics['underpowered']).sum())}"
        ),
        flush=True,
    )
    print(f"Outputs written to: {outdir}", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Simulation Experiment D: shape identifiability under noise + bin/smoothing sensitivity."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/simulations/expD_shape_identifiability",
        help="Output directory.",
    )
    parser.add_argument("--master_seed", type=int, default=123)
    parser.add_argument("--n_master_seeds", type=int, default=1)

    parser.add_argument("--N", type=int, default=20000)
    parser.add_argument("--D", type=int, default=10)
    parser.add_argument("--N_grid", type=int, nargs="*", default=None)
    parser.add_argument("--D_grid", type=int, nargs="*", default=None)
    parser.add_argument("--sigma_eta_grid", type=float, nargs="+", default=[0.0, 0.4])

    parser.add_argument("--n_perm", type=int, default=500)
    parser.add_argument("--alpha_sig", type=float, default=0.05)
    parser.add_argument("--bins_grid", type=int, nargs="+", default=[24, 36, 48])
    parser.add_argument("--w_grid", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--modes", type=str, nargs="+", default=["raw", "smoothed"])

    parser.add_argument("--pi_grid", type=float, nargs="+", default=[0.05, 0.2])
    parser.add_argument(
        "--beta_grid", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.25]
    )
    parser.add_argument(
        "--add_null_beta",
        action="store_true",
        help="Add beta=0.0 to grid for null calibration checks.",
    )
    parser.add_argument(
        "--dropout_grid", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3]
    )
    parser.add_argument("--dropout_symmetric", action="store_true")
    parser.add_argument("--dropout_p01_factor", type=float, default=0.5)

    parser.add_argument("--truth_classes", type=str, nargs="+", default=TRUTH_CLASSES)
    parser.add_argument("--patchy_width", type=float, default=0.45)
    parser.add_argument("--genes_per_class", type=int, default=200)
    parser.add_argument("--n_boot", type=int, default=0)
    parser.add_argument("--stability_thresh", type=float, default=0.7)

    parser.add_argument("--min_fg_total", type=int, default=50)
    parser.add_argument("--min_fg_per_donor", type=int, default=10)
    parser.add_argument("--min_bg_per_donor", type=int, default=10)
    parser.add_argument("--d_eff_min", type=int, default=2)
    parser.add_argument("--min_perm", type=int, default=200)

    parser.add_argument("--progress_every", type=int, default=200)
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

    if args.n_perm <= 0 or args.genes_per_class <= 0:
        raise ValueError("n_perm and genes_per_class must be positive.")
    if args.min_perm <= 0 or args.d_eff_min < 1:
        raise ValueError("min_perm must be positive and d_eff_min >= 1.")

    run_ctx = prepare_legacy_run(args, "expD_shape_identifiability", __file__)
    run_experiment(args)
    finalize_legacy_run(run_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
