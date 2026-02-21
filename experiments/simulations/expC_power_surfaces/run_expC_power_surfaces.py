#!/usr/bin/env python3
"""Simulation Experiment C: power surfaces + abstention boundaries.

This script quantifies power for BioRSP max-statistic anisotropy scoring under
controlled angular signals with donor-aware permutation testing.
"""

from __future__ import annotations

import argparse
import json
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
from scipy.stats import spearmanr

# Non-interactive matplotlib backend and writable cache dirs.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-biorsp-expC")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache-biorsp-expC")
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

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
    write_config,
)
from experiments.simulations._shared.models import simulate_unimodal_gene
from experiments.simulations._shared.plots import (
    plot_embedding_with_foreground,
    plot_rsp_polar,
    savefig,
    wilson_ci,
)
from experiments.simulations._shared.reporting import write_report_expC
from experiments.simulations._shared.runner import (
    finalize_legacy_run,
    prepare_legacy_run,
)
from experiments.simulations._shared.seeding import rng_from_seed, stable_seed
from experiments.simulations._sim_testmode import banner as testmode_banner
from experiments.simulations._sim_testmode import get_testmode_config, resolve_outdir

DEFAULT_N_GRID = [5000, 20000, 50000]
DEFAULT_D_GRID = [2, 5, 10, 15]
DEFAULT_SIGMA_ETA_GRID = [0.0, 0.4, 0.8]
DEFAULT_PI_GRID = [0.01, 0.05, 0.2, 0.6]
DEFAULT_BETA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def _parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer.")
    return vals


def _parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one float.")
    return vals


def _fmt_float(x: float) -> str:
    return f"{float(x):.3f}".rstrip("0").rstrip(".")


def _fmt_token(x: float | int) -> str:
    return str(x).replace(".", "p")


def _compute_bin_cache(theta: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    wrapped = np.mod(np.asarray(theta, dtype=float).ravel(), 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, int(n_bins) + 1, endpoint=True)
    bin_id = np.digitize(wrapped, edges, right=False) - 1
    bin_id = np.where(bin_id == int(n_bins), int(n_bins) - 1, bin_id).astype(np.int32)
    bin_counts_total = np.bincount(bin_id, minlength=int(n_bins)).astype(np.int64)
    return bin_id, bin_counts_total


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


def build_grid(
    N_grid: list[int],
    D_grid: list[int],
    sigma_eta_grid: list[float],
    seed_values: list[int],
) -> list[tuple[int, int, int, float]]:
    """Build dataset-level condition grid: (seed_run, N, D, sigma_eta)."""
    return [
        (int(seed_run), int(N), int(D), float(sigma_eta))
        for seed_run in seed_values
        for N in N_grid
        for D in D_grid
        for sigma_eta in sigma_eta_grid
    ]


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["N", "D", "sigma_eta", "pi_target", "beta"]

    for keys, grp in metrics.groupby(group_cols, sort=True):
        N, D, sigma_eta, pi_target, beta = keys
        n_total = int(grp.shape[0])
        valid = grp.loc[~grp["underpowered"].astype(bool)]
        n_analyzable = int(valid.shape[0])
        frac_underpowered = float(grp["underpowered"].mean())
        analyzable_rate = float(n_analyzable / n_total) if n_total > 0 else float("nan")
        mean_prev_obs = float(pd.to_numeric(grp["prev_obs"], errors="coerce").mean())
        mean_deff = float(pd.to_numeric(grp["D_eff"], errors="coerce").mean())
        analyzable_lo, analyzable_hi = wilson_ci(n_analyzable, n_total, alpha=0.05)

        pvals = pd.to_numeric(valid["p_T"], errors="coerce")
        sig_mask = np.isfinite(pvals.to_numpy(dtype=float)) & (
            pvals.to_numpy(dtype=float) <= 0.05
        )
        k_sig = int(np.sum(sig_mask))

        conditional_power = float("nan")
        conditional_lo = float("nan")
        conditional_hi = float("nan")
        if n_analyzable > 0:
            conditional_power = float(k_sig / n_analyzable)
            conditional_lo, conditional_hi = wilson_ci(k_sig, n_analyzable, alpha=0.05)

        operational_power = float(k_sig / n_total) if n_total > 0 else float("nan")
        operational_lo, operational_hi = wilson_ci(k_sig, n_total, alpha=0.05)

        power = float("nan")
        power_lo = float("nan")
        power_hi = float("nan")
        typeI = float("nan")
        typeI_lo = float("nan")
        typeI_hi = float("nan")
        typei_conditional = float("nan")
        typei_conditional_lo = float("nan")
        typei_conditional_hi = float("nan")
        typei_operational = float("nan")
        typei_operational_lo = float("nan")
        typei_operational_hi = float("nan")
        if float(beta) > 0.0:
            power = conditional_power
            power_lo, power_hi = conditional_lo, conditional_hi
        else:
            typeI = conditional_power
            typeI_lo, typeI_hi = conditional_lo, conditional_hi
            typei_conditional = conditional_power
            typei_conditional_lo, typei_conditional_hi = conditional_lo, conditional_hi
            typei_operational = operational_power
            typei_operational_lo, typei_operational_hi = operational_lo, operational_hi

        rows.append(
            {
                "N": int(N),
                "D": int(D),
                "sigma_eta": float(sigma_eta),
                "pi_target": float(pi_target),
                "beta": float(beta),
                "n_genes": n_total,
                "n_non_underpowered": n_analyzable,
                "frac_underpowered": frac_underpowered,
                "analyzable_rate": analyzable_rate,
                "analyzable_rate_ci_low": analyzable_lo,
                "analyzable_rate_ci_high": analyzable_hi,
                "conditional_power_alpha05": conditional_power,
                "conditional_power_alpha05_ci_low": conditional_lo,
                "conditional_power_alpha05_ci_high": conditional_hi,
                "operational_power_alpha05": operational_power,
                "operational_power_alpha05_ci_low": operational_lo,
                "operational_power_alpha05_ci_high": operational_hi,
                "power_alpha05": power,
                "power_alpha05_ci_low": power_lo,
                "power_alpha05_ci_high": power_hi,
                "typeI_alpha05": typeI,
                "typeI_alpha05_ci_low": typeI_lo,
                "typeI_alpha05_ci_high": typeI_hi,
                "typeI_alpha05_conditional": typei_conditional,
                "typeI_alpha05_conditional_ci_low": typei_conditional_lo,
                "typeI_alpha05_conditional_ci_high": typei_conditional_hi,
                "typeI_alpha05_operational": typei_operational,
                "typeI_alpha05_operational_ci_low": typei_operational_lo,
                "typeI_alpha05_operational_ci_high": typei_operational_hi,
                "mean_prev_obs": mean_prev_obs,
                "mean_D_eff": mean_deff,
            }
        )
    return pd.DataFrame(rows)


def _annotate_heatmap(
    ax: plt.Axes,
    value_mat: np.ndarray,
    n_mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    value_fmt: str = "{:.2f}",
) -> Any:
    im = ax.imshow(value_mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(value_mat.shape[0]):
        for j in range(value_mat.shape[1]):
            v = value_mat[i, j]
            n = n_mat[i, j]
            txt = "NA" if not np.isfinite(v) else value_fmt.format(v)
            if np.isfinite(n):
                txt = f"{txt}\nn={int(n)}"
            color = "white" if np.isfinite(v) and v > (vmin + vmax) / 2 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.7, color=color)
    return im


def _matrix_from_summary(
    summary: pd.DataFrame,
    *,
    row_vals: list[float] | list[int],
    col_vals: list[float] | list[int],
    row_key: str,
    col_key: str,
    value_col: str,
    n_col: str,
    filters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    m = np.full((len(row_vals), len(col_vals)), np.nan, dtype=float)
    n = np.full((len(row_vals), len(col_vals)), np.nan, dtype=float)
    sub = summary.copy()
    for k, v in filters.items():
        sub = sub.loc[sub[k] == v]
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            cell = sub.loc[(sub[row_key] == rv) & (sub[col_key] == cv)]
            if cell.empty:
                continue
            m[i, j] = (
                float(cell[value_col].iloc[0])
                if pd.notna(cell[value_col].iloc[0])
                else np.nan
            )
            n[i, j] = (
                float(cell[n_col].iloc[0]) if pd.notna(cell[n_col].iloc[0]) else np.nan
            )
    return m, n


def plot_power_heatmaps(
    summary: pd.DataFrame,
    *,
    sigma_eta_grid: list[float],
    pi_grid: list[float],
    D_grid: list[int],
    beta_grid: list[float],
    N_ref: int,
    plots_dir: Path,
    n_perm: int,
    bins: int,
    genes_per_condition: int,
) -> None:
    _plot_metric_heatmaps(
        summary=summary,
        sigma_eta_grid=sigma_eta_grid,
        pi_grid=pi_grid,
        D_grid=D_grid,
        beta_grid=beta_grid,
        N_ref=N_ref,
        plots_dir=plots_dir,
        n_perm=n_perm,
        bins=bins,
        genes_per_condition=genes_per_condition,
        value_col="power_alpha05",
        n_col="n_non_underpowered",
        out_prefix="power_heatmap",
        title_metric="power @ alpha=0.05",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )


def _plot_metric_heatmaps(
    *,
    summary: pd.DataFrame,
    sigma_eta_grid: list[float],
    pi_grid: list[float],
    D_grid: list[int],
    beta_grid: list[float],
    N_ref: int,
    plots_dir: Path,
    n_perm: int,
    bins: int,
    genes_per_condition: int,
    value_col: str,
    n_col: str,
    out_prefix: str,
    title_metric: str,
    vmin: float,
    vmax: float,
    cmap: str,
) -> None:
    if value_col not in summary.columns:
        return
    betas_pos = [float(b) for b in beta_grid if float(b) > 0.0]
    if not betas_pos:
        return
    row_labels = [_fmt_float(b) for b in betas_pos]
    col_labels = [str(int(d)) for d in D_grid]

    for sigma in sigma_eta_grid:
        for pi in pi_grid:
            mat, nmat = _matrix_from_summary(
                summary,
                row_vals=betas_pos,
                col_vals=[int(d) for d in D_grid],
                row_key="beta",
                col_key="D",
                value_col=value_col,
                n_col=n_col,
                filters={
                    "N": int(N_ref),
                    "sigma_eta": float(sigma),
                    "pi_target": float(pi),
                },
            )
            fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=False)
            im = _annotate_heatmap(
                ax,
                mat,
                nmat,
                row_labels,
                col_labels,
                (
                    f"ExpC {title_metric} heatmap (donor-stratified permutation, plus-one p-values)\n"
                    f"N={N_ref}, sigma_eta={_fmt_float(sigma)}, pi={_fmt_float(pi)}, bins={bins}, n_perm={n_perm}, G={genes_per_condition}"
                ),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            ax.set_xlabel("D")
            ax.set_ylabel("beta")
            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label(title_metric)
            savefig(
                fig,
                plots_dir
                / f"{out_prefix}_N{int(N_ref)}_sigma{_fmt_token(sigma)}_pi{_fmt_token(pi)}.png",
            )
            plt.close(fig)


def plot_tradeoff(
    summary: pd.DataFrame,
    *,
    sigma_eta_grid: list[float],
    pi_grid: list[float],
    D_grid: list[int],
    N_grid: list[int],
    beta_grid: list[float],
    plots_dir: Path,
    n_perm: int,
) -> None:
    if not pi_grid:
        return
    beta_target = min(
        [float(b) for b in beta_grid if float(b) > 0.0] or [1.0],
        key=lambda x: abs(x - 1.0),
    )
    pi_focus: list[float] = []
    if 0.2 in [round(x, 10) for x in pi_grid]:
        pi_focus.append(0.2)
    for pi in pi_grid:
        if float(pi) not in pi_focus:
            pi_focus.append(float(pi))
        if len(pi_focus) >= 2:
            break

    for sigma in sigma_eta_grid:
        for pi in pi_focus:
            beta_label = f"{beta_target:.1f}"
            fig, axes = plt.subplots(
                1, 2, figsize=(10.0, 3.5), constrained_layout=False
            )
            ax_d, ax_n = axes

            # Panel A: power vs D for each N at fixed beta.
            for N in N_grid:
                ys = []
                ns = []
                for D in D_grid:
                    row = summary.loc[
                        (summary["N"] == int(N))
                        & (summary["D"] == int(D))
                        & (summary["sigma_eta"] == float(sigma))
                        & (summary["pi_target"] == float(pi))
                        & (summary["beta"] == float(beta_target))
                    ]
                    if row.empty:
                        ys.append(np.nan)
                        ns.append(np.nan)
                    else:
                        ys.append(float(row["power_alpha05"].iloc[0]))
                        ns.append(float(row["n_non_underpowered"].iloc[0]))
                ax_d.plot(D_grid, ys, marker="o", linewidth=1.5, label=f"N={int(N)}")
                for x, y, n in zip(D_grid, ys, ns):
                    if np.isfinite(y) and np.isfinite(n):
                        ax_d.text(
                            float(x),
                            float(y) + 0.02,
                            f"n={int(n)}",
                            fontsize=6,
                            ha="center",
                        )
            ax_d.set_ylim(0.0, 1.0)
            ax_d.set_xlabel("D")
            ax_d.set_ylabel("power @ alpha=0.05")
            ax_d.set_title("Panel A: power vs D (fixed beta)")
            ax_d.grid(True, axis="y", linestyle=":", alpha=0.35)
            ax_d.legend(frameon=False)

            # Panel B: power vs N for each D at fixed beta.
            for D in D_grid:
                ys = []
                ns = []
                for N in N_grid:
                    row = summary.loc[
                        (summary["N"] == int(N))
                        & (summary["D"] == int(D))
                        & (summary["sigma_eta"] == float(sigma))
                        & (summary["pi_target"] == float(pi))
                        & (summary["beta"] == float(beta_target))
                    ]
                    if row.empty:
                        ys.append(np.nan)
                        ns.append(np.nan)
                    else:
                        ys.append(float(row["power_alpha05"].iloc[0]))
                        ns.append(float(row["n_non_underpowered"].iloc[0]))
                ax_n.plot(N_grid, ys, marker="o", linewidth=1.5, label=f"D={int(D)}")
                for x, y, n in zip(N_grid, ys, ns):
                    if np.isfinite(y) and np.isfinite(n):
                        ax_n.text(
                            float(x),
                            float(y) + 0.02,
                            f"n={int(n)}",
                            fontsize=6,
                            ha="center",
                        )
            ax_n.set_ylim(0.0, 1.0)
            ax_n.set_xlabel("N")
            ax_n.set_ylabel("power @ alpha=0.05")
            ax_n.set_title("Panel B: power vs N (fixed beta)")
            ax_n.grid(True, axis="y", linestyle=":", alpha=0.35)
            ax_n.legend(frameon=False, ncol=2)

            fig.suptitle(
                (
                    "ExpC donors-vs-cells tradeoff (donor-stratified permutation, plus-one p-values)\n"
                    f"sigma_eta={_fmt_float(sigma)}, pi={_fmt_float(pi)}, beta={_fmt_float(beta_target)}, n_perm={n_perm}"
                ),
                y=0.995,
            )
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
            savefig(
                fig,
                plots_dir
                / f"power_tradeoff_sigma{_fmt_token(sigma)}_pi{_fmt_token(pi)}_beta{beta_label}.png",
            )
            plt.close(fig)


def plot_abstention(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    N_ref: int,
    sigma_eta_grid: list[float],
    pi_grid: list[float],
    D_grid: list[int],
    plots_dir: Path,
) -> None:
    # frac_underpowered vs pi at N_ref.
    fig, ax = plt.subplots(figsize=(9.2, 4.2), constrained_layout=False)
    x = np.arange(len(pi_grid))
    colors = {
        float(s): c
        for s, c in zip(sigma_eta_grid, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    }
    styles = ["-", "--", ":", "-."]

    grp = (
        metrics.loc[metrics["N"] == int(N_ref)]
        .groupby(["sigma_eta", "D", "pi_target"], as_index=False)
        .agg(
            frac_underpowered=("underpowered", "mean"), n_genes=("underpowered", "size")
        )
    )
    for d_idx, D in enumerate(D_grid):
        for sigma in sigma_eta_grid:
            ys = []
            for pi in pi_grid:
                cell = grp.loc[
                    (grp["sigma_eta"] == float(sigma))
                    & (grp["D"] == int(D))
                    & (grp["pi_target"] == float(pi))
                ]
                ys.append(
                    float(cell["frac_underpowered"].iloc[0])
                    if not cell.empty
                    else np.nan
                )
            ax.plot(
                x,
                ys,
                marker="o",
                linewidth=1.4,
                linestyle=styles[d_idx % len(styles)],
                color=colors.get(float(sigma), "#444444"),
                label=f"D={int(D)}, sigma={_fmt_float(sigma)}",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([_fmt_float(pi) for pi in pi_grid])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("pi_target")
    ax.set_ylabel("frac underpowered")
    ax.set_title(
        "ExpC abstention boundary at N=20000\n"
        "(donor-stratified permutation, plus-one p-values)"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    savefig(fig, plots_dir / "abstention_vs_pi_N20000.png")
    plt.close(fig)

    # feasibility heatmap for each sigma_eta.
    for sigma in sigma_eta_grid:
        mat = np.full((len(pi_grid), len(D_grid)), np.nan, dtype=float)
        nmat = np.full((len(pi_grid), len(D_grid)), np.nan, dtype=float)
        for i, pi in enumerate(pi_grid):
            for j, D in enumerate(D_grid):
                cell = grp.loc[
                    (grp["sigma_eta"] == float(sigma))
                    & (grp["D"] == int(D))
                    & (grp["pi_target"] == float(pi))
                ]
                if not cell.empty:
                    mat[i, j] = float(cell["frac_underpowered"].iloc[0])
                    nmat[i, j] = float(cell["n_genes"].iloc[0])

        fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=False)
        im = _annotate_heatmap(
            ax,
            mat,
            nmat,
            [_fmt_float(pi) for pi in pi_grid],
            [str(int(d)) for d in D_grid],
            (
                "ExpC feasibility map (frac underpowered)\n"
                f"N={int(N_ref)}, sigma_eta={_fmt_float(sigma)}, donor-stratified permutation"
            ),
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
        ax.set_xlabel("D")
        ax.set_ylabel("pi_target")
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("frac underpowered")
        savefig(fig, plots_dir / f"underpowered_heatmap_sigma{_fmt_token(sigma)}.png")
        plt.close(fig)


def plot_power_curves(
    summary: pd.DataFrame,
    *,
    sigma_eta_grid: list[float],
    pi_grid: list[float],
    beta_grid: list[float],
    D_focus: list[int],
    N_ref: int,
    plots_dir: Path,
    n_perm: int,
) -> None:
    betas_pos = [float(b) for b in beta_grid if float(b) > 0.0]
    if not betas_pos:
        return

    for sigma in sigma_eta_grid:
        for D in D_focus:
            fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=False)
            for pi in pi_grid:
                ys = []
                ns = []
                for beta in betas_pos:
                    row = summary.loc[
                        (summary["N"] == int(N_ref))
                        & (summary["D"] == int(D))
                        & (summary["sigma_eta"] == float(sigma))
                        & (summary["pi_target"] == float(pi))
                        & (summary["beta"] == float(beta))
                    ]
                    if row.empty:
                        ys.append(np.nan)
                        ns.append(np.nan)
                    else:
                        ys.append(float(row["power_alpha05"].iloc[0]))
                        ns.append(float(row["n_non_underpowered"].iloc[0]))
                ax.plot(
                    betas_pos,
                    ys,
                    marker="o",
                    linewidth=1.5,
                    label=f"pi={_fmt_float(pi)}",
                )
                for x, y, n in zip(betas_pos, ys, ns):
                    if np.isfinite(y) and np.isfinite(n):
                        ax.text(x, y + 0.02, f"n={int(n)}", fontsize=6, ha="center")

            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("beta")
            ax.set_ylabel("power @ alpha=0.05")
            ax.grid(True, axis="y", linestyle=":", alpha=0.35)
            ax.legend(frameon=False)
            ax.set_title(
                (
                    "ExpC effect-size power curves\n"
                    f"N={int(N_ref)}, D={int(D)}, sigma_eta={_fmt_float(sigma)}, n_perm={n_perm}, "
                    "donor-stratified permutation"
                )
            )
            fig.tight_layout()
            savefig(
                fig, plots_dir / f"power_curves_sigma{_fmt_token(sigma)}_D{int(D)}.png"
            )
            plt.close(fig)


def plot_typei_heatmaps(
    summary: pd.DataFrame,
    *,
    sigma_eta_grid: list[float],
    pi_grid: list[float],
    D_grid: list[int],
    N_ref: int,
    plots_dir: Path,
) -> None:
    for sigma in sigma_eta_grid:
        mat, nmat = _matrix_from_summary(
            summary,
            row_vals=[float(pi) for pi in pi_grid],
            col_vals=[int(d) for d in D_grid],
            row_key="pi_target",
            col_key="D",
            value_col="typeI_alpha05",
            n_col="n_non_underpowered",
            filters={"N": int(N_ref), "sigma_eta": float(sigma), "beta": 0.0},
        )
        fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=False)
        im = _annotate_heatmap(
            ax,
            mat,
            nmat,
            [_fmt_float(pi) for pi in pi_grid],
            [str(int(d)) for d in D_grid],
            (
                "ExpC Type I sanity (beta=0)\n"
                f"N={int(N_ref)}, sigma_eta={_fmt_float(sigma)}, donor-stratified permutation"
            ),
            vmin=0.0,
            vmax=0.15,
            cmap="viridis",
        )
        ax.set_xlabel("D")
        ax.set_ylabel("pi_target")
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Type I @ alpha=0.05")
        savefig(fig, plots_dir / f"typeI_heatmap_beta0_sigma{_fmt_token(sigma)}.png")
        plt.close(fig)


def plot_deff_diagnostics(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    pi_grid: list[float],
    sigma_eta_grid: list[float],
    N_ref: int,
    beta_grid: list[float],
    plots_dir: Path,
) -> None:
    if metrics.empty or "D_eff" not in metrics.columns:
        return
    pi_ultra = float(min(pi_grid)) if pi_grid else 0.01
    sub = metrics.loc[np.isclose(metrics["pi_target"], pi_ultra)].copy()
    if not sub.empty:
        fig, axes = plt.subplots(
            1,
            max(1, len(sigma_eta_grid)),
            figsize=(3.8 * max(1, len(sigma_eta_grid)), 3.6),
            constrained_layout=False,
            squeeze=False,
        )
        for idx, sigma in enumerate(sigma_eta_grid):
            ax = axes[0, idx]
            ssub = sub.loc[np.isclose(sub["sigma_eta"], float(sigma))].copy()
            if ssub.empty:
                ax.set_axis_off()
                continue
            d_vals = sorted(int(x) for x in ssub["D"].unique().tolist())
            data = [
                ssub.loc[ssub["D"] == d, "D_eff"].to_numpy(dtype=float) for d in d_vals
            ]
            ax.boxplot(data, tick_labels=[str(d) for d in d_vals], showfliers=False)
            ax.set_title(f"sigma_eta={_fmt_float(sigma)}")
            ax.set_xlabel("Nominal D")
            ax.set_ylabel("D_eff")
            ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        fig.suptitle(
            f"ExpC D_eff distribution at ultra-rare prevalence (pi={_fmt_float(pi_ultra)})",
            y=0.995,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
        savefig(
            fig, plots_dir / f"deff_distribution_ultrarare_pi{_fmt_token(pi_ultra)}.png"
        )
        plt.close(fig)

    if summary.empty:
        return
    beta_pos = [float(b) for b in beta_grid if float(b) > 0.0]
    beta_focus = min(beta_pos or [1.0], key=lambda x: abs(x - 1.0))
    rows = summary.loc[
        (summary["N"] == int(N_ref))
        & np.isclose(summary["pi_target"], pi_ultra)
        & np.isclose(summary["beta"], beta_focus)
    ].copy()
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=False)
    for sigma in sigma_eta_grid:
        ssub = rows.loc[np.isclose(rows["sigma_eta"], float(sigma))].sort_values(
            "mean_D_eff"
        )
        if ssub.empty:
            continue
        x = pd.to_numeric(ssub["mean_D_eff"], errors="coerce").to_numpy(dtype=float)
        y_cond = pd.to_numeric(
            ssub.get("conditional_power_alpha05", ssub.get("power_alpha05")),
            errors="coerce",
        ).to_numpy(dtype=float)
        y_oper = pd.to_numeric(
            ssub.get("operational_power_alpha05"), errors="coerce"
        ).to_numpy(dtype=float)
        ax.plot(
            x,
            y_cond,
            marker="o",
            linewidth=1.4,
            label=f"sigma={_fmt_float(sigma)} cond",
        )
        if np.any(np.isfinite(y_oper)):
            ax.plot(
                x,
                y_oper,
                marker="s",
                linewidth=1.2,
                linestyle="--",
                label=f"sigma={_fmt_float(sigma)} oper",
            )
    ax.set_xlabel("mean D_eff")
    ax.set_ylabel("power @ alpha=0.05")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.set_title(
        f"ExpC power vs D_eff (N={int(N_ref)}, pi={_fmt_float(pi_ultra)}, beta={_fmt_float(beta_focus)})"
    )
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    savefig(
        fig,
        plots_dir
        / f"power_vs_deff_N{int(N_ref)}_pi{_fmt_token(pi_ultra)}_beta{_fmt_token(beta_focus)}.png",
    )
    plt.close(fig)


def run_validations(summary: pd.DataFrame) -> None:
    typei_col = (
        "typeI_alpha05_conditional"
        if "typeI_alpha05_conditional" in summary.columns
        else "typeI_alpha05"
    )
    power_col = (
        "conditional_power_alpha05"
        if "conditional_power_alpha05" in summary.columns
        else "power_alpha05"
    )
    offenders_typeI: list[dict[str, Any]] = []
    checked_typei = 0
    typei_slice = summary.loc[
        (summary["beta"] == 0.0)
        & (summary["pi_target"] >= 0.05)
        & (summary["D"] >= 5)
        & (summary["n_non_underpowered"] >= 5)
    ]
    for _, row in typei_slice.iterrows():
        checked_typei += 1
        val = float(row[typei_col]) if pd.notna(row[typei_col]) else np.nan
        if np.isfinite(val) and not (0.03 <= val <= 0.07):
            offenders_typeI.append(
                {
                    "N": int(row["N"]),
                    "D": int(row["D"]),
                    "sigma_eta": float(row["sigma_eta"]),
                    "pi_target": float(row["pi_target"]),
                    "typeI_alpha05": val,
                    "n_non_underpowered": int(row["n_non_underpowered"]),
                }
            )

    if checked_typei == 0:
        print(
            "Validation warning: Type I check skipped (no cells met n_non_underpowered >= 5 criterion).",
            flush=True,
        )
    elif offenders_typeI:
        print(
            "WARNING: Type I outside [0.03,0.07] in some well-powered beta=0 cells:",
            flush=True,
        )
        print(pd.DataFrame(offenders_typeI).to_string(index=False), flush=True)
    else:
        print(
            "Validation OK: Type I within [0.03,0.07] for checked well-powered beta=0 cells.",
            flush=True,
        )

    offenders_monotonic: list[dict[str, Any]] = []
    checked_mono = 0
    grp_cols = ["N", "D", "sigma_eta", "pi_target"]
    for keys, grp in summary.loc[
        (summary["beta"] > 0.0) & (summary["pi_target"] >= 0.05) & (summary["D"] >= 5)
    ].groupby(grp_cols):
        sub = grp.sort_values("beta")
        beta = pd.to_numeric(sub["beta"], errors="coerce").to_numpy(dtype=float)
        power = pd.to_numeric(sub[power_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(beta) & np.isfinite(power)
        if int(np.sum(mask)) < 3:
            continue
        checked_mono += 1
        if np.allclose(power[mask], power[mask][0]):
            rho = float("nan")
        else:
            rho, _ = spearmanr(beta[mask], power[mask], nan_policy="omit")
        if rho is None or not np.isfinite(float(rho)) or float(rho) <= 0.7:
            N, D, sigma, pi = keys
            offenders_monotonic.append(
                {
                    "N": int(N),
                    "D": int(D),
                    "sigma_eta": float(sigma),
                    "pi_target": float(pi),
                    "rho_beta_power": float(rho) if rho is not None else float("nan"),
                }
            )

    if checked_mono == 0:
        print(
            "Validation warning: monotonicity check skipped (need >=3 beta points with finite power per cell).",
            flush=True,
        )
    elif offenders_monotonic:
        print(
            "WARNING: power-vs-beta monotonicity soft-check failed (Spearman rho <= 0.7) for cells:",
            flush=True,
        )
        print(pd.DataFrame(offenders_monotonic).to_string(index=False), flush=True)
    else:
        print(
            "Validation OK: power-vs-beta Spearman rho > 0.7 for checked cells.",
            flush=True,
        )


def run_experiment(args: argparse.Namespace) -> None:
    test_cfg = (
        get_testmode_config(int(args.master_seed)) if bool(args.test_mode) else None
    )
    testmode_banner(bool(args.test_mode), test_cfg)

    outdir = resolve_outdir(args.outdir, bool(args.test_mode))
    results_dir = ensure_dir(outdir / "results")
    plots_dir = ensure_dir(outdir / "plots")

    if bool(args.test_mode):
        N_grid = [3000, 6000]
        D_grid = [2, 5]
        sigma_eta_grid = [0.4]
        pi_grid = [0.05, 0.2]
        beta_grid = [0.0, 0.5, 1.0]
        genes_per_condition = 12
        n_perm = 120
        n_master_seeds = 1
        min_perm = min(int(args.min_perm), int(n_perm))
    else:
        N_grid = _parse_int_list(args.N_grid)
        D_grid = _parse_int_list(args.D_grid)
        sigma_eta_grid = _parse_float_list(args.sigma_eta_grid)
        pi_grid = _parse_float_list(args.pi_grid)
        beta_grid = _parse_float_list(args.beta_grid)
        genes_per_condition = int(args.genes_per_condition)
        n_perm = int(args.n_perm)
        n_master_seeds = int(args.n_master_seeds)
        min_perm = int(args.min_perm)

    seed_values = [int(args.master_seed) + i for i in range(int(n_master_seeds))]
    dataset_grid = build_grid(N_grid, D_grid, sigma_eta_grid, seed_values)

    config = {
        "experiment": "Simulation Experiment C: Power surfaces + abstention boundaries",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit_short": git_commit_hash(cwd=REPO_ROOT),
        "master_seed": int(args.master_seed),
        "seed_values": [int(x) for x in seed_values],
        "n_master_seeds": int(n_master_seeds),
        "N_grid": [int(x) for x in N_grid],
        "D_grid": [int(x) for x in D_grid],
        "sigma_eta_grid": [float(x) for x in sigma_eta_grid],
        "pi_grid": [float(x) for x in pi_grid],
        "beta_grid": [float(x) for x in beta_grid],
        "genes_per_condition": int(genes_per_condition),
        "bins": int(args.bins),
        "n_perm": int(n_perm),
        "test_mode": bool(args.test_mode),
        "angles_origin": [0.0, 0.0],
        "geometry": "disk_gaussian",
        "statistic": "T = max_theta |E(theta)|",
        "permutation": "donor-stratified, plus-one correction",
        "underpowered": {
            "prev_obs_lt": float(args.prev_floor),
            "n_fg_total_lt": int(args.min_fg_total),
            "D_eff_lt": int(args.d_eff_min),
            "min_fg_per_donor": int(args.min_fg_per_donor),
            "min_bg_per_donor": int(args.min_bg_per_donor),
            "n_perm_lt": int(min_perm),
        },
    }
    write_config(outdir, config)

    total_conditions = (
        len(dataset_grid) * len(pi_grid) * len(beta_grid) * int(genes_per_condition)
    )
    print(
        (
            f"ExpC work plan: dataset_conditions={len(dataset_grid)}, "
            f"genes_per_condition={genes_per_condition}, total_gene_replicates={total_conditions}, "
            f"n_perm={n_perm}, bins={args.bins}"
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    example_payload: dict[str, Any] | None = None
    total_start = time.time()
    gene_counter = 0

    for ds_idx, (seed_run, N, D, sigma_eta) in enumerate(dataset_grid, start=1):
        seed_dataset = stable_seed(
            int(args.master_seed),
            "dataset",
            int(seed_run),
            int(N),
            int(D),
            float(sigma_eta),
        )
        rng_ds = rng_from_seed(seed_dataset)

        X_sim, _ = sample_disk_gaussian(int(N), rng_ds)
        theta = np.mod(np.arctan2(X_sim[:, 1], X_sim[:, 0]), 2.0 * np.pi)
        donor_ids = assign_donors(int(N), int(D), rng_ds)
        eta_d = sample_donor_effects(int(D), float(sigma_eta), rng_ds)
        eta_cell = donor_effect_vector(donor_ids, eta_d)
        bin_id, bin_counts_total = _compute_bin_cache(theta, int(args.bins))

        run_id = f"N{int(N)}_D{int(D)}_sigma{_fmt_float(sigma_eta)}_seed{int(seed_run)}"
        print(
            f"[{ds_idx}/{len(dataset_grid)}] {run_id} (genes={len(pi_grid)*len(beta_grid)*genes_per_condition})",
            flush=True,
        )

        for pi_target in pi_grid:
            for beta in beta_grid:
                for rep in range(int(genes_per_condition)):
                    gene_counter += 1
                    seed_gene = stable_seed(
                        int(args.master_seed),
                        "gene",
                        int(seed_run),
                        int(N),
                        int(D),
                        float(sigma_eta),
                        float(pi_target),
                        float(beta),
                        int(rep),
                    )
                    rng_gene = rng_from_seed(seed_gene)
                    theta0 = float(rng_gene.uniform(0.0, 2.0 * np.pi))

                    f = simulate_unimodal_gene(
                        pi_target=float(pi_target),
                        beta=float(beta),
                        theta=theta,
                        theta0=theta0,
                        donor_eta_per_cell=eta_cell,
                        rng=rng_gene,
                    )

                    prev_obs = float(np.mean(f))
                    n_fg_total = int(np.sum(f))

                    power = evaluate_underpowered(
                        donor_ids=donor_ids,
                        f=f,
                        n_perm=int(n_perm),
                        p_min=float(args.prev_floor),
                        min_fg_total=int(args.min_fg_total),
                        min_fg_per_donor=int(args.min_fg_per_donor),
                        min_bg_per_donor=int(args.min_bg_per_donor),
                        d_eff_min=int(args.d_eff_min),
                        min_perm=int(min_perm),
                    )
                    underpowered = bool(power["underpowered"])
                    abstain_reasons = list(power.get("abstain_reasons", []))
                    seed_perm = None

                    t_obs = float("nan")
                    p_t = float("nan")
                    if not underpowered:
                        seed_perm = stable_seed(
                            int(args.master_seed),
                            "perm",
                            int(seed_run),
                            int(N),
                            int(D),
                            float(sigma_eta),
                            float(pi_target),
                            float(beta),
                            int(rep),
                        )
                        perm = perm_null_T(
                            f=f,
                            angles=theta,
                            donor_ids=donor_ids,
                            n_bins=int(args.bins),
                            n_perm=int(n_perm),
                            seed=int(seed_perm),
                            mode="raw",
                            smooth_w=1,
                            donor_stratified=True,
                            return_null_T=False,
                            return_obs_profile=False,
                            bin_id=bin_id,
                            bin_counts_total=bin_counts_total,
                        )
                        t_obs = float(perm["T_obs"])
                        p_t = float(perm["p_T"])

                    rows.append(
                        {
                            "run_id": run_id,
                            "seed": int(seed_gene),
                            "N": int(N),
                            "D": int(D),
                            "sigma_eta": float(sigma_eta),
                            "pi_target": float(pi_target),
                            "beta": float(beta),
                            "theta0": theta0,
                            "prev_obs": prev_obs,
                            "n_fg_total": n_fg_total,
                            "D_eff": int(power["D_eff"]),
                            "underpowered": underpowered,
                            "n_analyzable_flag": bool(not underpowered),
                            "abstain_reasons": json.dumps(
                                abstain_reasons, separators=(",", ":")
                            ),
                            "underpowered_reasons": json.dumps(
                                dict(power.get("underpowered_reasons", {})),
                                separators=(",", ":"),
                            ),
                            "donor_fg_min": float(
                                power.get("donor_fg_min", float("nan"))
                            ),
                            "donor_fg_median": float(
                                power.get(
                                    "donor_fg_median",
                                    power.get("donor_fg_med", float("nan")),
                                )
                            ),
                            "donor_fg_max": float(
                                power.get("donor_fg_max", float("nan"))
                            ),
                            "donor_bg_min": float(
                                power.get("donor_bg_min", float("nan"))
                            ),
                            "donor_bg_median": float(
                                power.get(
                                    "donor_bg_median",
                                    power.get("donor_bg_med", float("nan")),
                                )
                            ),
                            "donor_bg_max": float(
                                power.get("donor_bg_max", float("nan"))
                            ),
                            "seed_perm": (
                                int(seed_perm) if seed_perm is not None else np.nan
                            ),
                            "T_obs": t_obs,
                            "p_T": p_t,
                        }
                    )
                    if (example_payload is None) and (not underpowered):
                        example_payload = {
                            "X": np.asarray(X_sim, dtype=float),
                            "f": np.asarray(f, dtype=bool),
                            "theta": np.asarray(theta, dtype=float),
                            "bin_id": np.asarray(bin_id, dtype=np.int32),
                            "bin_counts_total": np.asarray(
                                bin_counts_total, dtype=np.int64
                            ),
                            "bins": int(args.bins),
                            "run_id": run_id,
                            "pi_target": float(pi_target),
                            "beta": float(beta),
                        }

                    if (
                        gene_counter % int(args.progress_every) == 0
                        or gene_counter == total_conditions
                    ):
                        elapsed = time.time() - total_start
                        rate = gene_counter / elapsed if elapsed > 0 else float("nan")
                        print(
                            f"  progress: {gene_counter}/{total_conditions} genes ({rate:.2f} genes/s)",
                            flush=True,
                        )

    metrics = pd.DataFrame(rows)
    summary = summarize(metrics)

    atomic_write_csv(results_dir / "metrics_long.csv", metrics)
    atomic_write_csv(results_dir / "summary.csv", summary)

    _set_plot_style()
    plot_power_heatmaps(
        summary,
        sigma_eta_grid=sigma_eta_grid,
        pi_grid=pi_grid,
        D_grid=D_grid,
        beta_grid=beta_grid,
        N_ref=20000,
        plots_dir=plots_dir,
        n_perm=int(n_perm),
        bins=int(args.bins),
        genes_per_condition=int(genes_per_condition),
    )
    _plot_metric_heatmaps(
        summary=summary,
        sigma_eta_grid=sigma_eta_grid,
        pi_grid=pi_grid,
        D_grid=D_grid,
        beta_grid=beta_grid,
        N_ref=20000,
        plots_dir=plots_dir,
        n_perm=int(n_perm),
        bins=int(args.bins),
        genes_per_condition=int(genes_per_condition),
        value_col="conditional_power_alpha05",
        n_col="n_non_underpowered",
        out_prefix="conditional_power_heatmap",
        title_metric="conditional power @ alpha=0.05",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    _plot_metric_heatmaps(
        summary=summary,
        sigma_eta_grid=sigma_eta_grid,
        pi_grid=pi_grid,
        D_grid=D_grid,
        beta_grid=beta_grid,
        N_ref=20000,
        plots_dir=plots_dir,
        n_perm=int(n_perm),
        bins=int(args.bins),
        genes_per_condition=int(genes_per_condition),
        value_col="operational_power_alpha05",
        n_col="n_genes",
        out_prefix="operational_power_heatmap",
        title_metric="operational power @ alpha=0.05",
        vmin=0.0,
        vmax=1.0,
        cmap="plasma",
    )
    _plot_metric_heatmaps(
        summary=summary,
        sigma_eta_grid=sigma_eta_grid,
        pi_grid=pi_grid,
        D_grid=D_grid,
        beta_grid=beta_grid,
        N_ref=20000,
        plots_dir=plots_dir,
        n_perm=int(n_perm),
        bins=int(args.bins),
        genes_per_condition=int(genes_per_condition),
        value_col="analyzable_rate",
        n_col="n_genes",
        out_prefix="analyzable_rate_heatmap",
        title_metric="analyzable rate",
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
    )
    plot_tradeoff(
        summary,
        sigma_eta_grid=sigma_eta_grid,
        pi_grid=pi_grid,
        D_grid=D_grid,
        N_grid=N_grid,
        beta_grid=beta_grid,
        plots_dir=plots_dir,
        n_perm=int(n_perm),
    )
    plot_abstention(
        metrics,
        summary,
        N_ref=20000,
        sigma_eta_grid=sigma_eta_grid,
        pi_grid=pi_grid,
        D_grid=D_grid,
        plots_dir=plots_dir,
    )
    plot_power_curves(
        summary,
        sigma_eta_grid=sigma_eta_grid,
        pi_grid=pi_grid,
        beta_grid=beta_grid,
        D_focus=[d for d in [5, 10] if d in D_grid],
        N_ref=20000,
        plots_dir=plots_dir,
        n_perm=int(n_perm),
    )
    plot_typei_heatmaps(
        summary,
        sigma_eta_grid=sigma_eta_grid,
        pi_grid=pi_grid,
        D_grid=D_grid,
        N_ref=20000,
        plots_dir=plots_dir,
    )
    plot_deff_diagnostics(
        metrics,
        summary,
        pi_grid=pi_grid,
        sigma_eta_grid=sigma_eta_grid,
        N_ref=20000,
        beta_grid=beta_grid,
        plots_dir=plots_dir,
    )

    run_validations(summary)

    if example_payload is not None:
        fig_e, ax_e = plt.subplots(figsize=(5.2, 4.4), constrained_layout=False)
        plot_embedding_with_foreground(
            example_payload["X"],
            example_payload["f"],
            ax=ax_e,
            title=(
                "Representative embedding (ExpC)\n"
                f"{example_payload['run_id']}, pi={_fmt_float(example_payload['pi_target'])}, "
                f"beta={_fmt_float(example_payload['beta'])}"
            ),
            s=5.0,
            alpha_bg=0.28,
            alpha_fg=0.75,
        )
        savefig(fig_e, plots_dir / "embedding_example.png")
        plt.close(fig_e)

        E_obs, _, _ = compute_rsp_profile_from_boolean(
            example_payload["f"],
            example_payload["theta"],
            int(example_payload["bins"]),
            bin_id=example_payload["bin_id"],
            bin_counts_total=example_payload["bin_counts_total"],
        )
        theta_centers = np.linspace(
            0.0, 2.0 * np.pi, int(example_payload["bins"]), endpoint=False
        )
        fig_p, ax_p = plt.subplots(
            figsize=(5.2, 4.8),
            subplot_kw={"projection": "polar"},
            constrained_layout=False,
        )
        plot_rsp_polar(
            theta_centers,
            np.asarray(E_obs, dtype=float),
            ax=ax_p,
            title=(
                "Representative RSP profile (ExpC)\n"
                f"{example_payload['run_id']}, pi={_fmt_float(example_payload['pi_target'])}, "
                f"beta={_fmt_float(example_payload['beta'])}"
            ),
        )
        savefig(fig_p, plots_dir / "polar_rsp_example.png")
        plt.close(fig_p)
    write_report_expC(
        outdir=outdir,
        metrics_path=results_dir / "metrics_long.csv",
        summary_path=results_dir / "summary.csv",
    )

    elapsed = time.time() - total_start
    print(
        (
            f"Completed ExpC in {elapsed/60.0:.2f} min. "
            f"rows={metrics.shape[0]}, non_underpowered={int((~metrics['underpowered']).sum())}"
        ),
        flush=True,
    )
    print(f"Outputs written to: {outdir}", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Simulation Experiment C: power surfaces (effect size x prevalence x donors) "
            "with donor-aware permutation p-values."
        )
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/simulations/expC_power_surfaces",
        help="Output directory.",
    )
    parser.add_argument("--master_seed", type=int, default=123, help="Master seed.")
    parser.add_argument(
        "--n_master_seeds",
        type=int,
        default=3,
        help="Number of consecutive seeds to pool.",
    )
    parser.add_argument(
        "--n_perm", type=int, default=500, help="Permutation count per gene."
    )
    parser.add_argument(
        "--bins", type=int, default=36, help="Angular bin count for RSP profile."
    )
    parser.add_argument(
        "--genes_per_condition",
        type=int,
        default=200,
        help="Gene replicates per (N,D,sigma,pi,beta).",
    )

    parser.add_argument(
        "--N_grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_N_GRID),
        help="Comma-separated N grid.",
    )
    parser.add_argument(
        "--D_grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_D_GRID),
        help="Comma-separated donor grid.",
    )
    parser.add_argument(
        "--sigma_eta_grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SIGMA_ETA_GRID),
        help="Comma-separated sigma_eta grid.",
    )
    parser.add_argument(
        "--pi_grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_PI_GRID),
        help="Comma-separated prevalence grid.",
    )
    parser.add_argument(
        "--beta_grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BETA_GRID),
        help="Comma-separated beta grid.",
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
        "--prev_floor",
        type=float,
        default=0.005,
        help="Underpowered minimum observed prevalence.",
    )
    parser.add_argument(
        "--d_eff_min", type=int, default=5, help="Underpowered min informative donors."
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
        help="Progress logging frequency (genes).",
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
    if (
        args.n_perm <= 0
        or args.bins <= 0
        or args.genes_per_condition <= 0
        or args.n_master_seeds <= 0
    ):
        raise ValueError(
            "n_perm, bins, genes_per_condition, and n_master_seeds must be positive."
        )
    if not (0.0 <= float(args.prev_floor) < 1.0):
        raise ValueError("prev_floor must be in [0, 1).")
    run_ctx = prepare_legacy_run(args, "expC_power_surfaces", __file__)
    run_experiment(args)
    finalize_legacy_run(run_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
