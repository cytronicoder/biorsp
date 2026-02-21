#!/usr/bin/env python3
"""Simulation Experiment B: max-stat validity + binning/smoothing sensitivity."""

from __future__ import annotations

import argparse
import json
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
from scipy.stats import kstest

# Ensure non-interactive rendering and writable cache dirs in sandboxed runs.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-biorsp-expB")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache-biorsp-expB")
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from biorsp.permutation import (
    check_mode_consistency,
    mode_max_stat_from_profiles,
    perm_null_T,
    perm_null_T_and_profile,
)
from biorsp.power import evaluate_underpowered
from biorsp.rsp import compute_rsp_profile_from_boolean
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
    write_config,
)
from experiments.simulations._shared.io import (
    cache_get as io_cache_get,
)
from experiments.simulations._shared.io import (
    cache_set as io_cache_set,
)
from experiments.simulations._shared.models import simulate_null_gene
from experiments.simulations._shared.parallel import parallel_map
from experiments.simulations._shared.plots import (
    plot_embedding_with_foreground,
    plot_rsp_polar,
    savefig,
)
from experiments.simulations._shared.plots import (
    wilson_ci as shared_wilson_ci,
)
from experiments.simulations._shared.reporting import write_report_expB
from experiments.simulations._shared.runner import (
    finalize_legacy_run,
    prepare_legacy_run,
)
from experiments.simulations._shared.seeding import rng_from_seed, stable_seed
from experiments.simulations._sim_testmode import (
    banner as testmode_banner,
)
from experiments.simulations._sim_testmode import (
    get_testmode_config,
    resolve_outdir,
)
from experiments.simulations.expB_maxstat_sensitivity.cell_runner_b import (
    EdgeRuleConfigB,
    apply_foreground_edge_rule_b,
)

DEFAULT_GEOMETRIES = ["disk_gaussian", "ring_annulus", "density_gradient_disk"]
DEFAULT_D_GRID = [5, 10]
DEFAULT_PI_BINS = [0.01, 0.05, 0.2, 0.6, 0.9]
DEFAULT_BINS_GRID = [24, 36, 48, 72]
DEFAULT_SMOOTH_GRID = [1, 3, 5]
DEFAULT_MODES = ["raw", "smoothed"]


@dataclass(frozen=True)
class DatasetContext:
    geometry: str
    D: int
    X_sim: np.ndarray
    angles: np.ndarray
    donor_ids: np.ndarray
    eta_d: np.ndarray
    run_id: str
    seed_dataset: int
    bin_cache: dict[int, tuple[np.ndarray, np.ndarray]]
    cache_hit: bool = False


def _parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer in a comma-separated list.")
    return vals


def _parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one float in a comma-separated list.")
    return vals


def _parse_str_list(text: str) -> list[str]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one token in a comma-separated list.")
    return vals


def _normalize_smooth_grid(values: list[int]) -> tuple[list[int], str]:
    raw = [int(v) for v in values]
    if not raw:
        raise ValueError("smooth_grid cannot be empty.")
    if any(v < 0 for v in raw):
        raise ValueError("smooth_grid entries must be non-negative.")

    # Backward compatibility: historical configs used radius values (e.g., 0,1,2),
    # while current permutation code expects odd window widths (1,3,5).
    if 0 in raw:
        norm = sorted(set((2 * v) + 1 for v in raw))
        return norm, "radius_to_window"

    invalid = [v for v in raw if v < 1 or v % 2 == 0]
    if invalid:
        raise ValueError(
            "smooth_grid must contain odd integers >=1 (or include 0 to use legacy radius mode)."
        )
    return sorted(set(raw)), "window"


def _seed_from_fields(master_seed: int, *fields: Any) -> int:
    return int(stable_seed(int(master_seed), "expB", *fields))


def _fmt_pi(pi: float) -> str:
    return f"{float(pi):.3f}".rstrip("0").rstrip(".")


GEO_LABEL = {
    "disk_gaussian": "Disk (Gaussian)",
    "density_gradient_disk": "Disk (density gradient)",
}

MODE_LABEL = {
    "raw": "Raw",
    "smoothed": "Smoothed",
}


def _humanize_geometry_label(geometry: str) -> str:
    key = str(geometry)
    if key in GEO_LABEL:
        return GEO_LABEL[key]
    return key.replace("_", " ").strip().title()


def _humanize_mode_label(mode: str) -> str:
    key = str(mode).strip().lower()
    return MODE_LABEL.get(key, str(mode).strip().title())


def _heatmap_value_label(value_col: str) -> str:
    labels = {
        "neglog10_ks_pvalue": r"$-\log_{10}(p_{KS})$",
        "typeI_alpha05": r"Type I error ($\alpha=0.05$)",
        "dev_mean_p": r"$\Delta\,\mu(p_T)=\mathbb{E}[p_T]-0.5$",
        "dev_var_p": r"$\Delta\,\mathrm{Var}(p_T)=\mathrm{Var}(p_T)-\frac{1}{12}$",
    }
    return labels.get(str(value_col), str(value_col))


def _git_commit_hash() -> str:
    return git_commit_hash(cwd=REPO_ROOT)


def simulate_geometry(
    geometry: str, n_cells: int, rng: np.random.Generator
) -> np.ndarray:
    x, _ = sample_geometry(geometry, n_cells, rng)
    return x


def _build_bin_cache(
    angles: np.ndarray, bins_grid: list[int]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    # Keep canonical theta in [-pi, pi) externally, but wrap for [0, 2pi) binning.
    wrapped = np.mod(np.asarray(angles, dtype=float).ravel(), 2.0 * np.pi)
    for b in bins_grid:
        edges = np.linspace(0.0, 2.0 * np.pi, int(b) + 1, endpoint=True)
        bin_id = np.digitize(wrapped, edges, right=False) - 1
        bin_id = np.where(bin_id == int(b), int(b) - 1, bin_id).astype(np.int32)
        counts = np.bincount(bin_id, minlength=int(b)).astype(np.int64)
        cache[int(b)] = (bin_id, counts)
    return cache


def _assert_valid_n_bins(n_bins: int) -> None:
    if int(n_bins) < 8:
        raise AssertionError(
            f"n_bins sanity check failed: expected >=8, got {int(n_bins)}"
        )


def _build_donor_index_lists(donor_ids: np.ndarray) -> list[np.ndarray]:
    donor_arr = np.asarray(donor_ids).ravel()
    uniq = np.unique(donor_arr)
    return [np.flatnonzero(donor_arr == d).astype(np.int32) for d in uniq]


def _build_donor_perm_plan(
    donor_ids: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> np.ndarray:
    donor_lists = _build_donor_index_lists(donor_ids)
    n_cells = int(np.asarray(donor_ids).size)
    n_perm_int = int(n_perm)
    base = np.arange(n_cells, dtype=np.int32)
    out = np.empty((n_perm_int, n_cells), dtype=np.int32)
    rng = rng_from_seed(int(seed))
    for i in range(n_perm_int):
        row = base.copy()
        for idx in donor_lists:
            row[idx] = rng.permutation(idx)
        out[i, :] = row
    return out


def _gene_prevalence_schedule(
    g_total: int,
    pi_bins: list[float],
    rng: np.random.Generator,
) -> np.ndarray:
    bins = np.asarray(pi_bins, dtype=float)
    n_bins = bins.size
    if n_bins < 1:
        raise ValueError("At least one prevalence bin is required.")
    base = g_total // n_bins
    rem = g_total % n_bins
    counts = np.full(n_bins, base, dtype=int)
    counts[:rem] += 1
    pi = np.repeat(bins, counts)
    order = rng.permutation(pi.size)
    return pi[order]


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    _ = z  # Compatibility placeholder for previous call signature.
    return shared_wilson_ci(int(k), int(n), alpha=0.05)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _text_color_for_rgba(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, _ = rgba
    luminance = 0.2126 * float(r) + 0.7152 * float(g) + 0.0722 * float(b)
    return "black" if luminance > 0.6 else "white"


def _matrix_from_summary(
    summary: pd.DataFrame,
    *,
    geometry: str,
    d_value: int,
    mode: str,
    pi_target: float,
    bins_grid: list[int],
    smooth_grid: list[int],
    value_col: str,
) -> np.ndarray:
    mat = np.full((len(bins_grid), len(smooth_grid)), np.nan, dtype=float)
    for i, b in enumerate(bins_grid):
        for j, w in enumerate(smooth_grid):
            mask = (
                (summary["geometry"] == geometry)
                & (summary["D"] == int(d_value))
                & (summary["mode"] == mode)
                & (summary["pi_target"] == float(pi_target))
                & (summary["bins_B"] == int(b))
                & (summary["smooth_w"] == int(w))
            )
            if mask.any():
                val = summary.loc[mask, value_col].iloc[0]
                mat[i, j] = float(val) if pd.notna(val) else np.nan
    return mat


def plot_qq_small_multiples(
    metrics: pd.DataFrame,
    *,
    geometries: list[str],
    bins_grid: list[int],
    smooth_grid: list[int],
    modes: list[str],
    d_rep: int,
    pi_rep: float,
    n_perm: int,
    out_png: Path,
    q_max: float = 1.0,
) -> None:
    rows = len(geometries) * len(bins_grid)
    cols = len(modes) * len(smooth_grid)
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.0 * cols, 1.35 * rows), constrained_layout=False
    )
    axes_arr = np.atleast_2d(axes)
    p_label = _fmt_pi(pi_rep)

    for g_idx, geom in enumerate(geometries):
        for b_idx, b in enumerate(bins_grid):
            row = g_idx * len(bins_grid) + b_idx
            for m_idx, mode in enumerate(modes):
                for w_idx, w in enumerate(smooth_grid):
                    col = m_idx * len(smooth_grid) + w_idx
                    ax = axes_arr[row, col]
                    sub = metrics.loc[
                        (metrics["geometry"] == geom)
                        & (metrics["D"] == int(d_rep))
                        & (metrics["mode"] == mode)
                        & (metrics["pi_target"] == float(pi_rep))
                        & (metrics["bins_B"] == int(b))
                        & (metrics["smooth_w"] == int(w))
                        & (~metrics["underpowered"])
                    ]
                    pvals = pd.to_numeric(sub["p_T"], errors="coerce").to_numpy(
                        dtype=float
                    )
                    pvals = pvals[np.isfinite(pvals)]
                    if pvals.size >= 3:
                        p_sorted = np.sort(pvals)
                        expected = (
                            np.arange(1, p_sorted.size + 1, dtype=float) - 0.5
                        ) / p_sorted.size
                        if float(q_max) < 1.0:
                            tail_mask = (expected <= float(q_max)) & (
                                p_sorted <= float(q_max)
                            )
                            expected_plot = expected[tail_mask]
                            p_sorted_plot = p_sorted[tail_mask]
                        else:
                            expected_plot = expected
                            p_sorted_plot = p_sorted
                        if p_sorted_plot.size > 0:
                            ax.plot(
                                expected_plot,
                                p_sorted_plot,
                                ".",
                                color="#1f77b4",
                                markersize=1.5,
                                alpha=0.85,
                            )
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                "no tail pts",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                                color="#666",
                                fontsize=6.8,
                            )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "n<3",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            color="#666",
                        )
                    q_lim = float(np.clip(float(q_max), 1e-3, 1.0))
                    ax.plot(
                        [0, q_lim],
                        [0, q_lim],
                        linestyle="--",
                        color="#444",
                        linewidth=0.7,
                    )
                    ax.set_xlim(0, q_lim)
                    ax.set_ylim(0, q_lim)
                    if row == 0:
                        ax.set_title(f"{_humanize_mode_label(mode)}, w={w}")
                    if col == 0:
                        ax.set_ylabel(f"{_humanize_geometry_label(geom)}\nB={b}")
                    if row == rows - 1:
                        ax.set_xlabel("Expected U(0,1)")
                    ax.text(
                        0.97,
                        0.05,
                        f"n={pvals.size}",
                        ha="right",
                        va="bottom",
                        transform=ax.transAxes,
                        fontsize=6.3,
                    )

    qq_scope = "left-tail" if float(q_max) < 1.0 else "full-range"
    fig.suptitle(
        (
            f"Experiment B strict null QQ ({qq_scope}, non-underpowered)\n"
            f"D={d_rep}, pi={p_label}, n_perm={n_perm}, p_min={1.0/(n_perm+1):.4f}"
        ),
        y=0.997,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    savefig(fig, out_png)
    plt.close(fig)


def plot_p_hist_small_multiples(
    metrics: pd.DataFrame,
    *,
    geometries: list[str],
    bins_grid: list[int],
    smooth_grid: list[int],
    modes: list[str],
    d_rep: int,
    pi_rep: float,
    n_perm: int,
    out_png: Path,
) -> None:
    rows = len(geometries) * len(bins_grid)
    cols = len(modes) * len(smooth_grid)
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.0 * cols, 1.35 * rows), constrained_layout=False
    )
    axes_arr = np.atleast_2d(axes)
    p_label = _fmt_pi(pi_rep)
    hist_bins = np.linspace(0.0, 1.0, 21)

    for g_idx, geom in enumerate(geometries):
        for b_idx, b in enumerate(bins_grid):
            row = g_idx * len(bins_grid) + b_idx
            for m_idx, mode in enumerate(modes):
                for w_idx, w in enumerate(smooth_grid):
                    col = m_idx * len(smooth_grid) + w_idx
                    ax = axes_arr[row, col]
                    sub = metrics.loc[
                        (metrics["geometry"] == geom)
                        & (metrics["D"] == int(d_rep))
                        & (metrics["mode"] == mode)
                        & (metrics["pi_target"] == float(pi_rep))
                        & (metrics["bins_B"] == int(b))
                        & (metrics["smooth_w"] == int(w))
                        & (~metrics["underpowered"])
                    ]
                    pvals = pd.to_numeric(sub["p_T"], errors="coerce").to_numpy(
                        dtype=float
                    )
                    pvals = pvals[np.isfinite(pvals)]
                    if pvals.size > 0:
                        ax.hist(
                            pvals,
                            bins=hist_bins,
                            density=True,
                            color="#4C78A8",
                            alpha=0.85,
                            edgecolor="white",
                            linewidth=0.25,
                        )
                    ax.axhline(1.0, color="#444", linestyle="--", linewidth=0.6)
                    ax.set_xlim(0, 1)
                    if row == 0:
                        ax.set_title(f"{mode}, w={w}")
                    if col == 0:
                        ax.set_ylabel(f"{geom}\nB={b}")
                    if row == rows - 1:
                        ax.set_xlabel("p_T")
                    ax.text(
                        0.97,
                        0.92,
                        f"n={pvals.size}",
                        ha="right",
                        va="top",
                        transform=ax.transAxes,
                        fontsize=6.3,
                    )

    fig.suptitle(
        (
            "Experiment B strict null p-value histograms (non-underpowered)\n"
            f"D={d_rep}, pi={p_label}, n_perm={n_perm}, p_min={1.0/(n_perm+1):.4f}"
        ),
        y=0.997,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    savefig(fig, out_png)
    plt.close(fig)


def _plot_heatmap_panels(
    summary: pd.DataFrame,
    *,
    geometries: list[str],
    modes: list[str],
    pi_bins: list[float],
    bins_grid: list[int],
    smooth_grid: list[int],
    d_rep: int,
    value_col: str,
    title: str,
    out_png: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = "{:.3f}",
) -> None:
    rows = len(geometries) * len(modes)
    cols = len(pi_bins)
    fig = plt.figure(
        figsize=(2.55 * cols + 0.85, 1.9 * rows),
        constrained_layout=False,
    )
    gs = GridSpec(
        nrows=rows,
        ncols=cols + 1,
        figure=fig,
        width_ratios=[1.0] * cols + [0.06],
        left=0.10,
        right=0.92,
        top=0.90,
        bottom=0.09,
        wspace=0.26,
        hspace=0.34,
    )
    axes_arr = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            axes_arr[i, j] = fig.add_subplot(gs[i, j])
    cax = fig.add_subplot(gs[:, cols])
    last_im = None

    for g_idx, geom in enumerate(geometries):
        for m_idx, mode in enumerate(modes):
            row = g_idx * len(modes) + m_idx
            for p_idx, pi in enumerate(pi_bins):
                ax = axes_arr[row, p_idx]
                mat = _matrix_from_summary(
                    summary,
                    geometry=geom,
                    d_value=d_rep,
                    mode=mode,
                    pi_target=pi,
                    bins_grid=bins_grid,
                    smooth_grid=smooth_grid,
                    value_col=value_col,
                )
                im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                last_im = im
                ax.set_xticks(np.arange(len(smooth_grid)))
                ax.set_xticklabels([str(x) for x in smooth_grid])
                ax.set_yticks(np.arange(len(bins_grid)))
                ax.set_yticklabels([str(x) for x in bins_grid])
                if row == rows - 1:
                    ax.set_xlabel(r"$w$")
                if p_idx == 0:
                    ax.set_ylabel(r"$B$")
                    ax.text(
                        -0.52,
                        0.5,
                        f"{_humanize_geometry_label(geom)}\n{_humanize_mode_label(mode)}",
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=7.5,
                    )
                if row == 0:
                    ax.set_title(rf"$\pi_{{\mathrm{{target}}}}={_fmt_pi(pi)}$")
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        val = mat[i, j]
                        txt = "NA" if not np.isfinite(val) else fmt.format(val)
                        if np.isfinite(val):
                            rgba = im.cmap(im.norm(val))
                            color = _text_color_for_rgba(rgba)
                        else:
                            color = "black"
                        ax.text(
                            j,
                            i,
                            txt,
                            ha="center",
                            va="center",
                            fontsize=6.5,
                            color=color,
                        )

    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label(_heatmap_value_label(value_col))
    fig.suptitle(f"{title}\nD={d_rep}, strict null", y=0.997)
    savefig(fig, out_png)
    plt.close(fig)


def plot_underpowered_rate(
    summary: pd.DataFrame,
    *,
    geometries: list[str],
    d_grid: list[int],
    pi_bins: list[float],
    bins_rep: int,
    w_rep: int,
    modes: list[str],
    out_png: Path,
) -> None:
    x = np.arange(len(pi_bins))
    fig, axes = plt.subplots(
        len(geometries),
        1,
        figsize=(9.2, 3.0 * len(geometries)),
        sharex=True,
        constrained_layout=False,
    )
    axes_arr = np.atleast_1d(axes)
    colors = {"raw": "#1f77b4", "smoothed": "#d62728"}
    styles = ["-", "--", ":", "-."]

    for g_idx, geom in enumerate(geometries):
        ax = axes_arr[g_idx]
        for d_idx, d in enumerate(d_grid):
            for mode in modes:
                ys = []
                for pi in pi_bins:
                    sub = summary.loc[
                        (summary["geometry"] == geom)
                        & (summary["D"] == int(d))
                        & (summary["mode"] == mode)
                        & (summary["bins_B"] == int(bins_rep))
                        & (summary["smooth_w"] == int(w_rep))
                        & (summary["pi_target"] == float(pi))
                    ]
                    ys.append(
                        float(sub["frac_underpowered"].iloc[0])
                        if not sub.empty
                        else np.nan
                    )
                ax.plot(
                    x,
                    ys,
                    color=colors[mode],
                    linestyle=styles[d_idx % len(styles)],
                    marker="o",
                    linewidth=1.4,
                    markersize=4.0,
                    label=f"D={d}, mode={mode}",
                )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("frac underpowered")
        ax.set_title(f"{geom}")
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        ax.legend(loc="upper right", frameon=False, ncol=2)

    axes_arr[-1].set_xticks(x)
    axes_arr[-1].set_xticklabels([_fmt_pi(pi) for pi in pi_bins])
    axes_arr[-1].set_xlabel("pi_target")
    fig.suptitle(
        (
            "Experiment B strict null: underpowered rate vs prevalence\n"
            f"Representative sensitivity slice B={bins_rep}, w={w_rep}"
        ),
        y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    savefig(fig, out_png)
    plt.close(fig)


def plot_testmode_ks_heatmap(
    summary: pd.DataFrame,
    *,
    geometries: list[str],
    modes: list[str],
    bins_grid: list[int],
    smooth_grid: list[int],
    d_value: int,
    pi_target: float,
    out_png: Path,
) -> None:
    rows = len(geometries)
    cols = len(modes)
    fig = plt.figure(
        figsize=(3.8 * cols + 0.8, 3.0 * rows),
        constrained_layout=False,
    )
    gs = GridSpec(
        nrows=rows,
        ncols=cols + 1,
        figure=fig,
        width_ratios=[1.0] * cols + [0.06],
        left=0.08,
        right=0.92,
        top=0.88,
        bottom=0.10,
        wspace=0.25,
        hspace=0.25,
    )
    axes_arr = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            axes_arr[i, j] = fig.add_subplot(gs[i, j])
    cax = fig.add_subplot(gs[:, cols])

    vmax = 1.0
    mats_neglog: dict[tuple[int, int], np.ndarray] = {}
    mats_p: dict[tuple[int, int], np.ndarray] = {}

    for i, geom in enumerate(geometries):
        for j, mode in enumerate(modes):
            mat_neglog = _matrix_from_summary(
                summary,
                geometry=geom,
                d_value=d_value,
                mode=mode,
                pi_target=pi_target,
                bins_grid=bins_grid,
                smooth_grid=smooth_grid,
                value_col="neglog10_ks_pvalue",
            )
            mat_p = _matrix_from_summary(
                summary,
                geometry=geom,
                d_value=d_value,
                mode=mode,
                pi_target=pi_target,
                bins_grid=bins_grid,
                smooth_grid=smooth_grid,
                value_col="ks_pvalue",
            )
            mats_neglog[(i, j)] = mat_neglog
            mats_p[(i, j)] = mat_p
            if np.isfinite(mat_neglog).any():
                vmax = max(vmax, float(np.nanmax(mat_neglog)))

    for i, geom in enumerate(geometries):
        for j, mode in enumerate(modes):
            ax = axes_arr[i, j]
            mat_neglog = mats_neglog[(i, j)]
            mat_p = mats_p[(i, j)]
            im = ax.imshow(mat_neglog, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
            ax.set_xticks(np.arange(len(smooth_grid)))
            ax.set_xticklabels([str(x) for x in smooth_grid])
            ax.set_yticks(np.arange(len(bins_grid)))
            ax.set_yticklabels([str(x) for x in bins_grid])
            ax.set_xlabel(r"$w$")
            if j == 0:
                ax.set_ylabel(r"$B$")
                ax.text(
                    -0.46,
                    0.5,
                    _humanize_geometry_label(str(geom)),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=8,
                )
            ax.set_title("Raw" if str(mode).lower() == "raw" else "Smoothed")
            for r in range(mat_neglog.shape[0]):
                for c in range(mat_neglog.shape[1]):
                    v = mat_neglog[r, c]
                    p_raw = mat_p[r, c]
                    if not np.isfinite(v):
                        txt = "NA"
                        color = "black"
                    else:
                        txt = "NA" if not np.isfinite(p_raw) else f"{p_raw:.2f}"
                        rgba = im.cmap(im.norm(v))
                        color = _text_color_for_rgba(rgba)
                    ax.text(
                        c, r, txt, ha="center", va="center", fontsize=7, color=color
                    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$-\log_{10}(p_{KS})$")
    fig.suptitle(
        rf"Experiment B test mode: KS diagnostic (strict null, $\pi_{{\mathrm{{target}}}}={_fmt_pi(pi_target)}$, $D={int(d_value)}$)",
        y=0.995,
    )
    fig.text(
        0.5,
        0.045,
        r"Cell text: $p_{KS}$.  Color: $-\log_{10}(p_{KS})$.",
        ha="center",
        va="center",
        fontsize=7,
        color="0.3",
    )
    fig.text(
        0.5,
        0.022,
        r"Smoothed: moving average with window $w$.",
        ha="center",
        va="center",
        fontsize=7,
        color="0.3",
    )
    savefig(fig, out_png)
    plt.close(fig)


def plot_testmode_p_hist(
    metrics: pd.DataFrame,
    *,
    geometries: list[str],
    modes: list[str],
    bins_grid: list[int],
    smooth_grid: list[int],
    d_value: int,
    pi_target: float,
    out_png: Path,
    reduce_overplot: bool = True,
    reduce_mode: str = "facet_by_B",
) -> None:
    reduce_mode_norm = str(reduce_mode).strip().lower()
    if reduce_mode_norm not in {"facet_by_b", "facet_by_w"}:
        raise ValueError("reduce_mode must be one of {'facet_by_B','facet_by_w'}")

    hist_bins = np.linspace(0.0, 1.0, 11)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
    legend_handles: dict[str, Any] = {}
    if reduce_overplot:
        mode_selected = (
            "smoothed"
            if any(str(m).lower() == "smoothed" for m in modes)
            else str(modes[0])
        )
        if reduce_mode_norm == "facet_by_b":
            rows = len(geometries)
            cols = len(bins_grid)
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(4.2 * cols + 1.0, 2.9 * rows),
                constrained_layout=False,
            )
            axes_arr = np.atleast_2d(axes)
            smooth_show = sorted({int(smooth_grid[0]), int(smooth_grid[-1])})
            for i, geom in enumerate(geometries):
                for j, b in enumerate(bins_grid):
                    ax = axes_arr[i, j]
                    for idx, w in enumerate(smooth_show):
                        sub = metrics.loc[
                            (metrics["geometry"] == geom)
                            & (metrics["D"] == int(d_value))
                            & (metrics["mode"] == str(mode_selected))
                            & (metrics["bins_B"] == int(b))
                            & (metrics["smooth_w"] == int(w))
                            & (metrics["pi_target"] == float(pi_target))
                            & (~metrics["underpowered"])
                        ]
                        pvals = pd.to_numeric(sub["p_T"], errors="coerce").to_numpy(
                            dtype=float
                        )
                        pvals = pvals[np.isfinite(pvals)]
                        if pvals.size == 0:
                            continue
                        label = rf"$w={int(w)}$"
                        edge_color = colors[idx % len(colors)]
                        hist_out = ax.hist(
                            pvals,
                            bins=hist_bins,
                            density=True,
                            histtype="step",
                            linewidth=1.8,
                            color=edge_color,
                            label=label,
                        )
                        patch = hist_out[-1][0] if hist_out[-1] else None
                        if patch is not None and label not in legend_handles:
                            legend_handles[label] = patch
                    ax.axhline(
                        1.0, linestyle="--", color="0.3", linewidth=1.2, alpha=0.8
                    )
                    ax.set_xlim(0.0, 1.0)
                    ax.set_xlabel(r"$p_T$")
                    if j == 0:
                        ax.set_ylabel("Density")
                        ax.text(
                            -0.32,
                            0.5,
                            _humanize_geometry_label(str(geom)),
                            transform=ax.transAxes,
                            rotation=90,
                            va="center",
                            ha="center",
                            fontsize=8,
                        )
                    ax.set_title(rf"$B={int(b)}$")
            legend_title = ""
        else:
            rows = len(geometries)
            cols = len(smooth_grid)
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(4.2 * cols + 1.0, 2.9 * rows),
                constrained_layout=False,
            )
            axes_arr = np.atleast_2d(axes)
            bins_show = sorted({int(bins_grid[0]), int(bins_grid[-1])})
            for i, geom in enumerate(geometries):
                for j, w in enumerate(smooth_grid):
                    ax = axes_arr[i, j]
                    for idx, b in enumerate(bins_show):
                        sub = metrics.loc[
                            (metrics["geometry"] == geom)
                            & (metrics["D"] == int(d_value))
                            & (metrics["mode"] == str(mode_selected))
                            & (metrics["bins_B"] == int(b))
                            & (metrics["smooth_w"] == int(w))
                            & (metrics["pi_target"] == float(pi_target))
                            & (~metrics["underpowered"])
                        ]
                        pvals = pd.to_numeric(sub["p_T"], errors="coerce").to_numpy(
                            dtype=float
                        )
                        pvals = pvals[np.isfinite(pvals)]
                        if pvals.size == 0:
                            continue
                        label = rf"$B={int(b)}$"
                        edge_color = colors[idx % len(colors)]
                        hist_out = ax.hist(
                            pvals,
                            bins=hist_bins,
                            density=True,
                            histtype="step",
                            linewidth=1.8,
                            color=edge_color,
                            label=label,
                        )
                        patch = hist_out[-1][0] if hist_out[-1] else None
                        if patch is not None and label not in legend_handles:
                            legend_handles[label] = patch
                    ax.axhline(
                        1.0, linestyle="--", color="0.3", linewidth=1.2, alpha=0.8
                    )
                    ax.set_xlim(0.0, 1.0)
                    ax.set_xlabel(r"$p_T$")
                    if j == 0:
                        ax.set_ylabel("Density")
                        ax.text(
                            -0.32,
                            0.5,
                            _humanize_geometry_label(str(geom)),
                            transform=ax.transAxes,
                            rotation=90,
                            va="center",
                            ha="center",
                            fontsize=8,
                        )
                    ax.set_title(rf"$w={int(w)}$")
            legend_title = ""

        if legend_handles:
            labels = sorted(legend_handles.keys())
            handles = [legend_handles[k] for k in labels]
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(0.84, 0.5),
                frameon=False,
                fontsize=7,
                ncol=1,
                title=legend_title,
            )

        fig.suptitle(
            rf"Experiment B test mode: $p_T$ histograms (strict null, $\pi_{{\mathrm{{target}}}}={_fmt_pi(pi_target)}$, $D={int(d_value)}$)",
            y=0.995,
        )
        fig.text(
            0.5,
            0.04,
            "Dashed line: expected U(0,1) density (=1).",
            ha="center",
            va="center",
            fontsize=7,
            color="0.3",
        )
        fig.text(
            0.5,
            0.02,
            "Display mode: reduced overplot (2 curves per panel).",
            ha="center",
            va="center",
            fontsize=7,
            color="0.3",
        )
        fig.tight_layout(rect=[0.0, 0.07, 0.82, 0.97])
    else:
        rows = len(geometries)
        cols = len(modes)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(4.1 * cols + 1.4, 2.9 * rows),
            constrained_layout=False,
        )
        axes_arr = np.atleast_2d(axes)
        for i, geom in enumerate(geometries):
            for j, mode in enumerate(modes):
                ax = axes_arr[i, j]
                color_i = 0
                for b in bins_grid:
                    for w in smooth_grid:
                        sub = metrics.loc[
                            (metrics["geometry"] == geom)
                            & (metrics["D"] == int(d_value))
                            & (metrics["mode"] == mode)
                            & (metrics["bins_B"] == int(b))
                            & (metrics["smooth_w"] == int(w))
                            & (metrics["pi_target"] == float(pi_target))
                            & (~metrics["underpowered"])
                        ]
                        pvals = pd.to_numeric(sub["p_T"], errors="coerce").to_numpy(
                            dtype=float
                        )
                        pvals = pvals[np.isfinite(pvals)]
                        if pvals.size == 0:
                            continue
                        label = f"B={int(b)}, w={int(w)}"
                        edge_color = colors[color_i % len(colors)]
                        hist_out = ax.hist(
                            pvals,
                            bins=hist_bins,
                            density=True,
                            histtype="step",
                            linewidth=1.8,
                            color=edge_color,
                            label=label,
                        )
                        patch = hist_out[-1][0] if hist_out[-1] else None
                        if patch is not None and label not in legend_handles:
                            legend_handles[label] = patch
                        color_i += 1
                ax.axhline(1.0, linestyle="--", color="0.3", linewidth=1.2, alpha=0.8)
                ax.set_xlim(0.0, 1.0)
                ax.set_xlabel(r"$p_T$")
                if j == 0:
                    ax.set_ylabel("Density")
                    ax.text(
                        -0.34,
                        0.5,
                        _humanize_geometry_label(str(geom)),
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=8,
                    )
                ax.set_title("Raw" if str(mode).lower() == "raw" else "Smoothed")

        if legend_handles:
            labels = sorted(
                legend_handles.keys(),
                key=lambda x: (
                    int(x.split(",")[0].split("=")[1]),
                    int(x.split("=")[-1]),
                ),
            )
            handles = [legend_handles[k] for k in labels]
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(0.84, 0.5),
                frameon=False,
                fontsize=7,
                ncol=1,
            )

        fig.suptitle(
            rf"Experiment B test mode: $p_T$ histograms (strict null, $\pi_{{\mathrm{{target}}}}={_fmt_pi(pi_target)}$, $D={int(d_value)}$)",
            y=0.995,
        )
        fig.text(
            0.5,
            0.04,
            "Dashed line: expected U(0,1) density (=1).",
            ha="center",
            va="center",
            fontsize=7,
            color="0.3",
        )
        fig.tight_layout(rect=[0.0, 0.07, 0.82, 0.97])

    savefig(fig, out_png)
    plt.close(fig)


def aggregate_summary(metrics: pd.DataFrame, n_perm: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "geometry",
        "D",
        "sigma_eta",
        "bins_B",
        "smooth_w",
        "mode",
        "pi_target",
    ]
    for keys, grp in metrics.groupby(group_cols, sort=True):
        geometry, d, sigma_eta, b, w, mode, pi_target = keys
        n_genes = int(grp.shape[0])
        valid = grp.loc[~grp["underpowered"]]
        n_valid = int(valid.shape[0])
        frac_underpowered = float(grp["underpowered"].mean())
        if n_valid > 0:
            pvals = pd.to_numeric(valid["p_T"], errors="coerce").to_numpy(dtype=float)
            pvals = pvals[np.isfinite(pvals)]
        else:
            pvals = np.zeros(0, dtype=float)

        if pvals.size > 0:
            mean_p = float(np.mean(pvals))
            var_p = float(np.var(pvals))
            k05 = int(np.sum(pvals <= 0.05))
            typei05 = float(k05 / pvals.size)
            ci_low, ci_high = _wilson_ci(k05, int(pvals.size))
            if pvals.size >= 5:
                ks = kstest(pvals, "uniform")
                ks_p = float(ks.pvalue)
                ks_stat = float(ks.statistic)
            else:
                ks_p = float("nan")
                ks_stat = float("nan")
        else:
            mean_p = float("nan")
            var_p = float("nan")
            ks_p = float("nan")
            ks_stat = float("nan")
            typei05 = float("nan")
            ci_low, ci_high = float("nan"), float("nan")

        rows.append(
            {
                "geometry": str(geometry),
                "D": int(d),
                "sigma_eta": float(sigma_eta),
                "bins_B": int(b),
                "smooth_w": int(w),
                "mode": str(mode),
                "pi_target": float(pi_target),
                "n_genes_total": n_genes,
                "n_genes": n_genes,
                "n_non_underpowered": n_valid,
                "frac_underpowered": frac_underpowered,
                "mean_p": mean_p,
                "var_p": var_p,
                "ks_pvalue": ks_p,
                "ks_stat": ks_stat,
                "typeI_alpha05": typei05,
                "typeI_alpha05_ci_low": ci_low,
                "typeI_alpha05_ci_high": ci_high,
                "p_min_discrete": float(1.0 / (int(n_perm) + 1)),
            }
        )
    return pd.DataFrame(rows)


def derive_ks_uniformity(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "geometry",
        "D",
        "sigma_eta",
        "mode",
        "pi_target",
        "bins_B",
        "smooth_w",
        "n_non_underpowered",
        "ks_stat",
        "ks_pvalue",
        "mean_p",
        "var_p",
    ]
    have = [c for c in cols if c in summary.columns]
    out = summary[have].copy()
    if "ks_stat" in out.columns:
        out = out.rename(columns={"ks_stat": "ks_statistic"})
    return out


def derive_typei_by_cell(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "geometry",
        "D",
        "sigma_eta",
        "mode",
        "pi_target",
        "bins_B",
        "smooth_w",
        "n_non_underpowered",
        "typeI_alpha05",
        "typeI_alpha05_ci_low",
        "typeI_alpha05_ci_high",
        "mean_p",
        "ks_pvalue",
        "p_min_discrete",
    ]
    have = [c for c in cols if c in summary.columns]
    out = summary[have].copy()
    if "typeI_alpha05" in out.columns:
        out = out.rename(columns={"typeI_alpha05": "typeI_p05"})
    return out


def build_target_cell_diagnostics_expb(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    def _pick(
        *,
        geometry: str,
        d_value: int,
        mode: str,
        pi_target: float,
        bins_b: int,
        smooth_w: int,
        label: str,
    ) -> None:
        mask = (
            (summary["geometry"].astype(str) == str(geometry))
            & (summary["D"].astype(int) == int(d_value))
            & (summary["mode"].astype(str) == str(mode))
            & np.isclose(
                summary["pi_target"].astype(float),
                float(pi_target),
                atol=1e-12,
                rtol=0.0,
            )
            & (summary["bins_B"].astype(int) == int(bins_b))
            & (summary["smooth_w"].astype(int) == int(smooth_w))
        )
        picked = summary.loc[mask].copy()
        if picked.empty:
            return
        picked["target_label"] = str(label)
        rows.append(picked)

    _pick(
        geometry="density_gradient_disk",
        d_value=10,
        mode="smoothed",
        pi_target=0.9,
        bins_b=72,
        smooth_w=5,
        label="stress_primary",
    )
    _pick(
        geometry="density_gradient_disk",
        d_value=10,
        mode="smoothed",
        pi_target=0.9,
        bins_b=48,
        smooth_w=5,
        label="stress_companion",
    )
    for mode in ["raw", "smoothed"]:
        for w in [1, 3, 5]:
            _pick(
                geometry="ring_annulus",
                d_value=5,
                mode=mode,
                pi_target=0.2,
                bins_b=48,
                smooth_w=int(w),
                label="ring_borderline",
            )

    chunks: list[pd.DataFrame] = []
    if rows:
        chunks.extend(rows)
    worst_ks = summary.sort_values("ks_pvalue", ascending=True).head(10).copy()
    if not worst_ks.empty:
        worst_ks["target_label"] = "worst_ks"
        chunks.append(worst_ks)
    worst_typei = summary.copy()
    worst_typei["typeI_dev"] = (
        pd.to_numeric(worst_typei["typeI_alpha05"], errors="coerce") - 0.05
    ).abs()
    worst_typei = worst_typei.sort_values("typeI_dev", ascending=False).head(10).copy()
    if not worst_typei.empty:
        worst_typei["target_label"] = "worst_typei"
        chunks.append(worst_typei)

    if chunks:
        target = pd.concat(chunks, axis=0, ignore_index=False)
    else:
        target = pd.DataFrame(columns=list(summary.columns) + ["target_label"])
    target = target.reset_index(names="row_index")
    target["csv_line"] = target["row_index"].astype(int) + 2
    target = target.sort_values(
        ["target_label", "geometry", "mode", "bins_B", "smooth_w"]
    )
    return target


def _warn_ks_outliers(summary: pd.DataFrame) -> None:
    bad = summary.loc[
        (pd.to_numeric(summary["n_non_underpowered"], errors="coerce") >= 200)
        & (pd.to_numeric(summary["ks_pvalue"], errors="coerce") < 1e-3)
    ]
    for _, row in bad.iterrows():
        print(
            (
                "WARNING: KS non-uniformity flagged "
                f"[geometry={row['geometry']}, D={int(row['D'])}, mode={row['mode']}, "
                f"pi={_fmt_pi(float(row['pi_target']))}, B={int(row['bins_B'])}, w={int(row['smooth_w'])}, "
                f"n_non_underpowered={int(row['n_non_underpowered'])}, ks_p={float(row['ks_pvalue']):.3g}]"
            ),
            flush=True,
        )


def _run_sanity_checks(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    debug_cache: dict[
        tuple[str, int, str, int, int, float], list[dict[str, float | int]]
    ],
    *,
    n_perm: int,
    sanity_d: int,
    sanity_pi: float,
) -> None:
    checks = [
        ("disk_gaussian", int(sanity_d), "raw", 36, 1, float(sanity_pi)),
        ("disk_gaussian", int(sanity_d), "smoothed", 36, 3, float(sanity_pi)),
    ]
    for geom, d, mode, b, w, pi in checks:
        sub = summary.loc[
            (summary["geometry"] == geom)
            & (summary["D"] == int(d))
            & (summary["mode"] == mode)
            & (summary["bins_B"] == int(b))
            & (summary["smooth_w"] == int(w))
            & (summary["pi_target"] == float(pi))
        ]
        if sub.empty:
            print(
                f"SANITY WARNING: missing summary row for {(geom, d, mode, b, w, pi)}",
                flush=True,
            )
            continue
        row = sub.iloc[0]
        n_valid = int(row["n_non_underpowered"])
        mean_p = float(row["mean_p"]) if pd.notna(row["mean_p"]) else float("nan")
        ks_p = float(row["ks_pvalue"]) if pd.notna(row["ks_pvalue"]) else float("nan")
        ok_mean = bool(np.isfinite(mean_p) and (0.45 <= mean_p <= 0.55))
        ok_ks = bool((n_valid < 200) or (np.isfinite(ks_p) and ks_p > 1e-3))
        if ok_mean and ok_ks:
            print(
                (
                    f"SANITY OK [{geom}, D={d}, mode={mode}, B={b}, w={w}, pi={_fmt_pi(pi)}]: "
                    f"mean_p={mean_p:.4f}, ks_p={ks_p:.4g}, n_non_underpowered={n_valid}"
                ),
                flush=True,
            )
            continue

        print(
            (
                f"SANITY WARNING [{geom}, D={d}, mode={mode}, B={b}, w={w}, pi={_fmt_pi(pi)}]: "
                f"mean_p={mean_p:.4f}, ks_p={ks_p:.4g}, n_non_underpowered={n_valid}, "
                f"p_min={1.0/(n_perm+1):.4f}"
            ),
            flush=True,
        )
        sub_metrics = metrics.loc[
            (metrics["geometry"] == geom)
            & (metrics["D"] == int(d))
            & (metrics["mode"] == mode)
            & (metrics["bins_B"] == int(b))
            & (metrics["smooth_w"] == int(w))
            & (metrics["pi_target"] == float(pi))
            & (~metrics["underpowered"])
        ]
        pvals = pd.to_numeric(sub_metrics["p_T"], errors="coerce").to_numpy(dtype=float)
        pvals = pvals[np.isfinite(pvals)]
        if pvals.size > 0:
            q = np.quantile(pvals, [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
            print(
                (
                    "  p-value diagnostics: "
                    f"mean={float(np.mean(pvals)):.4f}, var={float(np.var(pvals)):.5f}, "
                    f"q01={q[0]:.3f}, q05={q[1]:.3f}, q10={q[2]:.3f}, q50={q[3]:.3f}, "
                    f"q90={q[4]:.3f}, q95={q[5]:.3f}, q99={q[6]:.3f}"
                ),
                flush=True,
            )
        else:
            print(
                "  p-value diagnostics: no non-underpowered genes in this cell.",
                flush=True,
            )

        debug_key = (geom, int(d), mode, int(b), int(w), float(pi))
        rows = debug_cache.get(debug_key, [])
        if rows:
            rng = np.random.default_rng(_seed_from_fields(1234567, *debug_key))
            take = min(3, len(rows))
            idx = rng.choice(len(rows), size=take, replace=False)
            print("  T_obs vs null_T summary for random genes:", flush=True)
            for i in idx.tolist():
                r = rows[int(i)]
                print(
                    (
                        f"    gene_index={int(r['gene_index'])}, T_obs={float(r['T_obs']):.6f}, "
                        f"null_mean={float(r['null_mean']):.6f}, null_median={float(r['null_median']):.6f}, "
                        f"null_q95={float(r['null_q95']):.6f}"
                    ),
                    flush=True,
                )


def _run_corner_debug_artifacts(
    *,
    candidates: list[dict[str, Any]],
    dataset_contexts: dict[tuple[int, str, int], DatasetContext],
    plots_dir: Path,
    results_dir: Path,
    n_perm: int,
    debug_k: int,
    master_seed: int,
    edge_cfg: EdgeRuleConfigB,
    save_full_profiles: bool = False,
) -> None:
    """Emit targeted diagnostics for the historically failing corner condition."""
    if not candidates:
        print(
            "Corner debug: no genes found for target condition; writing fallback embedding/profile plots.",
            flush=True,
        )
        fallback_ctx = next(iter(dataset_contexts.values()), None)
        if fallback_ctx is not None:
            rng_fb = np.random.default_rng(
                _seed_from_fields(master_seed, "corner_debug_fallback")
            )
            eta_cell_fb = donor_effect_vector(
                fallback_ctx.donor_ids, fallback_ctx.eta_d
            )
            f_fb = simulate_null_gene(
                pi_target=0.2, donor_eta_per_cell=eta_cell_fb, rng=rng_fb
            )

            fig_emb, ax_emb = plt.subplots(figsize=(5.2, 4.4), constrained_layout=False)
            plot_embedding_with_foreground(
                fallback_ctx.X_sim,
                np.asarray(f_fb, dtype=bool),
                ax=ax_emb,
                title="Representative embedding foreground (ExpB fallback)",
                s=6.0,
                alpha_bg=0.30,
                alpha_fg=0.75,
            )
            savefig(fig_emb, plots_dir / "embedding_example.png")
            plt.close(fig_emb)

            bins_b = sorted(fallback_ctx.bin_cache.keys())[0]
            bin_id, bin_counts_total = fallback_ctx.bin_cache[int(bins_b)]
            e_obs, _, _ = compute_rsp_profile_from_boolean(
                np.asarray(f_fb, dtype=bool),
                fallback_ctx.angles,
                int(bins_b),
                bin_id=bin_id,
                bin_counts_total=bin_counts_total,
            )
            theta_idx = np.linspace(0.0, 2.0 * np.pi, int(bins_b), endpoint=False)
            fig_pol, ax_pol = plt.subplots(
                figsize=(5.2, 4.8),
                subplot_kw={"projection": "polar"},
                constrained_layout=False,
            )
            plot_rsp_polar(
                theta_idx,
                np.asarray(e_obs, dtype=float),
                ax=ax_pol,
                title="Representative profile (ExpB fallback)",
            )
            savefig(fig_pol, plots_dir / "example_profile_fallback.png")
            plt.close(fig_pol)
        else:
            fig_emb, ax_emb = plt.subplots(figsize=(5.2, 4.4), constrained_layout=False)
            ax_emb.axis("off")
            ax_emb.text(0.5, 0.5, "ExpB fallback embedding", ha="center", va="center")
            savefig(fig_emb, plots_dir / "embedding_example.png")
            plt.close(fig_emb)

            theta_idx = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
            profile = np.zeros_like(theta_idx, dtype=float)
            fig_pol, ax_pol = plt.subplots(
                figsize=(5.2, 4.8),
                subplot_kw={"projection": "polar"},
                constrained_layout=False,
            )
            plot_rsp_polar(
                theta_idx,
                profile,
                ax=ax_pol,
                title="Representative profile (ExpB fallback)",
            )
            savefig(fig_pol, plots_dir / "example_profile_fallback.png")
            plt.close(fig_pol)
        return

    pvals_all = np.asarray([float(x["p_T"]) for x in candidates], dtype=float)
    mean_p = float(np.mean(pvals_all))
    var_p = float(np.var(pvals_all))
    if pvals_all.size >= 5:
        ks = kstest(pvals_all, "uniform")
        ks_p = float(ks.pvalue)
        ks_stat = float(ks.statistic)
    else:
        ks_p = float("nan")
        ks_stat = float("nan")
    print(
        (
            "Corner debug summary [density_gradient_disk, D=10, pi=0.9, mode=smoothed, B=72, w=5]: "
            f"n={pvals_all.size}, mean_p={mean_p:.4f}, var_p={var_p:.5f}, "
            f"ks_p={ks_p:.4g}, ks_stat={ks_stat:.4f}"
        ),
        flush=True,
    )

    rng = np.random.default_rng(
        _seed_from_fields(
            master_seed, "corner_debug_selection", pvals_all.size, debug_k
        )
    )
    take = min(int(debug_k), len(candidates))
    chosen_idx = rng.choice(len(candidates), size=take, replace=False)
    chosen = [candidates[int(i)] for i in chosen_idx.tolist()]

    debug_rows: list[dict[str, Any]] = []
    eta_cell_cache: dict[tuple[int, str, int], np.ndarray] = {}
    for ix, rec in enumerate(chosen):
        ctx_key = (int(rec["seed_run"]), str(rec["geometry"]), int(rec["D"]))
        context = dataset_contexts.get(ctx_key)
        if context is None:
            continue
        bins_b = int(rec.get("bins_B", 72))
        mode_name = str(rec.get("mode", "smoothed"))
        smooth_w = int(rec.get("smooth_w", 5))
        if ctx_key not in eta_cell_cache:
            eta_cell_cache[ctx_key] = donor_effect_vector(
                context.donor_ids, context.eta_d
            )
        eta_cell = eta_cell_cache[ctx_key]
        rng_gene = rng_from_seed(int(rec["seed_gene"]))
        f_raw = simulate_null_gene(
            pi_target=float(rec["pi_target"]),
            donor_eta_per_cell=eta_cell,
            rng=rng_gene,
        )
        f_test, _ = apply_foreground_edge_rule_b(
            np.asarray(f_raw, dtype=bool),
            pi_target=float(rec["pi_target"]),
            cfg=edge_cfg,
        )
        if bins_b not in context.bin_cache:
            continue
        bin_id, bin_counts_total = context.bin_cache[bins_b]
        perm = perm_null_T(
            f=np.asarray(f_test, dtype=bool),
            angles=context.angles,
            donor_ids=context.donor_ids,
            n_bins=int(bins_b),
            n_perm=int(n_perm),
            seed=int(rec["seed_perm"]),
            mode=mode_name,
            smooth_w=int(smooth_w),
            donor_stratified=True,
            return_null_T=True,
            return_obs_profile=True,
            return_null_profiles=True,
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
        null_e = np.asarray(perm["null_E_phi"], dtype=float)
        null_t = np.asarray(perm["null_T"], dtype=float)
        t_obs = float(perm["T_obs"])
        p_t = float(perm["p_T"])

        debug_rows.append(
            {
                "gene_id": str(rec["gene_id"]),
                "seed_run": int(rec["seed_run"]),
                "gene_index": int(rec["gene_index"]),
                "prev_obs": float(rec["prev_obs"]),
                "D_eff": int(rec["D_eff"]),
                "bins_B": int(bins_b),
                "mode": str(mode_name),
                "smooth_w": int(smooth_w),
                "underpowered": bool(rec.get("underpowered", False)),
                "T_obs": t_obs,
                "p_T": p_t,
                "null_mean": float(np.mean(null_t)),
                "null_sd": float(np.std(null_t)),
                "null_q95": float(np.quantile(null_t, 0.95)),
                "null_q99": float(np.quantile(null_t, 0.99)),
            }
        )

        if ix == 0:
            fig_emb, ax_emb = plt.subplots(figsize=(5.2, 4.4), constrained_layout=False)
            plot_embedding_with_foreground(
                context.X_sim,
                np.asarray(f_test, dtype=bool),
                ax=ax_emb,
                title="Representative embedding foreground (ExpB corner)",
                s=6.0,
                alpha_bg=0.30,
                alpha_fg=0.75,
            )
            savefig(fig_emb, plots_dir / "embedding_example.png")
            plt.close(fig_emb)

        theta_idx = np.linspace(0.0, 2.0 * np.pi, e_obs.size, endpoint=False)
        if bool(save_full_profiles):
            np.savez_compressed(
                results_dir / f"profile_{rec['gene_id']}.npz",
                theta=theta_idx,
                E_obs=e_obs,
                null_E=null_e,
                null_T=null_t,
                T_obs=np.asarray([t_obs], dtype=float),
                p_T=np.asarray([p_t], dtype=float),
            )
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), constrained_layout=True)
        ax_profile, ax_hist = axes
        gs = ax_profile.get_gridspec()
        ax_profile.remove()
        ax_profile = fig.add_subplot(gs[0, 0], projection="polar")

        sample_perm = min(8, null_e.shape[0])
        pick = np.linspace(
            0, max(0, null_e.shape[0] - 1), num=max(1, sample_perm), dtype=int
        )
        pick = np.unique(pick)
        for k, j in enumerate(pick.tolist()):
            plot_rsp_polar(
                theta_idx,
                null_e[int(j), :],
                ax=ax_profile,
                color="#b5b5b5",
                linewidth=0.8,
                label="Null draws" if k == 0 else None,
                title=None,
            )
        plot_rsp_polar(
            theta_idx,
            e_obs,
            ax=ax_profile,
            color="#D62728",
            linewidth=1.6,
            label="Observed",
            title=(
                f"{rec['gene_id']} • {_humanize_geometry_label(str(rec['geometry']))}"
            ),
        )
        profile_max = (
            np.nanmax(np.concatenate([e_obs.ravel(), null_e[pick, :].ravel()]))
            if pick.size > 0
            else np.nanmax(e_obs)
        )
        if np.isfinite(profile_max) and profile_max > 0:
            rticks = np.linspace(0.0, float(profile_max), 4)[1:]
            ax_profile.set_rticks(rticks)
        ax_profile.set_rlabel_position(135)
        ax_profile.grid(alpha=0.25)
        ax_profile.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

        ax_hist.hist(
            null_t,
            bins=28,
            color="#4C78A8",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.35,
        )
        ax_hist.axvline(
            t_obs,
            color="#D62728",
            linestyle="--",
            linewidth=1.6,
            label=f"T_obs={t_obs:.4f}",
        )
        ax_hist.set_xlabel("T null")
        ax_hist.set_ylabel("count")
        ax_hist.set_title(rf"$p_T={p_t:.4f}$")
        ax_hist.legend(loc="upper left", frameon=False)

        fig.suptitle(
            "ExpB corner debug (strict null)",
            y=0.99,
        )
        fig.text(
            0.5,
            0.02,
            (
                rf"$D={int(rec['D'])}$, $\pi_{{\mathrm{{target}}}}={_fmt_pi(float(rec['pi_target']))}$, "
                rf"mode={_humanize_mode_label(mode_name)}, $B={int(bins_b)}$, $w={int(smooth_w)}$"
            ),
            ha="center",
            va="center",
            fontsize=7,
            color="0.35",
        )
        out_png = plots_dir / (
            f"debug_profiles_{str(rec['geometry'])}_pi{_fmt_pi(float(rec['pi_target']))}_"
            f"B{int(bins_b)}_w{int(smooth_w)}_gene{rec['gene_id']}.png"
        )
        savefig(fig, out_png)
        plt.close(fig)

        if ix < 3:
            sample_rows = (
                null_e[pick[: min(3, sample_perm)], :] if sample_perm > 0 else null_e
            )
            print(
                (
                    f"Corner debug scaling {rec['gene_id']}: "
                    f"E_obs[min,max]=[{float(np.min(e_obs)):.5f}, {float(np.max(e_obs)):.5f}], "
                    f"null_E_sample[min,max]=[{float(np.min(sample_rows)):.5f}, {float(np.max(sample_rows)):.5f}]"
                ),
                flush=True,
            )

    pd.DataFrame(debug_rows).to_csv(
        results_dir / "debug_gene_summaries.csv", index=False
    )


def _simulate_dataset(
    *,
    geometry: str,
    d_value: int,
    n_cells: int,
    sigma_eta: float,
    master_seed: int,
    bins_grid: list[int],
    seed_label: int,
    cache_dir: str | Path | None = None,
) -> DatasetContext:
    seed_dataset = _seed_from_fields(
        master_seed,
        "dataset",
        int(seed_label),
        geometry,
        int(d_value),
        float(sigma_eta),
    )
    run_id = (
        f"{geometry}__D{int(d_value)}__sigma{float(sigma_eta):g}__seed{int(seed_label)}"
    )
    bins_key = ",".join(str(int(b)) for b in bins_grid)
    cache_key = (
        f"expB_dataset_v1|seed_dataset={seed_dataset}|geometry={geometry}|D={int(d_value)}|"
        f"N={int(n_cells)}|sigma_eta={float(sigma_eta):.6g}|bins={bins_key}"
    )

    if cache_dir is not None:
        cached = io_cache_get(cache_dir, cache_key)
        if isinstance(cached, dict) and {
            "X_sim",
            "angles",
            "donor_ids",
            "eta_d",
        }.issubset(set(cached.keys())):
            x_sim = np.asarray(cached["X_sim"], dtype=float)
            angles = np.asarray(cached["angles"], dtype=float)
            donor_ids = np.asarray(cached["donor_ids"])
            eta_d = np.asarray(cached["eta_d"], dtype=float)
            bin_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for b in bins_grid:
                bid_key = f"bin_id_{int(b)}"
                cnt_key = f"bin_counts_{int(b)}"
                if bid_key in cached and cnt_key in cached:
                    bin_cache[int(b)] = (
                        np.asarray(cached[bid_key], dtype=np.int32),
                        np.asarray(cached[cnt_key], dtype=np.int64),
                    )
            if len(bin_cache) != len(bins_grid):
                bin_cache = _build_bin_cache(angles, bins_grid)
            return DatasetContext(
                geometry=geometry,
                D=int(d_value),
                X_sim=x_sim,
                angles=angles,
                donor_ids=donor_ids,
                eta_d=eta_d,
                run_id=run_id,
                seed_dataset=seed_dataset,
                bin_cache=bin_cache,
                cache_hit=True,
            )

    rng = rng_from_seed(seed_dataset)
    x_sim = simulate_geometry(geometry, n_cells, rng)
    theta_raw = np.arctan2(x_sim[:, 1], x_sim[:, 0]).astype(float)  # [-pi, pi)
    # BioRSP validators expect [0, 2pi); retain fixed-origin arctan2 definition, wrapped.
    angles = np.mod(theta_raw, 2.0 * np.pi).astype(float)
    donor_ids = assign_donors(n_cells, int(d_value), rng)
    eta_d = sample_donor_effects(int(d_value), float(sigma_eta), rng)
    bin_cache = _build_bin_cache(angles, bins_grid)
    if cache_dir is not None:
        payload: dict[str, np.ndarray] = {
            "X_sim": np.asarray(x_sim, dtype=float),
            "angles": np.asarray(angles, dtype=float),
            "donor_ids": np.asarray(donor_ids),
            "eta_d": np.asarray(eta_d, dtype=float),
        }
        for b in bins_grid:
            bid, cnt = bin_cache[int(b)]
            payload[f"bin_id_{int(b)}"] = np.asarray(bid, dtype=np.int32)
            payload[f"bin_counts_{int(b)}"] = np.asarray(cnt, dtype=np.int64)
        io_cache_set(cache_dir, cache_key, payload)
    return DatasetContext(
        geometry=geometry,
        D=int(d_value),
        X_sim=x_sim,
        angles=angles,
        donor_ids=donor_ids,
        eta_d=eta_d,
        run_id=run_id,
        seed_dataset=seed_dataset,
        bin_cache=bin_cache,
        cache_hit=False,
    )


def build_grid(
    geometries: list[str],
    d_grid: list[int],
    seed_values: list[int],
) -> list[tuple[int, str, int]]:
    """Build dataset-condition grid for ExpB."""
    return [
        (int(seed), str(geometry), int(d))
        for seed in seed_values
        for geometry in geometries
        for d in d_grid
    ]


def _run_dataset_block_expb(item: dict[str, Any]) -> dict[str, Any]:
    """Process one dataset block (all genes) for ExpB."""
    t0 = time.time()
    seed_run = int(item["seed_run"])
    geometry = str(item["geometry"])
    d_value = int(item["d_value"])
    n_cells = int(item["n_cells"])
    g_total = int(item["g_total"])
    n_perm = int(item["n_perm"])
    sigma_eta = float(item["sigma_eta"])
    bins_grid = [int(x) for x in item["bins_grid"]]
    smooth_grid = [int(x) for x in item["smooth_grid"]]
    modes = [str(x) for x in item["modes"]]
    pi_bins = [float(x) for x in item["pi_bins"]]
    progress_every = int(item["progress_every"])
    progress_enabled = bool(item.get("progress", True))
    min_perm_eff = int(item["min_perm_eff"])
    edge_cfg = EdgeRuleConfigB(
        threshold=float(item["edge_threshold"]),
        strategy=str(item["edge_strategy"]),
    )
    p_min = float(item["p_min"])
    min_fg_total = int(item["min_fg_total"])
    min_fg_per_donor = int(item["min_fg_per_donor"])
    min_bg_per_donor = int(item["min_bg_per_donor"])
    d_eff_min = int(item["d_eff_min"])
    debug_targets = {
        (
            str(r["geometry"]),
            int(r["D"]),
            str(r["mode"]),
            int(r["bins_B"]),
            int(r["smooth_w"]),
            float(r["pi_target"]),
        )
        for r in item["debug_targets"]
    }
    corner_target = dict(item["corner_target"])

    context = _simulate_dataset(
        geometry=geometry,
        d_value=int(d_value),
        n_cells=int(n_cells),
        sigma_eta=float(sigma_eta),
        master_seed=int(seed_run),
        bins_grid=bins_grid,
        seed_label=int(seed_run),
        cache_dir=item.get("cache_dir"),
    )
    t_setup = time.time() - t0
    rng_schedule = rng_from_seed(
        _seed_from_fields(
            seed_run, "pi_schedule", geometry, int(d_value), float(sigma_eta)
        )
    )
    pi_targets = _gene_prevalence_schedule(int(g_total), pi_bins, rng_schedule)
    eta_cell = donor_effect_vector(context.donor_ids, context.eta_d)
    perm_plan_seed = _seed_from_fields(
        seed_run, "perm_plan", geometry, int(d_value), int(n_perm)
    )
    perm_plan_key = (
        f"expB_perm_plan_v1|seed_dataset={int(context.seed_dataset)}|n_perm={int(n_perm)}|"
        f"N={int(n_cells)}|D={int(d_value)}"
    )
    perm_plan_cache_hit = 0
    perm_indices: np.ndarray | None = None
    cache_dir = item.get("cache_dir")
    if cache_dir:
        cached_plan = io_cache_get(cache_dir, perm_plan_key)
        if isinstance(cached_plan, np.ndarray) and cached_plan.shape == (
            int(n_perm),
            int(n_cells),
        ):
            perm_indices = np.asarray(cached_plan, dtype=np.int32)
            perm_plan_cache_hit = 1
    if perm_indices is None:
        perm_indices = _build_donor_perm_plan(
            context.donor_ids,
            n_perm=int(n_perm),
            seed=int(perm_plan_seed),
        )
        if cache_dir:
            io_cache_set(
                cache_dir, perm_plan_key, np.asarray(perm_indices, dtype=np.int32)
            )

    metrics_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    corner_candidates: list[dict[str, Any]] = []

    t_gene_start = time.time()
    n_perm_calls = 0
    for g in range(int(g_total)):
        pi_target = float(pi_targets[g])
        pi_label = _fmt_pi(pi_target)
        seed_gene = _seed_from_fields(
            seed_run,
            "gene",
            geometry,
            int(d_value),
            float(sigma_eta),
            pi_label,
            int(g),
        )
        rng_gene = rng_from_seed(seed_gene)
        f = simulate_null_gene(
            pi_target=pi_target,
            donor_eta_per_cell=eta_cell,
            rng=rng_gene,
        )
        f_raw = np.asarray(f, dtype=bool)
        prev_obs = float(np.mean(f_raw))
        n_fg_raw = int(np.sum(f_raw))
        f_test, edge_info = apply_foreground_edge_rule_b(
            f_raw,
            pi_target=float(pi_target),
            cfg=edge_cfg,
        )
        n_fg_test = int(np.sum(f_test))

        power = evaluate_underpowered(
            donor_ids=context.donor_ids,
            f=f_test,
            n_perm=int(n_perm),
            p_min=p_min,
            min_fg_total=min_fg_total,
            min_fg_per_donor=min_fg_per_donor,
            min_bg_per_donor=min_bg_per_donor,
            d_eff_min=d_eff_min,
            min_perm=int(min_perm_eff),
        )
        under = bool(power["underpowered"])
        d_eff = int(power["D_eff"])

        if under:
            # Underpowered genes are excluded from calibration summaries; skip expensive permutations.
            for b in bins_grid:
                seed_perm = _seed_from_fields(
                    seed_run,
                    "perm",
                    geometry,
                    int(d_value),
                    float(sigma_eta),
                    int(b),
                    pi_label,
                    int(g),
                )
                for mode in modes:
                    for w in smooth_grid:
                        seed_row = _seed_from_fields(
                            seed_run,
                            "row",
                            geometry,
                            int(d_value),
                            int(b),
                            int(w),
                            mode,
                            pi_label,
                            int(g),
                        )
                        metrics_rows.append(
                            {
                                "run_id": context.run_id,
                                "seed": int(seed_row),
                                "seed_run": int(seed_run),
                                "seed_gene": int(seed_gene),
                                "seed_perm": int(seed_perm),
                                "geometry": geometry,
                                "D": int(d_value),
                                "sigma_eta": float(sigma_eta),
                                "pi_target": float(pi_target),
                                "prev_obs": prev_obs,
                                "n_fg": n_fg_test,
                                "n_fg_raw": n_fg_raw,
                                "D_eff": d_eff,
                                "bins_B": int(b),
                                "smooth_w": int(w),
                                "mode": mode,
                                "T_obs": float("nan"),
                                "p_T": float("nan"),
                                "underpowered": True,
                                "fg_fraction_raw": float(edge_info["fg_fraction_raw"]),
                                "fg_fraction_test": float(
                                    edge_info["fg_fraction_test"]
                                ),
                                "bg_fraction": float(edge_info["bg_fraction"]),
                                "fg_bg_contrast": float(edge_info["fg_bg_contrast"]),
                                "prevalence_edge_triggered": bool(
                                    edge_info["prevalence_edge_triggered"]
                                ),
                                "edge_trigger_reasons": json.dumps(
                                    edge_info["edge_trigger_reasons"],
                                    separators=(",", ":"),
                                ),
                                "fg_rule_applied": str(edge_info["fg_rule_applied"]),
                            }
                        )
                        if (
                            geometry == str(corner_target["geometry"])
                            and int(d_value) == int(corner_target["D"])
                            and int(b) == int(corner_target["bins_B"])
                            and mode == str(corner_target["mode"])
                            and int(w) == int(corner_target["smooth_w"])
                            and abs(
                                float(pi_target) - float(corner_target["pi_target"])
                            )
                            < 1e-12
                        ):
                            corner_candidates.append(
                                {
                                    "seed_run": int(seed_run),
                                    "geometry": str(geometry),
                                    "D": int(d_value),
                                    "sigma_eta": float(sigma_eta),
                                    "gene_index": int(g),
                                    "gene_id": f"seed{int(seed_run)}_g{int(g)}",
                                    "pi_target": float(pi_target),
                                    "prev_obs": prev_obs,
                                    "prev_test": float(np.mean(f_test)),
                                    "D_eff": d_eff,
                                    "seed_perm": int(seed_perm),
                                    "seed_gene": int(seed_gene),
                                    "bins_B": int(b),
                                    "mode": str(mode),
                                    "smooth_w": int(w),
                                    "edge_info": dict(edge_info),
                                    "T_obs": float("nan"),
                                    "p_T": float("nan"),
                                    "underpowered": True,
                                }
                            )
            continue

        for b in bins_grid:
            _assert_valid_n_bins(int(b))
            seed_perm = _seed_from_fields(
                seed_run,
                "perm",
                geometry,
                int(d_value),
                float(sigma_eta),
                int(b),
                pi_label,
                int(g),
            )
            bin_id, bin_counts_total = context.bin_cache[int(b)]
            perm_raw = perm_null_T_and_profile(
                f=f_test,
                angles=context.angles,
                donor_ids=context.donor_ids,
                n_bins=int(b),
                n_perm=int(n_perm),
                seed=int(seed_perm),
                donor_stratified=True,
                mode="raw",
                smooth_w=1,
                return_null_profiles=True,
                perm_indices=perm_indices,
                perm_start=0,
                perm_end=int(n_perm),
                bin_id=bin_id,
                bin_counts_total=bin_counts_total,
            )
            n_perm_calls += 1
            e_obs_raw = np.asarray(perm_raw["E_phi_obs"], dtype=float)
            if int(e_obs_raw.size) != int(b):
                raise AssertionError(
                    f"Logged bins_B ({int(b)}) does not match profile length ({int(e_obs_raw.size)})"
                )
            null_e_raw = np.asarray(perm_raw["null_E_phi"], dtype=float)

            mode_cache: dict[tuple[str, int], dict[str, Any]] = {}
            for mode in modes:
                for w in smooth_grid:
                    cache_key = (mode, 1 if mode == "raw" else int(w))
                    if cache_key not in mode_cache:
                        mode_cache[cache_key] = mode_max_stat_from_profiles(
                            E_obs_raw=e_obs_raw,
                            null_E_raw=null_e_raw,
                            mode=mode,
                            smooth_w=int(w),
                        )
                    mode_stats = mode_cache[cache_key]
                    t_obs = float(mode_stats["T_obs"])
                    p_t = float(mode_stats["p_T"])
                    null_t_mode = np.asarray(mode_stats["null_T"], dtype=float)

                    seed_row = _seed_from_fields(
                        seed_run,
                        "row",
                        geometry,
                        int(d_value),
                        int(b),
                        int(w),
                        mode,
                        pi_label,
                        int(g),
                    )
                    metrics_rows.append(
                        {
                            "run_id": context.run_id,
                            "seed": int(seed_row),
                            "seed_run": int(seed_run),
                            "seed_gene": int(seed_gene),
                            "seed_perm": int(seed_perm),
                            "geometry": geometry,
                            "D": int(d_value),
                            "sigma_eta": float(sigma_eta),
                            "pi_target": float(pi_target),
                            "prev_obs": prev_obs,
                            "n_fg": n_fg_test,
                            "n_fg_raw": n_fg_raw,
                            "D_eff": d_eff,
                            "bins_B": int(b),
                            "smooth_w": int(w),
                            "mode": mode,
                            "T_obs": t_obs,
                            "p_T": p_t,
                            "underpowered": under,
                            "fg_fraction_raw": float(edge_info["fg_fraction_raw"]),
                            "fg_fraction_test": float(edge_info["fg_fraction_test"]),
                            "bg_fraction": float(edge_info["bg_fraction"]),
                            "fg_bg_contrast": float(edge_info["fg_bg_contrast"]),
                            "prevalence_edge_triggered": bool(
                                edge_info["prevalence_edge_triggered"]
                            ),
                            "edge_trigger_reasons": json.dumps(
                                edge_info["edge_trigger_reasons"], separators=(",", ":")
                            ),
                            "fg_rule_applied": str(edge_info["fg_rule_applied"]),
                        }
                    )
                    dbg_key = (
                        geometry,
                        int(d_value),
                        mode,
                        int(b),
                        int(w),
                        float(pi_target),
                    )
                    if dbg_key in debug_targets and (not under):
                        debug_rows.append(
                            {
                                "geometry": str(geometry),
                                "D": int(d_value),
                                "mode": str(mode),
                                "bins_B": int(b),
                                "smooth_w": int(w),
                                "pi_target": float(pi_target),
                                "gene_index": int(g),
                                "T_obs": t_obs,
                                "null_mean": float(np.mean(null_t_mode)),
                                "null_median": float(np.median(null_t_mode)),
                                "null_q95": float(np.quantile(null_t_mode, 0.95)),
                            }
                        )

                    if (
                        geometry == str(corner_target["geometry"])
                        and int(d_value) == int(corner_target["D"])
                        and int(b) == int(corner_target["bins_B"])
                        and mode == str(corner_target["mode"])
                        and int(w) == int(corner_target["smooth_w"])
                        and abs(float(pi_target) - float(corner_target["pi_target"]))
                        < 1e-12
                    ):
                        corner_candidates.append(
                            {
                                "seed_run": int(seed_run),
                                "geometry": str(geometry),
                                "D": int(d_value),
                                "sigma_eta": float(sigma_eta),
                                "gene_index": int(g),
                                "gene_id": f"seed{int(seed_run)}_g{int(g)}",
                                "pi_target": float(pi_target),
                                "prev_obs": prev_obs,
                                "prev_test": float(np.mean(f_test)),
                                "D_eff": d_eff,
                                "seed_perm": int(seed_perm),
                                "seed_gene": int(seed_gene),
                                "bins_B": int(b),
                                "mode": str(mode),
                                "smooth_w": int(w),
                                "edge_info": dict(edge_info),
                                "T_obs": t_obs,
                                "p_T": p_t,
                                "underpowered": bool(under),
                            }
                        )
        if progress_enabled and (
            (g + 1) % int(progress_every) == 0 or (g + 1) == int(g_total)
        ):
            elapsed = time.time() - t_gene_start
            rate = (g + 1) / elapsed if elapsed > 0 else float("nan")
            print(
                f"  {context.run_id}: gene {g + 1}/{g_total} ({rate:.2f} genes/s, elapsed {elapsed/60.0:.1f} min)",
                flush=True,
            )

    t_genes = time.time() - t_gene_start
    timing = {
        "setup_sec": float(t_setup),
        "genes_sec": float(t_genes),
        "total_sec": float(time.time() - t0),
    }
    counters = {
        "n_cells": int(n_cells),
        "n_genes": int(g_total),
        "n_perm": int(n_perm),
        "bins_grid": [int(x) for x in bins_grid],
        "n_bins_options": int(len(bins_grid)),
        "bin_cache_entries": int(len(context.bin_cache)),
        "perm_calls": int(n_perm_calls),
        "cache_hit": int(context.cache_hit),
        "perm_plan_cache_hit": int(perm_plan_cache_hit),
    }
    return {
        "seed": int(seed_run),
        "run_key": f"{seed_run}|{geometry}|{d_value}",
        "seed_run": int(seed_run),
        "geometry": str(geometry),
        "D": int(d_value),
        "sigma_eta": float(sigma_eta),
        "context_seed_dataset": int(context.seed_dataset),
        "context_run_id": str(context.run_id),
        "metrics_rows": metrics_rows,
        "debug_rows": debug_rows,
        "corner_candidates": corner_candidates,
        "timing": timing,
        "counters": counters,
    }


def summarize(metrics: pd.DataFrame, n_perm: int) -> pd.DataFrame:
    """Aggregate calibration summaries for ExpB."""
    return aggregate_summary(metrics, int(n_perm))


def write_report_markdown(outdir: Path, results_dir: Path) -> Path:
    """Write REPORT.md from current ExpB CSV artifacts."""
    return write_report_expB(
        outdir=outdir,
        metrics_path=results_dir / "metrics_long.csv",
        summary_path=results_dir / "summary.csv",
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
        n_cells = int(test_cfg.N)
        g_total = int(test_cfg.G)
        n_perm = int(test_cfg.n_perm)
        sigma_eta = 0.4
        geometries = ["disk_gaussian", "density_gradient_disk"]
        d_grid = [10]
        pi_bins = [0.2, 0.9]
        bins_grid = [36, 72]
        smooth_grid = [1, 5]
        smooth_grid_mode = "window"
        n_seeds = 1
        progress_every = 25
    else:
        n_cells = int(args.N)
        g_total = int(args.G)
        n_perm = int(args.n_perm)
        sigma_eta = float(args.sigma_eta)
        geometries = _parse_str_list(args.geometries)
        d_grid = _parse_int_list(args.D_grid)
        if bool(args.include_d2_stress) and 2 not in d_grid:
            d_grid = [2] + d_grid
        pi_bins = _parse_float_list(args.pi_bins)
        bins_grid = _parse_int_list(args.bins_grid)
        smooth_grid_raw = _parse_int_list(args.smooth_grid)
        smooth_grid, smooth_grid_mode = _normalize_smooth_grid(smooth_grid_raw)
        n_seeds = int(args.n_seeds)
        progress_every = int(args.progress_every)
        if smooth_grid_mode == "radius_to_window":
            print(
                f"smooth_grid interpreted as legacy radius values {smooth_grid_raw}; using window widths {smooth_grid}.",
                flush=True,
            )
    modes = list(DEFAULT_MODES)
    n_perm_requested = int(n_perm)
    n_perm_pool = (
        int(args.n_perm_pool) if args.n_perm_pool is not None else int(n_perm_requested)
    )
    if bool(args.fast_mode):
        n_perm = int(min(int(n_perm_requested), max(1, int(n_perm_pool))))
    else:
        n_perm = int(n_perm_requested)
        if args.n_perm_pool is not None and int(args.n_perm_pool) != int(
            n_perm_requested
        ):
            print(
                "n_perm_pool provided but fast_mode is disabled; using n_perm.",
                flush=True,
            )
    if int(n_perm) != int(n_perm_requested):
        print(
            f"fast_mode enabled: using n_perm={int(n_perm)} (requested={int(n_perm_requested)}, n_perm_pool={int(n_perm_pool)})",
            flush=True,
        )
    min_perm_eff = min(int(args.min_perm), int(n_perm))
    edge_cfg = EdgeRuleConfigB(
        threshold=float(args.fg_edge_threshold),
        strategy=str(args.fg_edge_strategy),
    )

    if int(args.representative_D) not in d_grid:
        d_rep = int(d_grid[-1])
    else:
        d_rep = int(args.representative_D)
    pi_rep = float(args.representative_pi)
    if all(abs(pi_rep - x) > 1e-12 for x in pi_bins):
        pi_rep = float(pi_bins[0])

    cfg = {
        "experiment": "Simulation Experiment B: max-stat validity + binning/smoothing sensitivity",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit_short": _git_commit_hash(),
        "test_mode": bool(args.test_mode),
        "master_seed": int(args.master_seed),
        "n_seeds": int(n_seeds),
        "seed_values": [int(args.master_seed) + i for i in range(int(n_seeds))],
        "N": int(n_cells),
        "G": int(g_total),
        "n_perm": int(n_perm),
        "n_perm_requested": int(n_perm_requested),
        "n_perm_pool": int(n_perm_pool),
        "geometries": geometries,
        "D_grid": d_grid,
        "sigma_eta": float(sigma_eta),
        "pi_bins": pi_bins,
        "bins_grid": bins_grid,
        "smooth_grid": smooth_grid,
        "modes": modes,
        "angles_origin": [0.0, 0.0],
        "statistic": "T = max_theta |E(theta)|",
        "smoothing": "circular moving-average, odd window w; w=1 means raw",
        "permutation": "donor-stratified shuffling, plus-one correction",
        "p_min_discrete": float(1.0 / (int(n_perm) + 1)),
        "underpowered": {
            "p_min": float(args.p_min),
            "min_fg_total": int(args.min_fg_total),
            "min_fg_per_donor": int(args.min_fg_per_donor),
            "min_bg_per_donor": int(args.min_bg_per_donor),
            "d_eff_min": int(args.d_eff_min),
            "min_perm": int(min_perm_eff),
            "rule": "prev<p_min OR n_fg_total<min_fg_total OR D_eff<d_eff_min OR n_perm<min_perm",
        },
        "seed_scheme": (
            "Stable blake2 hash by condition tuple. "
            "Condition seeds include (geometry,D,bins_B,smooth_w,mode,pi_target)."
        ),
        "foreground_edge_rule": {
            "strategy": str(edge_cfg.strategy),
            "threshold": float(edge_cfg.threshold),
            "enabled": str(edge_cfg.strategy).lower() != "none",
        },
        "representative_plot_slice": {"D": d_rep, "pi_target": pi_rep},
        "corner_debug_target": {
            "geometry": "density_gradient_disk",
            "D": 10,
            "pi_target": 0.9,
            "mode": "smoothed",
            "bins_B": 72,
            "smooth_w": 5,
            "k_genes": int(args.debug_corner_k),
        },
        "fast_path": {
            "fast_mode": bool(args.fast_mode),
            "save_full_profiles": bool(args.save_full_profiles),
            "skip_heavy_examples": bool(args.skip_heavy_examples),
        },
    }
    write_config(outdir, cfg)
    cache_root = ensure_dir(
        Path(args.cache_dir) if getattr(args, "cache_dir", None) else (outdir / "cache")
    )
    dataset_cache_dir = ensure_dir(cache_root / "expB_dataset_cache")
    args.cache_dir = str(cache_root)

    if str(getattr(args, "plots", "all")) == "all" and bool(
        getattr(args, "skip_heavy_examples", False)
    ):
        raise ValueError("--skip_heavy_examples is not allowed when --plots=all.")

    metrics_rows: list[dict[str, Any]] = []
    debug_cache: dict[
        tuple[str, int, str, int, int, float], list[dict[str, float | int]]
    ] = {}
    debug_targets = {
        ("disk_gaussian", 10, "raw", 36, 1, 0.2),
        ("disk_gaussian", 10, "smoothed", 36, 3, 0.2),
    }
    corner_target = {
        "geometry": "density_gradient_disk",
        "D": 10,
        "pi_target": 0.9,
        "mode": "smoothed",
        "bins_B": 72,
        "smooth_w": 5,
    }
    corner_candidates: list[dict[str, Any]] = []
    dataset_contexts: dict[tuple[int, str, int], DatasetContext] = {}

    total_start = time.time()
    seed_values = [int(args.master_seed) + i for i in range(int(n_seeds))]
    dataset_grid = build_grid(geometries, d_grid, seed_values)
    n_dataset = len(dataset_grid)

    # Lightweight mode-consistency check once in test mode.
    consistency_checked = False
    if bool(args.test_mode):
        for seed_run, geometry, d_value in dataset_grid:
            if geometry != "disk_gaussian" or int(d_value) != int(d_rep):
                continue
            context = _simulate_dataset(
                geometry=geometry,
                d_value=int(d_value),
                n_cells=int(n_cells),
                sigma_eta=float(sigma_eta),
                master_seed=int(seed_run),
                bins_grid=bins_grid,
                seed_label=int(seed_run),
                cache_dir=dataset_cache_dir,
            )
            rng_schedule = rng_from_seed(
                _seed_from_fields(
                    seed_run, "pi_schedule", geometry, int(d_value), float(sigma_eta)
                )
            )
            pi_targets = _gene_prevalence_schedule(int(g_total), pi_bins, rng_schedule)
            cand_idx = np.flatnonzero(np.abs(pi_targets - float(pi_rep)) < 1e-12)
            if cand_idx.size < 1:
                continue
            g0 = int(cand_idx[0])
            pi_target = float(pi_targets[g0])
            pi_label = _fmt_pi(pi_target)
            seed_gene = _seed_from_fields(
                seed_run,
                "gene",
                geometry,
                int(d_value),
                float(sigma_eta),
                pi_label,
                int(g0),
            )
            eta_cell = donor_effect_vector(context.donor_ids, context.eta_d)
            rng0 = rng_from_seed(seed_gene)
            f0 = simulate_null_gene(
                pi_target=float(pi_target),
                donor_eta_per_cell=eta_cell,
                rng=rng0,
            )
            f0_test, _ = apply_foreground_edge_rule_b(
                np.asarray(f0, dtype=bool),
                pi_target=float(pi_target),
                cfg=edge_cfg,
            )
            if f0_test.size > 5000:
                rr = rng_from_seed(
                    _seed_from_fields(
                        seed_run, "consistency_subsample", geometry, int(d_value)
                    )
                )
                idx = np.sort(
                    rr.choice(
                        np.arange(f0_test.size, dtype=int), size=5000, replace=False
                    )
                )
                f0_check = np.asarray(f0_test, dtype=bool)[idx]
                angles_check = np.asarray(context.angles, dtype=float)[idx]
                donors_check = np.asarray(context.donor_ids)[idx]
            else:
                f0_check = np.asarray(f0_test, dtype=bool)
                angles_check = np.asarray(context.angles, dtype=float)
                donors_check = np.asarray(context.donor_ids)
            _ = check_mode_consistency(
                f=f0_check,
                angles=angles_check,
                donor_ids=donors_check,
                n_bins=36 if 36 in bins_grid else int(bins_grid[0]),
                n_perm=min(int(n_perm), 50),
                seed=_seed_from_fields(
                    seed_run, "consistency", geometry, int(d_value), int(g0)
                ),
                donor_stratified=True,
            )
            print(
                "Mode-consistency self-check passed (raw == smoothed w=1).", flush=True
            )
            consistency_checked = True
            break
        if not consistency_checked:
            print(
                "Mode-consistency self-check skipped: representative condition not found.",
                flush=True,
            )

    dataset_items: list[dict[str, Any]] = []
    for seed_run, geometry, d_value in dataset_grid:
        dataset_items.append(
            {
                "seed": int(
                    _seed_from_fields(seed_run, "dataset_item", geometry, int(d_value))
                ),
                "seed_run": int(seed_run),
                "geometry": str(geometry),
                "d_value": int(d_value),
                "n_cells": int(n_cells),
                "g_total": int(g_total),
                "n_perm": int(n_perm),
                "sigma_eta": float(sigma_eta),
                "bins_grid": [int(x) for x in bins_grid],
                "smooth_grid": [int(x) for x in smooth_grid],
                "smooth_grid_mode": str(smooth_grid_mode),
                "modes": [str(x) for x in modes],
                "pi_bins": [float(x) for x in pi_bins],
                "progress_every": int(progress_every),
                "progress": bool(getattr(args, "progress", True))
                and int(args.n_jobs) <= 1,
                "min_perm_eff": int(min_perm_eff),
                "edge_threshold": float(args.fg_edge_threshold),
                "edge_strategy": str(args.fg_edge_strategy),
                "p_min": float(args.p_min),
                "min_fg_total": int(args.min_fg_total),
                "min_fg_per_donor": int(args.min_fg_per_donor),
                "min_bg_per_donor": int(args.min_bg_per_donor),
                "d_eff_min": int(args.d_eff_min),
                "debug_targets": [
                    {
                        "geometry": x[0],
                        "D": int(x[1]),
                        "mode": x[2],
                        "bins_B": int(x[3]),
                        "smooth_w": int(x[4]),
                        "pi_target": float(x[5]),
                    }
                    for x in sorted(debug_targets)
                ],
                "corner_target": dict(corner_target),
                "cache_dir": str(dataset_cache_dir),
            }
        )

    if int(args.n_jobs) > 1:
        dataset_out = parallel_map(
            _run_dataset_block_expb,
            dataset_items,
            n_jobs=int(args.n_jobs),
            backend=str(args.backend),
            chunk_size=max(1, int(args.chunk_size)),
            progress=bool(getattr(args, "progress", True)),
        )
    else:
        dataset_out = [_run_dataset_block_expb(it) for it in dataset_items]
    dataset_out = sorted(
        dataset_out, key=lambda r: (int(r["seed_run"]), str(r["geometry"]), int(r["D"]))
    )

    total_setup = 0.0
    total_gene_stage = 0.0
    total_perm_calls = 0
    total_cache_hits = 0
    total_perm_plan_cache_hits = 0
    for i, out in enumerate(dataset_out, start=1):
        timing = out["timing"]
        counters = out["counters"]
        total_setup += float(timing["setup_sec"])
        total_gene_stage += float(timing["genes_sec"])
        total_perm_calls += int(counters["perm_calls"])
        total_cache_hits += int(counters.get("cache_hit", 0))
        total_perm_plan_cache_hits += int(counters.get("perm_plan_cache_hit", 0))
        print(
            (
                f"[{i}/{n_dataset}] dataset={out['context_run_id']} seed_dataset={out['context_seed_dataset']} "
                f"(N={counters['n_cells']}, G={counters['n_genes']}, n_perm={counters['n_perm']}, "
                f"bins_grid={counters['bins_grid']}, n_bins_options={counters['n_bins_options']}, "
                f"cache_hit={int(counters.get('cache_hit', 0))}, "
                f"perm_plan_cache_hit={int(counters.get('perm_plan_cache_hit', 0))}) "
                f"setup={float(timing['setup_sec']):.2f}s genes={float(timing['genes_sec']):.2f}s total={float(timing['total_sec']):.2f}s"
            ),
            flush=True,
        )
        metrics_rows.extend(out["metrics_rows"])
        for dbg_row in out["debug_rows"]:
            dbg_key = (
                str(dbg_row["geometry"]),
                int(dbg_row["D"]),
                str(dbg_row["mode"]),
                int(dbg_row["bins_B"]),
                int(dbg_row["smooth_w"]),
                float(dbg_row["pi_target"]),
            )
            debug_cache.setdefault(dbg_key, []).append(
                {
                    "gene_index": int(dbg_row["gene_index"]),
                    "T_obs": float(dbg_row["T_obs"]),
                    "null_mean": float(dbg_row["null_mean"]),
                    "null_median": float(dbg_row["null_median"]),
                    "null_q95": float(dbg_row["null_q95"]),
                }
            )
        corner_candidates.extend(out["corner_candidates"])

    print(
        (
            "ExpB stage timings: "
            f"datasets={n_dataset}, setup_sum={total_setup:.2f}s, "
            f"genes_sum={total_gene_stage:.2f}s, perm_calls={total_perm_calls}, cache_hits={total_cache_hits}, "
            f"perm_plan_cache_hits={total_perm_plan_cache_hits}"
        ),
        flush=True,
    )

    # Rebuild only contexts needed for corner debug artifacts.
    needed_ctx = sorted(
        {
            (int(x["seed_run"]), str(x["geometry"]), int(x["D"]))
            for x in corner_candidates
        }
    )
    corner_ctx_cache_hits = 0
    for seed_run, geometry, d_value in needed_ctx:
        ctx = _simulate_dataset(
            geometry=str(geometry),
            d_value=int(d_value),
            n_cells=int(n_cells),
            sigma_eta=float(sigma_eta),
            master_seed=int(seed_run),
            bins_grid=bins_grid,
            seed_label=int(seed_run),
            cache_dir=dataset_cache_dir,
        )
        dataset_contexts[(seed_run, geometry, d_value)] = ctx
        corner_ctx_cache_hits += int(ctx.cache_hit)
    if needed_ctx:
        print(
            f"Corner-context cache usage: requested={len(needed_ctx)}, cache_hits={corner_ctx_cache_hits}",
            flush=True,
        )

    metrics = pd.DataFrame(metrics_rows)
    if not metrics.empty:
        metrics = metrics.sort_values(
            ["seed_run", "geometry", "D", "seed_gene", "bins_B", "mode", "smooth_w"],
            kind="mergesort",
        ).reset_index(drop=True)
    summary = summarize(metrics, int(n_perm))
    _warn_ks_outliers(summary)

    atomic_write_csv(results_dir / "metrics_long.csv", metrics)
    atomic_write_csv(results_dir / "summary.csv", summary)
    atomic_write_csv(results_dir / "summary_by_bin.csv", summary)

    _set_plot_style()
    summary = pd.read_csv(results_dir / "summary.csv")
    summary["neglog10_ks_pvalue"] = -np.log10(
        np.maximum(pd.to_numeric(summary["ks_pvalue"], errors="coerce"), 1e-300)
    )
    summary["dev_mean_p"] = pd.to_numeric(summary["mean_p"], errors="coerce") - 0.5
    summary["dev_var_p"] = pd.to_numeric(summary["var_p"], errors="coerce") - (
        1.0 / 12.0
    )
    atomic_write_csv(results_dir / "summary.csv", summary)
    atomic_write_csv(results_dir / "summary_by_bin.csv", summary)

    ks_df = derive_ks_uniformity(summary)
    typei_df = derive_typei_by_cell(summary)
    target_diag_df = build_target_cell_diagnostics_expb(summary)
    atomic_write_csv(results_dir / "ks_uniformity.csv", ks_df)
    atomic_write_csv(results_dir / "typeI_by_cell.csv", typei_df)
    if not target_diag_df.empty:
        atomic_write_csv(results_dir / "target_cell_diagnostics.csv", target_diag_df)
        (results_dir / "target_cell_diagnostics.json").write_text(
            json.dumps(target_diag_df.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
    if bool(args.test_mode):
        plot_testmode_ks_heatmap(
            summary,
            geometries=geometries,
            modes=modes,
            bins_grid=bins_grid,
            smooth_grid=smooth_grid,
            d_value=int(d_rep),
            pi_target=0.9,
            out_png=plots_dir / "ks_heatmap_test.png",
        )
        plot_testmode_p_hist(
            metrics,
            geometries=geometries,
            modes=modes,
            bins_grid=bins_grid,
            smooth_grid=smooth_grid,
            d_value=int(d_rep),
            pi_target=0.9,
            out_png=plots_dir / "p_hist_test.png",
            reduce_overplot=True,
            reduce_mode="facet_by_B",
        )
    else:
        plot_qq_small_multiples(
            metrics,
            geometries=geometries,
            bins_grid=bins_grid,
            smooth_grid=smooth_grid,
            modes=modes,
            d_rep=d_rep,
            pi_rep=pi_rep,
            n_perm=int(n_perm),
            out_png=plots_dir / "qq_small_multiples.png",
            q_max=1.0,
        )
        plot_qq_small_multiples(
            metrics,
            geometries=geometries,
            bins_grid=bins_grid,
            smooth_grid=smooth_grid,
            modes=modes,
            d_rep=d_rep,
            pi_rep=pi_rep,
            n_perm=int(n_perm),
            out_png=plots_dir / "qq_lefttail_small_multiples.png",
            q_max=0.1,
        )
        plot_p_hist_small_multiples(
            metrics,
            geometries=geometries,
            bins_grid=bins_grid,
            smooth_grid=smooth_grid,
            modes=modes,
            d_rep=d_rep,
            pi_rep=pi_rep,
            n_perm=int(n_perm),
            out_png=plots_dir / "p_hist_small_multiples.png",
        )

        _plot_heatmap_panels(
            summary,
            geometries=geometries,
            modes=modes,
            pi_bins=pi_bins,
            bins_grid=bins_grid,
            smooth_grid=smooth_grid,
            d_rep=d_rep,
            value_col="neglog10_ks_pvalue",
            title=r"Experiment B strict null KS diagnostic ($-\log_{10}(p_{KS})$)",
            out_png=plots_dir / "ks_heatmap.png",
            cmap="magma",
            vmin=0.0,
            vmax=max(
                1.0,
                (
                    float(np.nanmax(summary["neglog10_ks_pvalue"]))
                    if np.isfinite(np.nanmax(summary["neglog10_ks_pvalue"]))
                    else 1.0
                ),
            ),
        )
        _plot_heatmap_panels(
            summary,
            geometries=geometries,
            modes=modes,
            pi_bins=pi_bins,
            bins_grid=bins_grid,
            smooth_grid=smooth_grid,
            d_rep=d_rep,
            value_col="typeI_alpha05",
            title=r"Experiment B strict null Type I error ($\alpha=0.05$)",
            out_png=plots_dir / "typeI_heatmap.png",
            cmap="viridis",
            vmin=0.0,
            vmax=0.15,
        )

        # Mean/variance deviation heatmaps combined in one figure.
        rows_mv = 2 * len(geometries) * len(modes)
        cols_mv = len(pi_bins)
        fig_mv = plt.figure(
            figsize=(2.55 * cols_mv + 0.95, 1.75 * rows_mv),
            constrained_layout=False,
        )
        gs_mv = GridSpec(
            nrows=rows_mv,
            ncols=cols_mv + 1,
            figure=fig_mv,
            width_ratios=[1.0] * cols_mv + [0.06],
            left=0.10,
            right=0.92,
            top=0.91,
            bottom=0.08,
            wspace=0.27,
            hspace=0.34,
        )
        axes_mv_arr = np.empty((rows_mv, cols_mv), dtype=object)
        for i in range(rows_mv):
            for j in range(cols_mv):
                axes_mv_arr[i, j] = fig_mv.add_subplot(gs_mv[i, j])
        cax_mean = fig_mv.add_subplot(gs_mv[0 : len(geometries) * len(modes), cols_mv])
        cax_var = fig_mv.add_subplot(
            gs_mv[len(geometries) * len(modes) : rows_mv, cols_mv]
        )
        block_last_im: dict[int, Any] = {}
        for metric_block, (value_col, cmap, lim) in enumerate(
            [
                ("dev_mean_p", "coolwarm", 0.08),
                ("dev_var_p", "coolwarm", 0.04),
            ]
        ):
            for g_idx, geom in enumerate(geometries):
                for m_idx, mode in enumerate(modes):
                    row = (
                        metric_block * (len(geometries) * len(modes))
                        + g_idx * len(modes)
                        + m_idx
                    )
                    for p_idx, pi in enumerate(pi_bins):
                        ax = axes_mv_arr[row, p_idx]
                        mat = _matrix_from_summary(
                            summary,
                            geometry=geom,
                            d_value=d_rep,
                            mode=mode,
                            pi_target=pi,
                            bins_grid=bins_grid,
                            smooth_grid=smooth_grid,
                            value_col=value_col,
                        )
                        im = ax.imshow(
                            mat, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim
                        )
                        block_last_im[metric_block] = im
                        ax.set_xticks(np.arange(len(smooth_grid)))
                        ax.set_xticklabels([str(x) for x in smooth_grid])
                        ax.set_yticks(np.arange(len(bins_grid)))
                        ax.set_yticklabels([str(x) for x in bins_grid])
                        if row == rows_mv - 1:
                            ax.set_xlabel(r"$w$")
                        if p_idx == 0:
                            ax.set_ylabel(r"$B$")
                            ax.text(
                                -0.54,
                                0.5,
                                f"{_humanize_geometry_label(str(geom))}\n{_humanize_mode_label(str(mode))}",
                                transform=ax.transAxes,
                                rotation=90,
                                va="center",
                                ha="center",
                                fontsize=7.5,
                            )
                        if row in [0, len(geometries) * len(modes)]:
                            ax.set_title(rf"$\pi_{{\mathrm{{target}}}}={_fmt_pi(pi)}$")
                        for i in range(mat.shape[0]):
                            for j in range(mat.shape[1]):
                                val = mat[i, j]
                                txt = "NA" if not np.isfinite(val) else f"{val:+.3f}"
                                if np.isfinite(val):
                                    rgba = im.cmap(im.norm(val))
                                    color = _text_color_for_rgba(rgba)
                                else:
                                    color = "black"
                                ax.text(
                                    j,
                                    i,
                                    txt,
                                    ha="center",
                                    va="center",
                                    fontsize=6.2,
                                    color=color,
                                )
        if 0 in block_last_im:
            cbar_mean = fig_mv.colorbar(block_last_im[0], cax=cax_mean)
            cbar_mean.set_label(_heatmap_value_label("dev_mean_p"))
        if 1 in block_last_im:
            cbar_var = fig_mv.colorbar(block_last_im[1], cax=cax_var)
            cbar_var.set_label(_heatmap_value_label("dev_var_p"))
        fig_mv.text(
            0.5,
            0.94,
            _heatmap_value_label("dev_mean_p"),
            ha="center",
            va="center",
            fontsize=8,
        )
        fig_mv.text(
            0.5,
            0.50,
            _heatmap_value_label("dev_var_p"),
            ha="center",
            va="center",
            fontsize=8,
        )
        fig_mv.suptitle(
            f"Experiment B strict null mean/variance deviation heatmaps (D={d_rep})",
            y=0.997,
        )
        savefig(fig_mv, plots_dir / "meanvar_heatmap.png")
        plt.close(fig_mv)

        bins_rep = 36 if 36 in bins_grid else bins_grid[0]
        w_rep = 3 if 3 in smooth_grid else smooth_grid[0]
        plot_underpowered_rate(
            summary,
            geometries=geometries,
            d_grid=d_grid,
            pi_bins=pi_bins,
            bins_rep=bins_rep,
            w_rep=w_rep,
            modes=modes,
            out_png=plots_dir / "underpowered_rate.png",
        )

        _run_sanity_checks(
            summary,
            metrics,
            debug_cache,
            n_perm=int(n_perm),
            sanity_d=d_rep,
            sanity_pi=pi_rep,
        )
        if (
            bool(getattr(args, "skip_heavy_examples", False))
            and str(getattr(args, "plots", "all")) != "all"
        ):
            print(
                "Skipping heavy corner examples (--skip_heavy_examples enabled).",
                flush=True,
            )
        else:
            _run_corner_debug_artifacts(
                candidates=corner_candidates,
                dataset_contexts=dataset_contexts,
                plots_dir=plots_dir,
                results_dir=results_dir,
                n_perm=int(n_perm),
                debug_k=int(args.debug_corner_k),
                master_seed=int(args.master_seed),
                edge_cfg=edge_cfg,
                save_full_profiles=bool(getattr(args, "save_full_profiles", False)),
            )

    if bool(args.test_mode):
        sub = summary.loc[
            (summary["geometry"] == "disk_gaussian")
            & (summary["D"] == int(d_rep))
            & (summary["mode"] == "raw")
            & (summary["bins_B"] == 36)
            & (summary["smooth_w"] == 1)
            & (summary["pi_target"] == 0.2)
        ]
        if not sub.empty:
            row = sub.iloc[0]
            ks_val = (
                float(row["ks_pvalue"]) if pd.notna(row["ks_pvalue"]) else float("nan")
            )
            typei = (
                float(row["typeI_alpha05"])
                if pd.notna(row["typeI_alpha05"])
                else float("nan")
            )
            if np.isfinite(typei) and typei > 0.15:
                raise RuntimeError(
                    f"Critical validation failed in test_mode: typeI_alpha05={typei:.3f} > 0.15"
                )
            if np.isfinite(ks_val) and ks_val < 1e-6:
                raise RuntimeError(
                    f"Critical validation failed in test_mode: ks_pvalue={ks_val:.3g} < 1e-6"
                )
        required = [
            results_dir / "summary.csv",
            plots_dir / "ks_heatmap_test.png",
            plots_dir / "p_hist_test.png",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"Missing required test_mode outputs: {missing}")

    report_path = write_report_markdown(outdir, results_dir)
    elapsed = time.time() - total_start
    n_non_under = int((~metrics["underpowered"]).sum())
    print(
        (
            f"Completed Experiment B in {elapsed/60.0:.2f} min. "
            f"rows={metrics.shape[0]}, non_underpowered={n_non_under}, "
            f"p_min={1.0/(int(n_perm)+1):.4f}"
        ),
        flush=True,
    )
    print(f"Report written to: {report_path}", flush=True)
    print(f"Outputs written to: {outdir}", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Simulation Experiment B: max-stat validity + binning/smoothing sensitivity."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/simulations/expB_maxstat_sensitivity",
        help="Output directory.",
    )
    parser.add_argument("--N", type=int, default=20000, help="Number of cells.")
    parser.add_argument(
        "--G", type=int, default=1000, help="Number of genes per dataset condition."
    )
    parser.add_argument(
        "--n_perm",
        type=int,
        default=500,
        help="Permutation count per gene and bin count.",
    )
    parser.add_argument(
        "--master_seed", type=int, default=123, help="Master random seed."
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run a tiny deterministic subset for CI/local sanity checks.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help="Number of consecutive seeds to pool (master_seed + i).",
    )
    parser.add_argument(
        "--geometries",
        type=str,
        default=",".join(DEFAULT_GEOMETRIES),
        help="Comma-separated geometry list.",
    )
    parser.add_argument(
        "--D_grid", type=str, default="5,10", help="Comma-separated donor-count grid."
    )
    parser.add_argument(
        "--include_d2_stress", action="store_true", help="Include D=2 stress condition."
    )
    parser.add_argument(
        "--sigma_eta",
        type=float,
        default=0.4,
        help="Donor random effect std-dev (logit scale).",
    )
    parser.add_argument(
        "--pi_bins",
        type=str,
        default="0.01,0.05,0.2,0.6,0.9",
        help="Comma-separated prevalence bins.",
    )
    parser.add_argument(
        "--bins_grid",
        type=str,
        default="24,36,48,72",
        help="Comma-separated bin counts.",
    )
    parser.add_argument(
        "--smooth_grid",
        type=str,
        default="1,3,5",
        help="Comma-separated odd smoothing windows.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50,
        help="Progress print interval in genes.",
    )

    parser.add_argument(
        "--p_min", type=float, default=0.005, help="Underpowered prevalence threshold."
    )
    parser.add_argument(
        "--min_fg_total",
        type=int,
        default=50,
        help="Underpowered minimum foreground count.",
    )
    parser.add_argument(
        "--min_fg_per_donor",
        type=int,
        default=10,
        help="D_eff gate: min foreground per donor.",
    )
    parser.add_argument(
        "--min_bg_per_donor",
        type=int,
        default=10,
        help="D_eff gate: min background per donor.",
    )
    parser.add_argument(
        "--d_eff_min", type=int, default=2, help="Minimum informative donors."
    )
    parser.add_argument(
        "--min_perm",
        type=int,
        default=200,
        help="Minimum permutations for non-underpowered.",
    )
    parser.add_argument(
        "--fg_edge_strategy",
        type=str,
        default="none",
        choices=["none", "complement"],
        help="Optional high-prevalence foreground rule (default none).",
    )
    parser.add_argument(
        "--fg_edge_threshold",
        type=float,
        default=0.8,
        help="Edge trigger threshold for pi_target or fg_fraction.",
    )

    parser.add_argument(
        "--representative_D",
        type=int,
        default=10,
        help="Representative D used in compact figures/sanity checks.",
    )
    parser.add_argument(
        "--representative_pi",
        type=float,
        default=0.2,
        help="Representative pi used in compact figures/sanity checks.",
    )
    parser.add_argument(
        "--debug_corner_k",
        type=int,
        default=20,
        help="Number of genes for targeted corner debug artifacts.",
    )
    add_common_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    warnings.simplefilter("default", RuntimeWarning)
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if bool(getattr(args, "dry_run", False)):
        print("dry_run=True: skipping execution for legacy runner.", flush=True)
        return 0
    if args.N <= 0 or args.G <= 0 or args.n_perm <= 0 or args.n_seeds <= 0:
        raise ValueError("N, G, n_perm, and n_seeds must be positive.")
    run_ctx = prepare_legacy_run(args, "expB_maxstat_sensitivity", __file__)
    run_experiment(args)
    finalize_legacy_run(run_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
