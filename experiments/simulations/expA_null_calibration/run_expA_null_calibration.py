#!/usr/bin/env python3
"""Simulation Experiment A: null calibration across prevalence and donor effects.

This script simulates strict-null genes (no geometry/QC signal), scores each gene
with BioRSP permutation testing, and generates calibration summaries and figures.
"""

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
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kstest

# Ensure non-interactive rendering and writable matplotlib cache in sandboxed envs.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-biorsp-expA"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "xdg-cache-biorsp-expA"),
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from biorsp.plotting.style import (
    DEFAULT_STYLE,
    apply_style,
    choose_text_color,
    finalize_fig,
    safe_suptitle,
    should_plot,
)
from biorsp.rsp import compute_rsp_profile_from_boolean
from biorsp.scoring import bh_fdr
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
from experiments.simulations._shared.models import simulate_null_gene
from experiments.simulations._shared.parallel import parallel_map
from experiments.simulations._shared.plots import (
    plot_embedding_with_foreground,
    plot_rsp_polar,
)
from experiments.simulations._shared.plots import wilson_ci as shared_wilson_ci
from experiments.simulations._shared.reporting import write_report_expA
from experiments.simulations._shared.runner import (
    finalize_legacy_run,
    prepare_legacy_run,
)
from experiments.simulations._shared.seeding import rng_from_seed, stable_seed
from experiments.simulations._sim_testmode import banner as testmode_banner
from experiments.simulations._sim_testmode import get_testmode_config, resolve_outdir
from experiments.simulations.expA_null_calibration.cell_runner import (
    EdgeRuleConfig,
    GateConfig,
    build_target_cell_diagnostics,
    run_single_null_gene,
)

DEFAULT_PREVALENCE_BINS = [0.002, 0.005, 0.01, 0.05, 0.2, 0.6, 0.9]
DEFAULT_P_MIN = 0.005
DEFAULT_MIN_FG_TOTAL = 50
DEFAULT_MIN_FG_PER_DONOR = 10
DEFAULT_MIN_BG_PER_DONOR = 10
DEFAULT_D_EFF_MIN = 2
DEFAULT_MIN_PERM = 200
DEFAULT_BH_VALIDATION_MODE = "panel_bh"
DEFAULT_BH_PANEL_SIZE = 15
DEFAULT_BH_PANEL_STRATEGY = "prevalence_stratified_fixed"
TAIL_TOLERANCES = {0.05: 0.01, 0.01: 0.005, 0.005: 0.0025}
P_FLOOR_TOL = 1e-12


@dataclass(frozen=True)
class RunContext:
    X_sim: np.ndarray
    angles: np.ndarray
    donor_ids: np.ndarray
    eta_d: np.ndarray
    library_size: np.ndarray
    pct_mt: np.ndarray
    bin_id: np.ndarray
    bin_counts_total: np.ndarray


def _parse_int_list(value: str) -> list[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def _parse_float_list(value: str) -> list[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one float value.")
    return out


def _stable_hash_run(donor_count: int, sigma_eta: float) -> int:
    """Deterministic hash in [0, 1e6) for seed derivation."""
    return int(
        stable_seed(0, "expA_run", int(donor_count), float(sigma_eta)) % 1_000_000
    )


def _format_prevalence_label(p: float) -> str:
    return f"{p:.3f}".rstrip("0").rstrip(".")


def _build_even_donor_assignment(n_cells: int, n_donors: int) -> np.ndarray:
    base = n_cells // n_donors
    remainder = n_cells % n_donors
    counts = np.full(n_donors, base, dtype=int)
    counts[:remainder] += 1
    donor_ids = np.repeat(np.arange(n_donors, dtype=np.int16), counts)
    if donor_ids.size != n_cells:
        raise AssertionError("Donor assignment length mismatch.")
    return donor_ids


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.size != y_arr.size or x_arr.size < 3:
        return float("nan")
    sx = float(np.std(x_arr))
    sy = float(np.std(y_arr))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _validate_qc_independence(
    run_id: str,
    x_sim: np.ndarray,
    library_size: np.ndarray,
    pct_mt: np.ndarray,
    corr_threshold: float,
) -> dict[str, float]:
    x = x_sim[:, 0]
    y = x_sim[:, 1]
    checks = {
        "corr_library_x": _corr(library_size, x),
        "corr_library_y": _corr(library_size, y),
        "corr_pctmt_x": _corr(pct_mt, x),
        "corr_pctmt_y": _corr(pct_mt, y),
    }
    max_abs = np.nanmax(np.abs(np.fromiter(checks.values(), dtype=float)))
    if np.isfinite(max_abs) and max_abs >= corr_threshold:
        warnings.warn(
            (
                f"[{run_id}] QC/geometry correlation check exceeded threshold "
                f"{corr_threshold:.3f}; max|corr|={max_abs:.4f}. "
                "Proceeding but this run should be interpreted cautiously."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    return checks


def _compute_bin_id(angles: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 2.0 * np.pi, int(n_bins) + 1, endpoint=True)
    bin_id = np.digitize(angles, edges, right=False) - 1
    bin_id = np.where(bin_id == int(n_bins), int(n_bins) - 1, bin_id).astype(np.int32)
    counts = np.bincount(bin_id, minlength=int(n_bins)).astype(np.int64)
    return bin_id, counts


def simulate_run_context(
    *,
    n_cells: int,
    n_bins: int,
    n_donors: int,
    sigma_eta: float,
    seed_run: int,
    mu0: float,
    sigma_l: float,
    qc_corr_threshold: float,
    run_id: str,
) -> tuple[RunContext, dict[str, float]]:
    rng = rng_from_seed(seed_run)

    x_sim, _ = sample_disk_gaussian(n_cells, rng)
    angles = np.mod(np.arctan2(x_sim[:, 1], x_sim[:, 0]), 2.0 * np.pi)

    donor_ids = assign_donors(n_cells, n_donors, rng)

    eta_d = sample_donor_effects(n_donors, float(sigma_eta), rng)

    delta_mu_d = rng.normal(loc=0.0, scale=0.3, size=n_donors).astype(float)
    mu_d = float(mu0) + delta_mu_d
    library_size = rng.lognormal(mean=mu_d[donor_ids], sigma=float(sigma_l)).astype(
        float
    )

    donor_mt_shift = rng.uniform(low=-0.02, high=0.02, size=n_donors).astype(float)
    donor_mt_mean = np.clip(0.07 + donor_mt_shift, 0.005, 0.5)
    mt_concentration = 180.0
    a_d = np.clip(donor_mt_mean * mt_concentration, 0.1, None)
    b_d = np.clip((1.0 - donor_mt_mean) * mt_concentration, 0.1, None)
    pct_mt = rng.beta(a=a_d[donor_ids], b=b_d[donor_ids]).astype(float)

    qc_corr = _validate_qc_independence(
        run_id=run_id,
        x_sim=x_sim,
        library_size=library_size,
        pct_mt=pct_mt,
        corr_threshold=qc_corr_threshold,
    )

    bin_id, bin_counts_total = _compute_bin_id(angles, n_bins)

    return (
        RunContext(
            X_sim=x_sim,
            angles=angles,
            donor_ids=donor_ids,
            eta_d=eta_d,
            library_size=library_size,
            pct_mt=pct_mt,
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        ),
        qc_corr,
    )


def _gene_prevalence_schedule(
    g_total: int,
    prevalence_bins: Iterable[float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    bins = np.asarray(list(prevalence_bins), dtype=float)
    n_bins = int(bins.size)
    if n_bins < 1:
        raise ValueError("Need at least one prevalence bin.")
    base = g_total // n_bins
    remainder = g_total % n_bins
    counts = np.full(n_bins, base, dtype=int)
    counts[:remainder] += 1
    pi_target = np.repeat(bins, counts)
    labels = np.repeat(np.array([_format_prevalence_label(x) for x in bins]), counts)
    order = rng.permutation(pi_target.size)
    return pi_target[order], labels[order]


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    _ = z  # Compatibility placeholder for previous call signature.
    return shared_wilson_ci(int(k), int(n), alpha=0.05)


def bh_feasibility_metrics(m_full: int, n_perm: int) -> dict[str, float | int | bool]:
    m_int = int(m_full)
    n_perm_int = int(n_perm)
    if m_int <= 0:
        raise ValueError("m_full must be positive.")
    if n_perm_int <= 0:
        raise ValueError("n_perm must be positive.")
    min_attainable_p = float(1.0 / (n_perm_int + 1))
    bh_min_rejectable_p_q05 = float(0.05 / m_int)
    bh_min_rejectable_p_q10 = float(0.10 / m_int)
    return {
        "m_full_tests": m_int,
        "n_perm": n_perm_int,
        "min_attainable_p": min_attainable_p,
        "bh_min_rejectable_p_q05": bh_min_rejectable_p_q05,
        "bh_min_rejectable_p_q10": bh_min_rejectable_p_q10,
        "bh_feasible_q05": bool(bh_min_rejectable_p_q05 >= min_attainable_p),
        "bh_feasible_q10": bool(bh_min_rejectable_p_q10 >= min_attainable_p),
    }


def required_n_for_margin(
    p: float = 0.05, half_width: float = 0.01, z: float = 1.96
) -> int:
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1).")
    if half_width <= 0.0:
        raise ValueError("half_width must be positive.")
    if z <= 0.0:
        raise ValueError("z must be positive.")
    return int(math.ceil(p * (1.0 - p) * (z / half_width) ** 2))


def _rate_ci(values: np.ndarray, threshold: float) -> tuple[float, float, float, int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n <= 0:
        return float("nan"), float("nan"), float("nan"), 0
    k = int(np.sum(arr <= float(threshold)))
    rate = float(k / n)
    ci_low, ci_high = wilson_ci(k, n)
    return rate, float(ci_low), float(ci_high), k


def _tail_inflation_flag(ci_low: float, alpha: float, tol: float) -> bool:
    if not np.isfinite(float(ci_low)):
        return False
    return bool(float(ci_low) > float(alpha) + float(tol))


def validate_pvalue_floor(
    metrics: pd.DataFrame, n_perm: int, tol: float = P_FLOOR_TOL
) -> float:
    pvals = pd.to_numeric(metrics["p_T"], errors="coerce").to_numpy(dtype=float)
    pvals = pvals[np.isfinite(pvals)]
    if pvals.size == 0:
        raise RuntimeError("No finite p_T values were produced.")
    if np.any(np.isclose(pvals, 0.0, atol=float(tol), rtol=0.0)):
        raise RuntimeError(
            "Invalid p-value floor: found p_T == 0 (plus-one correction violated)."
        )
    min_attainable_p = float(1.0 / (int(n_perm) + 1))
    min_p = float(np.min(pvals))
    if min_p < min_attainable_p - float(tol):
        raise RuntimeError(
            "Invalid p-value floor: observed min(p_T) below 1/(n_perm+1). "
            f"min(p_T)={min_p:.12g}, expected_floor={min_attainable_p:.12g}."
        )
    return min_p


def summarize_by_bin(
    metrics: pd.DataFrame, *, n_perm: int, m_full_tests: int
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    full_bh = bh_feasibility_metrics(m_full=m_full_tests, n_perm=n_perm)
    min_attainable_p = float(full_bh["min_attainable_p"])
    n_required = required_n_for_margin(p=0.05, half_width=0.01, z=1.96)
    group_cols = ["D", "sigma_eta", "prevalence_bin"]
    for (d, sigma_eta, prevalence_bin), grp in metrics.groupby(group_cols, sort=True):
        n_genes = int(grp.shape[0])
        mean_prev_obs = float(grp["prev_obs"].mean())
        frac_underpowered = float(grp["underpowered"].mean())
        d_eff_vals = pd.to_numeric(grp["D_eff"], errors="coerce").to_numpy(dtype=float)
        valid = grp.loc[~grp["underpowered"]]
        n_valid = int(valid.shape[0])
        pvals = pd.to_numeric(valid["p_T"], errors="coerce").to_numpy(dtype=float)
        pvals = pvals[np.isfinite(pvals)]
        qvals = pd.to_numeric(valid.get("q_T", np.nan), errors="coerce").to_numpy(
            dtype=float
        )
        qvals = qvals[np.isfinite(qvals)]

        typei_p05, p05_ci_low, p05_ci_high, _ = _rate_ci(pvals, 0.05)
        typei_p01, p01_ci_low, p01_ci_high, _ = _rate_ci(pvals, 0.01)
        typei_p005, p005_ci_low, p005_ci_high, _ = _rate_ci(pvals, 0.005)
        typei_p00333, p00333_ci_low, p00333_ci_high, _ = _rate_ci(
            pvals, min_attainable_p
        )
        typei_q05, q05_ci_low, q05_ci_high, _ = _rate_ci(qvals, 0.05)
        typei_q10, q10_ci_low, q10_ci_high, _ = _rate_ci(qvals, 0.10)

        q05_status = "ok"
        q10_status = "ok"
        if not bool(full_bh["bh_feasible_q05"]):
            typei_q05 = float("nan")
            q05_ci_low = float("nan")
            q05_ci_high = float("nan")
            q05_status = "BH infeasible; metric suppressed"
        if not bool(full_bh["bh_feasible_q10"]):
            typei_q10 = float("nan")
            q10_ci_low = float("nan")
            q10_ci_high = float("nan")
            q10_status = "BH infeasible; metric suppressed"

        se_typei_p05 = (
            float(math.sqrt(0.05 * 0.95 / n_valid)) if n_valid > 0 else float("nan")
        )

        rows.append(
            {
                "D": int(d),
                "sigma_eta": float(sigma_eta),
                "prevalence_bin": str(prevalence_bin),
                "n_genes": n_genes,
                "n_non_underpowered": n_valid,
                "mean_prev_obs": mean_prev_obs,
                "frac_underpowered": frac_underpowered,
                "mean_D_eff": float(np.nanmean(d_eff_vals)),
                "median_D_eff": float(np.nanmedian(d_eff_vals)),
                "min_D_eff": float(np.nanmin(d_eff_vals)),
                "max_D_eff": float(np.nanmax(d_eff_vals)),
                "typeI_p05": typei_p05,
                "typeI_p05_ci_low": p05_ci_low,
                "typeI_p05_ci_high": p05_ci_high,
                "typeI_p01": typei_p01,
                "typeI_p01_ci_low": p01_ci_low,
                "typeI_p01_ci_high": p01_ci_high,
                "typeI_p005": typei_p005,
                "typeI_p005_ci_low": p005_ci_low,
                "typeI_p005_ci_high": p005_ci_high,
                "typeI_p00333": typei_p00333,
                "typeI_p00333_ci_low": p00333_ci_low,
                "typeI_p00333_ci_high": p00333_ci_high,
                "tail_inflation_flag_p05": _tail_inflation_flag(
                    p05_ci_low, 0.05, TAIL_TOLERANCES[0.05]
                ),
                "tail_inflation_flag_p01": _tail_inflation_flag(
                    p01_ci_low, 0.01, TAIL_TOLERANCES[0.01]
                ),
                "tail_inflation_flag_p005": _tail_inflation_flag(
                    p005_ci_low, 0.005, TAIL_TOLERANCES[0.005]
                ),
                "se_typeI_p05": se_typei_p05,
                "n_required_pm01_95ci": int(n_required),
                "mc_sufficient_pm01_95ci": bool(n_valid >= n_required),
                "m_full_tests": int(full_bh["m_full_tests"]),
                "min_attainable_p": min_attainable_p,
                "bh_min_rejectable_p_q05": float(full_bh["bh_min_rejectable_p_q05"]),
                "bh_min_rejectable_p_q10": float(full_bh["bh_min_rejectable_p_q10"]),
                "bh_feasible_q05": bool(full_bh["bh_feasible_q05"]),
                "bh_feasible_q10": bool(full_bh["bh_feasible_q10"]),
                "typeI_q05": typei_q05,
                "typeI_q05_ci_low": q05_ci_low,
                "typeI_q05_ci_high": q05_ci_high,
                "typeI_q05_status": q05_status,
                "typeI_q10": typei_q10,
                "typeI_q10_ci_low": q10_ci_low,
                "typeI_q10_ci_high": q10_ci_high,
                "typeI_q10_status": q10_status,
            }
        )
    return pd.DataFrame(rows)


def summarize_ks(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    group_cols = ["D", "sigma_eta", "prevalence_bin"]
    for (d, sigma_eta, prevalence_bin), grp in metrics.groupby(group_cols, sort=True):
        valid = grp.loc[~grp["underpowered"], "p_T"].to_numpy(dtype=float)
        valid = valid[np.isfinite(valid)]
        n_valid = int(valid.size)
        if n_valid >= 5:
            ks = kstest(valid, "uniform")
            ks_stat = float(ks.statistic)
            ks_p = float(ks.pvalue)
            ks_neglog10 = float(-np.log10(max(ks_p, 1e-300)))
        else:
            ks_stat = float("nan")
            ks_p = float("nan")
            ks_neglog10 = float("nan")
        rows.append(
            {
                "D": int(d),
                "sigma_eta": float(sigma_eta),
                "prevalence_bin": str(prevalence_bin),
                "n_non_underpowered": n_valid,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "p_KS": ks_p,
                "neglog10_ks_pvalue": ks_neglog10,
                "neglog10_p_KS": ks_neglog10,
            }
        )
    return pd.DataFrame(rows)


def compute_panel_bh_validation(
    metrics: pd.DataFrame,
    *,
    bh_validation_mode: str,
    bh_panel_size: int,
    bh_panel_strategy: str,
    bh_panel_seed: int,
    n_perm: int,
) -> pd.DataFrame:
    panel_rows: list[dict[str, Any]] = []
    keys = ["D", "sigma_eta", "prevalence_bin"]
    if str(bh_validation_mode) != "panel_bh":
        for (d, sigma_eta, prevalence_bin), _ in metrics.groupby(keys, sort=True):
            panel_rows.append(
                {
                    "D": int(d),
                    "sigma_eta": float(sigma_eta),
                    "prevalence_bin": str(prevalence_bin),
                    "bh_validation_mode": str(bh_validation_mode),
                    "bh_panel_strategy": str(bh_panel_strategy),
                    "bh_panel_size_target": int(bh_panel_size),
                    "bh_panel_size_actual": 0,
                    "panel_sampling_scope": "disabled",
                    "panel_typeI_q05": float("nan"),
                    "panel_typeI_q05_ci_low": float("nan"),
                    "panel_typeI_q05_ci_high": float("nan"),
                    "panel_typeI_q10": float("nan"),
                    "panel_typeI_q10_ci_low": float("nan"),
                    "panel_typeI_q10_ci_high": float("nan"),
                    "panel_min_attainable_p": float(1.0 / (int(n_perm) + 1)),
                    "panel_bh_min_rejectable_p_q05": float("nan"),
                    "panel_bh_min_rejectable_p_q10": float("nan"),
                    "panel_bh_feasible_q05": False,
                    "panel_bh_feasible_q10": False,
                    "panel_typeI_q05_status": "panel BH disabled (uniformity_only)",
                    "panel_typeI_q10_status": "panel BH disabled (uniformity_only)",
                    "bh_panel_seed": int(bh_panel_seed),
                }
            )
        return pd.DataFrame(panel_rows)

    valid = metrics.loc[~metrics["underpowered"]].copy()
    if valid.empty:
        return pd.DataFrame(panel_rows)

    if str(bh_panel_strategy) == "prevalence_stratified_fixed":
        for (d, sigma_eta, prevalence_bin), grp in valid.groupby(keys, sort=True):
            n_valid = int(grp.shape[0])
            sample_n = min(int(bh_panel_size), n_valid)
            panel_pvals = np.array([], dtype=float)
            if sample_n > 0:
                rng = rng_from_seed(
                    stable_seed(
                        int(bh_panel_seed),
                        "expA_panel_bh",
                        str(bh_panel_strategy),
                        int(d),
                        float(sigma_eta),
                        str(prevalence_bin),
                    )
                )
                chosen = rng.choice(n_valid, size=sample_n, replace=False)
                panel_pvals = pd.to_numeric(
                    grp.iloc[chosen]["p_T"], errors="coerce"
                ).to_numpy(dtype=float)
                panel_pvals = panel_pvals[np.isfinite(panel_pvals)]
            panel_q = bh_fdr(panel_pvals) if panel_pvals.size > 0 else np.array([])
            panel_bh = (
                bh_feasibility_metrics(m_full=sample_n, n_perm=n_perm)
                if sample_n > 0
                else {
                    "min_attainable_p": float(1.0 / (int(n_perm) + 1)),
                    "bh_min_rejectable_p_q05": float("nan"),
                    "bh_min_rejectable_p_q10": float("nan"),
                    "bh_feasible_q05": False,
                    "bh_feasible_q10": False,
                }
            )
            t05, t05_lo, t05_hi, _ = _rate_ci(panel_q, 0.05)
            t10, t10_lo, t10_hi, _ = _rate_ci(panel_q, 0.10)
            q05_status = "ok"
            q10_status = "ok"
            if sample_n <= 0:
                q05_status = "no panel genes selected"
                q10_status = "no panel genes selected"
            elif not bool(panel_bh["bh_feasible_q05"]):
                t05, t05_lo, t05_hi = float("nan"), float("nan"), float("nan")
                q05_status = "BH infeasible on panel; metric suppressed"
            if sample_n > 0 and (not bool(panel_bh["bh_feasible_q10"])):
                t10, t10_lo, t10_hi = float("nan"), float("nan"), float("nan")
                q10_status = "BH infeasible on panel; metric suppressed"
            panel_rows.append(
                {
                    "D": int(d),
                    "sigma_eta": float(sigma_eta),
                    "prevalence_bin": str(prevalence_bin),
                    "bh_validation_mode": str(bh_validation_mode),
                    "bh_panel_strategy": str(bh_panel_strategy),
                    "bh_panel_size_target": int(bh_panel_size),
                    "bh_panel_size_actual": int(sample_n),
                    "panel_sampling_scope": "cell",
                    "panel_typeI_q05": t05,
                    "panel_typeI_q05_ci_low": t05_lo,
                    "panel_typeI_q05_ci_high": t05_hi,
                    "panel_typeI_q10": t10,
                    "panel_typeI_q10_ci_low": t10_lo,
                    "panel_typeI_q10_ci_high": t10_hi,
                    "panel_min_attainable_p": float(panel_bh["min_attainable_p"]),
                    "panel_bh_min_rejectable_p_q05": float(
                        panel_bh["bh_min_rejectable_p_q05"]
                    ),
                    "panel_bh_min_rejectable_p_q10": float(
                        panel_bh["bh_min_rejectable_p_q10"]
                    ),
                    "panel_bh_feasible_q05": bool(panel_bh["bh_feasible_q05"]),
                    "panel_bh_feasible_q10": bool(panel_bh["bh_feasible_q10"]),
                    "panel_typeI_q05_status": q05_status,
                    "panel_typeI_q10_status": q10_status,
                    "bh_panel_seed": int(bh_panel_seed),
                }
            )
        return pd.DataFrame(panel_rows)

    if str(bh_panel_strategy) != "random_fixed":
        raise ValueError(
            "bh_panel_strategy must be one of {'prevalence_stratified_fixed','random_fixed'}."
        )

    for (d, sigma_eta), run_grp in valid.groupby(["D", "sigma_eta"], sort=True):
        n_run = int(run_grp.shape[0])
        run_sample_n = min(int(bh_panel_size), n_run)
        run_panel = run_grp.iloc[0:0].copy()
        if run_sample_n > 0:
            rng = rng_from_seed(
                stable_seed(
                    int(bh_panel_seed),
                    "expA_panel_bh",
                    str(bh_panel_strategy),
                    int(d),
                    float(sigma_eta),
                )
            )
            idx = rng.choice(n_run, size=run_sample_n, replace=False)
            run_panel = run_grp.iloc[idx].copy()
            run_panel["_panel_q"] = bh_fdr(
                pd.to_numeric(run_panel["p_T"], errors="coerce").to_numpy(dtype=float)
            )
        run_bh = (
            bh_feasibility_metrics(m_full=run_sample_n, n_perm=n_perm)
            if run_sample_n > 0
            else {
                "min_attainable_p": float(1.0 / (int(n_perm) + 1)),
                "bh_min_rejectable_p_q05": float("nan"),
                "bh_min_rejectable_p_q10": float("nan"),
                "bh_feasible_q05": False,
                "bh_feasible_q10": False,
            }
        )
        for prevalence_bin, _cell_grp in run_grp.groupby("prevalence_bin", sort=True):
            cell_panel = run_panel.loc[
                run_panel["prevalence_bin"].astype(str) == str(prevalence_bin)
            ].copy()
            cell_q = pd.to_numeric(cell_panel.get("_panel_q", np.nan), errors="coerce")
            cell_q = cell_q.to_numpy(dtype=float)
            cell_q = cell_q[np.isfinite(cell_q)]
            cell_n = int(cell_q.size)
            t05, t05_lo, t05_hi, _ = _rate_ci(cell_q, 0.05)
            t10, t10_lo, t10_hi, _ = _rate_ci(cell_q, 0.10)
            q05_status = "ok"
            q10_status = "ok"
            if run_sample_n <= 0:
                q05_status = "no panel genes selected"
                q10_status = "no panel genes selected"
            elif cell_n <= 0:
                q05_status = "no sampled panel genes in prevalence bin"
                q10_status = "no sampled panel genes in prevalence bin"
            elif not bool(run_bh["bh_feasible_q05"]):
                t05, t05_lo, t05_hi = float("nan"), float("nan"), float("nan")
                q05_status = "BH infeasible on panel; metric suppressed"
            if (
                (run_sample_n > 0)
                and (cell_n > 0)
                and (not bool(run_bh["bh_feasible_q10"]))
            ):
                t10, t10_lo, t10_hi = float("nan"), float("nan"), float("nan")
                q10_status = "BH infeasible on panel; metric suppressed"
            panel_rows.append(
                {
                    "D": int(d),
                    "sigma_eta": float(sigma_eta),
                    "prevalence_bin": str(prevalence_bin),
                    "bh_validation_mode": str(bh_validation_mode),
                    "bh_panel_strategy": str(bh_panel_strategy),
                    "bh_panel_size_target": int(bh_panel_size),
                    "bh_panel_size_actual": int(cell_n),
                    "bh_panel_size_actual_run": int(run_sample_n),
                    "panel_sampling_scope": "run_global",
                    "panel_typeI_q05": t05,
                    "panel_typeI_q05_ci_low": t05_lo,
                    "panel_typeI_q05_ci_high": t05_hi,
                    "panel_typeI_q10": t10,
                    "panel_typeI_q10_ci_low": t10_lo,
                    "panel_typeI_q10_ci_high": t10_hi,
                    "panel_min_attainable_p": float(run_bh["min_attainable_p"]),
                    "panel_bh_min_rejectable_p_q05": float(
                        run_bh["bh_min_rejectable_p_q05"]
                    ),
                    "panel_bh_min_rejectable_p_q10": float(
                        run_bh["bh_min_rejectable_p_q10"]
                    ),
                    "panel_bh_feasible_q05": bool(run_bh["bh_feasible_q05"]),
                    "panel_bh_feasible_q10": bool(run_bh["bh_feasible_q10"]),
                    "panel_typeI_q05_status": q05_status,
                    "panel_typeI_q10_status": q10_status,
                    "bh_panel_seed": int(bh_panel_seed),
                }
            )
    return pd.DataFrame(panel_rows)


def _prevalence_float(label: str) -> float:
    try:
        return float(label)
    except ValueError:
        return float("nan")


N_MIN = int(DEFAULT_STYLE.n_min)
FIGSIZE_SMALL = DEFAULT_STYLE.fig_small
FIGSIZE_WIDE = DEFAULT_STYLE.fig_wide
ALPHA_LINE = DEFAULT_STYLE.alpha_line
ALPHA_SCATTER_BG = DEFAULT_STYLE.alpha_bg
ALPHA_SCATTER_FG = DEFAULT_STYLE.alpha_fg
COLORBAR_SHRINK = DEFAULT_STYLE.cbar_shrink
COLORBAR_PAD = DEFAULT_STYLE.cbar_pad
FONT_SIZES = {
    "suptitle": DEFAULT_STYLE.fs_suptitle,
    "title": DEFAULT_STYLE.fs_title,
    "label": DEFAULT_STYLE.fs_label,
    "tick": DEFAULT_STYLE.fs_tick,
    "legend": DEFAULT_STYLE.fs_legend,
    "annotation": DEFAULT_STYLE.fs_annot,
}


LABEL_DEFF_MATH = r"$D_{\mathrm{eff}}$"
LABEL_PI_TARGET_MATH = r"$\pi_{\mathrm{target}}$"
LABEL_PT_MATH = r"$p_T$"
LABEL_SIGMA_ETA_MATH = r"$\sigma_\eta$"
LABEL_NEGLOG10_KS_MATH = r"$-\log_{10}(p_{KS})$"


def fmt_params_math(
    D: int | None = None,
    sigma_eta: float | None = None,
    pi_target: float | None = None,
    *,
    use_commas: bool = True,
) -> str:
    parts: list[str] = []
    if D is not None:
        parts.append(rf"D={int(D)}")
    if sigma_eta is not None:
        parts.append(rf"\sigma_\eta={float(sigma_eta):g}")
    if pi_target is not None:
        parts.append(rf"\pi_{{\mathrm{{target}}}}={float(pi_target):.2f}")
    joiner = r",\ " if use_commas else r"\ "
    return r"$" + joiner.join(parts) + r"$"


def render_na(ax: plt.Axes, n: int, *, fontsize: float = 12) -> None:
    ax.cla()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5,
        0.56,
        "NA",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=fontsize,
    )
    ax.text(
        0.5,
        0.40,
        f"(n={int(n)})",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=fontsize * 0.8,
        color="0.35",
    )


def fmt_param_D(d: int) -> str:
    return fmt_params_math(D=int(d))


def fmt_param_sigma(sigma_eta: float) -> str:
    return fmt_params_math(sigma_eta=float(sigma_eta))


def fmt_param_pi(pi_target: float) -> str:
    return fmt_params_math(pi_target=float(pi_target))


def fmt_title_expA(
    prefix: str,
    D: int | None = None,
    sigma: float | None = None,
    pi: float | None = None,
) -> str:
    return f"{prefix}\n" + fmt_params_math(D=D, sigma_eta=sigma, pi_target=pi)


def annotate_heatmap(ax: plt.Axes, data: np.ndarray, im, fmt: str = "{:.3g}") -> None:
    arr = np.asarray(data, dtype=float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isfinite(v):
                ax.text(
                    j,
                    i,
                    "NA",
                    ha="center",
                    va="center",
                    fontsize=DEFAULT_STYLE.fs_annot - 1,
                    color="#222222",
                )
                continue
            rgba = im.cmap(im.norm(v))
            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=DEFAULT_STYLE.fs_annot - 1,
                color=choose_text_color(rgba),
            )


def _col_title_d(d: int) -> str:
    return fmt_params_math(D=int(d))


def _row_label(sigma_eta: float, prevalence_label: str) -> str:
    return fmt_params_math(
        sigma_eta=float(sigma_eta),
        pi_target=_prevalence_float(prevalence_label),
    )


def _build_grid_axes(
    *,
    n_rows: int,
    n_cols: int,
    figsize: tuple[float, float],
    left: float = 0.08,
    right: float = 0.88,
    bottom: float = 0.1,
    top: float = 0.86,
    wspace: float = 0.20,
    hspace: float = 0.24,
) -> tuple[plt.Figure, np.ndarray]:
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j] = fig.add_subplot(gs[i, j])
    return fig, axes


def plot_p_hist_grid(
    metrics: pd.DataFrame,
    prevalence_labels: list[str],
    donor_grid: list[int],
    sigma_grid: list[float],
    out_png: Path,
) -> None:
    rows = len(prevalence_labels) * len(sigma_grid)
    cols = len(donor_grid)
    fig, axes = _build_grid_axes(
        n_rows=rows,
        n_cols=cols,
        figsize=(max(8.0, 2.0 * cols), max(5.0, 1.3 * rows)),
        left=0.12,
        right=0.90,
        bottom=0.10,
        top=0.86,
        wspace=0.10,
        hspace=0.22,
    )
    axes = np.asarray(axes)

    bins = np.linspace(0.0, 1.0, 21)
    for s_idx, sigma_eta in enumerate(sigma_grid):
        for p_idx, p_label in enumerate(prevalence_labels):
            row = s_idx * len(prevalence_labels) + p_idx
            for d_idx, d in enumerate(donor_grid):
                ax = axes[row, d_idx]
                mask = (
                    (metrics["D"] == d)
                    & (metrics["sigma_eta"] == sigma_eta)
                    & (metrics["prevalence_bin"] == p_label)
                    & (~metrics["underpowered"])
                )
                pvals = pd.to_numeric(
                    metrics.loc[mask, "p_T"], errors="coerce"
                ).to_numpy(dtype=float)
                pvals = pvals[np.isfinite(pvals)]
                if should_plot(pvals.size, N_MIN):
                    ax.hist(
                        pvals,
                        bins=bins,
                        density=True,
                        color="#4C78A8",
                        alpha=DEFAULT_STYLE.alpha_line,
                        edgecolor="white",
                        linewidth=0.25,
                    )
                    ax.axhline(
                        1.0,
                        color="#666666",
                        linestyle="--",
                        linewidth=0.9,
                        alpha=0.9,
                    )
                else:
                    render_na(
                        ax, n=int(pvals.size), fontsize=DEFAULT_STYLE.fs_annot + 1
                    )
                if row == 0:
                    ax.set_title(_col_title_d(d))
                if d_idx == 0:
                    ax.text(
                        -0.34,
                        0.5,
                        _row_label(sigma_eta, p_label),
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=DEFAULT_STYLE.fs_tick,
                    )
                if row != rows - 1:
                    ax.tick_params(labelbottom=False)
                if should_plot(pvals.size, N_MIN):
                    ax.text(
                        0.98,
                        0.90,
                        f"n={pvals.size}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=DEFAULT_STYLE.fs_annot - 1,
                        color="#222222",
                    )
                ax.set_xlim(0.0, 1.0)
    fig.supxlabel(LABEL_PT_MATH, fontsize=DEFAULT_STYLE.fs_label)
    fig.supylabel("density", fontsize=DEFAULT_STYLE.fs_label)
    fig.text(
        0.12,
        0.03,
        "Dashed line: expected U(0,1) density",
        fontsize=DEFAULT_STYLE.fs_annot,
        ha="left",
    )
    safe_suptitle(
        fig,
        "Experiment A null calibration: p-value histograms by prevalence (non-underpowered)",
    )
    finalize_fig(fig, out_png)


def plot_qq_grid(
    metrics: pd.DataFrame,
    prevalence_labels: list[str],
    donor_grid: list[int],
    sigma_grid: list[float],
    out_png: Path,
    q_max: float | None = None,
) -> None:
    rows = len(prevalence_labels) * len(sigma_grid)
    cols = len(donor_grid)
    fig = plt.figure(
        figsize=(max(8.4, 2.35 * cols + 1.1), max(5.5, 1.55 * rows)),
        constrained_layout=False,
    )
    gs = GridSpec(
        rows,
        cols + 1,
        figure=fig,
        width_ratios=[0.55] + [1.0] * cols,
        left=0.06,
        right=0.98,
        top=0.88,
        bottom=0.10,
        wspace=0.25,
        hspace=0.35,
    )
    axes = np.empty((rows, cols), dtype=object)
    for row in range(rows):
        label_ax = fig.add_subplot(gs[row, 0])
        label_ax.axis("off")
        s_idx = row // len(prevalence_labels)
        p_idx = row % len(prevalence_labels)
        label_ax.text(
            0.98,
            0.5,
            _row_label(sigma_grid[s_idx], prevalence_labels[p_idx]),
            transform=label_ax.transAxes,
            ha="right",
            va="center",
            fontsize=DEFAULT_STYLE.fs_tick,
        )
        for d_idx in range(cols):
            axes[row, d_idx] = fig.add_subplot(gs[row, d_idx + 1])

    bound = float(q_max) if q_max is not None else 1.0

    for s_idx, sigma_eta in enumerate(sigma_grid):
        for p_idx, p_label in enumerate(prevalence_labels):
            row = s_idx * len(prevalence_labels) + p_idx
            for d_idx, d in enumerate(donor_grid):
                ax = axes[row, d_idx]
                mask = (
                    (metrics["D"] == d)
                    & (metrics["sigma_eta"] == sigma_eta)
                    & (metrics["prevalence_bin"] == p_label)
                    & (~metrics["underpowered"])
                )
                pvals = pd.to_numeric(
                    metrics.loc[mask, "p_T"], errors="coerce"
                ).to_numpy(dtype=float)
                pvals = pvals[np.isfinite(pvals)]
                is_valid = False
                if should_plot(pvals.size, N_MIN):
                    p_sorted = np.sort(pvals)
                    expected = (
                        np.arange(1, p_sorted.size + 1, dtype=float) - 0.5
                    ) / p_sorted.size
                    tail = expected <= bound
                    ex = expected[tail]
                    ob = p_sorted[tail]
                    if ex.size >= 2:
                        is_valid = True
                        ax.plot(
                            ex,
                            ob,
                            ".",
                            color="#4C78A8",
                            markersize=3.8,
                            alpha=0.95,
                        )
                    else:
                        render_na(
                            ax, n=int(pvals.size), fontsize=DEFAULT_STYLE.fs_annot + 1
                        )
                else:
                    render_na(
                        ax, n=int(pvals.size), fontsize=DEFAULT_STYLE.fs_annot + 1
                    )

                if is_valid:
                    ax.plot(
                        [0.0, bound],
                        [0.0, bound],
                        color="#444444",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.9,
                    )
                    ax.set_xlim(0.0, bound)
                    ax.set_ylim(0.0, bound)
                    ax.set_aspect("equal", adjustable="box")

                if row == 0:
                    ax.set_title(_col_title_d(d))
                if row != rows - 1:
                    ax.tick_params(labelbottom=False)
                if d_idx != 0:
                    ax.tick_params(labelleft=False)
                if is_valid:
                    ax.text(
                        0.98,
                        0.06,
                        f"n={pvals.size}",
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=DEFAULT_STYLE.fs_annot - 1,
                        color="0.45",
                    )
    fig.supxlabel("Expected U(0,1)", fontsize=DEFAULT_STYLE.fs_label)
    fig.text(0.02, 0.50, "Observed p-values", rotation=90, va="center", ha="center")
    title_suffix = "left-tail" if q_max is not None else "full-range"
    safe_suptitle(
        fig, f"Experiment A null calibration: QQ plots by prevalence ({title_suffix})"
    )
    finalize_fig(fig, out_png)


def _matrix_from_table(
    df: pd.DataFrame,
    *,
    prevalence_label: str,
    value_col: str,
    donor_grid: list[int],
    sigma_grid: list[float],
) -> np.ndarray:
    mat = np.full((len(sigma_grid), len(donor_grid)), np.nan, dtype=float)
    for i, sigma in enumerate(sigma_grid):
        for j, d in enumerate(donor_grid):
            mask = (
                (df["prevalence_bin"] == prevalence_label)
                & (df["D"] == d)
                & (df["sigma_eta"] == sigma)
            )
            if mask.any():
                val = df.loc[mask, value_col].iloc[0]
                mat[i, j] = float(val) if pd.notna(val) else np.nan
    return mat


def _plot_heatmap_small_multiples(
    matrices: list[np.ndarray],
    prevalence_labels: list[str],
    donor_grid: list[int],
    sigma_grid: list[float],
    *,
    out_png: Path,
    title: str,
    cbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    n_matrices: list[np.ndarray] | None = None,
    annotate_fmt: str = "{:.3f}",
    n_min: int = N_MIN,
    text_matrices: list[np.ndarray] | None = None,
) -> None:
    n = len(prevalence_labels)
    n_cols = min(3, max(1, n))
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = _build_grid_axes(
        n_rows=n_rows,
        n_cols=n_cols,
        figsize=(max(8.0, 3.0 * n_cols), max(4.8, 2.4 * n_rows + 0.8)),
        left=0.10,
        right=0.84,
        bottom=0.12,
        top=0.84,
        wspace=0.22,
        hspace=0.30,
    )
    axes_arr = np.asarray(axes).ravel()
    im = None

    for idx, (p_label, mat) in enumerate(zip(prevalence_labels, matrices)):
        ax = axes_arr[idx]
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(fmt_params_math(pi_target=_prevalence_float(p_label)))
        ax.set_xticks(np.arange(len(donor_grid)))
        ax.set_xticklabels([str(x) for x in donor_grid])
        ax.set_yticks(np.arange(len(sigma_grid)))
        ax.set_yticklabels([f"{x:g}" for x in sigma_grid])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                n_val = np.nan
                if n_matrices is not None:
                    n_val = float(n_matrices[idx][i, j])
                val = float(mat[i, j]) if np.isfinite(mat[i, j]) else np.nan
                if np.isfinite(n_val) and int(n_val) < int(n_min):
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1.0,
                            1.0,
                            facecolor="#f4f4f4",
                            edgecolor="white",
                            linewidth=1.0,
                            zorder=2,
                        )
                    )
                    ax.text(
                        j,
                        i,
                        f"NA\n(n={int(n_val)})",
                        ha="center",
                        va="center",
                        fontsize=DEFAULT_STYLE.fs_annot - 2,
                        color="#333333",
                        zorder=3,
                    )
                    continue
                if not np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        "NA",
                        ha="center",
                        va="center",
                        fontsize=DEFAULT_STYLE.fs_annot - 1,
                        color="#222222",
                        zorder=3,
                    )
                    continue
                text_val = val
                if text_matrices is not None:
                    tval = text_matrices[idx][i, j]
                    if np.isfinite(tval):
                        text_val = float(tval)
                label = annotate_fmt.format(text_val)
                rgba = im.cmap(im.norm(val))
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=DEFAULT_STYLE.fs_annot - 1,
                    color=choose_text_color(rgba),
                    zorder=3,
                )
        if idx // n_cols != n_rows - 1:
            ax.tick_params(labelbottom=False)
        if idx % n_cols != 0:
            ax.tick_params(labelleft=False)

    for idx in range(n, axes_arr.size):
        axes_arr[idx].axis("off")

    if im is not None:
        cax = fig.add_axes([0.87, 0.16, 0.02, 0.64])
        cbar = fig.colorbar(
            im, cax=cax, shrink=DEFAULT_STYLE.cbar_shrink, pad=DEFAULT_STYLE.cbar_pad
        )
        cbar.set_label(cbar_label)
    fig.supxlabel(r"$D$", fontsize=DEFAULT_STYLE.fs_label)
    fig.supylabel(LABEL_SIGMA_ETA_MATH, fontsize=DEFAULT_STYLE.fs_label)
    safe_suptitle(fig, title)
    finalize_fig(fig, out_png)


def plot_typei_heatmap_p(
    summary: pd.DataFrame,
    prevalence_labels: list[str],
    donor_grid: list[int],
    sigma_grid: list[float],
    out_png: Path,
) -> None:
    mats = [
        _matrix_from_table(
            summary,
            prevalence_label=p_label,
            value_col="typeI_p05",
            donor_grid=donor_grid,
            sigma_grid=sigma_grid,
        )
        for p_label in prevalence_labels
    ]
    n_mats = [
        _matrix_from_table(
            summary,
            prevalence_label=p_label,
            value_col="n_non_underpowered",
            donor_grid=donor_grid,
            sigma_grid=sigma_grid,
        )
        for p_label in prevalence_labels
    ]
    _plot_heatmap_small_multiples(
        mats,
        prevalence_labels,
        donor_grid,
        sigma_grid,
        out_png=out_png,
        title=r"Experiment A null calibration: empirical Type I error at $\alpha=0.05$ ($p_T$)",
        cbar_label="Type I error",
        cmap=DEFAULT_STYLE.cmap_heat,
        vmin=0.0,
        vmax=0.15,
        n_matrices=n_mats,
        n_min=N_MIN,
    )


def plot_underpowered_rate(
    summary: pd.DataFrame,
    prevalence_labels: list[str],
    donor_grid: list[int],
    sigma_grid: list[float],
    out_png: Path,
) -> None:
    x = np.arange(len(prevalence_labels))
    fig, ax = plt.subplots(figsize=DEFAULT_STYLE.fig_wide)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(donor_grid)))
    linestyles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D"]
    all_lines: list[np.ndarray] = []

    for d_idx, d in enumerate(donor_grid):
        for s_idx, sigma in enumerate(sigma_grid):
            y = []
            for p_label in prevalence_labels:
                mask = (
                    (summary["D"] == d)
                    & (summary["sigma_eta"] == sigma)
                    & (summary["prevalence_bin"] == p_label)
                )
                if mask.any():
                    y.append(float(summary.loc[mask, "frac_underpowered"].iloc[0]))
                else:
                    y.append(np.nan)
            y_arr = np.asarray(y, dtype=float)
            all_lines.append(y_arr)
            ax.plot(
                x,
                y_arr,
                color=colors[d_idx % len(colors)],
                linestyle=linestyles[s_idx % len(linestyles)],
                marker=markers[s_idx % len(markers)],
                linewidth=1.6,
                markersize=4.0,
                label=fmt_params_math(D=d, sigma_eta=sigma),
                alpha=DEFAULT_STYLE.alpha_line,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(prevalence_labels)
    ax.set_xlabel(rf"Target prevalence bin ({LABEL_PI_TARGET_MATH})")
    ax.set_ylabel("Fraction underpowered")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Experiment A diagnostics: underpowered fraction by prevalence")
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    if all(np.nanmax(np.nan_to_num(y, nan=0.0)) == 0.0 for y in all_lines):
        ax.text(
            0.5,
            0.93,
            "0 underpowered across bins",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=DEFAULT_STYLE.fs_annot,
        )
    if len(all_lines) > 1:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            ncol=1,
            frameon=False,
            fontsize=DEFAULT_STYLE.fs_legend - 1,
        )
    finalize_fig(fig, out_png)


def plot_ks_uniformity(
    ks_df: pd.DataFrame,
    prevalence_labels: list[str],
    donor_grid: list[int],
    sigma_grid: list[float],
    out_png: Path,
) -> None:
    mats = [
        _matrix_from_table(
            ks_df,
            prevalence_label=p_label,
            value_col="neglog10_ks_pvalue",
            donor_grid=donor_grid,
            sigma_grid=sigma_grid,
        )
        for p_label in prevalence_labels
    ]
    n_mats = [
        _matrix_from_table(
            ks_df,
            prevalence_label=p_label,
            value_col="n_non_underpowered",
            donor_grid=donor_grid,
            sigma_grid=sigma_grid,
        )
        for p_label in prevalence_labels
    ]
    text_mats = [
        _matrix_from_table(
            ks_df,
            prevalence_label=p_label,
            value_col="ks_pvalue",
            donor_grid=donor_grid,
            sigma_grid=sigma_grid,
        )
        for p_label in prevalence_labels
    ]
    flat = np.concatenate([m.ravel() for m in mats])
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        vmax = 1.0
    else:
        vmax = max(1.0, min(float(np.max(finite)), 12.0))
    _plot_heatmap_small_multiples(
        mats,
        prevalence_labels,
        donor_grid,
        sigma_grid,
        out_png=out_png,
        title="Experiment A null calibration: KS diagnostic (-log10 p) vs U(0,1)",
        cbar_label=LABEL_NEGLOG10_KS_MATH,
        cmap=DEFAULT_STYLE.cmap_ks,
        vmin=0.0,
        vmax=vmax,
        n_matrices=n_mats,
        n_min=N_MIN,
        text_matrices=text_mats,
        annotate_fmt="{:.3g}",
    )


def plot_deff_by_prevalence(
    metrics: pd.DataFrame,
    prevalence_labels: list[str],
    donor_grid: list[int],
    sigma_grid: list[float],
    out_png: Path,
) -> None:
    x = np.arange(len(prevalence_labels))
    fig, axes = _build_grid_axes(
        n_rows=1,
        n_cols=len(sigma_grid),
        figsize=(
            max(DEFAULT_STYLE.fig_wide[0], 3.4 * len(sigma_grid)),
            DEFAULT_STYLE.fig_wide[1],
        ),
        left=0.10,
        right=0.86,
        bottom=0.16,
        top=0.82,
        wspace=0.14,
    )
    axes_arr = np.asarray(axes).ravel()
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(donor_grid)))
    all_constant = True

    for s_idx, sigma_eta in enumerate(sigma_grid):
        ax = axes_arr[s_idx]
        for d_idx, d in enumerate(donor_grid):
            means = []
            sds = []
            for p_label in prevalence_labels:
                mask = (
                    (metrics["sigma_eta"] == sigma_eta)
                    & (metrics["D"] == d)
                    & (metrics["prevalence_bin"] == p_label)
                )
                vals = pd.to_numeric(
                    metrics.loc[mask, "D_eff"], errors="coerce"
                ).to_numpy(dtype=float)
                means.append(float(np.nanmean(vals)) if vals.size else np.nan)
                sds.append(float(np.nanstd(vals)) if vals.size else np.nan)
            means_arr = np.asarray(means, dtype=float)
            sds_arr = np.asarray(sds, dtype=float)
            if np.nanmax(means_arr) - np.nanmin(means_arr) > 1e-9:
                all_constant = False
            ax.errorbar(
                x,
                means_arr,
                yerr=sds_arr,
                marker="o",
                linewidth=1.6,
                color=colors[d_idx],
                capsize=2,
                label=fmt_params_math(D=d),
                alpha=DEFAULT_STYLE.alpha_line,
            )

        ax.set_title(fmt_params_math(sigma_eta=sigma_eta))
        ax.set_xticks(x)
        ax.set_xticklabels(prevalence_labels)
        ax.set_xlabel(LABEL_PI_TARGET_MATH)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    axes_arr[0].set_ylabel(rf"{LABEL_DEFF_MATH} (mean ± SD)")
    if len(donor_grid) > 1:
        axes_arr[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    if all_constant:
        fig.text(
            0.5,
            0.06,
            "All curves are constant across prevalence bins.",
            ha="center",
            fontsize=DEFAULT_STYLE.fs_annot,
        )
    safe_suptitle(
        fig,
        "Experiment A diagnostics: donor-effective support by prevalence",
    )
    finalize_fig(fig, out_png)


def plot_n_non_underpowered_heatmap(
    summary: pd.DataFrame,
    prevalence_labels: list[str],
    donor_grid: list[int],
    sigma_grid: list[float],
    out_png: Path,
) -> None:
    mats = [
        _matrix_from_table(
            summary,
            prevalence_label=p_label,
            value_col="n_non_underpowered",
            donor_grid=donor_grid,
            sigma_grid=sigma_grid,
        )
        for p_label in prevalence_labels
    ]
    vmax = max(1.0, float(np.nanmax(np.concatenate([m.ravel() for m in mats]))))
    _plot_heatmap_small_multiples(
        mats,
        prevalence_labels,
        donor_grid,
        sigma_grid,
        out_png=out_png,
        title="Experiment A diagnostics: non-underpowered gene counts",
        cbar_label="Non-underpowered genes (count)",
        cmap=DEFAULT_STYLE.cmap_counts,
        vmin=0.0,
        vmax=vmax,
        annotate_fmt="{:.0f}",
    )


def plot_testmode_p_hist(
    metrics: pd.DataFrame,
    prevalence_labels: list[str],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(prevalence_labels),
        figsize=(3.2 * len(prevalence_labels), 3.0),
        sharey=True,
    )
    axes_arr = np.atleast_1d(axes)
    bins = np.linspace(0.0, 1.0, 16)
    for i, prev_lbl in enumerate(prevalence_labels):
        ax = axes_arr[i]
        sub = metrics.loc[
            (metrics["prevalence_bin"] == prev_lbl) & (~metrics["underpowered"])
        ]
        pvals = pd.to_numeric(sub["p_T"], errors="coerce").to_numpy(dtype=float)
        pvals = pvals[np.isfinite(pvals)]
        if pvals.size > 0:
            ax.hist(
                pvals,
                bins=bins,
                density=True,
                color="#4C78A8",
                alpha=0.88,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.axhline(1.0, linestyle="--", color="#444", linewidth=0.8)
        ax.set_title(f"pi={prev_lbl}\nn={pvals.size}")
        ax.set_xlabel("p_T")
        if i == 0:
            ax.set_ylabel("density")
    fig.suptitle(
        "Experiment A test_mode: p-value histograms (non-underpowered)", y=0.995
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    finalize_fig(fig, out_png)


def plot_testmode_qq(
    metrics: pd.DataFrame,
    prevalence_labels: list[str],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(prevalence_labels),
        figsize=(3.2 * len(prevalence_labels), 3.0),
        sharex=True,
        sharey=True,
    )
    axes_arr = np.atleast_1d(axes)
    for i, prev_lbl in enumerate(prevalence_labels):
        ax = axes_arr[i]
        sub = metrics.loc[
            (metrics["prevalence_bin"] == prev_lbl) & (~metrics["underpowered"])
        ]
        pvals = pd.to_numeric(sub["p_T"], errors="coerce").to_numpy(dtype=float)
        pvals = pvals[np.isfinite(pvals)]
        if pvals.size >= 3:
            ps = np.sort(pvals)
            exp = (np.arange(1, ps.size + 1, dtype=float) - 0.5) / ps.size
            ax.plot(exp, ps, ".", color="#1f77b4", markersize=2.0, alpha=0.9)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#444", linewidth=0.8)
        ax.set_title(f"pi={prev_lbl}\nn={pvals.size}")
        ax.set_xlabel("Expected U(0,1)")
        if i == 0:
            ax.set_ylabel("Observed p_T")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    fig.suptitle("Experiment A test_mode: QQ plots (non-underpowered)", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    finalize_fig(fig, out_png)


def plot_testmode_typei_bar(
    summary: pd.DataFrame,
    prevalence_labels: list[str],
    out_png: Path,
) -> None:
    sub = summary.copy()
    order = {lbl: i for i, lbl in enumerate(prevalence_labels)}
    sub["order"] = sub["prevalence_bin"].map(order).astype(float)
    sub = sub.sort_values("order")
    y = pd.to_numeric(sub["typeI_p05"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(sub["typeI_p05_ci_low"], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(sub["typeI_p05_ci_high"], errors="coerce").to_numpy(dtype=float)
    x = np.arange(sub.shape[0], dtype=float)
    yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(x, y, color="#59A14F", alpha=0.9, edgecolor="white", linewidth=0.6)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#333", elinewidth=1.0, capsize=3)
    ax.axhline(
        0.05, linestyle="--", color="#D62728", linewidth=1.0, label="target alpha=0.05"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sub["prevalence_bin"].astype(str).tolist())
    ax.set_xlabel("prevalence_bin")
    ax.set_ylabel("Type I error (p<=0.05)")
    ax.set_ylim(0.0, max(0.15, float(np.nanmax(np.nan_to_num(y, nan=0.0))) + 0.03))
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Experiment A test_mode: Type I by prevalence (non-underpowered)")
    fig.tight_layout()
    finalize_fig(fig, out_png)


def _load_biorsp_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("biorsp")
    except PackageNotFoundError:
        pass
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        import biorsp  # type: ignore

        return str(getattr(biorsp, "__version__", "unknown"))
    except (ImportError, ModuleNotFoundError, AttributeError):
        return "unknown"


def build_grid(
    donor_grid: list[int], sigma_eta_grid: list[float]
) -> list[tuple[int, float]]:
    """Build condition grid for ExpA."""
    return [(int(d), float(sigma)) for d in donor_grid for sigma in sigma_eta_grid]


def _run_condition_item(item: dict[str, Any]) -> dict[str, Any]:
    """Compute all genes for one (D, sigma_eta) condition."""
    d = int(item["D"])
    sigma_eta = float(item["sigma_eta"])
    if "seed_run" in item:
        seed_run = int(item["seed_run"])
    elif "seed" in item:
        seed_run = int(item["seed"])
    else:
        raise KeyError("seed_run")
    run_id = str(item["run_id"])
    n_cells = int(item["N"])
    g_total = int(item["G"])
    n_bins = int(item["n_bins"])
    n_perm = int(item["n_perm"])
    prevalence_bins = [float(x) for x in item["prevalence_bins"]]
    progress_every = int(item["progress_every"])
    mu0 = float(item["mu0"])
    sigma_l = float(item["sigma_l"])
    qc_corr_threshold = float(item["qc_corr_threshold"])
    master_seed = int(item["master_seed"])
    gate_cfg = GateConfig(**dict(item["gate_cfg"]))
    edge_cfg = EdgeRuleConfig(**dict(item["edge_cfg"]))
    progress_enabled = bool(item.get("progress", True))

    context, qc_corr = simulate_run_context(
        n_cells=n_cells,
        n_bins=n_bins,
        n_donors=d,
        sigma_eta=sigma_eta,
        seed_run=seed_run,
        mu0=mu0,
        sigma_l=sigma_l,
        qc_corr_threshold=qc_corr_threshold,
        run_id=run_id,
    )
    qc_row = {
        "run_id": run_id,
        "seed_run": seed_run,
        "D": d,
        "sigma_eta": sigma_eta,
        **qc_corr,
    }

    rng_schedule = rng_from_seed(seed_run)
    pi_targets, pi_labels = _gene_prevalence_schedule(
        int(g_total),
        prevalence_bins,
        rng=rng_schedule,
    )

    metrics_rows: list[dict[str, Any]] = []
    prevalence_dev_rows: list[dict[str, Any]] = []
    run_start = time.time()

    for g in range(int(g_total)):
        seed_gene = seed_run + g
        pi_target = float(pi_targets[g])
        prevalence_bin = float(pi_labels[g])

        metric_row, _ = run_single_null_gene(
            context=context,
            pi_target=pi_target,
            prevalence_bin=prevalence_bin,
            seed_gene=int(seed_gene),
            n_bins=int(n_bins),
            n_perm=int(n_perm),
            gate_cfg=gate_cfg,
            edge_cfg=edge_cfg,
            run_id=run_id,
            seed_run=seed_run,
            master_seed=master_seed,
            include_null_t_values=False,
        )
        metric_row.update(
            {
                "N": int(n_cells),
                "D": int(d),
                "sigma_eta": float(sigma_eta),
            }
        )
        metrics_rows.append(metric_row)
        prevalence_dev_rows.append(
            {
                "run_id": run_id,
                "D": int(d),
                "sigma_eta": float(sigma_eta),
                "prevalence_bin": str(metric_row["prevalence_bin"]),
                "pi_target": pi_target,
                "prev_obs": float(metric_row["prev_obs"]),
                "abs_dev": abs(float(metric_row["prev_obs"]) - pi_target),
            }
        )
        if progress_enabled and (
            (g + 1) % int(progress_every) == 0 or (g + 1) == int(g_total)
        ):
            elapsed = time.time() - run_start
            rate = (g + 1) / elapsed if elapsed > 0 else float("nan")
            print(
                f"  {run_id}: gene {g + 1}/{g_total} "
                f"({rate:.2f} genes/s, elapsed {elapsed/60.0:.1f} min)",
                flush=True,
            )

    pvals_run = np.asarray([float(r["p_T"]) for r in metrics_rows], dtype=float)
    qvals_run = bh_fdr(pvals_run)
    for g in range(int(g_total)):
        metrics_rows[g]["q_T"] = float(qvals_run[g])

    prev_dev_run = pd.DataFrame(prevalence_dev_rows)
    max_abs_dev = (
        float(prev_dev_run["abs_dev"].max()) if not prev_dev_run.empty else float("nan")
    )
    p95_abs_dev = (
        float(prev_dev_run["abs_dev"].quantile(0.95))
        if not prev_dev_run.empty
        else float("nan")
    )
    if progress_enabled:
        print(
            f"  {run_id}: prevalence deviation max={max_abs_dev:.4f}, p95={p95_abs_dev:.4f}",
            flush=True,
        )

    return {
        "seed": int(seed_run),
        "run_id": run_id,
        "D": int(d),
        "sigma_eta": float(sigma_eta),
        "qc_row": qc_row,
        "metrics_rows": metrics_rows,
        "prevalence_dev_rows": prevalence_dev_rows,
    }


def summarize(
    metrics: pd.DataFrame,
    *,
    n_perm: int,
    m_full_tests: int,
    bh_validation_mode: str,
    bh_panel_size: int,
    bh_panel_strategy: str,
    bh_panel_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return summary tables for ExpA."""
    summary = summarize_by_bin(metrics, n_perm=n_perm, m_full_tests=m_full_tests)
    ks_df = summarize_ks(metrics)
    panel_df = compute_panel_bh_validation(
        metrics,
        bh_validation_mode=bh_validation_mode,
        bh_panel_size=bh_panel_size,
        bh_panel_strategy=bh_panel_strategy,
        bh_panel_seed=bh_panel_seed,
        n_perm=n_perm,
    )
    if not panel_df.empty:
        summary = summary.merge(
            panel_df,
            on=["D", "sigma_eta", "prevalence_bin"],
            how="left",
            validate="one_to_one",
        )
    return summary, ks_df, panel_df


def _assert_plot_outputs(plots_dir: Path, names: list[str]) -> None:
    missing_or_empty: list[str] = []
    for name in names:
        p = plots_dir / name
        if (not p.exists()) or (p.stat().st_size <= 0):
            missing_or_empty.append(str(p))
    if missing_or_empty:
        raise RuntimeError(
            f"Plot QA failed; missing or empty outputs: {missing_or_empty}"
        )


def make_plots(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    ks_df: pd.DataFrame,
    *,
    plots_dir: Path,
    prevalence_labels: list[str],
    donor_grid: list[int],
    sigma_eta_grid: list[float],
    test_mode: bool,
) -> None:
    """Generate figures for ExpA."""
    apply_style()
    if test_mode:
        plot_testmode_p_hist(metrics, prevalence_labels, plots_dir / "p_hist.png")
        plot_testmode_qq(metrics, prevalence_labels, plots_dir / "qq.png")
        summary_tm = summary.loc[
            (summary["D"] == int(donor_grid[0]))
            & (summary["sigma_eta"] == float(sigma_eta_grid[0]))
        ].copy()
        plot_testmode_typei_bar(
            summary_tm, prevalence_labels, plots_dir / "typeI_bar.png"
        )
        _assert_plot_outputs(plots_dir, ["p_hist.png", "qq.png", "typeI_bar.png"])
        return

    plot_p_hist_grid(
        metrics,
        prevalence_labels,
        donor_grid,
        sigma_eta_grid,
        plots_dir / "p_hist_grid.png",
    )
    plot_qq_grid(
        metrics,
        prevalence_labels,
        donor_grid,
        sigma_eta_grid,
        plots_dir / "qq_grid.png",
    )
    plot_qq_grid(
        metrics,
        prevalence_labels,
        donor_grid,
        sigma_eta_grid,
        plots_dir / "qq_lefttail_grid.png",
        q_max=0.1,
    )
    plot_typei_heatmap_p(
        summary,
        prevalence_labels,
        donor_grid,
        sigma_eta_grid,
        plots_dir / "typeI_heatmap_p.png",
    )
    plot_underpowered_rate(
        summary,
        prevalence_labels,
        donor_grid,
        sigma_eta_grid,
        plots_dir / "underpowered_rate.png",
    )
    plot_ks_uniformity(
        ks_df,
        prevalence_labels,
        donor_grid,
        sigma_eta_grid,
        plots_dir / "p_uniformity_ks.png",
    )
    plot_deff_by_prevalence(
        metrics,
        prevalence_labels,
        donor_grid,
        sigma_eta_grid,
        plots_dir / "deff_by_prevalence.png",
    )
    plot_n_non_underpowered_heatmap(
        summary,
        prevalence_labels,
        donor_grid,
        sigma_eta_grid,
        plots_dir / "n_non_underpowered_heatmap.png",
    )
    _assert_plot_outputs(
        plots_dir,
        [
            "deff_by_prevalence.png",
            "n_non_underpowered_heatmap.png",
            "p_hist_grid.png",
            "p_uniformity_ks.png",
            "qq_grid.png",
            "qq_lefttail_grid.png",
            "typeI_heatmap_p.png",
            "underpowered_rate.png",
        ],
    )


def plot_representative_embedding_and_profile(
    *,
    plots_dir: Path,
    n_cells: int,
    n_bins: int,
    donor_count: int,
    sigma_eta: float,
    master_seed: int,
    pi_target: float,
    mu0: float,
    sigma_l: float,
) -> None:
    seed_run = int(master_seed) + int(
        _stable_hash_run(int(donor_count), float(sigma_eta))
    )
    context, _ = simulate_run_context(
        n_cells=int(n_cells),
        n_bins=int(n_bins),
        n_donors=int(donor_count),
        sigma_eta=float(sigma_eta),
        seed_run=int(seed_run),
        mu0=float(mu0),
        sigma_l=float(sigma_l),
        qc_corr_threshold=float("inf"),
        run_id="representative",
    )
    seed_gene = int(seed_run)
    rng_gene = rng_from_seed(seed_gene)
    eta_cell = donor_effect_vector(context.donor_ids, context.eta_d)
    f = simulate_null_gene(
        pi_target=float(pi_target), donor_eta_per_cell=eta_cell, rng=rng_gene
    )

    fig1, ax1 = plt.subplots(figsize=DEFAULT_STYLE.fig_small, constrained_layout=False)
    plot_embedding_with_foreground(
        context.X_sim,
        f,
        ax=ax1,
        title="Representative embedding (ExpA)\n"
        + fmt_params_math(
            D=int(donor_count),
            sigma_eta=float(sigma_eta),
            pi_target=float(pi_target),
        ),
        s=4.5,
        alpha_bg=DEFAULT_STYLE.alpha_bg,
        alpha_fg=DEFAULT_STYLE.alpha_fg,
    )
    embed_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor="#6f6f6f",
            markeredgecolor="none",
            alpha=0.85,
            label="Background",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor="#d62728",
            markeredgecolor="none",
            alpha=0.95,
            label="Foreground",
        ),
    ]
    ax1.legend(
        handles=embed_handles,
        loc="upper right",
        frameon=False,
        fontsize=DEFAULT_STYLE.fs_legend,
    )
    finalize_fig(fig1, plots_dir / "embedding_example.png")

    E_obs, _, _ = compute_rsp_profile_from_boolean(
        f,
        context.angles,
        int(n_bins),
        bin_id=context.bin_id,
        bin_counts_total=context.bin_counts_total,
    )
    theta_centers = np.linspace(0.0, 2.0 * np.pi, int(n_bins), endpoint=False)
    fig2, ax2 = plt.subplots(
        figsize=DEFAULT_STYLE.fig_small,
        subplot_kw={"projection": "polar"},
        constrained_layout=False,
    )
    plot_rsp_polar(
        theta_centers,
        np.asarray(E_obs, dtype=float),
        ax=ax2,
        title="Representative RSP profile (ExpA)\n"
        + fmt_params_math(
            D=int(donor_count),
            sigma_eta=float(sigma_eta),
            pi_target=float(pi_target),
        ),
        linewidth=2.0,
    )
    max_rsp = (
        float(np.nanmax(np.asarray(E_obs, dtype=float))) if np.size(E_obs) else 1.0
    )
    if not np.isfinite(max_rsp) or max_rsp <= 0.0:
        max_rsp = 1.0
    rticks = np.linspace(0.0, max_rsp, 4)
    ax2.set_rticks(rticks)
    ax2.tick_params(axis="y", labelsize=DEFAULT_STYLE.fs_tick - 1)
    finalize_fig(fig2, plots_dir / "polar_rsp_example.png")


def write_report_markdown(outdir: Path, results_dir: Path) -> Path:
    """Write REPORT.md from current CSV artifacts."""
    return write_report_expA(
        outdir=outdir,
        metrics_path=results_dir / "metrics_long.csv",
        summary_path=results_dir / "summary_by_bin.csv",
        ks_path=results_dir / "ks_uniformity.csv",
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
        g_total = 150
        n_bins = 36
        n_perm = int(test_cfg.n_perm)
        donor_grid = [5]
        sigma_eta_grid = [0.4]
        prevalence_bins = [0.05, 0.2, 0.9]
        progress_every = 25
    else:
        n_cells = int(args.N)
        g_total = int(args.G)
        n_bins = int(args.bins)
        n_perm = int(args.n_perm)
        donor_grid = _parse_int_list(args.donor_grid)
        sigma_eta_grid = _parse_float_list(args.sigma_eta_grid)
        prevalence_bins = _parse_float_list(args.prevalence_bins)
        progress_every = int(args.progress_every)
    prevalence_labels = [_format_prevalence_label(x) for x in prevalence_bins]
    min_perm_eff = min(int(args.min_perm), int(n_perm))
    bh_panel_seed_eff = (
        int(args.master_seed) if args.bh_panel_seed is None else int(args.bh_panel_seed)
    )
    full_bh = bh_feasibility_metrics(m_full=int(g_total), n_perm=int(n_perm))
    gate_cfg = GateConfig(
        p_min=float(args.p_min),
        min_fg_total=int(args.min_fg_total),
        min_fg_per_donor=int(args.min_fg_per_donor),
        min_bg_per_donor=int(args.min_bg_per_donor),
        d_eff_min=int(args.d_eff_min),
        min_perm=int(min_perm_eff),
    )
    edge_cfg = EdgeRuleConfig(
        threshold=float(args.fg_edge_threshold),
        strategy=str(args.fg_edge_strategy),
    )

    config = {
        "experiment": "Simulation Experiment A: Null calibration across prevalence x donor effects",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit_short": git_commit_hash(cwd=REPO_ROOT),
        "biorsp_version": _load_biorsp_version(),
        "seed_derivation": (
            "seed_run = master_seed + stable_hash(D,sigma_eta) % 1e6; "
            "seed_gene = seed_run + g"
        ),
        "test_mode": bool(args.test_mode),
        "master_seed": int(args.master_seed),
        "N": int(n_cells),
        "G": int(g_total),
        "n_bins": int(n_bins),
        "n_perm": int(n_perm),
        "donor_grid": donor_grid,
        "sigma_eta_grid": sigma_eta_grid,
        "prevalence_bins": prevalence_bins,
        "qc": {
            "mu0": float(args.mu0),
            "sigma_L": float(args.sigma_L),
            "pct_mt_base_mean": 0.07,
            "pct_mt_shift_range": [-0.02, 0.02],
            "qc_corr_threshold_warn": float(args.qc_corr_threshold),
        },
        "underpowered_thresholds": {
            "p_min": float(args.p_min),
            "min_fg_total": int(args.min_fg_total),
            "min_fg_per_donor": int(args.min_fg_per_donor),
            "min_bg_per_donor": int(args.min_bg_per_donor),
            "d_eff_min": int(args.d_eff_min),
            "min_perm": int(min_perm_eff),
        },
        "underpowered_rule": (
            "prev<p_min OR n_fg_total<min_fg_total OR "
            "D_eff<d_eff_min OR n_perm<min_perm"
        ),
        "foreground_edge_rule": {
            "strategy": str(edge_cfg.strategy),
            "threshold": float(edge_cfg.threshold),
            "trigger": "prevalence_bin>=threshold OR fg_fraction>=threshold",
            "description": "If triggered and strategy=complement, test on complement foreground.",
        },
        "multiple_testing_validation": {
            "mode": str(args.bh_validation_mode),
            "bh_panel_size": int(args.bh_panel_size),
            "bh_panel_strategy": str(args.bh_panel_strategy),
            "bh_panel_seed": int(bh_panel_seed_eff),
            "m_full_tests": int(g_total),
            "min_attainable_p": float(full_bh["min_attainable_p"]),
            "bh_min_rejectable_p_q05": float(full_bh["bh_min_rejectable_p_q05"]),
            "bh_min_rejectable_p_q10": float(full_bh["bh_min_rejectable_p_q10"]),
            "bh_feasible_q05": bool(full_bh["bh_feasible_q05"]),
            "bh_feasible_q10": bool(full_bh["bh_feasible_q10"]),
        },
    }
    write_config(outdir, config)
    if not bool(full_bh["bh_feasible_q05"]):
        print(
            (
                "WARNING: Full-run BH at q=0.05 is infeasible under discrete p-value floor: "
                f"q/m={float(full_bh['bh_min_rejectable_p_q05']):.6g}, "
                f"min_attainable_p={float(full_bh['min_attainable_p']):.6g}. "
                "Full-BH metric will be suppressed."
            ),
            flush=True,
        )

    metrics_rows: list[dict[str, float | int | str | bool]] = []
    qc_rows: list[dict[str, float | int | str]] = []
    prevalence_dev_rows: list[dict[str, float | int | str]] = []

    total_start = time.time()
    condition_grid = build_grid(donor_grid, sigma_eta_grid)
    run_count = len(condition_grid)
    condition_items: list[dict[str, Any]] = []
    for run_idx, (d, sigma_eta) in enumerate(condition_grid, start=1):
        run_hash = _stable_hash_run(d, sigma_eta)
        seed_run = int(args.master_seed) + int(run_hash)
        run_id = f"D{d}_sigma{sigma_eta:g}"
        print(
            f"[{run_idx}/{run_count}] run_id={run_id} seed_run={seed_run} "
            f"(N={n_cells}, G={g_total}, n_perm={n_perm})",
            flush=True,
        )
        condition_items.append(
            {
                "seed_run": int(seed_run),
                "seed": int(seed_run),
                "run_id": run_id,
                "D": int(d),
                "sigma_eta": float(sigma_eta),
                "N": int(n_cells),
                "G": int(g_total),
                "n_bins": int(n_bins),
                "n_perm": int(n_perm),
                "prevalence_bins": [float(x) for x in prevalence_bins],
                "progress_every": int(progress_every),
                "mu0": float(args.mu0),
                "sigma_l": float(args.sigma_L),
                "qc_corr_threshold": float(args.qc_corr_threshold),
                "master_seed": int(args.master_seed),
                "gate_cfg": {
                    "p_min": float(gate_cfg.p_min),
                    "min_fg_total": int(gate_cfg.min_fg_total),
                    "min_fg_per_donor": int(gate_cfg.min_fg_per_donor),
                    "min_bg_per_donor": int(gate_cfg.min_bg_per_donor),
                    "d_eff_min": int(gate_cfg.d_eff_min),
                    "min_perm": int(gate_cfg.min_perm),
                },
                "edge_cfg": {
                    "threshold": float(edge_cfg.threshold),
                    "strategy": str(edge_cfg.strategy),
                },
                "progress": bool(getattr(args, "progress", True)),
            }
        )

    if int(args.n_jobs) > 1 and len(condition_items) > 1:
        condition_results = parallel_map(
            _run_condition_item,
            condition_items,
            n_jobs=int(args.n_jobs),
            backend=str(args.backend),
            chunk_size=max(1, int(args.chunk_size)),
            progress=bool(getattr(args, "progress", True)),
        )
    else:
        condition_results = [_run_condition_item(item) for item in condition_items]

    for out in condition_results:
        metrics_rows.extend(list(out["metrics_rows"]))
        qc_rows.append(dict(out["qc_row"]))
        prevalence_dev_rows.extend(list(out["prevalence_dev_rows"]))

    metrics = pd.DataFrame(metrics_rows)
    min_p_observed = validate_pvalue_floor(metrics, n_perm=int(n_perm), tol=P_FLOOR_TOL)
    summary, ks_df, panel_df = summarize(
        metrics,
        n_perm=int(n_perm),
        m_full_tests=int(g_total),
        bh_validation_mode=str(args.bh_validation_mode),
        bh_panel_size=int(args.bh_panel_size),
        bh_panel_strategy=str(args.bh_panel_strategy),
        bh_panel_seed=int(bh_panel_seed_eff),
    )
    qc_df = pd.DataFrame(qc_rows)
    prev_dev_df = pd.DataFrame(prevalence_dev_rows)

    atomic_write_csv(results_dir / "metrics_long.csv", metrics)
    atomic_write_csv(results_dir / "summary_by_bin.csv", summary)
    atomic_write_csv(results_dir / "summary.csv", summary)
    atomic_write_csv(results_dir / "ks_uniformity.csv", ks_df)
    atomic_write_csv(results_dir / "bh_panel_validation.csv", panel_df)
    atomic_write_csv(results_dir / "qc_correlation_checks.csv", qc_df)
    atomic_write_csv(results_dir / "prevalence_deviation.csv", prev_dev_df)

    make_plots(
        metrics,
        summary,
        ks_df,
        plots_dir=plots_dir,
        prevalence_labels=prevalence_labels,
        donor_grid=donor_grid,
        sigma_eta_grid=sigma_eta_grid,
        test_mode=bool(args.test_mode),
    )
    rep_pi = (
        0.2
        if any(abs(float(p) - 0.2) < 1e-9 for p in prevalence_bins)
        else float(prevalence_bins[0])
    )
    plot_representative_embedding_and_profile(
        plots_dir=plots_dir,
        n_cells=int(n_cells),
        n_bins=int(n_bins),
        donor_count=int(donor_grid[0]),
        sigma_eta=float(sigma_eta_grid[0]),
        master_seed=int(args.master_seed),
        pi_target=float(rep_pi),
        mu0=float(args.mu0),
        sigma_l=float(args.sigma_L),
    )
    _assert_plot_outputs(
        plots_dir,
        [
            "embedding_example.png",
            "polar_rsp_example.png",
        ],
    )

    elapsed_total = time.time() - total_start
    n_total = metrics.shape[0]
    n_non_under = int((~metrics["underpowered"]).sum())
    global_typei_p = float(
        (metrics.loc[~metrics["underpowered"], "p_T"] <= 0.05).mean()
    )
    global_typei_q = (
        float((metrics.loc[~metrics["underpowered"], "q_T"] <= 0.05).mean())
        if bool(full_bh["bh_feasible_q05"])
        else float("nan")
    )

    if bool(args.test_mode):
        ks_mask = (
            (ks_df["D"] == int(donor_grid[0]))
            & (ks_df["sigma_eta"] == float(sigma_eta_grid[0]))
            & (ks_df["prevalence_bin"] == _format_prevalence_label(0.2))
        )
        if ks_mask.any():
            ks_val = float(ks_df.loc[ks_mask, "ks_pvalue"].iloc[0])
            if np.isfinite(ks_val) and ks_val < 1e-6:
                raise RuntimeError(
                    f"Critical validation failed in test_mode: KS p-value too small ({ks_val:.3g})."
                )
        if np.isfinite(global_typei_p) and global_typei_p > 0.15:
            raise RuntimeError(
                f"Critical validation failed in test_mode: global Type I {global_typei_p:.3f} > 0.15."
            )
        required = [
            results_dir / "metrics_long.csv",
            results_dir / "summary_by_bin.csv",
            results_dir / "bh_panel_validation.csv",
            plots_dir / "p_hist.png",
            plots_dir / "qq.png",
            plots_dir / "typeI_bar.png",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"Missing required test_mode outputs: {missing}")
    else:
        target_cells = [
            (0.9, 2, 0.0),
            (0.6, 15, 0.4),
        ]
        target_diag_df = build_target_cell_diagnostics(
            metrics=metrics,
            ks_df=ks_df,
            targets=target_cells,
        )
        if not target_diag_df.empty:
            atomic_write_csv(
                results_dir / "target_cell_diagnostics.csv", target_diag_df
            )
            (results_dir / "target_cell_diagnostics.json").write_text(
                json.dumps(target_diag_df.to_dict(orient="records"), indent=2),
                encoding="utf-8",
            )
            for row in target_diag_df.to_dict(orient="records"):
                print(
                    (
                        f"Target diagnostic: {row['cell_key']} "
                        f"n_non_underpowered={int(row['n_non_underpowered'])} "
                        f"ks_pvalue={float(row['ks_pvalue']):.4g} "
                        f"mean_p={float(row['mean_p']):.4f} "
                        f"typeI_p05={float(row['typeI_p05']):.4f}"
                    ),
                    flush=True,
                )
    report_path = write_report_markdown(outdir, results_dir)
    if np.isfinite(global_typei_q):
        completion_msg = (
            f"Completed Experiment A in {elapsed_total/60.0:.2f} min. "
            f"genes={n_total}, non_underpowered={n_non_under}, "
            f"min_p={min_p_observed:.6g}, "
            f"global_typeI_p05={global_typei_p:.4f}, "
            f"global_typeI_q05={global_typei_q:.4f}"
        )
    else:
        completion_msg = (
            f"Completed Experiment A in {elapsed_total/60.0:.2f} min. "
            f"genes={n_total}, non_underpowered={n_non_under}, "
            f"min_p={min_p_observed:.6g}, "
            f"global_typeI_p05={global_typei_p:.4f}, "
            "global_typeI_q05=SUPPRESSED (BH infeasible)"
        )
    print(
        completion_msg,
        flush=True,
    )
    print(f"Report written to: {report_path}", flush=True)
    print(f"Outputs written to: {outdir}", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Simulation Experiment A: null calibration across prevalence x donor effects."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/simulations/expA_null_calibration",
        help="Output directory containing config, results/, and plots/.",
    )
    parser.add_argument("--N", type=int, default=20000, help="Number of cells.")
    parser.add_argument(
        "--G",
        type=int,
        default=2000,
        help="Number of null genes per run (set 1000 if runtime is too slow).",
    )
    parser.add_argument("--bins", type=int, default=36, help="Number of angular bins.")
    parser.add_argument(
        "--n_perm",
        type=int,
        default=300,
        help="Number of donor-stratified permutations per gene.",
    )
    parser.add_argument(
        "--master_seed",
        type=int,
        default=123,
        help="Master seed for deterministic run/gene seed derivation.",
    )
    parser.add_argument(
        "--donor_grid",
        type=str,
        default="2,5,10,15",
        help="Comma-separated donor-count grid.",
    )
    parser.add_argument(
        "--sigma_eta_grid",
        type=str,
        default="0.0,0.4,0.8",
        help="Comma-separated donor random-effect std-dev grid (logit scale).",
    )
    parser.add_argument(
        "--prevalence_bins",
        type=str,
        default=",".join(str(x) for x in DEFAULT_PREVALENCE_BINS),
        help="Comma-separated target prevalence bins.",
    )
    parser.add_argument(
        "--mu0", type=float, default=10.0, help="Baseline donor log-library mean."
    )
    parser.add_argument(
        "--sigma_L",
        type=float,
        default=0.5,
        help="Within-donor lognormal sigma for library size.",
    )
    parser.add_argument(
        "--qc_corr_threshold",
        type=float,
        default=0.03,
        help="Warn if abs(QC-coordinate correlation) exceeds this threshold.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print progress every N genes.",
    )
    parser.add_argument(
        "--p_min",
        type=float,
        default=DEFAULT_P_MIN,
        help="Underpowered prevalence threshold.",
    )
    parser.add_argument(
        "--min_fg_total",
        type=int,
        default=DEFAULT_MIN_FG_TOTAL,
        help="Underpowered minimum total foreground cells.",
    )
    parser.add_argument(
        "--min_fg_per_donor",
        type=int,
        default=DEFAULT_MIN_FG_PER_DONOR,
        help="Donor-effectiveness minimum foreground per donor.",
    )
    parser.add_argument(
        "--min_bg_per_donor",
        type=int,
        default=DEFAULT_MIN_BG_PER_DONOR,
        help="Donor-effectiveness minimum background per donor.",
    )
    parser.add_argument(
        "--d_eff_min",
        type=int,
        default=DEFAULT_D_EFF_MIN,
        help="Minimum number of informative donors (D_eff).",
    )
    parser.add_argument(
        "--min_perm",
        type=int,
        default=DEFAULT_MIN_PERM,
        help="Underpowered minimum permutations threshold.",
    )
    parser.add_argument(
        "--fg_edge_threshold",
        type=float,
        default=0.8,
        help="High-prevalence edge trigger threshold for prevalence_bin or fg_fraction.",
    )
    parser.add_argument(
        "--fg_edge_strategy",
        type=str,
        default="complement",
        choices=["none", "complement"],
        help="Foreground edge strategy when trigger is active.",
    )
    parser.add_argument(
        "--bh_validation_mode",
        type=str,
        default=DEFAULT_BH_VALIDATION_MODE,
        choices=["panel_bh", "uniformity_only"],
        help="Multiple-testing validation mode for Experiment A reporting.",
    )
    parser.add_argument(
        "--bh_panel_size",
        type=int,
        default=DEFAULT_BH_PANEL_SIZE,
        help="Panel size used for panel-BH validation.",
    )
    parser.add_argument(
        "--bh_panel_strategy",
        type=str,
        default=DEFAULT_BH_PANEL_STRATEGY,
        choices=["prevalence_stratified_fixed", "random_fixed"],
        help="Sampling strategy for panel-BH validation.",
    )
    parser.add_argument(
        "--bh_panel_seed",
        type=int,
        default=None,
        help="Optional seed override for panel-BH sampling (defaults to master_seed).",
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
    if args.N <= 0 or args.G <= 0 or args.bins <= 0 or args.n_perm <= 0:
        raise ValueError("N, G, bins, and n_perm must all be positive.")
    if args.min_fg_total < 0 or args.min_fg_per_donor < 0 or args.min_bg_per_donor < 0:
        raise ValueError("Underpowered count thresholds must be non-negative.")
    if args.d_eff_min < 1 or args.min_perm < 1:
        raise ValueError("d_eff_min and min_perm must be positive.")
    if args.fg_edge_threshold <= 0.0 or args.fg_edge_threshold >= 1.0:
        raise ValueError("fg_edge_threshold must be in (0, 1).")
    if args.bh_panel_size <= 0:
        raise ValueError("bh_panel_size must be positive.")
    run_ctx = prepare_legacy_run(args, "expA_null_calibration", __file__)
    run_experiment(args)
    finalize_legacy_run(run_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
