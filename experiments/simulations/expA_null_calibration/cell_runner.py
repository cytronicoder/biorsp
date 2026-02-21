"""Shared deterministic cell-level replay utilities for Experiment A."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kstest

from biorsp.permutation import perm_null_T
from biorsp.power import evaluate_underpowered
from biorsp.scoring import bh_fdr, robust_z
from experiments.simulations._shared.donors import (
    assign_donors,
    donor_effect_vector,
    sample_donor_effects,
)
from experiments.simulations._shared.geometry import sample_disk_gaussian
from experiments.simulations._shared.models import simulate_null_gene
from experiments.simulations._shared.plots import wilson_ci
from experiments.simulations._shared.seeding import rng_from_seed, stable_seed


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


@dataclass(frozen=True)
class GateConfig:
    p_min: float = 0.005
    min_fg_total: int = 50
    min_fg_per_donor: int = 10
    min_bg_per_donor: int = 10
    d_eff_min: int = 2
    min_perm: int = 200


@dataclass(frozen=True)
class EdgeRuleConfig:
    threshold: float = 0.8
    strategy: str = "complement"


def format_prevalence_label(p: float) -> str:
    return f"{float(p):.3f}".rstrip("0").rstrip(".")


def stable_hash_run(donor_count: int, sigma_eta: float) -> int:
    """Deterministic hash in [0, 1e6) for run seed derivation."""
    return int(
        stable_seed(0, "expA_run", int(donor_count), float(sigma_eta)) % 1_000_000
    )


def derive_seed_run(master_seed: int, donor_count: int, sigma_eta: float) -> int:
    return int(master_seed) + int(stable_hash_run(donor_count, sigma_eta))


def make_cell_key(
    prevalence_bin: float | str, donor_count: int, sigma_eta: float
) -> str:
    prev = (
        format_prevalence_label(float(prevalence_bin))
        if isinstance(prevalence_bin, (int, float, np.floating, np.integer))
        else str(prevalence_bin)
    )
    return f"prevalence_bin={prev}|D={int(donor_count)}|sigma_eta={float(sigma_eta):g}"


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


def validate_qc_independence(
    *,
    x_sim: np.ndarray,
    library_size: np.ndarray,
    pct_mt: np.ndarray,
) -> dict[str, float]:
    x = x_sim[:, 0]
    y = x_sim[:, 1]
    checks = {
        "corr_library_x": _corr(library_size, x),
        "corr_library_y": _corr(library_size, y),
        "corr_pctmt_x": _corr(pct_mt, x),
        "corr_pctmt_y": _corr(pct_mt, y),
    }
    finite = np.asarray([v for v in checks.values() if np.isfinite(v)], dtype=float)
    checks["max_abs_corr"] = (
        float(np.max(np.abs(finite))) if finite.size else float("nan")
    )
    return checks


def compute_bin_id(angles: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
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

    qc_corr = validate_qc_independence(
        x_sim=x_sim, library_size=library_size, pct_mt=pct_mt
    )
    bin_id, bin_counts_total = compute_bin_id(angles, n_bins)

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


def gene_prevalence_schedule(
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
    labels = np.repeat(np.array([format_prevalence_label(x) for x in bins]), counts)
    order = rng.permutation(pi_target.size)
    return pi_target[order], labels[order]


def apply_foreground_edge_rule(
    f_raw: np.ndarray,
    *,
    prevalence_bin: float,
    cfg: EdgeRuleConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    f = np.asarray(f_raw, dtype=bool).ravel()
    fg_fraction_raw = float(np.mean(f)) if f.size else float("nan")
    prev_bin = float(prevalence_bin)
    threshold = float(cfg.threshold)
    strategy = str(cfg.strategy).strip().lower()
    if strategy not in {"none", "complement"}:
        raise ValueError("Edge rule strategy must be one of {'none', 'complement'}.")

    reasons: list[str] = []
    if np.isfinite(prev_bin) and prev_bin >= threshold:
        reasons.append("prevalence_bin")
    if np.isfinite(fg_fraction_raw) and fg_fraction_raw >= threshold:
        reasons.append("fg_fraction")
    triggered = bool(reasons)

    if triggered and strategy == "complement":
        f_test = np.logical_not(f)
        applied = "complement"
    else:
        f_test = f.copy()
        applied = "none"

    fg_fraction = float(np.mean(f_test)) if f_test.size else float("nan")
    bg_fraction = float(1.0 - fg_fraction) if np.isfinite(fg_fraction) else float("nan")
    contrast = (
        float(fg_fraction - bg_fraction) if np.isfinite(fg_fraction) else float("nan")
    )

    info = {
        "prevalence_edge_triggered": bool(triggered),
        "edge_trigger_reasons": reasons,
        "fg_rule_applied": applied,
        "fg_fraction_raw": fg_fraction_raw,
        "fg_fraction": fg_fraction,
        "bg_fraction": bg_fraction,
        "fg_bg_contrast": contrast,
    }
    return f_test, info


def summarize_null_t(null_t: np.ndarray) -> dict[str, float]:
    arr = np.asarray(null_t, dtype=float).ravel()
    if arr.size == 0:
        return {
            "n": 0,
            "min": float("nan"),
            "q01": float("nan"),
            "q05": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    q = np.quantile(arr, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q10": float(q[2]),
        "q25": float(q[3]),
        "median": float(q[4]),
        "q75": float(q[5]),
        "q90": float(q[6]),
        "q95": float(q[7]),
        "q99": float(q[8]),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


def run_single_null_gene(
    *,
    context: RunContext,
    pi_target: float,
    prevalence_bin: float,
    seed_gene: int,
    n_bins: int,
    n_perm: int,
    gate_cfg: GateConfig,
    edge_cfg: EdgeRuleConfig,
    run_id: str,
    seed_run: int,
    master_seed: int,
    include_null_t_values: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng_gene = rng_from_seed(seed_gene)
    eta_cell = donor_effect_vector(context.donor_ids, context.eta_d)

    f_raw = simulate_null_gene(
        pi_target=pi_target, donor_eta_per_cell=eta_cell, rng=rng_gene
    )
    prev_obs = float(np.mean(f_raw))

    f_test, edge_info = apply_foreground_edge_rule(
        f_raw,
        prevalence_bin=float(prevalence_bin),
        cfg=edge_cfg,
    )

    perm = perm_null_T(
        f=f_test,
        angles=context.angles,
        donor_ids=context.donor_ids,
        n_bins=int(n_bins),
        n_perm=int(n_perm),
        seed=int(seed_gene),
        mode="raw",
        smooth_w=1,
        donor_stratified=True,
        return_null_T=True,
        return_obs_profile=True,
        bin_id=context.bin_id,
        bin_counts_total=context.bin_counts_total,
    )

    power = evaluate_underpowered(
        donor_ids=context.donor_ids,
        f=f_test,
        n_perm=int(n_perm),
        p_min=float(gate_cfg.p_min),
        min_fg_total=int(gate_cfg.min_fg_total),
        min_fg_per_donor=int(gate_cfg.min_fg_per_donor),
        min_bg_per_donor=int(gate_cfg.min_bg_per_donor),
        d_eff_min=int(gate_cfg.d_eff_min),
        min_perm=int(gate_cfg.min_perm),
    )

    t_obs = float(perm["T_obs"])
    p_t = float(perm["p_T"])
    null_t = np.asarray(perm["null_T"], dtype=float)
    null_summary = summarize_null_t(null_t)
    z_t = float(robust_z(t_obs, null_t)) if null_t.size > 0 else float("nan")
    tie_rate = (
        float(np.mean(np.isclose(null_t, t_obs, atol=1e-12, rtol=0.0)))
        if null_t.size > 0
        else float("nan")
    )
    t_obs_minus_null_mean = (
        float(t_obs - null_summary["mean"])
        if np.isfinite(null_summary["mean"])
        else float("nan")
    )

    donor_fg = np.asarray(power["n_fg_per_donor"], dtype=float)
    donor_bg = np.asarray(power["n_bg_per_donor"], dtype=float)
    reasons = [k for k, v in dict(power["underpowered_reasons"]).items() if bool(v)]

    p_min_value = float(1.0 / (int(n_perm) + 1))

    metric_row: dict[str, Any] = {
        "run_id": str(run_id),
        "seed": int(seed_gene),
        "pi_target": float(pi_target),
        "prevalence_bin": format_prevalence_label(float(prevalence_bin)),
        "prev_obs": prev_obs,
        "prev_test": float(np.mean(f_test)),
        "n_fg": int(np.sum(f_test)),
        "n_fg_raw": int(np.sum(f_raw)),
        "n_fg_total": int(power["n_fg_total"]),
        "n_bg_total": int(power["n_bg_total"]),
        "D_eff": int(power["D_eff"]),
        "donor_fg_min": float(np.min(donor_fg)) if donor_fg.size else float("nan"),
        "donor_fg_med": float(np.median(donor_fg)) if donor_fg.size else float("nan"),
        "donor_fg_max": float(np.max(donor_fg)) if donor_fg.size else float("nan"),
        "donor_bg_min": float(np.min(donor_bg)) if donor_bg.size else float("nan"),
        "donor_bg_med": float(np.median(donor_bg)) if donor_bg.size else float("nan"),
        "donor_bg_max": float(np.max(donor_bg)) if donor_bg.size else float("nan"),
        "T_obs": t_obs,
        "Z_T": z_t,
        "p_T": p_t,
        "q_T": float("nan"),
        "T_null_mean": float(null_summary["mean"]),
        "T_null_median": float(null_summary["median"]),
        "T_obs_minus_null_mean": t_obs_minus_null_mean,
        "tie_rate_obs_vs_null": tie_rate,
        "fg_fraction": float(edge_info["fg_fraction"]),
        "bg_fraction": float(edge_info["bg_fraction"]),
        "fg_bg_contrast": float(edge_info["fg_bg_contrast"]),
        "prevalence_edge_triggered": bool(edge_info["prevalence_edge_triggered"]),
        "fg_rule_applied": str(edge_info["fg_rule_applied"]),
        "edge_trigger_reasons": json.dumps(
            edge_info["edge_trigger_reasons"], separators=(",", ":")
        ),
        "used_donor_stratified": bool(perm["used_donor_stratified"]),
        "stratified_counts_signature": str(perm["stratified_counts_signature"]),
        "stratified_counts_signature_hash": str(
            perm.get("stratified_counts_signature_hash", "")
        ),
        "underpowered": bool(power["underpowered"]),
        "underpowered_reasons": json.dumps(
            dict(power["underpowered_reasons"]), separators=(",", ":")
        ),
        "p_min_value": p_min_value,
        "is_p_min": bool(np.isclose(p_t, p_min_value, atol=1e-12, rtol=0.0)),
    }

    verbose_row: dict[str, Any] = {
        "run_id": str(run_id),
        "seed_policy": "seed_run = master_seed + stable_hash(D,sigma_eta) % 1e6; seed_gene = seed_run + g",
        "rng_seeds": {
            "master_seed": int(master_seed),
            "seed_run": int(seed_run),
            "seed_gene": int(seed_gene),
            "perm_seed": int(seed_gene),
        },
        "pi_target": float(pi_target),
        "prevalence_bin": format_prevalence_label(float(prevalence_bin)),
        "fg_total": int(power["n_fg_total"]),
        "fg_per_donor": [
            int(x) for x in np.asarray(power["n_fg_per_donor"], dtype=int).tolist()
        ],
        "bg_per_donor": [
            int(x) for x in np.asarray(power["n_bg_per_donor"], dtype=int).tolist()
        ],
        "D_eff": int(power["D_eff"]),
        "abstention_gate_reasons": reasons,
        "underpowered": bool(power["underpowered"]),
        "fg_rule_applied": str(edge_info["fg_rule_applied"]),
        "edge_trigger_reasons": list(edge_info["edge_trigger_reasons"]),
        "fg_fraction_raw": float(edge_info["fg_fraction_raw"]),
        "fg_fraction": float(edge_info["fg_fraction"]),
        "bg_fraction": float(edge_info["bg_fraction"]),
        "fg_bg_contrast": float(edge_info["fg_bg_contrast"]),
        "permutation_mask_signature": {
            "stratified_counts_signature": str(perm["stratified_counts_signature"]),
            "stratified_counts_signature_hash": str(
                perm.get("stratified_counts_signature_hash", "")
            ),
        },
        "T_obs": t_obs,
        "T_perm_summary": null_summary,
        "T_obs_minus_mean_T_perm": t_obs_minus_null_mean,
        "p_value": p_t,
        "p_min_value": p_min_value,
        "is_p_min": bool(np.isclose(p_t, p_min_value, atol=1e-12, rtol=0.0)),
        "null_cache_key": None,
        "used_donor_stratified": bool(perm["used_donor_stratified"]),
    }
    if include_null_t_values:
        verbose_row["T_perm_values"] = [float(x) for x in null_t.tolist()]

    return metric_row, verbose_row


def summarize_cell_metrics(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {
            "n_genes": 0,
            "n_non_underpowered": 0,
        }

    valid = metrics.loc[~metrics["underpowered"].astype(bool)].copy()
    n_valid = int(valid.shape[0])
    pvals = pd.to_numeric(valid["p_T"], errors="coerce").to_numpy(dtype=float)
    pvals = pvals[np.isfinite(pvals)]

    if n_valid > 0:
        k_p = int(np.sum(pvals <= 0.05))
        typei_p = float(k_p / n_valid)
        p_ci_low, p_ci_high = wilson_ci(k_p, n_valid, alpha=0.05)
        mean_p = float(np.mean(pvals)) if pvals.size else float("nan")
        median_p = float(np.median(pvals)) if pvals.size else float("nan")
        min_p = float(np.min(pvals)) if pvals.size else float("nan")
        if pvals.size >= 5:
            ks = kstest(pvals, "uniform")
            ks_stat = float(ks.statistic)
            ks_p = float(ks.pvalue)
        else:
            ks_stat = float("nan")
            ks_p = float("nan")
    else:
        typei_p = float("nan")
        p_ci_low = float("nan")
        p_ci_high = float("nan")
        mean_p = float("nan")
        median_p = float("nan")
        min_p = float("nan")
        ks_stat = float("nan")
        ks_p = float("nan")

    return {
        "n_genes": int(metrics.shape[0]),
        "n_non_underpowered": n_valid,
        "frac_underpowered": float(metrics["underpowered"].astype(bool).mean()),
        "typeI_p05": typei_p,
        "typeI_p05_ci_low": float(p_ci_low),
        "typeI_p05_ci_high": float(p_ci_high),
        "mean_p": mean_p,
        "median_p": median_p,
        "min_p": min_p,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "mean_T_obs": float(
            np.nanmean(pd.to_numeric(valid.get("T_obs", np.nan), errors="coerce"))
        ),
        "mean_T_null": float(
            np.nanmean(pd.to_numeric(valid.get("T_null_mean", np.nan), errors="coerce"))
        ),
        "mean_T_obs_minus_null": float(
            np.nanmean(
                pd.to_numeric(
                    valid.get("T_obs_minus_null_mean", np.nan), errors="coerce"
                )
            )
        ),
    }


def run_cell_null_replay(
    *,
    n_cells: int,
    g_total: int,
    n_bins: int,
    n_perm: int,
    prevalence: float,
    donor_count: int,
    sigma_eta: float,
    prevalence_grid: Iterable[float],
    gate_cfg: GateConfig,
    edge_cfg: EdgeRuleConfig,
    mu0: float,
    sigma_l: float,
    master_seed: int,
    n_genes_subset: int | None = None,
    include_null_t_values: bool = False,
    seed_run_override: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    prevalence_label = format_prevalence_label(float(prevalence))
    if prevalence_label not in {
        format_prevalence_label(float(x)) for x in prevalence_grid
    }:
        raise ValueError(
            f"prevalence={prevalence} is not present in the configured prevalence grid."
        )

    seed_run = (
        int(seed_run_override)
        if seed_run_override is not None
        else derive_seed_run(master_seed, donor_count, sigma_eta)
    )
    run_id = f"D{int(donor_count)}_sigma{float(sigma_eta):g}"
    cell_key = make_cell_key(prevalence_label, donor_count, sigma_eta)

    context, qc_corr = simulate_run_context(
        n_cells=int(n_cells),
        n_bins=int(n_bins),
        n_donors=int(donor_count),
        sigma_eta=float(sigma_eta),
        seed_run=int(seed_run),
        mu0=float(mu0),
        sigma_l=float(sigma_l),
    )

    rng_schedule = rng_from_seed(seed_run)
    pi_targets, pi_labels = gene_prevalence_schedule(
        int(g_total), prevalence_grid, rng=rng_schedule
    )
    gene_indices = np.flatnonzero(pi_labels == prevalence_label).astype(int)
    if n_genes_subset is not None and int(n_genes_subset) > 0:
        gene_indices = gene_indices[: int(n_genes_subset)]

    rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []

    for local_i, g in enumerate(gene_indices):
        seed_gene = int(seed_run) + int(g)
        row, log = run_single_null_gene(
            context=context,
            pi_target=float(pi_targets[g]),
            prevalence_bin=float(prevalence),
            seed_gene=seed_gene,
            n_bins=int(n_bins),
            n_perm=int(n_perm),
            gate_cfg=gate_cfg,
            edge_cfg=edge_cfg,
            run_id=run_id,
            seed_run=int(seed_run),
            master_seed=int(master_seed),
            include_null_t_values=bool(include_null_t_values),
        )
        row.update(
            {
                "N": int(n_cells),
                "D": int(donor_count),
                "sigma_eta": float(sigma_eta),
                "gene_index": int(g),
                "gene_subset_index": int(local_i),
                "cell_key": cell_key,
            }
        )
        log.update(
            {
                "gene_index": int(g),
                "gene_subset_index": int(local_i),
                "cell_key": cell_key,
                "D": int(donor_count),
                "sigma_eta": float(sigma_eta),
            }
        )
        rows.append(row)
        logs.append(log)

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        qvals = bh_fdr(
            pd.to_numeric(metrics["p_T"], errors="coerce").to_numpy(dtype=float)
        )
        metrics["q_T"] = qvals

    summary = summarize_cell_metrics(metrics)
    summary.update(
        {
            "cell_key": cell_key,
            "D": int(donor_count),
            "sigma_eta": float(sigma_eta),
            "prevalence_bin": prevalence_label,
            "seed_run": int(seed_run),
            "master_seed": int(master_seed),
            "n_perm": int(n_perm),
            "n_genes_subset": int(metrics.shape[0]),
            "seed_run_override": (
                None if seed_run_override is None else int(seed_run_override)
            ),
            "qc_corr": qc_corr,
        }
    )

    return metrics, logs, summary


def build_target_cell_diagnostics(
    metrics: pd.DataFrame,
    ks_df: pd.DataFrame,
    targets: Iterable[tuple[float, int, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for prevalence, donor_count, sigma_eta in targets:
        p_label = format_prevalence_label(float(prevalence))
        mask = (
            (metrics["D"].astype(int) == int(donor_count))
            & (metrics["sigma_eta"].astype(float) == float(sigma_eta))
            & (metrics["prevalence_bin"].astype(str) == p_label)
            & (~metrics["underpowered"].astype(bool))
        )
        subset = metrics.loc[mask].copy()
        if subset.empty:
            continue

        pvals = pd.to_numeric(subset["p_T"], errors="coerce").to_numpy(dtype=float)
        zvals = pd.to_numeric(subset["Z_T"], errors="coerce").to_numpy(dtype=float)

        ks_mask = (
            (ks_df["D"].astype(int) == int(donor_count))
            & (ks_df["sigma_eta"].astype(float) == float(sigma_eta))
            & (ks_df["prevalence_bin"].astype(str) == p_label)
        )
        ks_pvalue = (
            float(ks_df.loc[ks_mask, "ks_pvalue"].iloc[0])
            if ks_mask.any()
            else float("nan")
        )
        ks_stat = (
            float(ks_df.loc[ks_mask, "ks_statistic"].iloc[0])
            if ks_mask.any()
            else float("nan")
        )

        rows.append(
            {
                "cell_key": make_cell_key(p_label, donor_count, sigma_eta),
                "D": int(donor_count),
                "sigma_eta": float(sigma_eta),
                "prevalence_bin": p_label,
                "n_non_underpowered": int(subset.shape[0]),
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "mean_p": float(np.nanmean(pvals)),
                "var_p": float(np.nanvar(pvals)),
                "expected_mean_p": 0.5,
                "expected_var_p": 1.0 / 12.0,
                "mean_Z_T": float(np.nanmean(zvals)),
                "median_Z_T": float(np.nanmedian(zvals)),
                "mean_T_obs": float(
                    np.nanmean(pd.to_numeric(subset["T_obs"], errors="coerce"))
                ),
                "median_T_obs": float(
                    np.nanmedian(pd.to_numeric(subset["T_obs"], errors="coerce"))
                ),
                "mean_null_T_mean": float(
                    np.nanmean(
                        pd.to_numeric(
                            subset.get("T_null_mean", np.nan), errors="coerce"
                        )
                    )
                ),
                "mean_null_T_median": float(
                    np.nanmean(
                        pd.to_numeric(
                            subset.get("T_null_median", np.nan), errors="coerce"
                        )
                    )
                ),
                "mean_T_obs_minus_mean_null": float(
                    np.nanmean(
                        pd.to_numeric(
                            subset.get("T_obs_minus_null_mean", np.nan), errors="coerce"
                        )
                    )
                ),
                "mean_tie_rate_T_obs_vs_null": float(
                    np.nanmean(
                        pd.to_numeric(
                            subset.get("tie_rate_obs_vs_null", np.nan), errors="coerce"
                        )
                    )
                ),
                "min_p": float(np.nanmin(pvals)) if pvals.size else float("nan"),
                "max_p": float(np.nanmax(pvals)) if pvals.size else float("nan"),
                "typeI_p05": (
                    float(np.nanmean(pvals <= 0.05)) if pvals.size else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)
