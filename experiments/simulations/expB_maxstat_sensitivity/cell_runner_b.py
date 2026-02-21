"""Deterministic single-cell replay helpers for Experiment B."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kstest

from biorsp.permutation import (
    mode_max_stat_from_profiles,
    perm_null_T,
    permute_foreground_within_donor,
)
from biorsp.power import evaluate_underpowered
from biorsp.scoring import bh_fdr
from experiments.simulations._shared.donors import (
    assign_donors,
    donor_effect_vector,
    sample_donor_effects,
)
from experiments.simulations._shared.geometry import sample_geometry
from experiments.simulations._shared.models import simulate_null_gene
from experiments.simulations._shared.plots import wilson_ci
from experiments.simulations._shared.seeding import rng_from_seed, stable_seed


@dataclass(frozen=True)
class RunContextB:
    geometry: str
    D: int
    sigma_eta: float
    seed_run: int
    seed_dataset: int
    X_sim: np.ndarray
    angles: np.ndarray
    donor_ids: np.ndarray
    eta_d: np.ndarray
    bin_id: np.ndarray
    bin_counts_total: np.ndarray


@dataclass(frozen=True)
class GateConfigB:
    p_min: float = 0.005
    min_fg_total: int = 50
    min_fg_per_donor: int = 10
    min_bg_per_donor: int = 10
    d_eff_min: int = 2
    min_perm: int = 200


@dataclass(frozen=True)
class EdgeRuleConfigB:
    threshold: float = 0.8
    strategy: str = "none"


@dataclass(frozen=True)
class CellSpecB:
    geometry: str
    mode: str
    pi_target: float
    D: int
    bins_B: int
    smooth_w: int
    sigma_eta: float
    seed: int
    n_perm: int


def format_pi_label(pi: float) -> str:
    return f"{float(pi):.3f}".rstrip("0").rstrip(".")


def make_cell_key(spec: CellSpecB) -> str:
    return (
        f"geometry={spec.geometry}|mode={spec.mode}|pi={format_pi_label(spec.pi_target)}|"
        f"D={int(spec.D)}|B={int(spec.bins_B)}|w={int(spec.smooth_w)}|"
        f"sigma_eta={float(spec.sigma_eta):g}|seed={int(spec.seed)}"
    )


def _seed_from_fields(master_seed: int, *fields: Any) -> int:
    return int(stable_seed(int(master_seed), "expB", *fields))


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


def simulate_qc_geometry_correlations(
    *,
    x_sim: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Generate independent synthetic QC covariates and correlate with geometry axes."""
    rng = rng_from_seed(int(seed))
    n_cells = int(x_sim.shape[0])
    library_size = rng.lognormal(mean=10.0, sigma=0.5, size=n_cells).astype(float)
    pct_mt = rng.beta(a=12.0, b=160.0, size=n_cells).astype(float)
    x = np.asarray(x_sim[:, 0], dtype=float)
    y = np.asarray(x_sim[:, 1], dtype=float)
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


def _build_bin_cache_for_b(
    angles: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    wrapped = np.mod(np.asarray(angles, dtype=float).ravel(), 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, int(n_bins) + 1, endpoint=True)
    bin_id = np.digitize(wrapped, edges, right=False) - 1
    bin_id = np.where(bin_id == int(n_bins), int(n_bins) - 1, bin_id).astype(np.int32)
    counts = np.bincount(bin_id, minlength=int(n_bins)).astype(np.int64)
    return bin_id, counts


def simulate_run_context_b(
    *,
    geometry: str,
    n_cells: int,
    d_value: int,
    sigma_eta: float,
    bins_b: int,
    seed_run: int,
) -> tuple[RunContextB, dict[str, float]]:
    seed_dataset = _seed_from_fields(
        int(seed_run),
        "dataset",
        int(seed_run),
        str(geometry),
        int(d_value),
        float(sigma_eta),
    )
    rng = rng_from_seed(seed_dataset)
    x_sim, _ = sample_geometry(str(geometry), int(n_cells), rng)
    theta_raw = np.arctan2(x_sim[:, 1], x_sim[:, 0]).astype(float)
    angles = np.mod(theta_raw, 2.0 * np.pi).astype(float)
    donor_ids = assign_donors(int(n_cells), int(d_value), rng)
    eta_d = sample_donor_effects(int(d_value), float(sigma_eta), rng)
    bin_id, bin_counts_total = _build_bin_cache_for_b(angles, int(bins_b))
    qc_corr = simulate_qc_geometry_correlations(
        x_sim=x_sim,
        seed=_seed_from_fields(
            int(seed_run),
            "qc",
            str(geometry),
            int(d_value),
            float(sigma_eta),
        ),
    )
    return (
        RunContextB(
            geometry=str(geometry),
            D=int(d_value),
            sigma_eta=float(sigma_eta),
            seed_run=int(seed_run),
            seed_dataset=int(seed_dataset),
            X_sim=x_sim,
            angles=angles,
            donor_ids=donor_ids,
            eta_d=eta_d,
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        ),
        qc_corr,
    )


def build_pi_schedule(
    g_total: int,
    pi_bins: Iterable[float],
    rng: np.random.Generator,
) -> np.ndarray:
    bins = np.asarray(list(pi_bins), dtype=float)
    if bins.size < 1:
        raise ValueError("At least one prevalence bin is required.")
    base = int(g_total) // int(bins.size)
    rem = int(g_total) % int(bins.size)
    counts = np.full(int(bins.size), base, dtype=int)
    counts[:rem] += 1
    pi = np.repeat(bins, counts)
    order = rng.permutation(pi.size)
    return pi[order]


def apply_foreground_edge_rule_b(
    f_raw: np.ndarray,
    *,
    pi_target: float,
    cfg: EdgeRuleConfigB,
) -> tuple[np.ndarray, dict[str, Any]]:
    f = np.asarray(f_raw, dtype=bool).ravel()
    fg_fraction_raw = float(np.mean(f)) if f.size else float("nan")
    threshold = float(cfg.threshold)
    strategy = str(cfg.strategy).strip().lower()
    if strategy not in {"none", "complement"}:
        raise ValueError("Edge rule strategy must be one of {'none', 'complement'}.")

    reasons: list[str] = []
    if np.isfinite(float(pi_target)) and float(pi_target) >= threshold:
        reasons.append("pi_target")
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
    return f_test, {
        "prevalence_edge_triggered": bool(triggered),
        "edge_trigger_reasons": reasons,
        "fg_rule_applied": str(applied),
        "fg_fraction_raw": fg_fraction_raw,
        "fg_fraction_test": fg_fraction,
        "bg_fraction": bg_fraction,
        "fg_bg_contrast": (
            float(fg_fraction - bg_fraction)
            if np.isfinite(fg_fraction)
            else float("nan")
        ),
    }


def _donor_to_idx_map(donor_ids: np.ndarray) -> dict[str, np.ndarray]:
    donor_arr = np.asarray(donor_ids)
    labels = np.unique(donor_arr)
    return {str(d): np.flatnonzero(donor_arr == d).astype(int) for d in labels}


def _first_perm_mask_hashes(
    *,
    f: np.ndarray,
    donor_ids: np.ndarray,
    seed_perm: int,
    n_perm: int,
    n_hashes: int = 5,
) -> tuple[str, dict[str, int], list[str]]:
    f_obs = np.asarray(f, dtype=bool).ravel()
    donor_to_idx = _donor_to_idx_map(np.asarray(donor_ids))
    n_take = int(min(max(0, int(n_hashes)), int(n_perm)))
    rng = np.random.default_rng(int(seed_perm))
    hashes: list[str] = []
    for _ in range(n_take):
        if len(donor_to_idx) >= 2:
            f_perm = permute_foreground_within_donor(f_obs, donor_to_idx, rng)
            scheme = "donor_stratified"
        else:
            f_perm = rng.permutation(f_obs)
            scheme = "global"
        payload = np.packbits(np.asarray(f_perm, dtype=np.uint8)).tobytes()
        hashes.append(hashlib.sha1(payload).hexdigest())
    strata_sizes = {k: int(v.size) for k, v in donor_to_idx.items()}
    return scheme, strata_sizes, hashes


def summarize_null_t(null_t: np.ndarray) -> dict[str, float]:
    arr = np.asarray(null_t, dtype=float).ravel()
    if arr.size == 0:
        return {
            "n": 0,
            "min": float("nan"),
            "q05": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "sd": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "max": float("nan"),
        }
    q = np.quantile(arr, [0.05, 0.5, 0.95, 0.99])
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "q05": float(q[0]),
        "median": float(q[1]),
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=0)),
        "q95": float(q[2]),
        "q99": float(q[3]),
        "max": float(np.max(arr)),
    }


def run_single_null_gene_b(
    *,
    context: RunContextB,
    spec: CellSpecB,
    gate_cfg: GateConfigB,
    edge_cfg: EdgeRuleConfigB,
    seed_run: int,
    seed_gene: int,
    gene_index: int,
    include_null_t_values: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    rng_gene = rng_from_seed(int(seed_gene))
    eta_cell = donor_effect_vector(context.donor_ids, context.eta_d)
    f_raw = simulate_null_gene(
        pi_target=float(spec.pi_target),
        donor_eta_per_cell=eta_cell,
        rng=rng_gene,
    )
    f_test, edge_info = apply_foreground_edge_rule_b(
        f_raw,
        pi_target=float(spec.pi_target),
        cfg=edge_cfg,
    )

    pi_label = format_pi_label(float(spec.pi_target))
    seed_perm = _seed_from_fields(
        int(seed_run),
        "perm",
        str(spec.geometry),
        int(spec.D),
        float(spec.sigma_eta),
        int(spec.bins_B),
        pi_label,
        int(gene_index),
    )

    power = evaluate_underpowered(
        donor_ids=context.donor_ids,
        f=f_test,
        n_perm=int(spec.n_perm),
        p_min=float(gate_cfg.p_min),
        min_fg_total=int(gate_cfg.min_fg_total),
        min_fg_per_donor=int(gate_cfg.min_fg_per_donor),
        min_bg_per_donor=int(gate_cfg.min_bg_per_donor),
        d_eff_min=int(gate_cfg.d_eff_min),
        min_perm=int(gate_cfg.min_perm),
    )
    perm_raw = perm_null_T(
        f=f_test,
        angles=context.angles,
        donor_ids=context.donor_ids,
        n_bins=int(spec.bins_B),
        n_perm=int(spec.n_perm),
        seed=int(seed_perm),
        mode="raw",
        smooth_w=1,
        donor_stratified=True,
        return_null_T=True,
        return_obs_profile=True,
        return_null_profiles=True,
        bin_id=context.bin_id,
        bin_counts_total=context.bin_counts_total,
    )
    mode_stats = mode_max_stat_from_profiles(
        E_obs_raw=np.asarray(perm_raw["E_phi_obs"], dtype=float),
        null_E_raw=np.asarray(perm_raw["null_E_phi"], dtype=float),
        mode=str(spec.mode),
        smooth_w=int(spec.smooth_w),
    )

    null_t = np.asarray(mode_stats["null_T"], dtype=float)
    t_obs = float(mode_stats["T_obs"])
    p_t = float(mode_stats["p_T"])
    null_summary = summarize_null_t(null_t)
    p_min_value = float(1.0 / (int(spec.n_perm) + 1))
    delta = (
        float(t_obs - null_summary["mean"])
        if np.isfinite(null_summary["mean"])
        else float("nan")
    )
    reasons = [k for k, v in dict(power["underpowered_reasons"]).items() if bool(v)]
    perm_scheme, donor_strata_sizes, perm_hashes = _first_perm_mask_hashes(
        f=f_test,
        donor_ids=context.donor_ids,
        seed_perm=int(seed_perm),
        n_perm=int(spec.n_perm),
        n_hashes=5,
    )

    metric_row: dict[str, Any] = {
        "run_id": (
            f"{spec.geometry}__D{int(spec.D)}__sigma{float(spec.sigma_eta):g}__seed{int(spec.seed)}"
        ),
        "seed": int(seed_gene),
        "seed_run": int(seed_run),
        "seed_gene": int(seed_gene),
        "seed_perm": int(seed_perm),
        "geometry": str(spec.geometry),
        "D": int(spec.D),
        "sigma_eta": float(spec.sigma_eta),
        "pi_target": float(spec.pi_target),
        "prev_obs": float(np.mean(f_raw)),
        "n_fg": int(np.sum(f_test)),
        "n_fg_raw": int(np.sum(f_raw)),
        "n_fg_total": int(power["n_fg_total"]),
        "n_bg_total": int(power["n_bg_total"]),
        "D_eff": int(power["D_eff"]),
        "bins_B": int(spec.bins_B),
        "smooth_w": int(spec.smooth_w),
        "mode": str(spec.mode),
        "T_obs": t_obs,
        "p_T": p_t,
        "T_null_mean": float(null_summary["mean"]),
        "T_null_median": float(null_summary["median"]),
        "T_obs_minus_null_mean": delta,
        "fg_fraction_raw": float(edge_info["fg_fraction_raw"]),
        "fg_fraction_test": float(edge_info["fg_fraction_test"]),
        "bg_fraction": float(edge_info["bg_fraction"]),
        "fg_bg_contrast": float(edge_info["fg_bg_contrast"]),
        "prevalence_edge_triggered": bool(edge_info["prevalence_edge_triggered"]),
        "fg_rule_applied": str(edge_info["fg_rule_applied"]),
        "edge_trigger_reasons": json.dumps(
            edge_info["edge_trigger_reasons"], separators=(",", ":")
        ),
        "underpowered": bool(power["underpowered"]),
        "underpowered_reasons": json.dumps(
            dict(power["underpowered_reasons"]), separators=(",", ":")
        ),
        "is_p_min": bool(np.isclose(p_t, p_min_value, atol=1e-12, rtol=0.0)),
        "p_min_value": p_min_value,
        "perm_scheme": str(perm_scheme),
        "perm_hashes_head": json.dumps(perm_hashes, separators=(",", ":")),
    }

    log_row: dict[str, Any] = {
        "cell_key": make_cell_key(spec),
        "cell_key_obj": {
            "geometry": str(spec.geometry),
            "mode": str(spec.mode),
            "pi": float(spec.pi_target),
            "D": int(spec.D),
            "B": int(spec.bins_B),
            "w": int(spec.smooth_w),
            "sigma_eta": float(spec.sigma_eta),
            "seed": int(spec.seed),
            "n_perm": int(spec.n_perm),
        },
        "fg_total": int(power["n_fg_total"]),
        "fg_per_donor": [
            int(x) for x in np.asarray(power["n_fg_per_donor"], dtype=int)
        ],
        "D_eff": int(power["D_eff"]),
        "abstain": bool(power["underpowered"]),
        "abstain_reasons": reasons,
        "rng": {
            "global_seed": int(spec.seed),
            "seed_run": int(seed_run),
            "seed_dataset": int(context.seed_dataset),
            "seed_gene": int(seed_gene),
            "seed_perm": int(seed_perm),
        },
        "perm_scheme": str(perm_scheme),
        "donor_strata_sizes": donor_strata_sizes,
        "perm_mask_hashes": perm_hashes,
        "T_obs": t_obs,
        "T_perm_summary": null_summary,
        "delta": delta,
        "p_value": p_t,
        "is_p_min": bool(np.isclose(p_t, p_min_value, atol=1e-12, rtol=0.0)),
        "p_min_value": p_min_value,
        "cache": {
            "null_cache_key": None,
            "null_cache_loaded": False,
        },
        "fg_fraction_raw": float(edge_info["fg_fraction_raw"]),
        "fg_fraction": float(edge_info["fg_fraction_test"]),
        "bg_fraction": float(edge_info["bg_fraction"]),
        "fg_bg_contrast": float(edge_info["fg_bg_contrast"]),
        "prevalence_edge_triggered": bool(edge_info["prevalence_edge_triggered"]),
        "fg_rule_applied": str(edge_info["fg_rule_applied"]),
        "edge_trigger_reasons": list(edge_info["edge_trigger_reasons"]),
    }
    if include_null_t_values:
        log_row["T_perm_values"] = [float(x) for x in null_t.tolist()]
    return metric_row, log_row, null_t


def summarize_cell_metrics_b(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {"n_genes": 0, "n_non_underpowered": 0}

    valid = metrics.loc[~metrics["underpowered"].astype(bool)].copy()
    pvals = pd.to_numeric(valid.get("p_T", np.nan), errors="coerce").to_numpy(
        dtype=float
    )
    pvals = pvals[np.isfinite(pvals)]
    n_valid = int(pvals.size)

    if n_valid > 0:
        k05 = int(np.sum(pvals <= 0.05))
        k01 = int(np.sum(pvals <= 0.01))
        ci_low, ci_high = wilson_ci(k05, n_valid, alpha=0.05)
        mean_p = float(np.mean(pvals))
        var_p = float(np.var(pvals))
        min_p = float(np.min(pvals))
        if n_valid >= 5:
            ks = kstest(pvals, "uniform")
            ks_stat = float(ks.statistic)
            ks_p = float(ks.pvalue)
        else:
            ks_stat = float("nan")
            ks_p = float("nan")
    else:
        k05 = 0
        k01 = 0
        ci_low = float("nan")
        ci_high = float("nan")
        mean_p = float("nan")
        var_p = float("nan")
        min_p = float("nan")
        ks_stat = float("nan")
        ks_p = float("nan")

    return {
        "n_genes": int(metrics.shape[0]),
        "n_non_underpowered": int(n_valid),
        "frac_underpowered": float(metrics["underpowered"].astype(bool).mean()),
        "mean_p": mean_p,
        "var_p": var_p,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "typeI_p05": float(k05 / n_valid) if n_valid > 0 else float("nan"),
        "typeI_p05_ci_low": float(ci_low),
        "typeI_p05_ci_high": float(ci_high),
        "count_p_lt_001": int(k01),
        "count_p_lt_005": int(k05),
        "min_p": min_p,
    }


def run_cell_null_replay_b(
    *,
    spec: CellSpecB,
    n_cells: int,
    g_total: int,
    pi_bins: Iterable[float],
    gate_cfg: GateConfigB,
    edge_cfg: EdgeRuleConfigB,
    n_genes_subset: int | None = None,
    compute_qvalues: bool = False,
    include_null_t_values: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], np.ndarray]:
    mode_norm = str(spec.mode).strip().lower()
    if mode_norm not in {"raw", "smoothed"}:
        raise ValueError("mode must be one of {'raw','smoothed'}.")
    if mode_norm == "raw":
        w_eff = 1
    else:
        w_eff = int(spec.smooth_w)
        if w_eff < 1 or w_eff % 2 == 0:
            raise ValueError("smooth_w must be odd >= 1 for smoothed mode.")

    spec_eff = CellSpecB(
        geometry=str(spec.geometry),
        mode=mode_norm,
        pi_target=float(spec.pi_target),
        D=int(spec.D),
        bins_B=int(spec.bins_B),
        smooth_w=int(w_eff),
        sigma_eta=float(spec.sigma_eta),
        seed=int(spec.seed),
        n_perm=int(spec.n_perm),
    )
    cell_key = make_cell_key(spec_eff)

    context, qc_corr = simulate_run_context_b(
        geometry=str(spec_eff.geometry),
        n_cells=int(n_cells),
        d_value=int(spec_eff.D),
        sigma_eta=float(spec_eff.sigma_eta),
        bins_b=int(spec_eff.bins_B),
        seed_run=int(spec_eff.seed),
    )

    rng_schedule = rng_from_seed(
        _seed_from_fields(
            int(spec_eff.seed),
            "pi_schedule",
            str(spec_eff.geometry),
            int(spec_eff.D),
            float(spec_eff.sigma_eta),
        )
    )
    pi_targets = build_pi_schedule(int(g_total), list(pi_bins), rng_schedule)
    matches = np.flatnonzero(
        np.isclose(
            pi_targets.astype(float),
            float(spec_eff.pi_target),
            atol=1e-12,
            rtol=0.0,
        )
    ).astype(int)
    if matches.size == 0:
        raise ValueError(
            f"No genes matched pi={spec_eff.pi_target} in schedule; check provided pi_bins."
        )
    if n_genes_subset is not None and int(n_genes_subset) > 0:
        matches = matches[: int(n_genes_subset)]

    rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    pooled_null_t: list[np.ndarray] = []

    for local_i, g in enumerate(matches.tolist()):
        seed_gene = _seed_from_fields(
            int(spec_eff.seed),
            "gene",
            str(spec_eff.geometry),
            int(spec_eff.D),
            float(spec_eff.sigma_eta),
            format_pi_label(float(spec_eff.pi_target)),
            int(g),
        )
        row, log, null_t = run_single_null_gene_b(
            context=context,
            spec=spec_eff,
            gate_cfg=gate_cfg,
            edge_cfg=edge_cfg,
            seed_run=int(spec_eff.seed),
            seed_gene=int(seed_gene),
            gene_index=int(g),
            include_null_t_values=bool(include_null_t_values),
        )
        row["gene_index"] = int(g)
        row["gene_subset_index"] = int(local_i)
        row["cell_key"] = cell_key
        log["gene_index"] = int(g)
        log["gene_subset_index"] = int(local_i)
        logs.append(log)
        rows.append(row)
        pooled_null_t.append(np.asarray(null_t, dtype=float))

    metrics = pd.DataFrame(rows)
    if compute_qvalues and not metrics.empty:
        qvals = bh_fdr(
            pd.to_numeric(metrics["p_T"], errors="coerce").to_numpy(dtype=float)
        )
        metrics["q_T"] = qvals

    summary = summarize_cell_metrics_b(metrics)
    summary.update(
        {
            "cell_key": cell_key,
            "geometry": str(spec_eff.geometry),
            "mode": str(spec_eff.mode),
            "pi_target": float(spec_eff.pi_target),
            "D": int(spec_eff.D),
            "bins_B": int(spec_eff.bins_B),
            "smooth_w": int(spec_eff.smooth_w),
            "sigma_eta": float(spec_eff.sigma_eta),
            "seed": int(spec_eff.seed),
            "n_perm": int(spec_eff.n_perm),
            "p_min": float(1.0 / (int(spec_eff.n_perm) + 1)),
            "qc_corr": qc_corr,
        }
    )
    pooled = (
        np.concatenate(pooled_null_t).astype(float)
        if pooled_null_t
        else np.zeros(0, dtype=float)
    )
    return metrics, logs, summary, pooled
