"""Staged scope-level discovery pipeline with deterministic caching."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from biorsp.permutation import perm_null_T_and_profile
from biorsp.pipeline_utils import bh_fdr
from biorsp.rsp import compute_rsp_profile_from_boolean, plot_rsp_polar
from biorsp.scope_cache import ScopeCache, build_or_load_scope_cache
from biorsp.scoring import coverage_from_null
from biorsp.utils import ensure_dir

EPS = 1e-12


@dataclass(frozen=True)
class ScopeParams:
    discovery_mode: bool
    pipeline_mode: str
    scope_level: str
    bins_screen: int
    bins_confirm: int
    stage1_top_k: int
    stage2_top_k: int
    min_fg: int
    min_prev: float
    perm_init: int
    perm_mid: int
    perm_final: int
    p_escalate_1: float
    p_escalate_2: float
    q_threshold: float
    plot_top_k: int
    skip_umap_plots: bool
    skip_pair_plots: bool
    skip_rsp_plots: bool
    seed: int


def _safe_scope_level(scope_level: str | None) -> str:
    level = str(scope_level or "global").strip().lower()
    if level not in {"global", "mega", "cluster"}:
        return "global"
    return level


def _resolve_scope_int(params: dict[str, Any], key: str, scope_level: str, default: int) -> int:
    scoped_key = f"{key}_{scope_level}"
    if scoped_key in params:
        return int(params[scoped_key])
    return int(params.get(key, default))


def _resolve_scope_float(params: dict[str, Any], key: str, scope_level: str, default: float) -> float:
    scoped_key = f"{key}_{scope_level}"
    if scoped_key in params:
        return float(params[scoped_key])
    return float(params.get(key, default))


def _build_scope_params(scope_ctx: dict[str, Any], params: dict[str, Any]) -> ScopeParams:
    scope_level = _safe_scope_level(scope_ctx.get("scope_level"))
    discovery_mode = bool(params.get("discovery_mode", True))
    pipeline_mode_default = "compute" if discovery_mode else "full"
    pipeline_mode = str(params.get("pipeline_mode", pipeline_mode_default)).strip().lower()
    if pipeline_mode not in {"compute", "plot", "full"}:
        pipeline_mode = pipeline_mode_default
    skip_defaults = bool(discovery_mode)
    return ScopeParams(
        discovery_mode=discovery_mode,
        pipeline_mode=pipeline_mode,
        scope_level=scope_level,
        bins_screen=int(params.get("bins_screen", 36)),
        bins_confirm=int(params.get("bins_confirm", 72)),
        stage1_top_k=_resolve_scope_int(params, "stage1_top_k", scope_level, 2000),
        stage2_top_k=_resolve_scope_int(params, "stage2_top_k", scope_level, 200),
        min_fg=_resolve_scope_int(params, "min_fg", scope_level, 50),
        min_prev=_resolve_scope_float(params, "min_prev", scope_level, 0.01),
        perm_init=int(params.get("perm_init", 100)),
        perm_mid=int(params.get("perm_mid", 300)),
        perm_final=int(params.get("perm_final", 1000)),
        p_escalate_1=float(params.get("p_escalate_1", 0.2)),
        p_escalate_2=float(params.get("p_escalate_2", 0.05)),
        q_threshold=float(params.get("q_threshold", 0.05)),
        plot_top_k=int(params.get("plot_top_k", 20)),
        skip_umap_plots=bool(params.get("skip_umap_plots", skip_defaults)),
        skip_pair_plots=bool(params.get("skip_pair_plots", skip_defaults)),
        skip_rsp_plots=bool(params.get("skip_rsp_plots", skip_defaults)),
        seed=int(params.get("seed", 0)),
    )


def _get_gene_column_dense(X, gene_idx: int) -> np.ndarray:
    col = X[:, int(gene_idx)]
    if sp.issparse(col):
        return col.toarray().ravel()
    return np.asarray(col).ravel()


def _get_fg_indices(X, gene_idx: int) -> np.ndarray:
    if sp.isspmatrix_csc(X):
        start = int(X.indptr[gene_idx])
        end = int(X.indptr[gene_idx + 1])
        return X.indices[start:end].astype(np.int32, copy=False)
    col = X[:, int(gene_idx)]
    if sp.issparse(col):
        return col.indices.astype(np.int32, copy=False)
    return np.flatnonzero(np.asarray(col).ravel() > 0).astype(np.int32)


def _is_ambient_gene(gene: str) -> bool:
    g = str(gene).upper()
    if g.startswith("MT-") or g.startswith("RPL") or g.startswith("RPS"):
        return True
    return False


def _stage1_scores(
    X,
    genes: list[str],
    labels: np.ndarray | None,
    min_fg: int,
    min_prev: float,
) -> tuple[pd.DataFrame, list[int], dict[int, dict[str, Any]]]:
    n_cells, n_genes = int(X.shape[0]), int(X.shape[1])
    if n_genes != len(genes):
        raise ValueError("genes length must match X.shape[1].")

    if sp.issparse(X):
        detect_n = np.asarray(X.getnnz(axis=0)).ravel().astype(np.int32)
        mean_expr = np.asarray(X.mean(axis=0)).ravel().astype(float)
    else:
        arr = np.asarray(X)
        detect_n = np.sum(arr > 0, axis=0).astype(np.int32)
        mean_expr = np.mean(arr, axis=0).astype(float)
    prev = detect_n / max(1, n_cells)

    if labels is not None:
        lab = np.asarray(labels)
        if lab.size != n_cells:
            raise ValueError("labels length must match n_cells.")
        uniq = pd.unique(lab)
        if uniq.size >= 2:
            region_mask = lab == uniq[0]
            rest_mask = ~region_mask
            n_region = int(np.sum(region_mask))
            n_rest = int(np.sum(rest_mask))
            if n_region > 0 and n_rest > 0:
                if sp.issparse(X):
                    det_region = np.asarray(X[region_mask].getnnz(axis=0)).ravel()
                    det_rest = np.asarray(X[rest_mask].getnnz(axis=0)).ravel()
                    mean_region = np.asarray(X[region_mask].mean(axis=0)).ravel()
                    mean_rest = np.asarray(X[rest_mask].mean(axis=0)).ravel()
                else:
                    arr = np.asarray(X)
                    det_region = np.sum(arr[region_mask] > 0, axis=0)
                    det_rest = np.sum(arr[rest_mask] > 0, axis=0)
                    mean_region = np.mean(arr[region_mask], axis=0)
                    mean_rest = np.mean(arr[rest_mask], axis=0)
                delta_det = np.abs(det_region / max(1, n_region) - det_rest / max(1, n_rest))
                delta_mean = np.abs(mean_region - mean_rest)
            else:
                delta_det = np.abs(prev - 0.5) * 2.0
                delta_mean = mean_expr
        else:
            delta_det = np.abs(prev - 0.5) * 2.0
            delta_mean = mean_expr
    else:
        delta_det = np.abs(prev - 0.5) * 2.0
        delta_mean = mean_expr

    max_delta_mean = float(np.max(delta_mean)) if np.max(delta_mean) > 0 else 1.0
    score = delta_det + (delta_mean / max_delta_mean)

    rows = []
    eligible_gene_idx: list[int] = []
    gene_meta: dict[int, dict[str, Any]] = {}
    for idx, gene in enumerate(genes):
        n_fg = int(detect_n[idx])
        prev_i = float(prev[idx])
        ambient = _is_ambient_gene(gene)
        skip_reason = ""
        eligible = True
        if ambient:
            eligible = False
            skip_reason = "excluded_ambient_gene"
        elif n_fg < int(min_fg):
            eligible = False
            skip_reason = "min_fg"
        elif prev_i < float(min_prev) or prev_i > float(1.0 - min_prev):
            eligible = False
            skip_reason = "prevalence_gate"

        rows.append(
            {
                "gene": gene,
                "gene_idx": int(idx),
                "n_fg": n_fg,
                "prevalence": prev_i,
                "delta_det": float(delta_det[idx]),
                "delta_mean": float(delta_mean[idx]),
                "screen_score": float(score[idx]),
                "eligible": bool(eligible),
                "skip_reason": skip_reason,
            }
        )
        gene_meta[int(idx)] = {
            "n_fg": n_fg,
            "prevalence": prev_i,
            "stage_reached": 0,
            "skip_reason": skip_reason,
            "n_perm_final_used": 0,
        }
        if eligible:
            eligible_gene_idx.append(int(idx))
            gene_meta[int(idx)]["stage_reached"] = 1

    return pd.DataFrame(rows), eligible_gene_idx, gene_meta


def _peak_count(E_phi: np.ndarray, frac: float = 0.5) -> int:
    arr = np.asarray(E_phi, dtype=float).ravel()
    if arr.size < 3:
        return 0
    threshold = float(np.max(arr) * frac)
    peaks = 0
    for i in range(arr.size):
        left = arr[(i - 1) % arr.size]
        right = arr[(i + 1) % arr.size]
        if arr[i] > left and arr[i] > right and arr[i] >= threshold:
            peaks += 1
    return int(peaks)


def _specificity_from_profile(E_phi: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(E_phi, dtype=float).ravel()
    shifted = arr - np.min(arr)
    total = float(np.sum(shifted))
    if total <= EPS:
        return 1.0, 0.0
    p = shifted / total
    entropy = float(-np.sum(p * np.log(p + EPS)) / np.log(max(2, p.size)))
    specificity = float(1.0 - entropy)
    return entropy, specificity


def _stage2_metrics(
    X,
    genes: list[str],
    gene_idx: list[int],
    cache: ScopeCache,
    min_prev: float,
    gene_meta: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx in gene_idx:
        fg_idx = _get_fg_indices(X, idx)
        prev = float(gene_meta[idx]["prevalence"])
        if prev < min_prev or prev > (1.0 - min_prev):
            gene_meta[idx]["skip_reason"] = "stage2_prevalence_gate"
            continue
        f_mask = np.zeros(cache.bin_id.size, dtype=bool)
        f_mask[fg_idx] = True
        try:
            E_phi, phi_max, e_max = compute_rsp_profile_from_boolean(
                f_mask,
                cache.angles,
                int(cache.metadata.bins),
                bin_id=cache.bin_id,
                bin_counts_total=cache.bin_counts_total,
            )
        except ValueError:
            gene_meta[idx]["skip_reason"] = "stage2_rsp_invalid"
            continue
        entropy, specificity = _specificity_from_profile(E_phi)
        rows.append(
            {
                "gene": genes[idx],
                "gene_idx": int(idx),
                "n_fg": int(gene_meta[idx]["n_fg"]),
                "prevalence": prev,
                "E_max": float(e_max),
                "phi_max_deg": float(phi_max * 180.0 / math.pi),
                "peak_count": int(_peak_count(E_phi, frac=0.5)),
                "entropy": float(entropy),
                "specificity_proxy": float(specificity),
                "strength_proxy": float(e_max),
                "stage2_score": float(e_max + specificity),
            }
        )
        gene_meta[idx]["stage_reached"] = 2
        gene_meta[idx]["skip_reason"] = ""
    return pd.DataFrame(rows)


def _adaptive_permutation(
    *,
    expr: np.ndarray,
    donor_ids: np.ndarray,
    cache: ScopeCache,
    params: ScopeParams,
) -> dict[str, Any]:
    current = perm_null_T_and_profile(
        expr=expr,
        angles=cache.angles,
        donor_ids=donor_ids,
        n_bins=params.bins_confirm,
        n_perm=params.perm_init,
        seed=params.seed,
        donor_stratified=True,
        perm_indices=cache.perm_indices,
        perm_start=0,
        perm_end=params.perm_init,
        bin_id=cache.bin_id,
        bin_counts_total=cache.bin_counts_total,
    )
    used = int(current["n_perm_used"])
    if float(current["p_T"]) < params.p_escalate_1 and params.perm_mid > used:
        current = perm_null_T_and_profile(
            expr=expr,
            angles=cache.angles,
            donor_ids=donor_ids,
            n_bins=params.bins_confirm,
            n_perm=params.perm_mid - used,
            seed=params.seed,
            donor_stratified=True,
            perm_indices=cache.perm_indices,
            perm_start=used,
            perm_end=params.perm_mid,
            previous_null_T=np.asarray(current["null_T"], dtype=float),
            previous_null_E_phi=np.asarray(current["null_E_phi"], dtype=float),
            bin_id=cache.bin_id,
            bin_counts_total=cache.bin_counts_total,
        )
        used = int(current["n_perm_used"])
    if float(current["p_T"]) < params.p_escalate_2 and params.perm_final > used:
        current = perm_null_T_and_profile(
            expr=expr,
            angles=cache.angles,
            donor_ids=donor_ids,
            n_bins=params.bins_confirm,
            n_perm=params.perm_final - used,
            seed=params.seed,
            donor_stratified=True,
            perm_indices=cache.perm_indices,
            perm_start=used,
            perm_end=params.perm_final,
            previous_null_T=np.asarray(current["null_T"], dtype=float),
            previous_null_E_phi=np.asarray(current["null_E_phi"], dtype=float),
            bin_id=cache.bin_id,
            bin_counts_total=cache.bin_counts_total,
        )

    null_E = np.asarray(current["null_E_phi"], dtype=float)
    E_obs = np.asarray(current["E_phi_obs"], dtype=float)
    coverage95 = float(coverage_from_null(E_obs, null_E, q=0.95))
    return {
        "T_obs": float(current["T_obs"]),
        "p_T": float(current["p_T"]),
        "n_perm_final_used": int(current["n_perm_used"]),
        "coverage_q95": coverage95,
    }


def _select_representatives(
    stage3_df: pd.DataFrame,
    plot_top_k: int,
    q_threshold: float,
) -> pd.DataFrame:
    if stage3_df.empty:
        return stage3_df.copy()

    confirmed = stage3_df[stage3_df["q_T"] <= float(q_threshold)].copy()
    if confirmed.empty:
        confirmed = stage3_df.copy()

    strength_thr = float(np.nanmedian(confirmed["strength_proxy"].values))
    spec_thr = float(np.nanmedian(confirmed["specificity_proxy"].values))

    def quadrant(row: pd.Series) -> int:
        hi_strength = float(row["strength_proxy"]) >= strength_thr
        hi_spec = float(row["specificity_proxy"]) >= spec_thr
        if hi_strength and hi_spec:
            return 1
        if hi_strength and not hi_spec:
            return 2
        if (not hi_strength) and hi_spec:
            return 3
        return 4

    confirmed = confirmed.copy()
    confirmed["quadrant"] = confirmed.apply(quadrant, axis=1)
    per_quad = max(1, int(math.ceil(plot_top_k / 4.0)))
    selected_frames = []
    for q in [1, 2, 3, 4]:
        q_df = confirmed[confirmed["quadrant"] == q].copy()
        q_df = q_df.sort_values(["q_T", "p_T", "strength_proxy"], ascending=[True, True, False]).head(per_quad)
        if not q_df.empty:
            selected_frames.append(q_df)

    if selected_frames:
        selected = pd.concat(selected_frames, ignore_index=True).drop_duplicates(subset=["gene"])
    else:
        selected = confirmed.head(0).copy()

    if selected.shape[0] < int(plot_top_k):
        already = set(selected["gene"].tolist())
        extra = confirmed[~confirmed["gene"].isin(already)].sort_values(
            ["q_T", "p_T", "strength_proxy"],
            ascending=[True, True, False],
        )
        selected = pd.concat([selected, extra], ignore_index=True).head(int(plot_top_k))
    else:
        selected = selected.head(int(plot_top_k))

    return selected.reset_index(drop=True)


def _funnel_report(stage1_df: pd.DataFrame, stage2_df: pd.DataFrame, stage3_df: pd.DataFrame, reps_df: pd.DataFrame) -> dict[str, int]:
    n_total = int(stage1_df.shape[0])
    n_eligible = int(stage1_df["eligible"].sum()) if "eligible" in stage1_df.columns else 0
    n_stage1_selected = int(stage1_df.get("stage1_selected", pd.Series([], dtype=bool)).sum())
    n_stage2_selected = int(stage2_df.get("stage2_selected", pd.Series([], dtype=bool)).sum())
    n_stage3_tested = int(stage3_df.shape[0])
    n_significant = int(np.sum(stage3_df["q_T"] <= 0.05)) if "q_T" in stage3_df.columns else 0
    n_plotted = int(reps_df.shape[0])
    return {
        "n_genes_total": n_total,
        "n_genes_eligible": n_eligible,
        "n_stage1_selected": n_stage1_selected,
        "n_stage2_selected": n_stage2_selected,
        "n_stage3_tested": n_stage3_tested,
        "n_significant": n_significant,
        "n_plotted": n_plotted,
    }


def _scope_logger(scope_ctx: dict[str, Any]) -> logging.Logger:
    logger = scope_ctx.get("logger")
    if isinstance(logger, logging.Logger):
        return logger
    return logging.getLogger("biorsp_staged")


def _ensure_stage_tables(
    out_dir: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    ensure_dir(out_dir.as_posix())
    return (
        out_dir / "stage1.csv",
        out_dir / "stage2.csv",
        out_dir / "stage3.csv",
        out_dir / "representatives.csv",
        out_dir / "funnel_gene_status.csv",
    )


def run_scope_staged(
    scope_ctx: dict[str, Any],
    X,
    genes: list[str],
    labels: np.ndarray | None,
    qc_covariates: pd.DataFrame | None,
    cache_dir: Path | str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run staged compute/plot pipeline for one analysis scope."""
    del qc_covariates  # reserved for future QC penalties

    p = _build_scope_params(scope_ctx, params)
    logger = _scope_logger(scope_ctx)
    scope_id = str(scope_ctx.get("scope_id", "scope"))
    scope_name = str(scope_ctx.get("scope_name", scope_id))
    out_dir = Path(scope_ctx["out_dir"])
    fig_dir = Path(scope_ctx.get("figure_dir", out_dir))
    ensure_dir(out_dir.as_posix())
    ensure_dir(fig_dir.as_posix())

    stage1_path, stage2_path, stage3_path, reps_path, funnel_gene_path = _ensure_stage_tables(out_dir)
    funnel_summary_csv = out_dir / "funnel_report.csv"
    funnel_summary_json = out_dir / "funnel_report.json"

    n_cells = int(X.shape[0])
    min_cells = _resolve_scope_int(params, "min_cells", p.scope_level, 1)
    if n_cells < min_cells:
        logger.info("Skipping scope=%s due to min_cells gate (%s < %s).", scope_name, n_cells, min_cells)
        empty_stage1 = pd.DataFrame(columns=["gene", "gene_idx", "eligible"])
        empty_stage2 = pd.DataFrame(columns=["gene", "gene_idx"])
        empty_stage3 = pd.DataFrame(columns=["gene", "gene_idx", "p_T", "q_T"])
        empty_rep = empty_stage3.copy()
        empty_funnel = pd.DataFrame(columns=["gene", "stage_reached", "skip_reason", "n_perm_final_used"])
        empty_stage1.to_csv(stage1_path, index=False)
        empty_stage2.to_csv(stage2_path, index=False)
        empty_stage3.to_csv(stage3_path, index=False)
        empty_rep.to_csv(reps_path, index=False)
        empty_funnel.to_csv(funnel_gene_path, index=False)
        summary = _funnel_report(empty_stage1, empty_stage2, empty_stage3, empty_rep)
        pd.DataFrame([summary]).to_csv(funnel_summary_csv, index=False)
        funnel_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {
            "stage1": empty_stage1,
            "stage2": empty_stage2,
            "stage3": empty_stage3,
            "representatives": empty_rep,
            "funnel_gene": empty_funnel,
            "scope_params": p,
        }

    angles = np.asarray(scope_ctx["angles"], dtype=float).ravel()
    donor_ids = np.asarray(scope_ctx["donor_ids"])
    if angles.size != n_cells:
        raise ValueError("scope_ctx['angles'] length must match X.shape[0].")
    if donor_ids.size != n_cells:
        raise ValueError("scope_ctx['donor_ids'] length must match X.shape[0].")

    run_compute = p.pipeline_mode in {"compute", "full"}
    run_plot = p.pipeline_mode in {"plot", "full"}
    degenerate_labels = labels is not None and pd.unique(np.asarray(labels)).size < 2

    if run_compute:
        if p.bins_confirm < p.perm_final:
            # no-op guard against accidental type mixups in malformed configs
            pass

        cache_root = Path(cache_dir)
        cache_screen = build_or_load_scope_cache(
            scope_id=f"{scope_id}.screen",
            angles=angles,
            donor_ids=donor_ids,
            bins=p.bins_screen,
            n_perm=p.perm_final,
            seed=p.seed,
            cache_dir=cache_root,
        )
        cache_confirm = build_or_load_scope_cache(
            scope_id=f"{scope_id}.confirm",
            angles=angles,
            donor_ids=donor_ids,
            bins=p.bins_confirm,
            n_perm=p.perm_final,
            seed=p.seed,
            cache_dir=cache_root,
        )

        stage1_df, eligible_idx, gene_meta = _stage1_scores(
            X=X,
            genes=genes,
            labels=labels,
            min_fg=p.min_fg,
            min_prev=p.min_prev,
        )
        if stage1_df.empty:
            stage1_df.to_csv(stage1_path, index=False)
            stage2_df = pd.DataFrame()
            stage3_df = pd.DataFrame()
            reps_df = pd.DataFrame()
            funnel_gene = pd.DataFrame()
        else:
            selected_stage1 = stage1_df[stage1_df["eligible"]].sort_values(
                ["screen_score", "n_fg"],
                ascending=[False, False],
            ).head(p.stage1_top_k)
            stage1_selected_idx = set(selected_stage1["gene_idx"].astype(int).tolist())
            stage1_df["stage1_selected"] = stage1_df["gene_idx"].astype(int).isin(stage1_selected_idx)

            if degenerate_labels:
                logger.info("Scope=%s has degenerate labels; skipping stage2/stage3.", scope_name)
                stage2_df = pd.DataFrame(columns=["gene", "gene_idx", "stage2_selected"])
                stage3_df = pd.DataFrame(columns=["gene", "gene_idx", "p_T", "q_T"])
                reps_df = pd.DataFrame(columns=["gene", "gene_idx"])
                for idx in stage1_selected_idx:
                    gene_meta[int(idx)]["skip_reason"] = "degenerate_labels"
            else:
                stage2_df = _stage2_metrics(
                    X=X,
                    genes=genes,
                    gene_idx=sorted(stage1_selected_idx),
                    cache=cache_screen,
                    min_prev=p.min_prev,
                    gene_meta=gene_meta,
                )
                if not stage2_df.empty:
                    stage2_df = stage2_df.sort_values(
                        ["stage2_score", "strength_proxy", "specificity_proxy"],
                        ascending=[False, False, False],
                    ).reset_index(drop=True)
                    stage2_selected = stage2_df.head(p.stage2_top_k).copy()
                    selected_stage2_idx = set(stage2_selected["gene_idx"].astype(int).tolist())
                    stage2_df["stage2_selected"] = stage2_df["gene_idx"].astype(int).isin(selected_stage2_idx)
                else:
                    selected_stage2_idx = set()
                    stage2_df["stage2_selected"] = False

                stage3_rows: list[dict[str, Any]] = []
                for idx in sorted(selected_stage2_idx):
                    expr = _get_gene_column_dense(X, idx)
                    perm_metrics = _adaptive_permutation(
                        expr=expr,
                        donor_ids=donor_ids,
                        cache=cache_confirm,
                        params=p,
                    )
                    gene_meta[idx]["stage_reached"] = 3
                    gene_meta[idx]["n_perm_final_used"] = int(perm_metrics["n_perm_final_used"])
                    gene_meta[idx]["skip_reason"] = ""
                    stage2_row = stage2_df.loc[stage2_df["gene_idx"] == idx].iloc[0]
                    stage3_rows.append(
                        {
                            "gene": genes[idx],
                            "gene_idx": int(idx),
                            "n_fg": int(gene_meta[idx]["n_fg"]),
                            "prevalence": float(gene_meta[idx]["prevalence"]),
                            "strength_proxy": float(stage2_row["strength_proxy"]),
                            "specificity_proxy": float(stage2_row["specificity_proxy"]),
                            "E_max": float(stage2_row["E_max"]),
                            "phi_max_deg": float(stage2_row["phi_max_deg"]),
                            "peak_count": int(stage2_row["peak_count"]),
                            "entropy": float(stage2_row["entropy"]),
                            "T_obs": float(perm_metrics["T_obs"]),
                            "p_T": float(perm_metrics["p_T"]),
                            "q_T": float("nan"),
                            "coverage_q95": float(perm_metrics["coverage_q95"]),
                            "n_perm_final_used": int(perm_metrics["n_perm_final_used"]),
                        }
                    )
                stage3_df = pd.DataFrame(stage3_rows)
                if not stage3_df.empty:
                    stage3_df["q_T"] = bh_fdr(stage3_df["p_T"].to_numpy(dtype=float))
                reps_df = _select_representatives(
                    stage3_df=stage3_df,
                    plot_top_k=p.plot_top_k,
                    q_threshold=p.q_threshold,
                )

            funnel_gene_rows = []
            for idx, gene in enumerate(genes):
                meta = gene_meta[idx]
                funnel_gene_rows.append(
                    {
                        "gene": gene,
                        "gene_idx": int(idx),
                        "stage_reached": int(meta["stage_reached"]),
                        "skip_reason": str(meta["skip_reason"]),
                        "n_perm_final_used": int(meta["n_perm_final_used"]),
                    }
                )
            funnel_gene = pd.DataFrame(funnel_gene_rows)

        stage1_df.to_csv(stage1_path, index=False)
        stage2_df.to_csv(stage2_path, index=False)
        stage3_df.to_csv(stage3_path, index=False)
        reps_df.to_csv(reps_path, index=False)
        funnel_gene.to_csv(funnel_gene_path, index=False)

        funnel_summary = _funnel_report(stage1_df, stage2_df, stage3_df, reps_df)
        funnel_summary.update(
            {
                "scope_id": scope_id,
                "scope_name": scope_name,
                "bins_screen": p.bins_screen,
                "bins_confirm": p.bins_confirm,
                "perm_init": p.perm_init,
                "perm_mid": p.perm_mid,
                "perm_final": p.perm_final,
                "seed": p.seed,
                "cache_screen_hit": bool(cache_screen.loaded_from_disk),
                "cache_confirm_hit": bool(cache_confirm.loaded_from_disk),
            }
        )
        pd.DataFrame([funnel_summary]).to_csv(funnel_summary_csv, index=False)
        funnel_summary_json.write_text(json.dumps(funnel_summary, indent=2), encoding="utf-8")
    else:
        stage1_df = pd.read_csv(stage1_path) if stage1_path.exists() else pd.DataFrame()
        stage2_df = pd.read_csv(stage2_path) if stage2_path.exists() else pd.DataFrame()
        stage3_df = pd.read_csv(stage3_path) if stage3_path.exists() else pd.DataFrame()
        reps_df = pd.read_csv(reps_path) if reps_path.exists() else pd.DataFrame()
        funnel_gene = pd.read_csv(funnel_gene_path) if funnel_gene_path.exists() else pd.DataFrame()
        cache_root = Path(cache_dir)
        cache_confirm = build_or_load_scope_cache(
            scope_id=f"{scope_id}.confirm",
            angles=angles,
            donor_ids=donor_ids,
            bins=p.bins_confirm,
            n_perm=p.perm_final,
            seed=p.seed,
            cache_dir=cache_root,
        )

    scope_results = {
        "scope_id": scope_id,
        "scope_name": scope_name,
        "stage1": stage1_df,
        "stage2": stage2_df,
        "stage3": stage3_df,
        "representatives": reps_df,
        "funnel_gene": funnel_gene,
        "scope_params": p,
        "out_dir": out_dir,
    }

    if run_plot:
        plot_from_results(
            scope_results=scope_results,
            scope_ctx=scope_ctx,
            X=X,
            genes=genes,
            cache_dir=cache_dir,
            params=params,
        )
    return scope_results


def _plot_umap_overlay(emb: np.ndarray, values: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    sca = ax.scatter(emb[:, 0], emb[:, 1], c=values, s=5, cmap="viridis", linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png.as_posix(), dpi=150)
    plt.close(fig)


def plot_from_results(
    scope_results: dict[str, Any],
    scope_ctx: dict[str, Any],
    X,
    genes: list[str],
    cache_dir: Path | str,
    params: dict[str, Any],
) -> None:
    """Generate representative-only plots from previously cached result tables."""
    del cache_dir
    p = _build_scope_params(scope_ctx, params)
    if p.skip_rsp_plots and p.skip_umap_plots and p.skip_pair_plots:
        return

    reps_df = scope_results.get("representatives")
    if not isinstance(reps_df, pd.DataFrame) or reps_df.empty:
        return
    fig_dir = Path(scope_ctx.get("figure_dir", scope_results["out_dir"]))
    ensure_dir(fig_dir.as_posix())
    figure_dirs_raw = scope_ctx.get("figure_dirs", {})
    rsp_dir = Path(figure_dirs_raw.get("rsp", fig_dir))
    umap_dir = Path(figure_dirs_raw.get("umap", fig_dir))
    pairs_dir = Path(figure_dirs_raw.get("pairs", fig_dir))
    ensure_dir(rsp_dir.as_posix())
    ensure_dir(umap_dir.as_posix())
    ensure_dir(pairs_dir.as_posix())
    angles = np.asarray(scope_ctx["angles"], dtype=float).ravel()
    umap_xy = scope_ctx.get("umap_xy")
    if umap_xy is not None:
        umap_xy = np.asarray(umap_xy, dtype=float)

    bins_confirm = int(params.get("bins_confirm", 72))
    for _, row in reps_df.iterrows():
        gene = str(row["gene"])
        if gene not in genes:
            continue
        idx = int(genes.index(gene))
        expr = _get_gene_column_dense(X, idx)
        f_mask = expr > 0
        try:
            E_phi, _, _ = compute_rsp_profile_from_boolean(f_mask, angles, bins_confirm)
        except ValueError:
            continue

        if not p.skip_rsp_plots:
            plot_rsp_polar(
                E_phi,
                (rsp_dir / f"{gene}_rsp_polar.png").as_posix(),
                f"RSP: {gene}",
            )
        if not p.skip_umap_plots and umap_xy is not None:
            _plot_umap_overlay(
                umap_xy,
                expr,
                umap_dir / f"{gene}_umap.png",
                f"{scope_results['scope_name']}: {gene}",
            )
