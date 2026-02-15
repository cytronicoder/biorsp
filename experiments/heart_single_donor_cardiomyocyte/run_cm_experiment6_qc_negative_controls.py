#!/usr/bin/env python3
"""CM Experiment #6 (single donor): QC/contamination negative controls for BioRSP.

Hypothesis (pre-registered):
A meaningful fraction of apparent localized BioRSP signals in cardiomyocytes can be
explained by technical covariates (library size / mitochondrial / ribosomal content),
ambient RNA contamination, or doublets rather than true biological programs.

Interpretation guardrail:
All directions are representation-conditional embedding geometry, not anatomy.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biorsp.core.compute import compute_rsp_profile_from_boolean
from biorsp.core.features import get_feature_vector, resolve_feature_index
from biorsp.core.geometry import (
    bin_theta,
    compute_theta,
    compute_vantage_point,
    theta_bin_centers,
)
from biorsp.pipeline.hierarchy import _resolve_expr_matrix
from biorsp.plotting.qc import save_numeric_umap
from biorsp.plotting.styles import DEFAULT_PLOT_STYLE, apply_plot_style
from biorsp.stats.permutation import perm_null_T_and_profile
from biorsp.stats.scoring import bh_fdr, coverage_from_null, peak_count, robust_z

DONOR_KEY_CANDIDATES = [
    "donor",
    "donor_id",
    "individual",
    "subject",
    "sample",
    "hubmap_id",
    "dataset",
]

LABEL_KEY_CANDIDATES = [
    "azimuth_label",
    "predicted_label",
    "predicted_CLID",
    "cell_type",
]

CM_PANEL = ["TNNT2", "TNNI3", "MYH6", "MYH7", "RYR2", "ATP2A2", "NPPA", "NPPB"]

AMBIENT_CANDIDATES = [
    "ALB",
    "APOA1",
    "APOA2",
    "APOC1",
    "HP",
    "HBB",
    "HBA1",
    "HBA2",
    "PPBP",
    "PF4",
    "LYZ",
    "S100A8",
    "S100A9",
]

UNDERPOWERED_PREV = 0.005
UNDERPOWERED_MIN_FG = 50
Q_SIG = 0.05
QC_RISK_THRESH = 0.35
SIM_QC_THRESH = 0.70


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    coords: np.ndarray
    params: dict[str, Any]


@dataclass(frozen=True)
class FeatureResolved:
    gene: str
    present: bool
    gene_idx: int | None
    resolved_gene: str
    status: str
    source: str
    symbol_column: str


@dataclass(frozen=True)
class EmbeddingGeom:
    key: str
    coords: np.ndarray
    center_xy: np.ndarray
    theta: np.ndarray
    bin_id: np.ndarray
    bin_counts_total: np.ndarray


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    x = str(value).strip().lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CM Experiment #6: QC/contamination negative controls for BioRSP localization."
    )
    p.add_argument("--h5ad", default="data/processed/HT_pca_umap.h5ad")
    p.add_argument(
        "--out",
        default="experiments/heart_single_donor_cardiomyocyte/results/cm_experiment6_qc_negative_controls",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_perm", type=int, default=300)
    p.add_argument("--n_bins", type=int, default=64)
    p.add_argument("--q", type=float, default=0.10)
    p.add_argument("--n_random_sets", type=int, default=50)
    p.add_argument("--enable_scrublet", type=_str2bool, default=True)
    p.add_argument("--extra_genes_csv", default="")
    p.add_argument("--top_extra", type=int, default=50)
    p.add_argument("--k_pca", type=int, default=50)
    p.add_argument("--layer", default=None)
    p.add_argument("--use_raw", action="store_true")
    p.add_argument("--donor_key", default=None)
    p.add_argument("--label_key", default=None)
    return p.parse_args()


def _save_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)


def _safe_numeric_obs(adata: ad.AnnData, keys: list[str]) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        if key not in adata.obs.columns:
            continue
        vals = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float)
        if int(np.isfinite(vals).sum()) == 0:
            continue
        if np.isnan(vals).any():
            vals = np.where(np.isfinite(vals), vals, float(np.nanmedian(vals)))
        return vals, key
    return None, None


def _resolve_key_required(adata: ad.AnnData, requested: str | None, candidates: list[str], purpose: str) -> str:
    if requested is not None:
        if requested in adata.obs.columns:
            return str(requested)
        raise KeyError(f"Requested {purpose} key '{requested}' not in adata.obs")
    for c in candidates:
        if c in adata.obs.columns:
            return c
    raise KeyError(f"No {purpose} key found. Tried: {', '.join(candidates)}")


def _choose_expression_source(adata: ad.AnnData, layer_arg: str | None, use_raw_arg: bool) -> tuple[Any, Any, str]:
    if layer_arg is not None or use_raw_arg:
        return _resolve_expr_matrix(adata, layer=layer_arg, use_raw=bool(use_raw_arg))
    if "counts" in adata.layers:
        return _resolve_expr_matrix(adata, layer="counts", use_raw=False)
    return _resolve_expr_matrix(adata, layer=None, use_raw=False)


def _is_cm_label(label: str) -> bool:
    x = str(label).strip().lower()
    if "cardio" in x or "cardiomyocyte" in x:
        return True
    tokens = (
        x.replace("/", " ")
        .replace("_", " ")
        .replace("-", " ")
        .replace("(", " ")
        .replace(")", " ")
        .split()
    )
    return "cm" in tokens


def _prepare_embedding_input(adata_cm: ad.AnnData, expr_matrix_cm: Any, expr_source: str) -> tuple[ad.AnnData, str]:
    import scanpy as sc

    adata_embed = ad.AnnData(
        X=expr_matrix_cm.copy() if hasattr(expr_matrix_cm, "copy") else np.array(expr_matrix_cm),
        obs=adata_cm.obs.copy(),
    )
    if expr_source.startswith("layer:counts"):
        sc.pp.normalize_total(adata_embed, target_sum=1e4)
        sc.pp.log1p(adata_embed)
        note = "counts->normalize_total(1e4)->log1p"
    elif expr_source in {"X", "raw"}:
        note = f"{expr_source}_as_is"
    else:
        note = f"{expr_source}_as_is"
    return adata_embed, note


def _compute_fixed_embeddings(adata_embed: ad.AnnData, seed: int, k_pca: int) -> tuple[list[EmbeddingSpec], int]:
    import scanpy as sc

    n_cells, n_vars = adata_embed.n_obs, adata_embed.n_vars
    n_pcs = int(max(2, min(int(k_pca), 50, n_vars - 1, n_cells - 1)))

    sc.pp.pca(adata_embed, n_comps=n_pcs, svd_solver="arpack", random_state=int(seed))
    pca = np.asarray(adata_embed.obsm["X_pca"], dtype=float)

    sc.pp.neighbors(
        adata_embed,
        n_neighbors=30,
        n_pcs=n_pcs,
        use_rep="X_pca",
        random_state=int(seed),
    )
    sc.tl.umap(adata_embed, min_dist=0.1, random_state=0)
    umap = np.asarray(adata_embed.obsm["X_umap"], dtype=float)

    specs = [
        EmbeddingSpec(
            key="pca2d",
            coords=pca[:, :2].copy(),
            params={"n_pcs": n_pcs, "seed": int(seed)},
        ),
        EmbeddingSpec(
            key="umap_repr",
            coords=umap[:, :2].copy(),
            params={"n_neighbors": 30, "min_dist": 0.1, "random_state": 0, "n_pcs": n_pcs},
        ),
    ]
    return specs, n_pcs


def _resolve_gene(adata_like: Any, gene: str) -> FeatureResolved:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            idx, label, symbol_col, source = resolve_feature_index(adata_like, gene)
        return FeatureResolved(
            gene=gene,
            present=True,
            gene_idx=int(idx),
            resolved_gene=str(label),
            status="resolved",
            source=str(source),
            symbol_column=str(symbol_col or ""),
        )
    except KeyError:
        return FeatureResolved(
            gene=gene,
            present=False,
            gene_idx=None,
            resolved_gene="",
            status="missing",
            source="",
            symbol_column="",
        )


def _safe_spearman(x: np.ndarray, y: np.ndarray | None) -> float:
    if y is None:
        return float("nan")
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(xv) & np.isfinite(yv)
    if int(m.sum()) < 3:
        return float("nan")
    xs = xv[m]
    ys = yv[m]
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return float("nan")
    rho = spearmanr(xs, ys, nan_policy="omit").correlation
    if rho is None or not np.isfinite(float(rho)):
        return float("nan")
    return float(rho)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= 1e-12 or ny <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def _top_q_mask(values: np.ndarray, q: float) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = int(max(1, round(float(q) * n)))
    order = np.argsort(x, kind="mergesort")
    keep = order[-k:]
    out = np.zeros(n, dtype=bool)
    out[keep] = True
    return out


def _to_dense(mat: Any) -> np.ndarray:
    if sp.issparse(mat):
        return mat.toarray().astype(float)
    return np.asarray(mat, dtype=float)


def _resolve_layer_name(expr_source: str) -> str | None:
    if expr_source.startswith("layer:"):
        return expr_source.split(":", 1)[1]
    return None


def _integer_like_counts(arr: np.ndarray, n_check: int = 20000) -> bool:
    x = np.asarray(arr, dtype=float).ravel()
    if x.size == 0:
        return False
    if x.size > n_check:
        idx = np.linspace(0, x.size - 1, n_check, dtype=int)
        x = x[idx]
    return bool(np.allclose(x, np.round(x), atol=1e-8))


def _build_embedding_geoms(specs: list[EmbeddingSpec], n_bins: int) -> dict[str, EmbeddingGeom]:
    out: dict[str, EmbeddingGeom] = {}
    for emb in specs:
        center = compute_vantage_point(emb.coords, method="mean")
        theta = compute_theta(emb.coords, center)
        _, bin_id = bin_theta(theta, bins=int(n_bins))
        bin_counts = np.bincount(bin_id, minlength=int(n_bins)).astype(float)
        out[emb.key] = EmbeddingGeom(
            key=emb.key,
            coords=emb.coords,
            center_xy=center,
            theta=theta,
            bin_id=bin_id,
            bin_counts_total=bin_counts,
        )
    return out


def _score_foreground(
    *,
    fg: np.ndarray,
    geom: EmbeddingGeom,
    n_bins: int,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    mask = np.asarray(fg, dtype=bool).ravel()
    n_cells = int(mask.size)
    n_fg = int(mask.sum())
    prev = float(n_fg / max(1, n_cells))
    underpowered = bool((prev < UNDERPOWERED_PREV) or (n_fg < UNDERPOWERED_MIN_FG))

    if n_fg == 0 or n_fg == n_cells:
        return {
            "prev": prev,
            "n_fg": n_fg,
            "n_cells": n_cells,
            "T_obs": 0.0,
            "p_T": np.nan,
            "Z_T": np.nan,
            "coverage_C": np.nan,
            "peaks_K": np.nan,
            "phi_hat_deg": np.nan,
            "underpowered_flag": True,
            "E_obs": np.zeros(int(n_bins), dtype=float),
            "null_E": None,
            "null_T": None,
        }

    e_obs, _, _, _ = compute_rsp_profile_from_boolean(
        mask,
        geom.theta,
        int(n_bins),
        bin_id=geom.bin_id,
        bin_counts_total=geom.bin_counts_total,
    )
    t_obs_raw = float(np.max(np.abs(e_obs)))
    phi_idx = int(np.argmax(np.abs(e_obs)))
    phi_hat = float(np.degrees(theta_bin_centers(int(n_bins))[phi_idx]) % 360.0)

    if underpowered:
        return {
            "prev": prev,
            "n_fg": n_fg,
            "n_cells": n_cells,
            "T_obs": t_obs_raw,
            "p_T": np.nan,
            "Z_T": np.nan,
            "coverage_C": np.nan,
            "peaks_K": np.nan,
            "phi_hat_deg": phi_hat,
            "underpowered_flag": True,
            "E_obs": e_obs,
            "null_E": None,
            "null_T": None,
        }

    perm = perm_null_T_and_profile(
        expr=mask.astype(float),
        theta=geom.theta,
        donor_ids=None,
        n_bins=int(n_bins),
        n_perm=int(n_perm),
        seed=int(seed),
        donor_stratified=False,
        bin_id=geom.bin_id,
        bin_counts_total=geom.bin_counts_total,
    )

    null_e = np.asarray(perm["null_E_phi"], dtype=float)
    null_t = np.asarray(perm["null_T"], dtype=float)
    t_obs = float(perm["T_obs"])
    p_t = float(perm["p_T"])

    return {
        "prev": prev,
        "n_fg": n_fg,
        "n_cells": n_cells,
        "T_obs": t_obs,
        "p_T": p_t,
        "Z_T": float(robust_z(t_obs, null_t)),
        "coverage_C": float(coverage_from_null(e_obs, null_e, q=0.95)),
        "peaks_K": float(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95)),
        "phi_hat_deg": phi_hat,
        "underpowered_flag": False,
        "E_obs": e_obs,
        "null_E": null_e,
        "null_T": null_t,
    }


def _ensure_qc_metrics(adata_cm: ad.AnnData, layer_for_qc: str | None) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    import scanpy as sc

    warn_log: list[str] = []
    info: dict[str, str] = {}

    var_names = pd.Index(adata_cm.var_names).astype(str)
    upper = var_names.str.upper()
    mito_mask = np.asarray(
        upper.str.startswith("MT-") | var_names.str.startswith("mt-"),
        dtype=bool,
    )
    ribo_mask = np.asarray(
        upper.str.startswith("RPL")
        | upper.str.startswith("RPS")
        | var_names.str.startswith("Rpl")
        | var_names.str.startswith("Rps"),
        dtype=bool,
    )
    adata_cm.var["mito"] = mito_mask
    adata_cm.var["ribo"] = ribo_mask

    have_total = "total_counts" in adata_cm.obs.columns
    have_ngenes = "n_genes_by_counts" in adata_cm.obs.columns
    have_mt = ("pct_counts_mt" in adata_cm.obs.columns) or ("pct_counts_mito" in adata_cm.obs.columns)
    have_ribo = "pct_counts_ribo" in adata_cm.obs.columns

    if not (have_total and have_ngenes and have_mt and have_ribo):
        qc_vars = []
        if bool(np.sum(adata_cm.var["mito"].to_numpy(dtype=bool))) > 0:
            qc_vars.append("mito")
        if bool(np.sum(adata_cm.var["ribo"].to_numpy(dtype=bool))) > 0:
            qc_vars.append("ribo")
        try:
            sc.pp.calculate_qc_metrics(
                adata_cm,
                qc_vars=qc_vars,
                inplace=True,
                layer=layer_for_qc,
                log1p=False,
                percent_top=None,
            )
            info["calculate_qc_metrics"] = f"ran(layer={layer_for_qc})"
        except Exception as exc:  # pragma: no cover - defensive runtime path.
            msg = f"scanpy.pp.calculate_qc_metrics failed: {exc}"
            warn_log.append(msg)
            info["calculate_qc_metrics"] = f"failed:{exc}"
    else:
        info["calculate_qc_metrics"] = "not_needed"

    # Harmonize aliases.
    if "pct_counts_mt" not in adata_cm.obs.columns and "pct_counts_mito" in adata_cm.obs.columns:
        adata_cm.obs["pct_counts_mt"] = pd.to_numeric(adata_cm.obs["pct_counts_mito"], errors="coerce")
        info["pct_counts_mt_alias"] = "pct_counts_mito"
    if "total_counts" not in adata_cm.obs.columns:
        arr, key = _safe_numeric_obs(adata_cm, ["n_counts", "total_umis", "nUMI"])
        if arr is not None and key is not None:
            adata_cm.obs["total_counts"] = arr
            info["total_counts_alias"] = key

    # Build summary table.
    rows = []
    for col in ["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo"]:
        if col not in adata_cm.obs.columns:
            rows.append(
                {
                    "metric": col,
                    "available": False,
                    "source": info.get(f"{col}_alias", "missing"),
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "missing_frac": 1.0,
                }
            )
            continue
        vals = pd.to_numeric(adata_cm.obs[col], errors="coerce").to_numpy(dtype=float)
        miss = float(np.mean(~np.isfinite(vals)))
        if np.isnan(vals).any():
            vals = np.where(np.isfinite(vals), vals, np.nanmedian(vals[np.isfinite(vals)]) if np.isfinite(vals).any() else 0.0)
        rows.append(
            {
                "metric": col,
                "available": True,
                "source": info.get(f"{col}_alias", "obs"),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "missing_frac": miss,
            }
        )
    return pd.DataFrame(rows), info, warn_log


def _try_scrublet(
    *,
    adata_cm: ad.AnnData,
    expr_matrix_cm: Any,
    expr_source: str,
    seed: int,
    enable_scrublet: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, str, str | None]:
    if not bool(enable_scrublet):
        return None, None, "disabled", None

    layer_name = _resolve_layer_name(expr_source)
    if layer_name != "counts":
        return None, None, "skipped_non_counts", "Scrublet skipped: counts layer not selected as expression source."

    try:
        import scanpy.external as sce
    except Exception as exc:  # pragma: no cover
        return None, None, f"unavailable:{exc}", "scanpy.external.scrublet unavailable."

    try:
        # Scrublet expects count-like matrix in X.
        x_counts = expr_matrix_cm.copy() if hasattr(expr_matrix_cm, "copy") else np.asarray(expr_matrix_cm)
        ad_scr = ad.AnnData(
            X=x_counts,
            obs=adata_cm.obs.copy(),
            var=adata_cm.var.copy(),
        )
        sce.pp.scrublet(ad_scr, expected_doublet_rate=0.05, random_state=int(seed), verbose=False)
        score = pd.to_numeric(ad_scr.obs.get("doublet_score"), errors="coerce").to_numpy(dtype=float)
        pred = ad_scr.obs.get("predicted_doublet")
        pred_arr = None
        if pred is not None:
            pred_arr = pd.Series(pred).astype(bool).to_numpy(dtype=bool)
        if score is None or int(np.isfinite(score).sum()) == 0:
            return None, None, "failed_no_score", "Scrublet ran but did not produce usable doublet_score."
        if np.isnan(score).any():
            score = np.where(np.isfinite(score), score, float(np.nanmedian(score[np.isfinite(score)])))
        return score, pred_arr, "ok", None
    except Exception as exc:  # pragma: no cover
        return None, None, f"failed:{exc}", f"Scrublet failed: {exc}"


def _ambient_score(expr_matrix: Any, indices: list[int]) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros(expr_matrix.shape[0], dtype=float)
    mat = expr_matrix[:, indices]
    arr = _to_dense(mat)
    return np.mean(np.log1p(np.maximum(arr, 0.0)), axis=1).ravel().astype(float)


def _random_sets_detection_matched(
    *,
    expr_matrix: Any,
    ambient_indices: list[int],
    n_sets: int,
    seed: int,
    n_bins_match: int = 10,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    n_cells, n_genes = int(expr_matrix.shape[0]), int(expr_matrix.shape[1])
    if n_genes <= 0:
        return [], np.zeros((0, n_cells), dtype=float), np.zeros(0, dtype=float)

    if sp.issparse(expr_matrix):
        det = np.asarray((expr_matrix > 0).sum(axis=0)).ravel().astype(float) / max(1.0, float(n_cells))
    else:
        det = (np.asarray(expr_matrix, dtype=float) > 0).mean(axis=0).astype(float)

    q_edges = np.quantile(det, np.linspace(0.0, 1.0, int(n_bins_match) + 1))
    q_edges = np.unique(q_edges)
    if q_edges.size <= 2:
        bins = np.zeros(n_genes, dtype=int)
        n_bins_used = 1
    else:
        bins = np.digitize(det, q_edges[1:-1], right=True).astype(int)
        n_bins_used = int(np.max(bins)) + 1

    ambient_idx = np.asarray(ambient_indices, dtype=int)
    if ambient_idx.size == 0:
        return [], np.zeros((0, n_cells), dtype=float), np.zeros(0, dtype=float)

    ambient_bins = bins[ambient_idx]
    all_idx = np.arange(n_genes, dtype=int)
    non_ambient_mask = np.ones(n_genes, dtype=bool)
    non_ambient_mask[ambient_idx] = False

    pool_by_bin: dict[int, np.ndarray] = {}
    for b in range(n_bins_used):
        cand = all_idx[(bins == b) & non_ambient_mask]
        pool_by_bin[b] = cand
    global_pool = all_idx[non_ambient_mask]

    rng = np.random.default_rng(int(seed))
    sets: list[list[int]] = []
    scores = np.zeros((int(n_sets), n_cells), dtype=float)
    mean_det = np.zeros(int(n_sets), dtype=float)

    for i in range(int(n_sets)):
        choice: list[int] = []
        for b in ambient_bins.tolist():
            pool = pool_by_bin.get(int(b), np.zeros(0, dtype=int))
            if pool.size == 0:
                pick = int(rng.choice(global_pool))
            else:
                pick = int(rng.choice(pool))
            choice.append(pick)
        sets.append(choice)
        mean_det[i] = float(np.mean(det[np.asarray(choice, dtype=int)]))
        scores[i, :] = _ambient_score(expr_matrix, choice)

    return sets, scores, mean_det


def _plot_polar_with_null(
    *,
    out_png: Path,
    title: str,
    e_obs: np.ndarray,
    null_e: np.ndarray | None,
    t_obs: float,
    null_t: np.ndarray | None,
    stats_text: str,
) -> None:
    fig = plt.figure(figsize=(10.8, 4.8))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax2 = fig.add_subplot(1, 2, 2)

    n_bins = int(np.asarray(e_obs).size)
    th_cent = theta_bin_centers(n_bins)
    th = np.concatenate([th_cent, th_cent[:1]])
    obs = np.concatenate([np.asarray(e_obs, dtype=float), np.asarray(e_obs[:1], dtype=float)])

    ax1.plot(th, obs, color="#8B0000", linewidth=2.2, label="obs")
    if null_e is not None and np.asarray(null_e).size > 0:
        ne = np.asarray(null_e, dtype=float)
        hi = np.quantile(ne, 0.95, axis=0)
        lo = np.quantile(ne, 0.05, axis=0)
        ax1.plot(th, np.concatenate([hi, hi[:1]]), color="#333333", linestyle="--", linewidth=1.2, label="null 95%")
        ax1.plot(th, np.concatenate([lo, lo[:1]]), color="#333333", linestyle="--", linewidth=1.0, label="null 5%")
        ax1.fill_between(th, np.concatenate([lo, lo[:1]]), np.concatenate([hi, hi[:1]]), color="#B9B9B9", alpha=0.18)
    ax1.set_theta_zero_location("E")
    ax1.set_theta_direction(1)
    ax1.set_thetagrids(np.arange(0, 360, 90))
    ax1.set_title("RSP profile")
    ax1.legend(loc="upper right", bbox_to_anchor=(1.2, 1.18), fontsize=8, frameon=True)
    ax1.text(
        0.02,
        0.02,
        stats_text,
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#999", "alpha": 0.85},
    )

    if null_t is None or np.asarray(null_t).size == 0:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "No null distribution", ha="center", va="center")
    else:
        nt = np.asarray(null_t, dtype=float)
        bins = int(min(45, max(12, np.ceil(np.sqrt(nt.size)))))
        ax2.hist(nt, bins=bins, color="#729FCF", edgecolor="white", alpha=0.9)
        ax2.axvline(float(t_obs), color="#8B0000", linestyle="--", linewidth=2.0)
        ax2.set_xlabel("null_T")
        ax2.set_ylabel("count")
        ax2.set_title("Null T distribution")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(
    *,
    out_png: Path,
    mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if mat.size == 0:
        _save_placeholder(out_png, title, "No data")
        return
    fig, ax = plt.subplots(figsize=(1.0 * len(col_labels) + 3.0, 0.36 * len(row_labels) + 2.4))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)


def _read_extra_genes(path: str, top_n: int) -> list[str]:
    if str(path).strip() == "":
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"extra_genes_csv not found: {p}")
    df = pd.read_csv(p)
    if "gene" in df.columns:
        genes = df["gene"].astype(str).tolist()
    else:
        genes = df.iloc[:, 0].astype(str).tolist()
    out = []
    seen = set()
    for g in genes:
        gg = g.strip()
        if gg == "" or gg in seen:
            continue
        seen.add(gg)
        out.append(gg)
        if len(out) >= int(top_n):
            break
    return out


def _write_readme(
    out_path: Path,
    *,
    args: argparse.Namespace,
    donor_key: str,
    label_key: str,
    donor_star: str,
    expr_source: str,
    embed_note: str,
    n_cells_donor: int,
    n_cells_cm: int,
    cm_labels: dict[str, int],
    qc_metric_sources: dict[str, str],
    scrublet_status: str,
    scrublet_note: str | None,
    ambient_present_genes: list[str],
    ambient_emp_rows: pd.DataFrame,
    flags_df: pd.DataFrame,
    warnings_log: list[str],
) -> None:
    lines: list[str] = []
    lines.append("CM Experiment #6 (Single-donor): QC/contamination negative controls and confounding diagnosis")
    lines.append("")
    lines.append("Hypothesis")
    lines.append(
        "A substantial fraction of apparent localized BioRSP signals in cardiomyocytes can be explained by technical "
        "covariates (library size / mitochondrial / ribosomal content), ambient RNA contamination, or doublets."
    )
    lines.append("")
    lines.append("Inference setup")
    lines.append("- Single donor only (donor_star selected by max cardiomyocyte count).")
    lines.append("- Localized signal based on permutation-calibrated q<=0.05 under top-quantile foreground.")
    lines.append("- QC-driven flag requires localized signal and (qc_risk>=0.35 OR profile_qc_risk>=0.70).")
    lines.append("")
    lines.append("Run metadata")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- n_perm: {args.n_perm}")
    lines.append(f"- n_bins: {args.n_bins}")
    lines.append(f"- q: {args.q}")
    lines.append(f"- n_random_sets: {args.n_random_sets}")
    lines.append(f"- donor_key_used: {donor_key}")
    lines.append(f"- label_key_used: {label_key}")
    lines.append(f"- donor_star: {donor_star}")
    lines.append(f"- expression_source_used: {expr_source}")
    lines.append(f"- embedding_input_prep: {embed_note}")
    lines.append(f"- donor_star_total_cells: {n_cells_donor}")
    lines.append(f"- donor_star_cardiomyocytes: {n_cells_cm}")
    lines.append("")
    lines.append("Cardiomyocyte labels included")
    for lab, count in cm_labels.items():
        lines.append(f"- {lab}: {count}")
    lines.append("")
    lines.append("QC metric sources")
    for k, v in qc_metric_sources.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(f"Scrublet status: {scrublet_status}")
    if scrublet_note:
        lines.append(f"Scrublet note: {scrublet_note}")
    lines.append("")
    lines.append("Ambient panel genes present")
    if len(ambient_present_genes) == 0:
        lines.append("- none")
    else:
        lines.append("- " + ", ".join(ambient_present_genes))
    lines.append("")
    if not ambient_emp_rows.empty:
        lines.append("Ambient random-set empirical p-values")
        for _, r in ambient_emp_rows.iterrows():
            lines.append(
                f"- {r['embedding']}: ambient_Z={float(r['ambient_Z_T']):.3f}, "
                f"empirical_p={float(r['empirical_p']):.4f}"
            )
        lines.append("")

    if not flags_df.empty:
        n_flag = int(np.sum(flags_df["qc_driven"].to_numpy(dtype=bool)))
        lines.append(f"QC-driven genes flagged: {n_flag}/{len(flags_df)}")

    if warnings_log:
        lines.append("")
        lines.append("Warnings")
        for w in warnings_log:
            lines.append(f"- {w}")

    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "QC-driven localization indicates representation-localized structure plausibly attributable to technical/ambient "
        "factors; it should not be interpreted as a biological cardiomyocyte program without stronger controls."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    apply_plot_style()

    out_root = Path(args.out)
    tables_dir = out_root / "tables"
    plots_dir = out_root / "plots"
    for d in [
        tables_dir,
        plots_dir / "00_overview",
        plots_dir / "01_qc_distributions",
        plots_dir / "02_qc_pseudofeature_rsp",
        plots_dir / "03_gene_vs_qc_association",
        plots_dir / "04_profile_similarity",
        plots_dir / "05_ambient_controls",
        plots_dir / "06_flagged_exemplars",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    warnings_log: list[str] = []

    adata = ad.read_h5ad(args.h5ad)

    donor_key = _resolve_key_required(adata, args.donor_key, DONOR_KEY_CANDIDATES, purpose="donor")
    label_key = _resolve_key_required(adata, args.label_key, LABEL_KEY_CANDIDATES, purpose="label")

    labels_all = adata.obs[label_key].astype("string").fillna("NA").astype(str)
    donor_ids_all = adata.obs[donor_key].astype("string").fillna("NA").astype(str)
    cm_mask_all = labels_all.map(_is_cm_label).to_numpy(dtype=bool)
    if int(cm_mask_all.sum()) == 0:
        raise RuntimeError("No cardiomyocyte cells detected by label matching")

    donor_choice = (
        pd.DataFrame({"donor_id": donor_ids_all.to_numpy(), "is_cm": cm_mask_all})
        .groupby("donor_id", as_index=False)
        .agg(n_cells_total=("is_cm", "size"), n_cm=("is_cm", "sum"))
        .sort_values(by=["n_cm", "n_cells_total", "donor_id"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    donor_star = str(donor_choice.iloc[0]["donor_id"])
    donor_choice["is_donor_star"] = donor_choice["donor_id"].astype(str) == donor_star
    donor_choice.to_csv(tables_dir / "donor_choice.csv", index=False)

    donor_mask = donor_ids_all.astype(str).to_numpy() == donor_star
    adata_donor = adata[donor_mask].copy()
    labels_donor = adata_donor.obs[label_key].astype("string").fillna("NA").astype(str)
    cm_mask_donor = labels_donor.map(_is_cm_label).to_numpy(dtype=bool)
    adata_cm = adata_donor[cm_mask_donor].copy()
    if int(adata_cm.n_obs) == 0:
        raise RuntimeError("donor_star cardiomyocyte subset is empty")

    cm_label_counts = labels_donor.loc[cm_mask_donor].value_counts().sort_index()
    cm_labels_included = cm_label_counts.index.astype(str).tolist()
    print("cm_labels_included=" + ", ".join(cm_labels_included))
    for lab, ct in cm_label_counts.items():
        print(f"cm_label_count[{lab}]={int(ct)}")

    expr_matrix_cm, adata_like_cm, expr_source = _choose_expression_source(
        adata_cm, layer_arg=args.layer, use_raw_arg=bool(args.use_raw)
    )

    expr_dense_probe = _to_dense(expr_matrix_cm[:, : min(32, int(expr_matrix_cm.shape[1]))])
    integer_like = _integer_like_counts(expr_dense_probe)
    if expr_source in {"X", "raw"} and not integer_like:
        msg = "Expression source appears log/continuous; absolute-count interpretation is limited."
        warnings_log.append(msg)
        print(f"WARNING: {msg}")

    adata_embed, embed_note = _prepare_embedding_input(adata_cm, expr_matrix_cm, expr_source)
    embeddings, n_pcs_used = _compute_fixed_embeddings(adata_embed, seed=int(args.seed), k_pca=int(args.k_pca))
    geom_map = _build_embedding_geoms(embeddings, n_bins=int(args.n_bins))

    # QC metrics.
    layer_for_qc = _resolve_layer_name(expr_source)
    qc_metrics_df, qc_info, qc_warns = _ensure_qc_metrics(adata_cm, layer_for_qc)
    warnings_log.extend(qc_warns)

    total_counts, total_key = _safe_numeric_obs(adata_cm, ["total_counts", "n_counts", "total_umis", "nUMI"])
    n_genes_by_counts, ng_key = _safe_numeric_obs(adata_cm, ["n_genes_by_counts"])
    pct_mt, mt_key = _safe_numeric_obs(adata_cm, ["pct_counts_mt", "pct_counts_mito", "percent.mt", "pct_mt"])
    pct_ribo, ribo_key = _safe_numeric_obs(adata_cm, ["pct_counts_ribo", "percent.ribo", "pct_ribo"])

    qc_metric_sources = {
        "total_counts": str(total_key),
        "n_genes_by_counts": str(ng_key),
        "pct_counts_mt": str(mt_key),
        "pct_counts_ribo": str(ribo_key),
        "calculate_qc_metrics": qc_info.get("calculate_qc_metrics", "na"),
    }
    qc_metrics_df.to_csv(tables_dir / "qc_metrics_summary.csv", index=False)

    # Optional Scrublet.
    doublet_score, predicted_doublet, scrublet_status, scrublet_note = _try_scrublet(
        adata_cm=adata_cm,
        expr_matrix_cm=expr_matrix_cm,
        expr_source=expr_source,
        seed=int(args.seed),
        enable_scrublet=bool(args.enable_scrublet),
    )
    if scrublet_note is not None:
        warnings_log.append(scrublet_note)
        print(f"WARNING: {scrublet_note}")

    if doublet_score is not None:
        adata_cm.obs["doublet_score"] = doublet_score
    if predicted_doublet is not None:
        adata_cm.obs["predicted_doublet"] = predicted_doublet

    # Resolve genes.
    extra_genes = _read_extra_genes(str(args.extra_genes_csv), top_n=int(args.top_extra))
    panel_genes = list(CM_PANEL) + extra_genes
    panel_genes = list(dict.fromkeys(panel_genes))

    resolved_panel = [_resolve_gene(adata_like_cm, g) for g in panel_genes]
    genes_present = [r for r in resolved_panel if r.present and r.gene_idx is not None]
    if len(genes_present) == 0:
        raise RuntimeError("No biological panel genes resolved")

    # Ambient genes and score.
    resolved_ambient = [_resolve_gene(adata_like_cm, g) for g in AMBIENT_CANDIDATES]
    ambient_present = [r for r in resolved_ambient if r.present and r.gene_idx is not None]
    ambient_indices = [int(r.gene_idx) for r in ambient_present]
    ambient_score = _ambient_score(expr_matrix_cm, ambient_indices)

    # Random matched sets for ambient control.
    random_sets, random_scores, random_mean_det = _random_sets_detection_matched(
        expr_matrix=expr_matrix_cm,
        ambient_indices=ambient_indices,
        n_sets=int(args.n_random_sets),
        seed=int(args.seed + 333),
        n_bins_match=10,
    )

    # Build feature values.
    feature_values: dict[str, np.ndarray] = {}
    feature_types: dict[str, str] = {}

    for r in genes_present:
        feature_values[r.gene] = get_feature_vector(expr_matrix_cm, int(r.gene_idx))
        feature_types[r.gene] = "gene"

    if total_counts is not None:
        feature_values["total_counts"] = np.asarray(total_counts, dtype=float)
        feature_types["total_counts"] = "qc"
    if pct_mt is not None:
        feature_values["pct_counts_mt"] = np.asarray(pct_mt, dtype=float)
        feature_types["pct_counts_mt"] = "qc"
    if pct_ribo is not None:
        feature_values["pct_counts_ribo"] = np.asarray(pct_ribo, dtype=float)
        feature_types["pct_counts_ribo"] = "qc"

    feature_values["ambient_score"] = np.asarray(ambient_score, dtype=float)
    feature_types["ambient_score"] = "ambient"

    if doublet_score is not None:
        feature_values["doublet_score"] = np.asarray(doublet_score, dtype=float)
        feature_types["doublet_score"] = "doublet"

    for i in range(random_scores.shape[0]):
        name = f"random_set_{i+1:03d}"
        feature_values[name] = random_scores[i, :]
        feature_types[name] = "randomset"

    # Scoring loop.
    rows: list[dict[str, Any]] = []
    detail_map: dict[tuple[str, str, str], dict[str, Any]] = {}

    scoring_plan: list[tuple[str, str]] = []
    for name, ftype in feature_types.items():
        if ftype == "gene":
            scoring_plan.append((name, "F0_detect"))
            scoring_plan.append((name, "F1_topq"))
        else:
            scoring_plan.append((name, "F1_topq"))

    total_tests = len(scoring_plan) * len(embeddings)
    test_count = 0

    for emb_i, emb in enumerate(embeddings):
        geom = geom_map[emb.key]
        for feat_i, (name, fg_mode) in enumerate(scoring_plan):
            vals = np.asarray(feature_values[name], dtype=float)
            if fg_mode == "F0_detect":
                fg = vals > 0.0
                q_param = np.nan
            else:
                fg = _top_q_mask(vals, q=float(args.q))
                q_param = float(args.q)

            score = _score_foreground(
                fg=fg,
                geom=geom,
                n_bins=int(args.n_bins),
                n_perm=int(args.n_perm),
                seed=int(args.seed + 100000 + emb_i * 10000 + feat_i * 17),
            )

            row = {
                "feature_name": name,
                "feature_type": feature_types[name],
                "embedding": emb.key,
                "foreground_mode": fg_mode,
                "q_param": q_param,
                "prev": score["prev"],
                "n_fg": score["n_fg"],
                "n_cells": score["n_cells"],
                "T_obs": score["T_obs"],
                "p_T": score["p_T"],
                "q_T": np.nan,
                "Z_T": score["Z_T"],
                "coverage_C": score["coverage_C"],
                "peaks_K": score["peaks_K"],
                "phi_hat_deg": score["phi_hat_deg"],
                "underpowered_flag": score["underpowered_flag"],
            }
            rows.append(row)

            # Store profiles for QC pseudo-features and genes in topq mode.
            if feature_types[name] in {"qc", "ambient", "doublet"} or (feature_types[name] == "gene" and fg_mode == "F1_topq"):
                detail_map[(emb.key, name, fg_mode)] = {
                    "E_obs": score["E_obs"],
                    "null_E": score["null_E"],
                    "null_T": score["null_T"],
                    "T_obs": score["T_obs"],
                    "values": vals,
                }

            test_count += 1
            if test_count % 50 == 0 or test_count == total_tests:
                print(f"[Scoring] {test_count}/{total_tests} tests completed")
                pd.DataFrame(rows).to_csv(tables_dir / "feature_scores_long.intermediate.csv", index=False)

    feature_scores = pd.DataFrame(rows)

    # BH within embedding x foreground x feature_type.
    if not feature_scores.empty:
        qvals = np.full(len(feature_scores), np.nan, dtype=float)
        for _, idx in feature_scores.groupby(["embedding", "foreground_mode", "feature_type"], sort=False).groups.items():
            part = feature_scores.loc[idx, "p_T"].to_numpy(dtype=float)
            fin = np.isfinite(part)
            if int(fin.sum()) == 0:
                continue
            qq = np.full(part.shape, np.nan, dtype=float)
            qq[fin] = bh_fdr(part[fin])
            qvals[np.asarray(list(idx), dtype=int)] = qq
        feature_scores["q_T"] = qvals

    feature_scores.to_csv(tables_dir / "feature_scores_long.csv", index=False)

    # QC pseudo-feature profile table.
    prof_rows = []
    for (emb, feat, fg_mode), det in detail_map.items():
        ftype = feature_types.get(feat, "")
        if ftype not in {"qc", "ambient", "doublet"}:
            continue
        sc_row = feature_scores.loc[
            (feature_scores["embedding"] == emb)
            & (feature_scores["feature_name"] == feat)
            & (feature_scores["foreground_mode"] == fg_mode)
        ]
        if sc_row.empty:
            continue
        rr = sc_row.iloc[0]
        prof_rows.append(
            {
                "embedding": emb,
                "feature_name": feat,
                "feature_type": ftype,
                "foreground_mode": fg_mode,
                "T_obs": float(rr["T_obs"]) if np.isfinite(float(rr["T_obs"])) else np.nan,
                "p_T": float(rr["p_T"]) if np.isfinite(float(rr["p_T"])) else np.nan,
                "q_T": float(rr["q_T"]) if np.isfinite(float(rr["q_T"])) else np.nan,
                "Z_T": float(rr["Z_T"]) if np.isfinite(float(rr["Z_T"])) else np.nan,
                "coverage_C": float(rr["coverage_C"]) if np.isfinite(float(rr["coverage_C"])) else np.nan,
                "peaks_K": float(rr["peaks_K"]) if np.isfinite(float(rr["peaks_K"])) else np.nan,
                "phi_hat_deg": float(rr["phi_hat_deg"]) if np.isfinite(float(rr["phi_hat_deg"])) else np.nan,
                "E_phi_json": json.dumps(np.asarray(det["E_obs"], dtype=float).tolist()),
            }
        )
    qc_profile_df = pd.DataFrame(prof_rows)
    qc_profile_df.to_csv(tables_dir / "qc_pseudofeature_profiles.csv", index=False)

    # Gene QC association (full data, topq foreground).
    qc_covars: dict[str, np.ndarray | None] = {
        "total_counts": np.asarray(total_counts, dtype=float) if total_counts is not None else None,
        "pct_counts_mt": np.asarray(pct_mt, dtype=float) if pct_mt is not None else None,
        "pct_counts_ribo": np.asarray(pct_ribo, dtype=float) if pct_ribo is not None else None,
        "ambient_score": np.asarray(ambient_score, dtype=float),
        "doublet_score": np.asarray(doublet_score, dtype=float) if doublet_score is not None else None,
    }

    assoc_rows = []
    for r in genes_present:
        vals = np.asarray(feature_values[r.gene], dtype=float)
        fg = _top_q_mask(vals, q=float(args.q)).astype(float)
        rho_total = _safe_spearman(fg, qc_covars["total_counts"])
        rho_mt = _safe_spearman(fg, qc_covars["pct_counts_mt"])
        rho_ribo = _safe_spearman(fg, qc_covars["pct_counts_ribo"])
        rho_amb = _safe_spearman(fg, qc_covars["ambient_score"])
        rho_dbl = _safe_spearman(fg, qc_covars["doublet_score"])
        vals_rho = np.array([rho_total, rho_mt, rho_ribo, rho_amb, rho_dbl], dtype=float)
        fin = vals_rho[np.isfinite(vals_rho)]
        qc_risk = float(np.max(np.abs(fin))) if fin.size > 0 else 0.0
        assoc_rows.append(
            {
                "gene": r.gene,
                "rho_total_counts": rho_total,
                "rho_pct_counts_mt": rho_mt,
                "rho_pct_counts_ribo": rho_ribo,
                "rho_ambient_score": rho_amb,
                "rho_doublet_score": rho_dbl,
                "qc_risk": qc_risk,
            }
        )
    assoc_df = pd.DataFrame(assoc_rows)
    assoc_df.to_csv(tables_dir / "gene_qc_association.csv", index=False)

    # Profile similarity genes vs QC pseudo-features.
    sim_rows = []
    qc_features_for_sim = ["total_counts", "pct_counts_mt", "pct_counts_ribo", "ambient_score", "doublet_score"]
    for emb in [e.key for e in embeddings]:
        for r in genes_present:
            gene_key = (emb, r.gene, "F1_topq")
            if gene_key not in detail_map:
                continue
            gprof = np.asarray(detail_map[gene_key]["E_obs"], dtype=float)
            sims = {}
            for qf in qc_features_for_sim:
                q_key = (emb, qf, "F1_topq")
                if q_key not in detail_map:
                    sims[f"sim_{qf}"] = np.nan
                    continue
                sims[f"sim_{qf}"] = _cosine_similarity(gprof, np.asarray(detail_map[q_key]["E_obs"], dtype=float))
            sim_vals = np.array([sims.get(f"sim_{x}", np.nan) for x in qc_features_for_sim], dtype=float)
            fin = sim_vals[np.isfinite(sim_vals)]
            risk = float(np.max(fin)) if fin.size > 0 else np.nan

            conf_name = "none"
            if fin.size > 0:
                sim_pairs = [(k, v) for k, v in sims.items() if np.isfinite(v)]
                if len(sim_pairs) > 0:
                    conf_name = str(max(sim_pairs, key=lambda kv: kv[1])[0])

            sim_rows.append(
                {
                    "gene": r.gene,
                    "embedding": emb,
                    **sims,
                    "profile_qc_risk": risk,
                    "top_profile_confounder": conf_name,
                }
            )
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(tables_dir / "gene_profile_similarity.csv", index=False)

    # QC-driven flags.
    topq_gene_scores = feature_scores.loc[
        (feature_scores["feature_type"] == "gene") & (feature_scores["foreground_mode"] == "F1_topq")
    ].copy()

    flag_rows = []
    for r in genes_present:
        g = r.gene
        g_scores = topq_gene_scores.loc[topq_gene_scores["feature_name"] == g]
        if g_scores.empty:
            continue
        loc_mask = (g_scores["q_T"].to_numpy(dtype=float) <= Q_SIG) & np.isfinite(g_scores["q_T"].to_numpy(dtype=float))
        localized_any = bool(np.any(loc_mask))
        best_idx = int(np.nanargmax(g_scores["Z_T"].to_numpy(dtype=float))) if np.isfinite(g_scores["Z_T"].to_numpy(dtype=float)).any() else 0
        best_row = g_scores.iloc[best_idx]

        assoc_row = assoc_df.loc[assoc_df["gene"] == g]
        assoc_one = assoc_row.iloc[0] if not assoc_row.empty else pd.Series(dtype=float)
        qc_risk = float(assoc_one.get("qc_risk", np.nan)) if not assoc_row.empty else np.nan

        g_sim = sim_df.loc[sim_df["gene"] == g]
        profile_qc_risk = float(np.nanmax(g_sim["profile_qc_risk"].to_numpy(dtype=float))) if not g_sim.empty else np.nan

        qc_driven = bool(
            localized_any
            and (
                (np.isfinite(qc_risk) and qc_risk >= QC_RISK_THRESH)
                or (np.isfinite(profile_qc_risk) and profile_qc_risk >= SIM_QC_THRESH)
            )
        )

        # Most likely confounder.
        conf_scores: list[tuple[str, float]] = []
        for k in ["rho_total_counts", "rho_pct_counts_mt", "rho_pct_counts_ribo", "rho_ambient_score", "rho_doublet_score"]:
            v = assoc_one.get(k, np.nan) if not assoc_row.empty else np.nan
            if np.isfinite(float(v)):
                conf_scores.append((k, abs(float(v))))
        if not g_sim.empty:
            for k in ["sim_total_counts", "sim_pct_counts_mt", "sim_pct_counts_ribo", "sim_ambient_score", "sim_doublet_score"]:
                vv = g_sim[k].to_numpy(dtype=float)
                if np.isfinite(vv).any():
                    conf_scores.append((k, float(np.nanmax(vv))))
        conf = "none"
        if len(conf_scores) > 0:
            conf = str(max(conf_scores, key=lambda kv: kv[1])[0])

        flag_rows.append(
            {
                "gene": g,
                "localized_any_topq": localized_any,
                "best_embedding": str(best_row["embedding"]),
                "best_q_T": float(best_row["q_T"]) if np.isfinite(float(best_row["q_T"])) else np.nan,
                "best_Z_T": float(best_row["Z_T"]) if np.isfinite(float(best_row["Z_T"])) else np.nan,
                "qc_risk": qc_risk,
                "profile_qc_risk": profile_qc_risk,
                "qc_driven": qc_driven,
                "most_likely_confounder": conf,
                "clean_localized": bool(localized_any and not qc_driven),
            }
        )
    flags_df = pd.DataFrame(flag_rows)
    flags_df.to_csv(tables_dir / "qc_driven_flags.csv", index=False)

    # Ambient randomset null table.
    ambient_rows = []
    for emb in [e.key for e in embeddings]:
        amb_row = feature_scores.loc[
            (feature_scores["feature_name"] == "ambient_score")
            & (feature_scores["embedding"] == emb)
            & (feature_scores["foreground_mode"] == "F1_topq")
        ]
        if amb_row.empty:
            continue
        amb_z = float(amb_row.iloc[0]["Z_T"]) if np.isfinite(float(amb_row.iloc[0]["Z_T"])) else np.nan

        rand_z = feature_scores.loc[
            (feature_scores["feature_type"] == "randomset")
            & (feature_scores["embedding"] == emb)
            & (feature_scores["foreground_mode"] == "F1_topq")
        ]["Z_T"].to_numpy(dtype=float)
        rand_z = rand_z[np.isfinite(rand_z)]
        if np.isfinite(amb_z) and rand_z.size > 0:
            emp_p = float((1 + np.sum(rand_z >= amb_z)) / (1 + rand_z.size))
        else:
            emp_p = np.nan

        ambient_rows.append(
            {
                "embedding": emb,
                "kind": "ambient",
                "set_id": "ambient_score",
                "ambient_Z_T": amb_z,
                "empirical_p": emp_p,
                "mean_detection": float(np.mean((ambient_score > 0).astype(float))),
            }
        )

        for i in range(random_scores.shape[0]):
            nm = f"random_set_{i+1:03d}"
            rr = feature_scores.loc[
                (feature_scores["feature_name"] == nm)
                & (feature_scores["embedding"] == emb)
                & (feature_scores["foreground_mode"] == "F1_topq")
            ]
            if rr.empty:
                continue
            ambient_rows.append(
                {
                    "embedding": emb,
                    "kind": "randomset",
                    "set_id": nm,
                    "ambient_Z_T": float(rr.iloc[0]["Z_T"]) if np.isfinite(float(rr.iloc[0]["Z_T"])) else np.nan,
                    "empirical_p": np.nan,
                    "mean_detection": float(random_mean_det[i]) if i < random_mean_det.size else np.nan,
                }
            )

    ambient_null_df = pd.DataFrame(ambient_rows)
    ambient_null_df.to_csv(tables_dir / "ambient_randomset_null.csv", index=False)

    # -----------------------
    # Plot 00: Overview.
    # -----------------------
    umap_xy = geom_map["umap_repr"].coords
    if total_counts is not None:
        save_numeric_umap(
            umap_xy,
            np.log1p(np.maximum(np.asarray(total_counts, dtype=float), 0.0)),
            plots_dir / "00_overview" / "umap_log1p_total_counts.png",
            title="CM UMAP: log1p(total_counts)",
            cmap="viridis",
            colorbar_label="log1p(total_counts)",
        )
    else:
        _save_placeholder(plots_dir / "00_overview" / "umap_log1p_total_counts.png", "total_counts", "Unavailable")

    if pct_mt is not None:
        save_numeric_umap(
            umap_xy,
            np.asarray(pct_mt, dtype=float),
            plots_dir / "00_overview" / "umap_pct_counts_mt.png",
            title="CM UMAP: pct_counts_mt",
            cmap="magma",
            colorbar_label="pct_counts_mt",
        )
    else:
        _save_placeholder(plots_dir / "00_overview" / "umap_pct_counts_mt.png", "pct_counts_mt", "Unavailable")

    save_numeric_umap(
        umap_xy,
        np.asarray(ambient_score, dtype=float),
        plots_dir / "00_overview" / "umap_ambient_score.png",
        title="CM UMAP: ambient_score",
        cmap="viridis",
        colorbar_label="ambient_score",
    )

    if predicted_doublet is not None and doublet_score is not None:
        save_numeric_umap(
            umap_xy,
            np.asarray(doublet_score, dtype=float),
            plots_dir / "00_overview" / "umap_doublet_score.png",
            title="CM UMAP: doublet_score",
            cmap="cividis",
            colorbar_label="doublet_score",
        )
    else:
        _save_placeholder(
            plots_dir / "00_overview" / "umap_doublet_score.png",
            "doublet_score",
            f"Unavailable ({scrublet_status})",
        )

    # QC metrics table figure.
    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    ax.axis("off")
    tbl_df = qc_metrics_df[["metric", "available", "source", "mean", "median", "std", "missing_frac"]].copy()
    tbl_df["mean"] = tbl_df["mean"].round(4)
    tbl_df["median"] = tbl_df["median"].round(4)
    tbl_df["std"] = tbl_df["std"].round(4)
    tbl = ax.table(cellText=tbl_df.values, colLabels=tbl_df.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.3)
    ax.set_title("QC metrics availability and summaries")
    fig.tight_layout()
    fig.savefig(plots_dir / "00_overview" / "qc_metrics_table.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)

    # -----------------------
    # Plot 01: QC distributions.
    # -----------------------
    fig1, axes1 = plt.subplots(1, 4, figsize=(16.0, 3.8))
    dist_specs = [
        ("total_counts", total_counts, "log1p(total_counts)"),
        ("n_genes_by_counts", n_genes_by_counts, "n_genes_by_counts"),
        ("pct_counts_mt", pct_mt, "pct_counts_mt"),
        ("pct_counts_ribo", pct_ribo, "pct_counts_ribo"),
    ]
    for ax, (name, vec, xlabel) in zip(axes1, dist_specs, strict=False):
        if vec is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{name} unavailable", ha="center", va="center")
            continue
        x = np.asarray(vec, dtype=float)
        if name == "total_counts":
            x = np.log1p(np.maximum(x, 0.0))
        bins = int(min(70, max(15, np.ceil(np.sqrt(x.size)))))
        ax.hist(x, bins=bins, color="#4c78a8", alpha=0.85, edgecolor="white")
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
    fig1.tight_layout()
    fig1.savefig(plots_dir / "01_qc_distributions" / "hist_qc_metrics.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    # Joint: log1p_total_counts vs log1p_n_genes_by_counts.
    fig2, ax2 = plt.subplots(figsize=(6.4, 5.6))
    if total_counts is not None and n_genes_by_counts is not None:
        ax2.hexbin(
            np.log1p(np.maximum(total_counts, 0.0)),
            np.log1p(np.maximum(n_genes_by_counts, 0.0)),
            gridsize=45,
            cmap="viridis",
            mincnt=1,
        )
        ax2.set_xlabel("log1p(total_counts)")
        ax2.set_ylabel("log1p(n_genes_by_counts)")
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "Counts or genes metric unavailable", ha="center", va="center")
    ax2.set_title("QC joint: counts vs genes")
    fig2.tight_layout()
    fig2.savefig(plots_dir / "01_qc_distributions" / "joint_counts_vs_genes.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)

    # Joint: mt vs counts.
    fig3, ax3 = plt.subplots(figsize=(6.4, 5.6))
    if total_counts is not None and pct_mt is not None:
        ax3.hexbin(
            np.log1p(np.maximum(total_counts, 0.0)),
            np.asarray(pct_mt, dtype=float),
            gridsize=45,
            cmap="magma",
            mincnt=1,
        )
        ax3.set_xlabel("log1p(total_counts)")
        ax3.set_ylabel("pct_counts_mt")
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "total_counts or pct_counts_mt unavailable", ha="center", va="center")
    ax3.set_title("QC joint: mt vs counts")
    fig3.tight_layout()
    fig3.savefig(plots_dir / "01_qc_distributions" / "joint_mt_vs_counts.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig3)

    # -----------------------
    # Plot 02: QC pseudo-feature RSP.
    # -----------------------
    pseudo_targets = ["total_counts", "pct_counts_mt", "pct_counts_ribo", "ambient_score", "doublet_score"]
    for feat in pseudo_targets:
        for emb in [e.key for e in embeddings]:
            key = (emb, feat, "F1_topq")
            if key not in detail_map:
                continue
            row = feature_scores.loc[
                (feature_scores["embedding"] == emb)
                & (feature_scores["feature_name"] == feat)
                & (feature_scores["foreground_mode"] == "F1_topq")
            ]
            if row.empty:
                continue
            rr = row.iloc[0]
            det = detail_map[key]
            stats_txt = (
                f"Z={float(rr['Z_T']):.2f}\n"
                f"q={float(rr['q_T']):.2e}\n"
                f"C={float(rr['coverage_C']):.3f}\n"
                f"K={int(rr['peaks_K']) if np.isfinite(rr['peaks_K']) else -1}\n"
                f"phi={float(rr['phi_hat_deg']):.1f}"
            )
            _plot_polar_with_null(
                out_png=plots_dir / "02_qc_pseudofeature_rsp" / f"qc_rsp_{feat}_{emb}.png",
                title=f"{emb}: {feat}",
                e_obs=np.asarray(det["E_obs"], dtype=float),
                null_e=np.asarray(det["null_E"], dtype=float) if det["null_E"] is not None else None,
                t_obs=float(det["T_obs"]),
                null_t=np.asarray(det["null_T"], dtype=float) if det["null_T"] is not None else None,
                stats_text=stats_txt,
            )

    # -----------------------
    # Plot 03: Gene vs QC association.
    # -----------------------
    # gene Z summary from topq across embeddings
    gene_best = (
        topq_gene_scores.groupby("feature_name", as_index=False)
        .agg(best_Z_T=("Z_T", "max"), best_q_T=("q_T", "min"))
        .rename(columns={"feature_name": "gene"})
    )
    assoc_plot_df = assoc_df.merge(gene_best, on="gene", how="left").merge(
        flags_df[["gene", "qc_driven"]], on="gene", how="left"
    )
    assoc_plot_df["qc_driven"] = assoc_plot_df["qc_driven"].fillna(False).astype(bool)

    fig4, ax4 = plt.subplots(figsize=(7.0, 5.8))
    if not assoc_plot_df.empty:
        m0 = ~assoc_plot_df["qc_driven"].to_numpy(dtype=bool)
        m1 = assoc_plot_df["qc_driven"].to_numpy(dtype=bool)
        x = assoc_plot_df["qc_risk"].to_numpy(dtype=float)
        y = assoc_plot_df["best_Z_T"].to_numpy(dtype=float)
        if int(np.sum(m0)) > 0:
            ax4.scatter(x[m0], y[m0], s=70, c="#4c78a8", alpha=0.85, edgecolors="black", linewidths=0.35, label="not flagged")
        if int(np.sum(m1)) > 0:
            ax4.scatter(x[m1], y[m1], s=105, c="#d62728", marker="*", alpha=0.95, edgecolors="black", linewidths=0.5, label="QC-driven")
        for _, rr in assoc_plot_df.iterrows():
            ax4.text(float(rr["qc_risk"]), float(rr["best_Z_T"]) + 0.02, str(rr["gene"]), fontsize=8)
        ax4.axvline(QC_RISK_THRESH, color="#333", linestyle="--")
        ax4.set_xlabel("qc_risk")
        ax4.set_ylabel("best gene Z_T (topq)")
        ax4.legend(loc="best")
    else:
        ax4.axis("off")
        ax4.text(0.5, 0.5, "No association data", ha="center", va="center")
    ax4.set_title("qc_risk vs gene Z_T")
    fig4.tight_layout()
    fig4.savefig(plots_dir / "03_gene_vs_qc_association" / "qc_risk_vs_gene_ZT.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig4)

    # Rho heatmap.
    rho_cols = ["rho_total_counts", "rho_pct_counts_mt", "rho_pct_counts_ribo", "rho_ambient_score", "rho_doublet_score"]
    if not assoc_df.empty:
        hdf = assoc_df.set_index("gene")[rho_cols]
        _plot_heatmap(
            out_png=plots_dir / "03_gene_vs_qc_association" / "heatmap_gene_qc_rho.png",
            mat=np.nan_to_num(hdf.to_numpy(dtype=float), nan=0.0),
            row_labels=hdf.index.astype(str).tolist(),
            col_labels=rho_cols,
            title="Spearman rho: gene foreground vs QC covariates",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
    else:
        _save_placeholder(
            plots_dir / "03_gene_vs_qc_association" / "heatmap_gene_qc_rho.png",
            "Gene-QC rho",
            "No genes",
        )

    # Top 10 QC-driven scatter gene vs top confounder.
    top_qc = flags_df.loc[flags_df["qc_driven"]].sort_values(by="best_Z_T", ascending=False).head(10)
    if top_qc.empty:
        _save_placeholder(
            plots_dir / "03_gene_vs_qc_association" / "top_qc_driven_gene_vs_confounder.png",
            "Top QC-driven",
            "No QC-driven genes flagged",
        )
    else:
        n = len(top_qc)
        n_cols = 2
        n_rows = int(np.ceil(n / n_cols))
        fig5, axes5 = plt.subplots(n_rows, n_cols, figsize=(7.2 * n_cols, 3.8 * n_rows))
        ax_arr = np.atleast_1d(axes5).ravel()
        for i, (_, rr) in enumerate(top_qc.iterrows()):
            ax = ax_arr[i]
            gene = str(rr["gene"])
            conf = str(rr["most_likely_confounder"])
            gvals = np.asarray(feature_values.get(gene, np.zeros(adata_cm.n_obs)), dtype=float)
            # choose scatter covariate by conf name fallback.
            if "total_counts" in conf:
                cvals = qc_covars["total_counts"]
                cname = "total_counts"
            elif "pct_counts_mt" in conf:
                cvals = qc_covars["pct_counts_mt"]
                cname = "pct_counts_mt"
            elif "pct_counts_ribo" in conf:
                cvals = qc_covars["pct_counts_ribo"]
                cname = "pct_counts_ribo"
            elif "ambient" in conf:
                cvals = qc_covars["ambient_score"]
                cname = "ambient_score"
            elif "doublet" in conf:
                cvals = qc_covars["doublet_score"]
                cname = "doublet_score"
            else:
                cvals = qc_covars["pct_counts_mt"]
                cname = "pct_counts_mt"
            if cvals is None:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{gene}: {cname} unavailable", ha="center", va="center")
                continue
            x = np.asarray(cvals, dtype=float)
            y = np.log1p(np.maximum(gvals, 0.0))
            ax.scatter(x, y, s=11, alpha=0.42, color="#4c78a8", edgecolors="none", rasterized=True)
            rho = _safe_spearman((y > np.quantile(y, 0.9)).astype(float), x)
            ax.set_title(f"{gene} vs {cname} (rho={rho:.2f})")
            ax.set_xlabel(cname)
            ax.set_ylabel(f"log1p({gene})")
        for j in range(i + 1, len(ax_arr)):
            ax_arr[j].axis("off")
        fig5.tight_layout()
        fig5.savefig(plots_dir / "03_gene_vs_qc_association" / "top_qc_driven_gene_vs_confounder.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig5)

    # -----------------------
    # Plot 04: Profile similarity.
    # -----------------------
    if sim_df.empty:
        _save_placeholder(plots_dir / "04_profile_similarity" / "heatmap_profile_similarity.png", "Profile similarity", "No data")
        _save_placeholder(plots_dir / "04_profile_similarity" / "profile_qc_risk_vs_gene_ZT.png", "Profile risk", "No data")
    else:
        sim_cols = ["sim_total_counts", "sim_pct_counts_mt", "sim_pct_counts_ribo", "sim_ambient_score", "sim_doublet_score"]
        sim_gene_max = (
            sim_df.groupby("gene", as_index=False)[sim_cols + ["profile_qc_risk"]]
            .max(numeric_only=True)
            .set_index("gene")
        )
        _plot_heatmap(
            out_png=plots_dir / "04_profile_similarity" / "heatmap_profile_similarity.png",
            mat=np.nan_to_num(sim_gene_max[sim_cols].to_numpy(dtype=float), nan=0.0),
            row_labels=sim_gene_max.index.astype(str).tolist(),
            col_labels=sim_cols,
            title="Cosine similarity: gene vs QC pseudo-feature profiles",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        sim_plot_df = flags_df.copy()
        if "best_Z_T" not in sim_plot_df.columns:
            sim_plot_df = sim_plot_df.merge(gene_best, on="gene", how="left")
        if "profile_qc_risk" not in sim_plot_df.columns:
            sim_plot_df = sim_plot_df.merge(
                sim_gene_max[["profile_qc_risk"]],
                left_on="gene",
                right_index=True,
                how="left",
            )

        fig6, ax6 = plt.subplots(figsize=(7.2, 5.8))
        if not sim_plot_df.empty:
            m1 = sim_plot_df["qc_driven"].to_numpy(dtype=bool)
            m0 = ~m1
            x = sim_plot_df["profile_qc_risk"].to_numpy(dtype=float)
            y = sim_plot_df["best_Z_T"].to_numpy(dtype=float)
            if int(np.sum(m0)) > 0:
                ax6.scatter(x[m0], y[m0], s=70, c="#4c78a8", alpha=0.86, edgecolors="black", linewidths=0.35)
            if int(np.sum(m1)) > 0:
                ax6.scatter(x[m1], y[m1], s=110, c="#d62728", marker="*", alpha=0.95, edgecolors="black", linewidths=0.5)
            for _, rr in sim_plot_df.iterrows():
                if np.isfinite(float(rr.get("profile_qc_risk", np.nan))) and np.isfinite(float(rr.get("best_Z_T", np.nan))):
                    ax6.text(float(rr["profile_qc_risk"]), float(rr["best_Z_T"]) + 0.02, str(rr["gene"]), fontsize=8)
            ax6.axvline(SIM_QC_THRESH, color="#333", linestyle="--")
            ax6.set_xlabel("profile_qc_risk")
            ax6.set_ylabel("best gene Z_T")
        else:
            ax6.axis("off")
        ax6.set_title("Profile QC risk vs gene Z_T")
        fig6.tight_layout()
        fig6.savefig(plots_dir / "04_profile_similarity" / "profile_qc_risk_vs_gene_ZT.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig6)

    # Overlay polar for top QC-driven genes.
    top_overlay = flags_df.loc[flags_df["qc_driven"]].sort_values(by="best_Z_T", ascending=False).head(6)
    if top_overlay.empty:
        _save_placeholder(
            plots_dir / "04_profile_similarity" / "overlay_top_qcdriven_profiles.png",
            "Overlay polar",
            "No QC-driven genes flagged",
        )
    else:
        n = len(top_overlay)
        n_cols = 3
        n_rows = int(np.ceil(n / n_cols))
        fig7 = plt.figure(figsize=(4.6 * n_cols, 4.2 * n_rows))
        for i, (_, rr) in enumerate(top_overlay.iterrows()):
            ax = fig7.add_subplot(n_rows, n_cols, i + 1, projection="polar")
            gene = str(rr["gene"])
            emb = str(rr["best_embedding"])
            g_key = (emb, gene, "F1_topq")
            if g_key not in detail_map:
                ax.axis("off")
                continue
            conf = str(rr["most_likely_confounder"])
            # map confounder to feature.
            if "total_counts" in conf:
                qf = "total_counts"
            elif "pct_counts_mt" in conf:
                qf = "pct_counts_mt"
            elif "pct_counts_ribo" in conf:
                qf = "pct_counts_ribo"
            elif "ambient" in conf:
                qf = "ambient_score"
            elif "doublet" in conf:
                qf = "doublet_score"
            else:
                qf = "pct_counts_mt"
            q_key = (emb, qf, "F1_topq")
            if q_key not in detail_map:
                ax.axis("off")
                continue
            ge = np.asarray(detail_map[g_key]["E_obs"], dtype=float)
            qe = np.asarray(detail_map[q_key]["E_obs"], dtype=float)
            th = theta_bin_centers(len(ge))
            th = np.concatenate([th, th[:1]])
            ax.plot(th, np.concatenate([ge, ge[:1]]), color="#8B0000", linewidth=2.0, label=gene)
            ax.plot(th, np.concatenate([qe, qe[:1]]), color="#2E8B57", linewidth=1.8, linestyle="--", label=qf)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_rticks([])
            ax.set_title(f"{gene} vs {qf} ({emb})", fontsize=8)
            ax.legend(loc="upper right", fontsize=7, frameon=True)
        fig7.tight_layout()
        fig7.savefig(plots_dir / "04_profile_similarity" / "overlay_top_qcdriven_profiles.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig7)

    # -----------------------
    # Plot 05: Ambient controls.
    # -----------------------
    # 1) histogram random Z with ambient line
    for emb in [e.key for e in embeddings]:
        sub = ambient_null_df.loc[ambient_null_df["embedding"] == emb]
        amb = sub.loc[sub["kind"] == "ambient", "ambient_Z_T"].to_numpy(dtype=float)
        rnd = sub.loc[sub["kind"] == "randomset", "ambient_Z_T"].to_numpy(dtype=float)
        if rnd.size == 0:
            _save_placeholder(
                plots_dir / "05_ambient_controls" / f"ambient_random_Z_{emb}.png",
                f"Ambient random null ({emb})",
                "No random set data",
            )
            continue
        fig8, ax8 = plt.subplots(figsize=(7.1, 5.4))
        bins = int(min(30, max(8, np.ceil(np.sqrt(rnd.size)))))
        ax8.hist(rnd, bins=bins, color="#729FCF", edgecolor="white", alpha=0.9)
        if amb.size > 0 and np.isfinite(amb[0]):
            ax8.axvline(float(amb[0]), color="#8B0000", linestyle="--", linewidth=2.0, label="ambient_score")
            ax8.legend(loc="best")
        ax8.set_xlabel("Z_T")
        ax8.set_ylabel("count")
        ax8.set_title(f"{emb}: random matched-set Z_T vs ambient_score")
        fig8.tight_layout()
        fig8.savefig(plots_dir / "05_ambient_controls" / f"ambient_random_Z_{emb}.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig8)

    # 2) gene Z vs sim_ambient
    sim_ambient_df = (
        sim_df.groupby("gene", as_index=False)["sim_ambient_score"].max(numeric_only=True)
        if not sim_df.empty and "sim_ambient_score" in sim_df.columns
        else pd.DataFrame(columns=["gene", "sim_ambient_score"])
    )
    amb_scatter_df = gene_best.merge(sim_ambient_df, on="gene", how="left").merge(
        flags_df[["gene", "qc_driven"]], on="gene", how="left"
    )
    fig9, ax9 = plt.subplots(figsize=(7.1, 5.8))
    if not amb_scatter_df.empty:
        qc_flag = amb_scatter_df["qc_driven"].fillna(False).to_numpy(dtype=bool)
        x = amb_scatter_df["sim_ambient_score"].to_numpy(dtype=float)
        y = amb_scatter_df["best_Z_T"].to_numpy(dtype=float)
        ax9.scatter(
            x,
            y,
            s=np.where(qc_flag, 105, 70),
            c=np.where(qc_flag, "#d62728", "#4c78a8"),
            alpha=0.88,
            edgecolors="black",
            linewidths=0.35,
        )
        for _, rr in amb_scatter_df.iterrows():
            if np.isfinite(float(rr.get("sim_ambient_score", np.nan))) and np.isfinite(float(rr.get("best_Z_T", np.nan))):
                ax9.text(float(rr["sim_ambient_score"]), float(rr["best_Z_T"]) + 0.02, str(rr["gene"]), fontsize=8)
        ax9.axvline(SIM_QC_THRESH, color="#333", linestyle="--")
        ax9.set_xlabel("sim_ambient")
        ax9.set_ylabel("best gene Z_T")
    else:
        ax9.axis("off")
    ax9.set_title("Gene Z_T vs ambient-profile similarity")
    fig9.tight_layout()
    fig9.savefig(plots_dir / "05_ambient_controls" / "gene_Z_vs_sim_ambient.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig9)

    # 3) if ambient localized, side-by-side UMAP ambient + top high-sim genes
    ambient_loc_any = bool(
        np.isfinite(
            feature_scores.loc[
                (feature_scores["feature_name"] == "ambient_score")
                & (feature_scores["foreground_mode"] == "F1_topq")
            ]["q_T"].to_numpy(dtype=float)
        ).any()
        and np.nanmin(
            feature_scores.loc[
                (feature_scores["feature_name"] == "ambient_score")
                & (feature_scores["foreground_mode"] == "F1_topq")
            ]["q_T"].to_numpy(dtype=float)
        )
        <= Q_SIG
    )

    if ambient_loc_any and (not sim_ambient_df.empty):
        top_sim_genes = (
            sim_ambient_df.sort_values(by="sim_ambient_score", ascending=False)["gene"].astype(str).tolist()[:3]
        )
        n_pan = 1 + len(top_sim_genes)
        fig10, axes10 = plt.subplots(1, n_pan, figsize=(5.0 * n_pan, 4.8))
        ax0 = axes10[0] if n_pan > 1 else axes10
        ord0 = np.argsort(ambient_score, kind="mergesort")
        sc0 = ax0.scatter(
            umap_xy[ord0, 0],
            umap_xy[ord0, 1],
            c=ambient_score[ord0],
            cmap="viridis",
            s=7,
            alpha=0.9,
            linewidths=0,
            rasterized=True,
        )
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_title("ambient_score")
        fig10.colorbar(sc0, ax=ax0, fraction=0.046, pad=0.03)

        for j, g in enumerate(top_sim_genes, start=1):
            ax = axes10[j]
            vals = np.log1p(np.maximum(np.asarray(feature_values[g], dtype=float), 0.0))
            ordg = np.argsort(vals, kind="mergesort")
            scg = ax.scatter(
                umap_xy[ordg, 0],
                umap_xy[ordg, 1],
                c=vals[ordg],
                cmap="Reds",
                s=7,
                alpha=0.9,
                linewidths=0,
                rasterized=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(g)
            fig10.colorbar(scg, ax=ax, fraction=0.046, pad=0.03)

        fig10.tight_layout()
        fig10.savefig(plots_dir / "05_ambient_controls" / "ambient_vs_high_sim_genes_umap.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig10)
    else:
        _save_placeholder(
            plots_dir / "05_ambient_controls" / "ambient_vs_high_sim_genes_umap.png",
            "Ambient localization",
            "Ambient not strongly localized or similarity table empty",
        )

    # -----------------------
    # Plot 06: Flagged exemplars.
    # -----------------------
    if flags_df.empty:
        _save_placeholder(plots_dir / "06_flagged_exemplars" / "exemplar_none.png", "Exemplars", "No gene flag rows")
    else:
        qc_ex = flags_df.loc[flags_df["qc_driven"]].sort_values(by="best_Z_T", ascending=False).head(6)
        clean_ex = flags_df.loc[(flags_df["clean_localized"])].sort_values(by="best_Z_T", ascending=False).head(6)
        exemplar_rows = []
        for _, rr in qc_ex.iterrows():
            exemplar_rows.append(("QCdriven", rr))
        for _, rr in clean_ex.iterrows():
            exemplar_rows.append(("clean", rr))

        if len(exemplar_rows) == 0:
            _save_placeholder(
                plots_dir / "06_flagged_exemplars" / "exemplar_none.png",
                "Exemplars",
                "No QC-driven or clean localized genes",
            )
        else:
            for kind, rr in exemplar_rows:
                gene = str(rr["gene"])
                emb = str(rr["best_embedding"])
                conf = str(rr["most_likely_confounder"])
                gvals = np.asarray(feature_values.get(gene, np.zeros(adata_cm.n_obs)), dtype=float)

                # map conf feature
                if "total_counts" in conf:
                    qf = "total_counts"
                elif "pct_counts_mt" in conf:
                    qf = "pct_counts_mt"
                elif "pct_counts_ribo" in conf:
                    qf = "pct_counts_ribo"
                elif "ambient" in conf:
                    qf = "ambient_score"
                elif "doublet" in conf:
                    qf = "doublet_score"
                else:
                    qf = "pct_counts_mt"

                fig11 = plt.figure(figsize=(15.5, 8.2))
                gs = fig11.add_gridspec(2, 2, wspace=0.26, hspace=0.28)

                # A) UMAP feature.
                axA = fig11.add_subplot(gs[0, 0])
                valsA = np.log1p(np.maximum(gvals, 0.0))
                ordA = np.argsort(valsA, kind="mergesort")
                axA.scatter(umap_xy[:, 0], umap_xy[:, 1], c="#d9d9d9", s=4, alpha=0.3, linewidths=0, rasterized=True)
                scA = axA.scatter(
                    umap_xy[ordA, 0],
                    umap_xy[ordA, 1],
                    c=valsA[ordA],
                    cmap="Reds",
                    s=7,
                    alpha=0.9,
                    linewidths=0,
                    rasterized=True,
                )
                axA.set_xticks([])
                axA.set_yticks([])
                axA.set_title(f"{gene} feature map")
                fig11.colorbar(scA, ax=axA, fraction=0.046, pad=0.03)

                # B) gene polar
                axB = fig11.add_subplot(gs[0, 1], projection="polar")
                g_key = (emb, gene, "F1_topq")
                if g_key in detail_map:
                    ge = np.asarray(detail_map[g_key]["E_obs"], dtype=float)
                    th = theta_bin_centers(len(ge))
                    th = np.concatenate([th, th[:1]])
                    axB.plot(th, np.concatenate([ge, ge[:1]]), color="#8B0000", linewidth=2.0)
                    grow = feature_scores.loc[
                        (feature_scores["feature_name"] == gene)
                        & (feature_scores["embedding"] == emb)
                        & (feature_scores["foreground_mode"] == "F1_topq")
                    ]
                    if not grow.empty:
                        gr = grow.iloc[0]
                        txt = f"Z={float(gr['Z_T']):.2f}\nq={float(gr['q_T']):.2e}\nC={float(gr['coverage_C']):.3f}\nK={int(gr['peaks_K']) if np.isfinite(gr['peaks_K']) else -1}"
                        axB.text(0.02, 0.02, txt, transform=axB.transAxes, fontsize=8, ha="left", va="bottom", bbox={"facecolor": "white", "edgecolor": "#999", "alpha": 0.85})
                else:
                    axB.text(0.5, 0.5, "profile missing", transform=axB.transAxes, ha="center", va="center")
                axB.set_theta_zero_location("E")
                axB.set_theta_direction(1)
                axB.set_rticks([])
                axB.set_title(f"{gene} polar ({emb})")

                # C) confounder polar
                axC = fig11.add_subplot(gs[1, 0], projection="polar")
                q_key = (emb, qf, "F1_topq")
                if q_key in detail_map:
                    qe = np.asarray(detail_map[q_key]["E_obs"], dtype=float)
                    th = theta_bin_centers(len(qe))
                    th = np.concatenate([th, th[:1]])
                    axC.plot(th, np.concatenate([qe, qe[:1]]), color="#2E8B57", linewidth=2.0)
                else:
                    axC.text(0.5, 0.5, "profile missing", transform=axC.transAxes, ha="center", va="center")
                axC.set_theta_zero_location("E")
                axC.set_theta_direction(1)
                axC.set_rticks([])
                axC.set_title(f"Confounder polar: {qf} ({emb})")

                # D) scatter gene vs confounder
                axD = fig11.add_subplot(gs[1, 1])
                cvals = qc_covars.get(qf, None)
                if cvals is None:
                    axD.axis("off")
                    axD.text(0.5, 0.5, f"{qf} unavailable", ha="center", va="center")
                else:
                    x = np.asarray(cvals, dtype=float)
                    y = np.log1p(np.maximum(gvals, 0.0))
                    axD.scatter(x, y, s=11, alpha=0.45, color="#4c78a8", edgecolors="none", rasterized=True)
                    rho = _safe_spearman((y > np.quantile(y, 0.9)).astype(float), x)
                    axD.set_xlabel(qf)
                    axD.set_ylabel(f"log1p({gene})")
                    axD.set_title(f"{gene} vs {qf} (rho={rho:.2f})")

                fig11.suptitle(
                    f"Exemplar {kind}: {gene} | conf={qf} | emb={emb}",
                    y=0.995,
                )
                fig11.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
                fig11.savefig(
                    plots_dir / "06_flagged_exemplars" / f"exemplar_{kind}_{gene}.png",
                    dpi=DEFAULT_PLOT_STYLE.dpi,
                )
                plt.close(fig11)

    # Write README.
    ambient_emp_rows = ambient_null_df.loc[ambient_null_df["kind"] == "ambient"].copy() if not ambient_null_df.empty else pd.DataFrame()
    _write_readme(
        out_root / "README.txt",
        args=args,
        donor_key=donor_key,
        label_key=label_key,
        donor_star=donor_star,
        expr_source=expr_source,
        embed_note=f"{embed_note}; n_pcs_used={n_pcs_used}",
        n_cells_donor=int(adata_donor.n_obs),
        n_cells_cm=int(adata_cm.n_obs),
        cm_labels={str(k): int(v) for k, v in cm_label_counts.to_dict().items()},
        qc_metric_sources=qc_metric_sources,
        scrublet_status=scrublet_status,
        scrublet_note=scrublet_note,
        ambient_present_genes=[r.gene for r in ambient_present],
        ambient_emp_rows=ambient_emp_rows,
        flags_df=flags_df,
        warnings_log=warnings_log,
    )

    # Verify required outputs.
    required = [
        tables_dir / "donor_choice.csv",
        tables_dir / "qc_metrics_summary.csv",
        tables_dir / "feature_scores_long.csv",
        tables_dir / "qc_pseudofeature_profiles.csv",
        tables_dir / "gene_qc_association.csv",
        tables_dir / "gene_profile_similarity.csv",
        tables_dir / "qc_driven_flags.csv",
        tables_dir / "ambient_randomset_null.csv",
        plots_dir / "00_overview",
        plots_dir / "01_qc_distributions",
        plots_dir / "02_qc_pseudofeature_rsp",
        plots_dir / "03_gene_vs_qc_association",
        plots_dir / "04_profile_similarity",
        plots_dir / "05_ambient_controls",
        plots_dir / "06_flagged_exemplars",
        out_root / "README.txt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing required outputs: " + ", ".join(missing))

    print(f"donor_key_used={donor_key}")
    print(f"label_key_used={label_key}")
    print(f"donor_star={donor_star}")
    print(f"expression_source_used={expr_source}")
    print(f"scrublet_status={scrublet_status}")
    print(f"ambient_genes_present={len(ambient_present)}")
    print(f"n_features_scored={int(len(feature_scores))}")
    print(f"n_genes_qc_flagged={int(np.sum(flags_df['qc_driven'].to_numpy(dtype=bool)) if not flags_df.empty else 0)}")
    print(f"results_root={out_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
