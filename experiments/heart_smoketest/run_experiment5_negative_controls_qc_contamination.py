#!/usr/bin/env python3
"""Experiment #5: negative controls for QC- and contamination-driven directionality."""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

# Headless backend for deterministic local/CI figure generation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

# Allow direct script execution from repository root.
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
from biorsp.pipeline.hierarchy import (
    _ensure_umap,
    _pct_mt_vector,
    _resolve_expr_matrix,
    _total_counts_vector,
)
from biorsp.plotting.qc import plot_categorical_umap, save_numeric_umap
from biorsp.plotting.styles import DEFAULT_PLOT_STYLE, apply_plot_style
from biorsp.stats.permutation import perm_null_T_and_profile
from biorsp.stats.scoring import (
    bh_fdr,
    classify_row,
    compute_T,
    coverage_from_null,
    peak_count,
    robust_z,
)

DONOR_CANDIDATES = [
    "donor",
    "sample",
    "individual",
    "hubmap_id",
    "donor_id",
    "sample_id",
    "patient",
    "orig.ident",
    "dataset",
    "batch",
]

LABEL_KEY_CANDIDATES = [
    "azimuth_label",
    "predicted_label",
    "predicted_CLID",
    "cell_type",
]

MARKER_PANEL_EXP1 = [
    "MYH6",
    "TNNT2",
    "RYR2",
    "PLN",
    "COL1A1",
    "COL1A2",
    "LUM",
    "DCN",
    "PECAM1",
    "VWF",
    "KDR",
    "ACTA2",
    "TAGLN",
    "RGS5",
    "PTPRC",
    "LST1",
    "LYZ",
]

HEMOGLOBIN_GENES = ["HBA1", "HBA2", "HBB", "HBD"]
PLATELET_GENES = ["PPBP", "PF4"]
MYELOID_SOUP_GENES = ["LYZ", "S100A8", "S100A9", "LST1", "TYROBP"]

FLAG_COLORS = {
    "QC-driven": "#D62728",
    "Contam-like": "#FF7F0E",
    "Both": "#8C564B",
    "Unflagged significant": "#1F77B4",
    "Not significant": "#BDBDBD",
}


@dataclass(frozen=True)
class ResolvedGene:
    gene: str
    found: bool
    gene_idx: int | None
    resolved_gene: str
    status: str
    source: str
    symbol_column: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment #5 negative-control audit for QC/contamination-driven BioRSP directionality."
    )
    p.add_argument(
        "--h5ad", default="data/processed/HT_pca_umap.h5ad", help="Input .h5ad path."
    )
    p.add_argument(
        "--out",
        default="experiments/heart_smoketest/results/experiment5_negative_controls_qc_contamination",
        help="Output directory.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--n_perm", type=int, default=300, help="Permutation count.")
    p.add_argument("--n_bins", type=int, default=64, help="Angular bins.")
    p.add_argument(
        "--max_genes",
        type=int,
        default=2000,
        help="Cap for default evaluation gene set.",
    )
    p.add_argument("--scope", choices=["global", "non_myeloid", "both"], default="both")
    p.add_argument(
        "--gene_subset", default=None, help="Optional path to newline-separated genes."
    )
    p.add_argument(
        "--sim_thresh",
        type=float,
        default=0.7,
        help="Similarity threshold for QC/contam flags.",
    )
    p.add_argument(
        "--qc_thresh",
        type=float,
        default=0.35,
        help="QC-risk threshold for QC-driven flag.",
    )
    p.add_argument(
        "--embedding_key", default=None, help="Optional embedding key override."
    )
    p.add_argument("--donor_key", default=None, help="Optional donor key override.")
    p.add_argument(
        "--label_key", default=None, help="Optional cell-type label key override."
    )
    p.add_argument("--layer", default=None, help="Optional expression layer override.")
    p.add_argument(
        "--use_raw", action="store_true", help="Use adata.raw instead of X/layers."
    )
    return p.parse_args()


def _save_placeholder(out_png: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)


def _resolve_embedding(
    adata: ad.AnnData, requested_key: str | None
) -> tuple[str, np.ndarray]:
    if requested_key is not None:
        if requested_key not in adata.obsm:
            raise KeyError(
                f"Requested embedding key '{requested_key}' missing in adata.obsm."
            )
        key = str(requested_key)
    else:
        key = "X_umap" if "X_umap" in adata.obsm else str(next(iter(adata.obsm.keys())))
    xy = np.asarray(adata.obsm[key], dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Embedding '{key}' must have shape (N,2+).")
    return key, xy[:, :2]


def _choose_expression_source(
    adata: ad.AnnData,
    layer_arg: str | None,
    use_raw_arg: bool,
) -> tuple[Any, Any, str]:
    if layer_arg is not None or use_raw_arg:
        return _resolve_expr_matrix(adata, layer=layer_arg, use_raw=bool(use_raw_arg))
    if "counts" in adata.layers:
        return _resolve_expr_matrix(adata, layer="counts", use_raw=False)
    if adata.raw is not None:
        return _resolve_expr_matrix(adata, layer=None, use_raw=True)
    return _resolve_expr_matrix(adata, layer=None, use_raw=False)


def _resolve_key(
    adata: ad.AnnData, requested: str | None, candidates: list[str]
) -> str | None:
    if requested is not None:
        return str(requested) if requested in adata.obs.columns else None
    for c in candidates:
        if c in adata.obs.columns:
            return c
    return None


def _resolve_donor_ids_optional(
    adata: ad.AnnData,
    requested_key: str | None,
) -> tuple[np.ndarray | None, str | None]:
    donor_key = _resolve_key(adata, requested_key, DONOR_CANDIDATES)
    if donor_key is None:
        return None, None
    ids = adata.obs[donor_key].astype("string").fillna("NA").astype(str).to_numpy()
    if np.unique(ids).size < 2:
        return None, donor_key
    return ids, donor_key


def _resolve_label_key(adata: ad.AnnData, requested_key: str | None) -> str | None:
    return _resolve_key(adata, requested_key, LABEL_KEY_CANDIDATES)


def _compute_pct_counts_ribo(
    adata: ad.AnnData,
    expr_matrix: Any,
    adata_like: Any,
    total_counts: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    if "pct_counts_ribo" in adata.obs.columns:
        vals = pd.to_numeric(adata.obs["pct_counts_ribo"], errors="coerce").to_numpy(
            dtype=float
        )
        if np.isfinite(vals).sum() > 0:
            fill = float(np.nanmedian(vals))
            vals = np.where(np.isfinite(vals), vals, fill)
            return vals, "obs:pct_counts_ribo"

    symbol_col = None
    if hasattr(adata_like, "var") and adata_like.var is not None:
        for c in ["hugo_symbol", "gene_name", "gene_symbol"]:
            if c in adata_like.var.columns:
                symbol_col = c
                break
    if symbol_col is None:
        return None, "missing"

    symbols = (
        adata_like.var[symbol_col].astype("string").fillna("").astype(str).str.upper()
    )
    ribo_mask = (
        symbols.str.startswith("RPL") | symbols.str.startswith("RPS")
    ).to_numpy(dtype=bool)
    if int(ribo_mask.sum()) == 0:
        return None, "missing"

    ribo_counts = (
        np.asarray(expr_matrix[:, ribo_mask].sum(axis=1)).ravel().astype(float)
    )
    pct = (
        np.divide(ribo_counts, np.maximum(np.asarray(total_counts, dtype=float), 1e-12))
        * 100.0
    )
    return pct, f"computed:{symbol_col}"


def _safe_spearman(x: np.ndarray, y: np.ndarray | None) -> float:
    if y is None:
        return float("nan")
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(xv) & np.isfinite(yv)
    if int(mask.sum()) < 3:
        return float("nan")
    xs = xv[mask]
    ys = yv[mask]
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return float("nan")
    rho = spearmanr(xs, ys, nan_policy="omit").correlation
    if rho is None or not np.isfinite(float(rho)):
        return float("nan")
    return float(rho)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size != y.size or x.size == 0:
        return float("nan")
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= 1e-12 or ny <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def _zscore_log1p(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    x = np.where(
        np.isfinite(x),
        x,
        np.nanmedian(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0,
    )
    x = np.log1p(np.maximum(x, 0.0))
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def _compute_continuous_profile(
    weights: np.ndarray,
    *,
    n_bins: int,
    bin_id: np.ndarray,
    bin_counts_total: np.ndarray,
) -> np.ndarray:
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != bin_id.size:
        raise ValueError("weights and bin_id size mismatch.")
    clean = np.where(np.isfinite(w), w, 0.0)
    if np.nanmin(clean) < 0:
        clean = clean - float(np.nanmin(clean))
    s = float(np.sum(clean))
    if s <= 1e-12:
        return np.zeros(int(n_bins), dtype=float)
    w_bin = np.bincount(bin_id, weights=clean, minlength=int(n_bins)).astype(float)
    p_w = w_bin / s
    p_bg = np.asarray(bin_counts_total, dtype=float) / float(bin_id.size)
    return p_w - p_bg


def _permute_weights_within_donor(
    values: np.ndarray,
    donor_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    d = np.asarray(donor_ids).astype(str)
    out = np.zeros_like(x)
    for donor in np.unique(d):
        idx = np.flatnonzero(d == donor)
        if idx.size <= 1:
            out[idx] = x[idx]
        else:
            out[idx] = x[idx[rng.permutation(idx.size)]]
    return out


def _perm_null_continuous_profile(
    weights: np.ndarray,
    *,
    donor_ids: np.ndarray | None,
    n_bins: int,
    n_perm: int,
    seed: int,
    bin_id: np.ndarray,
    bin_counts_total: np.ndarray,
) -> dict[str, Any]:
    x = np.asarray(weights, dtype=float).ravel()
    e_obs = _compute_continuous_profile(
        x,
        n_bins=int(n_bins),
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )
    t_obs = float(compute_T(e_obs))
    used_donor = bool(
        donor_ids is not None and np.unique(np.asarray(donor_ids).astype(str)).size >= 2
    )
    warning_msg = ""
    if not used_donor:
        warning_msg = "continuous null used global shuffling (donor unavailable)."

    rng = np.random.default_rng(int(seed))
    null_e = np.zeros((int(n_perm), int(n_bins)), dtype=float)
    null_t = np.zeros(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        if used_donor and donor_ids is not None:
            pvals = _permute_weights_within_donor(
                x, np.asarray(donor_ids).astype(str), rng
            )
        else:
            pvals = x[rng.permutation(x.size)]
        e_perm = _compute_continuous_profile(
            pvals,
            n_bins=int(n_bins),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        null_e[i, :] = e_perm
        null_t[i] = float(compute_T(e_perm))

    p_t = float((1.0 + np.sum(null_t >= t_obs)) / (1.0 + null_t.size))
    out = {
        "E_phi_obs": e_obs,
        "null_E_phi": null_e,
        "null_T": null_t,
        "T_obs": t_obs,
        "p_T": p_t,
        "used_donor_stratified": bool(used_donor),
    }
    if warning_msg:
        out["warning"] = warning_msg
    return out


def _resolve_symbol_series(adata_like: Any) -> pd.Series:
    if hasattr(adata_like, "var") and adata_like.var is not None:
        for c in ["hugo_symbol", "gene_name", "gene_symbol"]:
            if c in adata_like.var.columns:
                return adata_like.var[c].astype("string").fillna("").astype(str)
    return pd.Series(
        pd.Index(adata_like.var_names).astype(str).tolist(),
        index=adata_like.var_names,
        dtype="string",
    )


def _resolve_gene(adata_like: Any, gene: str) -> ResolvedGene:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            idx, label, symbol_col, source = resolve_feature_index(adata_like, gene)
        return ResolvedGene(
            gene=gene,
            found=True,
            gene_idx=int(idx),
            resolved_gene=str(label),
            status="resolved",
            source=str(source),
            symbol_column=str(symbol_col or ""),
        )
    except KeyError:
        return ResolvedGene(
            gene=gene,
            found=False,
            gene_idx=None,
            resolved_gene="",
            status="missing",
            source="",
            symbol_column="",
        )


def _read_gene_subset(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    genes: list[str] = []
    for ln in lines:
        g = ln.strip()
        if g == "" or g.startswith("#"):
            continue
        genes.append(g)
    return genes


def _detection_rate_top_genes(
    expr_matrix: Any,
    adata_like: Any,
    max_genes: int,
) -> list[str]:
    if sp.issparse(expr_matrix):
        detected = np.asarray((expr_matrix > 0).sum(axis=0)).ravel().astype(float)
    else:
        detected = np.sum(np.asarray(expr_matrix) > 0, axis=0).astype(float)
    rates = detected / max(1.0, float(expr_matrix.shape[0]))
    order = np.argsort(-rates, kind="mergesort")
    symbols = _resolve_symbol_series(adata_like).astype(str).tolist()
    var_names = pd.Index(adata_like.var_names).astype(str).tolist()
    out: list[str] = []
    seen: set[str] = set()
    for idx in order:
        sym = symbols[int(idx)].strip()
        name = sym if sym != "" else var_names[int(idx)]
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= int(max_genes):
            break
    return out


def _hvg_genes(adata_like: Any, max_genes: int) -> list[str]:
    if not hasattr(adata_like, "var") or adata_like.var is None:
        return []
    if "highly_variable" not in adata_like.var.columns:
        return []
    hv = adata_like.var["highly_variable"].astype(bool).to_numpy()
    idx = np.flatnonzero(hv)
    if idx.size == 0:
        return []
    symbols = _resolve_symbol_series(adata_like).astype(str).tolist()
    var_names = pd.Index(adata_like.var_names).astype(str).tolist()
    out: list[str] = []
    seen: set[str] = set()
    for i in idx:
        sym = symbols[int(i)].strip()
        name = sym if sym != "" else var_names[int(i)]
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out[: int(max_genes)]


def _build_contam_gene_sets(adata_like: Any) -> dict[str, list[str]]:
    symbols = _resolve_symbol_series(adata_like).astype(str).str.strip()
    upper = symbols.str.upper()

    mt = sorted(
        symbols[
            (upper.str.startswith("MT-") | upper.str.startswith("MT")) & (symbols != "")
        ]
        .unique()
        .tolist()
    )
    ribo = sorted(
        symbols[
            (upper.str.startswith("RPL") | upper.str.startswith("RPS"))
            & (symbols != "")
        ]
        .unique()
        .tolist()
    )

    def _present(genes: list[str]) -> list[str]:
        out: list[str] = []
        for g in genes:
            rg = _resolve_gene(adata_like, g)
            if rg.found:
                out.append(g)
        return out

    hemo = _present(HEMOGLOBIN_GENES)
    platelet = _present(PLATELET_GENES)
    soup = _present(MYELOID_SOUP_GENES)
    return {
        "mt_genes": mt,
        "ribo_genes": ribo,
        "hemoglobin_genes": hemo,
        "platelet_genes": platelet,
        "myeloid_soup_genes": soup,
    }


def _build_eval_genes(
    *,
    adata_like: Any,
    expr_matrix: Any,
    max_genes: int,
    gene_subset_path: Path | None,
    contam_sets: dict[str, list[str]],
) -> tuple[list[ResolvedGene], str]:
    if gene_subset_path is not None:
        base = _read_gene_subset(gene_subset_path)
        logic = f"gene_subset_file:{gene_subset_path.as_posix()}"
    else:
        hv = _hvg_genes(adata_like, int(max_genes))
        if len(hv) > 0:
            base = hv[: int(max_genes)]
            logic = f"hvg_cap_{int(max_genes)}"
        else:
            base = _detection_rate_top_genes(expr_matrix, adata_like, int(max_genes))
            logic = f"top_detection_cap_{int(max_genes)}"

    force_include = list(MARKER_PANEL_EXP1)
    for genes in contam_sets.values():
        force_include.extend(genes)

    merged: list[str] = []
    seen: set[str] = set()
    for g in list(base) + force_include:
        key = str(g).strip()
        if key == "" or key in seen:
            continue
        seen.add(key)
        merged.append(key)

    resolved: list[ResolvedGene] = [_resolve_gene(adata_like, g) for g in merged]
    return resolved, logic


def _is_myeloid_label(label: str) -> bool:
    low = str(label).lower()
    return bool(
        ("myeloid" in low)
        or ("macrophage" in low)
        or ("mono" in low)
        or ("dendritic" in low)
    )


def _build_scope_masks(
    adata: ad.AnnData,
    *,
    label_key: str | None,
    scope_mode: str,
    warnings_log: list[str],
) -> dict[str, np.ndarray]:
    n = int(adata.n_obs)
    masks: dict[str, np.ndarray] = {}
    if scope_mode in {"global", "both"}:
        masks["global"] = np.ones(n, dtype=bool)

    if scope_mode in {"non_myeloid", "both"}:
        if label_key is None or label_key not in adata.obs.columns:
            msg = "non_myeloid scope skipped: label key unavailable."
            warnings_log.append(msg)
            print(f"WARNING: {msg}")
        else:
            labels = adata.obs[label_key].astype("string").fillna("NA").astype(str)
            non_myeloid = ~labels.map(_is_myeloid_label).to_numpy(dtype=bool)
            if int(non_myeloid.sum()) == 0:
                msg = "non_myeloid scope skipped: no cells after excluding myeloid labels."
                warnings_log.append(msg)
                print(f"WARNING: {msg}")
            else:
                masks["non_myeloid"] = non_myeloid

    return masks


def _compute_signature_score(
    expr_matrix: Any,
    gene_indices: list[int],
    *,
    mode: str,
) -> np.ndarray:
    if len(gene_indices) == 0:
        return np.zeros(expr_matrix.shape[0], dtype=float)
    mat = expr_matrix[:, gene_indices]
    if sp.issparse(mat):
        arr = mat.toarray().astype(float)
    else:
        arr = np.asarray(mat, dtype=float)
    if mode == "sum":
        return np.sum(arr, axis=1).ravel().astype(float)
    if mode == "mean_log1p":
        return np.mean(np.log1p(np.maximum(arr, 0.0)), axis=1).ravel().astype(float)
    raise ValueError(f"Unsupported signature score mode: {mode}")


def _plot_profiles_grid(
    *,
    scope: str,
    name: str,
    values: np.ndarray,
    xy: np.ndarray,
    center_xy: np.ndarray,
    e_obs: np.ndarray,
    null_e: np.ndarray,
    null_t: np.ndarray,
    t_obs: float,
    out_png: Path,
    title_prefix: str,
    n_bins: int,
) -> None:
    fig = plt.figure(figsize=(14.8, 4.8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2, projection="polar")
    ax3 = fig.add_subplot(1, 3, 3)

    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ax1.scatter(
        xy[:, 0],
        xy[:, 1],
        c="#D1D1D1",
        s=4.0,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    sc = ax1.scatter(
        xy[order, 0],
        xy[order, 1],
        c=x[order],
        cmap="viridis",
        s=7.0,
        alpha=0.90,
        linewidths=0,
        rasterized=True,
    )
    ax1.scatter(
        [float(center_xy[0])],
        [float(center_xy[1])],
        marker="X",
        s=75,
        c="black",
        edgecolors="white",
        linewidths=0.8,
        zorder=10,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"{scope}: {name} on UMAP")
    fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03)

    centers = theta_bin_centers(int(n_bins))
    th = np.concatenate([centers, centers[:1]])
    obs_c = np.concatenate(
        [np.asarray(e_obs, dtype=float), np.asarray(e_obs[:1], dtype=float)]
    )
    q_hi = np.quantile(np.asarray(null_e, dtype=float), 0.95, axis=0)
    q_lo = np.quantile(np.asarray(null_e, dtype=float), 0.05, axis=0)
    qh = np.concatenate([q_hi, q_hi[:1]])
    ql = np.concatenate([q_lo, q_lo[:1]])
    ax2.plot(th, obs_c, color="#8B0000", linewidth=2.0, label="obs")
    ax2.plot(th, qh, color="#444444", linestyle="--", linewidth=1.2, label="null 95%")
    ax2.plot(th, ql, color="#444444", linestyle="--", linewidth=1.0, label="null 5%")
    ax2.fill_between(th, ql, qh, color="#B0B0B0", alpha=0.18)
    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(1)
    ax2.set_thetagrids(np.arange(0, 360, 90))
    ax2.set_title("RSP profile")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.18, 1.15), fontsize=8, frameon=True)

    bins = int(min(45, max(12, np.ceil(np.sqrt(np.asarray(null_t).size)))))
    ax3.hist(
        np.asarray(null_t, dtype=float),
        bins=bins,
        color="#779ECB",
        edgecolor="white",
        alpha=0.90,
    )
    ax3.axvline(float(t_obs), color="#8B0000", linestyle="--", linewidth=2.0)
    ax3.set_xlabel("null_T")
    ax3.set_ylabel("count")
    ax3.set_title("Null T distribution")

    fig.suptitle(f"{title_prefix}: {name}", y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight")
    plt.close(fig)


def _score_scope(
    *,
    scope_name: str,
    mask: np.ndarray,
    adata: ad.AnnData,
    embedding_xy: np.ndarray,
    expr_matrix: Any,
    resolved_genes: list[ResolvedGene],
    donor_ids_all: np.ndarray | None,
    donor_key_used: str | None,
    total_counts_all: np.ndarray,
    pct_mt_all: np.ndarray | None,
    pct_ribo_all: np.ndarray | None,
    contam_set_indices: dict[str, list[int]],
    sim_thresh: float,
    qc_thresh: float,
    n_bins: int,
    n_perm: int,
    seed: int,
    qc_profile_mode: str,
    out_qc_dir: Path,
    out_contam_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(int)
    if idx.size == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}, {}

    xy = np.asarray(embedding_xy, dtype=float)[idx, :2]
    center = compute_vantage_point(xy, method="median")
    theta = compute_theta(xy, center)
    _, bin_id = bin_theta(theta, int(n_bins))
    bin_counts_total = np.bincount(bin_id, minlength=int(n_bins)).astype(float)

    donor_ids_scope: np.ndarray | None = None
    donor_strat_available = False
    if donor_ids_all is not None:
        donor_ids_scope = np.asarray(donor_ids_all[idx]).astype(str)
        donor_strat_available = np.unique(donor_ids_scope).size >= 2
    if not donor_strat_available:
        donor_ids_scope = None

    total_counts = np.asarray(total_counts_all[idx], dtype=float)
    pct_mt = (
        np.asarray(pct_mt_all[idx], dtype=float) if pct_mt_all is not None else None
    )
    pct_ribo = (
        np.asarray(pct_ribo_all[idx], dtype=float) if pct_ribo_all is not None else None
    )

    # QC pseudo-feature continuous profiles.
    qc_rows: list[dict[str, Any]] = []
    qc_profiles: dict[str, np.ndarray] = {}
    qc_vectors: dict[str, np.ndarray | None] = {
        "total_counts": total_counts,
        "pct_counts_mt": pct_mt,
        "pct_counts_ribo": pct_ribo,
    }
    for qi, (qc_name, vals) in enumerate(qc_vectors.items()):
        if vals is None:
            qc_rows.append(
                {
                    "scope": scope_name,
                    "feature": qc_name,
                    "status": "missing",
                    "n_cells": int(idx.size),
                    "qc_profile_mode": qc_profile_mode,
                    "used_donor_stratified": False,
                    "T_obs": np.nan,
                    "p_T": np.nan,
                    "q_T": np.nan,
                    "Z_T": np.nan,
                }
            )
            continue
        weights = _zscore_log1p(np.asarray(vals, dtype=float))
        perm = _perm_null_continuous_profile(
            weights,
            donor_ids=donor_ids_scope,
            n_bins=int(n_bins),
            n_perm=int(n_perm),
            seed=int(seed + 10_000 + qi * 97),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
        null_e = np.asarray(perm["null_E_phi"], dtype=float)
        null_t = np.asarray(perm["null_T"], dtype=float)
        qc_profiles[qc_name] = e_obs
        qc_rows.append(
            {
                "scope": scope_name,
                "feature": qc_name,
                "status": "used",
                "n_cells": int(idx.size),
                "qc_profile_mode": qc_profile_mode,
                "used_donor_stratified": bool(perm["used_donor_stratified"]),
                "T_obs": float(perm["T_obs"]),
                "p_T": float(perm["p_T"]),
                "q_T": np.nan,
                "Z_T": float(robust_z(float(perm["T_obs"]), null_t)),
            }
        )
        _plot_profiles_grid(
            scope=scope_name,
            name=qc_name,
            values=np.asarray(vals, dtype=float),
            xy=xy,
            center_xy=np.asarray(center, dtype=float),
            e_obs=e_obs,
            null_e=null_e,
            null_t=null_t,
            t_obs=float(perm["T_obs"]),
            out_png=out_qc_dir / f"qc_{scope_name}_{qc_name}.png",
            title_prefix="QC pseudo-feature",
            n_bins=int(n_bins),
        )

    qc_df = pd.DataFrame(qc_rows)
    if not qc_df.empty:
        m = qc_df["status"] == "used"
        if int(m.sum()) > 0:
            qc_df.loc[m, "q_T"] = bh_fdr(
                pd.to_numeric(qc_df.loc[m, "p_T"], errors="coerce").to_numpy(
                    dtype=float
                )
            )

    # Contamination signature profiles.
    contam_rows: list[dict[str, Any]] = []
    contam_profiles: dict[str, np.ndarray] = {}
    contam_values_store: dict[str, np.ndarray] = {}
    contam_modes = {
        "mt_score": "sum",
        "ribo_score": "sum",
        "hemoglobin_score": "sum",
        "platelet_score": "sum",
        "myeloid_soup_score": "mean_log1p",
    }
    signature_map = {
        "mt_score": contam_set_indices.get("mt_genes", []),
        "ribo_score": contam_set_indices.get("ribo_genes", []),
        "hemoglobin_score": contam_set_indices.get("hemoglobin_genes", []),
        "platelet_score": contam_set_indices.get("platelet_genes", []),
        "myeloid_soup_score": contam_set_indices.get("myeloid_soup_genes", []),
    }
    for si, (sig, gidx_full) in enumerate(signature_map.items()):
        if len(gidx_full) == 0:
            contam_rows.append(
                {
                    "scope": scope_name,
                    "signature": sig,
                    "status": "missing",
                    "n_cells": int(idx.size),
                    "n_genes": 0,
                    "genes_used": "",
                    "used_donor_stratified": False,
                    "T_obs": np.nan,
                    "p_T": np.nan,
                    "q_T": np.nan,
                    "Z_T": np.nan,
                }
            )
            continue

        score = _compute_signature_score(
            expr_matrix[idx], gidx_full, mode=contam_modes[sig]
        )
        contam_values_store[sig] = score
        weights = _zscore_log1p(score)
        perm = _perm_null_continuous_profile(
            weights,
            donor_ids=donor_ids_scope,
            n_bins=int(n_bins),
            n_perm=int(n_perm),
            seed=int(seed + 20_000 + si * 131),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
        null_e = np.asarray(perm["null_E_phi"], dtype=float)
        null_t = np.asarray(perm["null_T"], dtype=float)
        contam_profiles[sig] = e_obs

        contam_rows.append(
            {
                "scope": scope_name,
                "signature": sig,
                "status": "used",
                "n_cells": int(idx.size),
                "n_genes": int(len(gidx_full)),
                "genes_used": ";".join(map(str, gidx_full)),
                "used_donor_stratified": bool(perm["used_donor_stratified"]),
                "T_obs": float(perm["T_obs"]),
                "p_T": float(perm["p_T"]),
                "q_T": np.nan,
                "Z_T": float(robust_z(float(perm["T_obs"]), null_t)),
            }
        )
        _plot_profiles_grid(
            scope=scope_name,
            name=sig,
            values=np.asarray(score, dtype=float),
            xy=xy,
            center_xy=np.asarray(center, dtype=float),
            e_obs=e_obs,
            null_e=null_e,
            null_t=null_t,
            t_obs=float(perm["T_obs"]),
            out_png=out_contam_dir / f"contam_{scope_name}_{sig}.png",
            title_prefix="Contamination signature",
            n_bins=int(n_bins),
        )

    contam_df = pd.DataFrame(contam_rows)
    if not contam_df.empty:
        m = contam_df["status"] == "used"
        if int(m.sum()) > 0:
            contam_df.loc[m, "q_T"] = bh_fdr(
                pd.to_numeric(contam_df.loc[m, "p_T"], errors="coerce").to_numpy(
                    dtype=float
                )
            )

    # Gene audit rows.
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    n_target_genes = sum(
        1 for g in resolved_genes if g.found and g.gene_idx is not None
    )
    scored_counter = 0
    resolved_lookup: dict[str, tuple[int, ResolvedGene]] = {}
    for gi, g in enumerate(resolved_genes):
        if not g.found or g.gene_idx is None:
            continue
        scored_counter += 1
        resolved_lookup[g.gene] = (gi, g)
        if (
            scored_counter == 1
            or scored_counter % 50 == 0
            or scored_counter == n_target_genes
        ):
            print(
                f"[{scope_name}] scoring {scored_counter}/{n_target_genes}: {g.gene}",
                flush=True,
            )
        expr_full = get_feature_vector(expr_matrix, int(g.gene_idx))
        expr = np.asarray(expr_full[idx], dtype=float)
        f = expr > 0.0
        prev = float(np.mean(f))
        n_fg = int(f.sum())

        if n_fg in {0, int(f.size)}:
            e_obs_direct = np.zeros(int(n_bins), dtype=float)
        else:
            e_obs_direct, _, _, _ = compute_rsp_profile_from_boolean(
                f,
                theta,
                int(n_bins),
                bin_id=bin_id,
                bin_counts_total=bin_counts_total,
            )

        perm = perm_null_T_and_profile(
            expr=np.asarray(expr, dtype=float),
            theta=theta,
            donor_ids=donor_ids_scope,
            n_bins=int(n_bins),
            n_perm=int(n_perm),
            seed=int(seed + 30_000 + gi * 23),
            donor_stratified=bool(donor_ids_scope is not None),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
        if e_obs.size != int(n_bins):
            e_obs = e_obs_direct
        null_e = np.asarray(perm["null_E_phi"], dtype=float)
        null_t = np.asarray(perm["null_T"], dtype=float)
        t_obs = float(perm["T_obs"])
        z_t = float(robust_z(t_obs, null_t))
        cov_c = float(coverage_from_null(e_obs, null_e, q=0.95))
        peaks_k = int(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95))
        p_idx = int(np.argmax(np.abs(e_obs))) if e_obs.size > 0 else 0
        centers = theta_bin_centers(int(n_bins))
        phi_hat = float(centers[p_idx]) if centers.size > 0 else 0.0

        rho_counts = _safe_spearman(f.astype(float), total_counts)
        rho_mt = _safe_spearman(f.astype(float), pct_mt)
        rho_ribo = _safe_spearman(f.astype(float), pct_ribo)
        finite_rho = [abs(v) for v in [rho_counts, rho_mt, rho_ribo] if np.isfinite(v)]
        qc_risk = float(max(finite_rho)) if finite_rho else 0.0

        sim_qc_vals: list[tuple[str, float]] = []
        for name, prof in qc_profiles.items():
            sim = _cosine_similarity(e_obs, prof)
            if np.isfinite(sim):
                sim_qc_vals.append((name, sim))
        if sim_qc_vals:
            best_qc_name, best_qc_sim = max(sim_qc_vals, key=lambda x: x[1])
        else:
            best_qc_name, best_qc_sim = "", float("nan")

        sim_contam_vals: list[tuple[str, float]] = []
        for name, prof in contam_profiles.items():
            sim = _cosine_similarity(e_obs, prof)
            if np.isfinite(sim):
                sim_contam_vals.append((name, sim))
        if sim_contam_vals:
            best_contam_name, best_contam_sim = max(sim_contam_vals, key=lambda x: x[1])
        else:
            best_contam_name, best_contam_sim = "", float("nan")

        row = {
            "gene": g.gene,
            "scope": scope_name,
            "prev": prev,
            "n_fg": n_fg,
            "n_cells": int(f.size),
            "T_obs": t_obs,
            "p_T": float(perm["p_T"]),
            "q_T": np.nan,
            "Z_T": z_t,
            "coverage_C": cov_c,
            "peaks_K": peaks_k,
            "phi_hat_rad": phi_hat,
            "phi_hat_deg": float(np.degrees(phi_hat) % 360.0),
            "score_1": z_t,
            "score_2": cov_c,
            "rho_counts": rho_counts,
            "rho_mt": rho_mt,
            "rho_ribo": rho_ribo,
            "qc_risk": qc_risk,
            "sim_qc": best_qc_sim,
            "sim_contam": best_contam_sim,
            "best_qc_feature": best_qc_name,
            "best_contam_signature": best_contam_name,
            "used_donor_stratified": bool(perm["used_donor_stratified"]),
            "donor_key_used": donor_key_used if donor_key_used is not None else "",
            "underpowered": bool((prev < 0.005) or (n_fg < 50)),
            "qc_driven": False,
            "contam_like": False,
            "class_label": "",
            "flag_category": "",
            "perm_warning": str(perm.get("warning", "")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, qc_df, contam_df, artifacts, qc_profiles, contam_profiles

    df["q_T"] = bh_fdr(df["p_T"].to_numpy(dtype=float))
    df["qc_driven"] = (df["q_T"] <= 0.05) & (
        (df["qc_risk"] >= float(qc_thresh)) | (df["sim_qc"] >= float(sim_thresh))
    )
    df["contam_like"] = (df["q_T"] <= 0.05) & (df["sim_contam"] >= float(sim_thresh))

    # Optional class label from shared scoring rules.
    df["class_label"] = df.apply(
        lambda r: classify_row(
            r,
            {
                "q_sig": 0.05,
                "high_prev": 0.6,
                "qc_thresh": float(qc_thresh),
            },
        ),
        axis=1,
    )

    def _flag_cat(r: pd.Series) -> str:
        sig = bool(float(r["q_T"]) <= 0.05)
        qc = bool(r["qc_driven"])
        ct = bool(r["contam_like"])
        if qc and ct:
            return "Both"
        if qc:
            return "QC-driven"
        if ct:
            return "Contam-like"
        if sig:
            return "Unflagged significant"
        return "Not significant"

    df["flag_category"] = df.apply(_flag_cat, axis=1)
    df = df.sort_values(
        by=["q_T", "gene"], ascending=[True, True], kind="mergesort"
    ).reset_index(drop=True)

    # Recompute full artifacts only for top flagged genes to keep memory bounded.
    top_qc = df.loc[df["qc_driven"]].sort_values(by="Z_T", ascending=False).head(15)
    top_ct = df.loc[df["contam_like"]].sort_values(by="Z_T", ascending=False).head(15)
    panel_genes = list(dict.fromkeys(top_qc["gene"].tolist() + top_ct["gene"].tolist()))
    for gene in panel_genes:
        if gene not in resolved_lookup:
            continue
        gi, g = resolved_lookup[gene]
        expr_full = get_feature_vector(expr_matrix, int(g.gene_idx))
        expr = np.asarray(expr_full[idx], dtype=float)
        perm = perm_null_T_and_profile(
            expr=np.asarray(expr, dtype=float),
            theta=theta,
            donor_ids=donor_ids_scope,
            n_bins=int(n_bins),
            n_perm=int(n_perm),
            seed=int(seed + 30_000 + gi * 23),
            donor_stratified=bool(donor_ids_scope is not None),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        artifacts[gene] = {
            "expr": expr,
            "E_phi_obs": np.asarray(perm["E_phi_obs"], dtype=float),
            "null_E_phi": np.asarray(perm["null_E_phi"], dtype=float),
            "null_T": np.asarray(perm["null_T"], dtype=float),
            "xy": xy,
            "center": np.asarray(center, dtype=float),
        }

    return df, qc_df, contam_df, artifacts, qc_profiles, contam_profiles


def _plot_overview(
    *,
    adata: ad.AnnData,
    umap_xy: np.ndarray,
    total_counts: np.ndarray,
    pct_mt: np.ndarray | None,
    pct_ribo: np.ndarray | None,
    donor_key: str | None,
    center_xy: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_numeric_umap(
        umap_xy=umap_xy,
        values=np.log1p(np.maximum(np.asarray(total_counts, dtype=float), 0.0)),
        out_png=out_dir / "umap_total_counts_log1p.png",
        title="UMAP: log1p(total_counts)",
        cmap="viridis",
        colorbar_label="log1p(total_counts)",
        vantage_point=(float(center_xy[0]), float(center_xy[1])),
    )
    if pct_mt is not None:
        save_numeric_umap(
            umap_xy=umap_xy,
            values=np.asarray(pct_mt, dtype=float),
            out_png=out_dir / "umap_pct_counts_mt.png",
            title="UMAP: pct_counts_mt",
            cmap="magma",
            colorbar_label="pct_counts_mt",
            vantage_point=(float(center_xy[0]), float(center_xy[1])),
        )
    else:
        _save_placeholder(
            out_dir / "umap_pct_counts_mt.png",
            "pct_counts_mt",
            "pct_counts_mt unavailable.",
        )

    if pct_ribo is not None:
        save_numeric_umap(
            umap_xy=umap_xy,
            values=np.asarray(pct_ribo, dtype=float),
            out_png=out_dir / "umap_pct_counts_ribo.png",
            title="UMAP: pct_counts_ribo",
            cmap="plasma",
            colorbar_label="pct_counts_ribo",
            vantage_point=(float(center_xy[0]), float(center_xy[1])),
        )
    else:
        _save_placeholder(
            out_dir / "umap_pct_counts_ribo.png",
            "pct_counts_ribo",
            "pct_counts_ribo unavailable.",
        )

    if donor_key is not None and donor_key in adata.obs.columns:
        plot_categorical_umap(
            umap_xy=umap_xy,
            labels=adata.obs[donor_key],
            title=f"UMAP by donor ({donor_key})",
            outpath=out_dir / "umap_by_donor.png",
            vantage_point=(float(center_xy[0]), float(center_xy[1])),
            annotate_cluster_medians=False,
        )
    else:
        _save_placeholder(
            out_dir / "umap_by_donor.png", "UMAP donor", "Donor key unavailable."
        )

    fig, axes = plt.subplots(1, 2 if pct_mt is not None else 1, figsize=(10.0, 4.0))
    if not isinstance(axes, np.ndarray):
        axes_arr = np.array([axes])
    else:
        axes_arr = axes
    axes_arr[0].hist(
        np.log1p(np.maximum(np.asarray(total_counts, dtype=float), 0.0)),
        bins=40,
        color="#5DA5DA",
        edgecolor="white",
    )
    axes_arr[0].set_title("log1p(total_counts)")
    axes_arr[0].set_xlabel("value")
    axes_arr[0].set_ylabel("count")
    if pct_mt is not None and len(axes_arr) > 1:
        axes_arr[1].hist(
            np.asarray(pct_mt, dtype=float), bins=40, color="#F17CB0", edgecolor="white"
        )
        axes_arr[1].set_title("pct_counts_mt")
        axes_arr[1].set_xlabel("value")
        axes_arr[1].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "qc_histograms.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)


def _plot_score_space(
    scope_df: pd.DataFrame,
    *,
    scope_name: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if scope_df.empty:
        _save_placeholder(
            out_dir / "score1_score2_scatter.png",
            f"{scope_name} score space",
            "No genes scored.",
        )
        _save_placeholder(
            out_dir / "qc_risk_vs_ZT.png",
            f"{scope_name} qc_risk vs Z",
            "No genes scored.",
        )
        _save_placeholder(
            out_dir / "sim_qc_vs_ZT.png",
            f"{scope_name} sim_qc vs Z",
            "No genes scored.",
        )
        _save_placeholder(
            out_dir / "sim_contam_vs_ZT.png",
            f"{scope_name} sim_contam vs Z",
            "No genes scored.",
        )
        _save_placeholder(
            out_dir / "similarity_map.png",
            f"{scope_name} similarity map",
            "No genes scored.",
        )
        return

    # 1) score space scatter by flag category.
    fig1, ax1 = plt.subplots(figsize=(8.3, 6.2))
    for cat, sub in scope_df.groupby("flag_category", observed=False):
        ax1.scatter(
            sub["score_1"].to_numpy(dtype=float),
            sub["score_2"].to_numpy(dtype=float),
            s=70,
            c=FLAG_COLORS.get(str(cat), "#333333"),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5,
            label=f"{cat} (n={sub.shape[0]})",
        )
    top = scope_df.sort_values(by="Z_T", ascending=False).head(20)
    for _, row in top.iterrows():
        ax1.text(
            float(row["score_1"]) + 0.04,
            float(row["score_2"]) + 0.004,
            str(row["gene"]),
            fontsize=7,
        )
    ax1.set_xlabel("score_1 = Z_T")
    ax1.set_ylabel("score_2 = coverage_C")
    ax1.set_title(f"{scope_name}: score space with control flags")
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    fig1.tight_layout()
    fig1.savefig(
        out_dir / "score1_score2_scatter.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
        bbox_inches="tight",
    )
    plt.close(fig1)

    # 2) qc_risk vs Z
    fig2, ax2 = plt.subplots(figsize=(7.6, 5.6))
    ax2.scatter(
        scope_df["qc_risk"].to_numpy(dtype=float),
        scope_df["Z_T"].to_numpy(dtype=float),
        c=[FLAG_COLORS.get(c, "#333333") for c in scope_df["flag_category"].tolist()],
        s=70,
        alpha=0.86,
        edgecolors="black",
        linewidths=0.5,
    )
    ax2.axvline(0.35, color="#8B0000", linestyle="--", linewidth=1.2)
    ax2.axhline(4.0, color="#404040", linestyle="-.", linewidth=1.2)
    ax2.set_xlabel("qc_risk")
    ax2.set_ylabel("Z_T")
    ax2.set_title(f"{scope_name}: qc_risk vs Z_T")
    ax2.grid(alpha=0.25, linewidth=0.6)
    fig2.tight_layout()
    fig2.savefig(out_dir / "qc_risk_vs_ZT.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)

    # 3) sim_qc / sim_contam vs Z_T
    for xcol, fname in [
        ("sim_qc", "sim_qc_vs_ZT.png"),
        ("sim_contam", "sim_contam_vs_ZT.png"),
    ]:
        mask = np.isfinite(scope_df[xcol].to_numpy(dtype=float))
        fig, ax = plt.subplots(figsize=(7.6, 5.6))
        sub = scope_df.loc[mask]
        ax.scatter(
            sub[xcol].to_numpy(dtype=float),
            sub["Z_T"].to_numpy(dtype=float),
            c=[FLAG_COLORS.get(c, "#333333") for c in sub["flag_category"].tolist()],
            s=70,
            alpha=0.86,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.axvline(0.7, color="#8B0000", linestyle="--", linewidth=1.2)
        ax.axhline(4.0, color="#404040", linestyle="-.", linewidth=1.2)
        ax.set_xlabel(xcol)
        ax.set_ylabel("Z_T")
        ax.set_title(f"{scope_name}: {xcol} vs Z_T")
        ax.grid(alpha=0.25, linewidth=0.6)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig)

    # 4) similarity map.
    mask = np.isfinite(scope_df["sim_qc"].to_numpy(dtype=float)) & np.isfinite(
        scope_df["sim_contam"].to_numpy(dtype=float)
    )
    fig4, ax4 = plt.subplots(figsize=(7.6, 5.8))
    sub = scope_df.loc[mask]
    ax4.scatter(
        sub["sim_qc"].to_numpy(dtype=float),
        sub["sim_contam"].to_numpy(dtype=float),
        c=[FLAG_COLORS.get(c, "#333333") for c in sub["flag_category"].tolist()],
        s=70,
        alpha=0.86,
        edgecolors="black",
        linewidths=0.5,
    )
    ax4.axvline(0.7, color="#8B0000", linestyle="--", linewidth=1.2)
    ax4.axhline(0.7, color="#8B0000", linestyle="--", linewidth=1.2)
    ax4.set_xlabel("sim_qc")
    ax4.set_ylabel("sim_contam")
    ax4.set_title(f"{scope_name}: QC-vs-contam similarity map")
    ax4.grid(alpha=0.25, linewidth=0.6)
    fig4.tight_layout()
    fig4.savefig(out_dir / "similarity_map.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig4)


def _plot_flagged_gene_panels(
    *,
    scope_name: str,
    scope_df: pd.DataFrame,
    artifacts: dict[str, dict[str, np.ndarray]],
    qc_profiles: dict[str, np.ndarray],
    contam_profiles: dict[str, np.ndarray],
    n_bins: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if scope_df.empty:
        _save_placeholder(
            out_dir / "no_flagged_genes.png",
            f"{scope_name} flagged genes",
            "No scored genes.",
        )
        return

    top_qc = (
        scope_df.loc[scope_df["qc_driven"]]
        .sort_values(by="Z_T", ascending=False)
        .head(15)
    )
    top_ct = (
        scope_df.loc[scope_df["contam_like"]]
        .sort_values(by="Z_T", ascending=False)
        .head(15)
    )
    genes = []
    seen: set[str] = set()
    for g in top_qc["gene"].tolist() + top_ct["gene"].tolist():
        if g in seen:
            continue
        seen.add(g)
        genes.append(g)
    if not genes:
        _save_placeholder(
            out_dir / "no_flagged_genes.png",
            f"{scope_name} flagged genes",
            "No QC-driven/contam-like genes.",
        )
        return

    for gene in genes:
        row = scope_df.loc[scope_df["gene"] == gene].iloc[0]
        if gene not in artifacts:
            continue
        art = artifacts[gene]
        expr = np.asarray(art["expr"], dtype=float)
        e_obs = np.asarray(art["E_phi_obs"], dtype=float)
        null_e = np.asarray(art["null_E_phi"], dtype=float)
        xy = np.asarray(art["xy"], dtype=float)
        center = np.asarray(art["center"], dtype=float)

        fig = plt.figure(figsize=(15.0, 4.8))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2, projection="polar")
        ax3 = fig.add_subplot(1, 3, 3, projection="polar")

        log_expr = np.log1p(np.maximum(expr, 0.0))
        order = np.argsort(log_expr, kind="mergesort")
        ax1.scatter(
            xy[:, 0],
            xy[:, 1],
            c="#D2D2D2",
            s=4.0,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        sc = ax1.scatter(
            xy[order, 0],
            xy[order, 1],
            c=log_expr[order],
            cmap="Reds",
            s=7.0,
            alpha=0.90,
            linewidths=0,
            rasterized=True,
        )
        ax1.scatter(
            [float(center[0])],
            [float(center[1])],
            marker="X",
            s=75,
            c="black",
            edgecolors="white",
            linewidths=0.8,
        )
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f"{gene}: feature map")
        fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03)

        centers = theta_bin_centers(int(n_bins))
        th = np.concatenate([centers, centers[:1]])
        obs_c = np.concatenate([e_obs, e_obs[:1]])
        q_hi = np.quantile(null_e, 0.95, axis=0)
        q_lo = np.quantile(null_e, 0.05, axis=0)
        ax2.plot(th, obs_c, color="#8B0000", linewidth=2.0, label="gene")
        ax2.plot(
            th,
            np.concatenate([q_hi, q_hi[:1]]),
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            label="null95",
        )
        ax2.plot(
            th,
            np.concatenate([q_lo, q_lo[:1]]),
            color="#444444",
            linestyle="--",
            linewidth=1.0,
            label="null5",
        )
        ax2.set_theta_zero_location("E")
        ax2.set_theta_direction(1)
        ax2.set_thetagrids(np.arange(0, 360, 90))
        ann = (
            f"q_T={float(row['q_T']):.2e}\n"
            f"Z_T={float(row['Z_T']):.2f}\n"
            f"sim_qc={float(row['sim_qc']):.2f}\n"
            f"sim_contam={float(row['sim_contam']):.2f}\n"
            f"qc_risk={float(row['qc_risk']):.2f}"
        )
        ax2.text(
            0.02,
            0.02,
            ann,
            transform=ax2.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.85},
        )
        ax2.set_title("Gene RSP + null")
        ax2.legend(
            loc="upper right", bbox_to_anchor=(1.2, 1.2), fontsize=8, frameon=True
        )

        # Overlay comparison with most similar control profiles.
        ax3.plot(th, obs_c, color="#8B0000", linewidth=2.0, label=f"{gene}")
        best_qc = str(row["best_qc_feature"])
        best_ct = str(row["best_contam_signature"])
        if best_qc in qc_profiles:
            qprof = np.asarray(qc_profiles[best_qc], dtype=float)
            ax3.plot(
                th,
                np.concatenate([qprof, qprof[:1]]),
                color="#1F77B4",
                linewidth=1.8,
                label=f"QC:{best_qc}",
            )
        if best_ct in contam_profiles:
            cprof = np.asarray(contam_profiles[best_ct], dtype=float)
            ax3.plot(
                th,
                np.concatenate([cprof, cprof[:1]]),
                color="#2CA02C",
                linewidth=1.8,
                label=f"Contam:{best_ct}",
            )
        ax3.set_theta_zero_location("E")
        ax3.set_theta_direction(1)
        ax3.set_thetagrids(np.arange(0, 360, 90))
        ax3.set_title("Profile overlay")
        ax3.legend(
            loc="upper right", bbox_to_anchor=(1.25, 1.2), fontsize=8, frameon=True
        )

        fig.suptitle(f"{scope_name}: flagged {gene}", y=1.01)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"flagged_{gene}.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


def _plot_comparisons(
    *,
    audits: dict[str, pd.DataFrame],
    marker_panel: set[str],
    out_dir: Path,
    seed: int,
    n_perm: int,
    n_bins: int,
    expr_matrix: Any,
    resolved_genes: list[ResolvedGene],
    theta_global: np.ndarray,
    donor_ids_global: np.ndarray | None,
    bin_id_global: np.ndarray,
    bin_counts_global: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "global" not in audits or audits["global"].empty:
        _save_placeholder(
            out_dir / "marker_vs_flagged.png",
            "Comparisons",
            "Global scope unavailable.",
        )
        _save_placeholder(
            out_dir / "myeloid_soup_sim_contam_global_vs_nonmyeloid.png",
            "Comparisons",
            "Data unavailable.",
        )
        _save_placeholder(
            out_dir / "random_gene_set_null.png", "Comparisons", "Data unavailable."
        )
        return

    gdf = audits["global"].copy()

    # 1) Marker panel vs flagged genes.
    fig1, ax1 = plt.subplots(figsize=(8.0, 6.0))
    non_marker = gdf.loc[~gdf["gene"].isin(marker_panel)]
    marker = gdf.loc[gdf["gene"].isin(marker_panel)]
    ax1.scatter(
        non_marker["Z_T"].to_numpy(dtype=float),
        non_marker["coverage_C"].to_numpy(dtype=float),
        c="#BDBDBD",
        s=55,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
        label="non-marker",
    )
    ax1.scatter(
        marker["Z_T"].to_numpy(dtype=float),
        marker["coverage_C"].to_numpy(dtype=float),
        c="#1F77B4",
        s=90,
        marker="D",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.6,
        label="Exp1 markers",
    )
    flagged = gdf.loc[gdf["qc_driven"] | gdf["contam_like"]]
    for _, row in flagged.sort_values(by="Z_T", ascending=False).head(12).iterrows():
        ax1.text(
            float(row["Z_T"]) + 0.04,
            float(row["coverage_C"]) + 0.003,
            str(row["gene"]),
            fontsize=8,
        )
    ax1.set_xlabel("Z_T")
    ax1.set_ylabel("coverage_C")
    ax1.set_title("Exp1 markers vs flagged genes (global)")
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.legend(loc="best", fontsize=8, frameon=True)
    fig1.tight_layout()
    fig1.savefig(out_dir / "marker_vs_flagged.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    # 2) Global vs non-myeloid sim_contam distributions for myeloid-soup candidates.
    if "non_myeloid" in audits and not audits["non_myeloid"].empty:
        ndf = audits["non_myeloid"].copy()
        candidates = set(
            [
                g
                for g in MYELOID_SOUP_GENES
                if g in set(gdf["gene"]) and g in set(ndf["gene"])
            ]
        )
        gvals = gdf.loc[gdf["gene"].isin(candidates), "sim_contam"].to_numpy(
            dtype=float
        )
        nvals = ndf.loc[ndf["gene"].isin(candidates), "sim_contam"].to_numpy(
            dtype=float
        )
        fig2, ax2 = plt.subplots(figsize=(6.8, 5.0))
        ax2.boxplot(
            [gvals, nvals],
            tick_labels=["global", "non_myeloid"],
            patch_artist=True,
            boxprops={"facecolor": "#9ecae1"},
        )
        rng = np.random.default_rng(0)
        ax2.scatter(
            np.full(gvals.size, 1.0) + rng.uniform(-0.05, 0.05, size=gvals.size),
            gvals,
            c="#1F77B4",
            s=35,
            alpha=0.8,
        )
        ax2.scatter(
            np.full(nvals.size, 2.0) + rng.uniform(-0.05, 0.05, size=nvals.size),
            nvals,
            c="#FF7F0E",
            s=35,
            alpha=0.8,
        )
        ax2.set_ylabel("sim_contam")
        ax2.set_title("Myeloid-soup candidate sim_contam: global vs non-myeloid")
        ax2.grid(axis="y", alpha=0.25, linewidth=0.6)
        fig2.tight_layout()
        fig2.savefig(
            out_dir / "myeloid_soup_sim_contam_global_vs_nonmyeloid.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
        )
        plt.close(fig2)
    else:
        _save_placeholder(
            out_dir / "myeloid_soup_sim_contam_global_vs_nonmyeloid.png",
            "Global vs non-myeloid sim_contam",
            "non_myeloid scope unavailable.",
        )

    # 3) Random gene-set controls (optional but included here).
    found_genes = [r for r in resolved_genes if r.found and r.gene_idx is not None]
    if len(found_genes) < 20:
        _save_placeholder(
            out_dir / "random_gene_set_null.png",
            "Random gene-set null",
            "Insufficient resolved genes for random set control.",
        )
    else:
        rng = np.random.default_rng(int(seed + 90_000))
        random_z: list[float] = []
        pool_idx = np.array([int(r.gene_idx) for r in found_genes], dtype=int)
        n_sets = 50
        set_size = 10
        for i in range(n_sets):
            if pool_idx.size < set_size:
                break
            chosen = rng.choice(pool_idx, size=set_size, replace=False)
            score = _compute_signature_score(
                expr_matrix, chosen.tolist(), mode="mean_log1p"
            )
            weights = _zscore_log1p(score)
            perm = _perm_null_continuous_profile(
                weights,
                donor_ids=donor_ids_global,
                n_bins=int(n_bins),
                n_perm=min(100, int(n_perm)),
                seed=int(seed + 100_000 + i * 37),
                bin_id=bin_id_global,
                bin_counts_total=bin_counts_global,
            )
            random_z.append(
                float(
                    robust_z(
                        float(perm["T_obs"]), np.asarray(perm["null_T"], dtype=float)
                    )
                )
            )
        fig3, ax3 = plt.subplots(figsize=(7.0, 5.0))
        ax3.hist(
            np.asarray(random_z, dtype=float),
            bins=18,
            color="#72B7B2",
            edgecolor="white",
            alpha=0.9,
        )
        ax3.set_xlabel("Z_T (random 10-gene signatures)")
        ax3.set_ylabel("count")
        ax3.set_title("Random signature null sanity check")
        ax3.grid(axis="y", alpha=0.25, linewidth=0.6)
        fig3.tight_layout()
        fig3.savefig(out_dir / "random_gene_set_null.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig3)


def main() -> int:
    args = parse_args()
    apply_plot_style()

    h5ad_path = Path(args.h5ad)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Input h5ad not found: {h5ad_path}")

    outdir = Path(args.out)
    tables_dir = outdir / "tables"
    plots_dir = outdir / "plots"
    p_overview = plots_dir / "00_overview"
    p_qc = plots_dir / "01_qc_profiles"
    p_contam = plots_dir / "02_contam_profiles"
    p_score = plots_dir / "03_score_space"
    p_flagged = plots_dir / "04_flagged_gene_panels"
    p_cmp = plots_dir / "05_comparisons"
    for d in [tables_dir, p_overview, p_qc, p_contam, p_score, p_flagged, p_cmp]:
        d.mkdir(parents=True, exist_ok=True)

    warnings_log: list[str] = []

    adata = ad.read_h5ad(h5ad_path)
    _ensure_umap(adata, seed=int(args.seed), recompute_if_missing=False)
    embedding_key, umap_xy = _resolve_embedding(adata, args.embedding_key)
    center_global = compute_vantage_point(umap_xy, method="median")
    theta_global = compute_theta(umap_xy, center_global)
    _, bin_id_global = bin_theta(theta_global, int(args.n_bins))
    bin_counts_global = np.bincount(bin_id_global, minlength=int(args.n_bins)).astype(
        float
    )

    expr_matrix, adata_like, expr_source = _choose_expression_source(
        adata,
        layer_arg=args.layer,
        use_raw_arg=bool(args.use_raw),
    )
    donor_ids_all, donor_key_used = _resolve_donor_ids_optional(adata, args.donor_key)
    if donor_ids_all is None:
        msg = "Donor key unavailable or <2 donors: using global permutations (no donor stratification)."
        warnings_log.append(msg)
        print(f"WARNING: {msg}")

    label_key = _resolve_label_key(adata, args.label_key)
    if label_key is None:
        msg = "No label key found; non_myeloid scope may be unavailable."
        warnings_log.append(msg)
        print(f"WARNING: {msg}")

    total_counts = _total_counts_vector(adata, expr_matrix)
    pct_mt_raw, pct_mt_source = _pct_mt_vector(adata, expr_matrix, adata_like)
    pct_mt = (
        None if pct_mt_source == "proxy:zeros" else np.asarray(pct_mt_raw, dtype=float)
    )
    pct_ribo, pct_ribo_source = _compute_pct_counts_ribo(
        adata, expr_matrix, adata_like, total_counts
    )
    if pct_mt is None:
        msg = "pct_counts_mt unavailable."
        warnings_log.append(msg)
        print(f"WARNING: {msg}")
    if pct_ribo is None:
        msg = "pct_counts_ribo unavailable."
        warnings_log.append(msg)
        print(f"WARNING: {msg}")

    print(f"embedding_key_used={embedding_key}")
    print(f"donor_key_used={donor_key_used if donor_key_used is not None else 'None'}")
    print(f"label_key_used={label_key if label_key is not None else 'None'}")
    print(f"expression_source_used={expr_source}")
    print(f"pct_counts_mt_source={pct_mt_source}")
    print(f"pct_counts_ribo_source={pct_ribo_source}")
    print(
        f"n_cells={int(adata.n_obs)} n_bins={int(args.n_bins)} n_perm={int(args.n_perm)} "
        f"max_genes={int(args.max_genes)} seed={int(args.seed)} scope={args.scope}"
    )

    _plot_overview(
        adata=adata,
        umap_xy=umap_xy,
        total_counts=np.asarray(total_counts, dtype=float),
        pct_mt=pct_mt,
        pct_ribo=pct_ribo,
        donor_key=donor_key_used,
        center_xy=np.asarray(center_global, dtype=float),
        out_dir=p_overview,
    )

    contam_sets_symbols = _build_contam_gene_sets(adata_like)
    contam_set_indices: dict[str, list[int]] = {}
    for key, genes in contam_sets_symbols.items():
        idxs: list[int] = []
        for g in genes:
            rg = _resolve_gene(adata_like, g)
            if rg.found and rg.gene_idx is not None:
                idxs.append(int(rg.gene_idx))
        contam_set_indices[key] = sorted(set(idxs))

    gene_subset_path = Path(args.gene_subset) if args.gene_subset is not None else None
    if gene_subset_path is not None and not gene_subset_path.exists():
        raise FileNotFoundError(f"--gene_subset file not found: {gene_subset_path}")

    resolved_genes, eval_logic = _build_eval_genes(
        adata_like=adata_like,
        expr_matrix=expr_matrix,
        max_genes=int(args.max_genes),
        gene_subset_path=gene_subset_path,
        contam_sets=contam_sets_symbols,
    )
    n_found = sum(1 for g in resolved_genes if g.found)
    n_missing = sum(1 for g in resolved_genes if not g.found)
    print(
        f"evaluation_genes_total={len(resolved_genes)} found={n_found} missing={n_missing} logic={eval_logic}"
    )

    scope_masks = _build_scope_masks(
        adata,
        label_key=label_key,
        scope_mode=str(args.scope),
        warnings_log=warnings_log,
    )
    if len(scope_masks) == 0:
        raise RuntimeError(
            "No scopes available to run after applying scope/label rules."
        )

    audits: dict[str, pd.DataFrame] = {}
    scope_qc_tables: list[pd.DataFrame] = []
    scope_contam_tables: list[pd.DataFrame] = []
    scope_artifacts: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    scope_qc_profiles: dict[str, dict[str, np.ndarray]] = {}
    scope_contam_profiles: dict[str, dict[str, np.ndarray]] = {}

    qc_profile_mode = "continuous_weighted"

    for si, (scope_name, mask) in enumerate(scope_masks.items()):
        scope_df, qc_df, contam_df, art, qprof, cprof = _score_scope(
            scope_name=scope_name,
            mask=mask,
            adata=adata,
            embedding_xy=umap_xy,
            expr_matrix=expr_matrix,
            resolved_genes=resolved_genes,
            donor_ids_all=donor_ids_all,
            donor_key_used=donor_key_used,
            total_counts_all=np.asarray(total_counts, dtype=float),
            pct_mt_all=pct_mt,
            pct_ribo_all=pct_ribo,
            contam_set_indices=contam_set_indices,
            sim_thresh=float(args.sim_thresh),
            qc_thresh=float(args.qc_thresh),
            n_bins=int(args.n_bins),
            n_perm=int(args.n_perm),
            seed=int(args.seed + si * 100_000),
            qc_profile_mode=qc_profile_mode,
            out_qc_dir=p_qc,
            out_contam_dir=p_contam,
        )
        audits[scope_name] = scope_df
        scope_qc_tables.append(qc_df)
        scope_contam_tables.append(contam_df)
        scope_artifacts[scope_name] = art
        scope_qc_profiles[scope_name] = qprof
        scope_contam_profiles[scope_name] = cprof

        # Scope score-space and flagged panels.
        _plot_score_space(scope_df, scope_name=scope_name, out_dir=p_score / scope_name)
        _plot_flagged_gene_panels(
            scope_name=scope_name,
            scope_df=scope_df,
            artifacts=art,
            qc_profiles=qprof,
            contam_profiles=cprof,
            n_bins=int(args.n_bins),
            out_dir=p_flagged / scope_name,
        )

    qc_features_df = (
        pd.concat(scope_qc_tables, ignore_index=True)
        if scope_qc_tables
        else pd.DataFrame()
    )
    contam_features_df = (
        pd.concat(scope_contam_tables, ignore_index=True)
        if scope_contam_tables
        else pd.DataFrame()
    )
    tables_dir.mkdir(parents=True, exist_ok=True)
    qc_features_csv = tables_dir / "qc_features_used.csv"
    contam_features_csv = tables_dir / "contam_signatures_used.csv"
    qc_features_df.to_csv(qc_features_csv, index=False)
    contam_features_df.to_csv(contam_features_csv, index=False)

    global_audit = audits.get("global", pd.DataFrame())
    non_myeloid_audit = audits.get("non_myeloid", pd.DataFrame())
    global_csv = tables_dir / "gene_audit_global.csv"
    non_myeloid_csv = tables_dir / "gene_audit_non_myeloid.csv"
    if not global_audit.empty:
        global_audit.to_csv(global_csv, index=False)
    else:
        pd.DataFrame().to_csv(global_csv, index=False)
    if "non_myeloid" in audits:
        non_myeloid_audit.to_csv(non_myeloid_csv, index=False)

    all_audits = (
        pd.concat([df for df in audits.values() if not df.empty], ignore_index=True)
        if audits
        else pd.DataFrame()
    )
    flagged_qc = (
        all_audits.loc[all_audits["qc_driven"]].copy()
        if not all_audits.empty
        else pd.DataFrame()
    )
    flagged_ct = (
        all_audits.loc[all_audits["contam_like"]].copy()
        if not all_audits.empty
        else pd.DataFrame()
    )
    flagged_qc_csv = tables_dir / "flagged_qc_driven_genes.csv"
    flagged_ct_csv = tables_dir / "flagged_contam_like_genes.csv"
    flagged_qc.to_csv(flagged_qc_csv, index=False)
    flagged_ct.to_csv(flagged_ct_csv, index=False)

    _plot_comparisons(
        audits=audits,
        marker_panel=set(MARKER_PANEL_EXP1),
        out_dir=p_cmp,
        seed=int(args.seed),
        n_perm=int(args.n_perm),
        n_bins=int(args.n_bins),
        expr_matrix=expr_matrix,
        resolved_genes=resolved_genes,
        theta_global=theta_global,
        donor_ids_global=donor_ids_all,
        bin_id_global=bin_id_global,
        bin_counts_global=bin_counts_global,
    )

    # README metadata and run flags.
    readme_lines = [
        "Experiment #5: Negative controls audit (QC + contamination signatures)",
        "",
        "Objective: identify genes whose embedding-localized BioRSP geometry may be QC-/contamination-driven.",
        "",
        f"embedding_key_used: {embedding_key}",
        f"donor_key_used: {donor_key_used if donor_key_used is not None else 'None'}",
        f"label_key_used: {label_key if label_key is not None else 'None'}",
        f"expression_source_used: {expr_source}",
        f"n_cells_total: {int(adata.n_obs)}",
        f"n_bins: {int(args.n_bins)}",
        f"n_perm: {int(args.n_perm)}",
        f"max_genes: {int(args.max_genes)}",
        f"scope_mode: {args.scope}",
        f"evaluation_logic: {eval_logic}",
        f"qc_profile_mode: {qc_profile_mode}",
        "",
        "Scopes run:",
    ]
    for k in audits.keys():
        readme_lines.append(f"- {k}: n_genes_scored={int(audits[k].shape[0])}")
    readme_lines.append("")
    readme_lines.append("Warnings:")
    if warnings_log:
        for msg in warnings_log:
            readme_lines.append(f"- {msg}")
    else:
        readme_lines.append("- none")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"qc_features_csv={qc_features_csv.as_posix()}")
    print(f"contam_features_csv={contam_features_csv.as_posix()}")
    print(f"gene_audit_global_csv={global_csv.as_posix()}")
    if "non_myeloid" in audits:
        print(f"gene_audit_non_myeloid_csv={non_myeloid_csv.as_posix()}")
    print(f"flagged_qc_csv={flagged_qc_csv.as_posix()} rows={int(flagged_qc.shape[0])}")
    print(
        f"flagged_contam_csv={flagged_ct_csv.as_posix()} rows={int(flagged_ct.shape[0])}"
    )
    print(f"results_root={outdir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
