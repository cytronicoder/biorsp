#!/usr/bin/env python3
"""Experiment #8: genome-wide triage with calibrated BioRSP nulls and QC/replication gates."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

# Use headless backend for deterministic local/CI plotting.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

# Allow direct script execution from repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from biorsp.stats.moran import morans_i
from biorsp.stats.permutation import perm_null_T_and_profile
from biorsp.stats.scoring import bh_fdr, coverage_from_null, peak_count, robust_z

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

QC_COVAR_CANDIDATES = {
    "total_counts": ["total_counts", "n_counts", "nCount_RNA", "nUMI", "total_umis"],
    "pct_counts_mt": ["pct_counts_mt", "percent.mt", "pct_mt"],
    "pct_counts_ribo": ["pct_counts_ribo", "percent.ribo", "pct_ribo"],
}

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

CONTAM_QC_CANDIDATES = [
    "HBA1",
    "HBA2",
    "HBB",
    "HBD",
    "PPBP",
    "PF4",
    "LYZ",
    "S100A8",
    "S100A9",
    "LST1",
    "TYROBP",
]

MODE_A = "A_binary"
MODE_B = "B_donor_quantile"
MODE_ORDER = [MODE_A, MODE_B]

CLASS_ORDER = [
    "Localized–unimodal",
    "Localized–multimodal",
    "QC-driven",
    "Ubiquitous (non-localized)",
    "Underpowered",
    "Uncertain",
]

CLASS_COLORS = {
    "Localized–unimodal": "#1F77B4",
    "Localized–multimodal": "#FF7F0E",
    "QC-driven": "#D62728",
    "Ubiquitous (non-localized)": "#2CA02C",
    "Underpowered": "#8C8C8C",
    "Uncertain": "#9467BD",
}

Q_SIG = 0.05
P_MIN = 0.005
MIN_FG = 50
HIGH_PREV = 0.60
QC_RISK_THRESH = 0.35
SIM_QC_THRESH = 0.70


@dataclass(frozen=True)
class ResolvedGene:
    requested_gene: str
    found: bool
    gene_idx: int | None
    resolved_gene: str
    status: str
    source: str
    symbol_column: str


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
        description="Run Experiment #8 genome-wide triage for BioRSP localization modes."
    )
    p.add_argument(
        "--h5ad",
        default="data/processed/HT_pca_umap.h5ad",
        help="Input .h5ad path.",
    )
    p.add_argument(
        "--out",
        default="experiments/heart_smoketest/results/experiment8_genomewide_triage",
        help="Output directory.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--n_perm", type=int, default=200, help="Permutation count.")
    p.add_argument("--n_bins", type=int, default=64, help="Angular bins.")
    p.add_argument("--k", type=int, default=30, help="kNN neighbors for Moran baseline.")
    p.add_argument(
        "--mode",
        choices=["hvg", "top_detected", "list"],
        default="hvg",
        help="Gene evaluation mode.",
    )
    p.add_argument("--max_genes", type=int, default=5000, help="Gene cap for hvg/top_detected modes.")
    p.add_argument("--gene_list", default=None, help="Path to newline-separated gene list for mode=list.")
    p.add_argument("--q", type=float, default=0.10, help="Top quantile for donor-quantile foreground mode B.")
    p.add_argument(
        "--run_de_baseline",
        type=_str2bool,
        default=False,
        help="Run optional DE baseline (scanpy rank_genes_groups).",
    )
    p.add_argument("--embedding_key", default=None, help="Optional embedding key override.")
    p.add_argument("--donor_key", default=None, help="Optional donor key override.")
    p.add_argument("--label_key", default=None, help="Optional label key override.")
    p.add_argument("--layer", default=None, help="Optional expression layer override.")
    p.add_argument("--use_raw", action="store_true", help="Use adata.raw instead of X/layers.")
    p.add_argument(
        "--top_candidates",
        type=int,
        default=100,
        help="Top localized candidates to write into top_candidates_localized.csv.",
    )
    return p.parse_args()


def _save_placeholder(out_png: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)


def _sanitize_name(text: str, max_len: int = 96) -> str:
    s = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(text))
    s = s.strip("_")
    if s == "":
        s = "id"
    return s[: int(max_len)]


def _resolve_embedding(
    adata: ad.AnnData, requested_key: str | None
) -> tuple[str, np.ndarray]:
    if requested_key is not None:
        if requested_key not in adata.obsm:
            raise KeyError(f"Requested embedding key '{requested_key}' missing in adata.obsm.")
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
    adata: ad.AnnData,
    requested: str | None,
    candidates: list[str],
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
    key = _resolve_key(adata, requested_key, DONOR_CANDIDATES)
    if key is None:
        return None, None
    ids = adata.obs[key].astype("string").fillna("NA").astype(str).to_numpy()
    if np.unique(ids).size < 2:
        return None, key
    return ids, key


def _resolve_label_key_optional(
    adata: ad.AnnData,
    requested_key: str | None,
) -> tuple[np.ndarray | None, str | None]:
    key = _resolve_key(adata, requested_key, LABEL_KEY_CANDIDATES)
    if key is None:
        return None, None
    vals = adata.obs[key].astype("string").fillna("NA").astype(str).to_numpy()
    return vals, key


def _compute_pct_counts_ribo(
    adata: ad.AnnData,
    expr_matrix: Any,
    adata_like: Any,
    total_counts: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    if "pct_counts_ribo" in adata.obs.columns:
        vals = pd.to_numeric(adata.obs["pct_counts_ribo"], errors="coerce").to_numpy(dtype=float)
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

    symbols = adata_like.var[symbol_col].astype("string").fillna("").astype(str).str.upper()
    ribo_mask = (symbols.str.startswith("RPL") | symbols.str.startswith("RPS")).to_numpy(dtype=bool)
    if int(ribo_mask.sum()) == 0:
        return None, "missing"

    ribo_counts = np.asarray(expr_matrix[:, ribo_mask].sum(axis=1)).ravel().astype(float)
    pct = np.divide(ribo_counts, np.maximum(np.asarray(total_counts, dtype=float), 1e-12)) * 100.0
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
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x)
    fill = float(np.nanmedian(x[finite]))
    x = np.where(np.isfinite(x), x, fill)
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
    t_obs = float(np.max(np.abs(e_obs)))

    used_donor = bool(donor_ids is not None and np.unique(np.asarray(donor_ids).astype(str)).size >= 2)
    warning_msg = ""
    if not used_donor:
        warning_msg = "continuous null used global shuffling (donor unavailable)."

    rng = np.random.default_rng(int(seed))
    null_e = np.zeros((int(n_perm), int(n_bins)), dtype=float)
    null_t = np.zeros(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        if used_donor and donor_ids is not None:
            pvals = _permute_weights_within_donor(x, np.asarray(donor_ids).astype(str), rng)
        else:
            pvals = x[rng.permutation(x.size)]
        e_perm = _compute_continuous_profile(
            pvals,
            n_bins=int(n_bins),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        null_e[i, :] = e_perm
        null_t[i] = float(np.max(np.abs(e_perm)))

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
            requested_gene=gene,
            found=True,
            gene_idx=int(idx),
            resolved_gene=str(label),
            status="resolved",
            source=str(source),
            symbol_column=str(symbol_col or ""),
        )
    except KeyError:
        return ResolvedGene(
            requested_gene=gene,
            found=False,
            gene_idx=None,
            resolved_gene="",
            status="missing",
            source="",
            symbol_column="",
        )


def _read_gene_list(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for ln in lines:
        g = ln.strip()
        if g == "" or g.startswith("#"):
            continue
        out.append(g)
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
        g = sym if sym != "" else var_names[int(i)]
        if g in seen:
            continue
        seen.add(g)
        out.append(g)
        if len(out) >= int(max_genes):
            break
    return out


def _top_detected_genes(
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
        g = sym if sym != "" else var_names[int(idx)]
        if g in seen:
            continue
        seen.add(g)
        out.append(g)
        if len(out) >= int(max_genes):
            break
    return out


def _build_eval_gene_set(
    *,
    mode: str,
    max_genes: int,
    gene_list_path: Path | None,
    expr_matrix: Any,
    adata_like: Any,
) -> tuple[list[ResolvedGene], str]:
    if mode == "list":
        if gene_list_path is None:
            raise ValueError("--gene_list is required when --mode=list")
        if not gene_list_path.exists():
            raise FileNotFoundError(f"--gene_list file not found: {gene_list_path}")
        base = _read_gene_list(gene_list_path)
        logic = f"list_file:{gene_list_path.as_posix()}"
    elif mode == "hvg":
        hv = _hvg_genes(adata_like, int(max_genes))
        if len(hv) > 0:
            base = hv[: int(max_genes)]
            logic = f"hvg_cap_{int(max_genes)}"
        else:
            base = _top_detected_genes(expr_matrix, adata_like, int(max_genes))
            logic = f"hvg_missing_fallback_top_detected_cap_{int(max_genes)}"
    else:
        base = _top_detected_genes(expr_matrix, adata_like, int(max_genes))
        logic = f"top_detected_cap_{int(max_genes)}"

    force_include = list(MARKER_PANEL_EXP1) + list(CONTAM_QC_CANDIDATES)

    merged: list[str] = []
    seen: set[str] = set()
    for g in list(base) + force_include:
        key = str(g).strip()
        if key == "" or key in seen:
            continue
        seen.add(key)
        merged.append(key)

    resolved = [_resolve_gene(adata_like, g) for g in merged]
    return resolved, logic


def _build_foreground_donor_quantile(
    expr: np.ndarray,
    donor_ids: np.ndarray | None,
    q: float,
) -> np.ndarray:
    x = np.asarray(expr, dtype=float).ravel()
    if donor_ids is None:
        thr = float(np.nanquantile(x, max(0.0, 1.0 - float(q))))
        return x >= thr

    donor_arr = np.asarray(donor_ids).astype(str)
    out = np.zeros(x.size, dtype=bool)
    for donor in np.unique(donor_arr):
        idx = np.flatnonzero(donor_arr == donor)
        if idx.size == 0:
            continue
        xd = x[idx]
        thr = float(np.nanquantile(xd, max(0.0, 1.0 - float(q))))
        out[idx] = xd >= thr
    return out


def _circular_diff_deg(a_deg: float, b_deg: float) -> float:
    a = float(a_deg) % 360.0
    b = float(b_deg) % 360.0
    return float(abs(((a - b + 180.0) % 360.0) - 180.0))


def _build_knn_weights(
    xy: np.ndarray,
    k: int,
) -> csr_matrix:
    xy_arr = np.asarray(xy, dtype=float)
    n = int(xy_arr.shape[0])
    if n < 3:
        raise ValueError("Need at least 3 cells to build kNN graph.")
    k_use = int(min(max(2, int(k)), max(2, n - 1)))
    nn = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean")
    nn.fit(xy_arr)
    idx = nn.kneighbors(xy_arr, return_distance=False)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n):
        for j in idx[i, 1:]:
            rows.append(i)
            cols.append(int(j))
            vals.append(1.0)

    w = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)
    w = w.maximum(w.T)
    w.setdiag(0.0)
    w.eliminate_zeros()
    return w


def _compute_moran_baseline(
    *,
    resolved_genes: list[ResolvedGene],
    expr_matrix: Any,
    w: csr_matrix,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_genes = sum(1 for g in resolved_genes if g.found and g.gene_idx is not None)
    counter = 0

    for g in resolved_genes:
        if not g.found or g.gene_idx is None:
            continue
        counter += 1
        if counter == 1 or counter % 200 == 0 or counter == n_genes:
            print(f"[Moran] scoring {counter}/{n_genes}: {g.requested_gene}", flush=True)
        x = np.asarray(get_feature_vector(expr_matrix, int(g.gene_idx)), dtype=float)
        x = np.log1p(np.maximum(x, 0.0))
        try:
            m = float(morans_i(x, w, row_standardize=True))
        except Exception:
            m = float("nan")
        rows.append({"gene": g.requested_gene, "moran_I": m, "moran_method": "biorsp.stats.moran"})
    return pd.DataFrame(rows)


def _run_de_baseline(
    *,
    adata: ad.AnnData,
    label_key: str | None,
    run_flag: bool,
    seed: int,
) -> tuple[pd.DataFrame, str, list[str]]:
    warnings_log: list[str] = []
    if not bool(run_flag):
        return pd.DataFrame(columns=["gene", "de_rank", "de_group"]), "disabled", warnings_log
    if label_key is None or label_key not in adata.obs.columns:
        warnings_log.append("DE baseline requested but no label key available; skipped.")
        return pd.DataFrame(columns=["gene", "de_rank", "de_group"]), "skipped_no_label", warnings_log

    try:
        import scanpy as sc  # type: ignore
    except Exception as exc:
        warnings_log.append(f"DE baseline skipped: scanpy import failed ({exc}).")
        return pd.DataFrame(columns=["gene", "de_rank", "de_group"]), "scanpy_unavailable", warnings_log

    try:
        ad = adata.copy()
        sc.tl.rank_genes_groups(
            ad,
            groupby=label_key,
            method="wilcoxon",
            n_genes=min(200, int(ad.n_vars)),
            tie_correct=True,
            pts=False,
            random_state=int(seed),
        )
        rg = ad.uns.get("rank_genes_groups", {})
        names = rg.get("names", None)
        if names is None:
            warnings_log.append("DE baseline skipped: rank_genes_groups produced no names.")
            return pd.DataFrame(columns=["gene", "de_rank", "de_group"]), "de_empty", warnings_log

        best: dict[str, tuple[int, str]] = {}
        groups = names.dtype.names if hasattr(names, "dtype") and names.dtype.names is not None else []
        for grp in groups:
            arr = np.asarray(names[grp]).astype(str)
            for r, gene in enumerate(arr, start=1):
                if gene == "" or gene.lower() == "nan":
                    continue
                if gene not in best or r < best[gene][0]:
                    best[gene] = (int(r), str(grp))
        rows = [
            {"gene": g, "de_rank": int(v[0]), "de_group": str(v[1])}
            for g, v in best.items()
        ]
        return pd.DataFrame(rows), "scanpy.rank_genes_groups.wilcoxon", warnings_log
    except Exception as exc:
        warnings_log.append(f"DE baseline failed ({exc}); skipped.")
        return pd.DataFrame(columns=["gene", "de_rank", "de_group"]), "de_failed", warnings_log


def _plot_overview(
    *,
    adata: ad.AnnData,
    umap_xy: np.ndarray,
    donor_key: str | None,
    label_key: str | None,
    total_counts: np.ndarray,
    pct_mt: np.ndarray | None,
    pct_ribo: np.ndarray | None,
    center_xy: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

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
        _save_placeholder(out_dir / "umap_by_donor.png", "UMAP donor", "Donor key unavailable.")

    if label_key is not None and label_key in adata.obs.columns:
        plot_categorical_umap(
            umap_xy=umap_xy,
            labels=adata.obs[label_key],
            title=f"UMAP by label ({label_key})",
            outpath=out_dir / "umap_by_label.png",
            vantage_point=(float(center_xy[0]), float(center_xy[1])),
            annotate_cluster_medians=False,
        )
    else:
        _save_placeholder(out_dir / "umap_by_label.png", "UMAP label", "Label key unavailable.")

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
        _save_placeholder(out_dir / "umap_pct_counts_mt.png", "pct_counts_mt", "pct_counts_mt unavailable.")

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
        _save_placeholder(out_dir / "umap_pct_counts_ribo.png", "pct_counts_ribo", "pct_counts_ribo unavailable.")


def _plot_distributions(
    *,
    mode_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode_df.empty:
        _save_placeholder(out_dir / "distributions.png", "Distributions", "No scored genes.")
        return

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 13.0))

    for mode, color in [(MODE_A, "#1F77B4"), (MODE_B, "#FF7F0E")]:
        sub = mode_df.loc[mode_df["mode"] == mode]
        axes[0, 0].hist(sub["Z_T"].to_numpy(dtype=float), bins=40, alpha=0.55, label=mode, color=color)
        q = np.maximum(sub["q_T"].to_numpy(dtype=float), 1e-300)
        axes[0, 1].hist(-np.log10(q), bins=40, alpha=0.55, label=mode, color=color)
        axes[1, 0].hist(sub["coverage_C"].to_numpy(dtype=float), bins=35, alpha=0.55, label=mode, color=color)
        axes[1, 1].hist(sub["peaks_K"].to_numpy(dtype=float), bins=np.arange(-0.5, 8.5, 1.0), alpha=0.55, label=mode, color=color)
        axes[2, 0].hist(sub["prev"].to_numpy(dtype=float), bins=35, alpha=0.55, label=mode, color=color)

    axes[0, 0].set_title("Z_T distribution")
    axes[0, 0].set_xlabel("Z_T")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("-log10(q_T) distribution")
    axes[0, 1].set_xlabel("-log10(q_T)")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_title("coverage_C distribution")
    axes[1, 0].set_xlabel("coverage_C")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("peaks_K distribution")
    axes[1, 1].set_xlabel("peaks_K")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend(fontsize=8)

    axes[2, 0].set_title("Prevalence distribution")
    axes[2, 0].set_xlabel("prevalence")
    axes[2, 0].set_ylabel("count")
    axes[2, 0].legend(fontsize=8)

    counts = summary_df["final_class"].value_counts().reindex(CLASS_ORDER).fillna(0.0)
    x = np.arange(counts.shape[0], dtype=float)
    axes[2, 1].bar(x, counts.to_numpy(dtype=float), color=[CLASS_COLORS.get(c, "#888888") for c in counts.index.tolist()], edgecolor="black", linewidth=0.5)
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(counts.index.tolist(), rotation=30, ha="right", fontsize=8)
    axes[2, 1].set_ylabel("count")
    axes[2, 1].set_title("Final class counts")

    for ax in axes.ravel():
        ax.grid(alpha=0.22, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(out_dir / "score_distributions.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)


def _plot_score_space(
    *,
    summary_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        _save_placeholder(out_dir / "score1_vs_score2_scatter.png", "Score space", "No scored genes.")
        return

    fig1, ax1 = plt.subplots(figsize=(8.4, 6.3))
    for cls, sub in summary_df.groupby("final_class", observed=False):
        ax1.scatter(
            sub["median_Z"].to_numpy(dtype=float),
            sub["median_coverage"].to_numpy(dtype=float),
            s=62,
            alpha=0.85,
            c=CLASS_COLORS.get(str(cls), "#666666"),
            edgecolors="black",
            linewidths=0.4,
            label=f"{cls} (n={sub.shape[0]})",
        )
    top = summary_df.sort_values(by="triage_score", ascending=False).head(20)
    for _, row in top.iterrows():
        ax1.text(float(row["median_Z"]) + 0.04, float(row["median_coverage"]) + 0.003, str(row["gene"]), fontsize=7)
    ax1.set_xlabel("score_1 = median_Z")
    ax1.set_ylabel("score_2 = median_coverage")
    ax1.set_title("Genome-wide triage score space")
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    fig1.tight_layout()
    fig1.savefig(out_dir / "score1_vs_score2_scatter.png", dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8.0, 5.8))
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=summary_df.shape[0])
    x = summary_df["peaks_K_median"].to_numpy(dtype=float) + jitter
    y = summary_df["median_Z"].to_numpy(dtype=float)
    c = [CLASS_COLORS.get(v, "#666666") for v in summary_df["final_class"].astype(str).tolist()]
    ax2.scatter(x, y, c=c, s=60, alpha=0.85, edgecolors="black", linewidths=0.4)
    ax2.set_xlabel("peaks_K_median")
    ax2.set_ylabel("median_Z")
    ax2.set_title("Unimodal vs multimodal boundary")
    ax2.grid(alpha=0.25, linewidth=0.6)
    fig2.tight_layout()
    fig2.savefig(out_dir / "peaksK_vs_medianZ.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(7.6, 6.0))
    xa = summary_df["Z_T_A"].to_numpy(dtype=float)
    yb = summary_df["Z_T_B"].to_numpy(dtype=float)
    ax3.scatter(
        xa,
        yb,
        c=[CLASS_COLORS.get(v, "#666666") for v in summary_df["final_class"].astype(str).tolist()],
        s=60,
        alpha=0.86,
        edgecolors="black",
        linewidths=0.4,
    )
    lim = max(float(np.nanmax(xa)), float(np.nanmax(yb)), 1.0)
    ax3.plot([0, lim], [0, lim], linestyle="--", color="#555555", linewidth=1.3)
    unstable = summary_df.loc[(summary_df["sig_A"] ^ summary_df["sig_B"])].sort_values(by="median_Z", ascending=False).head(20)
    for _, row in unstable.iterrows():
        ax3.text(float(row["Z_T_A"]) + 0.03, float(row["Z_T_B"]) + 0.03, str(row["gene"]), fontsize=7)
    ax3.set_xlabel("Z_T_A (binary)")
    ax3.set_ylabel("Z_T_B (donor-quantile)")
    ax3.set_title("Foreground robustness: mode A vs mode B")
    ax3.grid(alpha=0.25, linewidth=0.6)
    fig3.tight_layout()
    fig3.savefig(out_dir / "robustness_ZA_vs_ZB.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig3)


def _plot_baseline_comparisons(
    *,
    summary_df: pd.DataFrame,
    de_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        _save_placeholder(out_dir / "MoranI_vs_medianZ.png", "Baseline comparison", "No scored genes.")
        return

    fig1, ax1 = plt.subplots(figsize=(8.2, 6.0))
    mask = np.isfinite(summary_df["moran_I"].to_numpy(dtype=float))
    sub = summary_df.loc[mask].copy()
    ax1.scatter(
        sub["moran_I"].to_numpy(dtype=float),
        sub["median_Z"].to_numpy(dtype=float),
        c=[CLASS_COLORS.get(v, "#666666") for v in sub["final_class"].astype(str).tolist()],
        s=60,
        alpha=0.86,
        edgecolors="black",
        linewidths=0.4,
    )
    ax1.set_xlabel("Moran's I")
    ax1.set_ylabel("median_Z")
    ax1.set_title("Moran I vs median_Z")
    ax1.grid(alpha=0.25, linewidth=0.6)

    if sub.shape[0] > 0:
        mz_cut = float(np.nanquantile(sub["median_Z"].to_numpy(dtype=float), 0.90))
        mor_cut = float(np.nanmedian(sub["moran_I"].to_numpy(dtype=float)))
        cand = sub.loc[(sub["median_Z"] >= mz_cut) & (sub["moran_I"] <= mor_cut)].sort_values(by="median_Z", ascending=False).head(10)
        for _, row in cand.iterrows():
            ax1.text(float(row["moran_I"]) + 0.003, float(row["median_Z"]) + 0.03, str(row["gene"]), fontsize=8)

    fig1.tight_layout()
    fig1.savefig(out_dir / "MoranI_vs_medianZ.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8.2, 6.0))
    ax2.scatter(
        sub["moran_I"].to_numpy(dtype=float),
        sub["median_coverage"].to_numpy(dtype=float),
        c=[CLASS_COLORS.get(v, "#666666") for v in sub["final_class"].astype(str).tolist()],
        s=60,
        alpha=0.86,
        edgecolors="black",
        linewidths=0.4,
    )
    ax2.set_xlabel("Moran's I")
    ax2.set_ylabel("median_coverage")
    ax2.set_title("Moran I vs median_coverage")
    ax2.grid(alpha=0.25, linewidth=0.6)
    fig2.tight_layout()
    fig2.savefig(out_dir / "MoranI_vs_coverage.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)

    if de_df.empty or "de_rank" not in summary_df.columns:
        _save_placeholder(
            out_dir / "DE_rank_vs_triage_score.png",
            "DE baseline",
            "DE baseline unavailable or disabled.",
        )
    else:
        fig3, ax3 = plt.subplots(figsize=(8.2, 6.0))
        plot_df = summary_df.loc[np.isfinite(summary_df["de_rank"].to_numpy(dtype=float))].copy()
        if plot_df.empty:
            _save_placeholder(
                out_dir / "DE_rank_vs_triage_score.png",
                "DE baseline",
                "No genes with finite DE rank.",
            )
        else:
            ax3.scatter(
                plot_df["de_rank"].to_numpy(dtype=float),
                plot_df["triage_score"].to_numpy(dtype=float),
                c=[CLASS_COLORS.get(v, "#666666") for v in plot_df["final_class"].astype(str).tolist()],
                s=58,
                alpha=0.86,
                edgecolors="black",
                linewidths=0.4,
            )
            ax3.set_xscale("log")
            ax3.set_xlabel("Best DE rank (log scale)")
            ax3.set_ylabel("triage_score")
            ax3.set_title("DE rank vs triage score")
            ax3.grid(alpha=0.25, linewidth=0.6)
            hit = plot_df.loc[
                (plot_df["final_class"].isin(["Localized–unimodal", "Localized–multimodal"]))
                & (plot_df["de_rank"] > 100)
            ].sort_values(by="triage_score", ascending=False).head(10)
            for _, row in hit.iterrows():
                ax3.text(float(row["de_rank"]) * 1.02, float(row["triage_score"]) + 0.002, str(row["gene"]), fontsize=7)
            fig3.tight_layout()
            fig3.savefig(out_dir / "DE_rank_vs_triage_score.png", dpi=DEFAULT_PLOT_STYLE.dpi)
            plt.close(fig3)


def _plot_qc_controls(
    *,
    summary_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        _save_placeholder(out_dir / "qc_risk_vs_medianZ.png", "QC controls", "No scored genes.")
        return

    fig1, ax1 = plt.subplots(figsize=(7.8, 5.8))
    ax1.scatter(
        summary_df["qc_risk_max"].to_numpy(dtype=float),
        summary_df["median_Z"].to_numpy(dtype=float),
        c=[CLASS_COLORS.get(v, "#666666") for v in summary_df["final_class"].astype(str).tolist()],
        s=60,
        alpha=0.86,
        edgecolors="black",
        linewidths=0.4,
    )
    ax1.axvline(QC_RISK_THRESH, color="#8B0000", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("qc_risk_max")
    ax1.set_ylabel("median_Z")
    ax1.set_title("QC risk vs median_Z")
    ax1.grid(alpha=0.25, linewidth=0.6)
    fig1.tight_layout()
    fig1.savefig(out_dir / "qc_risk_vs_medianZ.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7.8, 5.8))
    ax2.scatter(
        summary_df["rho_mt_median"].to_numpy(dtype=float),
        summary_df["rho_counts_median"].to_numpy(dtype=float),
        c=[CLASS_COLORS.get(v, "#666666") for v in summary_df["final_class"].astype(str).tolist()],
        s=60,
        alpha=0.86,
        edgecolors="black",
        linewidths=0.4,
    )
    ax2.set_xlabel("rho_mt_median")
    ax2.set_ylabel("rho_counts_median")
    ax2.set_title("rho_mt vs rho_counts")
    ax2.grid(alpha=0.25, linewidth=0.6)
    fig2.tight_layout()
    fig2.savefig(out_dir / "rho_mt_vs_rho_counts.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(7.8, 5.8))
    ax3.scatter(
        summary_df["sim_qc_max"].to_numpy(dtype=float),
        summary_df["median_Z"].to_numpy(dtype=float),
        c=[CLASS_COLORS.get(v, "#666666") for v in summary_df["final_class"].astype(str).tolist()],
        s=60,
        alpha=0.86,
        edgecolors="black",
        linewidths=0.4,
    )
    ax3.axvline(SIM_QC_THRESH, color="#8B0000", linestyle="--", linewidth=1.2)
    ax3.set_xlabel("sim_qc_max")
    ax3.set_ylabel("median_Z")
    ax3.set_title("QC-profile similarity vs median_Z")
    ax3.grid(alpha=0.25, linewidth=0.6)
    fig3.tight_layout()
    fig3.savefig(out_dir / "sim_qc_vs_medianZ.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig3)


def _select_exemplar_genes(summary_df: pd.DataFrame) -> list[tuple[str, str]]:
    picks: list[tuple[str, str]] = []
    seen: set[str] = set()

    spec = [
        ("Localized–unimodal", 6),
        ("Localized–multimodal", 6),
        ("QC-driven", 4),
        ("Ubiquitous (non-localized)", 4),
    ]
    for cls, k in spec:
        sub = summary_df.loc[summary_df["final_class"] == cls].sort_values(by="triage_score", ascending=False)
        for _, row in sub.head(k).iterrows():
            gene = str(row["gene"])
            if gene in seen:
                continue
            seen.add(gene)
            picks.append((cls, gene))
    return picks


def _plot_exemplar_panels(
    *,
    exemplars: list[tuple[str, str]],
    summary_df: pd.DataFrame,
    gene_to_idx: dict[str, int],
    expr_matrix: Any,
    umap_xy: np.ndarray,
    center_xy: np.ndarray,
    theta: np.ndarray,
    bin_id: np.ndarray,
    bin_counts_total: np.ndarray,
    donor_ids: np.ndarray | None,
    n_bins: int,
    n_perm: int,
    seed: int,
    q: float,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(exemplars) == 0:
        _save_placeholder(out_dir / "no_exemplars.png", "Exemplar panels", "No exemplar genes selected.")
        return

    for i, (cls, gene) in enumerate(exemplars):
        if gene not in gene_to_idx:
            continue
        gidx = int(gene_to_idx[gene])
        expr = np.asarray(get_feature_vector(expr_matrix, gidx), dtype=float)
        fA = expr > 0.0
        fB = _build_foreground_donor_quantile(expr, donor_ids, q=q)

        permA = perm_null_T_and_profile(
            expr=fA.astype(float),
            theta=theta,
            donor_ids=donor_ids,
            n_bins=int(n_bins),
            n_perm=int(n_perm),
            seed=int(seed + 80_000 + i * 19),
            donor_stratified=bool(donor_ids is not None),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        permB = perm_null_T_and_profile(
            expr=fB.astype(float),
            theta=theta,
            donor_ids=donor_ids,
            n_bins=int(n_bins),
            n_perm=int(n_perm),
            seed=int(seed + 90_000 + i * 19),
            donor_stratified=bool(donor_ids is not None),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )

        eA = np.asarray(permA["E_phi_obs"], dtype=float)
        nEA = np.asarray(permA["null_E_phi"], dtype=float)
        tA = np.asarray(permA["null_T"], dtype=float)
        eB = np.asarray(permB["E_phi_obs"], dtype=float)
        nEB = np.asarray(permB["null_E_phi"], dtype=float)
        tB = np.asarray(permB["null_T"], dtype=float)

        row = summary_df.loc[summary_df["gene"] == gene].iloc[0]

        fig = plt.figure(figsize=(17.0, 4.9))
        ax1 = fig.add_subplot(1, 4, 1)
        ax2 = fig.add_subplot(1, 4, 2, projection="polar")
        ax3 = fig.add_subplot(1, 4, 3, projection="polar")
        ax4 = fig.add_subplot(1, 4, 4)

        log_expr = np.log1p(np.maximum(expr, 0.0))
        order = np.argsort(log_expr, kind="mergesort")
        ax1.scatter(umap_xy[:, 0], umap_xy[:, 1], c="#D2D2D2", s=3.5, alpha=0.33, linewidths=0, rasterized=True)
        sc = ax1.scatter(
            umap_xy[order, 0],
            umap_xy[order, 1],
            c=log_expr[order],
            cmap="Reds",
            s=6.5,
            alpha=0.9,
            linewidths=0,
            rasterized=True,
        )
        ax1.scatter([float(center_xy[0])], [float(center_xy[1])], marker="X", s=70, c="black", edgecolors="white", linewidths=0.8)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f"{gene}: log1p expr")
        fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03)

        th = theta_bin_centers(int(n_bins))
        thc = np.concatenate([th, th[:1]])

        def _polar(ax, e_obs, null_e, title, ann):
            obs_c = np.concatenate([e_obs, e_obs[:1]])
            q_hi = np.quantile(null_e, 0.95, axis=0)
            q_lo = np.quantile(null_e, 0.05, axis=0)
            ax.plot(thc, obs_c, color="#8B0000", linewidth=2.0)
            ax.plot(thc, np.concatenate([q_hi, q_hi[:1]]), color="#444444", linestyle="--", linewidth=1.1)
            ax.plot(thc, np.concatenate([q_lo, q_lo[:1]]), color="#444444", linestyle="--", linewidth=1.0)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_thetagrids(np.arange(0, 360, 90))
            ax.set_title(title)
            ax.text(0.02, 0.02, ann, transform=ax.transAxes, fontsize=7, ha="left", va="bottom", bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.8})

        annA = f"q={float(row['q_T_A']):.2e}\nZ={float(row['Z_T_A']):.2f}\nC={float(row['coverage_C_A']):.3f}\nK={int(row['peaks_K_A'])}"
        annB = f"q={float(row['q_T_B']):.2e}\nZ={float(row['Z_T_B']):.2f}\nC={float(row['coverage_C_B']):.3f}\nK={int(row['peaks_K_B'])}"
        _polar(ax2, eA, nEA, "Mode A polar", annA)
        _polar(ax3, eB, nEB, "Mode B polar", annB)

        bins = int(min(45, max(12, np.ceil(np.sqrt(max(tA.size, tB.size))))))
        ax4.hist(tA, bins=bins, color="#4C78A8", alpha=0.55, label="null_T A", edgecolor="white")
        ax4.hist(tB, bins=bins, color="#F58518", alpha=0.55, label="null_T B", edgecolor="white")
        ax4.axvline(float(permA["T_obs"]), color="#1F77B4", linestyle="--", linewidth=1.8)
        ax4.axvline(float(permB["T_obs"]), color="#B23A00", linestyle="--", linewidth=1.8)
        ax4.set_xlabel("null_T")
        ax4.set_ylabel("count")
        ax4.set_title("Null T histograms")
        ax4.legend(loc="upper right", fontsize=8, frameon=True)

        fig.suptitle(f"Exemplar [{cls}] {gene}", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"exemplar_{_sanitize_name(cls)}_{_sanitize_name(gene)}.png", dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    args = parse_args()
    apply_plot_style()
    t_start = time.time()

    h5ad_path = Path(args.h5ad)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Input h5ad not found: {h5ad_path}")

    outdir = Path(args.out)
    tables_dir = outdir / "tables"
    plots_dir = outdir / "plots"
    p_overview = plots_dir / "00_overview"
    p_dist = plots_dir / "01_distributions"
    p_score = plots_dir / "02_score_space"
    p_base = plots_dir / "03_baseline_comparisons"
    p_ex = plots_dir / "04_exemplar_panels"
    p_qc = plots_dir / "05_qc_controls"
    for d in [tables_dir, p_overview, p_dist, p_score, p_base, p_ex, p_qc]:
        d.mkdir(parents=True, exist_ok=True)

    warnings_log: list[str] = []

    adata = ad.read_h5ad(h5ad_path)
    _ensure_umap(adata, seed=int(args.seed), recompute_if_missing=False)
    embedding_key, umap_xy = _resolve_embedding(adata, args.embedding_key)
    center_xy = compute_vantage_point(umap_xy, method="median")
    theta = compute_theta(umap_xy, center_xy)
    _, bin_id = bin_theta(theta, int(args.n_bins))
    bin_counts_total = np.bincount(bin_id, minlength=int(args.n_bins)).astype(float)

    expr_matrix, adata_like, expr_source = _choose_expression_source(
        adata, layer_arg=args.layer, use_raw_arg=bool(args.use_raw)
    )

    donor_ids, donor_key_used = _resolve_donor_ids_optional(adata, args.donor_key)
    if donor_ids is None:
        warnings_log.append("Donor key missing/<2 donors: using global permutation nulls.")
    label_values, label_key_used = _resolve_label_key_optional(adata, args.label_key)

    total_counts = _total_counts_vector(adata, expr_matrix)
    pct_mt_raw, pct_mt_source = _pct_mt_vector(adata, expr_matrix, adata_like)
    pct_mt = None if pct_mt_source == "proxy:zeros" else np.asarray(pct_mt_raw, dtype=float)
    pct_ribo, pct_ribo_source = _compute_pct_counts_ribo(adata, expr_matrix, adata_like, total_counts)
    if pct_mt is None:
        warnings_log.append("pct_counts_mt unavailable.")
    if pct_ribo is None:
        warnings_log.append("pct_counts_ribo unavailable.")

    qc_covars = {
        "total_counts": np.asarray(total_counts, dtype=float),
        "pct_counts_mt": pct_mt,
        "pct_counts_ribo": pct_ribo,
    }

    # Gene set construction.
    gene_list_path = Path(args.gene_list) if args.gene_list is not None else None
    resolved_genes, eval_logic = _build_eval_gene_set(
        mode=str(args.mode),
        max_genes=int(args.max_genes),
        gene_list_path=gene_list_path,
        expr_matrix=expr_matrix,
        adata_like=adata_like,
    )

    genes_rows = []
    for g in resolved_genes:
        genes_rows.append(
            {
                "requested_gene": g.requested_gene,
                "found": bool(g.found),
                "resolved_gene": g.resolved_gene,
                "gene_idx": g.gene_idx if g.gene_idx is not None else "",
                "status": g.status,
                "resolution_source": g.source,
                "symbol_column": g.symbol_column,
            }
        )
    genes_scored_df = pd.DataFrame(genes_rows)

    found_genes = [g for g in resolved_genes if g.found and g.gene_idx is not None]
    gene_to_idx = {g.requested_gene: int(g.gene_idx) for g in found_genes}

    # QC pseudo-feature profiles for similarity gate.
    qc_profiles: dict[str, np.ndarray] = {}
    for i, (name, vals) in enumerate(qc_covars.items()):
        if vals is None:
            continue
        weights = _zscore_log1p(np.asarray(vals, dtype=float))
        perm = _perm_null_continuous_profile(
            weights,
            donor_ids=donor_ids,
            n_bins=int(args.n_bins),
            n_perm=int(args.n_perm),
            seed=int(args.seed + 10_000 + i * 31),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        qc_profiles[name] = np.asarray(perm["E_phi_obs"], dtype=float)

    # Per-gene, per-mode scoring.
    mode_rows: list[dict[str, Any]] = []
    n_target = len(found_genes)
    for gi, g in enumerate(found_genes, start=1):
        if gi == 1 or gi % 200 == 0 or gi == n_target:
            elapsed = time.time() - t_start
            print(f"[Scoring] gene {gi}/{n_target} elapsed={elapsed:.1f}s: {g.requested_gene}", flush=True)

        expr = np.asarray(get_feature_vector(expr_matrix, int(g.gene_idx)), dtype=float)
        f_A = expr > 0.0
        f_B = _build_foreground_donor_quantile(expr, donor_ids, q=float(args.q))

        for mi, (mode_name, f_mask) in enumerate([(MODE_A, f_A), (MODE_B, f_B)]):
            f = np.asarray(f_mask, dtype=bool)
            prev = float(np.mean(f))
            n_fg = int(f.sum())

            perm = perm_null_T_and_profile(
                expr=f.astype(float),
                theta=theta,
                donor_ids=donor_ids,
                n_bins=int(args.n_bins),
                n_perm=int(args.n_perm),
                seed=int(args.seed + 20_000 + gi * 13 + mi),
                donor_stratified=bool(donor_ids is not None),
                bin_id=bin_id,
                bin_counts_total=bin_counts_total,
            )
            e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
            null_e = np.asarray(perm["null_E_phi"], dtype=float)
            null_t = np.asarray(perm["null_T"], dtype=float)
            t_obs = float(perm["T_obs"])

            z_t = float(robust_z(t_obs, null_t))
            coverage = float(coverage_from_null(e_obs, null_e, q=0.95))
            peaks = int(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95))
            p_idx = int(np.argmax(np.abs(e_obs))) if e_obs.size > 0 else 0
            centers = theta_bin_centers(int(args.n_bins))
            phi = float(centers[p_idx]) if centers.size > 0 else 0.0

            rho_counts = _safe_spearman(f.astype(float), qc_covars["total_counts"])
            rho_mt = _safe_spearman(f.astype(float), qc_covars["pct_counts_mt"])
            rho_ribo = _safe_spearman(f.astype(float), qc_covars["pct_counts_ribo"])
            finite_rho = [abs(v) for v in [rho_counts, rho_mt, rho_ribo] if np.isfinite(v)]
            qc_risk = float(max(finite_rho)) if finite_rho else 0.0

            sim_vals = []
            for qname, qprof in qc_profiles.items():
                sim = _cosine_similarity(e_obs, qprof)
                if np.isfinite(sim):
                    sim_vals.append((qname, sim))
            if sim_vals:
                best_qc_name, sim_qc = max(sim_vals, key=lambda x: x[1])
            else:
                best_qc_name, sim_qc = "", float("nan")

            mode_rows.append(
                {
                    "gene": g.requested_gene,
                    "resolved_gene": g.resolved_gene,
                    "gene_idx": int(g.gene_idx),
                    "mode": mode_name,
                    "prev": prev,
                    "n_fg": int(n_fg),
                    "n_cells": int(f.size),
                    "T_obs": t_obs,
                    "p_T": float(perm["p_T"]),
                    "q_T": np.nan,
                    "Z_T": z_t,
                    "coverage_C": coverage,
                    "peaks_K": int(peaks),
                    "phi_hat_rad": phi,
                    "phi_hat_deg": float(np.degrees(phi) % 360.0),
                    "qc_risk": qc_risk,
                    "rho_counts": rho_counts,
                    "rho_mt": rho_mt,
                    "rho_ribo": rho_ribo,
                    "sim_qc": sim_qc,
                    "best_qc_feature": best_qc_name,
                    "used_donor_stratified": bool(perm["used_donor_stratified"]),
                    "donor_key_used": donor_key_used if donor_key_used is not None else "",
                    "perm_warning": str(perm.get("warning", "")),
                }
            )

    mode_df = pd.DataFrame(mode_rows)

    if not mode_df.empty:
        for mode_name in MODE_ORDER:
            mask = mode_df["mode"] == mode_name
            if int(mask.sum()) > 0:
                mode_df.loc[mask, "q_T"] = bh_fdr(mode_df.loc[mask, "p_T"].to_numpy(dtype=float))

    # Build per-gene summary by combining two modes.
    summary_rows: list[dict[str, Any]] = []
    for gene, sub in mode_df.groupby("gene", observed=False):
        rowA = sub.loc[sub["mode"] == MODE_A]
        rowB = sub.loc[sub["mode"] == MODE_B]
        if rowA.empty or rowB.empty:
            continue
        a = rowA.iloc[0]
        b = rowB.iloc[0]

        prevA = float(a["prev"])
        prevB = float(b["prev"])
        nfgA = int(a["n_fg"])
        nfgB = int(b["n_fg"])

        underpowered = bool(((prevA < P_MIN) or (nfgA < MIN_FG)) and ((prevB < P_MIN) or (nfgB < MIN_FG)))

        sigA = bool(float(a["q_T"]) <= Q_SIG)
        sigB = bool(float(b["q_T"]) <= Q_SIG)

        phi_diff = _circular_diff_deg(float(a["phi_hat_deg"]), float(b["phi_hat_deg"]))
        robust_localized = bool(
            sigA
            and sigB
            and (
                (phi_diff <= 45.0)
                or (
                    int(a["peaks_K"]) == 1
                    and int(b["peaks_K"]) == 1
                    and phi_diff <= 60.0
                )
            )
        )

        qc_risk_max = float(max(float(a["qc_risk"]), float(b["qc_risk"])))
        sim_qc_max = float(np.nanmax(np.asarray([a["sim_qc"], b["sim_qc"]], dtype=float)))
        qc_driven = bool((sigA or sigB) and ((qc_risk_max >= QC_RISK_THRESH) or (np.isfinite(sim_qc_max) and sim_qc_max >= SIM_QC_THRESH)))

        peaks_med = int(round(float(np.median([float(a["peaks_K"]), float(b["peaks_K"])]))))
        median_z = float(np.median([float(a["Z_T"]), float(b["Z_T"])]))
        median_cov = float(np.median([float(a["coverage_C"]), float(b["coverage_C"])]))
        triage_score = float(median_z * median_cov * (1.0 - np.clip(qc_risk_max, 0.0, 1.0)))

        if underpowered:
            final_class = "Underpowered"
        elif qc_driven:
            final_class = "QC-driven"
        elif (not sigA) and (not sigB) and (prevA >= HIGH_PREV):
            final_class = "Ubiquitous (non-localized)"
        elif robust_localized and peaks_med == 1:
            final_class = "Localized–unimodal"
        elif robust_localized and peaks_med >= 2:
            final_class = "Localized–multimodal"
        else:
            final_class = "Uncertain"

        summary_rows.append(
            {
                "gene": gene,
                "resolved_gene": str(a["resolved_gene"]),
                "gene_idx": int(a["gene_idx"]),
                "prev_A": prevA,
                "prev_B": prevB,
                "n_fg_A": nfgA,
                "n_fg_B": nfgB,
                "T_obs_A": float(a["T_obs"]),
                "T_obs_B": float(b["T_obs"]),
                "p_T_A": float(a["p_T"]),
                "p_T_B": float(b["p_T"]),
                "q_T_A": float(a["q_T"]),
                "q_T_B": float(b["q_T"]),
                "Z_T_A": float(a["Z_T"]),
                "Z_T_B": float(b["Z_T"]),
                "coverage_C_A": float(a["coverage_C"]),
                "coverage_C_B": float(b["coverage_C"]),
                "peaks_K_A": int(a["peaks_K"]),
                "peaks_K_B": int(b["peaks_K"]),
                "phi_hat_deg_A": float(a["phi_hat_deg"]),
                "phi_hat_deg_B": float(b["phi_hat_deg"]),
                "phi_diff_deg": float(phi_diff),
                "rho_counts_A": float(a["rho_counts"]),
                "rho_counts_B": float(b["rho_counts"]),
                "rho_mt_A": float(a["rho_mt"]),
                "rho_mt_B": float(b["rho_mt"]),
                "rho_ribo_A": float(a["rho_ribo"]),
                "rho_ribo_B": float(b["rho_ribo"]),
                "rho_counts_median": float(np.nanmedian(np.asarray([a["rho_counts"], b["rho_counts"]], dtype=float))),
                "rho_mt_median": float(np.nanmedian(np.asarray([a["rho_mt"], b["rho_mt"]], dtype=float))),
                "rho_ribo_median": float(np.nanmedian(np.asarray([a["rho_ribo"], b["rho_ribo"]], dtype=float))),
                "qc_risk_max": qc_risk_max,
                "sim_qc_A": float(a["sim_qc"]),
                "sim_qc_B": float(b["sim_qc"]),
                "sim_qc_max": sim_qc_max,
                "sig_A": sigA,
                "sig_B": sigB,
                "robust_localized": robust_localized,
                "underpowered": underpowered,
                "qc_driven": qc_driven,
                "peaks_K_median": peaks_med,
                "median_Z": median_z,
                "median_coverage": median_cov,
                "triage_score": triage_score,
                "final_class": final_class,
                "used_donor_stratified_A": bool(a["used_donor_stratified"]),
                "used_donor_stratified_B": bool(b["used_donor_stratified"]),
                "donor_key_used": donor_key_used if donor_key_used is not None else "",
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Moran baseline.
    w = _build_knn_weights(umap_xy, k=int(args.k))
    moran_df = _compute_moran_baseline(
        resolved_genes=found_genes,
        expr_matrix=expr_matrix,
        w=w,
    )
    summary_df = summary_df.merge(moran_df[["gene", "moran_I"]], on="gene", how="left")

    # Optional DE baseline.
    de_df, de_method, de_warn = _run_de_baseline(
        adata=adata,
        label_key=label_key_used,
        run_flag=bool(args.run_de_baseline),
        seed=int(args.seed),
    )
    warnings_log.extend(de_warn)
    if not de_df.empty:
        summary_df = summary_df.merge(de_df[["gene", "de_rank", "de_group"]], on="gene", how="left")
    else:
        summary_df["de_rank"] = np.nan
        summary_df["de_group"] = ""

    # Final tables.
    mode_df = mode_df.sort_values(by=["mode", "q_T", "gene"], ascending=[True, True, True], kind="mergesort")
    summary_df = summary_df.sort_values(by=["triage_score", "median_Z"], ascending=[False, False], kind="mergesort")
    top_candidates_df = summary_df.loc[
        summary_df["final_class"].isin(["Localized–unimodal", "Localized–multimodal"])
    ].head(int(args.top_candidates)).copy()
    qc_flag_df = summary_df.loc[summary_df["final_class"] == "QC-driven"].copy()

    genes_scored_df.to_csv(tables_dir / "genes_scored.csv", index=False)
    mode_df.to_csv(tables_dir / "per_gene_mode_scores.csv", index=False)
    summary_df.to_csv(tables_dir / "per_gene_summary.csv", index=False)
    top_candidates_df.to_csv(tables_dir / "top_candidates_localized.csv", index=False)
    qc_flag_df.to_csv(tables_dir / "qc_flagged_genes.csv", index=False)
    moran_df.to_csv(tables_dir / "moran_baseline.csv", index=False)
    de_df.to_csv(tables_dir / "de_baseline.csv", index=False)

    # Plots.
    _plot_overview(
        adata=adata,
        umap_xy=umap_xy,
        donor_key=donor_key_used,
        label_key=label_key_used,
        total_counts=np.asarray(total_counts, dtype=float),
        pct_mt=pct_mt,
        pct_ribo=pct_ribo,
        center_xy=np.asarray(center_xy, dtype=float),
        out_dir=p_overview,
    )
    _plot_distributions(mode_df=mode_df, summary_df=summary_df, out_dir=p_dist)
    _plot_score_space(summary_df=summary_df, out_dir=p_score)
    _plot_baseline_comparisons(summary_df=summary_df, de_df=de_df, out_dir=p_base)
    _plot_qc_controls(summary_df=summary_df, out_dir=p_qc)

    exemplars = _select_exemplar_genes(summary_df)
    _plot_exemplar_panels(
        exemplars=exemplars,
        summary_df=summary_df,
        gene_to_idx=gene_to_idx,
        expr_matrix=expr_matrix,
        umap_xy=umap_xy,
        center_xy=np.asarray(center_xy, dtype=float),
        theta=theta,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
        donor_ids=donor_ids,
        n_bins=int(args.n_bins),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        q=float(args.q),
        out_dir=p_ex,
    )

    elapsed = time.time() - t_start

    readme_lines = [
        "Experiment #8: Genome-wide triage for representation-conditional localized programs",
        "",
        "Interpretation: BioRSP captures representation-conditional localization modes; not tissue direction and not direct cell-type discovery.",
        "",
        "Metadata:",
        f"- embedding_key_used: {embedding_key}",
        f"- donor_key_used: {donor_key_used if donor_key_used is not None else 'None'}",
        f"- label_key_used: {label_key_used if label_key_used is not None else 'None'}",
        f"- expression_source_used: {expr_source}",
        f"- n_cells: {int(adata.n_obs)}",
        f"- mode: {args.mode}",
        f"- max_genes: {int(args.max_genes)}",
        f"- q_modeB: {float(args.q):.3f}",
        f"- n_bins: {int(args.n_bins)}",
        f"- n_perm: {int(args.n_perm)}",
        f"- k_neighbors: {int(args.k)}",
        f"- evaluation_logic: {eval_logic}",
        f"- run_de_baseline: {bool(args.run_de_baseline)}",
        f"- de_baseline_method: {de_method}",
        f"- runtime_seconds: {elapsed:.1f}",
        "",
        "Triage gates:",
        "- Underpowered if both modes have low prevalence/power.",
        "- QC-driven if significant in either mode and qc_risk>=0.35 or sim_qc>=0.7.",
        "- Robust localized requires q<=0.05 in both modes and phi consistency across modes.",
        "",
        "Summary:",
        f"- genes_requested: {int(genes_scored_df.shape[0])}",
        f"- genes_resolved: {int(sum(genes_scored_df['found'].astype(bool)))}",
        f"- mode_rows: {int(mode_df.shape[0])}",
        f"- summary_rows: {int(summary_df.shape[0])}",
        f"- localized_unimodal: {int((summary_df['final_class'] == 'Localized–unimodal').sum())}",
        f"- localized_multimodal: {int((summary_df['final_class'] == 'Localized–multimodal').sum())}",
        f"- qc_driven: {int((summary_df['final_class'] == 'QC-driven').sum())}",
        f"- ubiquitous: {int((summary_df['final_class'] == 'Ubiquitous (non-localized)').sum())}",
        f"- underpowered: {int((summary_df['final_class'] == 'Underpowered').sum())}",
        "",
        "Warnings:",
    ]
    if warnings_log:
        for msg in warnings_log:
            readme_lines.append(f"- {msg}")
    else:
        readme_lines.append("- none")
    (outdir / "README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"embedding_key_used={embedding_key}")
    print(f"donor_key_used={donor_key_used if donor_key_used is not None else 'None'}")
    print(f"label_key_used={label_key_used if label_key_used is not None else 'None'}")
    print(f"expression_source_used={expr_source}")
    print(
        f"n_cells={int(adata.n_obs)} n_bins={int(args.n_bins)} n_perm={int(args.n_perm)} k={int(args.k)} mode={args.mode} max_genes={int(args.max_genes)}"
    )
    print(f"evaluation_logic={eval_logic}")
    print(f"genes_requested={int(genes_scored_df.shape[0])} genes_resolved={int(sum(genes_scored_df['found'].astype(bool)))}")
    print(f"per_gene_mode_scores_csv={(tables_dir / 'per_gene_mode_scores.csv').as_posix()}")
    print(f"per_gene_summary_csv={(tables_dir / 'per_gene_summary.csv').as_posix()}")
    print(f"top_candidates_csv={(tables_dir / 'top_candidates_localized.csv').as_posix()}")
    print(f"moran_baseline_csv={(tables_dir / 'moran_baseline.csv').as_posix()}")
    print(f"results_root={outdir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

