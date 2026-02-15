#!/usr/bin/env python3
"""Experiment #3: endothelial arterio-venous axis benchmark (geometry validation)."""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

# Use headless backend for deterministic local/CI plotting.
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

# Allow running script directly via `python experiments/...` from repository root.
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
from biorsp.stats.moran import extract_weights, morans_i
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

ARTERIAL_MARKERS = ["EFNB2", "GJA5", "GJA4", "DLL4", "SOX17", "NRP1", "CXCR4", "UNC5B"]
VENOUS_MARKERS = ["EPHB4", "NR2F2", "NRP2", "ACKR1", "DAB2"]
PAN_EC_MARKERS = ["PECAM1", "VWF", "KDR", "CLDN5"]
QC_PSEUDOGENES = ["total_counts", "pct_counts_mt", "pct_counts_ribo"]

CLASS_ORDER = [
    "Underpowered",
    "Ubiquitous (non-localized)",
    "Localized–unimodal",
    "Localized–multimodal",
    "QC-driven",
    "Uncertain",
]

CLASS_COLORS = {
    "Underpowered": "#8C8C8C",
    "Ubiquitous (non-localized)": "#2CA02C",
    "Localized–unimodal": "#1F77B4",
    "Localized–multimodal": "#FF7F0E",
    "QC-driven": "#D62728",
    "Uncertain": "#9467BD",
}

AV_GROUP_COLORS = {
    "arterial": "#B22222",
    "venous": "#1E5AA8",
    "pan": "#2C8C4A",
    "qc": "#5F5F5F",
}

AV_GROUP_MARKERS = {
    "arterial": "^",
    "venous": "s",
    "pan": "o",
    "qc": "X",
}

Q_SIG = 0.05
P_MIN = 0.005
MIN_FG = 50
HIGH_PREV = 0.60
QC_THRESH = 0.35
SIM_QC_THRESH = 0.70
Z_STRONG = 4.0
COVERAGE_STRONG = 0.15


@dataclass(frozen=True)
class ResolvedFeature:
    feature: str
    feature_type: str  # gene | qc
    av_group: str
    found: bool
    gene_idx: int | None
    resolved_gene: str
    status: str
    provenance: str
    resolution_source: str
    symbol_column: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Experiment #3 endothelial arterio-venous axis benchmark."
    )
    p.add_argument(
        "--h5ad", default="data/processed/HT_pca_umap.h5ad", help="Input .h5ad path."
    )
    p.add_argument(
        "--out",
        default="experiments/heart_smoketest/results/experiment3_endothelial_av_axis",
        help="Output directory.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument(
        "--n_perm", type=int, default=300, help="Permutation count (default smoke=300)."
    )
    p.add_argument("--n_bins", type=int, default=64, help="Angular bin count.")
    p.add_argument(
        "--min_cells",
        type=int,
        default=2000,
        help="Subset underpowered warning threshold.",
    )
    p.add_argument(
        "--embedding_key", default=None, help="Optional embedding key override."
    )
    p.add_argument("--donor_key", default=None, help="Optional donor key override.")
    p.add_argument("--label_key", default=None, help="Optional label key override.")
    p.add_argument("--layer", default=None, help="Optional expression layer override.")
    p.add_argument(
        "--use_raw", action="store_true", help="Use adata.raw instead of X/layers."
    )
    return p.parse_args()


def _save_placeholder(out_png: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
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
                f"Requested embedding '{requested_key}' missing from adata.obsm."
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


def _resolve_donor_ids(
    adata: ad.AnnData,
    requested_key: str | None,
) -> tuple[np.ndarray | None, str | None]:
    key = _resolve_key(adata, requested_key, DONOR_CANDIDATES)
    if key is None:
        return None, None
    donor_ids = adata.obs[key].astype("string").fillna("NA").astype(str).to_numpy()
    if np.unique(donor_ids).size < 2:
        return None, key
    return donor_ids, key


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
    pct_ribo = (
        np.divide(ribo_counts, np.maximum(np.asarray(total_counts, dtype=float), 1e-12))
        * 100.0
    )
    return pct_ribo, f"computed:{symbol_col}"


def _is_ec_label(label: str) -> bool:
    text = str(label).strip()
    low = text.lower()
    token_match = any(
        tok in low
        for tok in [
            "endo",
            "endothelial",
            "capillary",
            "artery",
            "arterial",
            "vein",
            "venous",
        ]
    )
    exact_match = text in {"EC", "Endo", "CapEC", "ArtEC", "VenEC"}
    return bool(token_match or exact_match)


def _build_ec_mask(labels: pd.Series) -> tuple[np.ndarray, list[str], pd.Series]:
    labels_str = labels.astype("string").fillna("NA").astype(str)
    keep = labels_str.map(_is_ec_label).to_numpy(dtype=bool)
    included = sorted(labels_str[keep].unique().tolist())
    counts = labels_str[keep].value_counts().sort_values(ascending=False)
    return keep, included, counts


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


def _resolve_av_panel(
    adata_like: Any,
    qc_available: dict[str, bool],
) -> tuple[list[ResolvedFeature], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    resolved: list[ResolvedFeature] = []
    used_idx: set[int] = set()

    panel_genes: list[tuple[str, str, str]] = []
    panel_genes.extend(
        [
            (
                g,
                "arterial",
                "Pre-registered arterial-enriched marker for Experiment #3.",
            )
            for g in ARTERIAL_MARKERS
        ]
    )
    panel_genes.extend(
        [
            (g, "venous", "Pre-registered venous-enriched marker for Experiment #3.")
            for g in VENOUS_MARKERS
        ]
    )
    panel_genes.extend(
        [
            (
                g,
                "pan",
                "Pre-registered pan-endothelial sanity marker for Experiment #3.",
            )
            for g in PAN_EC_MARKERS
        ]
    )

    for gene, av_group, provenance in panel_genes:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                idx, label, symbol_col, source = resolve_feature_index(adata_like, gene)
            idx_i = int(idx)
            if idx_i in used_idx:
                item = ResolvedFeature(
                    feature=gene,
                    feature_type="gene",
                    av_group=av_group,
                    found=False,
                    gene_idx=None,
                    resolved_gene="",
                    status="duplicate_index",
                    provenance=provenance,
                    resolution_source=source,
                    symbol_column=symbol_col or "",
                )
            else:
                used_idx.add(idx_i)
                item = ResolvedFeature(
                    feature=gene,
                    feature_type="gene",
                    av_group=av_group,
                    found=True,
                    gene_idx=idx_i,
                    resolved_gene=str(label),
                    status="resolved",
                    provenance=provenance,
                    resolution_source=source,
                    symbol_column=symbol_col or "",
                )
        except KeyError:
            item = ResolvedFeature(
                feature=gene,
                feature_type="gene",
                av_group=av_group,
                found=False,
                gene_idx=None,
                resolved_gene="",
                status="missing",
                provenance=provenance,
                resolution_source="",
                symbol_column="",
            )

        rows.append(
            {
                "feature": item.feature,
                "feature_type": item.feature_type,
                "av_group": item.av_group,
                "status": item.status,
                "found": item.found,
                "resolved_gene": item.resolved_gene,
                "gene_idx": item.gene_idx if item.gene_idx is not None else "",
                "provenance": item.provenance,
                "resolution_source": item.resolution_source,
                "symbol_column": item.symbol_column,
            }
        )
        resolved.append(item)

    for qc_name in QC_PSEUDOGENES:
        present = bool(qc_available.get(qc_name, False))
        status = "available" if present else "unavailable"
        item = ResolvedFeature(
            feature=qc_name,
            feature_type="qc",
            av_group="qc",
            found=present,
            gene_idx=None,
            resolved_gene=qc_name,
            status=status,
            provenance="Technical/QC pseudo-feature control for Experiment #3.",
            resolution_source="obs_or_computed",
            symbol_column="",
        )
        rows.append(
            {
                "feature": item.feature,
                "feature_type": item.feature_type,
                "av_group": item.av_group,
                "status": item.status,
                "found": item.found,
                "resolved_gene": item.resolved_gene,
                "gene_idx": "",
                "provenance": item.provenance,
                "resolution_source": item.resolution_source,
                "symbol_column": item.symbol_column,
            }
        )
        resolved.append(item)

    return resolved, pd.DataFrame(rows)


def _compute_continuous_profile(
    weights: np.ndarray,
    *,
    n_bins: int,
    bin_id: np.ndarray,
    bin_counts_total: np.ndarray,
) -> np.ndarray:
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != bin_id.size:
        raise ValueError("weights/bin_id length mismatch.")
    clean = np.where(np.isfinite(w), w, 0.0)
    if np.nanmin(clean) < 0:
        clean = clean - float(np.nanmin(clean))
    total_w = float(np.sum(clean))
    if total_w <= 1e-12:
        return np.zeros(int(n_bins), dtype=float)
    w_bin = np.bincount(bin_id, weights=clean, minlength=int(n_bins)).astype(float)
    p_w = w_bin / total_w
    p_bg = np.asarray(bin_counts_total, dtype=float) / float(bin_id.size)
    return p_w - p_bg


def _permute_weights_within_donor(
    weights: np.ndarray,
    donor_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(weights, dtype=float).ravel()
    d = np.asarray(donor_ids).astype(str)
    out = np.zeros_like(x)
    for donor in np.unique(d):
        idx = np.flatnonzero(d == donor)
        if idx.size <= 1:
            out[idx] = x[idx]
            continue
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

    used_donor_stratified = False
    warning_msg = ""
    if donor_ids is not None and np.unique(np.asarray(donor_ids).astype(str)).size >= 2:
        used_donor_stratified = True
    else:
        warning_msg = (
            "continuous permutation fallback to global shuffling (donor unavailable)."
        )

    rng = np.random.default_rng(int(seed))
    null_e = np.zeros((int(n_perm), int(n_bins)), dtype=float)
    null_t = np.zeros(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        if used_donor_stratified and donor_ids is not None:
            perm_w = _permute_weights_within_donor(
                x, np.asarray(donor_ids).astype(str), rng
            )
        else:
            perm_w = x[rng.permutation(x.size)]
        e_perm = _compute_continuous_profile(
            perm_w,
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
        "used_donor_stratified": bool(used_donor_stratified),
    }
    if warning_msg:
        out["warning"] = warning_msg
    return out


def _ensure_subset_connectivities(
    subset_adata: ad.AnnData, subset_xy: np.ndarray, *, k: int = 15
) -> None:
    n = int(subset_adata.n_obs)
    if n < 3:
        raise ValueError("Need at least 3 cells to compute Moran's I weights.")

    if "connectivities" in subset_adata.obsp:
        conn = subset_adata.obsp["connectivities"]
        if sp.issparse(conn) and conn.shape == (n, n):
            return

    k_use = int(min(max(2, k), max(2, n - 1)))
    nn = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean")
    nn.fit(np.asarray(subset_xy, dtype=float))
    _, idx = nn.kneighbors(np.asarray(subset_xy, dtype=float))

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n):
        for j in idx[i, 1:]:
            rows.append(i)
            cols.append(int(j))
            vals.append(1.0)

    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)
    mat = mat.maximum(mat.T)
    mat.setdiag(0.0)
    mat.eliminate_zeros()
    subset_adata.obsp["connectivities"] = mat


def _classify_row(row: pd.Series) -> str:
    q_t = float(row["q_T"]) if np.isfinite(float(row["q_T"])) else 1.0
    prev = float(row["prev"]) if np.isfinite(float(row["prev"])) else 0.0
    n_fg = int(row["n_fg"]) if np.isfinite(float(row["n_fg"])) else 0
    peaks = int(row["peaks_K"]) if np.isfinite(float(row["peaks_K"])) else 0
    qc_risk = float(row["qc_risk"]) if np.isfinite(float(row["qc_risk"])) else 0.0
    sim_qc = float(row["sim_qc"]) if np.isfinite(float(row["sim_qc"])) else 0.0

    donor_key_present = str(row["donor_key_used"]).strip() != ""
    donors_n = int(row["donors_n"]) if np.isfinite(float(row["donors_n"])) else 0
    subset_underpowered = bool(row["subset_underpowered"])

    underpowered = bool(
        (prev < P_MIN)
        or (n_fg < MIN_FG)
        or (subset_underpowered)
        or (donor_key_present and donors_n < 2)
    )
    if underpowered:
        return "Underpowered"

    qc_driven = bool(
        (q_t <= Q_SIG) and ((qc_risk >= QC_THRESH) or (sim_qc >= SIM_QC_THRESH))
    )
    if q_t > Q_SIG and prev >= HIGH_PREV:
        return "Ubiquitous (non-localized)"
    if q_t <= Q_SIG and qc_driven:
        return "QC-driven"
    if q_t <= Q_SIG and peaks == 1 and not qc_driven:
        return "Localized–unimodal"
    if q_t <= Q_SIG and peaks >= 2 and not qc_driven:
        return "Localized–multimodal"
    return "Uncertain"


def _plot_overview(
    *,
    adata: ad.AnnData,
    umap_xy: np.ndarray,
    labels: pd.Series,
    ec_mask: np.ndarray,
    ec_counts: pd.Series,
    donor_key_used: str | None,
    ec_center: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) all-cells UMAP with EC highlighted by label and non-EC in gray.
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    non_ec = ~np.asarray(ec_mask, dtype=bool)
    ax.scatter(
        umap_xy[non_ec, 0],
        umap_xy[non_ec, 1],
        c="#CFCFCF",
        s=4.0,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
        label="non-EC",
    )
    ec_labels = labels.astype("string").fillna("NA").astype(str).to_numpy()
    ec_unique = sorted(pd.Index(ec_labels[ec_mask]).unique().tolist())
    cmap = plt.get_cmap("tab20")
    for i, label in enumerate(ec_unique):
        idx = np.flatnonzero(ec_mask & (ec_labels == label))
        if idx.size == 0:
            continue
        ax.scatter(
            umap_xy[idx, 0],
            umap_xy[idx, 1],
            s=7.0,
            alpha=0.9,
            linewidths=0,
            rasterized=True,
            color=cmap(i % 20),
            label=str(label),
        )
    ax.scatter(
        [float(ec_center[0])],
        [float(ec_center[1])],
        marker="X",
        s=90,
        c="black",
        edgecolors="white",
        linewidths=0.9,
        zorder=10,
        label="EC vantage",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("All cells UMAP with endothelial subset highlighted")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(
        out_dir / "umap_all_with_ec_highlight.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 2) EC subset by donor.
    ec_xy = umap_xy[np.asarray(ec_mask, dtype=bool)]
    if donor_key_used is not None and donor_key_used in adata.obs.columns:
        plot_categorical_umap(
            umap_xy=ec_xy,
            labels=adata.obs.loc[np.asarray(ec_mask, dtype=bool), donor_key_used],
            title=f"EC subset UMAP by donor ({donor_key_used})",
            outpath=out_dir / "umap_ec_by_donor.png",
            vantage_point=(float(ec_center[0]), float(ec_center[1])),
            annotate_cluster_medians=False,
        )
    else:
        _save_placeholder(
            out_dir / "umap_ec_by_donor.png",
            "EC subset by donor",
            "Donor key unavailable.",
        )

    # 3) counts per included EC label.
    fig2, ax2 = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(ec_counts.shape[0], dtype=float)
    ax2.bar(
        x,
        ec_counts.to_numpy(dtype=float),
        color="#5DA5DA",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(ec_counts.index.tolist(), rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Cell count")
    ax2.set_title("EC subset counts per included label")
    ax2.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig2.tight_layout()
    fig2.savefig(out_dir / "ec_label_counts.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)


def _plot_gene_panels(
    *,
    ec_xy: np.ndarray,
    ec_center: np.ndarray,
    scores_df: pd.DataFrame,
    artifacts: dict[str, dict[str, np.ndarray]],
    n_bins: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_rows = scores_df.loc[scores_df["feature_type"] == "gene"].copy()
    if gene_rows.empty:
        _save_placeholder(
            out_dir / "no_gene_panels.png", "Gene panels", "No resolved genes to plot."
        )
        return

    # Consistent robust expression scale across all genes.
    log_vals: list[np.ndarray] = []
    for gene in gene_rows["gene"].tolist():
        if gene not in artifacts:
            continue
        log_vals.append(
            np.log1p(np.maximum(np.asarray(artifacts[gene]["expr"], dtype=float), 0.0))
        )
    if log_vals:
        all_log = np.concatenate(log_vals)
        vmin = float(np.nanpercentile(all_log, 1.0))
        vmax = float(np.nanpercentile(all_log, 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            vmin, vmax = 0.0, float(
                np.nanmax(all_log) if np.isfinite(np.nanmax(all_log)) else 1.0
            )
    else:
        vmin, vmax = 0.0, 1.0

    for _, row in gene_rows.iterrows():
        gene = str(row["gene"])
        if gene not in artifacts:
            continue
        art = artifacts[gene]
        expr = np.asarray(art["expr"], dtype=float)
        e_obs = np.asarray(art["E_phi_obs"], dtype=float)
        null_e = np.asarray(art["null_E_phi"], dtype=float)
        null_t = np.asarray(art["null_T"], dtype=float)

        fig = plt.figure(figsize=(15.0, 4.8))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2, projection="polar")
        ax3 = fig.add_subplot(1, 3, 3)

        log_expr = np.log1p(np.maximum(expr, 0.0))
        order = np.argsort(log_expr, kind="mergesort")
        ax1.scatter(
            ec_xy[:, 0],
            ec_xy[:, 1],
            c="#D2D2D2",
            s=4.0,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        sc = ax1.scatter(
            ec_xy[order, 0],
            ec_xy[order, 1],
            c=log_expr[order],
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
            s=7.0,
            alpha=0.90,
            linewidths=0,
            rasterized=True,
        )
        ax1.scatter(
            [float(ec_center[0])],
            [float(ec_center[1])],
            marker="X",
            s=75,
            c="black",
            edgecolors="white",
            linewidths=0.8,
            zorder=10,
        )
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f"{gene}: EC feature map")
        cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03)
        cbar.set_label("log1p(expr)")

        centers = theta_bin_centers(int(n_bins))
        theta_c = np.concatenate([centers, centers[:1]])
        obs_c = np.concatenate([e_obs, e_obs[:1]])
        q_hi = np.quantile(null_e, 0.95, axis=0)
        q_lo = np.quantile(null_e, 0.05, axis=0)
        q_hi_c = np.concatenate([q_hi, q_hi[:1]])
        q_lo_c = np.concatenate([q_lo, q_lo[:1]])
        ax2.plot(theta_c, obs_c, color="#8B0000", linewidth=2.0, label="E_phi obs")
        ax2.plot(
            theta_c,
            q_hi_c,
            color="#444444",
            linestyle="--",
            linewidth=1.3,
            label="null 95%",
        )
        ax2.plot(
            theta_c,
            q_lo_c,
            color="#444444",
            linestyle="--",
            linewidth=1.0,
            label="null 5%",
        )
        ax2.fill_between(theta_c, q_lo_c, q_hi_c, color="#B0B0B0", alpha=0.18)
        ax2.set_theta_zero_location("E")
        ax2.set_theta_direction(1)
        ax2.set_thetagrids(np.arange(0, 360, 90))
        ax2.set_title("RSP profile + null envelope")
        ann = (
            f"Z_T={float(row['Z_T']):.2f}\n"
            f"q_T={float(row['q_T']):.2e}\n"
            f"C={float(row['coverage_C']):.3f}\n"
            f"K={int(row['peaks_K'])}\n"
            f"phi={float(row['phi_hat_deg']):.1f}°"
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
        ax2.legend(
            loc="upper right", bbox_to_anchor=(1.23, 1.2), fontsize=8, frameon=True
        )

        bins = int(min(45, max(12, np.ceil(np.sqrt(null_t.size)))))
        ax3.hist(null_t, bins=bins, color="#779ECB", edgecolor="white", alpha=0.90)
        ax3.axvline(float(row["T_obs"]), color="#8B0000", linestyle="--", linewidth=2.0)
        ax3.set_xlabel("null_T")
        ax3.set_ylabel("count")
        ax3.set_title("Null T distribution")

        fig.suptitle(f"{gene} ({row['av_group']})", y=1.02, fontsize=12)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"gene_{gene}.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


def _plot_av_axis_summary(scores_df: pd.DataFrame, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_df = scores_df.loc[scores_df["feature_type"] == "gene"].copy()
    if gene_df.empty:
        _save_placeholder(
            out_dir / "av_direction_circle.png",
            "A–V direction",
            "No gene rows to plot.",
        )
        _save_placeholder(
            out_dir / "axis_vs_islands.png", "Axis vs islands", "No gene rows to plot."
        )
        return

    sig = gene_df.loc[gene_df["q_T"] <= Q_SIG].copy()
    if sig.shape[0] >= 3:
        plot_df = sig
        subtitle = "significant genes (q_T<=0.05)"
        alpha = 0.95
    else:
        plot_df = gene_df
        subtitle = "all genes (few significant)"
        alpha = 0.65

    # 1) circular phi_hat plot.
    fig1 = plt.figure(figsize=(6.8, 6.5))
    ax1 = fig1.add_subplot(111, projection="polar")
    for _, row in plot_df.iterrows():
        av_group = str(row["av_group"])
        phi = float(row["phi_hat_rad"])
        size = 50.0 + 12.0 * max(0.0, float(row["Z_T"]))
        ax1.scatter(
            [phi],
            [1.0],
            s=size,
            c=AV_GROUP_COLORS.get(av_group, "#333333"),
            marker=AV_GROUP_MARKERS.get(av_group, "o"),
            alpha=alpha,
            edgecolors="black",
            linewidths=0.7,
        )
        ax1.text(phi, 1.08, str(row["gene"]), fontsize=8, ha="center", va="center")
    ax1.set_ylim(0.0, 1.2)
    ax1.set_rticks([])
    ax1.set_theta_zero_location("E")
    ax1.set_theta_direction(1)
    ax1.set_thetagrids(np.arange(0, 360, 90))
    ax1.set_title(f"Arterial vs venous phi-hat ({subtitle})")
    handles = [
        mlines.Line2D(
            [],
            [],
            color=AV_GROUP_COLORS[g],
            marker=AV_GROUP_MARKERS[g],
            linestyle="None",
            label=g,
        )
        for g in ["arterial", "venous", "pan"]
    ]
    ax1.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.25, 1.15),
        fontsize=8,
        frameon=True,
    )
    fig1.tight_layout()
    fig1.savefig(
        out_dir / "av_direction_circle.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
        bbox_inches="tight",
    )
    plt.close(fig1)

    # 2) axis vs islands plot.
    fig2, ax2 = plt.subplots(figsize=(8.0, 6.0))
    rng = np.random.default_rng(0)
    for _, row in gene_df.iterrows():
        cls = str(row["class_label"])
        x = float(row["peaks_K"]) + rng.uniform(-0.10, 0.10)
        y = float(row["coverage_C"])
        size = 35.0 + 9.0 * max(0.0, float(row["Z_T"]))
        edge = "#000000"
        if cls == "Localized–unimodal":
            edge = "#1F77B4"
        elif cls == "Localized–multimodal":
            edge = "#FF7F0E"
        ax2.scatter(
            [x],
            [y],
            s=size,
            c=AV_GROUP_COLORS.get(str(row["av_group"]), "#333333"),
            marker=AV_GROUP_MARKERS.get(str(row["av_group"]), "o"),
            alpha=0.90,
            edgecolors=edge,
            linewidths=1.1,
        )
        if cls in {"Localized–unimodal", "Localized–multimodal"}:
            ax2.text(x + 0.03, y + 0.004, str(row["gene"]), fontsize=8)
    ax2.set_xlabel("peaks_K (jittered)")
    ax2.set_ylabel("coverage_C")
    ax2.set_title("Axis-like (unimodal) vs island-like (multimodal)")
    ax2.grid(alpha=0.25, linewidth=0.6)
    fig2.tight_layout()
    fig2.savefig(
        out_dir / "axis_vs_islands.png", dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight"
    )
    plt.close(fig2)


def _plot_score_space(scores_df: pd.DataFrame, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if scores_df.empty:
        _save_placeholder(
            out_dir / "score1_score2_scatter.png", "Score space", "No rows."
        )
        _save_placeholder(
            out_dir / "classification_scatter.png", "Classification scatter", "No rows."
        )
        _save_placeholder(out_dir / "class_counts.png", "Class counts", "No rows.")
        return

    # 1) score space scatter.
    fig1, ax1 = plt.subplots(figsize=(8.5, 6.3))
    for _, row in scores_df.iterrows():
        cls = str(row["class_label"])
        av_group = str(row["av_group"])
        ax1.scatter(
            float(row["score_1"]),
            float(row["score_2"]),
            s=95,
            c=CLASS_COLORS.get(cls, "#333333"),
            marker=AV_GROUP_MARKERS.get(av_group, "o"),
            edgecolors="black",
            linewidths=0.7,
            alpha=0.88,
        )
    top = scores_df.sort_values(by="Z_T", ascending=False).head(10)
    for _, row in top.iterrows():
        ax1.text(
            float(row["score_1"]) + 0.05,
            float(row["score_2"]) + 0.003,
            str(row["gene"]),
            fontsize=8,
        )
    ax1.set_xlabel("score_1 = Z_T")
    ax1.set_ylabel("score_2 = coverage_C")
    ax1.set_title("EC A–V score space")
    ax1.grid(alpha=0.25, linewidth=0.6)
    class_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color=CLASS_COLORS[c],
            label=c,
            markersize=8,
        )
        for c in CLASS_ORDER
    ]
    group_handles = [
        mlines.Line2D(
            [],
            [],
            marker=AV_GROUP_MARKERS[g],
            linestyle="None",
            color="black",
            markerfacecolor="white",
            label=g,
            markersize=8,
        )
        for g in ["arterial", "venous", "pan", "qc"]
    ]
    leg1 = ax1.legend(
        handles=class_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="class",
    )
    ax1.add_artist(leg1)
    ax1.legend(
        handles=group_handles,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        title="av_group",
    )
    fig1.tight_layout()
    fig1.savefig(
        out_dir / "score1_score2_scatter.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
        bbox_inches="tight",
    )
    plt.close(fig1)

    # 2) classification scatter with references.
    fig2, ax2 = plt.subplots(figsize=(8.5, 6.3))
    for cls in CLASS_ORDER:
        sub = scores_df.loc[scores_df["class_label"] == cls]
        if sub.empty:
            continue
        ax2.scatter(
            sub["score_1"].to_numpy(dtype=float),
            sub["score_2"].to_numpy(dtype=float),
            s=90,
            c=CLASS_COLORS.get(cls, "#333333"),
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            alpha=0.86,
            label=f"{cls} (n={sub.shape[0]})",
        )
    ax2.axvline(Z_STRONG, color="black", linestyle="--", linewidth=1.2)
    ax2.axhline(COVERAGE_STRONG, color="black", linestyle="-.", linewidth=1.2)
    ax2.set_xlabel("Z_T")
    ax2.set_ylabel("coverage_C")
    ax2.set_title("Classification map with heuristic references")
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    fig2.tight_layout()
    fig2.savefig(
        out_dir / "classification_scatter.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
        bbox_inches="tight",
    )
    plt.close(fig2)

    # 3) class counts with av-group breakdown.
    grp = (
        scores_df.pivot_table(
            index="class_label",
            columns="av_group",
            values="gene",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(CLASS_ORDER, fill_value=0)
        .reindex(columns=["arterial", "venous", "pan", "qc"], fill_value=0)
    )
    fig3, ax3 = plt.subplots(figsize=(9.2, 4.8))
    bottoms = np.zeros(grp.shape[0], dtype=float)
    x = np.arange(grp.shape[0], dtype=float)
    for group in grp.columns:
        vals = grp[group].to_numpy(dtype=float)
        ax3.bar(
            x,
            vals,
            bottom=bottoms,
            color=AV_GROUP_COLORS.get(group, "#333333"),
            edgecolor="black",
            linewidth=0.5,
            label=group,
        )
        bottoms += vals
    ax3.set_xticks(x)
    ax3.set_xticklabels(grp.index.tolist(), rotation=28, ha="right")
    ax3.set_ylabel("Feature count")
    ax3.set_title("Class counts (stacked by av_group)")
    ax3.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    fig3.tight_layout()
    fig3.savefig(
        out_dir / "class_counts.png", dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight"
    )
    plt.close(fig3)


def _plot_moran_baseline(
    scores_df: pd.DataFrame,
    *,
    panels_dir: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_df = scores_df.loc[
        (scores_df["feature_type"] == "gene") & np.isfinite(scores_df["MoranI"])
    ].copy()
    if gene_df.empty:
        _save_placeholder(
            out_dir / "MoranI_vs_ZT.png", "MoranI vs Z_T", "No finite MoranI values."
        )
        return

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for cls in CLASS_ORDER:
        sub = gene_df.loc[gene_df["class_label"] == cls]
        if sub.empty:
            continue
        ax.scatter(
            sub["MoranI"].to_numpy(dtype=float),
            sub["Z_T"].to_numpy(dtype=float),
            s=85,
            c=CLASS_COLORS.get(cls, "#333333"),
            alpha=0.88,
            edgecolors="black",
            linewidths=0.6,
            label=cls,
        )
    med_m = float(np.nanmedian(gene_df["MoranI"].to_numpy(dtype=float)))
    med_z = float(np.nanmedian(gene_df["Z_T"].to_numpy(dtype=float)))
    ax.axvline(med_m, color="#333333", linestyle="--", linewidth=1.1)
    ax.axhline(med_z, color="#333333", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Moran's I (continuous expression)")
    ax.set_ylabel("Z_T")
    ax.set_title("MoranI vs BioRSP anisotropy (within EC)")
    ax.grid(alpha=0.25, linewidth=0.6)

    # Label one salient point per quadrant.
    qdefs = [
        ("Q1_highM_highZ", (gene_df["MoranI"] >= med_m) & (gene_df["Z_T"] >= med_z)),
        ("Q2_highM_lowZ", (gene_df["MoranI"] >= med_m) & (gene_df["Z_T"] < med_z)),
        ("Q3_lowM_highZ", (gene_df["MoranI"] < med_m) & (gene_df["Z_T"] >= med_z)),
        ("Q4_lowM_lowZ", (gene_df["MoranI"] < med_m) & (gene_df["Z_T"] < med_z)),
    ]
    exemplars: list[tuple[str, str]] = []
    for qname, mask in qdefs:
        sub = gene_df.loc[mask]
        if sub.empty:
            continue
        chosen = sub.sort_values(by="Z_T", ascending=False).iloc[0]
        ax.text(
            float(chosen["MoranI"]) + 0.003,
            float(chosen["Z_T"]) + 0.05,
            str(chosen["gene"]),
            fontsize=8,
        )
        exemplars.append((qname, str(chosen["gene"])))

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(
        out_dir / "MoranI_vs_ZT.png", dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight"
    )
    plt.close(fig)

    # Copy quadrant exemplars from per-gene panels for quick figure assembly.
    ex_dir = out_dir / "quadrant_exemplars"
    ex_dir.mkdir(parents=True, exist_ok=True)
    for qname, gene in exemplars:
        src = panels_dir / f"gene_{gene}.png"
        if src.exists():
            dst = ex_dir / f"{qname}_{gene}.png"
            shutil.copy2(src, dst)


def _plot_controls(
    scores_df: pd.DataFrame,
    donor_diag_df: pd.DataFrame | None,
    *,
    out_dir: Path,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings_out: list[str] = []

    if scores_df.empty:
        _save_placeholder(out_dir / "qc_risk_vs_ZT.png", "qc_risk vs Z_T", "No rows.")
        _save_placeholder(out_dir / "sim_qc_vs_ZT.png", "sim_qc vs Z_T", "No rows.")
        _save_placeholder(
            out_dir / "donor_directionality_diagnostic.png",
            "Donor diagnostic",
            "No rows.",
        )
        return warnings_out

    # 1) qc_risk vs Z_T
    fig1, ax1 = plt.subplots(figsize=(7.5, 5.7))
    ax1.scatter(
        scores_df["qc_risk"].to_numpy(dtype=float),
        scores_df["Z_T"].to_numpy(dtype=float),
        c=[CLASS_COLORS.get(c, "#333333") for c in scores_df["class_label"].tolist()],
        s=80,
        alpha=0.86,
        edgecolors="black",
        linewidths=0.6,
    )
    ax1.axvline(QC_THRESH, color="#8B0000", linestyle="--", linewidth=1.2)
    ax1.axhline(Z_STRONG, color="#404040", linestyle="-.", linewidth=1.2)
    qc_driven = scores_df.loc[scores_df["class_label"] == "QC-driven"]
    for _, row in qc_driven.iterrows():
        ax1.text(
            float(row["qc_risk"]) + 0.01,
            float(row["Z_T"]) + 0.05,
            str(row["gene"]),
            fontsize=8,
        )
    ax1.set_xlabel("qc_risk = max |Spearman rho|")
    ax1.set_ylabel("Z_T")
    ax1.set_title("QC mimicry: qc_risk vs Z_T")
    ax1.grid(alpha=0.25, linewidth=0.6)
    fig1.tight_layout()
    fig1.savefig(out_dir / "qc_risk_vs_ZT.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    # 2) sim_qc vs Z_T
    sim_mask = np.isfinite(scores_df["sim_qc"].to_numpy(dtype=float))
    if int(sim_mask.sum()) > 0:
        sim_df = scores_df.loc[sim_mask]
        fig2, ax2 = plt.subplots(figsize=(7.5, 5.7))
        ax2.scatter(
            sim_df["sim_qc"].to_numpy(dtype=float),
            sim_df["Z_T"].to_numpy(dtype=float),
            c=[CLASS_COLORS.get(c, "#333333") for c in sim_df["class_label"].tolist()],
            s=80,
            alpha=0.86,
            edgecolors="black",
            linewidths=0.6,
        )
        ax2.axvline(SIM_QC_THRESH, color="#8B0000", linestyle="--", linewidth=1.2)
        ax2.axhline(Z_STRONG, color="#404040", linestyle="-.", linewidth=1.2)
        for _, row in sim_df.loc[sim_df["class_label"] == "QC-driven"].iterrows():
            ax2.text(
                float(row["sim_qc"]) + 0.01,
                float(row["Z_T"]) + 0.05,
                str(row["gene"]),
                fontsize=8,
            )
        ax2.set_xlabel("sim_qc = max cosine(E_phi, E_phi_qc)")
        ax2.set_ylabel("Z_T")
        ax2.set_title("QC mimicry: sim_qc vs Z_T")
        ax2.grid(alpha=0.25, linewidth=0.6)
        fig2.tight_layout()
        fig2.savefig(out_dir / "sim_qc_vs_ZT.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig2)
    else:
        _save_placeholder(
            out_dir / "sim_qc_vs_ZT.png",
            "sim_qc vs Z_T",
            "QC similarity unavailable (no QC profiles).",
        )

    # 3) donor-directionality diagnostic.
    if donor_diag_df is None or donor_diag_df.empty:
        _save_placeholder(
            out_dir / "donor_directionality_diagnostic.png",
            "Donor directionality diagnostic",
            "Donor key unavailable.",
        )
        return warnings_out

    fig3, ax3 = plt.subplots(figsize=(7.8, 5.8))
    ax3.scatter(
        scores_df["Z_T"].to_numpy(dtype=float),
        scores_df["coverage_C"].to_numpy(dtype=float),
        c="#BFBFBF",
        s=55,
        alpha=0.70,
        edgecolors="white",
        linewidths=0.7,
        label="features",
    )
    ax3.scatter(
        donor_diag_df["Z_T"].to_numpy(dtype=float),
        donor_diag_df["coverage_C"].to_numpy(dtype=float),
        c="#D62728",
        marker="X",
        s=125,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.8,
        label="donor vs all",
    )
    for _, row in donor_diag_df.iterrows():
        ax3.text(
            float(row["Z_T"]) + 0.05,
            float(row["coverage_C"]) + 0.004,
            str(row["donor_id"]),
            fontsize=7,
        )
    ax3.axvline(Z_STRONG, color="black", linestyle="--", linewidth=1.1)
    ax3.axhline(COVERAGE_STRONG, color="black", linestyle="-.", linewidth=1.1)
    ax3.set_xlabel("Z_T")
    ax3.set_ylabel("coverage_C")
    ax3.set_title("Donor-directionality diagnostic")
    ax3.grid(alpha=0.25, linewidth=0.6)
    ax3.legend(loc="best", fontsize=8, frameon=True)
    fig3.tight_layout()
    fig3.savefig(
        out_dir / "donor_directionality_diagnostic.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig3)

    strong = donor_diag_df.loc[
        (donor_diag_df["q_T"] <= Q_SIG) & (donor_diag_df["Z_T"] >= Z_STRONG)
    ]
    if not strong.empty:
        warnings_out.append(
            f"Donor-directionality strong for {strong.shape[0]} donor(s) with q_T<=0.05 and Z_T>=4."
        )
    return warnings_out


def main() -> int:
    args = parse_args()
    apply_plot_style()

    h5ad_path = Path(args.h5ad)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Input h5ad missing: {h5ad_path}")

    outdir = Path(args.out)
    tables_dir = outdir / "tables"
    plots_dir = outdir / "plots"
    p_overview = plots_dir / "00_overview"
    p_gene_panels = plots_dir / "01_gene_panels"
    p_axis = plots_dir / "02_av_axis_summary"
    p_score = plots_dir / "03_score_space"
    p_moran = plots_dir / "04_morans_baseline"
    p_controls = plots_dir / "05_controls"
    for d in [
        tables_dir,
        p_overview,
        p_gene_panels,
        p_axis,
        p_score,
        p_moran,
        p_controls,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    warnings_log: list[str] = []

    adata = ad.read_h5ad(h5ad_path)
    _ensure_umap(adata, seed=int(args.seed), recompute_if_missing=False)
    embedding_key, umap_xy = _resolve_embedding(adata, args.embedding_key)
    expr_matrix, adata_like, expr_source = _choose_expression_source(
        adata,
        layer_arg=args.layer,
        use_raw_arg=bool(args.use_raw),
    )

    donor_ids_all, donor_key_used = _resolve_donor_ids(adata, args.donor_key)
    if donor_ids_all is None:
        msg = "Donor key unavailable or <2 donors. Falling back to global permutation where needed."
        print(f"WARNING: {msg}")
        warnings_log.append(msg)

    label_key = _resolve_label_key(adata, args.label_key)
    if label_key is None:
        raise RuntimeError(
            "No label key found. Tried: azimuth_label, predicted_label, predicted_CLID, cell_type."
        )

    labels = adata.obs[label_key].astype("string").fillna("NA").astype(str)
    ec_mask, ec_labels_included, ec_counts = _build_ec_mask(labels)
    if int(np.sum(ec_mask)) == 0:
        raise RuntimeError(
            "Endothelial subset is empty after label filtering. "
            "Check label key and endothelial token matching rules."
        )

    for lbl, cnt in ec_counts.items():
        print(f"ec_label_included: {lbl} -> {int(cnt)}")

    subset_underpowered = bool(int(np.sum(ec_mask)) < int(args.min_cells))
    if subset_underpowered:
        msg = (
            f"EC subset has n_cells={int(np.sum(ec_mask))} < min_cells={int(args.min_cells)}; "
            "continuing with subset_underpowered=True."
        )
        print(f"WARNING: {msg}")
        warnings_log.append(msg)

    total_counts = _total_counts_vector(adata, expr_matrix)
    pct_mt_raw, pct_mt_source = _pct_mt_vector(adata, expr_matrix, adata_like)
    pct_mt_all = (
        None if pct_mt_source == "proxy:zeros" else np.asarray(pct_mt_raw, dtype=float)
    )
    pct_ribo_all, pct_ribo_source = _compute_pct_counts_ribo(
        adata, expr_matrix, adata_like, total_counts
    )
    if pct_mt_all is None:
        msg = "pct_counts_mt unavailable; qc_risk excludes this covariate."
        print(f"WARNING: {msg}")
        warnings_log.append(msg)
    if pct_ribo_all is None:
        msg = "pct_counts_ribo unavailable; qc_risk excludes this covariate."
        print(f"WARNING: {msg}")
        warnings_log.append(msg)

    qc_available = {
        "total_counts": True,
        "pct_counts_mt": pct_mt_all is not None,
        "pct_counts_ribo": pct_ribo_all is not None,
    }
    resolved_features, panel_df = _resolve_av_panel(
        adata_like, qc_available=qc_available
    )
    panel_csv = tables_dir / "av_marker_panel.csv"
    panel_df.to_csv(panel_csv, index=False)

    missing_features = (
        panel_df.loc[~panel_df["found"].astype(bool), "feature"].astype(str).tolist()
    )
    if missing_features:
        print(f"missing_features={','.join(missing_features)}")

    # Subset arrays.
    ec_idx = np.flatnonzero(ec_mask).astype(int)
    ec_xy = umap_xy[ec_idx]
    ec_center = compute_vantage_point(ec_xy, method="median")
    theta = compute_theta(ec_xy, ec_center)
    _, bin_id = bin_theta(theta, int(args.n_bins))
    bin_counts_total = np.bincount(bin_id, minlength=int(args.n_bins)).astype(float)

    donor_ids = donor_ids_all[ec_idx] if donor_ids_all is not None else None
    donors_n = int(np.unique(donor_ids).size) if donor_ids is not None else 0
    total_counts_sub = np.asarray(total_counts[ec_idx], dtype=float)
    pct_mt_sub = (
        np.asarray(pct_mt_all[ec_idx], dtype=float) if pct_mt_all is not None else None
    )
    pct_ribo_sub = (
        np.asarray(pct_ribo_all[ec_idx], dtype=float)
        if pct_ribo_all is not None
        else None
    )

    # Prepare Moran's I weights for subset.
    subset_adata = adata[ec_idx].copy()
    try:
        _ensure_subset_connectivities(subset_adata, ec_xy, k=15)
        weights = extract_weights(subset_adata)
    except Exception as exc:
        weights = None
        msg = f"Moran's I weights unavailable ({exc}); MoranI will be NaN."
        print(f"WARNING: {msg}")
        warnings_log.append(msg)

    print(f"embedding_key_used={embedding_key}")
    print(f"label_key_used={label_key}")
    print(f"donor_key_used={donor_key_used if donor_key_used is not None else 'None'}")
    print(f"expression_source_used={expr_source}")
    print(f"pct_counts_mt_source={pct_mt_source}")
    print(f"pct_counts_ribo_source={pct_ribo_source}")
    print(
        f"ec_n_cells={int(ec_idx.size)} n_bins={int(args.n_bins)} n_perm={int(args.n_perm)} "
        f"seed={int(args.seed)}"
    )

    # Plot overview block first.
    _plot_overview(
        adata=adata,
        umap_xy=umap_xy,
        labels=labels,
        ec_mask=ec_mask,
        ec_counts=ec_counts,
        donor_key_used=donor_key_used,
        ec_center=ec_center,
        out_dir=p_overview,
    )

    # Score QC pseudo-features as continuous weights and keep their profiles.
    qc_vectors = {
        "total_counts": total_counts_sub,
        "pct_counts_mt": pct_mt_sub,
        "pct_counts_ribo": pct_ribo_sub,
    }
    qc_profiles: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, np.ndarray]] = {}

    row_counter = 0
    for qc_name in QC_PSEUDOGENES:
        vals = qc_vectors.get(qc_name)
        if vals is None:
            continue
        row_counter += 1
        perm = _perm_null_continuous_profile(
            np.asarray(vals, dtype=float),
            donor_ids=donor_ids,
            n_bins=int(args.n_bins),
            n_perm=int(args.n_perm),
            seed=int(args.seed + 10_000 + row_counter * 17),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
        null_e = np.asarray(perm["null_E_phi"], dtype=float)
        null_t = np.asarray(perm["null_T"], dtype=float)
        qc_profiles[qc_name] = e_obs

        prev_qc = float(np.mean(np.asarray(vals, dtype=float) > 0.0))
        n_fg_qc = int(np.sum(np.asarray(vals, dtype=float) > 0.0))
        peak_idx = int(np.argmax(np.abs(e_obs))) if e_obs.size > 0 else 0
        centers = theta_bin_centers(int(args.n_bins))
        phi_hat = float(centers[peak_idx]) if centers.size > 0 else 0.0
        moran_qc = float("nan")
        if weights is not None:
            try:
                if float(np.nanstd(vals)) > 0:
                    moran_qc = float(morans_i(np.asarray(vals, dtype=float), weights))
            except Exception:
                moran_qc = float("nan")

        rows.append(
            {
                "gene": qc_name,
                "feature_type": "qc",
                "av_group": "qc",
                "prev": prev_qc,
                "n_fg": n_fg_qc,
                "n_cells": int(ec_idx.size),
                "T_obs": float(perm["T_obs"]),
                "p_T": float(perm["p_T"]),
                "q_T": float("nan"),
                "Z_T": float(robust_z(float(perm["T_obs"]), null_t)),
                "coverage_C": float(coverage_from_null(e_obs, null_e, q=0.95)),
                "peaks_K": int(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95)),
                "phi_hat_rad": phi_hat,
                "phi_hat_deg": float(np.degrees(phi_hat) % 360.0),
                "qc_risk": float("nan"),
                "sim_qc": float("nan"),
                "donor_key_used": donor_key_used if donor_key_used is not None else "",
                "donors_n": int(donors_n),
                "used_donor_stratified": bool(perm["used_donor_stratified"]),
                "subset_underpowered": bool(subset_underpowered),
                "MoranI": moran_qc,
                "score_1": float(robust_z(float(perm["T_obs"]), null_t)),
                "score_2": float(coverage_from_null(e_obs, null_e, q=0.95)),
                "class_label": "",
                "perm_warning": str(perm.get("warning", "")),
            }
        )

    # Score gene panel rows.
    for item in resolved_features:
        if item.feature_type != "gene" or not item.found or item.gene_idx is None:
            continue
        row_counter += 1
        expr_full = get_feature_vector(expr_matrix, int(item.gene_idx))
        expr = np.asarray(expr_full[ec_idx], dtype=float)
        f = expr > 0.0

        if int(f.sum()) in {0, f.size}:
            e_obs_direct = np.zeros(int(args.n_bins), dtype=float)
        else:
            e_obs_direct, _, _, _ = compute_rsp_profile_from_boolean(
                f,
                theta,
                int(args.n_bins),
                bin_id=bin_id,
                bin_counts_total=bin_counts_total,
            )

        perm = perm_null_T_and_profile(
            expr=expr,
            theta=theta,
            donor_ids=donor_ids,
            n_bins=int(args.n_bins),
            n_perm=int(args.n_perm),
            seed=int(args.seed + 20_000 + row_counter * 31),
            donor_stratified=True,
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
        if e_obs.size != int(args.n_bins):
            e_obs = e_obs_direct
        null_e = np.asarray(perm["null_E_phi"], dtype=float)
        null_t = np.asarray(perm["null_T"], dtype=float)
        t_obs = float(perm["T_obs"])
        z_t = float(robust_z(t_obs, null_t))

        coverage_c = float(coverage_from_null(e_obs, null_e, q=0.95))
        peaks_k = int(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95))
        peak_idx = int(np.argmax(np.abs(e_obs))) if e_obs.size > 0 else 0
        centers = theta_bin_centers(int(args.n_bins))
        phi_hat = float(centers[peak_idx]) if centers.size > 0 else 0.0

        rho_total = _safe_spearman(f.astype(float), total_counts_sub)
        rho_mt = _safe_spearman(f.astype(float), pct_mt_sub)
        rho_ribo = _safe_spearman(f.astype(float), pct_ribo_sub)
        finite_rho = [abs(v) for v in [rho_total, rho_mt, rho_ribo] if np.isfinite(v)]
        qc_risk = float(max(finite_rho)) if finite_rho else 0.0

        sims: list[float] = []
        for q_prof in qc_profiles.values():
            sim = _cosine_similarity(e_obs, q_prof)
            if np.isfinite(sim):
                sims.append(sim)
        sim_qc = float(max(sims)) if sims else float("nan")

        moran_val = float("nan")
        if weights is not None:
            try:
                if float(np.nanstd(expr)) > 0:
                    moran_val = float(morans_i(expr, weights))
            except Exception:
                moran_val = float("nan")

        rows.append(
            {
                "gene": item.feature,
                "feature_type": "gene",
                "av_group": item.av_group,
                "prev": float(np.mean(f)),
                "n_fg": int(f.sum()),
                "n_cells": int(f.size),
                "T_obs": t_obs,
                "p_T": float(perm["p_T"]),
                "q_T": float("nan"),
                "Z_T": z_t,
                "coverage_C": coverage_c,
                "peaks_K": peaks_k,
                "phi_hat_rad": phi_hat,
                "phi_hat_deg": float(np.degrees(phi_hat) % 360.0),
                "qc_risk": qc_risk,
                "sim_qc": sim_qc,
                "donor_key_used": donor_key_used if donor_key_used is not None else "",
                "donors_n": int(donors_n),
                "used_donor_stratified": bool(perm["used_donor_stratified"]),
                "subset_underpowered": bool(subset_underpowered),
                "MoranI": moran_val,
                "score_1": z_t,
                "score_2": coverage_c,
                "class_label": "",
                "perm_warning": str(perm.get("warning", "")),
            }
        )
        artifacts[item.feature] = {
            "expr": expr,
            "E_phi_obs": e_obs,
            "null_E_phi": null_e,
            "null_T": null_t,
        }

    scores_df = pd.DataFrame(rows)
    if scores_df.empty:
        raise RuntimeError("No features were scored in EC subset; cannot continue.")

    scores_df["q_T"] = bh_fdr(
        pd.to_numeric(scores_df["p_T"], errors="coerce").to_numpy(dtype=float)
    )
    scores_df["class_label"] = scores_df.apply(_classify_row, axis=1)
    scores_df = scores_df.sort_values(
        by=["feature_type", "av_group", "q_T", "gene"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # Donor-directionality diagnostic for controls plot.
    donor_diag_df: pd.DataFrame | None = None
    if donor_ids is not None and donors_n >= 2:
        diag_rows: list[dict[str, Any]] = []
        donor_arr = np.asarray(donor_ids).astype(str)
        for i, donor in enumerate(sorted(np.unique(donor_arr).tolist())):
            expr_d = (donor_arr == donor).astype(float)
            perm = perm_null_T_and_profile(
                expr=expr_d,
                theta=theta,
                donor_ids=None,
                n_bins=int(args.n_bins),
                n_perm=int(args.n_perm),
                seed=int(args.seed + 40_000 + i * 13),
                donor_stratified=False,
                bin_id=bin_id,
                bin_counts_total=bin_counts_total,
            )
            e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
            null_e = np.asarray(perm["null_E_phi"], dtype=float)
            null_t = np.asarray(perm["null_T"], dtype=float)
            diag_rows.append(
                {
                    "donor_id": donor,
                    "n_cells": int(np.sum(donor_arr == donor)),
                    "T_obs": float(perm["T_obs"]),
                    "p_T": float(perm["p_T"]),
                    "q_T": float("nan"),
                    "Z_T": float(robust_z(float(perm["T_obs"]), null_t)),
                    "coverage_C": float(coverage_from_null(e_obs, null_e, q=0.95)),
                }
            )
        donor_diag_df = pd.DataFrame(diag_rows)
        donor_diag_df["q_T"] = bh_fdr(donor_diag_df["p_T"].to_numpy(dtype=float))
        donor_diag_df.to_csv(
            tables_dir / "donor_directionality_scores.csv", index=False
        )

    scores_csv = tables_dir / "ec_av_scores.csv"
    scores_df.to_csv(scores_csv, index=False)

    # Plot blocks.
    _plot_gene_panels(
        ec_xy=ec_xy,
        ec_center=ec_center,
        scores_df=scores_df,
        artifacts=artifacts,
        n_bins=int(args.n_bins),
        out_dir=p_gene_panels,
    )
    _plot_av_axis_summary(scores_df, out_dir=p_axis)
    _plot_score_space(scores_df, out_dir=p_score)
    _plot_moran_baseline(scores_df, panels_dir=p_gene_panels, out_dir=p_moran)
    warnings_log.extend(_plot_controls(scores_df, donor_diag_df, out_dir=p_controls))

    # Save supplementary EC QC map for reviewer convenience.
    if pct_mt_sub is not None:
        save_numeric_umap(
            umap_xy=ec_xy,
            values=np.asarray(pct_mt_sub, dtype=float),
            out_png=p_overview / "umap_ec_pct_mt.png",
            title="EC subset: pct_counts_mt",
            cmap="magma",
            colorbar_label="pct_counts_mt",
            vantage_point=(float(ec_center[0]), float(ec_center[1])),
        )

    # README metadata and warnings.
    readme_lines = [
        "Experiment #3: Endothelial A–V axis benchmark (geometry validation)",
        "",
        "Representation-conditional note: directional scores are interpreted in embedding space only.",
        "No physical tissue-direction claims are made.",
        "",
        f"embedding_key_used: {embedding_key}",
        f"donor_key_used: {donor_key_used if donor_key_used is not None else 'None'}",
        f"celltype_key_used: {label_key}",
        f"expression_source_used: {expr_source}",
        f"n_cells_total: {int(adata.n_obs)}",
        f"n_cells_ec_subset: {int(ec_idx.size)}",
        f"n_bins: {int(args.n_bins)}",
        f"n_perm: {int(args.n_perm)}",
        "",
        "Included EC labels (count):",
    ]
    for lbl, cnt in ec_counts.items():
        readme_lines.append(f"- {lbl}: {int(cnt)}")
    readme_lines.append("")
    readme_lines.append("Warnings:")
    if warnings_log:
        for msg in warnings_log:
            readme_lines.append(f"- {msg}")
    else:
        readme_lines.append("- none")
    (outdir / "README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    # Console summary.
    cls_counts = (
        scores_df["class_label"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
    )
    cls_txt = "; ".join([f"{k}={v}" for k, v in cls_counts.items()])
    print(f"classification_summary={cls_txt}")
    print(f"panel_csv={panel_csv.as_posix()}")
    print(f"scores_csv={scores_csv.as_posix()}")
    print(f"results_root={outdir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
