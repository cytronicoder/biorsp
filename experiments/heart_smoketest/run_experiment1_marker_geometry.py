#!/usr/bin/env python3
"""Experiment #1: directional marker signatures across the full embedding."""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

# Force headless backend for reproducible local/CI execution.
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Allow running this script directly via `python experiments/...` without external PYTHONPATH setup.
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
from biorsp.stats.scoring import bh_fdr, coverage_from_null, peak_count, robust_z

MARKER_PANEL: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("Cardiomyocyte", ["MYH6", "TNNT2", "RYR2", "PLN"]),
        ("Fibroblast/ECM", ["COL1A1", "COL1A2", "LUM", "DCN"]),
        ("Endothelial", ["PECAM1", "VWF", "KDR"]),
        ("Mural", ["ACTA2", "TAGLN", "RGS5"]),
        ("Immune", ["PTPRC", "LST1", "LYZ"]),
    ]
)

PANEL_PROVENANCE = (
    "Pre-registered heart marker panel from the heart case-study plan "
    "(Experiment #1 directional marker geometry)."
)

DONOR_KEY_CANDIDATES = [
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

LABEL_KEY_CANDIDATES = ["azimuth_label", "predicted_label"]

GROUP_COLORS = {
    "Cardiomyocyte": "#A63A50",
    "Fibroblast/ECM": "#2E6F95",
    "Endothelial": "#4CAF50",
    "Mural": "#9C6B3D",
    "Immune": "#7F4E97",
}

CLASS_ORDER = [
    "Underpowered",
    "Ubiquitous",
    "Localized–unimodal",
    "Localized–multimodal",
    "QC-driven",
    "Uncertain",
]

CLASS_COLORS = {
    "Underpowered": "#8C8C8C",
    "Ubiquitous": "#2ca02c",
    "Localized–unimodal": "#1f77b4",
    "Localized–multimodal": "#ff7f0e",
    "QC-driven": "#d62728",
    "Uncertain": "#9467bd",
}

CLASS_MARKERS = {
    "Underpowered": "o",
    "Ubiquitous": "s",
    "Localized–unimodal": "^",
    "Localized–multimodal": "D",
    "QC-driven": "X",
    "Uncertain": "P",
}

QC_THRESH = 0.35
Q_SIG = 0.05
UBIQUITOUS_PREV = 0.60
UNDERPOWERED_MIN_CELLS = 20
UNDERPOWERED_MIN_FRAC = 0.005
Z_STRONG = 4.0
COVERAGE_STRONG = 0.15


@dataclass(frozen=True)
class MarkerResolution:
    gene: str
    marker_group: str
    found: bool
    gene_idx: int | None
    resolved_gene: str
    resolution_source: str
    symbol_column: str
    status: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Experiment #1 directional marker signatures across full heart embedding."
    )
    p.add_argument(
        "--h5ad", default="data/processed/HT_pca_umap.h5ad", help="Input .h5ad file."
    )
    p.add_argument(
        "--out",
        default="experiments/heart_smoketest/results/experiment1_marker_geometry",
        help="Output directory.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument(
        "--n_perm", type=int, default=300, help="Permutation count (default smoke=300)."
    )
    p.add_argument("--n_bins", type=int, default=64, help="Number of angular bins.")
    p.add_argument("--embedding_key", default=None, help="Embedding key in adata.obsm.")
    p.add_argument(
        "--donor_key", default=None, help="Optional donor key override in adata.obs."
    )
    p.add_argument(
        "--label_key", default=None, help="Optional label key override in adata.obs."
    )
    p.add_argument("--layer", default=None, help="Optional expression layer override.")
    p.add_argument(
        "--use_raw", action="store_true", help="Use adata.raw instead of X/layers."
    )
    return p.parse_args()


def _resolve_embedding(
    adata: ad.AnnData, requested_key: str | None
) -> tuple[str, np.ndarray]:
    if requested_key is not None:
        if requested_key not in adata.obsm:
            raise KeyError(
                f"Requested embedding_key '{requested_key}' missing in adata.obsm."
            )
        key = str(requested_key)
    else:
        if "X_umap" in adata.obsm:
            key = "X_umap"
        else:
            umap_like = [k for k in adata.obsm.keys() if "umap" in str(k).lower()]
            if umap_like:
                key = str(umap_like[0])
            elif len(adata.obsm.keys()) > 0:
                key = str(next(iter(adata.obsm.keys())))
            else:
                raise ValueError("No embedding found in adata.obsm.")

    xy = np.asarray(adata.obsm[key], dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(
            f"Embedding '{key}' must have shape (N, 2+) but got {xy.shape}."
        )
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


def _resolve_donor_ids(
    adata: ad.AnnData,
    requested_key: str | None,
) -> tuple[np.ndarray | None, str | None]:
    keys_to_try: list[str] = []
    if requested_key is not None:
        keys_to_try.append(str(requested_key))
    keys_to_try.extend([k for k in DONOR_KEY_CANDIDATES if k not in keys_to_try])

    for key in keys_to_try:
        if key not in adata.obs.columns:
            continue
        donor_ids = adata.obs[key].astype("string").fillna("NA").astype(str).to_numpy()
        if np.unique(donor_ids).size >= 2:
            return donor_ids, key
    return None, None


def _resolve_label_key(adata: ad.AnnData, requested_key: str | None) -> str | None:
    if requested_key is not None:
        return str(requested_key) if requested_key in adata.obs.columns else None
    for key in LABEL_KEY_CANDIDATES:
        if key in adata.obs.columns:
            return key
    return None


def _safe_spearman_binary(
    binary_mask: np.ndarray, covariate: np.ndarray | None
) -> float:
    if covariate is None:
        return float("nan")
    x = np.asarray(binary_mask, dtype=float).ravel()
    y = np.asarray(covariate, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.allclose(xx, xx[0]) or np.allclose(yy, yy[0]):
        return float("nan")
    rho = spearmanr(xx, yy, nan_policy="omit").correlation
    if rho is None or not np.isfinite(float(rho)):
        return float("nan")
    return float(rho)


def _resolve_marker_panel(
    adata_like: Any,
) -> tuple[list[MarkerResolution], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    resolved: list[MarkerResolution] = []
    used_idx: set[int] = set()

    for marker_group, genes in MARKER_PANEL.items():
        for gene in genes:
            try:
                idx, label, symbol_col, source = resolve_feature_index(adata_like, gene)
                idx_i = int(idx)
                if idx_i in used_idx:
                    entry = MarkerResolution(
                        gene=gene,
                        marker_group=marker_group,
                        found=False,
                        gene_idx=None,
                        resolved_gene="",
                        resolution_source=source,
                        symbol_column=symbol_col or "",
                        status="duplicate_index",
                    )
                else:
                    used_idx.add(idx_i)
                    entry = MarkerResolution(
                        gene=gene,
                        marker_group=marker_group,
                        found=True,
                        gene_idx=idx_i,
                        resolved_gene=str(label),
                        resolution_source=source,
                        symbol_column=symbol_col or "",
                        status="resolved",
                    )
            except KeyError:
                entry = MarkerResolution(
                    gene=gene,
                    marker_group=marker_group,
                    found=False,
                    gene_idx=None,
                    resolved_gene="",
                    resolution_source="",
                    symbol_column="",
                    status="missing",
                )

            rows.append(
                {
                    "gene": entry.gene,
                    "group": entry.marker_group,
                    "marker_group": entry.marker_group,
                    "provenance": PANEL_PROVENANCE,
                    "status": entry.status,
                    "found": entry.found,
                    "resolved_gene": entry.resolved_gene,
                    "gene_idx": entry.gene_idx if entry.gene_idx is not None else "",
                    "resolution_source": entry.resolution_source,
                    "symbol_column": entry.symbol_column,
                }
            )
            resolved.append(entry)

    marker_panel_df = pd.DataFrame(rows)
    return resolved, marker_panel_df


def _compute_pct_counts_ribo(
    adata: ad.AnnData,
    expr_matrix: Any,
    adata_like: Any,
    total_counts: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    if "pct_counts_ribo" in adata.obs.columns:
        arr = pd.to_numeric(adata.obs["pct_counts_ribo"], errors="coerce").to_numpy(
            dtype=float
        )
        if np.isfinite(arr).sum() > 0:
            fill = float(np.nanmedian(arr))
            arr = np.where(np.isfinite(arr), arr, fill)
            return arr, "obs:pct_counts_ribo"

    symbol_col = None
    if hasattr(adata_like, "var") and adata_like.var is not None:
        for candidate in ["hugo_symbol", "gene_name", "gene_symbol"]:
            if candidate in adata_like.var.columns:
                symbol_col = candidate
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


def _save_placeholder_plot(out_png: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)


def _classify_row(row: pd.Series) -> str:
    if bool(row["underpowered"]):
        return "Underpowered"
    if bool(row["qc_driven"]):
        return "QC-driven"
    q_t = float(row["q_T"]) if np.isfinite(float(row["q_T"])) else 1.0
    if q_t <= Q_SIG:
        if int(row["peaks_K"]) >= 2:
            return "Localized–multimodal"
        return "Localized–unimodal"
    if float(row["prev"]) >= UBIQUITOUS_PREV:
        return "Ubiquitous"
    return "Uncertain"


def _plot_gene_panel(
    *,
    out_png: Path,
    gene: str,
    marker_group: str,
    expr: np.ndarray,
    umap_xy: np.ndarray,
    center_xy: np.ndarray,
    e_phi_obs: np.ndarray,
    null_e_phi: np.ndarray,
    null_t: np.ndarray,
    n_bins: int,
    row: pd.Series,
) -> None:
    fig = plt.figure(figsize=(16.0, 4.9))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2, projection="polar")
    ax3 = fig.add_subplot(1, 3, 3)

    x = np.asarray(expr, dtype=float)
    x_plot = np.log1p(np.maximum(x, 0.0))
    order = np.argsort(x_plot, kind="mergesort")

    ax1.scatter(
        umap_xy[:, 0],
        umap_xy[:, 1],
        c="#D7D7D7",
        s=4.0,
        alpha=0.30,
        linewidths=0,
        rasterized=True,
    )
    sc = ax1.scatter(
        umap_xy[order, 0],
        umap_xy[order, 1],
        c=x_plot[order],
        cmap="Reds",
        s=6.0,
        alpha=0.85,
        linewidths=0,
        rasterized=True,
    )
    ax1.scatter(
        [float(center_xy[0])],
        [float(center_xy[1])],
        marker="X",
        s=70,
        c="black",
        edgecolors="white",
        linewidths=0.8,
        zorder=10,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"{gene} ({marker_group}) feature map")
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03)
    cbar.set_label("log1p(expression)")

    centers = theta_bin_centers(n_bins)
    theta_closed = np.concatenate([centers, centers[:1]])
    obs_closed = np.concatenate(
        [np.asarray(e_phi_obs, dtype=float), np.asarray(e_phi_obs[:1], dtype=float)]
    )
    q_hi = np.quantile(np.asarray(null_e_phi, dtype=float), 0.95, axis=0)
    q_lo = np.quantile(np.asarray(null_e_phi, dtype=float), 0.05, axis=0)
    q_hi_closed = np.concatenate([q_hi, q_hi[:1]])
    q_lo_closed = np.concatenate([q_lo, q_lo[:1]])

    ax2.plot(
        theta_closed, obs_closed, color="#8B0000", linewidth=2.0, label="E_phi obs"
    )
    ax2.plot(
        theta_closed,
        q_hi_closed,
        color="#444444",
        linestyle="--",
        linewidth=1.4,
        label="null 95%",
    )
    ax2.plot(
        theta_closed,
        q_lo_closed,
        color="#444444",
        linestyle="--",
        linewidth=1.0,
        label="null 5%",
    )
    ax2.fill_between(
        theta_closed, q_lo_closed, q_hi_closed, color="#B0B0B0", alpha=0.20
    )
    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(1)
    ax2.set_thetagrids(np.arange(0, 360, 90))
    ax2.set_title("RSP profile + null envelope")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.25, 1.2), fontsize=8, frameon=True)

    summary_text = (
        f"T_obs={float(row['T_obs']):.4f}\n"
        f"Z_T={float(row['Z_T']):.2f}\n"
        f"q_T={float(row['q_T']):.2e}\n"
        f"coverage_C={float(row['coverage_C']):.3f}\n"
        f"peaks_K={int(row['peaks_K'])}\n"
        f"peak_dir={float(row['peak_dir_phi_deg']):.1f}°"
    )
    ax2.text(
        0.02,
        0.02,
        summary_text,
        transform=ax2.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#888888", "alpha": 0.85},
    )

    nt = np.asarray(null_t, dtype=float).ravel()
    bins = int(min(40, max(12, np.ceil(np.sqrt(nt.size)))))
    ax3.hist(nt, bins=bins, color="#779ECB", edgecolor="white", alpha=0.90)
    ax3.axvline(float(row["T_obs"]), color="#8B0000", linestyle="--", linewidth=2.0)
    ax3.set_title("null_T distribution")
    ax3.set_xlabel("T under null")
    ax3.set_ylabel("count")

    fig.suptitle(
        f"{gene}: full-embedding directional marker geometry", y=1.02, fontsize=12
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_score_space(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if metrics_df.empty:
        _save_placeholder_plot(
            out_dir / "score1_vs_score2_scatter.png",
            "score_1 vs score_2",
            "No marker genes were resolved in this dataset.",
        )
        _save_placeholder_plot(
            out_dir / "classification_scatter.png",
            "classification scatter",
            "No marker genes were resolved in this dataset.",
        )
        _save_placeholder_plot(
            out_dir / "class_counts.png",
            "class counts",
            "No marker genes were resolved in this dataset.",
        )
        _save_placeholder_plot(
            out_dir / "heatmap_scores.png",
            "heatmap scores",
            "No marker genes were resolved in this dataset.",
        )
        return

    # 1) score1 vs score2 with group colors + class marker shape.
    fig1, ax1 = plt.subplots(figsize=(8.0, 6.0))
    for _, row in metrics_df.iterrows():
        cls = str(row["class_label"])
        grp = str(row["marker_group"])
        ax1.scatter(
            float(row["score_1"]),
            float(row["score_2"]),
            s=90,
            marker=CLASS_MARKERS.get(cls, "o"),
            c=GROUP_COLORS.get(grp, "#222222"),
            edgecolors=CLASS_COLORS.get(cls, "#111111"),
            linewidths=1.2,
            alpha=0.92,
            zorder=3,
        )
    for _, row in (
        metrics_df.sort_values(by="score_1", ascending=False).head(8).iterrows()
    ):
        ax1.text(
            float(row["score_1"]) + 0.05,
            float(row["score_2"]) + 0.004,
            str(row["gene"]),
            fontsize=8,
        )
    ax1.set_xlabel("score_1 = Z_T")
    ax1.set_ylabel("score_2 = coverage_C")
    ax1.set_title("Score space (color=marker_group, marker/edge=class_label)")
    ax1.grid(alpha=0.25, linewidth=0.6)

    group_handles = [
        mlines.Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markerfacecolor=GROUP_COLORS[g],
            markeredgecolor="black",
            label=g,
            markersize=8,
        )
        for g in MARKER_PANEL.keys()
    ]
    class_handles = [
        mlines.Line2D(
            [],
            [],
            linestyle="None",
            marker=CLASS_MARKERS[c],
            markerfacecolor="white",
            markeredgecolor=CLASS_COLORS[c],
            label=c,
            markersize=8,
        )
        for c in CLASS_ORDER
    ]
    leg1 = ax1.legend(
        handles=group_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="marker_group",
    )
    ax1.add_artist(leg1)
    ax1.legend(
        handles=class_handles,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        title="class_label",
    )
    fig1.tight_layout()
    fig1.savefig(
        out_dir / "score1_vs_score2_scatter.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
        bbox_inches="tight",
    )
    plt.close(fig1)

    # 2) classification scatter with heuristic lines.
    fig2, ax2 = plt.subplots(figsize=(8.0, 6.0))
    for cls in CLASS_ORDER:
        sub = metrics_df.loc[metrics_df["class_label"] == cls]
        if sub.empty:
            continue
        ax2.scatter(
            sub["score_1"].to_numpy(dtype=float),
            sub["score_2"].to_numpy(dtype=float),
            s=90,
            marker=CLASS_MARKERS.get(cls, "o"),
            c=CLASS_COLORS.get(cls, "#222222"),
            edgecolors="black",
            linewidths=0.7,
            alpha=0.88,
            label=f"{cls} (n={sub.shape[0]})",
        )
    ax2.axvline(
        Z_STRONG, color="black", linestyle="--", linewidth=1.2, label="Z_T=4 heuristic"
    )
    ax2.axhline(
        COVERAGE_STRONG,
        color="black",
        linestyle="-.",
        linewidth=1.2,
        label="coverage_C=0.15 heuristic",
    )
    ax2.set_xlabel("score_1 = Z_T")
    ax2.set_ylabel("score_2 = coverage_C")
    ax2.set_title("Classification scatter")
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=8)
    fig2.tight_layout()
    fig2.savefig(
        out_dir / "classification_scatter.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
        bbox_inches="tight",
    )
    plt.close(fig2)

    # 3) class counts with marker-group breakdown (stacked bar).
    class_group = (
        metrics_df.pivot_table(
            index="class_label",
            columns="marker_group",
            values="gene",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(CLASS_ORDER, fill_value=0)
        .reindex(columns=list(MARKER_PANEL.keys()), fill_value=0)
    )
    fig3, ax3 = plt.subplots(figsize=(9.0, 4.8))
    bottoms = np.zeros(class_group.shape[0], dtype=float)
    x = np.arange(class_group.shape[0], dtype=float)
    for group in class_group.columns:
        vals = class_group[group].to_numpy(dtype=float)
        ax3.bar(
            x,
            vals,
            bottom=bottoms,
            color=GROUP_COLORS.get(group, "#333333"),
            edgecolor="black",
            linewidth=0.5,
            label=group,
        )
        bottoms += vals
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_group.index.tolist(), rotation=25, ha="right")
    ax3.set_ylabel("Marker count")
    ax3.set_title("Class counts (stacked by marker_group)")
    ax3.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=8)
    ax3.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig3.tight_layout()
    fig3.savefig(
        out_dir / "class_counts.png", dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight"
    )
    plt.close(fig3)

    # 4) heatmap of requested score columns.
    hm = metrics_df.copy()
    hm["neglog10_q_T"] = -np.log10(
        np.clip(hm["q_T"].to_numpy(dtype=float), 1e-300, 1.0)
    )
    score_cols = ["Z_T", "neglog10_q_T", "coverage_C", "peaks_K", "prev", "qc_risk"]
    hm = hm.sort_values(by=["marker_group", "gene"], kind="mergesort").reset_index(
        drop=True
    )
    mat_raw = hm[score_cols].to_numpy(dtype=float)
    col_means = np.nanmean(mat_raw, axis=0)
    col_stds = np.nanstd(mat_raw, axis=0)
    col_stds = np.where(col_stds > 1e-12, col_stds, 1.0)
    mat = (mat_raw - col_means) / col_stds

    fig4, ax4 = plt.subplots(figsize=(8.4, 6.8))
    im = ax4.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax4.set_yticks(np.arange(hm.shape[0]))
    ax4.set_yticklabels(hm["gene"].tolist(), fontsize=8)
    ax4.set_xticks(np.arange(len(score_cols)))
    ax4.set_xticklabels(
        ["Z_T", "-log10(q_T)", "coverage_C", "peaks_K", "prev", "qc_risk"],
        rotation=25,
        ha="right",
    )
    ax4.set_title("Marker score heatmap (column-wise z-scored for display)")
    prev_group = None
    for i, grp in enumerate(hm["marker_group"].tolist()):
        if prev_group is None:
            prev_group = grp
            continue
        if grp != prev_group:
            ax4.axhline(i - 0.5, color="black", linewidth=1.0)
            prev_group = grp
    cbar = fig4.colorbar(im, ax=ax4, fraction=0.046, pad=0.03)
    cbar.set_label("z-score")
    fig4.tight_layout()
    fig4.savefig(
        out_dir / "heatmap_scores.png", dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight"
    )
    plt.close(fig4)


def _plot_controls(
    *,
    metrics_df: pd.DataFrame,
    donor_diag_df: pd.DataFrame | None,
    out_dir: Path,
    readme_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) QC mimicry plot.
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    colors = [CLASS_COLORS.get(str(c), "#333333") for c in metrics_df["class_label"]]
    ax.scatter(
        metrics_df["qc_risk"].to_numpy(dtype=float),
        metrics_df["Z_T"].to_numpy(dtype=float),
        c=colors,
        s=80,
        alpha=0.88,
        edgecolors="black",
        linewidths=0.6,
    )
    ax.axvline(
        QC_THRESH, color="#8B0000", linestyle="--", linewidth=1.2, label="qc_risk=0.35"
    )
    ax.axhline(Z_STRONG, color="#404040", linestyle="-.", linewidth=1.2, label="Z_T=4")
    qc_driven = metrics_df.loc[metrics_df["qc_driven"].astype(bool)]
    for _, row in qc_driven.iterrows():
        ax.text(
            float(row["qc_risk"]) + 0.01,
            float(row["Z_T"]) + 0.05,
            str(row["gene"]),
            fontsize=8,
        )
    ax.set_xlabel("qc_risk = max |Spearman rho|")
    ax.set_ylabel("Z_T")
    ax.set_title("QC mimicry diagnostic")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "qc_mimicry_qc_risk_vs_zt.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)

    # 2) Donor directionality diagnostic.
    if donor_diag_df is None or donor_diag_df.empty:
        _save_placeholder_plot(
            out_dir / "donor_directionality_diagnostic.png",
            "Donor directionality diagnostic",
            "Skipped: donor key unavailable.",
        )
        return

    fig2, ax2 = plt.subplots(figsize=(7.8, 6.0))
    ax2.scatter(
        metrics_df["score_1"].to_numpy(dtype=float),
        metrics_df["score_2"].to_numpy(dtype=float),
        s=60,
        c="#B7B7B7",
        edgecolors="white",
        linewidths=0.8,
        alpha=0.7,
        label="marker genes",
    )
    ax2.scatter(
        donor_diag_df["Z_T"].to_numpy(dtype=float),
        donor_diag_df["coverage_C"].to_numpy(dtype=float),
        s=110,
        c="#D62728",
        marker="X",
        edgecolors="black",
        linewidths=0.8,
        alpha=0.95,
        label="donor vs all",
    )
    for _, row in donor_diag_df.iterrows():
        ax2.text(
            float(row["Z_T"]) + 0.05,
            float(row["coverage_C"]) + 0.004,
            str(row["donor_id"]),
            fontsize=7,
        )
    ax2.axvline(Z_STRONG, color="black", linestyle="--", linewidth=1.2)
    ax2.axhline(COVERAGE_STRONG, color="black", linestyle="-.", linewidth=1.2)
    ax2.set_xlabel("score_1 = Z_T")
    ax2.set_ylabel("score_2 = coverage_C")
    ax2.set_title("Donor-directionality diagnostic in marker score space")
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.legend(loc="best", fontsize=8, frameon=True)
    fig2.tight_layout()
    fig2.savefig(
        out_dir / "donor_directionality_diagnostic.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig2)

    strong_donor = donor_diag_df.loc[
        (donor_diag_df["q_T"] <= Q_SIG) & (donor_diag_df["Z_T"] >= Z_STRONG)
    ]
    if not strong_donor.empty:
        lines = [
            "WARNING: donor-directionality diagnostic indicates strong donor directional structure.",
            "",
            "Criterion: donor q_T <= 0.05 and donor Z_T >= 4.0",
            "",
            "Affected donors:",
        ]
        for _, row in strong_donor.sort_values(by="Z_T", ascending=False).iterrows():
            lines.append(
                f"- donor_id={row['donor_id']}, Z_T={float(row['Z_T']):.2f}, "
                f"coverage_C={float(row['coverage_C']):.3f}, q_T={float(row['q_T']):.2e}"
            )
        readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_donor_diagnostic(
    *,
    donor_ids: np.ndarray | None,
    theta: np.ndarray,
    n_bins: int,
    n_perm: int,
    seed: int,
    bin_id: np.ndarray,
    bin_counts_total: np.ndarray,
) -> pd.DataFrame | None:
    if donor_ids is None:
        return None

    donor_arr = np.asarray(donor_ids).astype(str)
    uniq = sorted(pd.Index(donor_arr).unique().tolist())
    rows: list[dict[str, Any]] = []
    for i, donor in enumerate(uniq):
        foreground = donor_arr == donor
        expr = foreground.astype(float)
        perm = perm_null_T_and_profile(
            expr=expr,
            theta=theta,
            donor_ids=None,
            n_bins=int(n_bins),
            n_perm=int(n_perm),
            seed=int(seed + 50_000 + i),
            donor_stratified=False,
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
        null_e = np.asarray(perm["null_E_phi"], dtype=float)
        null_t = np.asarray(perm["null_T"], dtype=float)
        rows.append(
            {
                "donor_id": donor,
                "n_cells": int(foreground.sum()),
                "prev": float(np.mean(foreground)),
                "T_obs": float(perm["T_obs"]),
                "p_T": float(perm["p_T"]),
                "Z_T": float(robust_z(float(perm["T_obs"]), null_t)),
                "coverage_C": float(
                    coverage_from_null(np.abs(e_obs), np.abs(null_e), q=0.95)
                ),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["q_T"] = bh_fdr(df["p_T"].to_numpy(dtype=float))
    return df


def _compute_marker_scores(
    *,
    resolutions: list[MarkerResolution],
    expr_matrix: Any,
    umap_xy: np.ndarray,
    theta: np.ndarray,
    bin_id: np.ndarray,
    bin_counts_total: np.ndarray,
    donor_ids: np.ndarray | None,
    donor_key_used: str | None,
    total_counts: np.ndarray,
    pct_counts_mt: np.ndarray | None,
    pct_counts_ribo: np.ndarray | None,
    n_bins: int,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    n_cells = int(umap_xy.shape[0])
    min_fg_required = max(
        UNDERPOWERED_MIN_CELLS, int(np.ceil(UNDERPOWERED_MIN_FRAC * n_cells))
    )
    centers = theta_bin_centers(int(n_bins))

    scored_counter = 0
    for res in resolutions:
        if not res.found or res.gene_idx is None:
            continue
        scored_counter += 1
        expr = get_feature_vector(expr_matrix, int(res.gene_idx))
        f = np.asarray(expr, dtype=float) > 0.0
        prev = float(np.mean(f))
        n_fg = int(f.sum())

        if 0 < n_fg < n_cells:
            e_obs_direct, _, _, _ = compute_rsp_profile_from_boolean(
                f,
                theta,
                int(n_bins),
                bin_id=bin_id,
                bin_counts_total=bin_counts_total,
            )
        else:
            e_obs_direct = np.zeros(int(n_bins), dtype=float)

        perm = perm_null_T_and_profile(
            expr=expr,
            theta=theta,
            donor_ids=donor_ids,
            n_bins=int(n_bins),
            n_perm=int(n_perm),
            seed=int(seed + 1_000 + scored_counter),
            donor_stratified=True,
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )

        e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
        if e_obs.size != int(n_bins):
            e_obs = np.asarray(e_obs_direct, dtype=float)
        null_e = np.asarray(perm["null_E_phi"], dtype=float)
        null_t = np.asarray(perm["null_T"], dtype=float)
        t_obs = float(perm["T_obs"])

        coverage_c = float(coverage_from_null(np.abs(e_obs), np.abs(null_e), q=0.95))
        peaks_k = int(
            peak_count(np.abs(e_obs), np.abs(null_e), smooth_w=3, q_prom=0.95)
        )

        peak_idx = int(np.argmax(np.abs(e_obs))) if e_obs.size > 0 else 0
        peak_phi = float(centers[peak_idx]) if centers.size > 0 else 0.0
        peak_phi_deg = float(np.degrees(peak_phi) % 360.0)

        rho_total = _safe_spearman_binary(f.astype(float), total_counts)
        rho_mt = _safe_spearman_binary(f.astype(float), pct_counts_mt)
        rho_ribo = _safe_spearman_binary(f.astype(float), pct_counts_ribo)
        finite_rhos = [abs(v) for v in [rho_total, rho_mt, rho_ribo] if np.isfinite(v)]
        qc_risk = float(max(finite_rhos)) if finite_rhos else 0.0

        row = {
            "gene": res.gene,
            "resolved_gene": res.resolved_gene,
            "marker_group": res.marker_group,
            "prev": prev,
            "n_fg": n_fg,
            "n_cells": n_cells,
            "T_obs": t_obs,
            "p_T": float(perm["p_T"]),
            "q_T": float("nan"),
            "Z_T": float(robust_z(t_obs, null_t)),
            "coverage_C": coverage_c,
            "peaks_K": peaks_k,
            "peak_dir_phi": peak_phi,
            "peak_dir_phi_rad": peak_phi,
            "peak_dir_phi_deg": peak_phi_deg,
            "rho_total_counts": rho_total,
            "rho_pct_counts_mt": rho_mt,
            "rho_pct_counts_ribo": rho_ribo,
            "qc_risk": qc_risk,
            "qc_driven": False,
            "used_donor_stratified": bool(perm["used_donor_stratified"]),
            "donor_key_used": donor_key_used or "",
            "underpowered": bool(n_fg < min_fg_required),
            "score_1": float(robust_z(t_obs, null_t)),
            "score_2": coverage_c,
            "class_label": "",
            "perm_warning": str(perm.get("warning", "")),
        }
        rows.append(row)
        artifacts[res.gene] = {
            "expr": np.asarray(expr, dtype=float),
            "E_phi_obs": e_obs,
            "null_E_phi": null_e,
            "null_T": null_t,
        }

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df, artifacts

    metrics_df["q_T"] = bh_fdr(metrics_df["p_T"].to_numpy(dtype=float))
    metrics_df["qc_driven"] = (metrics_df["qc_risk"] >= QC_THRESH) & (
        metrics_df["q_T"] <= Q_SIG
    )
    metrics_df["class_label"] = metrics_df.apply(_classify_row, axis=1)
    metrics_df = metrics_df.sort_values(
        by=["marker_group", "q_T", "gene"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return metrics_df, artifacts


def _plot_overview(
    *,
    adata: ad.AnnData,
    umap_xy: np.ndarray,
    center_xy: np.ndarray,
    label_key: str | None,
    donor_key: str | None,
    total_counts: np.ndarray,
    pct_counts_mt: np.ndarray | None,
    pct_counts_ribo: np.ndarray | None,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if label_key is not None:
        plot_categorical_umap(
            umap_xy=umap_xy,
            labels=adata.obs[label_key],
            title=f"UMAP colored by {label_key}",
            outpath=out_dir / "umap_labels.png",
            vantage_point=(float(center_xy[0]), float(center_xy[1])),
            annotate_cluster_medians=False,
        )
    else:
        _save_placeholder_plot(
            out_dir / "umap_labels.png",
            "UMAP labels",
            "No label key available (tried: azimuth_label, predicted_label).",
        )

    if donor_key is not None and donor_key in adata.obs.columns:
        plot_categorical_umap(
            umap_xy=umap_xy,
            labels=adata.obs[donor_key],
            title=f"UMAP colored by donor ({donor_key})",
            outpath=out_dir / "umap_donor.png",
            vantage_point=(float(center_xy[0]), float(center_xy[1])),
            annotate_cluster_medians=False,
        )
    else:
        _save_placeholder_plot(
            out_dir / "umap_donor.png",
            "UMAP donor",
            "No donor key available; donor mixing plot skipped.",
        )

    save_numeric_umap(
        umap_xy=umap_xy,
        values=np.log1p(np.maximum(total_counts, 0.0)),
        out_png=out_dir / "umap_qc_total_counts.png",
        title="UMAP: log1p(total_counts)",
        cmap="viridis",
        colorbar_label="log1p(total_counts)",
        vantage_point=(float(center_xy[0]), float(center_xy[1])),
    )

    if pct_counts_mt is not None:
        save_numeric_umap(
            umap_xy=umap_xy,
            values=np.asarray(pct_counts_mt, dtype=float),
            out_png=out_dir / "umap_qc_pct_mt.png",
            title="UMAP: pct_counts_mt",
            cmap="magma",
            colorbar_label="pct_counts_mt",
            vantage_point=(float(center_xy[0]), float(center_xy[1])),
        )
    else:
        _save_placeholder_plot(
            out_dir / "umap_qc_pct_mt.png",
            "UMAP pct_counts_mt",
            "pct_counts_mt unavailable in this dataset.",
        )

    if pct_counts_ribo is not None:
        save_numeric_umap(
            umap_xy=umap_xy,
            values=np.asarray(pct_counts_ribo, dtype=float),
            out_png=out_dir / "umap_qc_pct_ribo.png",
            title="UMAP: pct_counts_ribo",
            cmap="plasma",
            colorbar_label="pct_counts_ribo",
            vantage_point=(float(center_xy[0]), float(center_xy[1])),
        )
    else:
        _save_placeholder_plot(
            out_dir / "umap_qc_pct_ribo.png",
            "UMAP pct_counts_ribo",
            "pct_counts_ribo unavailable in this dataset.",
        )


def main() -> int:
    args = parse_args()
    apply_plot_style()

    h5ad_path = Path(args.h5ad)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Input h5ad not found: {h5ad_path}")

    outdir = Path(args.out)
    tables_dir = outdir / "tables"
    plots_dir = outdir / "plots"
    overview_dir = plots_dir / "00_overview"
    per_gene_dir = plots_dir / "01_gene_panels"
    score_dir = plots_dir / "02_score_space"
    controls_dir = plots_dir / "03_controls"
    for d in [tables_dir, overview_dir, per_gene_dir, score_dir, controls_dir]:
        d.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(h5ad_path)
    _ensure_umap(adata, seed=int(args.seed), recompute_if_missing=False)
    embedding_key, umap_xy = _resolve_embedding(adata, args.embedding_key)
    center_xy = compute_vantage_point(umap_xy, method="median")
    theta = compute_theta(umap_xy, center_xy)
    _, bin_id = bin_theta(theta, int(args.n_bins))
    bin_counts_total = np.bincount(bin_id, minlength=int(args.n_bins)).astype(float)

    expr_matrix, adata_like, expr_source = _choose_expression_source(
        adata,
        layer_arg=args.layer,
        use_raw_arg=bool(args.use_raw),
    )

    donor_ids, donor_key_used = _resolve_donor_ids(adata, args.donor_key)
    if donor_ids is None:
        print(
            "WARNING: donor key unavailable (or <2 unique donors). "
            "Falling back to global permutation (used_donor_stratified=False)."
        )

    label_key_used = _resolve_label_key(adata, args.label_key)
    if label_key_used is None:
        print(
            "WARNING: no label key found for UMAP label overview (azimuth_label/predicted_label)."
        )

    total_counts = _total_counts_vector(adata, expr_matrix)
    pct_mt, pct_mt_source = _pct_mt_vector(adata, expr_matrix, adata_like)
    if pct_mt_source == "proxy:zeros":
        pct_mt_vec: np.ndarray | None = None
    else:
        pct_mt_vec = np.asarray(pct_mt, dtype=float)
    pct_ribo_vec, pct_ribo_source = _compute_pct_counts_ribo(
        adata, expr_matrix, adata_like, total_counts
    )

    print(f"embedding_key_used={embedding_key}")
    print(f"donor_key_used={donor_key_used if donor_key_used is not None else 'None'}")
    print(f"expression_source_used={expr_source}")
    print(f"pct_counts_mt_source={pct_mt_source}")
    print(f"pct_counts_ribo_source={pct_ribo_source}")
    print(
        f"n_cells={int(adata.n_obs)} n_bins={int(args.n_bins)} n_perm={int(args.n_perm)} seed={int(args.seed)}"
    )

    resolutions, marker_panel_df = _resolve_marker_panel(adata_like)
    marker_panel_csv = tables_dir / "marker_panel.csv"
    marker_panel_df.to_csv(marker_panel_csv, index=False)

    missing_markers = (
        marker_panel_df.loc[~marker_panel_df["found"].astype(bool), "gene"]
        .astype(str)
        .tolist()
    )
    if missing_markers:
        print(f"missing_markers={','.join(missing_markers)}")

    metrics_df, artifacts = _compute_marker_scores(
        resolutions=resolutions,
        expr_matrix=expr_matrix,
        umap_xy=umap_xy,
        theta=theta,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
        donor_ids=donor_ids,
        donor_key_used=donor_key_used,
        total_counts=np.asarray(total_counts, dtype=float),
        pct_counts_mt=pct_mt_vec,
        pct_counts_ribo=pct_ribo_vec,
        n_bins=int(args.n_bins),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
    )

    metrics_csv = tables_dir / "marker_scores_full_embedding.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    _plot_overview(
        adata=adata,
        umap_xy=umap_xy,
        center_xy=center_xy,
        label_key=label_key_used,
        donor_key=donor_key_used,
        total_counts=np.asarray(total_counts, dtype=float),
        pct_counts_mt=pct_mt_vec,
        pct_counts_ribo=pct_ribo_vec,
        out_dir=overview_dir,
    )

    for _, row in metrics_df.iterrows():
        gene = str(row["gene"])
        if gene not in artifacts:
            continue
        gene_out = per_gene_dir / f"gene_{gene}.png"
        art = artifacts[gene]
        _plot_gene_panel(
            out_png=gene_out,
            gene=gene,
            marker_group=str(row["marker_group"]),
            expr=np.asarray(art["expr"], dtype=float),
            umap_xy=umap_xy,
            center_xy=np.asarray(center_xy, dtype=float),
            e_phi_obs=np.asarray(art["E_phi_obs"], dtype=float),
            null_e_phi=np.asarray(art["null_E_phi"], dtype=float),
            null_t=np.asarray(art["null_T"], dtype=float),
            n_bins=int(args.n_bins),
            row=row,
        )

    _plot_score_space(metrics_df, score_dir)

    donor_diag_df = _compute_donor_diagnostic(
        donor_ids=donor_ids,
        theta=theta,
        n_bins=int(args.n_bins),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )
    if donor_diag_df is not None:
        donor_diag_df.to_csv(
            tables_dir / "donor_directionality_scores.csv", index=False
        )

    _plot_controls(
        metrics_df=metrics_df,
        donor_diag_df=donor_diag_df,
        out_dir=controls_dir,
        readme_path=outdir / "README.txt",
    )

    class_counts = (
        metrics_df["class_label"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
        if not metrics_df.empty
        else {k: 0 for k in CLASS_ORDER}
    )
    class_summary = "; ".join([f"{k}={v}" for k, v in class_counts.items()])
    print(f"classification_summary: {class_summary}")
    print(f"marker_panel_csv={marker_panel_csv.as_posix()}")
    print(f"marker_scores_csv={metrics_csv.as_posix()}")
    print(f"results_root={outdir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
