#!/usr/bin/env python3
"""Experiment #4: donor-reproducible geometry as default evidence standard."""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

# Force non-interactive backend for deterministic batch plotting.
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
from biorsp.plotting.qc import plot_categorical_umap
from biorsp.plotting.styles import DEFAULT_PLOT_STYLE, apply_plot_style
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

CANONICAL_PANEL = {
    "Cardiomyocyte": ["MYH6", "TNNT2", "RYR2", "PLN"],
    "Fibroblast/ECM": ["COL1A1", "COL1A2", "LUM", "DCN"],
    "Endothelial": ["PECAM1", "VWF", "KDR"],
    "Mural": ["ACTA2", "TAGLN", "RGS5"],
    "Immune": ["PTPRC", "LST1", "LYZ"],
}

HOUSEKEEPING_CONTROLS = ["ACTB", "GAPDH", "RPLP0"]

# "Small candidate set" for automatic challenger selection.
CHALLENGER_CANDIDATE_SET = [
    "MYH7",
    "TTN",
    "ACTC1",
    "COL3A1",
    "FBLN5",
    "PDGFRA",
    "POSTN",
    "CXCL12",
    "S100A8",
    "S100A9",
    "IL1B",
    "FCGR3A",
    "AIF1",
    "VCAN",
    "C1QA",
    "C1QC",
    "CLDN5",
    "EFNB2",
    "EPHB4",
    "NR2F2",
    "GJA5",
    "SOX17",
]

CLASS_ORDER = [
    "Donor-replicated localized program",
    "Donor-unstable",
    "QC-driven",
    "Underpowered",
]

CLASS_COLORS = {
    "Donor-replicated localized program": "#1F77B4",
    "Donor-unstable": "#FF7F0E",
    "QC-driven": "#D62728",
    "Underpowered": "#8C8C8C",
}

QC_THRESH = 0.35
Q_SIG = 0.05
P_MIN = 0.005
MIN_FG = 50
MIN_CELLS_D = 500
DONOR_Z_MIN = 2.0
HIGH_PREV = 0.60


@dataclass(frozen=True)
class ResolvedGene:
    gene: str
    panel_role: str
    marker_group: str
    found: bool
    gene_idx: int | None
    resolved_gene: str
    status: str
    resolution_source: str
    symbol_column: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment #4: donor-level reproducibility benchmark for BioRSP geometry."
    )
    p.add_argument(
        "--h5ad", default="data/processed/HT_pca_umap.h5ad", help="Input .h5ad path."
    )
    p.add_argument(
        "--out",
        default="experiments/heart_smoketest/results/experiment4_donor_reproducibility",
        help="Output directory.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument(
        "--n_perm",
        type=int,
        default=300,
        help="Permutation count for global/donor tests.",
    )
    p.add_argument("--n_bins", type=int, default=64, help="Angular bins.")
    p.add_argument(
        "--bootstrap", type=int, default=200, help="Donor bootstrap iterations for CI."
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
    fig, ax = plt.subplots(figsize=(6.2, 4.7))
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
            raise KeyError(f"Requested embedding key '{requested_key}' not found.")
        key = str(requested_key)
    else:
        key = "X_umap" if "X_umap" in adata.obsm else str(next(iter(adata.obsm.keys())))
    xy = np.asarray(adata.obsm[key], dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Embedding '{key}' must be shape (N,2+), got {xy.shape}.")
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


def _resolve_donor_ids_required(
    adata: ad.AnnData,
    requested_key: str | None,
) -> tuple[np.ndarray, str]:
    donor_key = _resolve_key(adata, requested_key, DONOR_CANDIDATES)
    if donor_key is None:
        raise RuntimeError(
            "Experiment #4 requires donor IDs. No donor key found in obs candidates."
        )
    donor_ids = (
        adata.obs[donor_key].astype("string").fillna("NA").astype(str).to_numpy()
    )
    n_donor = int(np.unique(donor_ids).size)
    if n_donor < 2:
        raise RuntimeError(
            f"Experiment #4 requires >=2 donors, got {n_donor} from key '{donor_key}'."
        )
    return donor_ids, donor_key


def _resolve_label_key(adata: ad.AnnData, requested_key: str | None) -> str | None:
    return _resolve_key(adata, requested_key, LABEL_KEY_CANDIDATES)


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


def _circular_stats(phi: np.ndarray) -> tuple[float, float, float]:
    ang = np.asarray(phi, dtype=float).ravel()
    if ang.size == 0:
        return float("nan"), float("nan"), float("nan")
    z = np.exp(1j * ang)
    m = np.mean(z)
    mu = float(np.angle(m) % (2.0 * np.pi))
    R = float(np.abs(m))
    R_clip = float(max(R, 1e-12))
    circ_sd = float(np.sqrt(max(0.0, -2.0 * np.log(R_clip))))
    return mu, R, circ_sd


def _circular_diff(phi: np.ndarray, mu: float) -> np.ndarray:
    ang = np.asarray(phi, dtype=float)
    return np.angle(np.exp(1j * (ang - float(mu))))


def _bootstrap_median_ci(
    values: np.ndarray, *, n_boot: int, seed: int
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    boots = np.zeros(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        sample = x[rng.integers(0, x.size, size=x.size)]
        boots[i] = float(np.median(sample))
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _resolve_gene(
    adata_like: Any,
    gene: str,
) -> tuple[bool, int | None, str, str, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            idx, label, symbol_col, source = resolve_feature_index(adata_like, gene)
        return True, int(idx), str(label), str(source), str(symbol_col or "")
    except KeyError:
        return False, None, "", "", ""


def _compute_global_z_for_gene(
    *,
    expr: np.ndarray,
    theta: np.ndarray,
    donor_ids: np.ndarray,
    n_bins: int,
    n_perm: int,
    seed: int,
    bin_id: np.ndarray,
    bin_counts_total: np.ndarray,
) -> float:
    perm = perm_null_T_and_profile(
        expr=np.asarray(expr, dtype=float),
        theta=np.asarray(theta, dtype=float),
        donor_ids=np.asarray(donor_ids).astype(str),
        n_bins=int(n_bins),
        n_perm=int(n_perm),
        seed=int(seed),
        donor_stratified=True,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )
    return float(
        robust_z(float(perm["T_obs"]), np.asarray(perm["null_T"], dtype=float))
    )


def _build_gene_panel(
    *,
    adata_like: Any,
    expr_matrix: Any,
    theta: np.ndarray,
    donor_ids: np.ndarray,
    n_bins: int,
    n_perm: int,
    seed: int,
    bin_id: np.ndarray,
    bin_counts_total: np.ndarray,
) -> tuple[list[ResolvedGene], pd.DataFrame]:
    canonical_rows: list[ResolvedGene] = []
    panel_rows: list[dict[str, Any]] = []

    canonical_set = [g for genes in CANONICAL_PANEL.values() for g in genes]
    used_gene_names: set[str] = set()

    for group, genes in CANONICAL_PANEL.items():
        for gene in genes:
            found, idx, resolved, source, symbol_col = _resolve_gene(adata_like, gene)
            status = "resolved" if found else "missing"
            row = ResolvedGene(
                gene=gene,
                panel_role="canonical",
                marker_group=group,
                found=found,
                gene_idx=idx,
                resolved_gene=resolved,
                status=status,
                resolution_source=source,
                symbol_column=symbol_col,
            )
            canonical_rows.append(row)
            panel_rows.append(
                {
                    "gene": gene,
                    "panel_role": "canonical",
                    "marker_group": group,
                    "status": status,
                    "found": found,
                    "resolved_gene": resolved,
                    "gene_idx": idx if idx is not None else "",
                    "resolution_source": source,
                    "symbol_column": symbol_col,
                }
            )
            if found:
                used_gene_names.add(gene)

    # Compute global Z for challenger candidates and pick top 3 not already in canonical.
    candidate_pool = canonical_set + CHALLENGER_CANDIDATE_SET
    candidate_unique = []
    seen: set[str] = set()
    for g in candidate_pool:
        if g in seen:
            continue
        seen.add(g)
        candidate_unique.append(g)

    candidate_scores: list[tuple[str, float]] = []
    for i, gene in enumerate(candidate_unique):
        found, idx, _, _, _ = _resolve_gene(adata_like, gene)
        if not found or idx is None:
            continue
        expr = get_feature_vector(expr_matrix, int(idx))
        try:
            z = _compute_global_z_for_gene(
                expr=np.asarray(expr, dtype=float),
                theta=theta,
                donor_ids=donor_ids,
                n_bins=int(n_bins),
                n_perm=int(n_perm),
                seed=int(seed + 200_000 + i * 19),
                bin_id=bin_id,
                bin_counts_total=bin_counts_total,
            )
            candidate_scores.append((gene, z))
        except Exception:
            continue

    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    challengers: list[str] = []
    for gene, _ in candidate_scores:
        if gene in canonical_set:
            continue
        if gene in HOUSEKEEPING_CONTROLS:
            continue
        challengers.append(gene)
        if len(challengers) >= 3:
            break

    for gene in challengers:
        found, idx, resolved, source, symbol_col = _resolve_gene(adata_like, gene)
        status = "resolved" if found else "missing"
        row = ResolvedGene(
            gene=gene,
            panel_role="challenger",
            marker_group="challenger",
            found=found,
            gene_idx=idx,
            resolved_gene=resolved,
            status=status,
            resolution_source=source,
            symbol_column=symbol_col,
        )
        canonical_rows.append(row)
        panel_rows.append(
            {
                "gene": gene,
                "panel_role": "challenger",
                "marker_group": "challenger",
                "status": status,
                "found": found,
                "resolved_gene": resolved,
                "gene_idx": idx if idx is not None else "",
                "resolution_source": source,
                "symbol_column": symbol_col,
            }
        )
        if found:
            used_gene_names.add(gene)

    # Add housekeeping controls.
    for gene in HOUSEKEEPING_CONTROLS:
        found, idx, resolved, source, symbol_col = _resolve_gene(adata_like, gene)
        status = "resolved" if found else "missing"
        row = ResolvedGene(
            gene=gene,
            panel_role="housekeeping",
            marker_group="housekeeping",
            found=found,
            gene_idx=idx,
            resolved_gene=resolved,
            status=status,
            resolution_source=source,
            symbol_column=symbol_col,
        )
        canonical_rows.append(row)
        panel_rows.append(
            {
                "gene": gene,
                "panel_role": "housekeeping",
                "marker_group": "housekeeping",
                "status": status,
                "found": found,
                "resolved_gene": resolved,
                "gene_idx": idx if idx is not None else "",
                "resolution_source": source,
                "symbol_column": symbol_col,
            }
        )

    return canonical_rows, pd.DataFrame(panel_rows)


def _plot_overview(
    *,
    adata: ad.AnnData,
    umap_xy: np.ndarray,
    donor_key: str,
    label_key: str | None,
    center_xy: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) UMAP colored by donor.
    plot_categorical_umap(
        umap_xy=umap_xy,
        labels=adata.obs[donor_key],
        title=f"All cells UMAP by donor ({donor_key})",
        outpath=out_dir / "umap_by_donor.png",
        vantage_point=(float(center_xy[0]), float(center_xy[1])),
        annotate_cluster_medians=False,
    )

    # 2) cell counts per donor.
    donor_counts = (
        adata.obs[donor_key]
        .astype("string")
        .fillna("NA")
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
    )
    fig1, ax1 = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(donor_counts.shape[0], dtype=float)
    ax1.bar(
        x,
        donor_counts.to_numpy(dtype=float),
        color="#5DA5DA",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        donor_counts.index.tolist(), rotation=35, ha="right", fontsize=8
    )
    ax1.set_ylabel("# cells")
    ax1.set_title("Cells per donor")
    ax1.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig1.tight_layout()
    fig1.savefig(out_dir / "cells_per_donor.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    # 3) stacked label composition by donor.
    if label_key is not None and label_key in adata.obs.columns:
        comp = pd.crosstab(
            adata.obs[donor_key].astype("string").fillna("NA").astype(str),
            adata.obs[label_key].astype("string").fillna("NA").astype(str),
            dropna=False,
        )
        comp = comp.loc[donor_counts.index, :]
        frac = comp.div(comp.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        fig2, ax2 = plt.subplots(figsize=(9.8, 5.2))
        bottoms = np.zeros(frac.shape[0], dtype=float)
        cmap = plt.get_cmap("tab20")
        for i, col in enumerate(frac.columns.tolist()):
            vals = frac[col].to_numpy(dtype=float)
            ax2.bar(
                np.arange(frac.shape[0], dtype=float),
                vals,
                bottom=bottoms,
                color=cmap(i % 20),
                edgecolor="white",
                linewidth=0.2,
                label=str(col),
            )
            bottoms += vals
        ax2.set_xticks(np.arange(frac.shape[0], dtype=float))
        ax2.set_xticklabels(frac.index.tolist(), rotation=35, ha="right", fontsize=8)
        ax2.set_ylabel("Cell-type fraction")
        ax2.set_title(f"Cell-type composition by donor ({label_key})")
        ax2.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=True
        )
        fig2.tight_layout()
        fig2.savefig(
            out_dir / "label_composition_by_donor.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
            bbox_inches="tight",
        )
        plt.close(fig2)
    else:
        _save_placeholder(
            out_dir / "label_composition_by_donor.png",
            "Cell-type composition by donor",
            "Cell-type label key unavailable.",
        )


def _plot_gene_donor_panel(
    *,
    gene: str,
    global_row: pd.Series,
    global_art: dict[str, np.ndarray],
    donor_gene_df: pd.DataFrame,
    umap_xy: np.ndarray,
    center_xy: np.ndarray,
    n_bins: int,
    out_png: Path,
) -> None:
    fig = plt.figure(figsize=(15.5, 9.5))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2, projection="polar")
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4, projection="polar")
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    expr = np.asarray(global_art["expr"], dtype=float)
    e_obs = np.asarray(global_art["E_phi_obs"], dtype=float)
    null_e = np.asarray(global_art["null_E_phi"], dtype=float)

    # 1) global UMAP feature.
    log_expr = np.log1p(np.maximum(expr, 0.0))
    order = np.argsort(log_expr, kind="mergesort")
    ax1.scatter(
        umap_xy[:, 0],
        umap_xy[:, 1],
        c="#D3D3D3",
        s=4.0,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    sc = ax1.scatter(
        umap_xy[order, 0],
        umap_xy[order, 1],
        c=log_expr[order],
        cmap="Reds",
        s=7.0,
        alpha=0.88,
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
    ax1.set_title(f"{gene}: global feature map")
    cb = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label("log1p(expr)")

    # 2) global RSP polar with null envelope.
    centers = theta_bin_centers(int(n_bins))
    theta_c = np.concatenate([centers, centers[:1]])
    obs_c = np.concatenate([e_obs, e_obs[:1]])
    q95 = np.quantile(null_e, 0.95, axis=0)
    q05 = np.quantile(null_e, 0.05, axis=0)
    q95_c = np.concatenate([q95, q95[:1]])
    q05_c = np.concatenate([q05, q05[:1]])
    ax2.plot(theta_c, obs_c, color="#8B0000", linewidth=2.0, label="E_phi global")
    ax2.plot(
        theta_c, q95_c, color="#444444", linestyle="--", linewidth=1.2, label="null 95%"
    )
    ax2.plot(
        theta_c, q05_c, color="#444444", linestyle="--", linewidth=1.0, label="null 5%"
    )
    ax2.fill_between(theta_c, q05_c, q95_c, color="#B0B0B0", alpha=0.18)
    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(1)
    ax2.set_thetagrids(np.arange(0, 360, 90))
    ax2.set_title("Global RSP + donor-strat null")
    ann = (
        f"Z_global={float(global_row['Z_T_global']):.2f}\n"
        f"q_global={float(global_row['q_T_global']):.2e}\n"
        f"C_global={float(global_row['coverage_C_global']):.3f}\n"
        f"K_global={int(global_row['peaks_K_global'])}\n"
        f"phi_global={float(global_row['phi_global_deg']):.1f}°"
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
    ax2.legend(loc="upper right", bbox_to_anchor=(1.2, 1.2), fontsize=8, frameon=True)

    # 3) donor effect Z plot.
    ddf = donor_gene_df.sort_values(
        by="Z_T_d", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    x = np.arange(ddf.shape[0], dtype=float)
    colors = np.where(ddf["underpowered_d"].to_numpy(dtype=bool), "#B0B0B0", "#1F77B4")
    ax3.scatter(
        x,
        ddf["Z_T_d"].to_numpy(dtype=float),
        c=colors,
        s=70,
        edgecolors="black",
        linewidths=0.5,
    )
    ax3.axhline(DONOR_Z_MIN, color="#333333", linestyle="--", linewidth=1.0)
    ax3.set_xticks(x)
    ax3.set_xticklabels(ddf["donor_id"].tolist(), rotation=40, ha="right", fontsize=7)
    ax3.set_ylabel("Z_T_d")
    ax3.set_title("Donor-level effect sizes")
    ax3.grid(axis="y", alpha=0.25, linewidth=0.6)

    # 4) donor direction circular plot.
    phi = ddf["phi_d_rad"].to_numpy(dtype=float)
    under = ddf["underpowered_d"].to_numpy(dtype=bool)
    ax4.scatter(
        phi[~under],
        np.ones(np.sum(~under)),
        c="#1F77B4",
        s=70,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
        label="eligible",
    )
    if np.any(under):
        ax4.scatter(
            phi[under],
            np.full(np.sum(under), 0.8),
            c="#B0B0B0",
            s=70,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
            label="underpowered",
        )
    mu = float(global_row["mu_phi"])
    R = float(global_row["R"])
    if np.isfinite(mu):
        ax4.plot([mu, mu], [0.0, 1.15], color="#D62728", linewidth=2.0, label="mu_phi")
    ax4.set_theta_zero_location("E")
    ax4.set_theta_direction(1)
    ax4.set_thetagrids(np.arange(0, 360, 90))
    ax4.set_rticks([])
    ax4.set_title(
        f"Donor phi_d (R={R:.2f}, circ_sd={float(global_row['circ_sd']):.2f})"
    )
    ax4.legend(loc="upper right", bbox_to_anchor=(1.25, 1.2), fontsize=8, frameon=True)

    # 5) donor T distribution vs global.
    ax5.boxplot(
        ddf["T_d"].to_numpy(dtype=float),
        vert=True,
        widths=0.45,
        patch_artist=True,
        boxprops={"facecolor": "#9ecae1"},
    )
    jitter = np.random.default_rng(0).uniform(-0.06, 0.06, size=ddf.shape[0])
    ax5.scatter(
        np.ones(ddf.shape[0]) + jitter,
        ddf["T_d"].to_numpy(dtype=float),
        c=colors,
        s=45,
        edgecolors="black",
        linewidths=0.4,
    )
    ax5.axhline(
        float(global_row["T_obs_global"]),
        color="#8B0000",
        linestyle="--",
        linewidth=1.7,
        label="T_global",
    )
    ax5.set_xticks([1.0])
    ax5.set_xticklabels(["donor T_d"])
    ax5.set_ylabel("T statistic")
    ax5.set_title("Donor T_d distribution")
    ax5.legend(loc="best", fontsize=8, frameon=True)
    ax5.grid(axis="y", alpha=0.25, linewidth=0.6)

    # 6) summary text block.
    ax6.axis("off")
    summary = (
        f"final_label: {global_row['final_repro_label']}\n"
        f"median_Z: {float(global_row['median_Z']):.2f}\n"
        f"donor_support: {float(global_row['donor_support']):.2f}\n"
        f"class_stability: {float(global_row['class_stability']):.2f}\n"
        f"qc_risk_global: {float(global_row['qc_risk_global']):.2f}\n"
        f"q_T_global: {float(global_row['q_T_global']):.2e}"
    )
    ax6.text(0.02, 0.95, summary, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle(f"{gene}: donor reproducibility panel", y=1.01, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_repro_summary(
    summary_df: pd.DataFrame, *, out_dir: Path, rank_csv: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        _save_placeholder(
            out_dir / "replication_map.png", "Replication map", "No data."
        )
        _save_placeholder(
            out_dir / "summary_heatmap.png", "Summary heatmap", "No data."
        )
        _save_placeholder(out_dir / "rank_table.png", "Rank table", "No data.")
        return

    # 1) replication map.
    fig1, ax1 = plt.subplots(figsize=(8.4, 6.2))
    for _, row in summary_df.iterrows():
        lbl = str(row["final_repro_label"])
        size = 80 + 260 * max(
            0.0, float(row["R"]) if np.isfinite(float(row["R"])) else 0.0
        )
        ax1.scatter(
            float(row["median_Z"]),
            float(row["donor_support"]),
            s=size,
            c=CLASS_COLORS.get(lbl, "#333333"),
            alpha=0.88,
            edgecolors="black",
            linewidths=0.7,
        )
    for _, row in summary_df.loc[summary_df["panel_role"] == "canonical"].iterrows():
        ax1.text(
            float(row["median_Z"]) + 0.05,
            float(row["donor_support"]) + 0.01,
            str(row["gene"]),
            fontsize=8,
        )
    ax1.set_xlabel("median_Z")
    ax1.set_ylabel("donor_support")
    ax1.set_title("Replication map")
    ax1.grid(alpha=0.25, linewidth=0.6)
    handles = [
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
    ax1.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        frameon=True,
    )
    fig1.tight_layout()
    fig1.savefig(
        out_dir / "replication_map.png", dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight"
    )
    plt.close(fig1)

    # 2) summary heatmap.
    cols = [
        "median_Z",
        "donor_support",
        "R",
        "circ_sd",
        "class_stability",
        "qc_risk_global",
        "coverage_C_global",
        "peaks_K_global",
    ]
    hm = summary_df.copy()
    hm = hm.sort_values(
        by=["final_repro_label", "median_Z"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)
    mat_raw = hm[cols].to_numpy(dtype=float)
    means = np.nanmean(mat_raw, axis=0)
    stds = np.nanstd(mat_raw, axis=0)
    stds = np.where(stds > 1e-12, stds, 1.0)
    mat = (mat_raw - means) / stds
    fig2, ax2 = plt.subplots(figsize=(8.4, 7.1))
    im = ax2.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-2.7, vmax=2.7)
    ax2.set_yticks(np.arange(hm.shape[0]))
    ax2.set_yticklabels(hm["gene"].tolist(), fontsize=8)
    ax2.set_xticks(np.arange(len(cols)))
    ax2.set_xticklabels(cols, rotation=28, ha="right")
    ax2.set_title("Gene reproducibility metrics heatmap (z-scored columns)")
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    fig2.tight_layout()
    fig2.savefig(
        out_dir / "summary_heatmap.png", dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight"
    )
    plt.close(fig2)

    # 3) rank table + CSV.
    rank = summary_df.copy()
    rank["qc_risk_clipped"] = np.clip(
        rank["qc_risk_global"].to_numpy(dtype=float), 0.0, 1.0
    )
    rank["replication_score"] = (
        rank["median_Z"].to_numpy(dtype=float)
        * rank["donor_support"].to_numpy(dtype=float)
        * rank["R"].to_numpy(dtype=float)
        * (1.0 - rank["qc_risk_clipped"].to_numpy(dtype=float))
    )
    rank = rank.sort_values(
        by="replication_score", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    rank.to_csv(rank_csv, index=False)

    top = rank.head(15).copy()
    fig3, ax3 = plt.subplots(figsize=(9.6, 0.52 * max(4, top.shape[0] + 3)))
    ax3.axis("off")
    lines = ["gene | final_label | median_Z | donor_support | R | qc_risk | repl_score"]
    for _, row in top.iterrows():
        lines.append(
            f"{row['gene']} | {row['final_repro_label']} | {float(row['median_Z']):.2f} | "
            f"{float(row['donor_support']):.2f} | {float(row['R']):.2f} | "
            f"{float(row['qc_risk_global']):.2f} | {float(row['replication_score']):.2f}"
        )
    ax3.text(
        0.01,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )
    ax3.set_title("Top replication-ranked genes")
    fig3.tight_layout()
    fig3.savefig(
        out_dir / "rank_table.png", dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight"
    )
    plt.close(fig3)


def _plot_direction_stability(
    donor_level_df: pd.DataFrame, summary_df: pd.DataFrame, *, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if donor_level_df.empty or summary_df.empty:
        _save_placeholder(
            out_dir / "delta_phi_small_multiples.png", "Delta phi", "No data."
        )
        _save_placeholder(out_dir / "R_vs_medianZ.png", "R vs median_Z", "No data.")
        return

    # 1) small-multiple histograms of delta-phi by gene.
    genes = summary_df["gene"].tolist()
    n = len(genes)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig1, axes = plt.subplots(
        nrows, ncols, figsize=(4.1 * ncols, 2.6 * nrows), squeeze=False
    )
    for i, gene in enumerate(genes):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        sub = donor_level_df.loc[
            (donor_level_df["gene"] == gene)
            & (~donor_level_df["underpowered_d"].astype(bool))
        ]
        vals = np.degrees(sub["delta_phi_to_mu"].to_numpy(dtype=float))
        if vals.size == 0:
            ax.text(
                0.5, 0.5, "no eligible donors", ha="center", va="center", fontsize=8
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(gene, fontsize=9)
            continue
        ax.hist(vals, bins=12, color="#72B7B2", alpha=0.90, edgecolor="white")
        ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(gene, fontsize=9)
        ax.set_xlabel("Δphi (deg)")
        ax.set_ylabel("count")
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")
    fig1.suptitle("Direction stability: donor Δphi histograms", y=1.01)
    fig1.tight_layout()
    fig1.savefig(
        out_dir / "delta_phi_small_multiples.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
        bbox_inches="tight",
    )
    plt.close(fig1)

    # 2) R vs median_Z.
    fig2, ax2 = plt.subplots(figsize=(7.8, 6.0))
    for _, row in summary_df.iterrows():
        lbl = str(row["final_repro_label"])
        ax2.scatter(
            float(row["median_Z"]),
            float(row["R"]),
            s=95,
            c=CLASS_COLORS.get(lbl, "#333333"),
            alpha=0.90,
            edgecolors="black",
            linewidths=0.7,
        )
        if lbl == "Donor-replicated localized program":
            ax2.text(
                float(row["median_Z"]) + 0.04,
                float(row["R"]) + 0.01,
                str(row["gene"]),
                fontsize=8,
            )
    ax2.axvline(DONOR_Z_MIN, color="#333333", linestyle="--", linewidth=1.0)
    ax2.axhline(0.6, color="#333333", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("median_Z")
    ax2.set_ylabel("R (direction stability)")
    ax2.set_title("Direction stability vs effect size")
    ax2.grid(alpha=0.25, linewidth=0.6)
    fig2.tight_layout()
    fig2.savefig(
        out_dir / "R_vs_medianZ.png", dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight"
    )
    plt.close(fig2)


def _plot_qc_controls(
    summary_df: pd.DataFrame,
    qc_profiles: dict[str, np.ndarray],
    *,
    n_bins: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        _save_placeholder(
            out_dir / "qc_risk_vs_medianZ.png", "QC controls", "No summary rows."
        )
        _save_placeholder(
            out_dir / "qc_polar_profiles.png", "QC profiles", "No summary rows."
        )
        return

    # 1) qc_risk_global vs median_Z.
    fig1, ax1 = plt.subplots(figsize=(7.6, 5.8))
    ax1.scatter(
        summary_df["qc_risk_global"].to_numpy(dtype=float),
        summary_df["median_Z"].to_numpy(dtype=float),
        c=[
            CLASS_COLORS.get(c, "#333333")
            for c in summary_df["final_repro_label"].tolist()
        ],
        s=85,
        alpha=0.88,
        edgecolors="black",
        linewidths=0.6,
    )
    ax1.axvline(QC_THRESH, color="#8B0000", linestyle="--", linewidth=1.2)
    ax1.axhline(DONOR_Z_MIN, color="#404040", linestyle="-.", linewidth=1.2)
    for _, row in summary_df.loc[
        summary_df["final_repro_label"] == "QC-driven"
    ].iterrows():
        ax1.text(
            float(row["qc_risk_global"]) + 0.01,
            float(row["median_Z"]) + 0.05,
            str(row["gene"]),
            fontsize=8,
        )
    ax1.set_xlabel("qc_risk_global")
    ax1.set_ylabel("median_Z")
    ax1.set_title("QC coupling vs donor-reproducible effect")
    ax1.grid(alpha=0.25, linewidth=0.6)
    fig1.tight_layout()
    fig1.savefig(out_dir / "qc_risk_vs_medianZ.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    # 2) QC pseudo-feature polar profiles.
    if not qc_profiles:
        _save_placeholder(
            out_dir / "qc_polar_profiles.png",
            "QC pseudo-feature polar profiles",
            "No QC covariates available.",
        )
        return

    names = list(qc_profiles.keys())
    n = len(names)
    fig2 = plt.figure(figsize=(4.6 * n, 4.6))
    for i, name in enumerate(names):
        ax = fig2.add_subplot(1, n, i + 1, projection="polar")
        prof = np.asarray(qc_profiles[name], dtype=float)
        centers = theta_bin_centers(int(n_bins))
        th = np.concatenate([centers, centers[:1]])
        yy = np.concatenate([prof, prof[:1]])
        ax.plot(th, yy, color="#8B0000", linewidth=2.0)
        ax.fill_between(th, 0.0, yy, color="#F08080", alpha=0.25)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetagrids(np.arange(0, 360, 90))
        ax.set_title(name)
    fig2.suptitle("QC pseudo-feature RSP profiles (global)")
    fig2.tight_layout()
    fig2.savefig(
        out_dir / "qc_polar_profiles.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
        bbox_inches="tight",
    )
    plt.close(fig2)


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
    p_gene = plots_dir / "01_per_gene_donor_panels"
    p_repro = plots_dir / "02_repro_summary"
    p_dir = plots_dir / "03_direction_stability"
    p_qc = plots_dir / "04_qc_controls"
    for d in [tables_dir, p_overview, p_gene, p_repro, p_dir, p_qc]:
        d.mkdir(parents=True, exist_ok=True)

    warnings_log: list[str] = []

    adata = ad.read_h5ad(h5ad_path)
    _ensure_umap(adata, seed=int(args.seed), recompute_if_missing=False)
    embedding_key, umap_xy = _resolve_embedding(adata, args.embedding_key)
    center_xy = compute_vantage_point(umap_xy, method="median")
    theta_all = compute_theta(umap_xy, center_xy)
    _, bin_id_all = bin_theta(theta_all, int(args.n_bins))
    bin_counts_total_all = np.bincount(bin_id_all, minlength=int(args.n_bins)).astype(
        float
    )

    donor_ids, donor_key = _resolve_donor_ids_required(adata, args.donor_key)
    label_key = _resolve_label_key(adata, args.label_key)
    expr_matrix, adata_like, expr_source = _choose_expression_source(
        adata,
        layer_arg=args.layer,
        use_raw_arg=bool(args.use_raw),
    )

    total_counts = _total_counts_vector(adata, expr_matrix)
    pct_mt_raw, pct_mt_source = _pct_mt_vector(adata, expr_matrix, adata_like)
    pct_mt = (
        None if pct_mt_source == "proxy:zeros" else np.asarray(pct_mt_raw, dtype=float)
    )
    pct_ribo, pct_ribo_source = _compute_pct_counts_ribo(
        adata, expr_matrix, adata_like, total_counts
    )
    if pct_mt is None:
        msg = "pct_counts_mt unavailable; qc_risk_global excludes this covariate."
        print(f"WARNING: {msg}")
        warnings_log.append(msg)
    if pct_ribo is None:
        msg = "pct_counts_ribo unavailable; qc_risk_global excludes this covariate."
        print(f"WARNING: {msg}")
        warnings_log.append(msg)

    print(f"embedding_key_used={embedding_key}")
    print(f"donor_key_used={donor_key}")
    print(f"label_key_used={label_key if label_key is not None else 'None'}")
    print(f"expression_source_used={expr_source}")
    print(f"pct_counts_mt_source={pct_mt_source}")
    print(f"pct_counts_ribo_source={pct_ribo_source}")
    print(
        f"n_cells={int(adata.n_obs)} n_donors={int(np.unique(donor_ids).size)} "
        f"n_bins={int(args.n_bins)} n_perm={int(args.n_perm)} bootstrap={int(args.bootstrap)} seed={int(args.seed)}"
    )

    # Plot overview first.
    _plot_overview(
        adata=adata,
        umap_xy=umap_xy,
        donor_key=donor_key,
        label_key=label_key,
        center_xy=center_xy,
        out_dir=p_overview,
    )

    resolved_genes, panel_df = _build_gene_panel(
        adata_like=adata_like,
        expr_matrix=expr_matrix,
        theta=theta_all,
        donor_ids=donor_ids,
        n_bins=int(args.n_bins),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        bin_id=bin_id_all,
        bin_counts_total=bin_counts_total_all,
    )
    panel_csv = tables_dir / "gene_panel.csv"
    panel_df.to_csv(panel_csv, index=False)
    missing = panel_df.loc[~panel_df["found"].astype(bool), "gene"].astype(str).tolist()
    if missing:
        print(f"missing_genes={','.join(missing)}")

    donors_unique = sorted(np.unique(donor_ids).tolist())
    donor_index = {d: np.flatnonzero(donor_ids == d).astype(int) for d in donors_unique}

    # QC pseudo-feature profiles (global continuous) for optional comparisons.
    qc_profiles: dict[str, np.ndarray] = {}
    qc_vecs = {
        "total_counts": np.asarray(total_counts, dtype=float),
        "pct_counts_mt": (
            np.asarray(pct_mt, dtype=float) if pct_mt is not None else None
        ),
        "pct_counts_ribo": (
            np.asarray(pct_ribo, dtype=float) if pct_ribo is not None else None
        ),
    }
    for qc_name, vals in qc_vecs.items():
        if vals is None:
            continue
        prof = np.asarray(vals, dtype=float)
        # Continuous profile as weighted bin distribution difference vs background.
        clean = np.where(np.isfinite(prof), prof, 0.0)
        if np.nanmin(clean) < 0:
            clean = clean - float(np.nanmin(clean))
        if float(np.sum(clean)) <= 1e-12:
            qc_profiles[qc_name] = np.zeros(int(args.n_bins), dtype=float)
        else:
            w_bin = np.bincount(
                bin_id_all, weights=clean, minlength=int(args.n_bins)
            ).astype(float)
            p_w = w_bin / float(np.sum(clean))
            p_bg = bin_counts_total_all / float(bin_id_all.size)
            qc_profiles[qc_name] = p_w - p_bg

    donor_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    global_artifacts: dict[str, dict[str, np.ndarray]] = {}

    scored_genes = [g for g in resolved_genes if g.found and g.gene_idx is not None]
    for gi, ginfo in enumerate(scored_genes):
        gene = ginfo.gene
        expr = get_feature_vector(expr_matrix, int(ginfo.gene_idx))
        f_global = np.asarray(expr, dtype=float) > 0.0

        # Global donor-stratified baseline.
        perm_global = perm_null_T_and_profile(
            expr=np.asarray(expr, dtype=float),
            theta=theta_all,
            donor_ids=donor_ids,
            n_bins=int(args.n_bins),
            n_perm=int(args.n_perm),
            seed=int(args.seed + 300_000 + gi * 101),
            donor_stratified=True,
            bin_id=bin_id_all,
            bin_counts_total=bin_counts_total_all,
        )
        e_obs_global = np.asarray(perm_global["E_phi_obs"], dtype=float)
        null_e_global = np.asarray(perm_global["null_E_phi"], dtype=float)
        null_t_global = np.asarray(perm_global["null_T"], dtype=float)
        t_obs_global = float(perm_global["T_obs"])
        z_global = float(robust_z(t_obs_global, null_t_global))
        coverage_global = float(coverage_from_null(e_obs_global, null_e_global, q=0.95))
        peaks_global = int(
            peak_count(e_obs_global, null_e_global, smooth_w=3, q_prom=0.95)
        )
        peak_idx = int(np.argmax(np.abs(e_obs_global))) if e_obs_global.size > 0 else 0
        centers = theta_bin_centers(int(args.n_bins))
        phi_global = float(centers[peak_idx]) if centers.size > 0 else 0.0

        global_artifacts[gene] = {
            "expr": np.asarray(expr, dtype=float),
            "E_phi_obs": e_obs_global,
            "null_E_phi": null_e_global,
            "null_T": null_t_global,
        }

        # Global QC coupling.
        rho_total = _safe_spearman(
            f_global.astype(float), np.asarray(total_counts, dtype=float)
        )
        rho_mt = _safe_spearman(f_global.astype(float), pct_mt)
        rho_ribo = _safe_spearman(f_global.astype(float), pct_ribo)
        rho_finite = [abs(v) for v in [rho_total, rho_mt, rho_ribo] if np.isfinite(v)]
        qc_risk_global = float(max(rho_finite)) if rho_finite else 0.0
        sim_vals: list[float] = []
        for qprof in qc_profiles.values():
            s = _cosine_similarity(e_obs_global, qprof)
            if np.isfinite(s):
                sim_vals.append(s)
        sim_qc_global = float(max(sim_vals)) if sim_vals else float("nan")

        # Donor-level metrics with fixed global vantage point.
        donor_effect_rows: list[dict[str, Any]] = []
        for di, donor in enumerate(donors_unique):
            idx = donor_index[donor]
            expr_d = np.asarray(expr[idx], dtype=float)
            xy_d = umap_xy[idx, :]
            theta_d = compute_theta(xy_d, center_xy)  # fixed global vantage point
            _, bin_id_d = bin_theta(theta_d, int(args.n_bins))
            bin_counts_d = np.bincount(bin_id_d, minlength=int(args.n_bins)).astype(
                float
            )
            f_d = expr_d > 0.0

            n_cells_d = int(idx.size)
            prev_d = float(np.mean(f_d))
            n_fg_d = int(f_d.sum())
            low_power_perms = bool(n_cells_d < MIN_CELLS_D)
            n_perm_d = int(100 if low_power_perms else int(args.n_perm))

            if n_fg_d in {0, n_cells_d}:
                e_obs_d = np.zeros(int(args.n_bins), dtype=float)
                null_e_d = np.zeros((n_perm_d, int(args.n_bins)), dtype=float)
                null_t_d = np.zeros(n_perm_d, dtype=float)
                t_d = 0.0
                p_d = 1.0
                z_d = 0.0
                cov_d = 0.0
                peaks_d = 0
                phi_d = 0.0
                perm_warning = "Degenerate foreground (all/none cells)."
            else:
                perm_d = perm_null_T_and_profile(
                    expr=np.asarray(expr_d, dtype=float),
                    theta=theta_d,
                    donor_ids=None,
                    n_bins=int(args.n_bins),
                    n_perm=n_perm_d,
                    seed=int(args.seed + 500_000 + gi * 307 + di * 17),
                    donor_stratified=False,
                    bin_id=bin_id_d,
                    bin_counts_total=bin_counts_d,
                )
                e_obs_d = np.asarray(perm_d["E_phi_obs"], dtype=float)
                null_e_d = np.asarray(perm_d["null_E_phi"], dtype=float)
                null_t_d = np.asarray(perm_d["null_T"], dtype=float)
                t_d = float(perm_d["T_obs"])
                p_d = float(perm_d["p_T"])
                z_d = float(robust_z(t_d, null_t_d))
                cov_d = float(coverage_from_null(e_obs_d, null_e_d, q=0.95))
                peaks_d = int(peak_count(e_obs_d, null_e_d, smooth_w=3, q_prom=0.95))
                peak_idx_d = int(np.argmax(np.abs(e_obs_d)))
                phi_d = float(theta_bin_centers(int(args.n_bins))[peak_idx_d])
                perm_warning = str(perm_d.get("warning", ""))

            underpowered_d = bool(
                (prev_d < P_MIN) or (n_fg_d < MIN_FG) or (n_cells_d < MIN_CELLS_D)
            )
            localized_d = bool((p_d <= 0.05) and (z_d >= DONOR_Z_MIN))
            if localized_d and peaks_d == 1:
                donor_class = "localized-unimodal"
            elif localized_d and peaks_d >= 2:
                donor_class = "localized-multimodal"
            else:
                donor_class = "not-localized"

            drow = {
                "gene": gene,
                "panel_role": ginfo.panel_role,
                "marker_group": ginfo.marker_group,
                "donor_id": donor,
                "cells_d": n_cells_d,
                "prev_d": prev_d,
                "n_fg_d": n_fg_d,
                "n_perm_d": n_perm_d,
                "low_power_perms": low_power_perms,
                "underpowered_d": underpowered_d,
                "T_d": t_d,
                "p_T_d": p_d,
                "Z_T_d": z_d,
                "coverage_C_d": cov_d,
                "peaks_K_d": peaks_d,
                "phi_d_rad": phi_d,
                "phi_d_deg": float(np.degrees(phi_d) % 360.0),
                "localized_d": localized_d,
                "donor_class_d": donor_class,
                "perm_warning_d": perm_warning,
            }
            donor_rows.append(drow)
            donor_effect_rows.append(drow)

        donor_gene_df = pd.DataFrame(donor_effect_rows)
        eligible = donor_gene_df.loc[
            ~donor_gene_df["underpowered_d"].astype(bool)
        ].copy()
        n_eligible = int(eligible.shape[0])

        if n_eligible > 0:
            z_vals = eligible["Z_T_d"].to_numpy(dtype=float)
            p_vals = eligible["p_T_d"].to_numpy(dtype=float)
            phi_vals = eligible["phi_d_rad"].to_numpy(dtype=float)
            median_z = float(np.median(z_vals))
            iqr_z = float(np.quantile(z_vals, 0.75) - np.quantile(z_vals, 0.25))
            ci_low, ci_high = _bootstrap_median_ci(
                z_vals,
                n_boot=int(args.bootstrap),
                seed=int(args.seed + 700_000 + gi * 41),
            )
            donor_support = float(np.mean((z_vals >= DONOR_Z_MIN) & (p_vals <= 0.05)))
            mu_phi, R, circ_sd = _circular_stats(phi_vals)
            delta_phi = _circular_diff(phi_vals, mu_phi)
            eligible_classes = eligible["donor_class_d"].astype(str).tolist()
            consensus_class = pd.Series(eligible_classes).value_counts().idxmax()
            class_stability = float(
                np.mean(pd.Series(eligible_classes) == consensus_class)
            )
        else:
            median_z = float("nan")
            iqr_z = float("nan")
            ci_low = float("nan")
            ci_high = float("nan")
            donor_support = float("nan")
            mu_phi = float("nan")
            R = float("nan")
            circ_sd = float("nan")
            consensus_class = "none"
            class_stability = float("nan")
            delta_phi = np.zeros(0, dtype=float)

        # Write delta_phi back into donor rows for direction plots.
        if n_eligible > 0:
            eligible_idx = eligible.index.to_numpy(dtype=int)
            for j, idx_val in enumerate(eligible_idx):
                donor_rows_idx = (
                    len(donor_rows)
                    - donor_gene_df.shape[0]
                    + int(np.where(donor_gene_df.index == idx_val)[0][0])
                )
                donor_rows[donor_rows_idx]["delta_phi_to_mu"] = float(delta_phi[j])
        # Fill non-eligible.
        for j in range(len(donor_rows) - donor_gene_df.shape[0], len(donor_rows)):
            if "delta_phi_to_mu" not in donor_rows[j]:
                donor_rows[j]["delta_phi_to_mu"] = float("nan")

        summary_rows.append(
            {
                "gene": gene,
                "panel_role": ginfo.panel_role,
                "marker_group": ginfo.marker_group,
                "n_donors_total": int(donor_gene_df.shape[0]),
                "n_donors_eligible": n_eligible,
                "median_Z": median_z,
                "IQR_Z": iqr_z,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "donor_support": donor_support,
                "mu_phi": mu_phi,
                "R": R,
                "circ_sd": circ_sd,
                "consensus_class": consensus_class,
                "class_stability": class_stability,
                "qc_risk_global": qc_risk_global,
                "sim_qc_global": sim_qc_global,
                "T_obs_global": t_obs_global,
                "p_T_global": float(perm_global["p_T"]),
                "q_T_global": float("nan"),
                "Z_T_global": z_global,
                "coverage_C_global": coverage_global,
                "peaks_K_global": peaks_global,
                "phi_global_rad": phi_global,
                "phi_global_deg": float(np.degrees(phi_global) % 360.0),
                "prev_global": float(np.mean(f_global)),
                "n_fg_global": int(f_global.sum()),
                "n_cells_global": int(f_global.size),
                "donor_key_used": donor_key,
                "qc_driven_global": False,
                "final_repro_label": "",
            }
        )

    donor_df = pd.DataFrame(donor_rows)
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise RuntimeError("No genes were scored for Experiment #4.")

    # Global BH-FDR across scored genes.
    summary_df["q_T_global"] = bh_fdr(summary_df["p_T_global"].to_numpy(dtype=float))
    summary_df["qc_driven_global"] = (summary_df["qc_risk_global"] >= QC_THRESH) & (
        summary_df["q_T_global"] <= Q_SIG
    )

    # Final reproducibility label.
    labels_out: list[str] = []
    for _, row in summary_df.iterrows():
        underpowered_gene = bool(
            (int(row["n_donors_eligible"]) < 2)
            or (float(row["prev_global"]) < P_MIN)
            or (int(row["n_fg_global"]) < MIN_FG)
        )
        replicated = bool(
            (float(row["donor_support"]) >= 0.60)
            and ((float(row["R"]) >= 0.60) or (float(row["circ_sd"]) <= 0.90))
            and (float(row["class_stability"]) >= 0.60)
            and (not bool(row["qc_driven_global"]))
        )
        if underpowered_gene:
            labels_out.append("Underpowered")
        elif bool(row["qc_driven_global"]):
            labels_out.append("QC-driven")
        elif replicated:
            labels_out.append("Donor-replicated localized program")
        else:
            labels_out.append("Donor-unstable")
    summary_df["final_repro_label"] = labels_out

    # Save tables.
    donor_csv = tables_dir / "donor_level_metrics.csv"
    summary_csv = tables_dir / "gene_repro_summary.csv"
    donor_df.to_csv(donor_csv, index=False)
    summary_df = summary_df.sort_values(
        by=["final_repro_label", "median_Z", "gene"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary_df.to_csv(summary_csv, index=False)

    # Per-gene donor panels.
    for _, row in summary_df.iterrows():
        gene = str(row["gene"])
        if gene not in global_artifacts:
            continue
        ddf = donor_df.loc[donor_df["gene"] == gene].copy()
        _plot_gene_donor_panel(
            gene=gene,
            global_row=row,
            global_art=global_artifacts[gene],
            donor_gene_df=ddf,
            umap_xy=umap_xy,
            center_xy=center_xy,
            n_bins=int(args.n_bins),
            out_png=p_gene / f"gene_{gene}_donor_repro.png",
        )

    rank_csv = tables_dir / "replication_ranked.csv"
    _plot_repro_summary(summary_df, out_dir=p_repro, rank_csv=rank_csv)
    _plot_direction_stability(donor_df, summary_df, out_dir=p_dir)
    _plot_qc_controls(summary_df, qc_profiles, n_bins=int(args.n_bins), out_dir=p_qc)

    # README metadata.
    donor_counts = (
        adata.obs[donor_key]
        .astype("string")
        .fillna("NA")
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
    )
    readme_lines = [
        "Experiment #4: Donor-reproducible geometry benchmark",
        "",
        "Representation-conditional note: directional geometry is interpreted in embedding space, not tissue axes.",
        "",
        f"embedding_key_used: {embedding_key}",
        f"donor_key_used: {donor_key}",
        f"celltype_key_used: {label_key if label_key is not None else 'None'}",
        f"expression_source_used: {expr_source}",
        f"n_cells_total: {int(adata.n_obs)}",
        f"n_donors: {int(len(donor_counts))}",
        f"n_bins: {int(args.n_bins)}",
        f"n_perm: {int(args.n_perm)}",
        f"bootstrap: {int(args.bootstrap)}",
        "",
        "Cells per donor:",
    ]
    for donor, cnt in donor_counts.items():
        readme_lines.append(f"- {donor}: {int(cnt)}")
    readme_lines.append("")
    readme_lines.append("Warnings:")
    if warnings_log:
        for msg in warnings_log:
            readme_lines.append(f"- {msg}")
    else:
        readme_lines.append("- none")
    (outdir / "README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    cls_counts = (
        summary_df["final_repro_label"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
    )
    cls_txt = "; ".join([f"{k}={v}" for k, v in cls_counts.items()])
    print(f"classification_summary={cls_txt}")
    print(f"gene_panel_csv={panel_csv.as_posix()}")
    print(f"donor_level_csv={donor_csv.as_posix()}")
    print(f"gene_summary_csv={summary_csv.as_posix()}")
    print(f"results_root={outdir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
