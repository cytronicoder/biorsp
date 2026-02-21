#!/usr/bin/env python3
"""CM Experiment #1 (single donor): embedding-stable cardiomyocyte axis programs.

Hypothesis (pre-registered for this run):
Within one donor with many cardiomyocytes, cardiomyocyte expression programs can manifest
as representation-conditional BioRSP-localized unimodal axis-like patterns. These patterns
should be robust across embedding families/hyperparameters (PCA-2D, UMAP grid, t-SNE grid)
and should not be trivially explained by QC gradients (library size / mt% / ribo%).

Important interpretation boundary:
BioRSP peak direction phi is embedding-conditional geometry, not physical tissue direction.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

# Headless plotting backend for deterministic script execution.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

# Allow direct execution: python experiments/.../run_cm_experiment1_embedding_stability.py
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
    _pct_mt_vector,
    _resolve_expr_matrix,
    _total_counts_vector,
)
from biorsp.plotting.qc import plot_categorical_umap, save_numeric_umap
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

CM_MARKER_PANEL: dict[str, list[str]] = {
    "Contractile/core": ["MYH6", "MYH7", "TTN", "TNNT2", "TNNI3", "ACTC1"],
    "Calcium handling": ["RYR2", "PLN", "ATP2A2"],
}

CM_PANEL_PROVENANCE = (
    "Pre-registered CM panel for single-donor Experiment #1: contractile/core, calcium handling, "
    "and stress-associated markers (NPPA/NPPB)."
)

QC_CANDIDATES = {
    "total_counts": [
        "total_counts",
        "n_counts",
        "total_umis",
        "nUMI",
        "n_genes_by_counts",
    ],
    "pct_counts_mt": ["pct_counts_mt", "percent.mt", "pct_mt"],
    "pct_counts_ribo": ["pct_counts_ribo", "percent.ribo", "pct_ribo"],
}

CLASS_ORDER = [
    "Localized-unimodal",
    "Localized-multimodal",
    "Not-localized",
    "Underpowered",
]

CLASS_COLORS = {
    "Localized-unimodal": "#1f77b4",
    "Localized-multimodal": "#ff7f0e",
    "Not-localized": "#8a8a8a",
    "Underpowered": "#d62728",
}

UNDERPOWERED_PREV = 0.005
UNDERPOWERED_MIN_FG = 50
Q_SIG = 0.05


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    embedding_type: str
    params: dict[str, Any]
    coords: np.ndarray


@dataclass(frozen=True)
class GeneStatus:
    gene: str
    marker_group: str
    present: bool
    status: str
    resolved_gene: str
    gene_idx: int | None
    resolution_source: str
    symbol_column: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CM Experiment #1: single-donor cardiomyocyte embedding stability for BioRSP axis-like programs."
        )
    )
    p.add_argument(
        "--h5ad", default="data/processed/HT_pca_umap.h5ad", help="Input AnnData .h5ad"
    )
    p.add_argument(
        "--out",
        default="experiments/heart_single_donor_cardiomyocyte/results/cm_experiment1_embedding_stability",
        help="Output directory",
    )
    p.add_argument("--seed", type=int, default=0, help="Global seed")
    p.add_argument(
        "--n_perm", type=int, default=300, help="Permutations per gene per embedding"
    )
    p.add_argument("--n_bins", type=int, default=64, help="Angular bins")
    p.add_argument(
        "--k_pca",
        type=int,
        default=50,
        help="PCA components for neighbor graph and t-SNE input",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Reduce embedding combinations for quick iteration",
    )
    p.add_argument("--layer", default=None, help="Optional layer override")
    p.add_argument(
        "--use_raw", action="store_true", help="Use adata.raw as expression source"
    )
    p.add_argument("--donor_key", default=None, help="Optional donor key override")
    p.add_argument("--label_key", default=None, help="Optional label key override")
    return p.parse_args()


def _resolve_key_required(
    adata: ad.AnnData,
    requested: str | None,
    candidates: list[str],
    purpose: str,
) -> str:
    if requested is not None:
        if requested in adata.obs.columns:
            return str(requested)
        raise KeyError(f"Requested {purpose} key '{requested}' not found in adata.obs.")
    for key in candidates:
        if key in adata.obs.columns:
            return key
    raise KeyError(
        f"No {purpose} key found. Tried: {', '.join(candidates)}. "
        f"This experiment requires {purpose} labels."
    )


def _choose_expression_source(
    adata: ad.AnnData,
    layer_arg: str | None,
    use_raw_arg: bool,
) -> tuple[Any, Any, str, bool]:
    if layer_arg is not None or use_raw_arg:
        expr_matrix, adata_like, source = _resolve_expr_matrix(
            adata, layer=layer_arg, use_raw=bool(use_raw_arg)
        )
        uses_binary_detection_warning = source in {"X", "raw"}
        return expr_matrix, adata_like, source, uses_binary_detection_warning
    if "counts" in adata.layers:
        expr_matrix, adata_like, source = _resolve_expr_matrix(
            adata, layer="counts", use_raw=False
        )
        return expr_matrix, adata_like, source, False
    expr_matrix, adata_like, source = _resolve_expr_matrix(
        adata, layer=None, use_raw=False
    )
    return expr_matrix, adata_like, source, True


def _safe_numeric_obs(
    adata: ad.AnnData, keys: list[str]
) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        if key not in adata.obs.columns:
            continue
        vals = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float)
        if int(np.isfinite(vals).sum()) == 0:
            continue
        fill = float(np.nanmedian(vals)) if np.isnan(vals).any() else 0.0
        vals = np.where(np.isfinite(vals), vals, fill)
        return vals, key
    return None, None


def _compute_pct_counts_ribo(
    adata: ad.AnnData,
    expr_matrix: Any,
    adata_like: Any,
    total_counts: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    arr, key = _safe_numeric_obs(adata, QC_CANDIDATES["pct_counts_ribo"])
    if arr is not None:
        return arr, f"obs:{key}"

    symbol_col = None
    if hasattr(adata_like, "var") and adata_like.var is not None:
        for col in ["hugo_symbol", "gene_name", "gene_symbol"]:
            if col in adata_like.var.columns:
                symbol_col = col
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
    pct_ribo = np.divide(ribo_counts, np.maximum(total_counts, 1e-12)) * 100.0
    return pct_ribo, f"computed:{symbol_col}"


def _is_cm_label(label: str) -> bool:
    x = str(label).strip().lower()
    if "cardio" in x or "cardiomyocyte" in x:
        return True
    # "cm" as standalone token.
    tokens = (
        x.replace("/", " ")
        .replace("_", " ")
        .replace("-", " ")
        .replace("(", " ")
        .replace(")", " ")
        .split()
    )
    return "cm" in tokens


def _resolve_gene_panel(adata_like: Any) -> tuple[list[GeneStatus], pd.DataFrame]:
    statuses: list[GeneStatus] = []
    rows: list[dict[str, Any]] = []
    used_indices: set[int] = set()

    for group, genes in CM_MARKER_PANEL.items():
        for gene in genes:
            try:
                idx, label, symbol_col, source = resolve_feature_index(adata_like, gene)
                idx_i = int(idx)
                if idx_i in used_indices:
                    status = GeneStatus(
                        gene=gene,
                        marker_group=group,
                        present=False,
                        status="duplicate_index",
                        resolved_gene="",
                        gene_idx=None,
                        resolution_source=source,
                        symbol_column=symbol_col or "",
                    )
                else:
                    used_indices.add(idx_i)
                    status = GeneStatus(
                        gene=gene,
                        marker_group=group,
                        present=True,
                        status="resolved",
                        resolved_gene=str(label),
                        gene_idx=idx_i,
                        resolution_source=source,
                        symbol_column=symbol_col or "",
                    )
            except KeyError:
                status = GeneStatus(
                    gene=gene,
                    marker_group=group,
                    present=False,
                    status="missing",
                    resolved_gene="",
                    gene_idx=None,
                    resolution_source="",
                    symbol_column="",
                )

            statuses.append(status)
            rows.append(
                {
                    "gene": status.gene,
                    "marker_group": status.marker_group,
                    "present": status.present,
                    "status": status.status,
                    "resolved_gene": status.resolved_gene,
                    "gene_idx": status.gene_idx if status.gene_idx is not None else "",
                    "resolution_source": status.resolution_source,
                    "symbol_column": status.symbol_column,
                    "provenance": CM_PANEL_PROVENANCE,
                }
            )

    return statuses, pd.DataFrame(rows)


def _circular_diff_deg(a: float, b: float) -> float:
    """Return wrapped absolute angular difference in degrees in [0, 180]."""
    d = abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)
    return float(d)


def _circular_stats_deg(phi_deg: np.ndarray) -> tuple[float, float, float]:
    """Return (mu_deg, R, circ_sd_rad)."""
    arr = np.asarray(phi_deg, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    ang = np.deg2rad(arr)
    z = np.exp(1j * ang)
    mean_vec = np.mean(z)
    mu = float(np.mod(np.angle(mean_vec), 2.0 * np.pi))
    R = float(np.abs(mean_vec))
    circ_sd = float(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12)))))
    return float(np.rad2deg(mu)), R, circ_sd


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


def _class_from_row(q: float, peaks_k: int, underpowered: bool) -> str:
    if underpowered:
        return "Underpowered"
    if np.isfinite(q) and q <= Q_SIG:
        if int(peaks_k) >= 2:
            return "Localized-multimodal"
        return "Localized-unimodal"
    return "Not-localized"


def _save_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)


def _prepare_embedding_input(
    adata_cm: ad.AnnData,
    expr_matrix_cm: Any,
    expr_source: str,
) -> tuple[ad.AnnData, str]:
    """Build a working AnnData for PCA/UMAP/t-SNE generation."""
    import scanpy as sc

    adata_embed = ad.AnnData(
        X=(
            expr_matrix_cm.copy()
            if hasattr(expr_matrix_cm, "copy")
            else np.array(expr_matrix_cm)
        ),
        obs=adata_cm.obs.copy(),
    )

    prep_note = ""
    if expr_source.startswith("layer:counts"):
        sc.pp.normalize_total(adata_embed, target_sum=1e4)
        sc.pp.log1p(adata_embed)
        prep_note = "counts->normalize_total(1e4)->log1p"
    elif expr_source == "X":
        prep_note = "X_as_is"
    elif expr_source == "raw":
        prep_note = "raw_as_is"
    else:
        prep_note = f"{expr_source}_as_is"
    return adata_embed, prep_note


def _compute_embedding_grid(
    adata_embed: ad.AnnData,
    *,
    seed: int,
    k_pca: int,
    fast: bool,
) -> tuple[list[EmbeddingSpec], int]:
    import scanpy as sc

    n_cells, n_vars = adata_embed.n_obs, adata_embed.n_vars
    n_pcs = int(max(2, min(int(k_pca), 50, n_vars - 1, n_cells - 1)))

    sc.pp.pca(
        adata_embed,
        n_comps=n_pcs,
        svd_solver="arpack",
        random_state=int(seed),
    )
    pca_all = np.asarray(adata_embed.obsm["X_pca"], dtype=float)

    specs: list[EmbeddingSpec] = [
        EmbeddingSpec(
            key="pca2d",
            embedding_type="PCA",
            params={"n_pcs": n_pcs, "seed": int(seed)},
            coords=pca_all[:, :2].copy(),
        )
    ]

    if fast:
        umap_neighbors = [30]
        umap_min_dist = [0.10, 0.50]
        umap_seeds = [0]
        tsne_perplexities = [30]
        tsne_seeds = [0]
    else:
        umap_neighbors = [15, 30, 50]
        umap_min_dist = [0.10, 0.50]
        umap_seeds = [0, 1, 2]
        tsne_perplexities = [30, 50]
        tsne_seeds = [0, 1, 2]

    for nn in umap_neighbors:
        sc.pp.neighbors(
            adata_embed,
            n_neighbors=int(nn),
            n_pcs=n_pcs,
            use_rep="X_pca",
            random_state=int(seed),
        )
        for md in umap_min_dist:
            for rs in umap_seeds:
                sc.tl.umap(adata_embed, min_dist=float(md), random_state=int(rs))
                key = f"umap_nn{nn}_md{md:.2f}_seed{rs}"
                specs.append(
                    EmbeddingSpec(
                        key=key,
                        embedding_type="UMAP",
                        params={
                            "n_neighbors": int(nn),
                            "min_dist": float(md),
                            "random_state": int(rs),
                            "n_pcs": n_pcs,
                        },
                        coords=np.asarray(adata_embed.obsm["X_umap"], dtype=float)[
                            :, :2
                        ].copy(),
                    )
                )

    for perp in tsne_perplexities:
        for rs in tsne_seeds:
            tsne = TSNE(
                n_components=2,
                perplexity=float(perp),
                random_state=int(rs),
                init="pca",
                learning_rate="auto",
                max_iter=1000,
                method="barnes_hut",
            )
            coords = np.asarray(tsne.fit_transform(pca_all), dtype=float)
            key = f"tsne_perp{int(perp)}_seed{rs}"
            specs.append(
                EmbeddingSpec(
                    key=key,
                    embedding_type="tSNE",
                    params={
                        "perplexity": float(perp),
                        "random_state": int(rs),
                        "n_pcs": n_pcs,
                    },
                    coords=coords,
                )
            )

    return specs, n_pcs


def _score_embeddings(
    *,
    embedding_specs: list[EmbeddingSpec],
    gene_statuses: list[GeneStatus],
    expr_matrix_cm: Any,
    n_bins: int,
    n_perm: int,
    seed: int,
    out_tables_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rep_cache: dict[str, dict[str, Any]] = {}

    representative_embedding_key = "umap_nn30_md0.10_seed0"
    if representative_embedding_key not in {e.key for e in embedding_specs}:
        representative_embedding_key = embedding_specs[0].key

    for emb_i, emb in enumerate(embedding_specs):
        center = compute_vantage_point(emb.coords, method="median")
        theta = compute_theta(emb.coords, center)
        _, bin_id = bin_theta(theta, bins=int(n_bins))
        bin_counts_total = np.bincount(bin_id, minlength=int(n_bins)).astype(float)

        for gene_i, status in enumerate(gene_statuses):
            if not status.present or status.gene_idx is None:
                continue

            expr = get_feature_vector(expr_matrix_cm, int(status.gene_idx))
            f = np.asarray(expr, dtype=float) > 0.0
            n_fg = int(f.sum())
            n_cells = int(f.size)
            prev = float(n_fg / max(1, n_cells))
            underpowered = bool(prev < UNDERPOWERED_PREV or n_fg < UNDERPOWERED_MIN_FG)

            e_obs: np.ndarray
            t_obs: float
            phi_hat_deg: float
            null_e: np.ndarray | None = None
            null_t: np.ndarray | None = None
            p_t: float
            z_t: float
            coverage_c: float
            peaks_k: int

            if n_fg == 0 or n_fg == n_cells:
                e_obs = np.zeros(int(n_bins), dtype=float)
                t_obs = 0.0
                phi_hat_deg = float("nan")
                p_t = 1.0
                z_t = float("nan")
                coverage_c = float("nan")
                peaks_k = 0
                underpowered = True
            else:
                e_obs, _, _, _ = compute_rsp_profile_from_boolean(
                    f,
                    theta,
                    int(n_bins),
                    bin_id=bin_id,
                    bin_counts_total=bin_counts_total,
                )
                t_obs = float(np.max(np.abs(e_obs)))
                phi_idx = int(np.argmax(np.abs(e_obs)))
                centers = theta_bin_centers(int(n_bins))
                phi_hat_deg = float(np.degrees(centers[phi_idx]) % 360.0)

                if underpowered:
                    p_t = 1.0
                    z_t = float("nan")
                    coverage_c = float("nan")
                    peaks_k = 0
                else:
                    perm_seed = int(seed + emb_i * 10000 + gene_i * 97 + 7)
                    perm = perm_null_T_and_profile(
                        expr=expr,
                        theta=theta,
                        donor_ids=None,
                        n_bins=int(n_bins),
                        n_perm=int(n_perm),
                        seed=perm_seed,
                        donor_stratified=False,
                        bin_id=bin_id,
                        bin_counts_total=bin_counts_total,
                    )
                    null_e = np.asarray(perm["null_E_phi"], dtype=float)
                    null_t = np.asarray(perm["null_T"], dtype=float)
                    p_t = float(perm["p_T"])
                    z_t = float(robust_z(float(perm["T_obs"]), null_t))
                    coverage_c = float(coverage_from_null(e_obs, null_e, q=0.95))
                    peaks_k = int(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95))

            row = {
                "donor_id": "",
                "gene": status.gene,
                "resolved_gene": status.resolved_gene,
                "gene_idx": int(status.gene_idx),
                "marker_group": status.marker_group,
                "embedding_key": emb.key,
                "embedding_type": emb.embedding_type,
                "params_json": json.dumps(emb.params, sort_keys=True),
                "n_cells": int(f.size),
                "prev": prev,
                "n_fg": int(n_fg),
                "T_obs": float(t_obs),
                "p_T": float(p_t),
                "q_T_within_embedding": np.nan,
                "q_T_global": np.nan,
                "Z_T": float(z_t),
                "coverage_C": float(coverage_c),
                "peaks_K": int(peaks_k),
                "phi_hat_deg": float(phi_hat_deg),
                "underpowered_flag": bool(underpowered),
            }
            rows.append(row)

            if emb.key == representative_embedding_key:
                rep_cache[status.gene] = {
                    "E_phi_obs": np.asarray(e_obs, dtype=float),
                    "null_E_phi": (
                        None if null_e is None else np.asarray(null_e, dtype=float)
                    ),
                    "null_T": (
                        None if null_t is None else np.asarray(null_t, dtype=float)
                    ),
                    "theta": np.asarray(theta, dtype=float),
                    "center": np.asarray(center, dtype=float),
                    "embedding_key": emb.key,
                }

        if (emb_i + 1) % 5 == 0 or emb_i + 1 == len(embedding_specs):
            tmp_df = pd.DataFrame(rows)
            tmp_path = out_tables_dir / "per_embedding_gene_scores.intermediate.csv"
            tmp_df.to_csv(tmp_path, index=False)
            print(
                f"[Progress] scored embeddings {emb_i + 1}/{len(embedding_specs)}; "
                f"intermediate table -> {tmp_path}"
            )

    scores_df = pd.DataFrame(rows)
    if scores_df.empty:
        return scores_df, rep_cache

    # BH-FDR within each embedding.
    for emb_key, idx in scores_df.groupby("embedding_key").groups.items():
        pvals = scores_df.loc[idx, "p_T"].to_numpy(dtype=float)
        qvals = bh_fdr(pvals)
        scores_df.loc[idx, "q_T_within_embedding"] = qvals

    # BH-FDR across all tests.
    scores_df["q_T_global"] = bh_fdr(scores_df["p_T"].to_numpy(dtype=float))

    scores_df["class_label"] = [
        _class_from_row(float(q), int(k), bool(u))
        for q, k, u in zip(
            scores_df["q_T_within_embedding"],
            scores_df["peaks_K"],
            scores_df["underpowered_flag"],
            strict=False,
        )
    ]

    return scores_df, rep_cache


def _gene_stability_summary(scores_df: pd.DataFrame) -> pd.DataFrame:
    if scores_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for gene, sub in scores_df.groupby("gene", sort=False):
        z = sub["Z_T"].to_numpy(dtype=float)
        cov = sub["coverage_C"].to_numpy(dtype=float)
        q = sub["q_T_within_embedding"].to_numpy(dtype=float)
        cls = sub["class_label"].astype(str)

        frac_sig = float(np.mean(q <= Q_SIG))
        z_fin = z[np.isfinite(z)]
        cov_fin = cov[np.isfinite(cov)]
        if z_fin.size > 0:
            med_z = float(np.median(z_fin))
            iqr_z = float(np.quantile(z_fin, 0.75) - np.quantile(z_fin, 0.25))
        else:
            med_z = float("nan")
            iqr_z = float("nan")
        if cov_fin.size > 0:
            med_c = float(np.median(cov_fin))
            iqr_c = float(np.quantile(cov_fin, 0.75) - np.quantile(cov_fin, 0.25))
        else:
            med_c = float("nan")
            iqr_c = float("nan")

        counts = cls.value_counts(dropna=False)
        dominant_class = str(counts.index[0]) if len(counts) > 0 else "Not-localized"
        class_stability = float(counts.iloc[0] / max(1, len(sub)))

        sig_phi = sub.loc[sub["q_T_within_embedding"] <= Q_SIG, "phi_hat_deg"].to_numpy(
            dtype=float
        )
        sig_phi = sig_phi[np.isfinite(sig_phi)]
        if sig_phi.size > 0:
            mu_deg, R, circ_sd = _circular_stats_deg(sig_phi)
        else:
            mu_deg, R, circ_sd = float("nan"), float("nan"), float("nan")

        robust_axis_like = bool(
            frac_sig >= 0.70
            and class_stability >= 0.70
            and np.isfinite(R)
            and R >= 0.60
            and dominant_class == "Localized-unimodal"
        )

        rows.append(
            {
                "gene": str(gene),
                "marker_group": str(sub["marker_group"].iloc[0]),
                "n_embeddings": int(len(sub)),
                "frac_sig": frac_sig,
                "median_Z": med_z,
                "IQR_Z": iqr_z,
                "median_coverage": med_c,
                "IQR_coverage": iqr_c,
                "dominant_class": dominant_class,
                "class_stability": class_stability,
                "phi_mean_deg": mu_deg,
                "R": R,
                "circ_sd": circ_sd,
                "robust_axis_like": robust_axis_like,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            by=["robust_axis_like", "frac_sig", "median_Z"],
            ascending=[False, False, False],
        )
    return out


def _plot_overview(
    *,
    out_dir: Path,
    donor_counts: pd.DataFrame,
    donor_star: str,
    label_counts_donor: pd.Series,
    rep_coords: np.ndarray,
    rep_key: str,
    tnnt2_expr: np.ndarray | None,
    qc_total_counts: np.ndarray | None,
    qc_pct_mt: np.ndarray | None,
    qc_pct_ribo: np.ndarray | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Representative embedding overlays.
    if tnnt2_expr is not None:
        save_numeric_umap(
            rep_coords,
            np.log1p(np.maximum(np.asarray(tnnt2_expr, dtype=float), 0.0)),
            out_dir / "rep_embedding_tnnt2.png",
            title=f"Representative embedding ({rep_key}) colored by TNNT2",
            cmap="Reds",
            colorbar_label="log1p(TNNT2)",
        )
    else:
        _save_placeholder(
            out_dir / "rep_embedding_tnnt2.png",
            "Representative embedding TNNT2",
            "TNNT2 missing from selected expression namespace.",
        )

    if qc_total_counts is not None:
        save_numeric_umap(
            rep_coords,
            np.log1p(np.maximum(qc_total_counts, 0.0)),
            out_dir / "rep_embedding_total_counts.png",
            title=f"Representative embedding ({rep_key}) - log1p(total_counts)",
            cmap="viridis",
            colorbar_label="log1p(total_counts)",
        )
    else:
        _save_placeholder(
            out_dir / "rep_embedding_total_counts.png",
            "total_counts",
            "total_counts unavailable.",
        )

    if qc_pct_mt is not None:
        save_numeric_umap(
            rep_coords,
            qc_pct_mt,
            out_dir / "rep_embedding_pct_counts_mt.png",
            title=f"Representative embedding ({rep_key}) - pct_counts_mt",
            cmap="magma",
            colorbar_label="pct_counts_mt",
        )
    else:
        _save_placeholder(
            out_dir / "rep_embedding_pct_counts_mt.png",
            "pct_counts_mt",
            "pct_counts_mt unavailable.",
        )

    if qc_pct_ribo is not None:
        save_numeric_umap(
            rep_coords,
            qc_pct_ribo,
            out_dir / "rep_embedding_pct_counts_ribo.png",
            title=f"Representative embedding ({rep_key}) - pct_counts_ribo",
            cmap="plasma",
            colorbar_label="pct_counts_ribo",
        )

    # Donor CM counts.
    fig1, ax1 = plt.subplots(figsize=(10.0, 4.6))
    bars = ax1.bar(
        donor_counts["donor_id"].astype(str),
        donor_counts["n_cm"].to_numpy(dtype=float),
        color="#9aa0a6",
        edgecolor="white",
        linewidth=0.8,
    )
    for i, did in enumerate(donor_counts["donor_id"].astype(str).tolist()):
        if did == donor_star:
            bars[i].set_color("#d62728")
    ax1.set_ylabel("Cardiomyocyte cells")
    ax1.set_xlabel("Donor")
    ax1.set_title("Cardiomyocyte counts per donor (donor_star highlighted)")
    ax1.tick_params(axis="x", rotation=70)
    fig1.tight_layout()
    fig1.savefig(
        out_dir / "cardiomyocyte_counts_per_donor.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig1)

    # Donor_star label composition.
    fig2, ax2 = plt.subplots(figsize=(9.6, 5.0))
    lc = label_counts_donor.sort_values(ascending=False)
    ax2.bar(
        lc.index.astype(str),
        lc.values,
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.8,
    )
    ax2.set_title(f"Cell-type composition within donor_star={donor_star}")
    ax2.set_ylabel("Cell count")
    ax2.set_xlabel("Label")
    ax2.tick_params(axis="x", rotation=70)
    fig2.tight_layout()
    fig2.savefig(
        out_dir / "donor_star_celltype_composition.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig2)


def _plot_gene_panels(
    *,
    out_dir: Path,
    genes_present: list[str],
    scores_df: pd.DataFrame,
    embedding_map: dict[str, EmbeddingSpec],
    expr_by_gene: dict[str, np.ndarray],
    rep_cache: dict[str, dict[str, Any]],
    n_bins: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(genes_present) == 0:
        _save_placeholder(
            out_dir / "no_genes.png", "Per-gene panels", "No CM panel genes resolved."
        )
        return

    fixed_keys = [
        "pca2d",
        "umap_nn30_md0.10_seed0",
        "umap_nn50_md0.50_seed1",
        "tsne_perp30_seed0",
        "tsne_perp50_seed1",
    ]

    for gene in genes_present:
        sub = scores_df.loc[scores_df["gene"] == gene].copy()
        if sub.empty:
            continue

        expr = np.asarray(expr_by_gene[gene], dtype=float)
        x_plot = np.log1p(np.maximum(expr, 0.0))
        if np.isfinite(x_plot).sum() > 0:
            vmin = float(np.quantile(x_plot, 0.01))
            vmax = float(np.quantile(x_plot, 0.99))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
                vmin = float(np.nanmin(x_plot))
                vmax = float(np.nanmax(x_plot) + 1e-6)
        else:
            vmin, vmax = 0.0, 1.0

        # Worst-case embedding by minimal Z_T for this gene.
        sub_nonan = sub.copy()
        sub_nonan["Z_T"] = pd.to_numeric(sub_nonan["Z_T"], errors="coerce")
        if sub_nonan["Z_T"].notna().any():
            worst_row = sub_nonan.sort_values(by="Z_T", ascending=True).iloc[0]
            worst_key = str(worst_row["embedding_key"])
        else:
            worst_key = str(sub.iloc[0]["embedding_key"])

        panel_keys = []
        for k in fixed_keys:
            if k in embedding_map and k not in panel_keys:
                panel_keys.append(k)
        if worst_key in embedding_map and worst_key not in panel_keys:
            panel_keys.append(worst_key)
        if len(panel_keys) < 6:
            for k in embedding_map.keys():
                if k not in panel_keys:
                    panel_keys.append(k)
                if len(panel_keys) >= 6:
                    break
        panel_keys = panel_keys[:6]

        fig = plt.figure(figsize=(19.5, 9.8))
        gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.95], wspace=0.18, hspace=0.30)

        last_sc = None
        for i, key in enumerate(panel_keys):
            ax = fig.add_subplot(gs[0, i])
            coords = embedding_map[key].coords
            order = np.argsort(x_plot, kind="mergesort")
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c="#dddddd",
                s=5,
                alpha=0.25,
                linewidths=0,
                rasterized=True,
            )
            last_sc = ax.scatter(
                coords[order, 0],
                coords[order, 1],
                c=x_plot[order],
                cmap="Reds",
                s=7,
                alpha=0.92,
                linewidths=0,
                rasterized=True,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(key, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        if last_sc is not None:
            cax = fig.add_axes([0.92, 0.61, 0.015, 0.23])
            cb = fig.colorbar(last_sc, cax=cax)
            cb.set_label("log1p(expr)")

        rep_key = "umap_nn30_md0.10_seed0"
        if rep_key not in embedding_map:
            rep_key = panel_keys[0]
        rep_row = sub.loc[sub["embedding_key"] == rep_key]
        if rep_row.empty:
            rep_row = sub.iloc[[0]]
        rep_row_s = rep_row.iloc[0]

        rep = rep_cache.get(gene, None)
        ax_polar = fig.add_subplot(gs[1, 0:3], projection="polar")
        ax_hist = fig.add_subplot(gs[1, 3:5])
        ax_text = fig.add_subplot(gs[1, 5])
        ax_text.axis("off")

        if rep is not None and rep.get("E_phi_obs") is not None:
            e_obs = np.asarray(rep["E_phi_obs"], dtype=float)
            centers = theta_bin_centers(int(n_bins))
            th = np.concatenate([centers, centers[:1]])
            e_plot = np.concatenate([e_obs, e_obs[:1]])
            ax_polar.plot(th, e_plot, color="#8B0000", linewidth=2.0, label="E_phi obs")

            null_e = rep.get("null_E_phi", None)
            null_t = rep.get("null_T", None)
            if null_e is not None:
                null_e_arr = np.asarray(null_e, dtype=float)
                hi = np.quantile(null_e_arr, 0.95, axis=0)
                lo = np.quantile(null_e_arr, 0.05, axis=0)
                hi_plot = np.concatenate([hi, hi[:1]])
                lo_plot = np.concatenate([lo, lo[:1]])
                ax_polar.plot(
                    th,
                    hi_plot,
                    color="#333333",
                    linestyle="--",
                    linewidth=1.3,
                    label="null 95%",
                )
                ax_polar.plot(
                    th,
                    lo_plot,
                    color="#333333",
                    linestyle="--",
                    linewidth=1.0,
                    label="null 5%",
                )
                ax_polar.fill_between(th, lo_plot, hi_plot, color="#999999", alpha=0.22)

                nt = np.asarray(null_t, dtype=float).ravel()
                bins_hist = int(min(40, max(10, np.ceil(np.sqrt(max(10, nt.size))))))
                ax_hist.hist(
                    nt, bins=bins_hist, color="#7aa6d6", edgecolor="white", alpha=0.95
                )
                ax_hist.axvline(
                    float(rep_row_s["T_obs"]),
                    color="#8B0000",
                    linestyle="--",
                    linewidth=2.0,
                )
                ax_hist.set_title(f"null_T histogram ({rep_key})")
                ax_hist.set_xlabel("T under null")
                ax_hist.set_ylabel("count")
            else:
                ax_hist.text(
                    0.5,
                    0.5,
                    "Underpowered\n(no permutation null)",
                    ha="center",
                    va="center",
                )
                ax_hist.set_xticks([])
                ax_hist.set_yticks([])

            ax_polar.set_theta_zero_location("E")
            ax_polar.set_theta_direction(1)
            ax_polar.set_title(f"RSP polar ({rep_key})")
            ax_polar.legend(loc="upper right", fontsize=8)
        else:
            ax_polar.text(
                0.5, 0.5, "No representative RSP profile", ha="center", va="center"
            )
            ax_polar.set_xticks([])
            ax_polar.set_yticks([])
            ax_hist.axis("off")

        text = (
            f"Gene: {gene}\n"
            f"Rep embedding: {rep_key}\n"
            f"Z_T: {float(rep_row_s['Z_T']):.2f}\n"
            f"q_T_within: {float(rep_row_s['q_T_within_embedding']):.2e}\n"
            f"coverage_C: {float(rep_row_s['coverage_C']):.3f}\n"
            f"peaks_K: {int(rep_row_s['peaks_K'])}\n"
            f"phi_hat: {float(rep_row_s['phi_hat_deg']):.1f} deg\n"
            f"worst-case embedding: {worst_key}"
        )
        ax_text.text(0.02, 0.98, text, va="top", ha="left", fontsize=10)

        fig.suptitle(
            f"{gene}: embedding sensitivity panel (single-donor cardiomyocytes)",
            y=0.995,
            fontsize=13,
        )
        fig.tight_layout(rect=[0.0, 0.0, 0.91, 0.97])
        fig.savefig(out_dir / f"gene_{gene}.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig)


def _ordered_gene_index(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 2:
        return np.arange(matrix.shape[0], dtype=int)
    arr = np.asarray(matrix, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        dist = pdist(arr, metric="euclidean")
        if not np.all(np.isfinite(dist)) or dist.size == 0:
            return np.arange(arr.shape[0], dtype=int)
        z = linkage(dist, method="average")
        return leaves_list(z).astype(int)
    except Exception:
        return np.arange(arr.shape[0], dtype=int)


def _plot_embedding_stability(
    *,
    out_dir: Path,
    scores_df: pd.DataFrame,
    stability_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if scores_df.empty or stability_df.empty:
        _save_placeholder(
            out_dir / "heatmap_ZT.png", "Embedding stability", "No scored genes."
        )
        _save_placeholder(
            out_dir / "heatmap_qT.png", "Embedding stability", "No scored genes."
        )
        _save_placeholder(
            out_dir / "robustness_scatter.png",
            "Embedding stability",
            "No scored genes.",
        )
        return

    pivot_z = scores_df.pivot(index="gene", columns="embedding_key", values="Z_T")
    pivot_q = scores_df.pivot(
        index="gene", columns="embedding_key", values="q_T_within_embedding"
    )

    gene_order_idx = _ordered_gene_index(
        np.nan_to_num(pivot_z.to_numpy(dtype=float), nan=0.0)
    )
    gene_order = pivot_z.index[gene_order_idx]

    emb_order = list(pivot_z.columns)
    mat_z = pivot_z.loc[gene_order, emb_order].to_numpy(dtype=float)
    mat_q = pivot_q.loc[gene_order, emb_order].to_numpy(dtype=float)

    fig1, ax1 = plt.subplots(
        figsize=(1.1 * len(emb_order) + 4.5, 0.55 * len(gene_order) + 2.2)
    )
    im1 = ax1.imshow(np.nan_to_num(mat_z, nan=0.0), aspect="auto", cmap="magma")
    ax1.set_title("Genes x embeddings: Z_T")
    ax1.set_xlabel("Embedding")
    ax1.set_ylabel("Gene")
    ax1.set_xticks(np.arange(len(emb_order)))
    ax1.set_xticklabels(emb_order, rotation=90, fontsize=7)
    ax1.set_yticks(np.arange(len(gene_order)))
    ax1.set_yticklabels(gene_order, fontsize=8)
    cb1 = fig1.colorbar(im1, ax=ax1, shrink=0.86)
    cb1.set_label("Z_T")
    fig1.tight_layout()
    fig1.savefig(
        out_dir / "heatmap_genes_embeddings_ZT.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig1)

    fig2, ax2 = plt.subplots(
        figsize=(1.1 * len(emb_order) + 4.5, 0.55 * len(gene_order) + 2.2)
    )
    q_cap = np.minimum(np.nan_to_num(mat_q, nan=1.0), 0.2)
    im2 = ax2.imshow(q_cap, aspect="auto", cmap="viridis_r", vmin=0.0, vmax=0.2)
    ax2.set_title("Genes x embeddings: q_T within embedding (capped at 0.2)")
    ax2.set_xlabel("Embedding")
    ax2.set_ylabel("Gene")
    ax2.set_xticks(np.arange(len(emb_order)))
    ax2.set_xticklabels(emb_order, rotation=90, fontsize=7)
    ax2.set_yticks(np.arange(len(gene_order)))
    ax2.set_yticklabels(gene_order, fontsize=8)
    cb2 = fig2.colorbar(im2, ax=ax2, shrink=0.86)
    cb2.set_label("q_T_within_embedding")
    fig2.tight_layout()
    fig2.savefig(
        out_dir / "heatmap_genes_embeddings_qT.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8.2, 6.2))
    for cls in [
        "Localized-unimodal",
        "Localized-multimodal",
        "Not-localized",
        "Underpowered",
    ]:
        sub = stability_df.loc[stability_df["dominant_class"] == cls]
        if sub.empty:
            continue
        sizes = 60.0 + 280.0 * np.nan_to_num(sub["R"].to_numpy(dtype=float), nan=0.0)
        ax3.scatter(
            sub["median_Z"].to_numpy(dtype=float),
            sub["frac_sig"].to_numpy(dtype=float),
            s=sizes,
            c=CLASS_COLORS.get(cls, "#555555"),
            alpha=0.88,
            edgecolors="black",
            linewidths=0.5,
            label=cls,
        )

    top = stability_df.sort_values(by="median_Z", ascending=False).head(8)
    for _, row in top.iterrows():
        ax3.text(
            float(row["median_Z"]),
            float(row["frac_sig"]) + 0.015,
            str(row["gene"]),
            fontsize=8,
        )

    ax3.axhline(0.70, color="#444444", linestyle="--", linewidth=1.0)
    ax3.axvline(4.0, color="#444444", linestyle=":", linewidth=1.0)
    ax3.set_xlabel("median_Z")
    ax3.set_ylabel("frac_sig (q<=0.05 across embeddings)")
    ax3.set_title("Embedding robustness scatter (size ~ R, color ~ dominant class)")
    ax3.legend(loc="best", fontsize=8, frameon=True)
    fig3.tight_layout()
    fig3.savefig(
        out_dir / "robustness_scatter_medianZ_vs_fracsig.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
    )
    plt.close(fig3)


def _plot_direction_stability(
    *,
    out_dir: Path,
    scores_df: pd.DataFrame,
    stability_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if scores_df.empty or stability_df.empty:
        _save_placeholder(
            out_dir / "phi_circular_small_multiples.png",
            "Direction stability",
            "No scored genes.",
        )
        _save_placeholder(
            out_dir / "R_vs_frac_sig.png", "Direction stability", "No scored genes."
        )
        return

    genes = stability_df["gene"].astype(str).tolist()
    n = len(genes)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))

    fig = plt.figure(figsize=(4.2 * n_cols, 3.8 * n_rows))
    for i, gene in enumerate(genes):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="polar")
        sub = scores_df.loc[
            (scores_df["gene"] == gene) & (scores_df["q_T_within_embedding"] <= Q_SIG)
        ]
        phi = sub["phi_hat_deg"].to_numpy(dtype=float)
        phi = phi[np.isfinite(phi)]
        if phi.size == 0:
            ax.text(
                0.5,
                0.5,
                "No sig embeddings",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
            )
        else:
            rad = np.deg2rad(phi)
            ax.scatter(rad, np.ones_like(rad), s=26, c="#1f77b4", alpha=0.85)
            mu_deg, R, circ_sd = _circular_stats_deg(phi)
            mu = np.deg2rad(mu_deg)
            ax.plot([mu, mu], [0.0, 1.1], color="#d62728", linewidth=2.0)
            ax.text(
                0.02,
                0.02,
                f"n_sig={phi.size}\nR={R:.2f}\ncirc_sd={circ_sd:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.8},
            )
        ax.set_rticks([])
        ax.set_title(gene, fontsize=9)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)

    fig.suptitle("Direction stability across significant embeddings (phi-hat)", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(
        out_dir / "phi_circular_small_multiples.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7.8, 5.8))
    ax2.scatter(
        stability_df["R"].to_numpy(dtype=float),
        stability_df["frac_sig"].to_numpy(dtype=float),
        c=[
            CLASS_COLORS.get(str(c), "#555555")
            for c in stability_df["dominant_class"].astype(str)
        ],
        s=100,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.4,
    )
    for _, row in stability_df.iterrows():
        ax2.text(
            float(row["R"]),
            float(row["frac_sig"]) + 0.012,
            str(row["gene"]),
            fontsize=8,
        )
    ax2.axvline(0.60, color="#333333", linestyle="--", linewidth=1.0)
    ax2.axhline(0.70, color="#333333", linestyle=":", linewidth=1.0)
    ax2.set_xlabel("R (circular concentration)")
    ax2.set_ylabel("frac_sig")
    ax2.set_title("Direction stability vs significance stability")
    fig2.tight_layout()
    fig2.savefig(out_dir / "R_vs_frac_sig.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)


def _plot_qc_controls(
    *,
    out_dir: Path,
    rep_coords: np.ndarray,
    rep_key: str,
    qc_total_counts: np.ndarray | None,
    qc_pct_mt: np.ndarray | None,
    qc_pct_ribo: np.ndarray | None,
    scores_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Representative QC heatmaps.
    if qc_total_counts is not None and qc_pct_mt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
        order1 = np.argsort(qc_total_counts, kind="mergesort")
        s1 = axes[0].scatter(
            rep_coords[order1, 0],
            rep_coords[order1, 1],
            c=np.log1p(np.maximum(qc_total_counts[order1], 0.0)),
            cmap="viridis",
            s=8,
            alpha=0.9,
            linewidths=0,
            rasterized=True,
        )
        axes[0].set_title(f"{rep_key}: log1p(total_counts)")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        cb1 = fig.colorbar(s1, ax=axes[0], fraction=0.046, pad=0.03)
        cb1.set_label("log1p(total_counts)")

        order2 = np.argsort(qc_pct_mt, kind="mergesort")
        s2 = axes[1].scatter(
            rep_coords[order2, 0],
            rep_coords[order2, 1],
            c=qc_pct_mt[order2],
            cmap="magma",
            s=8,
            alpha=0.9,
            linewidths=0,
            rasterized=True,
        )
        axes[1].set_title(f"{rep_key}: pct_counts_mt")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        cb2 = fig.colorbar(s2, ax=axes[1], fraction=0.046, pad=0.03)
        cb2.set_label("pct_counts_mt")

        fig.tight_layout()
        fig.savefig(
            out_dir / "rep_embedding_qc_totalcounts_pctmt.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
        )
        plt.close(fig)
    else:
        _save_placeholder(
            out_dir / "rep_embedding_qc_totalcounts_pctmt.png",
            "QC controls",
            "Insufficient QC covariates for representative UMAP panel.",
        )

    # QC correlation scatter from per-embedding scores.
    if scores_df.empty:
        _save_placeholder(
            out_dir / "qc_rho_mt_vs_rho_counts.png",
            "QC audit",
            "No scored genes.",
        )
        return

    qc_rows: list[dict[str, Any]] = []
    for gene, sub in scores_df.groupby("gene", sort=False):
        # Use representative row for plotting to match panel's representative embedding.
        rep = sub.loc[sub["embedding_key"] == "umap_nn30_md0.10_seed0"]
        row = rep.iloc[0] if not rep.empty else sub.iloc[0]
        qc_rows.append(
            {
                "gene": str(gene),
                "rho_counts": float(row.get("rho_counts", np.nan)),
                "rho_mt": float(row.get("rho_mt", np.nan)),
                "rho_ribo": float(row.get("rho_ribo", np.nan)),
                "Z_T": float(row.get("Z_T", np.nan)),
                "q_T_within_embedding": float(row.get("q_T_within_embedding", np.nan)),
            }
        )
    qcdf = pd.DataFrame(qc_rows)

    fig2, ax2 = plt.subplots(figsize=(7.8, 6.1))
    sizes = 60 + 35 * np.nan_to_num(np.abs(qcdf["Z_T"].to_numpy(dtype=float)), nan=0.0)
    colors = np.where(
        qcdf["q_T_within_embedding"].to_numpy(dtype=float) <= Q_SIG,
        "#d62728",
        "#4c78a8",
    )
    ax2.scatter(
        qcdf["rho_counts"].to_numpy(dtype=float),
        qcdf["rho_mt"].to_numpy(dtype=float),
        s=sizes,
        c=colors,
        alpha=0.88,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, row in qcdf.iterrows():
        ax2.text(
            float(row["rho_counts"]),
            float(row["rho_mt"]) + 0.01,
            str(row["gene"]),
            fontsize=8,
        )
    ax2.axvline(0.0, color="#666666", linewidth=0.9)
    ax2.axhline(0.0, color="#666666", linewidth=0.9)
    ax2.set_xlabel("Spearman rho(f, total_counts)")
    ax2.set_ylabel("Spearman rho(f, pct_counts_mt)")
    ax2.set_title(
        "QC correlation audit (size ~ |Z_T|, red=significant representative embedding)"
    )
    fig2.tight_layout()
    fig2.savefig(out_dir / "qc_rho_mt_vs_rho_counts.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)

    if qc_pct_ribo is not None and "rho_ribo" in scores_df.columns:
        fig3, ax3 = plt.subplots(figsize=(7.8, 6.0))
        ribo_pts = qcdf["rho_ribo"].to_numpy(dtype=float)
        ax3.scatter(
            qcdf["rho_counts"].to_numpy(dtype=float),
            ribo_pts,
            s=95,
            c="#7f3c8d",
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
        )
        for _, row in qcdf.iterrows():
            ax3.text(
                float(row["rho_counts"]),
                float(row["rho_ribo"]) + 0.01,
                str(row["gene"]),
                fontsize=8,
            )
        ax3.axvline(0.0, color="#666666", linewidth=0.9)
        ax3.axhline(0.0, color="#666666", linewidth=0.9)
        ax3.set_xlabel("Spearman rho(f, total_counts)")
        ax3.set_ylabel("Spearman rho(f, pct_counts_ribo)")
        ax3.set_title("QC audit: ribosomal correlation")
        fig3.tight_layout()
        fig3.savefig(
            out_dir / "qc_rho_ribo_vs_rho_counts.png", dpi=DEFAULT_PLOT_STYLE.dpi
        )
        plt.close(fig3)


def _write_readme(
    *,
    out_path: Path,
    seed: int,
    n_perm: int,
    n_bins: int,
    k_pca: int,
    donor_key: str,
    label_key: str,
    donor_star: str,
    cm_labels_included: list[str],
    cm_label_counts: dict[str, int],
    expr_source: str,
    expr_warning: bool,
    embedding_prep_note: str,
    n_embeddings: int,
    n_cells_donor: int,
    n_cells_cm: int,
    cm_underpowered: bool,
    qc_sources: dict[str, str],
    stability_df: pd.DataFrame,
) -> None:
    robust_genes = []
    if not stability_df.empty:
        robust_genes = (
            stability_df.loc[stability_df["robust_axis_like"], "gene"]
            .astype(str)
            .tolist()
        )

    lines: list[str] = []
    lines.append(
        "CM Experiment #1 (Single-donor): Cardiomyocyte axis stability across embeddings"
    )
    lines.append("")
    lines.append("Hypothesis")
    lines.append(
        "Within one donor with many cardiomyocytes, continuous cardiomyocyte programs should show "
        "representation-conditional BioRSP-localized unimodal axis-like geometry that is stable across "
        "embedding families/hyperparameters and not explained by QC gradients."
    )
    lines.append("")
    lines.append("Interpretation guardrail")
    lines.append(
        "BioRSP direction phi is representation-conditional (embedding geometry), not physical tissue direction."
    )
    lines.append("")
    lines.append("Run metadata")
    lines.append(f"- seed: {seed}")
    lines.append(f"- n_perm: {n_perm}")
    lines.append(f"- n_bins: {n_bins}")
    lines.append(f"- k_pca: {k_pca}")
    lines.append(f"- donor_key_used: {donor_key}")
    lines.append(f"- label_key_used: {label_key}")
    lines.append(f"- donor_star: {donor_star}")
    lines.append(f"- expression_source_used: {expr_source}")
    lines.append(f"- embedding_input_prep: {embedding_prep_note}")
    lines.append(f"- embeddings_tested: {n_embeddings}")
    lines.append(f"- donor_star_total_cells: {n_cells_donor}")
    lines.append(f"- donor_star_cardiomyocytes: {n_cells_cm}")
    lines.append(f"- cm_underpowered: {cm_underpowered}")
    lines.append("")
    lines.append("Cardiomyocyte labels included")
    for label in cm_labels_included:
        lines.append(f"- {label}: {cm_label_counts.get(label, 0)}")
    lines.append("")
    lines.append("QC feature sources")
    for key, source in qc_sources.items():
        lines.append(f"- {key}: {source}")
    lines.append("")
    if expr_warning:
        lines.append("Warning")
        lines.append(
            "Counts layer not used. Detection foreground (expr>0) may be less interpretable when source is log-normalized X/raw."
        )
        lines.append("")

    lines.append("Embedding-robust axis-like CM programs")
    if robust_genes:
        for gene in robust_genes:
            lines.append(f"- {gene}")
    else:
        lines.append("- none met the strict robust_axis_like criteria in this run")
    lines.append("")

    lines.append("Single-donor rigor note")
    lines.append(
        "No donor replication is possible in this experiment. Evidence strength is from embedding sensitivity "
        "analysis and QC negative controls; donor resampling/replication is deferred to later experiments."
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
        plots_dir / "01_per_gene_panels",
        plots_dir / "02_embedding_stability",
        plots_dir / "03_direction_stability",
        plots_dir / "04_qc_controls",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad)

    donor_key = _resolve_key_required(
        adata, args.donor_key, DONOR_KEY_CANDIDATES, purpose="donor"
    )
    label_key = _resolve_key_required(
        adata, args.label_key, LABEL_KEY_CANDIDATES, purpose="cell-type label"
    )

    labels_all = adata.obs[label_key].astype("string").fillna("NA").astype(str)
    cm_mask_all = labels_all.map(_is_cm_label).to_numpy(dtype=bool)
    if int(cm_mask_all.sum()) == 0:
        raise RuntimeError(
            "Cardiomyocyte subset is empty using label contains any of ['cardio', 'cardiomyocyte', 'cm']"
        )

    donor_ids_all = adata.obs[donor_key].astype("string").fillna("NA").astype(str)
    donor_summary = (
        pd.DataFrame({"donor_id": donor_ids_all.to_numpy(), "is_cm": cm_mask_all})
        .groupby("donor_id", as_index=False)
        .agg(n_cells_total=("is_cm", "size"), n_cm=("is_cm", "sum"))
        .sort_values(
            by=["n_cm", "n_cells_total", "donor_id"], ascending=[False, False, True]
        )
        .reset_index(drop=True)
    )
    donor_star = str(donor_summary.iloc[0]["donor_id"])
    donor_summary["is_donor_star"] = donor_summary["donor_id"].astype(str) == donor_star

    donor_mask = donor_ids_all.astype(str).to_numpy() == donor_star
    adata_donor = adata[donor_mask].copy()
    labels_donor = adata_donor.obs[label_key].astype("string").fillna("NA").astype(str)
    cm_mask_donor = labels_donor.map(_is_cm_label).to_numpy(dtype=bool)

    if int(cm_mask_donor.sum()) == 0:
        raise RuntimeError(
            f"Selected donor_star={donor_star} has zero cardiomyocytes after filtering."
        )

    adata_cm = adata_donor[cm_mask_donor].copy()
    cm_underpowered = bool(int(adata_cm.n_obs) < 2000)
    if cm_underpowered:
        print(
            "WARNING: adata_cm.n_obs < 2000. Proceeding with cm_underpowered=True "
            f"(n_cm={int(adata_cm.n_obs)})."
        )

    expr_matrix_cm, adata_like_cm, expr_source, expr_warning = (
        _choose_expression_source(
            adata_cm,
            layer_arg=args.layer,
            use_raw_arg=bool(args.use_raw),
        )
    )

    expr_matrix_donor, adata_like_donor, _, _ = _choose_expression_source(
        adata_donor,
        layer_arg=args.layer,
        use_raw_arg=bool(args.use_raw),
    )

    # Resolve panel genes.
    gene_statuses, gene_panel_df = _resolve_gene_panel(adata_like_cm)
    gene_panel_df.to_csv(tables_dir / "gene_panel_status.csv", index=False)

    # Build QC covariates on CM subset for audit.
    qc_total_counts, total_key = _safe_numeric_obs(
        adata_cm, QC_CANDIDATES["total_counts"]
    )
    if qc_total_counts is None:
        qc_total_counts = _total_counts_vector(adata_cm, expr_matrix_cm)
        total_key = "computed:expr_sum"

    qc_pct_mt_obs, mt_key = _safe_numeric_obs(adata_cm, QC_CANDIDATES["pct_counts_mt"])
    if qc_pct_mt_obs is None:
        qc_pct_mt_obs, mt_source = _pct_mt_vector(
            adata_cm, expr_matrix_cm, adata_like_cm
        )
        mt_key = mt_source
    qc_pct_ribo, ribo_key = _compute_pct_counts_ribo(
        adata_cm,
        expr_matrix_cm,
        adata_like_cm,
        np.asarray(qc_total_counts, dtype=float),
    )

    qc_sources = {
        "total_counts": str(total_key),
        "pct_counts_mt": str(mt_key),
        "pct_counts_ribo": str(ribo_key),
    }

    # Build embeddings on CM subset.
    adata_embed, embed_prep_note = _prepare_embedding_input(
        adata_cm,
        expr_matrix_cm,
        expr_source,
    )
    embedding_specs, n_pcs_used = _compute_embedding_grid(
        adata_embed,
        seed=int(args.seed),
        k_pca=int(args.k_pca),
        fast=bool(args.fast),
    )
    embedding_map = {spec.key: spec for spec in embedding_specs}
    representative_key = "umap_nn30_md0.10_seed0"
    if representative_key not in embedding_map:
        representative_key = embedding_specs[0].key

    # Score genes across embeddings.
    scores_df, rep_cache = _score_embeddings(
        embedding_specs=embedding_specs,
        gene_statuses=gene_statuses,
        expr_matrix_cm=expr_matrix_cm,
        n_bins=int(args.n_bins),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        out_tables_dir=tables_dir,
    )
    if not scores_df.empty:
        scores_df["donor_id"] = donor_star

    # Add QC correlations per row for audit (f = expr > 0).
    expr_by_gene: dict[str, np.ndarray] = {}
    qc_rows = []
    for status in gene_statuses:
        if not status.present or status.gene_idx is None:
            continue
        expr = get_feature_vector(expr_matrix_cm, int(status.gene_idx))
        expr_by_gene[status.gene] = np.asarray(expr, dtype=float)
        f = (np.asarray(expr, dtype=float) > 0.0).astype(float)
        rho_counts = _safe_spearman(f, qc_total_counts)
        rho_mt = _safe_spearman(f, qc_pct_mt_obs)
        rho_ribo = _safe_spearman(f, qc_pct_ribo)
        qc_rows.append(
            {
                "gene": status.gene,
                "rho_counts": rho_counts,
                "rho_mt": rho_mt,
                "rho_ribo": rho_ribo,
            }
        )
    qc_df = pd.DataFrame(qc_rows)
    if not scores_df.empty and not qc_df.empty:
        scores_df = scores_df.merge(qc_df, on="gene", how="left")

    scores_df.to_csv(tables_dir / "per_embedding_gene_scores.csv", index=False)

    stability_df = _gene_stability_summary(scores_df)
    if not stability_df.empty and not qc_df.empty:
        qc_medians = (
            scores_df.groupby("gene", as_index=False)[
                ["rho_counts", "rho_mt", "rho_ribo"]
            ]
            .median(numeric_only=True)
            .rename(
                columns={
                    "rho_counts": "rho_counts_median",
                    "rho_mt": "rho_mt_median",
                    "rho_ribo": "rho_ribo_median",
                }
            )
        )
        stability_df = stability_df.merge(qc_medians, on="gene", how="left")
        stability_df["qc_risk"] = np.nanmax(
            np.abs(
                stability_df[
                    ["rho_counts_median", "rho_mt_median", "rho_ribo_median"]
                ].to_numpy(dtype=float)
            ),
            axis=1,
        )

    stability_df.to_csv(tables_dir / "gene_stability_summary.csv", index=False)
    donor_summary.to_csv(tables_dir / "donor_choice.csv", index=False)

    # Overview plots.
    rep_coords = embedding_map[representative_key].coords
    tnnt2_expr = expr_by_gene.get("TNNT2", None)
    label_counts_donor = labels_donor.value_counts(dropna=False)
    _plot_overview(
        out_dir=plots_dir / "00_overview",
        donor_counts=donor_summary,
        donor_star=donor_star,
        label_counts_donor=label_counts_donor,
        rep_coords=rep_coords,
        rep_key=representative_key,
        tnnt2_expr=tnnt2_expr,
        qc_total_counts=(
            np.asarray(qc_total_counts, dtype=float)
            if qc_total_counts is not None
            else None
        ),
        qc_pct_mt=(
            np.asarray(qc_pct_mt_obs, dtype=float)
            if qc_pct_mt_obs is not None
            else None
        ),
        qc_pct_ribo=(
            np.asarray(qc_pct_ribo, dtype=float) if qc_pct_ribo is not None else None
        ),
    )

    # Save representative categorical map for CM subset labels (optional context).
    try:
        plot_categorical_umap(
            rep_coords,
            labels=labels_donor.loc[cm_mask_donor].reset_index(drop=True),
            title=f"donor_star={donor_star} cardiomyocyte labels on {representative_key}",
            outpath=plots_dir / "00_overview" / "rep_embedding_labels.png",
            annotate_cluster_medians=False,
        )
    except Exception as exc:
        _save_placeholder(
            plots_dir / "00_overview" / "rep_embedding_labels.png",
            "Representative labels",
            f"Skipped due to plotting error: {exc}",
        )

    # Gene-level panels.
    genes_present = [
        g.gene for g in gene_statuses if g.present and g.gene_idx is not None
    ]
    _plot_gene_panels(
        out_dir=plots_dir / "01_per_gene_panels",
        genes_present=genes_present,
        scores_df=scores_df,
        embedding_map=embedding_map,
        expr_by_gene=expr_by_gene,
        rep_cache=rep_cache,
        n_bins=int(args.n_bins),
    )

    # Stability plots.
    _plot_embedding_stability(
        out_dir=plots_dir / "02_embedding_stability",
        scores_df=scores_df,
        stability_df=stability_df,
    )
    _plot_direction_stability(
        out_dir=plots_dir / "03_direction_stability",
        scores_df=scores_df,
        stability_df=stability_df,
    )
    _plot_qc_controls(
        out_dir=plots_dir / "04_qc_controls",
        rep_coords=rep_coords,
        rep_key=representative_key,
        qc_total_counts=(
            np.asarray(qc_total_counts, dtype=float)
            if qc_total_counts is not None
            else None
        ),
        qc_pct_mt=(
            np.asarray(qc_pct_mt_obs, dtype=float)
            if qc_pct_mt_obs is not None
            else None
        ),
        qc_pct_ribo=(
            np.asarray(qc_pct_ribo, dtype=float) if qc_pct_ribo is not None else None
        ),
        scores_df=scores_df,
    )

    # README metadata.
    cm_labels_included = sorted(labels_all.loc[cm_mask_all].unique().tolist())
    cm_label_counts = labels_all.loc[cm_mask_all].value_counts().to_dict()
    _write_readme(
        out_path=out_root / "README.txt",
        seed=int(args.seed),
        n_perm=int(args.n_perm),
        n_bins=int(args.n_bins),
        k_pca=int(args.k_pca),
        donor_key=donor_key,
        label_key=label_key,
        donor_star=donor_star,
        cm_labels_included=cm_labels_included,
        cm_label_counts={str(k): int(v) for k, v in cm_label_counts.items()},
        expr_source=expr_source,
        expr_warning=bool(expr_warning),
        embedding_prep_note=f"{embed_prep_note}; n_pcs_used={n_pcs_used}",
        n_embeddings=int(len(embedding_specs)),
        n_cells_donor=int(adata_donor.n_obs),
        n_cells_cm=int(adata_cm.n_obs),
        cm_underpowered=cm_underpowered,
        qc_sources=qc_sources,
        stability_df=stability_df,
    )

    # Required output existence check.
    required_paths = [
        tables_dir / "donor_choice.csv",
        tables_dir / "gene_panel_status.csv",
        tables_dir / "per_embedding_gene_scores.csv",
        tables_dir / "gene_stability_summary.csv",
        out_root / "README.txt",
        plots_dir / "00_overview",
        plots_dir / "01_per_gene_panels",
        plots_dir / "02_embedding_stability",
        plots_dir / "03_direction_stability",
        plots_dir / "04_qc_controls",
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise RuntimeError("Missing required outputs: " + ", ".join(missing))

    print(f"donor_key_used={donor_key}")
    print(f"label_key_used={label_key}")
    print(f"donor_star={donor_star}")
    print(f"expression_source_used={expr_source}")
    print(f"embedding_count={len(embedding_specs)}")
    print(f"cm_labels_included={json.dumps(cm_labels_included)}")
    print(f"results_root={out_root}")

    if not stability_df.empty:
        robust = stability_df.loc[stability_df["robust_axis_like"]]
        print(f"robust_axis_like_genes={int(len(robust))}")
        if len(robust) > 0:
            print(
                "robust_axis_like_gene_list="
                + ",".join(robust["gene"].astype(str).tolist())
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
