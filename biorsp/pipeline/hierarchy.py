"""Heart case-study orchestration: global -> mega -> cluster scopes."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

# Use a non-interactive backend for reproducible headless runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from biorsp._version import __version__
from biorsp.core.compute import compute_rsp
from biorsp.core.features import get_feature_vector, resolve_feature_index
from biorsp.core.geometry import bin_theta, compute_theta, compute_vantage_point
from biorsp.core.types import NullConfig, RSPConfig
from biorsp.pipeline.io import write_json
from biorsp.plotting.classification import plot_classification_suite
from biorsp.plotting.qc import (
    plot_categorical_umap,
    save_numeric_umap,
    write_cluster_celltype_counts,
)
from biorsp.plotting.rsp import plot_rsp_to_file, plot_umap_rsp_pair
from biorsp.plotting.styles import apply_plot_style, plot_style_dict
from biorsp.plotting.utils import sanitize_feature_label
from biorsp.stats.moran import extract_weights, morans_i
from biorsp.stats.permutation import perm_null_T_and_profile
from biorsp.stats.scoring import (
    bh_fdr,
    coverage_from_null,
    evaluate_underpowered,
    peak_count,
    qc_metrics,
    robust_z,
)

DONOR_CANDIDATES = [
    "hubmap_id",
    "donor",
    "sample",
    "orig.ident",
    "batch",
    "patient",
    "donor_id",
    "sample_id",
]
CELLTYPE_CANDIDATES = ["azimuth_label", "predicted_label"]
CLUSTER_CANDIDATES = ["azimuth_id", "predicted_CLID"]

MARKER_PANEL: dict[str, list[str]] = {
    "cardiomyocyte": ["TNNT2", "TTN", "MYH6", "MYH7", "ACTC1", "PLN", "RYR2"],
    "fibroblast_ecm": ["COL1A1", "COL1A2", "DCN", "LUM"],
    "endothelial": ["PECAM1", "VWF", "KDR"],
    "pericyte_smooth_muscle": ["RGS5", "ACTA2", "TAGLN"],
    "immune": ["PTPRC", "LST1", "LYZ"],
}
PANEL_GENE_ORDER = [gene for genes in MARKER_PANEL.values() for gene in genes]
PANEL_GENE_TO_GROUP = {
    gene: group for group, genes in MARKER_PANEL.items() for gene in genes
}

PANEL_MIN_FOUND = 12
CANONICAL_PAIR_LIMIT = 10
QC_LIKE_THRESHOLD = 0.35
STABILITY_SEED_OFFSETS = [0, 1, 2]
STABILITY_BIN_OFFSETS = [-12, 0, 12]
UNDERPOWERED_PREV = 0.005
UNDERPOWERED_MIN_FG = 50
UNDERPOWERED_MIN_FG_PER_DONOR = 10
UNDERPOWERED_MIN_BG_PER_DONOR = 10
UNDERPOWERED_D_EFF_MIN = 3
UNDERPOWERED_MIN_PERM = 200
Q_SIG = 0.05


def _classify_gene_row(
    *,
    q_t: float,
    prevalence: float,
    peaks_k: float,
    qc_driven: bool,
    underpowered: bool,
) -> str:
    if underpowered:
        return "Underpowered"
    if np.isfinite(float(q_t)) and float(q_t) > Q_SIG and float(prevalence) >= 0.6:
        return "Ubiquitous (non-localized)"
    if np.isfinite(float(q_t)) and float(q_t) <= Q_SIG:
        if qc_driven:
            return "QC-driven"
        if np.isfinite(float(peaks_k)) and float(peaks_k) >= 2.0:
            return "Localized–multimodal"
        return "Localized–unimodal"
    return "Uncertain"


@dataclass(frozen=True)
class ResolvedFeature:
    requested_gene: str
    gene: str
    gene_idx: int
    panel_group: str
    auto_gene: bool
    symbol_column: str | None
    resolution_source: str


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_obs_key(
    adata: ad.AnnData,
    provided: str | None,
    candidates: list[str],
) -> str | None:
    if provided is not None:
        return str(provided) if provided in adata.obs.columns else None
    for candidate in candidates:
        if candidate in adata.obs.columns:
            return str(candidate)
    return None


def _resolve_expr_matrix(
    adata: ad.AnnData,
    layer: str | None,
    use_raw: bool,
) -> tuple[Any, Any, str]:
    if layer is not None and use_raw:
        raise ValueError("Use either layer or use_raw, not both.")
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True requested but adata.raw is missing.")
        return adata.raw.X, adata.raw, "raw"
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")
        return adata.layers[layer], adata, f"layer:{layer}"
    return adata.X, adata, "X"


def _ensure_umap(
    adata: ad.AnnData,
    *,
    seed: int,
    recompute_if_missing: bool,
) -> None:
    if "X_umap" in adata.obsm:
        xy = np.asarray(adata.obsm["X_umap"], dtype=float)
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError("adata.obsm['X_umap'] must have shape (N, 2+).")
        if xy.shape[1] > 2:
            adata.obsm["X_umap"] = xy[:, :2]
        return
    if not recompute_if_missing:
        raise ValueError(
            "Missing adata.obsm['X_umap']; case study does not recompute UMAP by default. "
            "Set recompute_umap_if_missing=True to allow recomputation."
        )
    import scanpy as sc

    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata, random_state=int(seed))


def _safe_numeric_obs(adata: ad.AnnData, key: str | None) -> np.ndarray | None:
    if key is None or key not in adata.obs.columns:
        return None
    values = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(values).sum() == 0:
        return None
    if not np.isfinite(values).all():
        fill = float(np.nanmedian(values))
        values = np.where(np.isfinite(values), values, fill)
    return values


def _total_counts_vector(adata: ad.AnnData, expr_matrix: Any) -> np.ndarray:
    obs_counts = _safe_numeric_obs(adata, "total_counts")
    if obs_counts is not None:
        return obs_counts
    fallback = np.asarray(expr_matrix.sum(axis=1)).ravel().astype(float)
    return fallback


def _pct_mt_vector(
    adata: ad.AnnData, expr_matrix: Any, adata_like: Any
) -> tuple[np.ndarray, str]:
    obs_pct_mt = _safe_numeric_obs(adata, "pct_counts_mt")
    if obs_pct_mt is not None:
        return obs_pct_mt, "obs:pct_counts_mt"

    mt_mask = None
    if hasattr(adata_like, "var") and adata_like.var is not None:
        if "mt" in adata_like.var.columns:
            mt_mask = adata_like.var["mt"].astype(bool).to_numpy()
        else:
            for symbol_col in ["hugo_symbol", "gene_name", "gene_symbol"]:
                if symbol_col in adata_like.var.columns:
                    symbols = (
                        adata_like.var[symbol_col]
                        .astype("string")
                        .fillna("")
                        .astype(str)
                        .str.upper()
                    )
                    mt_mask = symbols.str.startswith("MT-").to_numpy()
                    break

    if mt_mask is None or int(np.sum(mt_mask)) == 0:
        return np.zeros(int(adata.n_obs), dtype=float), "proxy:zeros"

    total = np.asarray(expr_matrix.sum(axis=1)).ravel().astype(float)
    mt = np.asarray(expr_matrix[:, mt_mask].sum(axis=1)).ravel().astype(float)
    pct = np.divide(mt, np.maximum(total, 1e-12)) * 100.0
    return pct, "computed:mt_fraction"


def _ribosomal_fraction_vector(
    expr_matrix: Any,
    adata_like: Any,
    total_counts: np.ndarray,
) -> tuple[np.ndarray, str]:
    ribo_mask = None
    if hasattr(adata_like, "var") and adata_like.var is not None:
        for symbol_col in ["hugo_symbol", "gene_name", "gene_symbol"]:
            if symbol_col in adata_like.var.columns:
                symbols = (
                    adata_like.var[symbol_col]
                    .astype("string")
                    .fillna("")
                    .astype(str)
                    .str.upper()
                )
                ribo_mask = (
                    symbols.str.startswith("RPL") | symbols.str.startswith("RPS")
                ).to_numpy()
                break

    if ribo_mask is None or int(np.sum(ribo_mask)) == 0:
        return np.asarray(total_counts, dtype=float), "proxy:total_counts"

    ribo_counts = (
        np.asarray(expr_matrix[:, ribo_mask].sum(axis=1)).ravel().astype(float)
    )
    ribo_frac = np.divide(ribo_counts, np.maximum(total_counts, 1e-12))
    return ribo_frac, "computed:ribo_fraction"


def _top_quantile_mask(values: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=bool)
    threshold = float(np.nanquantile(x, quantile))
    mask = x >= threshold
    if int(mask.sum()) in {0, int(mask.size)}:
        idx = np.argsort(x, kind="mergesort")
        k = max(1, int(round(0.1 * x.size)))
        keep = idx[-k:]
        mask = np.zeros(x.size, dtype=bool)
        mask[keep] = True
    return mask


def _library_size_quantile_strata(total_counts: np.ndarray) -> np.ndarray:
    x = np.asarray(total_counts, dtype=float)
    ranked = pd.Series(x).rank(method="first")
    strata = pd.qcut(ranked, q=4, labels=["lib_q1", "lib_q2", "lib_q3", "lib_q4"])
    return strata.astype("string").fillna("lib_q1").astype(str).to_numpy()


def _resolve_inference_strata(
    adata_scope: ad.AnnData,
    donor_key: str | None,
    total_counts: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    events: list[dict[str, str]] = []
    if donor_key is not None and donor_key in adata_scope.obs.columns:
        donor_ids = (
            adata_scope.obs[donor_key]
            .astype("string")
            .fillna("NA")
            .astype(str)
            .to_numpy()
        )
        if np.unique(donor_ids).size >= 2:
            return donor_ids, {
                "mode": "donor",
                "donor_key": donor_key,
                "inference_limited": False,
                "events": events,
            }

    strata = _library_size_quantile_strata(total_counts)
    events.append(
        {
            "kind": "limited_inference",
            "detail": "Donor replication unavailable; using library-size quantile strata.",
        }
    )
    return strata, {
        "mode": "library_quantile",
        "donor_key": donor_key,
        "inference_limited": True,
        "events": events,
    }


def _display_label_from_index(
    adata_like: Any, gene_idx: int, symbol_col: str | None
) -> str:
    label = str(pd.Index(adata_like.var_names)[int(gene_idx)])
    if symbol_col is not None and symbol_col in adata_like.var.columns:
        symbol = str(adata_like.var.iloc[int(gene_idx)][symbol_col]).strip()
        if symbol:
            return symbol
    return label


def _resolve_marker_panel(
    adata_like: Any,
    expr_matrix: Any,
) -> tuple[list[ResolvedFeature], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    resolved: list[ResolvedFeature] = []
    used_idx: set[int] = set()

    for gene in PANEL_GENE_ORDER:
        panel_group = PANEL_GENE_TO_GROUP[gene]
        try:
            idx, label, symbol_col, source = resolve_feature_index(adata_like, gene)
            idx = int(idx)
            if idx in used_idx:
                rows.append(
                    {
                        "requested_gene": gene,
                        "panel_group": panel_group,
                        "found": False,
                        "gene": "",
                        "gene_idx": "",
                        "auto_gene": False,
                        "symbol_column": symbol_col or "",
                        "resolution_source": source,
                        "status": "duplicate_gene_index",
                    }
                )
                continue
            used_idx.add(idx)
            resolved.append(
                ResolvedFeature(
                    requested_gene=gene,
                    gene=str(label),
                    gene_idx=idx,
                    panel_group=panel_group,
                    auto_gene=False,
                    symbol_column=symbol_col,
                    resolution_source=source,
                )
            )
            rows.append(
                {
                    "requested_gene": gene,
                    "panel_group": panel_group,
                    "found": True,
                    "gene": str(label),
                    "gene_idx": idx,
                    "auto_gene": False,
                    "symbol_column": symbol_col or "",
                    "resolution_source": source,
                    "status": "marker_found",
                }
            )
        except KeyError:
            rows.append(
                {
                    "requested_gene": gene,
                    "panel_group": panel_group,
                    "found": False,
                    "gene": "",
                    "gene_idx": "",
                    "auto_gene": False,
                    "symbol_column": "",
                    "resolution_source": "",
                    "status": "marker_missing",
                }
            )

    n_found = sum(1 for item in resolved if not item.auto_gene)
    if n_found < PANEL_MIN_FOUND:
        n_auto = PANEL_MIN_FOUND - n_found
        mean_expr = np.asarray(expr_matrix.mean(axis=0)).ravel().astype(float)
        symbol_col = None
        for candidate in ["hugo_symbol", "gene_name", "gene_symbol"]:
            if hasattr(adata_like, "var") and candidate in adata_like.var.columns:
                symbol_col = candidate
                break
        top_idx = np.argsort(-mean_expr)
        for idx in top_idx:
            gene_idx = int(idx)
            if gene_idx in used_idx:
                continue
            label = _display_label_from_index(adata_like, gene_idx, symbol_col)
            used_idx.add(gene_idx)
            resolved.append(
                ResolvedFeature(
                    requested_gene=label,
                    gene=label,
                    gene_idx=gene_idx,
                    panel_group="auto",
                    auto_gene=True,
                    symbol_column=symbol_col,
                    resolution_source="auto_top_expression",
                )
            )
            rows.append(
                {
                    "requested_gene": label,
                    "panel_group": "auto",
                    "found": True,
                    "gene": label,
                    "gene_idx": gene_idx,
                    "auto_gene": True,
                    "symbol_column": symbol_col or "",
                    "resolution_source": "auto_top_expression",
                    "status": "auto_gene",
                }
            )
            if len([item for item in resolved if item.auto_gene]) >= n_auto:
                break

    marker_table = pd.DataFrame(rows)
    return resolved, marker_table


def _moran_pair(
    expr: np.ndarray,
    weights: Any | None,
) -> tuple[float, float]:
    if weights is None:
        return float("nan"), float("nan")

    cont = float("nan")
    binary = float("nan")
    try:
        if np.nanstd(expr) > 0:
            cont = float(morans_i(expr, weights))
    except Exception:
        cont = float("nan")

    detected = (np.asarray(expr, dtype=float) > 0.0).astype(float)
    try:
        if np.nanstd(detected) > 0:
            binary = float(morans_i(detected, weights))
    except Exception:
        binary = float("nan")

    return cont, binary


def _score_feature(
    *,
    gene_label: str,
    requested_gene: str,
    panel_group: str,
    auto_gene: bool,
    gene_idx: int,
    expr: np.ndarray,
    umap_xy: np.ndarray,
    center_xy: np.ndarray,
    theta: np.ndarray,
    bin_id: np.ndarray,
    bin_counts_total: np.ndarray,
    rsp_cfg: RSPConfig,
    null_cfg: NullConfig,
    inference_strata: np.ndarray,
    obs_df: pd.DataFrame,
    weights: Any | None,
    control_tag: bool,
    seed_offset: int,
) -> tuple[dict[str, Any], Any]:
    result = compute_rsp(
        expr=expr,
        embedding_xy=umap_xy,
        config=rsp_cfg,
        center_xy=center_xy,
        feature_label=gene_label,
        feature_index=gene_idx,
    )

    perm = perm_null_T_and_profile(
        expr=expr,
        theta=theta,
        donor_ids=inference_strata,
        n_bins=int(rsp_cfg.bins),
        n_perm=int(null_cfg.n_perm),
        seed=int(null_cfg.seed + seed_offset),
        donor_stratified=True,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )

    prevalence = float(np.mean(np.asarray(expr, dtype=float) > 0.0))
    n_fg = int(np.sum(np.asarray(expr, dtype=float) > 0.0))
    n_cells = int(np.asarray(expr, dtype=float).size)
    power = evaluate_underpowered(
        donor_ids=np.asarray(inference_strata),
        f=np.asarray(expr, dtype=float) > 0.0,
        n_perm=int(null_cfg.n_perm),
        p_min=UNDERPOWERED_PREV,
        min_fg_total=UNDERPOWERED_MIN_FG,
        min_fg_per_donor=UNDERPOWERED_MIN_FG_PER_DONOR,
        min_bg_per_donor=UNDERPOWERED_MIN_BG_PER_DONOR,
        d_eff_min=UNDERPOWERED_D_EFF_MIN,
        min_perm=UNDERPOWERED_MIN_PERM,
    )
    underpowered = bool(power["underpowered"])

    e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
    null_e = np.asarray(perm["null_E_phi"], dtype=float)
    null_t = np.asarray(perm["null_T"], dtype=float)
    t_obs = float(perm["T_obs"])
    z_t = float(robust_z(t_obs, null_t))
    coverage_c = float(coverage_from_null(np.abs(e_obs), np.abs(null_e), q=0.95))
    peaks_k = int(peak_count(np.abs(e_obs), np.abs(null_e), smooth_w=3, q_prom=0.95))

    moran_cont, moran_bin = _moran_pair(np.asarray(expr, dtype=float), weights)
    qc = qc_metrics(
        np.asarray(expr, dtype=float),
        obs_df,
        {
            "total_counts": ["total_counts", "n_counts"],
            "pct_counts_mt": ["pct_counts_mt"],
            "pct_counts_ribo": ["pct_counts_ribo"],
        },
    )
    qc_risk = float(qc.get("qc_risk", 0.0))
    if not np.isfinite(qc_risk):
        qc_risk = 0.0
    qc_like = bool(control_tag or qc_risk >= QC_LIKE_THRESHOLD)

    row = {
        "requested_gene": requested_gene,
        "gene": gene_label,
        "gene_idx": int(gene_idx),
        "panel_group": panel_group,
        "auto_gene": bool(auto_gene),
        "prevalence": prevalence,
        "n_fg": n_fg,
        "n_fg_total": int(power["n_fg_total"]),
        "n_bg_total": int(power["n_bg_total"]),
        "D_eff": int(power["D_eff"]),
        "donor_fg_min": float(power["donor_fg_min"]),
        "donor_fg_med": float(power["donor_fg_med"]),
        "donor_fg_max": float(power["donor_fg_max"]),
        "n_cells": n_cells,
        "underpowered_flag": underpowered,
        "anisotropy": float(result.anisotropy),
        "peak_direction_rad": float(result.peak_direction),
        "T_obs": t_obs,
        "Z_T": z_t,
        "coverage_C": coverage_c,
        "peaks_K": peaks_k,
        "score_1": z_t,
        "score_2": coverage_c,
        "p_T": float(perm["p_T"]),
        "n_perm": int(perm["n_perm_used"]),
        "moran_continuous": moran_cont,
        "moran_binary": moran_bin,
        "qc_risk": qc_risk,
        "qc_like_flag": qc_like,
        "control_tag": bool(control_tag),
    }
    if "warning" in perm:
        row["perm_warning"] = str(perm["warning"])

    return row, result


def _plot_sanity_distribution(
    *,
    values: np.ndarray,
    observed: float,
    out_png: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(values, bins=25, color="#D8A7A7", edgecolor="white")
    ax.axvline(observed, color="#8B0000", linestyle="--", linewidth=2)
    ax.set_xlabel("anisotropy")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_qc_mimicry(gene_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if gene_df.empty:
        ax.text(0.5, 0.5, "No rows", ha="center", va="center")
        ax.axis("off")
    else:
        x = pd.to_numeric(gene_df["qc_risk"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(gene_df["score_1"], errors="coerce").to_numpy(dtype=float)
        c = np.where(
            gene_df["control_tag"].astype(bool).to_numpy(), "#d62728", "#1f77b4"
        )
        ax.scatter(x, y, s=24, alpha=0.8, c=c, linewidths=0)
        ax.axvline(QC_LIKE_THRESHOLD, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("qc_risk")
        ax.set_ylabel("score_1 (Z_T)")
        ax.set_title("QC mimicry diagnostic")
        ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _run_within_strata_sanity_check(
    *,
    expr: np.ndarray,
    umap_xy: np.ndarray,
    center_xy: np.ndarray,
    rsp_cfg: RSPConfig,
    strata: np.ndarray,
    seed: int,
    n_perm: int,
) -> dict[str, Any]:
    x = np.asarray(expr, dtype=float)
    observed = compute_rsp(
        expr=x,
        embedding_xy=umap_xy,
        config=rsp_cfg,
        center_xy=center_xy,
        feature_label="sanity",
    )

    rng = np.random.default_rng(int(seed))
    labels = np.asarray(strata)
    perm_values = np.zeros(int(n_perm), dtype=float)
    uniq = np.unique(labels)

    for i in range(int(n_perm)):
        permuted = x.copy()
        for group in uniq:
            idx = np.flatnonzero(labels == group)
            if idx.size <= 1:
                continue
            permuted[idx] = permuted[idx[rng.permutation(idx.size)]]
        perm_result = compute_rsp(
            expr=permuted,
            embedding_xy=umap_xy,
            config=rsp_cfg,
            center_xy=center_xy,
            feature_label="sanity_perm",
        )
        perm_values[i] = float(perm_result.anisotropy)

    observed_value = float(observed.anisotropy)
    p_like = float(
        (1.0 + np.sum(perm_values >= observed_value)) / (1.0 + perm_values.size)
    )

    return {
        "observed_anisotropy": observed_value,
        "perm_median": float(np.median(perm_values)),
        "perm_mean": float(np.mean(perm_values)),
        "attenuation_ratio": float(np.median(perm_values) / max(observed_value, 1e-12)),
        "p_perm_like": p_like,
        "perm_values": perm_values,
    }


def _compute_stability_table(
    *,
    features: list[ResolvedFeature],
    expr_matrix: Any,
    umap_xy: np.ndarray,
    center_xy: np.ndarray,
    bins: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = [item for item in features if not item.auto_gene][:5]
    if not selected:
        selected = features[:5]

    for feature in selected:
        expr = get_feature_vector(expr_matrix, feature.gene_idx)
        for seed_offset in STABILITY_SEED_OFFSETS:
            for bin_offset in STABILITY_BIN_OFFSETS:
                n_bins = max(24, int(bins + bin_offset))
                cfg = RSPConfig(
                    basis="X_umap",
                    bins=n_bins,
                    center_method="median",
                    threshold=0.0,
                )
                out = compute_rsp(
                    expr=expr,
                    embedding_xy=umap_xy,
                    config=cfg,
                    center_xy=center_xy,
                    feature_label=feature.gene,
                    feature_index=feature.gene_idx,
                )
                rows.append(
                    {
                        "gene": feature.gene,
                        "gene_idx": feature.gene_idx,
                        "bins": n_bins,
                        "seed": int(seed + seed_offset),
                        "anisotropy": float(out.anisotropy),
                        "peak_direction_rad": float(out.peak_direction),
                    }
                )

    return pd.DataFrame(rows)


def _plot_stability_table(stability_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    if stability_df.empty:
        ax.text(
            0.5, 0.5, "No stability diagnostics available", ha="center", va="center"
        )
        ax.axis("off")
    else:
        stable = stability_df.copy()
        stable["config"] = stable.apply(
            lambda row: f"b{int(row['bins'])}/s{int(row['seed'])}",
            axis=1,
        )
        for gene, sub in stable.groupby("gene", observed=False):
            ax.plot(
                sub["config"],
                sub["anisotropy"],
                marker="o",
                linewidth=1.2,
                label=str(gene),
            )
        ax.set_ylabel("anisotropy")
        ax.set_title("F5 stability diagnostic (not biological validation)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _mega_split_by_cluster_centroids(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, int], pd.DataFrame]:
    umap_xy = np.asarray(adata.obsm["X_umap"], dtype=float)[:, :2]
    clusters = adata.obs[cluster_key].astype("string").fillna("NA").astype(str)
    centroids = (
        pd.DataFrame(
            {"cluster": clusters.to_numpy(), "x": umap_xy[:, 0], "y": umap_xy[:, 1]}
        )
        .groupby("cluster", observed=False, as_index=False)
        .median()
        .sort_values("cluster", kind="mergesort")
    )
    if centroids.shape[0] < 2:
        raise ValueError("Need at least two clusters to build mega split.")

    km = KMeans(n_clusters=2, random_state=int(seed), n_init=10)
    labels = km.fit_predict(centroids[["x", "y"]].to_numpy(dtype=float)).astype(int)
    centroids["mega_id"] = labels
    mapping = {str(r["cluster"]): int(r["mega_id"]) for _, r in centroids.iterrows()}

    cluster_values = clusters.to_numpy(dtype=str)
    assign = np.array([mapping.get(c, -1) for c in cluster_values], dtype=int)
    return assign, mapping, centroids


def _scope_dirs(scope_outdir: Path) -> dict[str, Path]:
    paths = {
        "root": scope_outdir,
        "plots_rsp": scope_outdir / "plots" / "rsp",
        "plots_pairs": scope_outdir / "plots" / "pairs",
        "plots_qc": scope_outdir / "plots" / "qc",
        "plots_meta": scope_outdir / "plots" / "meta",
        "tables": scope_outdir / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _run_scope(
    *,
    adata_subset: ad.AnnData,
    scope_outdir: Path,
    scope_name: str,
    scope_kind: str,
    seed: int,
    bins: int,
    n_perm: int,
    layer: str | None,
    use_raw: bool,
    donor_key: str | None,
    celltype_key: str | None,
    cluster_key: str | None,
    canonical_pair_limit: int,
    recompute_umap_if_missing: bool,
) -> dict[str, Any]:
    dirs = _scope_dirs(scope_outdir)

    adata_scope = adata_subset
    _ensure_umap(
        adata_scope,
        seed=int(seed),
        recompute_if_missing=bool(recompute_umap_if_missing),
    )
    umap_xy = np.asarray(adata_scope.obsm["X_umap"], dtype=float)[:, :2]
    center = compute_vantage_point(umap_xy, method="median")
    theta = compute_theta(umap_xy, center)
    _, bin_id = bin_theta(theta, int(bins))
    bin_counts_total = np.bincount(bin_id, minlength=int(bins)).astype(float)

    expr_matrix, adata_like, expr_source = _resolve_expr_matrix(
        adata_scope,
        layer=layer,
        use_raw=use_raw,
    )

    total_counts = _total_counts_vector(adata_scope, expr_matrix)
    pct_mt, pct_mt_source = _pct_mt_vector(adata_scope, expr_matrix, adata_like)
    ribo_fraction, ribo_source = _ribosomal_fraction_vector(
        expr_matrix, adata_like, total_counts
    )

    inference_strata, inference_meta = _resolve_inference_strata(
        adata_scope,
        donor_key=donor_key,
        total_counts=total_counts,
    )

    save_numeric_umap(
        umap_xy=umap_xy,
        values=total_counts,
        out_png=dirs["plots_qc"] / "f2_total_counts_umap.png",
        title="F2 total_counts on UMAP",
        cmap="viridis",
        colorbar_label="total_counts",
        vantage_point=(float(center[0]), float(center[1])),
    )
    save_numeric_umap(
        umap_xy=umap_xy,
        values=pct_mt,
        out_png=dirs["plots_qc"] / "f2_pct_counts_mt_umap.png",
        title="F2 pct_counts_mt on UMAP",
        cmap="magma",
        colorbar_label="pct_counts_mt",
        vantage_point=(float(center[0]), float(center[1])),
    )

    if celltype_key is not None and celltype_key in adata_scope.obs.columns:
        plot_categorical_umap(
            umap_xy=umap_xy,
            labels=adata_scope.obs[celltype_key],
            title=f"F3 UMAP by {celltype_key}",
            outpath=dirs["plots_meta"] / "f3_umap_celltype.png",
            vantage_point=(float(center[0]), float(center[1])),
            annotate_cluster_medians=False,
        )

    cluster_celltype_counts_path = ""
    if cluster_key is not None and cluster_key in adata_scope.obs.columns:
        plot_categorical_umap(
            umap_xy=umap_xy,
            labels=adata_scope.obs[cluster_key],
            title=f"F3 UMAP by {cluster_key}",
            outpath=dirs["plots_meta"] / "f3_umap_cluster.png",
            vantage_point=(float(center[0]), float(center[1])),
            annotate_cluster_medians=True,
        )
        if celltype_key is not None and celltype_key in adata_scope.obs.columns:
            counts_csv = dirs["tables"] / "cluster_celltype_counts.csv"
            write_cluster_celltype_counts(
                cluster_labels=adata_scope.obs[cluster_key],
                celltype_labels=adata_scope.obs[celltype_key],
                cluster_key=cluster_key,
                celltype_key=celltype_key,
                out_csv=counts_csv,
            )
            cluster_celltype_counts_path = counts_csv.as_posix()

    features, marker_table = _resolve_marker_panel(adata_like, expr_matrix)
    marker_table_path = dirs["tables"] / "marker_panel_found_missing.csv"
    marker_table.to_csv(marker_table_path, index=False)

    weights = None
    try:
        weights = extract_weights(adata_scope)
    except Exception:
        weights = None

    rsp_cfg = RSPConfig(
        basis="X_umap", bins=int(bins), center_method="median", threshold=0.0
    )
    null_cfg = NullConfig(n_perm=int(n_perm), seed=int(seed), donor_stratified=True)

    marker_for_pairs = [item for item in features if not item.auto_gene][
        :canonical_pair_limit
    ]
    marker_for_pairs_set = {item.gene_idx for item in marker_for_pairs}

    rows: list[dict[str, Any]] = []
    for feature in features:
        expr = get_feature_vector(expr_matrix, feature.gene_idx)
        try:
            row, result = _score_feature(
                gene_label=feature.gene,
                requested_gene=feature.requested_gene,
                panel_group=feature.panel_group,
                auto_gene=feature.auto_gene,
                gene_idx=feature.gene_idx,
                expr=expr,
                umap_xy=umap_xy,
                center_xy=center,
                theta=theta,
                bin_id=bin_id,
                bin_counts_total=bin_counts_total,
                rsp_cfg=rsp_cfg,
                null_cfg=null_cfg,
                inference_strata=inference_strata,
                obs_df=adata_scope.obs,
                weights=weights,
                control_tag=False,
                seed_offset=int(feature.gene_idx),
            )
            row["scope"] = scope_name

            safe = sanitize_feature_label(feature.gene, max_len=64)
            rsp_png = dirs["plots_rsp"] / f"rsp_{safe}.png"
            plot_rsp_to_file(result, rsp_png, title=f"RSP: {feature.gene}")

            if feature.gene_idx in marker_for_pairs_set:
                pair_png = dirs["plots_pairs"] / f"f4_pair_{safe}.png"
                plot_umap_rsp_pair(
                    embedding_xy=umap_xy,
                    expr=expr,
                    result=result,
                    out_png=pair_png,
                    title_prefix="F4",
                )
        except ValueError as exc:
            row = {
                "requested_gene": feature.requested_gene,
                "gene": feature.gene,
                "gene_idx": int(feature.gene_idx),
                "panel_group": feature.panel_group,
                "auto_gene": bool(feature.auto_gene),
                "prevalence": float(np.mean(np.asarray(expr, dtype=float) > 0.0)),
                "anisotropy": float("nan"),
                "peak_direction_rad": float("nan"),
                "p_T": float("nan"),
                "n_perm": int(n_perm),
                "moran_continuous": float("nan"),
                "moran_binary": float("nan"),
                "qc_risk": float("nan"),
                "qc_like_flag": False,
                "control_tag": False,
                "scope": scope_name,
                "error": str(exc),
            }
        rows.append(row)

    control_vectors = [
        (
            "pct_counts_mt_high",
            _top_quantile_mask(pct_mt).astype(float),
            "negative_control",
            True,
        )
    ]
    if ribo_source == "proxy:total_counts":
        control_vectors.append(
            (
                "total_counts_high_proxy",
                _top_quantile_mask(total_counts).astype(float),
                "negative_control",
                True,
            )
        )

    rng_controls = np.random.default_rng(int(seed) + 9000)
    shuffle_source_expr: np.ndarray
    if features:
        shuffle_source_expr = get_feature_vector(expr_matrix, features[0].gene_idx)
    else:
        shuffle_source_expr = np.asarray(total_counts, dtype=float)
    for i in range(2):
        control_vectors.append(
            (
                f"fake_shuffle_expr_{i+1}",
                rng_controls.permutation(np.asarray(shuffle_source_expr, dtype=float)),
                "negative_control",
                True,
            )
        )
    if ribo_source != "proxy:total_counts":
        control_vectors.append(
            (
                "ribo_fraction_high",
                _top_quantile_mask(ribo_fraction).astype(float),
                "negative_control",
                True,
            )
        )

    for i, (name, ctrl_expr, panel_group, control_tag) in enumerate(control_vectors):
        try:
            row, result = _score_feature(
                gene_label=name,
                requested_gene=name,
                panel_group=panel_group,
                auto_gene=True,
                gene_idx=-1000 - i,
                expr=ctrl_expr,
                umap_xy=umap_xy,
                center_xy=center,
                theta=theta,
                bin_id=bin_id,
                bin_counts_total=bin_counts_total,
                rsp_cfg=rsp_cfg,
                null_cfg=null_cfg,
                inference_strata=inference_strata,
                obs_df=adata_scope.obs,
                weights=weights,
                control_tag=control_tag,
                seed_offset=5000 + i,
            )
            row["scope"] = scope_name
            safe = sanitize_feature_label(name, max_len=64)
            pair_png = dirs["plots_pairs"] / f"f5_control_pair_{safe}.png"
            plot_umap_rsp_pair(
                embedding_xy=umap_xy,
                expr=ctrl_expr,
                result=result,
                out_png=pair_png,
                title_prefix="F5",
            )
        except ValueError as exc:
            row = {
                "requested_gene": name,
                "gene": name,
                "gene_idx": -1000 - i,
                "panel_group": panel_group,
                "auto_gene": True,
                "prevalence": float(np.mean(np.asarray(ctrl_expr, dtype=float) > 0.0)),
                "anisotropy": float("nan"),
                "peak_direction_rad": float("nan"),
                "p_T": float("nan"),
                "n_perm": int(n_perm),
                "moran_continuous": float("nan"),
                "moran_binary": float("nan"),
                "qc_risk": float("nan"),
                "qc_like_flag": True,
                "control_tag": bool(control_tag),
                "scope": scope_name,
                "error": str(exc),
            }
        rows.append(row)

    gene_df = pd.DataFrame(rows)
    if not gene_df.empty:
        gene_df["q_T"] = bh_fdr(
            pd.to_numeric(gene_df["p_T"], errors="coerce").to_numpy(dtype=float)
        )
        gene_df["qc_driven"] = gene_df["qc_like_flag"].astype(bool)
        gene_df["class_label"] = [
            _classify_gene_row(
                q_t=float(q),
                prevalence=float(prev),
                peaks_k=float(peaks),
                qc_driven=bool(qc_driven),
                underpowered=bool(underpowered),
            )
            for q, prev, peaks, qc, qc_driven, underpowered in zip(
                gene_df["q_T"],
                gene_df["prevalence"],
                gene_df["peaks_K"],
                gene_df["qc_risk"],
                gene_df["qc_driven"],
                gene_df["underpowered_flag"],
                strict=False,
            )
        ]
    gene_df = gene_df.sort_values(
        by=["control_tag", "auto_gene", "p_T", "gene"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    gene_summary_path = dirs["tables"] / "gene_summary.csv"
    gene_df.to_csv(gene_summary_path, index=False)
    gene_scores_path = dirs["tables"] / "gene_scores.csv"
    score_cols = [
        "scope",
        "gene",
        "requested_gene",
        "panel_group",
        "prevalence",
        "p_T",
        "q_T",
        "Z_T",
        "score_1",
        "coverage_C",
        "score_2",
        "peaks_K",
        "class_label",
        "qc_risk",
        "qc_like_flag",
        "qc_driven",
        "underpowered_flag",
        "control_tag",
        "moran_continuous",
        "moran_binary",
    ]
    score_cols = [c for c in score_cols if c in gene_df.columns]
    gene_df[score_cols].to_csv(gene_scores_path, index=False)

    cls_df = gene_df.loc[~gene_df["control_tag"].astype(bool)].copy()
    if cls_df.empty:
        cls_df = gene_df.copy()
    plot_classification_suite(
        cls_df,
        dirs["plots_meta"],
        z_strong_threshold=4.0,
        coverage_strong_threshold=0.15,
    )
    _plot_qc_mimicry(gene_df, dirs["plots_qc"] / "qc_mimicry.png")

    sanity_target_feature = next(
        (item for item in features if not item.auto_gene), None
    )
    sanity_rows: list[dict[str, Any]] = []
    if sanity_target_feature is not None:
        expr = get_feature_vector(expr_matrix, sanity_target_feature.gene_idx)
        sanity = _run_within_strata_sanity_check(
            expr=expr,
            umap_xy=umap_xy,
            center_xy=center,
            rsp_cfg=rsp_cfg,
            strata=inference_strata,
            seed=int(seed + 700),
            n_perm=min(100, max(30, int(n_perm))),
        )
        sanity_rows.append(
            {
                "scope": scope_name,
                "feature": sanity_target_feature.gene,
                "stratification_mode": inference_meta["mode"],
                "observed_anisotropy": sanity["observed_anisotropy"],
                "perm_median": sanity["perm_median"],
                "perm_mean": sanity["perm_mean"],
                "attenuation_ratio": sanity["attenuation_ratio"],
                "p_perm_like": sanity["p_perm_like"],
            }
        )
        _plot_sanity_distribution(
            values=np.asarray(sanity["perm_values"], dtype=float),
            observed=float(sanity["observed_anisotropy"]),
            out_png=dirs["plots_qc"] / "f5_sanity_permutation.png",
            title=f"F5 sanity permutation ({sanity_target_feature.gene})",
        )

    stability_df = _compute_stability_table(
        features=features,
        expr_matrix=expr_matrix,
        umap_xy=umap_xy,
        center_xy=center,
        bins=int(bins),
        seed=int(seed),
    )
    stability_path = dirs["tables"] / "stability_diagnostics.csv"
    stability_df.to_csv(stability_path, index=False)
    _plot_stability_table(
        stability_df, dirs["plots_qc"] / "f5_stability_diagnostic.png"
    )

    sanity_df = pd.DataFrame(sanity_rows)
    sanity_path = dirs["tables"] / "negative_control_sanity.csv"
    sanity_df.to_csv(sanity_path, index=False)

    symbol_cols = marker_table.loc[
        marker_table["found"].astype(bool)
        & (marker_table["symbol_column"].astype(str) != ""),
        "symbol_column",
    ]
    symbol_column_used = ""
    if not symbol_cols.empty:
        symbol_column_used = str(symbol_cols.value_counts().idxmax())

    metadata = {
        "timestamp_utc": _now_utc_iso(),
        "scope_name": scope_name,
        "scope_kind": scope_kind,
        "status": "PASS",
        "n_cells": int(adata_scope.n_obs),
        "n_genes_scored": int(gene_df.shape[0]),
        "expression_source": expr_source,
        "vantage_point": {
            "x0": float(center[0]),
            "y0": float(center[1]),
            "method": "median",
        },
        "theta_convention": {
            "zero_direction": "east (+UMAP1)",
            "rotation": "counterclockwise",
            "north_angle_deg": 90,
        },
        "scientific_rationale": {
            "representation_conditional": True,
            "primary_goal": (
                "Heart case study for method validation: quantify directional enrichment for known "
                "heart markers and QC confounds while controlling obvious confounding."
            ),
            "discovery_claims": "No biological discovery claims are made by this workflow.",
        },
        "inference": inference_meta,
        "controls": {
            "pct_counts_mt_source": pct_mt_source,
            "ribo_control_source": ribo_source,
            "qc_like_threshold": QC_LIKE_THRESHOLD,
        },
        "gene_resolution_column_used": symbol_column_used,
        "paths": {
            "gene_summary_csv": gene_summary_path.as_posix(),
            "gene_scores_csv": gene_scores_path.as_posix(),
            "marker_panel_csv": marker_table_path.as_posix(),
            "stability_csv": stability_path.as_posix(),
            "negative_control_sanity_csv": sanity_path.as_posix(),
            "cluster_celltype_counts_csv": cluster_celltype_counts_path,
        },
    }
    write_json(dirs["root"] / "metadata.json", metadata)

    return {
        "scope_name": scope_name,
        "scope_kind": scope_kind,
        "status": "PASS",
        "n_cells": int(adata_scope.n_obs),
        "vantage_point": {"x0": float(center[0]), "y0": float(center[1])},
        "gene_summary_csv": gene_summary_path.as_posix(),
        "gene_scores_csv": gene_scores_path.as_posix(),
        "marker_panel_csv": marker_table_path.as_posix(),
        "stability_csv": stability_path.as_posix(),
        "negative_control_sanity_csv": sanity_path.as_posix(),
        "cluster_celltype_counts_csv": cluster_celltype_counts_path,
        "metadata_path": (dirs["root"] / "metadata.json").as_posix(),
    }


def _write_runlog(
    *,
    outdir: Path,
    adata: ad.AnnData,
    summary: dict[str, Any],
    warnings: list[str],
) -> None:
    lines = [
        "# Heart Case Study Run Log",
        "",
        f"- timestamp_utc: {_now_utc_iso()}",
        f"- status: {summary.get('status', 'unknown')}",
        f"- n_cells: {int(adata.n_obs)}",
        f"- n_genes: {int(adata.n_vars)}",
        f"- donor_key: {summary['keys'].get('donor_key')}",
        f"- cluster_key: {summary['keys'].get('cluster_key')}",
        f"- celltype_key: {summary['keys'].get('celltype_key')}",
        "",
        "## Scientific Framing",
        "",
        "This run is a case study / method validation. Directionality is interpreted only in the embedding coordinate frame.",
        "No subtype discovery claims are made.",
        "",
        "## Warnings",
    ]
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")

    runlog_path = outdir / "logs" / "runlog.md"
    runlog_path.parent.mkdir(parents=True, exist_ok=True)
    runlog_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case_study(
    *,
    adata: ad.AnnData,
    outdir: str | Path,
    do_hierarchy: bool,
    layer: str | None = None,
    use_raw: bool = False,
    bins: int = 72,
    n_perm: int = 200,
    seed: int = 0,
    min_cells_per_cluster: int = 300,
    min_cells_per_mega: int = 500,
    donor_key: str | None = None,
    celltype_key: str | None = None,
    cluster_key: str | None = None,
    recompute_umap_if_missing: bool = False,
) -> dict[str, Any]:
    if int(bins) <= 0:
        raise ValueError("bins must be positive")
    if int(n_perm) <= 0:
        raise ValueError("n_perm must be positive")

    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    adata_work = adata
    _ensure_umap(
        adata_work,
        seed=int(seed),
        recompute_if_missing=bool(recompute_umap_if_missing),
    )

    donor_key_resolved = _resolve_obs_key(adata_work, donor_key, DONOR_CANDIDATES)
    celltype_key_resolved = _resolve_obs_key(
        adata_work, celltype_key, CELLTYPE_CANDIDATES
    )
    cluster_key_resolved = _resolve_obs_key(adata_work, cluster_key, CLUSTER_CANDIDATES)

    hierarchy_root = root / "hierarchy"
    hierarchy_root.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    global_result = _run_scope(
        adata_subset=adata_work,
        scope_outdir=hierarchy_root / "global",
        scope_name="global",
        scope_kind="global",
        seed=int(seed),
        bins=int(bins),
        n_perm=int(n_perm),
        layer=layer,
        use_raw=bool(use_raw),
        donor_key=donor_key_resolved,
        celltype_key=celltype_key_resolved,
        cluster_key=cluster_key_resolved,
        canonical_pair_limit=CANONICAL_PAIR_LIMIT,
        recompute_umap_if_missing=bool(recompute_umap_if_missing),
    )

    if donor_key_resolved is None:
        warnings.append("donor_key missing; inference limited to library-size strata.")

    summary: dict[str, Any] = {
        "status": "PASS",
        "timestamp_utc": _now_utc_iso(),
        "global": global_result,
        "keys": {
            "donor_key": donor_key_resolved,
            "celltype_key": celltype_key_resolved,
            "cluster_key": cluster_key_resolved,
        },
        "warnings": warnings,
    }

    aggregated_gene_paths = [Path(global_result["gene_summary_csv"])]
    aggregated_gene_score_paths = [Path(global_result["gene_scores_csv"])]

    if do_hierarchy:
        if cluster_key_resolved is None:
            raise RuntimeError(
                "Hierarchy requires a cluster key (for this dataset, use azimuth_id)."
            )

        assign, mapping, centroids = _mega_split_by_cluster_centroids(
            adata_work,
            cluster_key=cluster_key_resolved,
            seed=int(seed),
        )

        mega_root = hierarchy_root / "mega"
        mega_root.mkdir(parents=True, exist_ok=True)
        mapping_csv = mega_root / "cluster_to_mega.csv"
        pd.DataFrame(
            [
                {"cluster": cluster_name, "mega_id": mega_id}
                for cluster_name, mega_id in sorted(mapping.items(), key=lambda x: x[0])
            ]
        ).to_csv(mapping_csv, index=False)
        centroids_csv = mega_root / "cluster_centroids.csv"
        centroids.to_csv(centroids_csv, index=False)

        umap_xy = np.asarray(adata_work.obsm["X_umap"], dtype=float)[:, :2]
        center = compute_vantage_point(umap_xy, method="median")
        mega_labels = pd.Series(
            np.where(assign == 0, "mega0", "mega1"),
            index=adata_work.obs.index,
            dtype="string",
        )
        plot_categorical_umap(
            umap_xy=umap_xy,
            labels=mega_labels,
            title="F3 mega split overlay",
            outpath=root / "plots" / "meta" / "f3_mega_split_overlay.png",
            vantage_point=(float(center[0]), float(center[1])),
            annotate_cluster_medians=False,
        )

        mega_results: dict[str, Any] = {}
        for mega_id in [0, 1]:
            mask = assign == mega_id
            n_cells = int(mask.sum())
            if n_cells < int(min_cells_per_mega):
                mega_results[f"mega{mega_id}"] = {
                    "status": "SKIP",
                    "n_cells": n_cells,
                    "reason": f"below min_cells_per_mega={min_cells_per_mega}",
                }
                continue
            result = _run_scope(
                adata_subset=adata_work[mask],
                scope_outdir=mega_root / f"mega{mega_id}",
                scope_name=f"mega{mega_id}",
                scope_kind="mega",
                seed=int(seed),
                bins=int(bins),
                n_perm=int(n_perm),
                layer=layer,
                use_raw=bool(use_raw),
                donor_key=donor_key_resolved,
                celltype_key=celltype_key_resolved,
                cluster_key=cluster_key_resolved,
                canonical_pair_limit=CANONICAL_PAIR_LIMIT,
                recompute_umap_if_missing=bool(recompute_umap_if_missing),
            )
            mega_results[f"mega{mega_id}"] = result
            aggregated_gene_paths.append(Path(result["gene_summary_csv"]))
            aggregated_gene_score_paths.append(Path(result["gene_scores_csv"]))

        clusters_root = hierarchy_root / "clusters"
        clusters_root.mkdir(parents=True, exist_ok=True)
        cluster_values = (
            adata_work.obs[cluster_key_resolved]
            .astype("string")
            .fillna("NA")
            .astype(str)
        )
        cluster_results: dict[str, Any] = {}
        skipped_clusters: list[dict[str, Any]] = []
        for cluster_id in sorted(cluster_values.unique().tolist()):
            mask = cluster_values == cluster_id
            n_cells = int(mask.sum())
            if n_cells < int(min_cells_per_cluster):
                skipped_clusters.append(
                    {
                        "cluster_id": str(cluster_id),
                        "n_cells": n_cells,
                        "reason": f"below min_cells_per_cluster={min_cells_per_cluster}",
                    }
                )
                continue

            result = _run_scope(
                adata_subset=adata_work[mask.to_numpy()],
                scope_outdir=clusters_root / str(cluster_id),
                scope_name=str(cluster_id),
                scope_kind="cluster",
                seed=int(seed),
                bins=int(bins),
                n_perm=int(n_perm),
                layer=layer,
                use_raw=bool(use_raw),
                donor_key=donor_key_resolved,
                celltype_key=celltype_key_resolved,
                cluster_key=cluster_key_resolved,
                canonical_pair_limit=CANONICAL_PAIR_LIMIT,
                recompute_umap_if_missing=bool(recompute_umap_if_missing),
            )
            cluster_results[str(cluster_id)] = result
            aggregated_gene_paths.append(Path(result["gene_summary_csv"]))
            aggregated_gene_score_paths.append(Path(result["gene_scores_csv"]))

        summary["mega"] = mega_results
        summary["clusters"] = {
            "processed_count": len(cluster_results),
            "skipped": skipped_clusters,
            "results": cluster_results,
        }
        summary["mega_split"] = {
            "mapping_csv": mapping_csv.as_posix(),
            "centroids_csv": centroids_csv.as_posix(),
        }

    hierarchy_summary_path = hierarchy_root / "hierarchy_summary.json"
    write_json(hierarchy_summary_path, summary)

    tables_root = root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    plots_root = root / "plots"
    for sub in ["qc", "meta", "pairs", "rsp"]:
        (plots_root / sub).mkdir(parents=True, exist_ok=True)

    gene_tables = [pd.read_csv(path) for path in aggregated_gene_paths if path.exists()]
    if gene_tables:
        all_genes = pd.concat(gene_tables, ignore_index=True)
        all_genes.to_csv(tables_root / "gene_summary.csv", index=False)
    gene_score_tables = [
        pd.read_csv(path) for path in aggregated_gene_score_paths if path.exists()
    ]
    if gene_score_tables:
        all_gene_scores = pd.concat(gene_score_tables, ignore_index=True)
        all_gene_scores.to_csv(tables_root / "gene_scores.csv", index=False)

    global_marker_csv = Path(global_result["marker_panel_csv"])
    if global_marker_csv.exists():
        pd.read_csv(global_marker_csv).to_csv(
            tables_root / "marker_panel_found_missing.csv", index=False
        )

    global_cluster_counts = Path(global_result.get("cluster_celltype_counts_csv", ""))
    if global_cluster_counts.exists():
        pd.read_csv(global_cluster_counts).to_csv(
            tables_root / "cluster_celltype_counts.csv", index=False
        )

    for fig_name in [
        "f2_total_counts_umap.png",
        "f2_pct_counts_mt_umap.png",
        "f5_sanity_permutation.png",
        "f5_stability_diagnostic.png",
        "qc_mimicry.png",
    ]:
        src = hierarchy_root / "global" / "plots" / "qc" / fig_name
        if src.exists():
            target = plots_root / "qc" / fig_name
            target.write_bytes(src.read_bytes())

    for fig_name in ["f3_umap_celltype.png", "f3_umap_cluster.png"]:
        src = hierarchy_root / "global" / "plots" / "meta" / fig_name
        if src.exists():
            target = plots_root / "meta" / fig_name
            target.write_bytes(src.read_bytes())
    for fig_name in [
        "score1_score2_scatter.png",
        "classification_scatter.png",
        "class_counts.png",
    ]:
        src = hierarchy_root / "global" / "plots" / "meta" / fig_name
        if src.exists():
            target = plots_root / "meta" / fig_name
            target.write_bytes(src.read_bytes())

    for src in sorted(
        (hierarchy_root / "global" / "plots" / "pairs").glob("f4_pair_*.png")
    ):
        (plots_root / "pairs" / src.name).write_bytes(src.read_bytes())

    for src in sorted((hierarchy_root / "global" / "plots" / "rsp").glob("rsp_*.png")):
        (plots_root / "rsp" / src.name).write_bytes(src.read_bytes())

    root_metadata = {
        "timestamp_utc": _now_utc_iso(),
        "status": summary["status"],
        "biorsp_version": __version__,
        "python_version": platform.python_version(),
        "versions": {
            "scanpy": importlib_metadata.version("scanpy"),
            "anndata": importlib_metadata.version("anndata"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "parameters": {
            "do_hierarchy": bool(do_hierarchy),
            "layer": layer,
            "use_raw": bool(use_raw),
            "bins": int(bins),
            "n_perm": int(n_perm),
            "seed": int(seed),
            "min_cells_per_cluster": int(min_cells_per_cluster),
            "min_cells_per_mega": int(min_cells_per_mega),
            "donor_key": donor_key,
            "celltype_key": celltype_key,
            "cluster_key": cluster_key,
        },
        "keys_resolved": summary["keys"],
        "theta_convention": {
            "zero_direction": "east (+UMAP1)",
            "rotation": "counterclockwise",
            "north_angle_deg": 90,
        },
        "scientific_rationale": {
            "representation_conditional": True,
            "goal": (
                "Method validation in a heart case study for directional enrichment patterns in "
                "known biology and QC confounds."
            ),
            "not_goal": "Subtype discovery.",
        },
        "plot_style": plot_style_dict(),
        "paths": {
            "hierarchy_summary_json": hierarchy_summary_path.as_posix(),
            "gene_summary_csv": (tables_root / "gene_summary.csv").as_posix(),
            "gene_scores_csv": (tables_root / "gene_scores.csv").as_posix(),
            "marker_panel_csv": (
                tables_root / "marker_panel_found_missing.csv"
            ).as_posix(),
            "cluster_celltype_counts_csv": (
                tables_root / "cluster_celltype_counts.csv"
            ).as_posix(),
        },
        "warnings": warnings,
    }

    checklist_rows = [
        {
            "hypothesis": "individual_gene_scoring_classification",
            "required_output": "tables/gene_scores.csv",
            "path": (tables_root / "gene_scores.csv").as_posix(),
            "exists": (tables_root / "gene_scores.csv").exists(),
        },
        {
            "hypothesis": "score_space_with_classes",
            "required_output": "plots/meta/score1_score2_scatter.png",
            "path": (plots_root / "meta" / "score1_score2_scatter.png").as_posix(),
            "exists": (plots_root / "meta" / "score1_score2_scatter.png").exists(),
        },
        {
            "hypothesis": "class_count_plot",
            "required_output": "plots/meta/class_counts.png",
            "path": (plots_root / "meta" / "class_counts.png").as_posix(),
            "exists": (plots_root / "meta" / "class_counts.png").exists(),
        },
        {
            "hypothesis": "qc_mimicry_control",
            "required_output": "plots/qc/qc_mimicry.png",
            "path": (plots_root / "qc" / "qc_mimicry.png").as_posix(),
            "exists": (plots_root / "qc" / "qc_mimicry.png").exists(),
        },
    ]
    pd.DataFrame(checklist_rows).to_csv(
        tables_root / "results_ready_checklist.csv", index=False
    )

    write_json(root / "metadata.json", root_metadata)
    _write_runlog(outdir=root, adata=adata_work, summary=summary, warnings=warnings)
    return summary
