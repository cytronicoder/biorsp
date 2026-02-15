#!/usr/bin/env python3
"""Experiment #2: within-cell-type gradients without subclustering."""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

# Headless backend for deterministic batch execution.
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Allow running directly via `python experiments/...` from repo root.
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
    "leiden",
    "cluster",
]

FIBRO_PANEL = [
    "COL1A1",
    "COL1A2",
    "COL3A1",
    "DCN",
    "LUM",
    "FBLN5",
    "CXCL12",
    "CXCL14",
    "PDGFRA",
    "POSTN",
]

MYELOID_PANEL = [
    "LST1",
    "LYZ",
    "TYROBP",
    "AIF1",
    "FCGR3A",
    "S100A8",
    "S100A9",
    "IL1B",
    "C1QA",
    "C1QB",
    "C1QC",
    "VCAN",
]

HOUSEKEEPING_PANEL = ["ACTB", "GAPDH", "RPLP0", "B2M"]
QC_PSEUDOGENES = ["total_counts", "pct_counts_mt", "pct_counts_ribo"]

PANEL_PROVENANCE = {
    "fibroblast_program": (
        "Pre-registered fibroblast panel from Experiment #2 plan: ECM + activation/remodeling genes."
    ),
    "myeloid_program": (
        "Pre-registered myeloid panel from Experiment #2 plan: identity + inflammatory/chemokine programs."
    ),
    "housekeeping_control": (
        "Pre-registered housekeeping-ish controls from Experiment #2 plan."
    ),
}

GENE_CATEGORY = {
    # Fibro panel categories.
    "COL1A1": "ECM",
    "COL1A2": "ECM",
    "COL3A1": "ECM",
    "DCN": "ECM",
    "LUM": "ECM",
    "FBLN5": "ECM",
    "CXCL12": "activation",
    "CXCL14": "activation",
    "PDGFRA": "identity",
    "POSTN": "activation",
    # Myeloid panel categories.
    "LST1": "identity",
    "LYZ": "identity",
    "TYROBP": "identity",
    "AIF1": "identity",
    "FCGR3A": "identity",
    "S100A8": "activation",
    "S100A9": "activation",
    "IL1B": "activation",
    "C1QA": "macrophage_state",
    "C1QB": "macrophage_state",
    "C1QC": "macrophage_state",
    "VCAN": "activation",
    # Housekeeping.
    "ACTB": "housekeeping",
    "GAPDH": "housekeeping",
    "RPLP0": "housekeeping",
    "B2M": "housekeeping",
}

CLASS_ORDER = [
    "Underpowered",
    "Ubiquitous (non-localized)",
    "Localized–unimodal",
    "Localized–multimodal",
    "QC-driven",
    "Uncertain",
]

CLASS_COLORS = {
    "Underpowered": "#8A8A8A",
    "Ubiquitous (non-localized)": "#2CA02C",
    "Localized–unimodal": "#1F77B4",
    "Localized–multimodal": "#FF7F0E",
    "QC-driven": "#D62728",
    "Uncertain": "#9467BD",
}

CATEGORY_MARKERS = {
    "ECM": "o",
    "identity": "s",
    "activation": "^",
    "macrophage_state": "D",
    "housekeeping": "P",
    "other": "X",
}

MODE_ORDER = ["binary", "donor_quantile"]
MODE_TITLES = {"binary": "A: binary detection", "donor_quantile": "B: donor quantile"}

Q_SIG = 0.05
QC_RISK_THRESH = 0.35
QC_PROFILE_SIM_THRESH = 0.70
UBIQUITOUS_PREV = 0.60
PREV_MIN = 0.005
MIN_FG = 50
Z_STRONG = 4.0
COVERAGE_STRONG = 0.15


@dataclass(frozen=True)
class ResolvedGene:
    gene: str
    category: str
    panel_group: str
    found: bool
    gene_idx: int | None
    resolved_gene: str
    status: str
    resolution_source: str
    symbol_column: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment #2: quantify within-type directional localization modes without subclustering."
        )
    )
    p.add_argument(
        "--h5ad", default="data/processed/HT_pca_umap.h5ad", help="Input .h5ad path."
    )
    p.add_argument(
        "--out",
        default="experiments/heart_smoketest/results/experiment2_within_type_gradients",
        help="Output root directory.",
    )
    p.add_argument("--seed", type=int, default=0, help="Global random seed.")
    p.add_argument(
        "--n_perm", type=int, default=300, help="Permutation count (default smoke=300)."
    )
    p.add_argument("--n_bins", type=int, default=64, help="Angular bins.")
    p.add_argument(
        "--q",
        type=float,
        default=0.10,
        help="Foreground quantile for donor_quantile mode.",
    )
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
    p.add_argument(
        "--label_key", default=None, help="Optional cell-type label key override."
    )
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
            raise KeyError(f"Requested embedding '{requested_key}' not in adata.obsm.")
        key = str(requested_key)
    else:
        key = "X_umap" if "X_umap" in adata.obsm else str(next(iter(adata.obsm.keys())))
    xy = np.asarray(adata.obsm[key], dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Embedding '{key}' must have shape (N, 2+).")
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
    ids = adata.obs[key].astype("string").fillna("NA").astype(str).to_numpy()
    if np.unique(ids).size < 2:
        return None, key
    return ids, key


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
    pct_ribo = (
        np.divide(ribo_counts, np.maximum(np.asarray(total_counts, dtype=float), 1e-12))
        * 100.0
    )
    return pct_ribo, f"computed:{symbol_col}"


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


def _build_subset_masks(
    labels: pd.Series,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    labels_str = labels.astype("string").fillna("NA").astype(str)
    lower = labels_str.str.lower()

    fibro_mask = lower.str.contains("fibro", regex=False)
    myeloid_mask = (
        lower.str.contains("myeloid", regex=False)
        | lower.str.contains("macrophage", regex=False)
        | lower.str.contains("mono", regex=False)
        | lower.str.contains("dendritic", regex=False)
    )

    masks = {
        "fibroblast": fibro_mask.to_numpy(dtype=bool),
        "myeloid": myeloid_mask.to_numpy(dtype=bool),
    }
    matched_labels = {
        "fibroblast": sorted(labels_str[fibro_mask].unique().tolist()),
        "myeloid": sorted(labels_str[myeloid_mask].unique().tolist()),
    }
    return masks, matched_labels


def _resolve_panel(
    adata_like: Any, subset_name: str
) -> tuple[list[ResolvedGene], pd.DataFrame]:
    if subset_name == "fibroblast":
        core_genes = FIBRO_PANEL
        core_group = "fibroblast_program"
    elif subset_name == "myeloid":
        core_genes = MYELOID_PANEL
        core_group = "myeloid_program"
    else:
        raise ValueError(f"Unknown subset_name: {subset_name}")

    requested: list[tuple[str, str]] = []
    requested.extend([(g, core_group) for g in core_genes])
    requested.extend([(g, "housekeeping_control") for g in HOUSEKEEPING_PANEL])

    rows: list[dict[str, Any]] = []
    resolved: list[ResolvedGene] = []
    used_idx: set[int] = set()

    for gene, panel_group in requested:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                idx, label, symbol_col, source = resolve_feature_index(adata_like, gene)
            idx_i = int(idx)
            if idx_i in used_idx:
                item = ResolvedGene(
                    gene=gene,
                    category=GENE_CATEGORY.get(gene, "other"),
                    panel_group=panel_group,
                    found=False,
                    gene_idx=None,
                    resolved_gene="",
                    status="duplicate_index",
                    resolution_source=source,
                    symbol_column=symbol_col or "",
                )
            else:
                used_idx.add(idx_i)
                item = ResolvedGene(
                    gene=gene,
                    category=GENE_CATEGORY.get(gene, "other"),
                    panel_group=panel_group,
                    found=True,
                    gene_idx=idx_i,
                    resolved_gene=str(label),
                    status="resolved",
                    resolution_source=source,
                    symbol_column=symbol_col or "",
                )
        except KeyError:
            item = ResolvedGene(
                gene=gene,
                category=GENE_CATEGORY.get(gene, "other"),
                panel_group=panel_group,
                found=False,
                gene_idx=None,
                resolved_gene="",
                status="missing",
                resolution_source="",
                symbol_column="",
            )

        rows.append(
            {
                "gene": item.gene,
                "category": item.category,
                "panel_group": item.panel_group,
                "provenance": PANEL_PROVENANCE[item.panel_group],
                "status": item.status,
                "found": item.found,
                "resolved_gene": item.resolved_gene,
                "gene_idx": item.gene_idx if item.gene_idx is not None else "",
                "resolution_source": item.resolution_source,
                "symbol_column": item.symbol_column,
            }
        )
        resolved.append(item)
    return resolved, pd.DataFrame(rows)


def _build_foreground_donor_quantile(
    expr: np.ndarray,
    donor_ids: np.ndarray | None,
    q: float,
) -> tuple[np.ndarray, str]:
    x = np.asarray(expr, dtype=float).ravel()
    qf = float(q)
    if not np.isfinite(qf) or qf <= 0.0 or qf >= 1.0:
        raise ValueError("q must be in (0,1).")
    f = np.zeros(x.size, dtype=bool)
    if donor_ids is None:
        thr = float(np.quantile(x, 1.0 - qf))
        f = x >= thr
        return f, "global"

    d = np.asarray(donor_ids).astype(str)
    for donor in np.unique(d):
        idx = np.flatnonzero(d == donor)
        if idx.size == 0:
            continue
        thr = float(np.quantile(x[idx], 1.0 - qf))
        f[idx] = x[idx] >= thr
    return f, "donor"


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
    if np.nanmax(np.abs(w)) <= 1e-12:
        return np.zeros(int(n_bins), dtype=float)
    w_clean = np.where(np.isfinite(w), w, 0.0)
    if np.nanmin(w_clean) < 0:
        w_clean = w_clean - float(np.nanmin(w_clean))
    w_sum = float(np.sum(w_clean))
    if w_sum <= 1e-12:
        return np.zeros(int(n_bins), dtype=float)
    w_bin = np.bincount(bin_id, weights=w_clean, minlength=int(n_bins)).astype(float)
    p_w = w_bin / w_sum
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
    if x.size == 0:
        raise ValueError("weights must be non-empty.")
    e_obs = _compute_continuous_profile(
        x,
        n_bins=int(n_bins),
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )
    t_obs = float(np.max(np.abs(e_obs)))

    used_donor_stratified = False
    warning_msg = ""
    donor_valid = False
    if donor_ids is not None:
        dd = np.asarray(donor_ids).astype(str)
        donor_valid = np.unique(dd).size >= 2
        if donor_valid:
            used_donor_stratified = True
        else:
            warning_msg = (
                "continuous permutation fallback: <2 donors, using global shuffle."
            )
    else:
        warning_msg = (
            "continuous permutation fallback: donor_ids missing, using global shuffle."
        )

    rng = np.random.default_rng(int(seed))
    null_e = np.zeros((int(n_perm), int(n_bins)), dtype=float)
    null_t = np.zeros(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        if used_donor_stratified and donor_valid and donor_ids is not None:
            x_perm = _permute_weights_within_donor(
                x, np.asarray(donor_ids).astype(str), rng
            )
        else:
            x_perm = x[rng.permutation(x.size)]
        e_perm = _compute_continuous_profile(
            x_perm,
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
        "T_obs": t_obs,
        "null_T": null_t,
        "p_T": p_t,
        "used_donor_stratified": bool(used_donor_stratified),
    }
    if warning_msg:
        out["warning"] = warning_msg
    return out


def _classify_row(row: pd.Series) -> str:
    q_t = float(row["q_T"]) if np.isfinite(float(row["q_T"])) else 1.0
    prev = float(row["prev"])
    peaks = int(row["peaks_K"]) if np.isfinite(float(row["peaks_K"])) else 0
    qc_risk = float(row["qc_risk"]) if np.isfinite(float(row["qc_risk"])) else 0.0
    qc_ps = (
        float(row["qc_profile_similarity"])
        if np.isfinite(float(row["qc_profile_similarity"]))
        else 0.0
    )
    qc_driven = bool(
        ((qc_risk >= QC_RISK_THRESH) or (qc_ps >= QC_PROFILE_SIM_THRESH))
        and (q_t <= Q_SIG)
    )

    if bool(row["underpowered"]):
        return "Underpowered"
    if q_t > Q_SIG and prev >= UBIQUITOUS_PREV:
        return "Ubiquitous (non-localized)"
    if q_t <= Q_SIG and qc_driven:
        return "QC-driven"
    if q_t <= Q_SIG and peaks == 1 and not qc_driven:
        return "Localized–unimodal"
    if q_t <= Q_SIG and peaks >= 2 and not qc_driven:
        return "Localized–multimodal"
    return "Uncertain"


def _score_gene_mode(
    *,
    subset_name: str,
    gene: str,
    category: str,
    panel_group: str,
    expr: np.ndarray,
    foreground_mode: str,
    q: float,
    donor_ids: np.ndarray | None,
    donor_key_used: str | None,
    theta: np.ndarray,
    n_bins: int,
    n_perm: int,
    seed: int,
    bin_id: np.ndarray,
    bin_counts_total: np.ndarray,
    total_counts: np.ndarray,
    pct_counts_mt: np.ndarray | None,
    pct_counts_ribo: np.ndarray | None,
    qc_profiles: dict[str, np.ndarray],
    subset_underpowered: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    x = np.asarray(expr, dtype=float).ravel()
    if foreground_mode == "binary":
        f = x > 0.0
        quantile_scope = "binary"
    elif foreground_mode == "donor_quantile":
        f, quantile_scope = _build_foreground_donor_quantile(x, donor_ids, q=q)
    else:
        raise ValueError(f"Unsupported foreground_mode: {foreground_mode}")

    if int(f.sum()) in {0, f.size}:
        e_obs = np.zeros(int(n_bins), dtype=float)
    else:
        e_obs, _, _, _ = compute_rsp_profile_from_boolean(
            f,
            theta,
            int(n_bins),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )

    perm = perm_null_T_and_profile(
        expr=f.astype(float),
        theta=theta,
        donor_ids=donor_ids,
        n_bins=int(n_bins),
        n_perm=int(n_perm),
        seed=int(seed),
        donor_stratified=True,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )

    null_e = np.asarray(perm["null_E_phi"], dtype=float)
    null_t = np.asarray(perm["null_T"], dtype=float)
    e_obs_perm = np.asarray(perm["E_phi_obs"], dtype=float)
    if e_obs_perm.shape[0] == int(n_bins):
        e_obs = e_obs_perm
    t_obs = float(perm["T_obs"])

    coverage_c = float(coverage_from_null(e_obs, null_e, q=0.95))
    peaks_k = int(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95))
    z_t = float(robust_z(t_obs, null_t))

    peak_idx = int(np.argmax(np.abs(e_obs))) if e_obs.size > 0 else 0
    centers = theta_bin_centers(int(n_bins))
    peak_phi = float(centers[peak_idx]) if centers.size > 0 else 0.0

    prev = float(np.mean(f))
    n_fg = int(f.sum())
    n_cells = int(f.size)
    donors_n = int(np.unique(donor_ids).size) if donor_ids is not None else 0
    underpowered = bool(
        (prev < PREV_MIN)
        or (n_fg < MIN_FG)
        or ((donor_ids is not None) and (donors_n < 2))
    )

    rho_total = _safe_spearman(f.astype(float), total_counts)
    rho_mt = _safe_spearman(f.astype(float), pct_counts_mt)
    rho_ribo = _safe_spearman(f.astype(float), pct_counts_ribo)
    finite = [abs(v) for v in [rho_total, rho_mt, rho_ribo] if np.isfinite(v)]
    qc_risk = float(max(finite)) if finite else 0.0

    sims: list[float] = []
    for profile in qc_profiles.values():
        sim = _cosine_similarity(e_obs, profile)
        if np.isfinite(sim):
            sims.append(sim)
    qc_profile_similarity = float(max(sims)) if sims else float("nan")

    row = {
        "subset_name": subset_name,
        "gene": gene,
        "category": category,
        "panel_group": panel_group,
        "foreground_mode": foreground_mode,
        "q": float(q) if foreground_mode == "donor_quantile" else float("nan"),
        "foreground_scope": quantile_scope,
        "prev": prev,
        "n_fg": n_fg,
        "n_cells": n_cells,
        "donors_n": donors_n,
        "donor_key_used": donor_key_used if donor_key_used is not None else "",
        "T_obs": t_obs,
        "p_T": float(perm["p_T"]),
        "q_T": float("nan"),
        "Z_T": z_t,
        "coverage_C": coverage_c,
        "peaks_K": peaks_k,
        "peak_dir_phi": peak_phi,
        "peak_dir_phi_deg": float(np.degrees(peak_phi) % 360.0),
        "qc_risk": qc_risk,
        "qc_profile_similarity": qc_profile_similarity,
        "used_donor_stratified": bool(perm["used_donor_stratified"]),
        "underpowered": underpowered,
        "subset_underpowered": bool(subset_underpowered),
        "rho_total_counts": rho_total,
        "rho_pct_counts_mt": rho_mt,
        "rho_pct_counts_ribo": rho_ribo,
        "perm_warning": str(perm.get("warning", "")),
        "score_1": z_t,
        "score_2": coverage_c,
        "class_label": "",
        "qc_driven": False,
    }
    artifacts = {
        "expr": x,
        "foreground": f.astype(float),
        "E_phi_obs": e_obs,
        "null_E_phi": null_e,
        "null_T": null_t,
    }
    return row, artifacts


def _apply_fdr_and_classification(scores_df: pd.DataFrame) -> pd.DataFrame:
    if scores_df.empty:
        return scores_df
    out = scores_df.copy()
    out["q_T"] = np.nan
    for mode in MODE_ORDER:
        idx = out.index[out["foreground_mode"] == mode].to_numpy(dtype=int)
        if idx.size == 0:
            continue
        pvals = pd.to_numeric(out.loc[idx, "p_T"], errors="coerce").to_numpy(
            dtype=float
        )
        out.loc[idx, "q_T"] = bh_fdr(pvals)
    out["qc_driven"] = (
        (out["qc_risk"] >= QC_RISK_THRESH)
        | (out["qc_profile_similarity"] >= QC_PROFILE_SIM_THRESH)
    ) & (out["q_T"] <= Q_SIG)
    out["class_label"] = out.apply(_classify_row, axis=1)
    out = out.sort_values(
        by=["foreground_mode", "q_T", "gene"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def _plot_label_count_bar(
    labels: pd.Series,
    *,
    fibro_labels: list[str],
    myeloid_labels: list[str],
    out_png: Path,
) -> None:
    counts = (
        labels.astype("string")
        .fillna("NA")
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
    )
    lab = counts.index.tolist()
    vals = counts.to_numpy(dtype=float)
    colors = []
    for label in lab:
        if label in fibro_labels:
            colors.append("#2E6F95")
        elif label in myeloid_labels:
            colors.append("#C44536")
        else:
            colors.append("#BBBBBB")

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    x = np.arange(len(lab))
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Cell count")
    ax.set_title("Cell counts per label (fibro/myeloid highlighted)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    handles = [
        mlines.Line2D(
            [], [], color="#2E6F95", marker="s", linestyle="None", label="fibro labels"
        ),
        mlines.Line2D(
            [],
            [],
            color="#C44536",
            marker="s",
            linestyle="None",
            label="myeloid labels",
        ),
        mlines.Line2D(
            [], [], color="#BBBBBB", marker="s", linestyle="None", label="other labels"
        ),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)


def _plot_overview(
    *,
    adata: ad.AnnData,
    umap_xy: np.ndarray,
    label_key: str | None,
    donor_key: str | None,
    center_xy: np.ndarray,
    matched_labels: dict[str, list[str]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if label_key is not None:
        plot_categorical_umap(
            umap_xy=umap_xy,
            labels=adata.obs[label_key],
            title=f"All cells UMAP by {label_key}",
            outpath=out_dir / "umap_all_labels.png",
            vantage_point=(float(center_xy[0]), float(center_xy[1])),
            annotate_cluster_medians=False,
        )
        _plot_label_count_bar(
            adata.obs[label_key],
            fibro_labels=matched_labels["fibroblast"],
            myeloid_labels=matched_labels["myeloid"],
            out_png=out_dir / "label_counts_highlighted.png",
        )
    else:
        _save_placeholder(
            out_dir / "umap_all_labels.png",
            "All cells UMAP labels",
            "No label key available.",
        )
        _save_placeholder(
            out_dir / "label_counts_highlighted.png",
            "Label counts",
            "No label key available.",
        )

    if donor_key is not None and donor_key in adata.obs.columns:
        plot_categorical_umap(
            umap_xy=umap_xy,
            labels=adata.obs[donor_key],
            title=f"All cells UMAP by donor ({donor_key})",
            outpath=out_dir / "umap_all_donor.png",
            vantage_point=(float(center_xy[0]), float(center_xy[1])),
            annotate_cluster_medians=False,
        )
    else:
        _save_placeholder(
            out_dir / "umap_all_donor.png",
            "All cells UMAP donor",
            "Donor key unavailable.",
        )


def _plot_subset_umaps(
    *,
    subset_name: str,
    subset_xy: np.ndarray,
    subset_center: np.ndarray,
    donor_labels: pd.Series | None,
    pct_mt: np.ndarray | None,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if donor_labels is not None:
        plot_categorical_umap(
            umap_xy=subset_xy,
            labels=donor_labels,
            title=f"{subset_name}: UMAP by donor",
            outpath=out_dir / f"{subset_name}_umap_donor.png",
            vantage_point=(float(subset_center[0]), float(subset_center[1])),
            annotate_cluster_medians=False,
        )
    else:
        _save_placeholder(
            out_dir / f"{subset_name}_umap_donor.png",
            f"{subset_name}: donor UMAP",
            "Donor key unavailable.",
        )

    if pct_mt is not None:
        save_numeric_umap(
            umap_xy=subset_xy,
            values=np.asarray(pct_mt, dtype=float),
            out_png=out_dir / f"{subset_name}_umap_pct_mt.png",
            title=f"{subset_name}: pct_counts_mt",
            cmap="magma",
            colorbar_label="pct_counts_mt",
            vantage_point=(float(subset_center[0]), float(subset_center[1])),
        )
    else:
        _save_placeholder(
            out_dir / f"{subset_name}_umap_pct_mt.png",
            f"{subset_name}: pct_counts_mt UMAP",
            "pct_counts_mt unavailable.",
        )

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(
        subset_xy[:, 0],
        subset_xy[:, 1],
        s=6.0,
        c="#4C78A8",
        alpha=0.80,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        [float(subset_center[0])],
        [float(subset_center[1])],
        marker="X",
        s=80,
        c="black",
        edgecolors="white",
        linewidths=0.8,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"{subset_name}: subset UMAP")
    fig.tight_layout()
    fig.savefig(out_dir / f"{subset_name}_umap_plain.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)


def _plot_gene_panel_both_modes(
    *,
    subset_name: str,
    gene: str,
    gene_category: str,
    subset_xy: np.ndarray,
    subset_center: np.ndarray,
    rows_by_mode: dict[str, pd.Series],
    artifacts_by_mode: dict[str, dict[str, np.ndarray]],
    n_bins: int,
    out_png: Path,
) -> None:
    fig = plt.figure(figsize=(15.0, 11.0))
    for col, mode in enumerate(MODE_ORDER):
        row = rows_by_mode[mode]
        art = artifacts_by_mode[mode]
        expr = np.asarray(art["expr"], dtype=float)
        e_obs = np.asarray(art["E_phi_obs"], dtype=float)
        null_e = np.asarray(art["null_E_phi"], dtype=float)
        null_t = np.asarray(art["null_T"], dtype=float)

        # Top: feature UMAP.
        ax_u = fig.add_subplot(3, 2, 1 + col)
        expr_plot = np.log1p(np.maximum(expr, 0.0))
        order = np.argsort(expr_plot, kind="mergesort")
        ax_u.scatter(
            subset_xy[:, 0],
            subset_xy[:, 1],
            s=4.0,
            c="#D8D8D8",
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        sc = ax_u.scatter(
            subset_xy[order, 0],
            subset_xy[order, 1],
            c=expr_plot[order],
            cmap="Reds",
            s=6.0,
            alpha=0.85,
            linewidths=0,
            rasterized=True,
        )
        ax_u.scatter(
            [float(subset_center[0])],
            [float(subset_center[1])],
            marker="X",
            s=70,
            c="black",
            edgecolors="white",
            linewidths=0.8,
        )
        ax_u.set_xticks([])
        ax_u.set_yticks([])
        ax_u.set_title(f"{MODE_TITLES[mode]}: feature map")
        cbar = fig.colorbar(sc, ax=ax_u, fraction=0.040, pad=0.02)
        cbar.set_label("log1p(expr)")

        # Middle: polar profile + null envelope.
        ax_p = fig.add_subplot(3, 2, 3 + col, projection="polar")
        centers = theta_bin_centers(int(n_bins))
        theta_closed = np.concatenate([centers, centers[:1]])
        obs_closed = np.concatenate([e_obs, e_obs[:1]])
        q_hi = np.quantile(null_e, 0.95, axis=0)
        q_lo = np.quantile(null_e, 0.05, axis=0)
        q_hi_c = np.concatenate([q_hi, q_hi[:1]])
        q_lo_c = np.concatenate([q_lo, q_lo[:1]])
        ax_p.plot(
            theta_closed, obs_closed, color="#8B0000", linewidth=2.0, label="E_phi obs"
        )
        ax_p.plot(
            theta_closed,
            q_hi_c,
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            label="null 95%",
        )
        ax_p.plot(
            theta_closed,
            q_lo_c,
            color="#444444",
            linestyle="--",
            linewidth=1.0,
            label="null 5%",
        )
        ax_p.fill_between(theta_closed, q_lo_c, q_hi_c, color="#B0B0B0", alpha=0.18)
        ax_p.set_theta_zero_location("E")
        ax_p.set_theta_direction(1)
        ax_p.set_thetagrids(np.arange(0, 360, 90))
        text = (
            f"T={float(row['T_obs']):.4f}\n"
            f"Z={float(row['Z_T']):.2f}\n"
            f"q={float(row['q_T']):.2e}\n"
            f"C={float(row['coverage_C']):.3f}\n"
            f"K={int(row['peaks_K'])}\n"
            f"class={row['class_label']}"
        )
        ax_p.text(
            0.02,
            0.02,
            text,
            transform=ax_p.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.85},
        )
        if col == 0:
            ax_p.legend(
                loc="upper right", bbox_to_anchor=(1.25, 1.2), fontsize=8, frameon=True
            )
        ax_p.set_title("RSP + null envelope")

        # Bottom: null_T histogram.
        ax_h = fig.add_subplot(3, 2, 5 + col)
        bins = int(min(45, max(12, np.ceil(np.sqrt(null_t.size)))))
        ax_h.hist(null_t, bins=bins, color="#779ECB", edgecolor="white", alpha=0.90)
        ax_h.axvline(
            float(row["T_obs"]), color="#8B0000", linestyle="--", linewidth=2.0
        )
        ax_h.set_xlabel("null_T")
        ax_h.set_ylabel("count")
        ax_h.set_title("Null T distribution")

    fig.suptitle(
        f"{subset_name} | {gene} ({gene_category}) | binary vs donor-quantile foreground",
        y=1.01,
        fontsize=12,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_score_space_for_subset(
    subset_df: pd.DataFrame,
    *,
    subset_name: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if subset_df.empty:
        _save_placeholder(
            out_dir / "empty.png", f"{subset_name} score space", "No scored genes."
        )
        return

    # 1) per-mode score scatter.
    for mode in MODE_ORDER:
        sub = subset_df.loc[subset_df["foreground_mode"] == mode].copy()
        fig, ax = plt.subplots(figsize=(8.3, 6.0))
        for cls in CLASS_ORDER:
            cls_sub = sub.loc[sub["class_label"] == cls]
            if cls_sub.empty:
                continue
            for category, cat_sub in cls_sub.groupby("category", observed=False):
                ax.scatter(
                    cat_sub["score_1"].to_numpy(dtype=float),
                    cat_sub["score_2"].to_numpy(dtype=float),
                    s=90,
                    marker=CATEGORY_MARKERS.get(str(category), "X"),
                    c=CLASS_COLORS.get(cls, "#333333"),
                    edgecolors="black",
                    linewidths=0.6,
                    alpha=0.85,
                )
        top = sub.sort_values(by="Z_T", ascending=False).head(8)
        for _, row in top.iterrows():
            ax.text(
                float(row["score_1"]) + 0.06,
                float(row["score_2"]) + 0.004,
                str(row["gene"]),
                fontsize=8,
            )
        ax.set_xlabel("score_1 = Z_T")
        ax.set_ylabel("score_2 = coverage_C")
        ax.set_title(f"{subset_name}: score space ({mode})")
        ax.grid(alpha=0.25, linewidth=0.6)

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
        cat_handles = [
            mlines.Line2D(
                [],
                [],
                marker=CATEGORY_MARKERS[c],
                linestyle="None",
                color="black",
                markerfacecolor="white",
                label=c,
                markersize=8,
            )
            for c in sorted(set(sub["category"].astype(str)))
        ]
        leg1 = ax.legend(
            handles=class_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            title="class",
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=cat_handles,
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            title="category",
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / f"score1_vs_score2_scatter_{mode}.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

    # 2) mode overlay with connectors.
    piv = (
        subset_df.pivot_table(
            index=["gene", "category"],
            columns="foreground_mode",
            values=["score_1", "score_2", "class_label"],
            aggfunc="first",
        )
    ).reset_index()
    if ("score_1", "binary") in piv.columns and (
        "score_1",
        "donor_quantile",
    ) in piv.columns:
        fig2, ax2 = plt.subplots(figsize=(8.3, 6.0))
        deltas: list[tuple[float, str, float, float]] = []
        for _, row in piv.iterrows():
            x1 = float(row[("score_1", "binary")])
            y1 = float(row[("score_2", "binary")])
            x2 = float(row[("score_1", "donor_quantile")])
            y2 = float(row[("score_2", "donor_quantile")])
            cat = str(row["category"])
            gene = str(row["gene"])
            ax2.plot(
                [x1, x2], [y1, y2], color="#999999", alpha=0.7, linewidth=1.0, zorder=1
            )
            ax2.scatter(
                [x1],
                [y1],
                c="#1F77B4",
                marker=CATEGORY_MARKERS.get(cat, "X"),
                s=75,
                edgecolors="black",
            )
            ax2.scatter(
                [x2],
                [y2],
                c="#D62728",
                marker=CATEGORY_MARKERS.get(cat, "X"),
                s=75,
                edgecolors="black",
            )
            deltas.append((abs(x2 - x1), gene, x2, y2))
        deltas.sort(reverse=True, key=lambda x: x[0])
        for _, gene, x2, y2 in deltas[:8]:
            ax2.text(float(x2) + 0.05, float(y2) + 0.004, gene, fontsize=8)
        ax2.set_xlabel("score_1 = Z_T")
        ax2.set_ylabel("score_2 = coverage_C")
        ax2.set_title(f"{subset_name}: mode overlay (binary -> donor_quantile)")
        ax2.grid(alpha=0.25, linewidth=0.6)
        handles = [
            mlines.Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                color="#1F77B4",
                label="binary",
                markersize=8,
            ),
            mlines.Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                color="#D62728",
                label="donor_quantile",
                markersize=8,
            ),
        ]
        ax2.legend(handles=handles, loc="best", frameon=True, fontsize=8)
        fig2.tight_layout()
        fig2.savefig(
            out_dir / "mode_overlay_scatter.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
            bbox_inches="tight",
        )
        plt.close(fig2)
    else:
        _save_placeholder(
            out_dir / "mode_overlay_scatter.png",
            f"{subset_name} mode overlay",
            "Missing one or both foreground modes.",
        )

    # 3) mode heatmaps with shared scaling.
    heat_cols = [
        "Z_T",
        "coverage_C",
        "peaks_K",
        "prev",
        "qc_risk",
        "qc_profile_similarity",
    ]
    combined = subset_df.copy()
    combined["neglog10_q_T"] = -np.log10(
        np.clip(combined["q_T"].to_numpy(dtype=float), 1e-300, 1.0)
    )
    heat_cols = [
        "Z_T",
        "neglog10_q_T",
        "coverage_C",
        "peaks_K",
        "prev",
        "qc_risk",
        "qc_profile_similarity",
    ]
    means = combined[heat_cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
    stds = (
        combined[heat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .std(axis=0)
        .replace(0.0, 1.0)
    )

    for mode in MODE_ORDER:
        sub = subset_df.loc[subset_df["foreground_mode"] == mode].copy()
        if sub.empty:
            _save_placeholder(
                out_dir / f"heatmap_scores_{mode}.png",
                f"{subset_name} heatmap {mode}",
                "No scored genes for mode.",
            )
            continue
        sub["neglog10_q_T"] = -np.log10(
            np.clip(sub["q_T"].to_numpy(dtype=float), 1e-300, 1.0)
        )
        sub = sub.sort_values(by=["panel_group", "gene"], kind="mergesort").reset_index(
            drop=True
        )
        mat_raw = sub[heat_cols].apply(pd.to_numeric, errors="coerce")
        mat = (mat_raw - means) / stds
        arr = mat.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8.2, 6.8))
        im = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-2.8, vmax=2.8)
        ax.set_yticks(np.arange(sub.shape[0]))
        ax.set_yticklabels(sub["gene"].tolist(), fontsize=8)
        ax.set_xticks(np.arange(len(heat_cols)))
        ax.set_xticklabels(
            [
                "Z_T",
                "-log10(q_T)",
                "coverage_C",
                "peaks_K",
                "prev",
                "qc_risk",
                "qc_profile_sim",
            ],
            rotation=25,
            ha="right",
        )
        ax.set_title(f"{subset_name}: heatmap ({mode}) shared scale")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("z-score")
        fig.tight_layout()
        fig.savefig(
            out_dir / f"heatmap_scores_{mode}.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


def _build_stability_summary(scores_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scores_df.empty:
        return pd.DataFrame(rows)
    for subset_name in sorted(scores_df["subset_name"].unique().tolist()):
        sub = scores_df.loc[scores_df["subset_name"] == subset_name]
        genes = sorted(sub["gene"].unique().tolist())
        for gene in genes:
            gdf = sub.loc[sub["gene"] == gene]
            a = gdf.loc[gdf["foreground_mode"] == "binary"]
            b = gdf.loc[gdf["foreground_mode"] == "donor_quantile"]
            if a.empty or b.empty:
                continue
            ra = a.iloc[0]
            rb = b.iloc[0]
            phi_a = float(ra["peak_dir_phi"])
            phi_b = float(rb["peak_dir_phi"])
            delta_phi = float(np.angle(np.exp(1j * (phi_b - phi_a))))
            rows.append(
                {
                    "subset_name": subset_name,
                    "gene": gene,
                    "Z_T_A": float(ra["Z_T"]),
                    "q_T_A": float(ra["q_T"]),
                    "peaks_K_A": int(ra["peaks_K"]),
                    "coverage_C_A": float(ra["coverage_C"]),
                    "class_A": str(ra["class_label"]),
                    "Z_T_B": float(rb["Z_T"]),
                    "q_T_B": float(rb["q_T"]),
                    "peaks_K_B": int(rb["peaks_K"]),
                    "coverage_C_B": float(rb["coverage_C"]),
                    "class_B": str(rb["class_label"]),
                    "delta_Z": float(rb["Z_T"] - ra["Z_T"]),
                    "delta_C": float(rb["coverage_C"] - ra["coverage_C"]),
                    "delta_phi": delta_phi,
                    "delta_phi_deg": float(np.degrees(delta_phi)),
                    "class_changed": bool(
                        str(ra["class_label"]) != str(rb["class_label"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_stability_for_subset(
    stability_df: pd.DataFrame, *, subset_name: str, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = stability_df.loc[stability_df["subset_name"] == subset_name].copy()
    if sub.empty:
        _save_placeholder(
            out_dir / f"{subset_name}_stability_empty.png",
            f"{subset_name} stability",
            "No data.",
        )
        return

    # 1) delta Z vs binary Z.
    fig1, ax1 = plt.subplots(figsize=(7.2, 5.2))
    ax1.scatter(
        sub["Z_T_A"],
        sub["delta_Z"],
        c="#4C78A8",
        s=80,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.6,
    )
    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax1.set_xlabel("Z_T(binary)")
    ax1.set_ylabel("ΔZ_T = Z_T(donor_quantile) - Z_T(binary)")
    ax1.set_title(f"{subset_name}: ΔZ_T stability")
    ax1.grid(alpha=0.25, linewidth=0.6)
    top = sub.reindex(sub["delta_Z"].abs().sort_values(ascending=False).head(8).index)
    for _, row in top.iterrows():
        ax1.text(
            float(row["Z_T_A"]) + 0.05,
            float(row["delta_Z"]) + 0.05,
            str(row["gene"]),
            fontsize=8,
        )
    fig1.tight_layout()
    fig1.savefig(
        out_dir / f"{subset_name}_deltaZ_vs_Zbinary.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig1)

    # 2) label stability confusion matrix.
    conf = pd.crosstab(sub["class_A"], sub["class_B"], dropna=False).reindex(
        index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0
    )
    fig2, ax2 = plt.subplots(figsize=(7.2, 6.0))
    im = ax2.imshow(conf.to_numpy(dtype=float), cmap="Blues", aspect="auto")
    ax2.set_xticks(np.arange(len(CLASS_ORDER)))
    ax2.set_yticks(np.arange(len(CLASS_ORDER)))
    ax2.set_xticklabels(CLASS_ORDER, rotation=35, ha="right", fontsize=8)
    ax2.set_yticklabels(CLASS_ORDER, fontsize=8)
    ax2.set_xlabel("class_label (donor_quantile)")
    ax2.set_ylabel("class_label (binary)")
    ax2.set_title(f"{subset_name}: label stability confusion")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax2.text(
                j,
                i,
                str(int(conf.iloc[i, j])),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    fig2.tight_layout()
    fig2.savefig(
        out_dir / f"{subset_name}_label_confusion.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig2)

    # 3) peak direction stability (hist + scatter).
    fig3 = plt.figure(figsize=(10.0, 4.4))
    ax3a = fig3.add_subplot(1, 2, 1)
    ax3b = fig3.add_subplot(1, 2, 2)
    ax3a.hist(
        sub["delta_phi_deg"], bins=18, color="#72B7B2", alpha=0.9, edgecolor="white"
    )
    ax3a.set_xlabel("Δphi (deg, wrapped)")
    ax3a.set_ylabel("count")
    ax3a.set_title("Peak direction Δphi histogram")
    z_ref = np.maximum(sub["Z_T_A"].to_numpy(dtype=float), 1e-6)
    ax3b.scatter(
        z_ref,
        sub["delta_phi_deg"],
        c="#F58518",
        s=75,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.6,
    )
    ax3b.set_xlabel("Z_T(binary)")
    ax3b.set_ylabel("Δphi (deg)")
    ax3b.set_title("Δphi vs Z_T(binary)")
    ax3b.grid(alpha=0.25, linewidth=0.6)
    fig3.tight_layout()
    fig3.savefig(
        out_dir / f"{subset_name}_peak_direction_stability.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
    )
    plt.close(fig3)

    # 4) robust gradient-like genes plot/table.
    robust = sub.loc[
        (sub["class_A"] == "Localized–unimodal")
        & (sub["class_B"] == "Localized–unimodal")
        & (sub["q_T_A"] <= Q_SIG)
        & (sub["q_T_B"] <= Q_SIG)
    ].copy()
    fig4, ax4 = plt.subplots(figsize=(8.5, 0.55 * max(2, robust.shape[0] + 2)))
    ax4.axis("off")
    title = f"{subset_name}: robust gradient-like genes (unimodal in both modes)"
    ax4.set_title(title)
    if robust.empty:
        ax4.text(
            0.01,
            0.80,
            "No robust gradient-like genes under current thresholds.",
            fontsize=10,
        )
    else:
        robust = robust.sort_values(by="Z_T_A", ascending=False).reset_index(drop=True)
        lines = ["gene | Z_A | Z_B | q_A | q_B | delta_Z | delta_phi_deg"]
        for _, row in robust.iterrows():
            lines.append(
                f"{row['gene']} | {float(row['Z_T_A']):.2f} | {float(row['Z_T_B']):.2f} | "
                f"{float(row['q_T_A']):.2e} | {float(row['q_T_B']):.2e} | "
                f"{float(row['delta_Z']):.2f} | {float(row['delta_phi_deg']):.1f}"
            )
        ax4.text(
            0.01,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
        )
    fig4.tight_layout()
    fig4.savefig(
        out_dir / f"{subset_name}_robust_gradient_like_genes.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
    )
    plt.close(fig4)


def _plot_controls_for_subset(
    *,
    subset_name: str,
    scores_df: pd.DataFrame,
    qc_scores_df: pd.DataFrame,
    donor_diag_df: pd.DataFrame | None,
    out_dir: Path,
    warning_lines: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = scores_df.loc[scores_df["subset_name"] == subset_name].copy()
    if sub.empty:
        _save_placeholder(
            out_dir / f"{subset_name}_controls_empty.png",
            f"{subset_name} controls",
            "No scores.",
        )
        return

    # 1) QC mimicry scatter.
    fig1, ax1 = plt.subplots(figsize=(7.4, 5.5))
    for mode, marker in [("binary", "o"), ("donor_quantile", "^")]:
        msub = sub.loc[sub["foreground_mode"] == mode]
        ax1.scatter(
            msub["qc_risk"].to_numpy(dtype=float),
            msub["Z_T"].to_numpy(dtype=float),
            c=[CLASS_COLORS.get(c, "#333333") for c in msub["class_label"].tolist()],
            s=80,
            marker=marker,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.85,
            label=mode,
        )
    ax1.axvline(QC_RISK_THRESH, color="#8B0000", linestyle="--", linewidth=1.1)
    ax1.axhline(Z_STRONG, color="#404040", linestyle="-.", linewidth=1.1)
    qc_driven = sub.loc[sub["class_label"] == "QC-driven"]
    for _, row in qc_driven.iterrows():
        ax1.text(
            float(row["qc_risk"]) + 0.01,
            float(row["Z_T"]) + 0.05,
            str(row["gene"]),
            fontsize=8,
        )
    ax1.set_xlabel("qc_risk = max |Spearman rho|")
    ax1.set_ylabel("Z_T")
    ax1.set_title(f"{subset_name}: QC mimicry diagnostic")
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.legend(loc="best", fontsize=8, frameon=True)
    fig1.tight_layout()
    fig1.savefig(out_dir / f"{subset_name}_qc_mimicry.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    # 2) overlay QC pseudo-genes in score space.
    fig2, ax2 = plt.subplots(figsize=(7.6, 5.8))
    base = sub.loc[sub["foreground_mode"] == "binary"]
    ax2.scatter(
        base["Z_T"].to_numpy(dtype=float),
        base["coverage_C"].to_numpy(dtype=float),
        c="#BDBDBD",
        s=65,
        edgecolors="white",
        linewidths=0.7,
        alpha=0.75,
        label="genes (binary)",
    )
    if not qc_scores_df.empty:
        qsub = qc_scores_df.loc[qc_scores_df["subset_name"] == subset_name]
        for _, row in qsub.iterrows():
            ax2.scatter(
                float(row["Z_T"]),
                float(row["coverage_C"]),
                c="#D62728",
                s=140,
                marker="X",
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
            )
            ax2.text(
                float(row["Z_T"]) + 0.05,
                float(row["coverage_C"]) + 0.004,
                str(row["gene"]),
                fontsize=8,
            )
    ax2.set_xlabel("Z_T")
    ax2.set_ylabel("coverage_C")
    ax2.set_title(f"{subset_name}: QC pseudo-genes in score space")
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.legend(loc="best", fontsize=8, frameon=True)
    fig2.tight_layout()
    fig2.savefig(
        out_dir / f"{subset_name}_qc_pseudogene_overlay.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig2)

    # 3) donor-directionality diagnostic.
    if donor_diag_df is None or donor_diag_df.empty:
        _save_placeholder(
            out_dir / f"{subset_name}_donor_directionality.png",
            f"{subset_name} donor-directionality",
            "Skipped: donor key unavailable.",
        )
        return
    dsub = donor_diag_df.loc[donor_diag_df["subset_name"] == subset_name].copy()
    fig3, ax3 = plt.subplots(figsize=(7.6, 5.8))
    ax3.scatter(
        base["Z_T"].to_numpy(dtype=float),
        base["coverage_C"].to_numpy(dtype=float),
        c="#CFCFCF",
        s=55,
        edgecolors="white",
        linewidths=0.6,
        alpha=0.70,
        label="genes (binary)",
    )
    ax3.scatter(
        dsub["Z_T"].to_numpy(dtype=float),
        dsub["coverage_C"].to_numpy(dtype=float),
        c="#D62728",
        marker="X",
        s=120,
        edgecolors="black",
        linewidths=0.8,
        alpha=0.95,
        label="donor vs all",
    )
    for _, row in dsub.iterrows():
        ax3.text(
            float(row["Z_T"]) + 0.04,
            float(row["coverage_C"]) + 0.004,
            str(row["donor_id"]),
            fontsize=7,
        )
    ax3.axvline(Z_STRONG, color="black", linestyle="--", linewidth=1.1)
    ax3.axhline(COVERAGE_STRONG, color="black", linestyle="-.", linewidth=1.1)
    ax3.set_xlabel("Z_T")
    ax3.set_ylabel("coverage_C")
    ax3.set_title(f"{subset_name}: donor-directionality diagnostic")
    ax3.grid(alpha=0.25, linewidth=0.6)
    ax3.legend(loc="best", fontsize=8, frameon=True)
    fig3.tight_layout()
    fig3.savefig(
        out_dir / f"{subset_name}_donor_directionality.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig3)

    strong = dsub.loc[(dsub["q_T"] <= Q_SIG) & (dsub["Z_T"] >= Z_STRONG)]
    if not strong.empty:
        warning_lines.append(
            f"[{subset_name}] donor-directionality strong for {strong.shape[0]} donor(s) "
            f"(criterion q_T<=0.05 and Z_T>=4.0)."
        )


def _compute_qc_profiles_and_scores(
    *,
    subset_name: str,
    covariates: dict[str, np.ndarray | None],
    donor_ids: np.ndarray | None,
    n_bins: int,
    n_perm: int,
    seed: int,
    bin_id: np.ndarray,
    bin_counts_total: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    qc_profiles: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    idx = 0
    for cov_name in QC_PSEUDOGENES:
        vals = covariates.get(cov_name)
        if vals is None:
            continue
        idx += 1
        perm = _perm_null_continuous_profile(
            np.asarray(vals, dtype=float),
            donor_ids=donor_ids,
            n_bins=int(n_bins),
            n_perm=int(n_perm),
            seed=int(seed + 100_000 + idx),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
        null_e = np.asarray(perm["null_E_phi"], dtype=float)
        null_t = np.asarray(perm["null_T"], dtype=float)
        qc_profiles[cov_name] = e_obs
        rows.append(
            {
                "subset_name": subset_name,
                "gene": cov_name,
                "foreground_mode": "qc_continuous",
                "T_obs": float(perm["T_obs"]),
                "p_T": float(perm["p_T"]),
                "q_T": float("nan"),
                "Z_T": float(robust_z(float(perm["T_obs"]), null_t)),
                "coverage_C": float(coverage_from_null(e_obs, null_e, q=0.95)),
                "peaks_K": int(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95)),
                "used_donor_stratified": bool(perm["used_donor_stratified"]),
                "warning": str(perm.get("warning", "")),
            }
        )
    qc_df = pd.DataFrame(rows)
    if not qc_df.empty:
        qc_df["q_T"] = bh_fdr(qc_df["p_T"].to_numpy(dtype=float))
    return qc_profiles, qc_df


def _compute_donor_directionality(
    *,
    subset_name: str,
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
    d = np.asarray(donor_ids).astype(str)
    uniq = sorted(np.unique(d).tolist())
    if len(uniq) < 2:
        return None
    rows: list[dict[str, Any]] = []
    for i, donor in enumerate(uniq):
        f = (d == donor).astype(float)
        perm = perm_null_T_and_profile(
            expr=f,
            theta=theta,
            donor_ids=None,
            n_bins=int(n_bins),
            n_perm=int(n_perm),
            seed=int(seed + 200_000 + i),
            donor_stratified=False,
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
        null_e = np.asarray(perm["null_E_phi"], dtype=float)
        null_t = np.asarray(perm["null_T"], dtype=float)
        rows.append(
            {
                "subset_name": subset_name,
                "donor_id": donor,
                "n_cells": int(np.sum(d == donor)),
                "prev": float(np.mean(d == donor)),
                "T_obs": float(perm["T_obs"]),
                "p_T": float(perm["p_T"]),
                "Z_T": float(robust_z(float(perm["T_obs"]), null_t)),
                "coverage_C": float(coverage_from_null(e_obs, null_e, q=0.95)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_T"] = bh_fdr(out["p_T"].to_numpy(dtype=float))
    return out


def _score_subset(
    *,
    subset_name: str,
    subset_idx: np.ndarray,
    resolved_panel: list[ResolvedGene],
    expr_matrix: Any,
    umap_xy: np.ndarray,
    donor_ids_all: np.ndarray | None,
    donor_key_used: str | None,
    total_counts_all: np.ndarray,
    pct_mt_all: np.ndarray | None,
    pct_ribo_all: np.ndarray | None,
    q: float,
    n_bins: int,
    n_perm: int,
    seed: int,
    min_cells: int,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, dict[str, np.ndarray]]],
    pd.DataFrame,
    pd.DataFrame,
]:
    subset_idx_arr = np.asarray(subset_idx, dtype=int)
    subset_xy = umap_xy[subset_idx_arr]
    subset_center = compute_vantage_point(subset_xy, method="median")
    theta = compute_theta(subset_xy, subset_center)
    _, bin_id = bin_theta(theta, int(n_bins))
    bin_counts_total = np.bincount(bin_id, minlength=int(n_bins)).astype(float)

    donor_subset = donor_ids_all[subset_idx_arr] if donor_ids_all is not None else None
    total_counts = np.asarray(total_counts_all[subset_idx_arr], dtype=float)
    pct_mt = (
        np.asarray(pct_mt_all[subset_idx_arr], dtype=float)
        if pct_mt_all is not None
        else None
    )
    pct_ribo = (
        np.asarray(pct_ribo_all[subset_idx_arr], dtype=float)
        if pct_ribo_all is not None
        else None
    )

    subset_underpowered = int(subset_idx_arr.size) < int(min_cells)
    if subset_underpowered:
        print(
            f"WARNING: subset '{subset_name}' has n_cells={subset_idx_arr.size} < min_cells={int(min_cells)}; "
            "continuing with subset_underpowered=True."
        )

    covariates = {
        "total_counts": total_counts,
        "pct_counts_mt": pct_mt,
        "pct_counts_ribo": pct_ribo,
    }
    qc_profiles, qc_scores = _compute_qc_profiles_and_scores(
        subset_name=subset_name,
        covariates=covariates,
        donor_ids=donor_subset,
        n_bins=int(n_bins),
        n_perm=int(n_perm),
        seed=int(seed),
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )

    if len(qc_profiles) == 0:
        print(
            f"WARNING: subset '{subset_name}' has no QC pseudo-gene profiles; "
            "qc_profile_similarity will be NaN."
        )

    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    row_counter = 0
    for item in resolved_panel:
        if not item.found or item.gene_idx is None:
            continue
        expr_full = get_feature_vector(expr_matrix, int(item.gene_idx))
        expr_sub = np.asarray(expr_full[subset_idx_arr], dtype=float)
        artifacts[item.gene] = {}

        for mode_idx, mode in enumerate(MODE_ORDER):
            row_counter += 1
            row_seed = int(seed + 10_000 + row_counter * 29 + mode_idx * 997)
            row, art = _score_gene_mode(
                subset_name=subset_name,
                gene=item.gene,
                category=item.category,
                panel_group=item.panel_group,
                expr=expr_sub,
                foreground_mode=mode,
                q=float(q),
                donor_ids=donor_subset,
                donor_key_used=donor_key_used,
                theta=theta,
                n_bins=int(n_bins),
                n_perm=int(n_perm),
                seed=row_seed,
                bin_id=bin_id,
                bin_counts_total=bin_counts_total,
                total_counts=total_counts,
                pct_counts_mt=pct_mt,
                pct_counts_ribo=pct_ribo,
                qc_profiles=qc_profiles,
                subset_underpowered=bool(subset_underpowered),
            )
            rows.append(row)
            artifacts[item.gene][mode] = art

    scores_df = _apply_fdr_and_classification(pd.DataFrame(rows))
    donor_diag = _compute_donor_directionality(
        subset_name=subset_name,
        donor_ids=donor_subset,
        theta=theta,
        n_bins=int(n_bins),
        n_perm=int(n_perm),
        seed=int(seed),
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )
    if donor_diag is None:
        donor_diag = pd.DataFrame()

    return scores_df, artifacts, qc_scores, donor_diag


def main() -> int:
    args = parse_args()
    apply_plot_style()

    h5ad_path = Path(args.h5ad)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Input h5ad not found: {h5ad_path}")

    outdir = Path(args.out)
    tables_dir = outdir / "tables"
    plots_dir = outdir / "plots"
    dirs = {
        "overview": plots_dir / "00_overview",
        "subset_umaps": plots_dir / "01_subset_umaps",
        "gene_panels": plots_dir / "02_gene_panels",
        "score_space": plots_dir / "03_score_space",
        "stability": plots_dir / "04_stability",
        "controls": plots_dir / "05_controls",
    }
    for d in [tables_dir, *dirs.values()]:
        d.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(h5ad_path)
    _ensure_umap(adata, seed=int(args.seed), recompute_if_missing=False)
    embedding_key, umap_xy = _resolve_embedding(adata, args.embedding_key)
    center_xy = compute_vantage_point(umap_xy, method="median")

    expr_matrix, adata_like, expr_source = _choose_expression_source(
        adata,
        layer_arg=args.layer,
        use_raw_arg=bool(args.use_raw),
    )

    donor_ids, donor_key_used = _resolve_donor_ids(adata, args.donor_key)
    if donor_ids is None:
        print(
            "WARNING: donor key unavailable (or <2 donors). Using global fallback where required."
        )
    label_key = _resolve_label_key(adata, args.label_key)
    if label_key is None:
        raise RuntimeError(
            "No label key found. Tried: azimuth_label, predicted_label, predicted_CLID, cell_type, leiden, cluster."
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
        print(
            "WARNING: pct_counts_mt unavailable; QC-risk correlations will omit this covariate."
        )
    if pct_ribo is None:
        print(
            "WARNING: pct_counts_ribo unavailable; QC-risk correlations will omit this covariate."
        )

    print(f"embedding_key_used={embedding_key}")
    print(f"label_key_used={label_key}")
    print(f"donor_key_used={donor_key_used if donor_key_used is not None else 'None'}")
    print(f"expression_source_used={expr_source}")
    print(f"pct_counts_mt_source={pct_mt_source}")
    print(f"pct_counts_ribo_source={pct_ribo_source}")
    print(
        f"n_cells={int(adata.n_obs)} n_bins={int(args.n_bins)} n_perm={int(args.n_perm)} "
        f"seed={int(args.seed)} q={float(args.q):.3f}"
    )

    labels = adata.obs[label_key].astype("string").fillna("NA").astype(str)
    subset_masks, matched_labels = _build_subset_masks(labels)

    _plot_overview(
        adata=adata,
        umap_xy=umap_xy,
        label_key=label_key,
        donor_key=donor_key_used,
        center_xy=center_xy,
        matched_labels=matched_labels,
        out_dir=dirs["overview"],
    )

    subset_results: dict[str, pd.DataFrame] = {}
    subset_artifacts: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    all_qc_scores: list[pd.DataFrame] = []
    all_donor_diag: list[pd.DataFrame] = []
    warning_lines: list[str] = []

    for subset_i, subset_name in enumerate(["fibroblast", "myeloid"]):
        mask = np.asarray(subset_masks[subset_name], dtype=bool)
        idx = np.flatnonzero(mask).astype(int)
        if idx.size == 0:
            print(
                f"WARNING: subset '{subset_name}' is empty; writing placeholder outputs."
            )
            subset_results[subset_name] = pd.DataFrame()
            subset_artifacts[subset_name] = {}
            _save_placeholder(
                dirs["subset_umaps"] / f"{subset_name}_umap_plain.png",
                f"{subset_name} subset",
                "No cells selected.",
            )
            continue

        resolved_panel, panel_df = _resolve_panel(adata_like, subset_name=subset_name)
        panel_csv = tables_dir / f"gene_panel_{subset_name}.csv"
        panel_df.to_csv(panel_csv, index=False)
        missing = (
            panel_df.loc[~panel_df["found"].astype(bool), "gene"].astype(str).tolist()
        )
        if missing:
            print(f"subset={subset_name} missing_genes={','.join(missing)}")

        subset_xy = umap_xy[idx]
        subset_center = compute_vantage_point(subset_xy, method="median")
        donor_labels = (
            adata.obs.iloc[idx][donor_key_used]
            if (donor_key_used is not None and donor_key_used in adata.obs.columns)
            else None
        )
        pct_mt_sub = pct_mt[idx] if pct_mt is not None else None
        _plot_subset_umaps(
            subset_name=subset_name,
            subset_xy=subset_xy,
            subset_center=subset_center,
            donor_labels=donor_labels,
            pct_mt=pct_mt_sub,
            out_dir=dirs["subset_umaps"] / subset_name,
        )

        scores_df, artifacts, qc_scores_df, donor_diag_df = _score_subset(
            subset_name=subset_name,
            subset_idx=idx,
            resolved_panel=resolved_panel,
            expr_matrix=expr_matrix,
            umap_xy=umap_xy,
            donor_ids_all=donor_ids,
            donor_key_used=donor_key_used,
            total_counts_all=np.asarray(total_counts, dtype=float),
            pct_mt_all=pct_mt,
            pct_ribo_all=pct_ribo,
            q=float(args.q),
            n_bins=int(args.n_bins),
            n_perm=int(args.n_perm),
            seed=int(args.seed + subset_i * 10_000),
            min_cells=int(args.min_cells),
        )
        subset_results[subset_name] = scores_df
        subset_artifacts[subset_name] = artifacts
        all_qc_scores.append(qc_scores_df)
        all_donor_diag.append(donor_diag_df)

        scores_csv = tables_dir / f"scores_{subset_name}.csv"
        scores_df.to_csv(scores_csv, index=False)
        print(
            f"subset={subset_name} scores_rows={scores_df.shape[0]} -> {scores_csv.as_posix()}"
        )
        if not scores_df.empty:
            cls_counts = (
                scores_df["class_label"]
                .value_counts()
                .reindex(CLASS_ORDER, fill_value=0)
                .astype(int)
                .to_dict()
            )
            cls_txt = "; ".join([f"{k}={v}" for k, v in cls_counts.items()])
            print(f"subset={subset_name} class_summary={cls_txt}")

        # Per-gene two-mode panel plots.
        gene_panel_dir = dirs["gene_panels"] / subset_name
        gene_panel_dir.mkdir(parents=True, exist_ok=True)
        for gene in sorted(artifacts.keys()):
            mode_art = artifacts[gene]
            if not all(m in mode_art for m in MODE_ORDER):
                continue
            rows = {
                m: scores_df.loc[
                    (scores_df["gene"] == gene) & (scores_df["foreground_mode"] == m)
                ].iloc[0]
                for m in MODE_ORDER
            }
            _plot_gene_panel_both_modes(
                subset_name=subset_name,
                gene=gene,
                gene_category=str(rows["binary"]["category"]),
                subset_xy=subset_xy,
                subset_center=subset_center,
                rows_by_mode=rows,
                artifacts_by_mode=mode_art,
                n_bins=int(args.n_bins),
                out_png=gene_panel_dir / f"gene_{gene}.png",
            )

        # Score space / controls per subset.
        _plot_score_space_for_subset(
            scores_df,
            subset_name=subset_name,
            out_dir=dirs["score_space"] / subset_name,
        )
        _plot_controls_for_subset(
            subset_name=subset_name,
            scores_df=scores_df,
            qc_scores_df=qc_scores_df,
            donor_diag_df=donor_diag_df,
            out_dir=dirs["controls"] / subset_name,
            warning_lines=warning_lines,
        )

    # Stability summary across subsets.
    all_scores = pd.concat(
        [df for df in subset_results.values() if not df.empty], ignore_index=True
    )
    stability_df = _build_stability_summary(all_scores)
    stability_csv = tables_dir / "stability_summary.csv"
    stability_df.to_csv(stability_csv, index=False)
    for subset_name in ["fibroblast", "myeloid"]:
        _plot_stability_for_subset(
            stability_df,
            subset_name=subset_name,
            out_dir=dirs["stability"] / subset_name,
        )

    # Optional additional tables for QC pseudo-gene and donor diagnostics.
    qc_all = (
        pd.concat([df for df in all_qc_scores if not df.empty], ignore_index=True)
        if all_qc_scores
        else pd.DataFrame()
    )
    donor_all = (
        pd.concat([df for df in all_donor_diag if not df.empty], ignore_index=True)
        if all_donor_diag
        else pd.DataFrame()
    )
    if not qc_all.empty:
        qc_all.to_csv(tables_dir / "qc_pseudogene_scores.csv", index=False)
    if not donor_all.empty:
        donor_all.to_csv(tables_dir / "donor_directionality_scores.csv", index=False)

    # Write warning README if needed.
    if warning_lines:
        readme_txt = [
            "WARNING: donor-directionality diagnostic indicates strong donor structure in subset space.",
            "",
            "Criteria: q_T <= 0.05 and Z_T >= 4.0 for donor-vs-all foreground.",
            "",
            "Findings:",
        ]
        readme_txt.extend([f"- {line}" for line in warning_lines])
        (outdir / "README.txt").write_text(
            "\n".join(readme_txt) + "\n", encoding="utf-8"
        )

    print(f"stability_summary_csv={stability_csv.as_posix()}")
    print(f"results_root={outdir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
