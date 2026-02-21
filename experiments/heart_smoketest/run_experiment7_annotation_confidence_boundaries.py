#!/usr/bin/env python3
"""Experiment #7: annotation-confidence geometry and boundary diagnostics."""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

# Deterministic headless plotting.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.neighbors import NearestNeighbors

# Allow running from repository root via `python experiments/...`.
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
]

SCORE_CANDIDATES = [
    "prediction_score",
    "predicted_label.score",
]

MATCHTYPE_CANDIDATE = "cl_match_type"
PREDICTED_SCORE_REGEX = re.compile(r"predicted\..*\.score$", re.IGNORECASE)

BOUNDARY_MARKER_PANEL = {
    "Cardiomyocyte": ["MYH6", "TNNT2"],
    "Endothelial": ["PECAM1", "VWF"],
    "Fibroblast": ["COL1A1", "LUM"],
    "Myeloid": ["LST1", "LYZ"],
}

DUAL_MARKER_PAIRS = [
    ("PECAM1", "COL1A1"),
    ("TNNT2", "PECAM1"),
    ("LST1", "COL1A1"),
]

QC_COVAR_CANDIDATES = {
    "total_counts": ["total_counts", "n_counts", "nCount_RNA", "nUMI", "total_umis"],
    "pct_counts_mt": ["pct_counts_mt", "percent.mt", "pct_mt"],
    "pct_counts_ribo": ["pct_counts_ribo", "percent.ribo", "pct_ribo"],
}

QC_RISK_THRESH = 0.35
QC_SIM_THRESH = 0.70


@dataclass(frozen=True)
class LowConfDefinition:
    name: str
    definition_type: str
    source_key: str
    low_mask: np.ndarray
    high_mask: np.ndarray
    notes: str
    threshold_low: float | None
    threshold_high: float | None


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
        description="Run Experiment #7 annotation-confidence boundary diagnostics."
    )
    p.add_argument(
        "--h5ad",
        default="data/processed/HT_pca_umap.h5ad",
        help="Input .h5ad path.",
    )
    p.add_argument(
        "--out",
        default="experiments/heart_smoketest/results/experiment7_annotation_confidence_boundaries",
        help="Output directory.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--n_perm", type=int, default=300, help="Permutation count.")
    p.add_argument("--n_bins", type=int, default=64, help="Angular bin count.")
    p.add_argument("--k", type=int, default=30, help="kNN neighborhood size.")
    p.add_argument("--q_low", type=float, default=0.10, help="Low-confidence quantile.")
    p.add_argument(
        "--q_high", type=float, default=0.90, help="High-confidence quantile."
    )
    p.add_argument(
        "--random_reps",
        type=int,
        default=25,
        help="Random foreground replicate count.",
    )
    p.add_argument(
        "--save_per_cell",
        type=_str2bool,
        default=False,
        help="Write boundary_metrics_per_cell.csv (can be large).",
    )
    p.add_argument(
        "--embedding_key", default=None, help="Optional embedding key override."
    )
    p.add_argument("--donor_key", default=None, help="Optional donor key override.")
    p.add_argument("--label_key", default=None, help="Optional label key override.")
    p.add_argument("--layer", default=None, help="Optional expression layer override.")
    p.add_argument(
        "--use_raw",
        action="store_true",
        help="Use adata.raw instead of X/layers.",
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
    donor_ids = adata.obs[key].astype("string").fillna("NA").astype(str).to_numpy()
    if np.unique(donor_ids).size < 2:
        return None, key
    return donor_ids, key


def _resolve_label_key_required(
    adata: ad.AnnData,
    requested_key: str | None,
) -> tuple[np.ndarray, str]:
    key = _resolve_key(adata, requested_key, LABEL_KEY_CANDIDATES)
    if key is None:
        raise RuntimeError(
            "Experiment #7 requires labels. No label key found among "
            f"{LABEL_KEY_CANDIDATES}."
        )
    labels = adata.obs[key].astype("string").fillna("NA").astype(str).to_numpy()
    if np.unique(labels).size < 2:
        raise RuntimeError(
            f"Experiment #7 requires >=2 labels. Found {int(np.unique(labels).size)} in '{key}'."
        )
    return labels, key


def _find_numeric_confidence_key(
    adata: ad.AnnData,
) -> tuple[str | None, np.ndarray | None, str]:
    for key in SCORE_CANDIDATES:
        if key in adata.obs.columns:
            vals = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(vals).sum() > 0:
                return key, vals, "preferred"

    regex_hits = [c for c in adata.obs.columns if PREDICTED_SCORE_REGEX.match(str(c))]
    for key in sorted(regex_hits):
        vals = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(vals).sum() > 0:
            return str(key), vals, "regex"
    return None, None, "missing"


def _find_matchtype_key(adata: ad.AnnData) -> tuple[str | None, np.ndarray | None]:
    if MATCHTYPE_CANDIDATE in adata.obs.columns:
        vals = (
            adata.obs[MATCHTYPE_CANDIDATE]
            .astype("string")
            .fillna("NA")
            .astype(str)
            .to_numpy()
        )
        return MATCHTYPE_CANDIDATE, vals
    return None, None


def _is_ambiguous_matchtype(text: str) -> bool:
    t = str(text).lower()
    if any(
        x in t for x in ["low", "ambig", "uncertain", "unknown", "partial", "mixed"]
    ):
        return True
    # explicit "no" token-like forms to avoid matching arbitrary words.
    if re.search(r"(^|[^a-z])no([^a-z]|$)", t) is not None:
        return True
    return False


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


def _compute_boundary_metrics(
    *,
    xy: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    xy_arr = np.asarray(xy, dtype=float)
    lbl = np.asarray(labels).astype(str)
    n = int(xy_arr.shape[0])
    if n < 3:
        raise ValueError("Need at least 3 cells for boundary metrics.")

    label_cat = pd.Categorical(lbl, categories=sorted(pd.Index(lbl).unique().tolist()))
    codes = label_cat.codes.astype(int)
    n_labels = int(len(label_cat.categories))

    k_use = int(min(max(2, int(k)), max(2, n - 1)))
    nn = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean")
    nn.fit(xy_arr)
    nbr_idx = nn.kneighbors(xy_arr, return_distance=False)[:, 1:]

    nbr_codes = codes[nbr_idx]
    counts = np.zeros((n, n_labels), dtype=float)
    row_idx = np.repeat(np.arange(n, dtype=int), k_use)
    col_idx = nbr_codes.ravel().astype(int)
    np.add.at(counts, (row_idx, col_idx), 1.0)
    p = counts / float(k_use)

    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(p > 0.0, p * np.log(p), 0.0), axis=1)
    denom = np.log(max(2, n_labels))
    h_norm = np.clip(entropy / float(denom), 0.0, 1.0)
    max_frac = np.max(p, axis=1)
    nndis = 1.0 - max_frac
    boundary = 0.5 * h_norm + 0.5 * nndis

    metrics_df = pd.DataFrame(
        {
            "label": lbl,
            "label_entropy": entropy,
            "label_entropy_norm": h_norm,
            "nn_disagreement": nndis,
            "boundary_score": boundary,
            "k_neighbors": int(k_use),
        }
    )
    return metrics_df, nbr_idx


def _build_lowconf_definitions(
    *,
    score_key: str | None,
    score_vals: np.ndarray | None,
    matchtype_key: str | None,
    matchtype_vals: np.ndarray | None,
    q_low: float,
    q_high: float,
    warnings_log: list[str],
) -> list[LowConfDefinition]:
    defs: list[LowConfDefinition] = []
    n = int(
        score_vals.size
        if score_vals is not None
        else (matchtype_vals.size if matchtype_vals is not None else 0)
    )

    if score_key is not None and score_vals is not None:
        x = np.asarray(score_vals, dtype=float)
        finite = np.isfinite(x)
        if int(finite.sum()) < 10:
            warnings_log.append(
                f"Score key '{score_key}' has too few finite values; score definition skipped."
            )
        else:
            low_thr = float(np.nanquantile(x[finite], float(q_low)))
            high_thr = float(np.nanquantile(x[finite], float(q_high)))
            low_mask = finite & (x <= low_thr)
            high_mask = finite & (x >= high_thr)
            if int(low_mask.sum()) == 0:
                warnings_log.append(
                    f"Score definition produced zero low-confidence cells at q_low={float(q_low):.2f}; skipped."
                )
            elif int(high_mask.sum()) == 0:
                warnings_log.append(
                    f"Score definition produced zero high-confidence cells at q_high={float(q_high):.2f}; using complement for high_conf."
                )
                high_mask = ~low_mask
            defs.append(
                LowConfDefinition(
                    name="score_low",
                    definition_type="score",
                    source_key=score_key,
                    low_mask=low_mask.astype(bool),
                    high_mask=high_mask.astype(bool),
                    notes=f"low=score<=q{float(q_low):.2f}, high=score>=q{float(q_high):.2f}",
                    threshold_low=low_thr,
                    threshold_high=high_thr,
                )
            )

    if matchtype_key is not None and matchtype_vals is not None:
        vals = np.asarray(matchtype_vals).astype(str)
        low_mask = np.array([_is_ambiguous_matchtype(v) for v in vals], dtype=bool)
        if int(low_mask.sum()) == 0:
            warnings_log.append(
                f"Match-type definition skipped: no ambiguous/low-quality values matched in '{matchtype_key}'."
            )
        elif int(low_mask.sum()) == int(n):
            warnings_log.append(
                f"Match-type definition skipped: all cells matched low-confidence in '{matchtype_key}'."
            )
        else:
            high_mask = ~low_mask
            defs.append(
                LowConfDefinition(
                    name="matchtype_low",
                    definition_type="matchtype",
                    source_key=matchtype_key,
                    low_mask=low_mask,
                    high_mask=high_mask,
                    notes="low_conf via substrings: low/ambig/uncertain/unknown/no/partial/mixed",
                    threshold_low=None,
                    threshold_high=None,
                )
            )

    return defs


def _resolve_marker_vectors(
    adata_like: Any,
    expr_matrix: Any,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    vecs: dict[str, np.ndarray] = {}
    for group, genes in BOUNDARY_MARKER_PANEL.items():
        for gene in genes:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    idx, label, symbol_col, source = resolve_feature_index(
                        adata_like, gene
                    )
                idx = int(idx)
                expr = get_feature_vector(expr_matrix, idx)
                expr = np.asarray(expr, dtype=float).ravel()
                vecs[gene] = expr
                rows.append(
                    {
                        "gene": gene,
                        "panel_group": group,
                        "status": "resolved",
                        "resolved_gene": str(label),
                        "gene_idx": int(idx),
                        "symbol_column": str(symbol_col or ""),
                        "resolution_source": str(source),
                    }
                )
            except KeyError:
                rows.append(
                    {
                        "gene": gene,
                        "panel_group": group,
                        "status": "missing",
                        "resolved_gene": "",
                        "gene_idx": "",
                        "symbol_column": "",
                        "resolution_source": "",
                    }
                )
    return pd.DataFrame(rows), vecs


def _foreground_rsp(
    *,
    f_mask: np.ndarray,
    theta: np.ndarray,
    n_bins: int,
    n_perm: int,
    seed: int,
    donor_ids: np.ndarray | None,
    bin_id: np.ndarray,
    bin_counts_total: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    f = np.asarray(f_mask, dtype=bool).ravel()
    if int(f.sum()) in {0, int(f.size)}:
        e_obs = np.zeros(int(n_bins), dtype=float)
        null_e = np.zeros((int(n_perm), int(n_bins)), dtype=float)
        null_t = np.zeros(int(n_perm), dtype=float)
        out = {
            "T_obs": 0.0,
            "p_T": 1.0,
            "Z_T": 0.0,
            "coverage_C": 0.0,
            "peaks_K": 0,
            "phi_hat_rad": 0.0,
            "phi_hat_deg": 0.0,
            "used_donor_stratified": False,
            "perm_warning": "Degenerate foreground (all/none cells).",
        }
        art = {
            "E_phi_obs": e_obs,
            "null_E_phi": null_e,
            "null_T": null_t,
            "T_obs": np.asarray([0.0], dtype=float),
        }
        return out, art

    perm = perm_null_T_and_profile(
        expr=f.astype(float),
        theta=np.asarray(theta, dtype=float),
        donor_ids=donor_ids,
        n_bins=int(n_bins),
        n_perm=int(n_perm),
        seed=int(seed),
        donor_stratified=bool(donor_ids is not None),
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )
    e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
    null_e = np.asarray(perm["null_E_phi"], dtype=float)
    null_t = np.asarray(perm["null_T"], dtype=float)
    t_obs = float(perm["T_obs"])
    z_t = float(robust_z(t_obs, null_t))
    cov = float(coverage_from_null(e_obs, null_e, q=0.95))
    peaks = int(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95))
    idx = int(np.argmax(np.abs(e_obs))) if e_obs.size > 0 else 0
    centers = theta_bin_centers(int(n_bins))
    phi = float(centers[idx]) if centers.size > 0 else 0.0
    out = {
        "T_obs": t_obs,
        "p_T": float(perm["p_T"]),
        "Z_T": z_t,
        "coverage_C": cov,
        "peaks_K": peaks,
        "phi_hat_rad": phi,
        "phi_hat_deg": float(np.degrees(phi) % 360.0),
        "used_donor_stratified": bool(perm["used_donor_stratified"]),
        "perm_warning": str(perm.get("warning", "")),
    }
    art = {
        "E_phi_obs": e_obs,
        "null_E_phi": null_e,
        "null_T": null_t,
        "T_obs": np.asarray([t_obs], dtype=float),
    }
    return out, art


def _mean_boundary_random_controls(
    *,
    low_mask: np.ndarray,
    boundary_score: np.ndarray,
    theta: np.ndarray,
    n_bins: int,
    random_reps: int,
    seed: int,
    bin_id: np.ndarray,
    bin_counts_total: np.ndarray,
) -> dict[str, np.ndarray]:
    f_low = np.asarray(low_mask, dtype=bool).ravel()
    n = int(f_low.size)
    n_fg = int(f_low.sum())
    b = np.asarray(boundary_score, dtype=float).ravel()
    rng = np.random.default_rng(int(seed))

    mean_b = np.zeros(int(random_reps), dtype=float)
    t_rand = np.zeros(int(random_reps), dtype=float)
    pooled_b: list[np.ndarray] = []
    for r in range(int(random_reps)):
        idx = rng.choice(n, size=n_fg, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        pooled_b.append(b[mask])
        mean_b[r] = float(np.mean(b[mask]))
        e_r, _, _, _ = compute_rsp_profile_from_boolean(
            mask,
            np.asarray(theta, dtype=float),
            int(n_bins),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        t_rand[r] = float(np.max(np.abs(np.asarray(e_r, dtype=float))))

    return {
        "mean_boundary_random": mean_b,
        "T_random": t_rand,
        "boundary_random_pooled": (
            np.concatenate(pooled_b) if pooled_b else np.zeros(0, dtype=float)
        ),
    }


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float).ravel())
    if x.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
    return x, y


def _cliffs_delta_from_u(u_stat: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return float((2.0 * float(u_stat) / (float(n1) * float(n2))) - 1.0)


def _plot_overview(
    *,
    umap_xy: np.ndarray,
    labels: np.ndarray,
    donor_ids: np.ndarray | None,
    label_key: str,
    donor_key: str | None,
    center_xy: np.ndarray,
    primary_low_mask: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_categorical_umap(
        umap_xy=umap_xy,
        labels=pd.Series(labels),
        title=f"UMAP by labels ({label_key})",
        outpath=out_dir / "umap_labels.png",
        vantage_point=(float(center_xy[0]), float(center_xy[1])),
        annotate_cluster_medians=False,
    )
    if donor_ids is not None and donor_key is not None:
        plot_categorical_umap(
            umap_xy=umap_xy,
            labels=pd.Series(donor_ids),
            title=f"UMAP by donor ({donor_key})",
            outpath=out_dir / "umap_donor.png",
            vantage_point=(float(center_xy[0]), float(center_xy[1])),
            annotate_cluster_medians=False,
        )
    else:
        _save_placeholder(
            out_dir / "umap_donor.png", "UMAP by donor", "Donor key unavailable."
        )

    labels_s = pd.Series(labels.astype(str), name="label")
    counts = labels_s.value_counts().sort_values(ascending=False)
    low_frac = labels_s.groupby(labels_s).apply(
        lambda s: float(
            np.mean(
                np.asarray(primary_low_mask, dtype=bool)[s.index.to_numpy(dtype=int)]
            )
        )
    )
    low_frac = low_frac.reindex(counts.index).fillna(0.0)

    fig, ax1 = plt.subplots(figsize=(max(8.0, 0.42 * counts.shape[0] + 4.0), 5.2))
    x = np.arange(counts.shape[0], dtype=float)
    ax1.bar(
        x,
        counts.to_numpy(dtype=float),
        color="#5DA5DA",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("# cells")
    ax1.set_xticks(x)
    ax1.set_xticklabels(counts.index.tolist(), rotation=35, ha="right", fontsize=8)
    ax1.set_title("Label counts with low-confidence fraction overlay")
    ax1.grid(axis="y", alpha=0.25, linewidth=0.6)

    ax2 = ax1.twinx()
    ax2.plot(
        x, low_frac.to_numpy(dtype=float), color="#D62728", marker="o", linewidth=1.6
    )
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("low_conf fraction")
    fig.tight_layout()
    fig.savefig(
        out_dir / "label_counts_with_lowconf_fraction.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig)


def _plot_lowconf_umaps(
    *,
    defn: LowConfDefinition,
    umap_xy: np.ndarray,
    center_xy: np.ndarray,
    score_vals: np.ndarray | None,
    boundary_score: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    low = np.asarray(defn.low_mask, dtype=bool)

    fig1, ax1 = plt.subplots(figsize=(7.3, 6.0))
    ax1.scatter(
        umap_xy[~low, 0],
        umap_xy[~low, 1],
        c="#D0D0D0",
        s=4.0,
        alpha=0.33,
        linewidths=0,
        rasterized=True,
        label="others",
    )
    ax1.scatter(
        umap_xy[low, 0],
        umap_xy[low, 1],
        c="#D62728",
        s=8.0,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
        label=f"{defn.name} low_conf",
    )
    ax1.scatter(
        [float(center_xy[0])],
        [float(center_xy[1])],
        marker="X",
        s=80,
        c="black",
        edgecolors="white",
        linewidths=0.8,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"{defn.name}: low-confidence highlight")
    ax1.legend(loc="upper left", fontsize=8, frameon=True)
    fig1.tight_layout()
    fig1.savefig(
        out_dir / f"{defn.name}_lowconf_highlight.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7.3, 6.0))
    ax2.scatter(
        umap_xy[:, 0],
        umap_xy[:, 1],
        c="#D8D8D8",
        s=3.0,
        alpha=0.28,
        linewidths=0,
        rasterized=True,
    )
    if int(low.sum()) > 0:
        hb = ax2.hexbin(
            umap_xy[low, 0],
            umap_xy[low, 1],
            gridsize=42,
            cmap="Reds",
            mincnt=1,
            linewidths=0.0,
            alpha=0.95,
        )
        fig2.colorbar(
            hb, ax=ax2, fraction=0.046, pad=0.03, label="low_conf local count"
        )
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title(f"{defn.name}: low-confidence density")
    fig2.tight_layout()
    fig2.savefig(
        out_dir / f"{defn.name}_lowconf_density.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig2)

    if score_vals is not None:
        save_numeric_umap(
            umap_xy=umap_xy,
            values=np.asarray(score_vals, dtype=float),
            out_png=out_dir / f"{defn.name}_confidence_score_umap.png",
            title="Confidence score",
            cmap="viridis",
            colorbar_label="confidence score",
            vantage_point=(float(center_xy[0]), float(center_xy[1])),
        )
    else:
        _save_placeholder(
            out_dir / f"{defn.name}_confidence_score_umap.png",
            "Confidence score",
            "No numeric confidence score available.",
        )

    save_numeric_umap(
        umap_xy=umap_xy,
        values=np.asarray(boundary_score, dtype=float),
        out_png=out_dir / f"{defn.name}_boundary_score_umap.png",
        title="Boundary score",
        cmap="magma",
        colorbar_label="boundary score",
        vantage_point=(float(center_xy[0]), float(center_xy[1])),
    )


def _plot_biorsp_profiles(
    *,
    defn: LowConfDefinition,
    biorsp_row: pd.Series,
    art: dict[str, np.ndarray],
    high_profile: np.ndarray,
    n_bins: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    e_obs = np.asarray(art["E_phi_obs"], dtype=float)
    null_e = np.asarray(art["null_E_phi"], dtype=float)
    null_t = np.asarray(art["null_T"], dtype=float)

    fig = plt.figure(figsize=(15.0, 4.8))
    ax1 = fig.add_subplot(1, 3, 1, projection="polar")
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection="polar")

    th = theta_bin_centers(int(n_bins))
    th_c = np.concatenate([th, th[:1]])
    obs_c = np.concatenate([e_obs, e_obs[:1]])
    q_hi = np.quantile(null_e, 0.95, axis=0)
    q_lo = np.quantile(null_e, 0.05, axis=0)
    ax1.plot(th_c, obs_c, color="#8B0000", linewidth=2.0, label="low_conf")
    ax1.plot(
        th_c,
        np.concatenate([q_hi, q_hi[:1]]),
        color="#444444",
        linestyle="--",
        linewidth=1.2,
        label="null95",
    )
    ax1.plot(
        th_c,
        np.concatenate([q_lo, q_lo[:1]]),
        color="#444444",
        linestyle="--",
        linewidth=1.0,
        label="null5",
    )
    ax1.fill_between(
        th_c,
        np.concatenate([q_lo, q_lo[:1]]),
        np.concatenate([q_hi, q_hi[:1]]),
        color="#B0B0B0",
        alpha=0.17,
    )
    ax1.set_theta_zero_location("E")
    ax1.set_theta_direction(1)
    ax1.set_thetagrids(np.arange(0, 360, 90))
    ann = (
        f"T={float(biorsp_row['T_obs']):.3f}\n"
        f"Z={float(biorsp_row['Z_T']):.2f}\n"
        f"q={float(biorsp_row['q_T']):.2e}\n"
        f"C={float(biorsp_row['coverage_C']):.3f}\n"
        f"K={int(biorsp_row['peaks_K'])}\n"
        f"phi={float(biorsp_row['phi_hat_deg']):.1f}°"
    )
    ax1.text(
        0.02,
        0.02,
        ann,
        transform=ax1.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.85},
    )
    ax1.set_title("Low-conf BioRSP profile")
    ax1.legend(loc="upper right", bbox_to_anchor=(1.18, 1.2), fontsize=8, frameon=True)

    bins = int(min(45, max(12, np.ceil(np.sqrt(null_t.size)))))
    ax2.hist(null_t, bins=bins, color="#779ECB", edgecolor="white", alpha=0.9)
    ax2.axvline(
        float(biorsp_row["T_obs"]), color="#8B0000", linestyle="--", linewidth=2.0
    )
    ax2.set_xlabel("null_T")
    ax2.set_ylabel("count")
    ax2.set_title("Null T distribution")

    high_c = np.concatenate(
        [
            np.asarray(high_profile, dtype=float),
            np.asarray(high_profile[:1], dtype=float),
        ]
    )
    ax3.plot(th_c, obs_c, color="#D62728", linewidth=2.0, label="low_conf")
    ax3.plot(th_c, high_c, color="#1F77B4", linewidth=1.8, label="high_conf")
    ax3.set_theta_zero_location("E")
    ax3.set_theta_direction(1)
    ax3.set_thetagrids(np.arange(0, 360, 90))
    ax3.set_title("Low vs high profile")
    ax3.legend(loc="upper right", bbox_to_anchor=(1.18, 1.2), fontsize=8, frameon=True)

    fig.suptitle(f"{defn.name}: BioRSP diagnostics", y=1.02)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{defn.name}_biorsp_profiles.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_boundary_metrics(
    *,
    defn: LowConfDefinition,
    boundary_score: np.ndarray,
    random_controls: dict[str, np.ndarray],
    score_vals: np.ndarray | None,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    low = np.asarray(defn.low_mask, dtype=bool)
    high = np.asarray(defn.high_mask, dtype=bool)
    b = np.asarray(boundary_score, dtype=float)
    b_low = b[low]
    b_high = b[high]
    b_rand = np.asarray(random_controls["boundary_random_pooled"], dtype=float)

    fig = plt.figure(figsize=(16.0, 4.9))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    data = [b_low, b_high, b_rand]
    labels = ["low_conf", "high_conf", "random"]
    ax1.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        boxprops={"facecolor": "#9ecae1"},
        medianprops={"color": "#8B0000", "linewidth": 1.3},
    )
    ax1.set_ylabel("boundary_score")
    ax1.set_title("Boundary score distribution")
    ax1.grid(axis="y", alpha=0.25, linewidth=0.6)

    x_low, y_low = _ecdf(b_low)
    x_high, y_high = _ecdf(b_high)
    ax2.plot(x_low, y_low, color="#D62728", linewidth=2.0, label="low_conf")
    ax2.plot(x_high, y_high, color="#1F77B4", linewidth=2.0, label="high_conf")
    ax2.set_xlabel("boundary_score")
    ax2.set_ylabel("ECDF")
    ax2.set_title("Boundary ECDF")
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.legend(loc="lower right", fontsize=8, frameon=True)

    if score_vals is not None:
        s = np.asarray(score_vals, dtype=float)
        finite = np.isfinite(s) & np.isfinite(b)
        ax3.scatter(
            s[finite],
            b[finite],
            c=np.where(low[finite], "#D62728", "#9A9A9A"),
            s=8.0,
            alpha=0.45,
            linewidths=0,
            rasterized=True,
        )
        if int(finite.sum()) >= 3:
            order = np.argsort(s[finite], kind="mergesort")
            xs = s[finite][order]
            ys = b[finite][order]
            # running median smoother for interpretability
            win = max(25, int(0.03 * xs.size))
            med_x = []
            med_y = []
            for start in range(0, xs.size, max(1, win // 5)):
                stop = min(xs.size, start + win)
                if stop - start < 10:
                    continue
                med_x.append(float(np.median(xs[start:stop])))
                med_y.append(float(np.median(ys[start:stop])))
            if len(med_x) >= 2:
                ax3.plot(
                    np.asarray(med_x), np.asarray(med_y), color="black", linewidth=2.0
                )
        ax3.set_xlabel("confidence score")
        ax3.set_ylabel("boundary_score")
        ax3.set_title("Confidence vs boundary")
        ax3.grid(alpha=0.25, linewidth=0.6)
    else:
        ax3.axis("off")
        ax3.set_title("Confidence vs boundary")
        ax3.text(
            0.5, 0.5, "No numeric confidence score available.", ha="center", va="center"
        )

    fig.suptitle(f"{defn.name}: boundary metrics", y=1.02)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{defn.name}_boundary_metrics.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_random_controls(
    *,
    defn: LowConfDefinition,
    mean_b_low: float,
    t_obs: float,
    random_controls: dict[str, np.ndarray],
    p_emp: float,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_rand = np.asarray(random_controls["mean_boundary_random"], dtype=float)
    t_rand = np.asarray(random_controls["T_random"], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    ax1.hist(
        mean_rand,
        bins=max(8, min(18, mean_rand.size)),
        color="#72B7B2",
        edgecolor="white",
        alpha=0.92,
    )
    ax1.axvline(float(mean_b_low), color="#8B0000", linestyle="--", linewidth=1.8)
    ax1.set_xlabel("mean(boundary_score) random foreground")
    ax1.set_ylabel("count")
    ax1.set_title(f"Random control means (p_emp={float(p_emp):.3f})")
    ax1.grid(axis="y", alpha=0.25, linewidth=0.6)

    ax2.hist(
        t_rand,
        bins=max(8, min(18, t_rand.size)),
        color="#4C78A8",
        edgecolor="white",
        alpha=0.92,
    )
    ax2.axvline(float(t_obs), color="#8B0000", linestyle="--", linewidth=1.8)
    ax2.set_xlabel("T_random")
    ax2.set_ylabel("count")
    ax2.set_title("Random foreground BioRSP T")
    ax2.grid(axis="y", alpha=0.25, linewidth=0.6)

    fig.suptitle(f"{defn.name}: random-mask controls", y=1.02)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{defn.name}_random_controls.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_marker_checks(
    *,
    defn: LowConfDefinition,
    marker_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = marker_summary.loc[marker_summary["definition"] == defn.name].copy()
    gene_rows = sub.loc[sub["feature_type"] == "gene"].copy()
    pair_rows = sub.loc[sub["feature_type"] == "pair"].copy()

    if gene_rows.empty:
        _save_placeholder(
            out_dir / f"{defn.name}_marker_checks.png",
            f"{defn.name} marker checks",
            "No marker genes resolved.",
        )
        return

    pivot_mean = gene_rows.pivot_table(
        index="contrast_group",
        columns="feature",
        values="mean_log1p",
        aggfunc="first",
    )
    pivot_frac = gene_rows.pivot_table(
        index="contrast_group",
        columns="feature",
        values="frac_expr",
        aggfunc="first",
    )
    markers = pivot_mean.columns.astype(str).tolist()
    for required in markers:
        if required not in pivot_frac.columns:
            pivot_frac[required] = np.nan

    low_mean = (
        pivot_mean.loc["low_conf", markers].to_numpy(dtype=float)
        if "low_conf" in pivot_mean.index
        else np.full(len(markers), np.nan)
    )
    high_mean = (
        pivot_mean.loc["high_conf", markers].to_numpy(dtype=float)
        if "high_conf" in pivot_mean.index
        else np.full(len(markers), np.nan)
    )
    low_frac = (
        pivot_frac.loc["low_conf", markers].to_numpy(dtype=float)
        if "low_conf" in pivot_frac.index
        else np.full(len(markers), np.nan)
    )
    high_frac = (
        pivot_frac.loc["high_conf", markers].to_numpy(dtype=float)
        if "high_conf" in pivot_frac.index
        else np.full(len(markers), np.nan)
    )
    d_mean = low_mean - high_mean
    d_frac = low_frac - high_frac

    fig = plt.figure(figsize=(15.8, 4.9))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    mat = np.vstack([low_mean, high_mean])
    im = ax1.imshow(mat, aspect="auto", cmap="viridis")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["low_conf", "high_conf"])
    ax1.set_xticks(np.arange(len(markers), dtype=int))
    ax1.set_xticklabels(markers, rotation=35, ha="right", fontsize=8)
    ax1.set_title("Marker mean(log1p expr)")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cbar.set_label("mean log1p expr")

    if pair_rows.empty:
        ax2.axis("off")
        ax2.set_title("Dual-marker co-expression")
        ax2.text(0.5, 0.5, "No dual-marker pairs resolved.", ha="center", va="center")
    else:
        pairs = sorted(pair_rows["feature"].astype(str).unique().tolist())
        x = np.arange(len(pairs), dtype=float)
        width = 0.38
        low_vals = []
        high_vals = []
        for p in pairs:
            low_row = pair_rows.loc[
                (pair_rows["feature"] == p)
                & (pair_rows["contrast_group"] == "low_conf")
            ]
            high_row = pair_rows.loc[
                (pair_rows["feature"] == p)
                & (pair_rows["contrast_group"] == "high_conf")
            ]
            low_vals.append(
                float(low_row["coexpr_rate"].iloc[0]) if not low_row.empty else np.nan
            )
            high_vals.append(
                float(high_row["coexpr_rate"].iloc[0]) if not high_row.empty else np.nan
            )
        ax2.bar(x - width / 2, low_vals, width=width, color="#D62728", label="low_conf")
        ax2.bar(
            x + width / 2, high_vals, width=width, color="#1F77B4", label="high_conf"
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(pairs, rotation=35, ha="right", fontsize=8)
        ax2.set_ylabel("co-expression rate")
        ax2.set_title("Dual-marker co-expression")
        ax2.set_ylim(
            0.0,
            max(1e-6, np.nanmax(np.asarray(low_vals + high_vals, dtype=float)) * 1.2),
        )
        ax2.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax2.legend(loc="upper right", fontsize=8, frameon=True)

    ax3.scatter(
        d_mean, d_frac, c="#2C7FB8", s=65, edgecolors="black", linewidths=0.5, alpha=0.9
    )
    for i, gene in enumerate(markers):
        ax3.text(float(d_mean[i]) + 0.01, float(d_frac[i]) + 0.002, gene, fontsize=8)
    ax3.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    ax3.axvline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    ax3.set_xlabel("delta mean expr (low-high)")
    ax3.set_ylabel("delta frac expr (low-high)")
    ax3.set_title("Marker delta summary")
    ax3.grid(alpha=0.25, linewidth=0.6)

    fig.suptitle(f"{defn.name}: marker ambiguity checks", y=1.02)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{defn.name}_marker_checks.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_qc_controls(
    *,
    defn: LowConfDefinition,
    umap_xy: np.ndarray,
    center_xy: np.ndarray,
    score_vals: np.ndarray | None,
    qc_covars: dict[str, np.ndarray | None],
    rho_row: dict[str, Any],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    low = np.asarray(defn.low_mask, dtype=bool)

    available = [(k, v) for k, v in qc_covars.items() if v is not None]
    if len(available) == 0:
        _save_placeholder(
            out_dir / f"{defn.name}_qc_umap_overlay.png",
            "QC overlays",
            "No QC covariates available.",
        )
    else:
        n_pan = len(available)
        fig, axes = plt.subplots(1, n_pan, figsize=(6.0 * n_pan, 5.1))
        if not isinstance(axes, np.ndarray):
            axes_arr = np.array([axes])
        else:
            axes_arr = axes
        for ax, (name, vals) in zip(axes_arr, available):
            x = np.asarray(vals, dtype=float)
            order = np.argsort(x, kind="mergesort")
            ax.scatter(
                umap_xy[:, 0],
                umap_xy[:, 1],
                c="#D4D4D4",
                s=3.0,
                alpha=0.25,
                linewidths=0,
                rasterized=True,
            )
            sc = ax.scatter(
                umap_xy[order, 0],
                umap_xy[order, 1],
                c=x[order],
                cmap="viridis",
                s=6.0,
                alpha=0.9,
                linewidths=0,
                rasterized=True,
            )
            ax.scatter(
                umap_xy[low, 0],
                umap_xy[low, 1],
                facecolors="none",
                edgecolors="#D62728",
                s=12.0,
                linewidths=0.4,
                alpha=0.8,
                rasterized=True,
            )
            ax.scatter(
                [float(center_xy[0])],
                [float(center_xy[1])],
                marker="X",
                s=65,
                c="black",
                edgecolors="white",
                linewidths=0.8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{name} + low_conf outline")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"{defn.name}_qc_umap_overlay.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

    if score_vals is not None and len(available) > 0:
        fig2, axes2 = plt.subplots(
            1, len(available), figsize=(5.5 * len(available), 4.9)
        )
        if not isinstance(axes2, np.ndarray):
            axes2_arr = np.array([axes2])
        else:
            axes2_arr = axes2
        s = np.asarray(score_vals, dtype=float)
        finite_s = np.isfinite(s)
        for ax, (name, vals) in zip(axes2_arr, available):
            y = np.asarray(vals, dtype=float)
            finite = finite_s & np.isfinite(y)
            ax.scatter(
                s[finite],
                y[finite],
                c=np.where(low[finite], "#D62728", "#808080"),
                s=6.0,
                alpha=0.45,
                linewidths=0,
                rasterized=True,
            )
            ax.set_xlabel("confidence score")
            ax.set_ylabel(name)
            ax.set_title(f"{name} vs confidence")
            ax.grid(alpha=0.25, linewidth=0.6)
        fig2.tight_layout()
        fig2.savefig(
            out_dir / f"{defn.name}_qc_vs_confidence.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
            bbox_inches="tight",
        )
        plt.close(fig2)
    else:
        _save_placeholder(
            out_dir / f"{defn.name}_qc_vs_confidence.png",
            "QC vs confidence",
            "Numeric confidence score unavailable or QC covariates missing.",
        )

    fig3, ax3 = plt.subplots(figsize=(7.6, 4.8))
    ax3.axis("off")
    lines = [
        f"Definition: {defn.name}",
        f"rho_total_counts: {rho_row.get('rho_total_counts', np.nan):.4f}",
        f"rho_pct_counts_mt: {rho_row.get('rho_pct_counts_mt', np.nan):.4f}",
        f"rho_pct_counts_ribo: {rho_row.get('rho_pct_counts_ribo', np.nan):.4f}",
        f"qc_risk: {rho_row.get('qc_risk', np.nan):.4f}",
        f"sim_qc_profile_max: {rho_row.get('sim_qc_profile_max', np.nan):.4f}",
    ]
    ax3.text(
        0.03,
        0.95,
        "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
    )
    ax3.set_title("QC correlation summary")
    fig3.tight_layout()
    fig3.savefig(
        out_dir / f"{defn.name}_qc_rho_summary.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
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
    p_lowumap = plots_dir / "01_lowconf_umaps"
    p_biorsp = plots_dir / "02_biorsp_profiles"
    p_boundary = plots_dir / "03_boundary_metrics"
    p_random = plots_dir / "04_random_controls"
    p_marker = plots_dir / "05_marker_checks"
    p_qc = plots_dir / "06_qc_controls"
    for d in [
        tables_dir,
        p_overview,
        p_lowumap,
        p_biorsp,
        p_boundary,
        p_random,
        p_marker,
        p_qc,
    ]:
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
        warnings_log.append(
            "Donor key unavailable or <2 donors: permutations use global shuffling (no donor stratification)."
        )

    labels, label_key_used = _resolve_label_key_required(adata, args.label_key)

    score_key, score_vals, score_source = _find_numeric_confidence_key(adata)
    matchtype_key, matchtype_vals = _find_matchtype_key(adata)
    if score_key is None and matchtype_key is None:
        raise RuntimeError(
            "Experiment #7 requires a confidence proxy: missing numeric score and cl_match_type."
        )

    defs = _build_lowconf_definitions(
        score_key=score_key,
        score_vals=score_vals,
        matchtype_key=matchtype_key,
        matchtype_vals=matchtype_vals,
        q_low=float(args.q_low),
        q_high=float(args.q_high),
        warnings_log=warnings_log,
    )
    if len(defs) == 0:
        raise RuntimeError(
            "No valid low-confidence definitions could be constructed (all definitions skipped)."
        )

    boundary_df, _ = _compute_boundary_metrics(
        xy=umap_xy,
        labels=labels,
        k=int(args.k),
    )

    total_counts = _total_counts_vector(adata, expr_matrix)
    pct_mt_raw, pct_mt_source = _pct_mt_vector(adata, expr_matrix, adata_like)
    pct_mt = (
        None if pct_mt_source == "proxy:zeros" else np.asarray(pct_mt_raw, dtype=float)
    )
    pct_ribo, pct_ribo_source = _compute_pct_counts_ribo(
        adata, expr_matrix, adata_like, total_counts
    )
    qc_covars: dict[str, np.ndarray | None] = {
        "total_counts": np.asarray(total_counts, dtype=float),
        "pct_counts_mt": pct_mt,
        "pct_counts_ribo": pct_ribo,
    }

    # QC pseudo-feature BioRSP profiles (negative controls).
    qc_profile_rows: list[dict[str, Any]] = []
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
            seed=int(args.seed + 10_000 + i * 61),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        e_obs = np.asarray(perm["E_phi_obs"], dtype=float)
        null_e = np.asarray(perm["null_E_phi"], dtype=float)
        null_t = np.asarray(perm["null_T"], dtype=float)
        qc_profiles[name] = e_obs
        qc_profile_rows.append(
            {
                "definition": f"qc::{name}",
                "definition_type": "qc_pseudofeature",
                "source_key": name,
                "n_fg": int(np.isfinite(vals).sum()),
                "prev": np.nan,
                "T_obs": float(perm["T_obs"]),
                "p_T": float(perm["p_T"]),
                "q_T": np.nan,
                "Z_T": float(robust_z(float(perm["T_obs"]), null_t)),
                "coverage_C": float(coverage_from_null(e_obs, null_e, q=0.95)),
                "peaks_K": int(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95)),
                "phi_hat_deg": float(
                    np.degrees(
                        theta_bin_centers(int(args.n_bins))[
                            int(np.argmax(np.abs(e_obs)))
                        ]
                    )
                    % 360.0
                ),
                "used_donor_stratified": bool(perm["used_donor_stratified"]),
                "notes": "QC pseudo-feature continuous control",
            }
        )

    biorsp_rows: list[dict[str, Any]] = []
    biorsp_artifacts: dict[str, dict[str, np.ndarray]] = {}
    random_controls_map: dict[str, dict[str, np.ndarray]] = {}
    boundary_rows: list[dict[str, Any]] = []
    lowdef_rows: list[dict[str, Any]] = []
    qc_assoc_rows: dict[str, dict[str, Any]] = {}

    label_counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    major_labels = (
        label_counts.loc[label_counts >= max(50, int(0.01 * labels.size))]
        .index.astype(str)
        .tolist()
    )

    for di, defn in enumerate(defs):
        low = np.asarray(defn.low_mask, dtype=bool)
        high = np.asarray(defn.high_mask, dtype=bool)
        n_fg = int(low.sum())
        n_high = int(high.sum())
        prev = float(np.mean(low))
        lowdef_rows.append(
            {
                "definition": defn.name,
                "definition_type": defn.definition_type,
                "source_key": defn.source_key,
                "status": "used",
                "n_fg": n_fg,
                "prev": prev,
                "n_high": n_high,
                "high_prev": float(np.mean(high)),
                "threshold_low": defn.threshold_low,
                "threshold_high": defn.threshold_high,
                "notes": defn.notes,
            }
        )

        summary, art = _foreground_rsp(
            f_mask=low,
            theta=theta,
            n_bins=int(args.n_bins),
            n_perm=int(args.n_perm),
            seed=int(args.seed + 20_000 + di * 1000),
            donor_ids=donor_ids,
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        biorsp_artifacts[defn.name] = art

        high_profile, _, _, _ = compute_rsp_profile_from_boolean(
            high,
            theta,
            int(args.n_bins),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )

        sim_qc_vals: list[tuple[str, float]] = []
        for qname, qprof in qc_profiles.items():
            sim = _cosine_similarity(
                np.asarray(art["E_phi_obs"], dtype=float),
                np.asarray(qprof, dtype=float),
            )
            if np.isfinite(sim):
                sim_qc_vals.append((qname, sim))
        if sim_qc_vals:
            best_qc_name, best_qc_sim = max(sim_qc_vals, key=lambda x: x[1])
        else:
            best_qc_name, best_qc_sim = "", float("nan")

        rho_counts = _safe_spearman(low.astype(float), qc_covars["total_counts"])
        rho_mt = _safe_spearman(low.astype(float), qc_covars["pct_counts_mt"])
        rho_ribo = _safe_spearman(low.astype(float), qc_covars["pct_counts_ribo"])
        finite_rho = [abs(v) for v in [rho_counts, rho_mt, rho_ribo] if np.isfinite(v)]
        qc_risk = float(max(finite_rho)) if finite_rho else 0.0

        row = {
            "definition": defn.name,
            "definition_type": defn.definition_type,
            "source_key": defn.source_key,
            "n_fg": int(n_fg),
            "prev": float(prev),
            "T_obs": float(summary["T_obs"]),
            "p_T": float(summary["p_T"]),
            "q_T": np.nan,
            "Z_T": float(summary["Z_T"]),
            "coverage_C": float(summary["coverage_C"]),
            "peaks_K": int(summary["peaks_K"]),
            "phi_hat_deg": float(summary["phi_hat_deg"]),
            "used_donor_stratified": bool(summary["used_donor_stratified"]),
            "best_qc_feature": best_qc_name,
            "sim_qc_profile_max": float(best_qc_sim),
            "rho_total_counts": float(rho_counts),
            "rho_pct_counts_mt": float(rho_mt),
            "rho_pct_counts_ribo": float(rho_ribo),
            "qc_risk": float(qc_risk),
            "qc_driven_like": bool(
                (float(summary["p_T"]) <= 0.05)
                and (
                    (float(qc_risk) >= QC_RISK_THRESH)
                    or (
                        np.isfinite(best_qc_sim) and float(best_qc_sim) >= QC_SIM_THRESH
                    )
                )
            ),
            "perm_warning": str(summary["perm_warning"]),
        }
        biorsp_rows.append(row)
        qc_assoc_rows[defn.name] = row

        random_controls = _mean_boundary_random_controls(
            low_mask=low,
            boundary_score=boundary_df["boundary_score"].to_numpy(dtype=float),
            theta=theta,
            n_bins=int(args.n_bins),
            random_reps=int(args.random_reps),
            seed=int(args.seed + 30_000 + di * 1000),
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        random_controls_map[defn.name] = random_controls

        b = boundary_df["boundary_score"].to_numpy(dtype=float)
        low_b = b[low]
        high_b = b[high]
        if int(low_b.size) > 0 and int(high_b.size) > 0:
            mw = mannwhitneyu(low_b, high_b, alternative="two-sided")
            cliffs = _cliffs_delta_from_u(
                float(mw.statistic), int(low_b.size), int(high_b.size)
            )
            mw_u = float(mw.statistic)
            mw_p = float(mw.pvalue)
        else:
            mw_u = float("nan")
            mw_p = float("nan")
            cliffs = float("nan")

        mean_low = float(np.mean(low_b)) if low_b.size > 0 else float("nan")
        mean_high = float(np.mean(high_b)) if high_b.size > 0 else float("nan")
        rand_mean = np.asarray(random_controls["mean_boundary_random"], dtype=float)
        p_emp = (
            float((1.0 + np.sum(rand_mean >= mean_low)) / (1.0 + rand_mean.size))
            if rand_mean.size > 0
            else float("nan")
        )

        boundary_rows.append(
            {
                "definition": defn.name,
                "scope": "overall",
                "label": "ALL",
                "n_low": int(low_b.size),
                "n_high": int(high_b.size),
                "mean_B_low": mean_low,
                "mean_B_high": mean_high,
                "mw_U": mw_u,
                "mw_p": mw_p,
                "cliffs_delta": cliffs,
                "mean_B_random_mean": (
                    float(np.mean(rand_mean)) if rand_mean.size > 0 else float("nan")
                ),
                "mean_B_random_sd": (
                    float(np.std(rand_mean)) if rand_mean.size > 0 else float("nan")
                ),
                "p_emp_random": p_emp,
                "T_obs": float(summary["T_obs"]),
                "T_random_mean": float(
                    np.mean(np.asarray(random_controls["T_random"], dtype=float))
                ),
                "T_random_sd": float(
                    np.std(np.asarray(random_controls["T_random"], dtype=float))
                ),
            }
        )

        # Within-label boundary enrichment on major labels.
        for lbl in major_labels:
            lbl_mask = labels == lbl
            low_lbl = low & lbl_mask
            high_lbl = high & lbl_mask
            if int(low_lbl.sum()) < 10 or int(high_lbl.sum()) < 10:
                continue
            b_low_lbl = b[low_lbl]
            b_high_lbl = b[high_lbl]
            mw_l = mannwhitneyu(b_low_lbl, b_high_lbl, alternative="two-sided")
            delta_l = _cliffs_delta_from_u(
                float(mw_l.statistic), int(b_low_lbl.size), int(b_high_lbl.size)
            )
            boundary_rows.append(
                {
                    "definition": defn.name,
                    "scope": "within_label",
                    "label": str(lbl),
                    "n_low": int(b_low_lbl.size),
                    "n_high": int(b_high_lbl.size),
                    "mean_B_low": float(np.mean(b_low_lbl)),
                    "mean_B_high": float(np.mean(b_high_lbl)),
                    "mw_U": float(mw_l.statistic),
                    "mw_p": float(mw_l.pvalue),
                    "cliffs_delta": float(delta_l),
                    "mean_B_random_mean": np.nan,
                    "mean_B_random_sd": np.nan,
                    "p_emp_random": np.nan,
                    "T_obs": np.nan,
                    "T_random_mean": np.nan,
                    "T_random_sd": np.nan,
                }
            )

        _plot_lowconf_umaps(
            defn=defn,
            umap_xy=umap_xy,
            center_xy=np.asarray(center_xy, dtype=float),
            score_vals=score_vals,
            boundary_score=boundary_df["boundary_score"].to_numpy(dtype=float),
            out_dir=p_lowumap,
        )
        _plot_biorsp_profiles(
            defn=defn,
            biorsp_row=pd.Series(row),
            art=art,
            high_profile=np.asarray(high_profile, dtype=float),
            n_bins=int(args.n_bins),
            out_dir=p_biorsp,
        )
        _plot_boundary_metrics(
            defn=defn,
            boundary_score=boundary_df["boundary_score"].to_numpy(dtype=float),
            random_controls=random_controls,
            score_vals=score_vals,
            out_dir=p_boundary,
        )
        _plot_random_controls(
            defn=defn,
            mean_b_low=float(mean_low),
            t_obs=float(summary["T_obs"]),
            random_controls=random_controls,
            p_emp=float(p_emp),
            out_dir=p_random,
        )

    lowconf_biorsp_df = pd.DataFrame(biorsp_rows)
    if not lowconf_biorsp_df.empty:
        lowconf_biorsp_df["q_T"] = bh_fdr(
            lowconf_biorsp_df["p_T"].to_numpy(dtype=float)
        )

    boundary_enrich_df = pd.DataFrame(boundary_rows)

    # Marker ambiguity summaries.
    marker_resolve_df, marker_vectors = _resolve_marker_vectors(adata_like, expr_matrix)
    marker_rows: list[dict[str, Any]] = []
    for defn in defs:
        low = np.asarray(defn.low_mask, dtype=bool)
        high = np.asarray(defn.high_mask, dtype=bool)
        for group, genes in BOUNDARY_MARKER_PANEL.items():
            for gene in genes:
                if gene not in marker_vectors:
                    continue
                x = np.asarray(marker_vectors[gene], dtype=float)
                for gname, gmask in [("low_conf", low), ("high_conf", high)]:
                    vals = x[gmask]
                    marker_rows.append(
                        {
                            "definition": defn.name,
                            "definition_type": defn.definition_type,
                            "contrast_group": gname,
                            "feature_type": "gene",
                            "feature": gene,
                            "panel_group": group,
                            "mean_log1p": (
                                float(np.mean(np.log1p(np.maximum(vals, 0.0))))
                                if vals.size > 0
                                else np.nan
                            ),
                            "frac_expr": (
                                float(np.mean(vals > 0.0)) if vals.size > 0 else np.nan
                            ),
                            "coexpr_rate": np.nan,
                        }
                    )

        for a, b in DUAL_MARKER_PAIRS:
            if a not in marker_vectors or b not in marker_vectors:
                continue
            xa = np.asarray(marker_vectors[a], dtype=float)
            xb = np.asarray(marker_vectors[b], dtype=float)
            both = (xa > 0.0) & (xb > 0.0)
            for gname, gmask in [("low_conf", low), ("high_conf", high)]:
                marker_rows.append(
                    {
                        "definition": defn.name,
                        "definition_type": defn.definition_type,
                        "contrast_group": gname,
                        "feature_type": "pair",
                        "feature": f"{a}+{b}",
                        "panel_group": "dual_marker",
                        "mean_log1p": np.nan,
                        "frac_expr": np.nan,
                        "coexpr_rate": (
                            float(np.mean(both[gmask]))
                            if int(gmask.sum()) > 0
                            else np.nan
                        ),
                    }
                )

    marker_summary_df = pd.DataFrame(marker_rows)

    _plot_overview(
        umap_xy=umap_xy,
        labels=labels,
        donor_ids=donor_ids,
        label_key=label_key_used,
        donor_key=donor_key_used,
        center_xy=np.asarray(center_xy, dtype=float),
        primary_low_mask=np.asarray(defs[0].low_mask, dtype=bool),
        out_dir=p_overview,
    )

    for defn in defs:
        _plot_marker_checks(
            defn=defn,
            marker_summary=marker_summary_df,
            out_dir=p_marker,
        )
        _plot_qc_controls(
            defn=defn,
            umap_xy=umap_xy,
            center_xy=np.asarray(center_xy, dtype=float),
            score_vals=score_vals,
            qc_covars=qc_covars,
            rho_row=qc_assoc_rows.get(defn.name, {}),
            out_dir=p_qc,
        )

    # Tables.
    confidence_keys_rows = [
        {
            "key_type": "label_key",
            "key_used": label_key_used,
            "status": "used",
            "notes": "Required for this experiment.",
        },
        {
            "key_type": "donor_key",
            "key_used": donor_key_used if donor_key_used is not None else "",
            "status": "used" if donor_ids is not None else "missing",
            "notes": "Missing donor triggers global permutations.",
        },
        {
            "key_type": "confidence_score_key",
            "key_used": score_key if score_key is not None else "",
            "status": "used" if score_key is not None else "missing",
            "notes": f"source={score_source}",
        },
        {
            "key_type": "cl_match_type_key",
            "key_used": matchtype_key if matchtype_key is not None else "",
            "status": "used" if matchtype_key is not None else "missing",
            "notes": "Used for match-type ambiguity definition if ambiguous categories match.",
        },
        {
            "key_type": "embedding_key",
            "key_used": embedding_key,
            "status": "used",
            "notes": "2D embedding used for geometry and kNN.",
        },
        {
            "key_type": "expression_source",
            "key_used": expr_source,
            "status": "used",
            "notes": "Expression source for marker checks and QC proxies.",
        },
        {
            "key_type": "pct_counts_mt_source",
            "key_used": pct_mt_source,
            "status": "used" if pct_mt is not None else "missing",
            "notes": "QC covariate source.",
        },
        {
            "key_type": "pct_counts_ribo_source",
            "key_used": pct_ribo_source,
            "status": "used" if pct_ribo is not None else "missing",
            "notes": "QC covariate source.",
        },
    ]
    pd.DataFrame(confidence_keys_rows).to_csv(
        tables_dir / "confidence_keys_used.csv", index=False
    )
    pd.DataFrame(lowdef_rows).to_csv(
        tables_dir / "lowconf_definitions.csv", index=False
    )

    if bool(args.save_per_cell):
        per_cell = boundary_df.copy()
        per_cell["cell_id"] = adata.obs_names.to_numpy(dtype=str)
        per_cell["label"] = labels
        if donor_ids is not None:
            per_cell["donor_id"] = donor_ids
        if score_vals is not None:
            per_cell["confidence_score"] = np.asarray(score_vals, dtype=float)
        for defn in defs:
            per_cell[f"is_{defn.name}"] = np.asarray(defn.low_mask, dtype=bool)
            per_cell[f"is_{defn.name}_high"] = np.asarray(defn.high_mask, dtype=bool)
        per_cell.to_csv(tables_dir / "boundary_metrics_per_cell.csv", index=False)

    lowconf_out = pd.concat(
        [lowconf_biorsp_df, pd.DataFrame(qc_profile_rows)], ignore_index=True
    )
    lowconf_out.to_csv(tables_dir / "lowconf_biorsp_summary.csv", index=False)
    boundary_enrich_df.to_csv(
        tables_dir / "boundary_enrichment_summary.csv", index=False
    )
    marker_out = marker_summary_df.merge(
        marker_resolve_df[["gene", "panel_group", "status"]].rename(
            columns={"gene": "feature", "status": "feature_status"}
        ),
        on=["feature", "panel_group"],
        how="left",
    )
    marker_out.to_csv(tables_dir / "marker_ambiguity_summary.csv", index=False)

    # README interpretation summary.
    summary_lines: list[str] = []
    for defn in defs:
        row = lowconf_biorsp_df.loc[lowconf_biorsp_df["definition"] == defn.name]
        b_row = boundary_enrich_df.loc[
            (boundary_enrich_df["definition"] == defn.name)
            & (boundary_enrich_df["scope"] == "overall")
        ]
        if row.empty or b_row.empty:
            continue
        rr = row.iloc[0]
        bb = b_row.iloc[0]
        qc_driven = bool(rr.get("qc_driven_like", False))
        summary_lines.extend(
            [
                f"Definition: {defn.name}",
                f"- BioRSP anisotropy: p_T={float(rr['p_T']):.3e}, q_T={float(rr['q_T']):.3e}, Z_T={float(rr['Z_T']):.3f}, coverage_C={float(rr['coverage_C']):.3f}, peaks_K={int(rr['peaks_K'])}",
                f"- Boundary enrichment: mean_B_low={float(bb['mean_B_low']):.4f}, mean_B_high={float(bb['mean_B_high']):.4f}, Cliff's delta={float(bb['cliffs_delta']):.4f}, p_emp_random={float(bb['p_emp_random']):.4f}",
                f"- QC association: qc_risk={float(rr['qc_risk']):.4f}, sim_qc_profile_max={float(rr['sim_qc_profile_max']):.4f}, qc_driven_like={qc_driven}",
            ]
        )

    # Marker/doublet plausibility brief.
    dual_rows = marker_summary_df.loc[
        marker_summary_df["feature_type"] == "pair"
    ].copy()
    marker_lines: list[str] = []
    for defn in defs:
        sub = dual_rows.loc[dual_rows["definition"] == defn.name]
        if sub.empty:
            continue
        for pair in sorted(sub["feature"].astype(str).unique().tolist()):
            low_row = sub.loc[
                (sub["feature"] == pair) & (sub["contrast_group"] == "low_conf")
            ]
            high_row = sub.loc[
                (sub["feature"] == pair) & (sub["contrast_group"] == "high_conf")
            ]
            if low_row.empty or high_row.empty:
                continue
            low_rate = float(low_row["coexpr_rate"].iloc[0])
            high_rate = float(high_row["coexpr_rate"].iloc[0])
            marker_lines.append(
                f"- {defn.name} {pair}: coexpr low={low_rate:.4f}, high={high_rate:.4f}, delta={low_rate - high_rate:+.4f}"
            )

    readme_lines = [
        "Experiment #7: Annotation confidence geometry and boundary diagnostics",
        "",
        "Objective: quantify whether low-confidence annotation cells concentrate at label boundaries/transitions.",
        "Boundary enrichment suggests annotation uncertainty concentrates at transitional/ambiguous regions; does not imply new cell types.",
        "BioRSP directions are representation-conditional (embedding direction is not tissue direction).",
        "",
        "Metadata:",
        f"- embedding_key_used: {embedding_key}",
        f"- donor_key_used: {donor_key_used if donor_key_used is not None else 'None'}",
        f"- label_key_used: {label_key_used}",
        f"- confidence_score_key_used: {score_key if score_key is not None else 'None'}",
        f"- cl_match_type_key_used: {matchtype_key if matchtype_key is not None else 'None'}",
        f"- expression_source_used: {expr_source}",
        f"- n_cells: {int(adata.n_obs)}",
        f"- n_bins: {int(args.n_bins)}",
        f"- n_perm: {int(args.n_perm)}",
        f"- k_neighbors: {int(args.k)}",
        f"- q_low: {float(args.q_low):.2f}",
        f"- q_high: {float(args.q_high):.2f}",
        f"- random_reps: {int(args.random_reps)}",
        f"- save_per_cell: {bool(args.save_per_cell)}",
        "",
        "Interpretation guidance:",
        "- Significant low-confidence anisotropy (low p_T / high Z_T) indicates spatially coherent uncertainty regions on the embedding.",
        "- Significant boundary enrichment (empirical p from random controls) supports concentration at label boundaries/transitions.",
        "- High QC association (rho with QC covariates or high profile similarity to QC controls) indicates possible QC-driven uncertainty patterns.",
        "- Dual-marker enrichment is descriptive and can reflect intermediates or potential doublets; it does not define new cell types.",
        "",
        "Per-definition summary:",
    ]
    if summary_lines:
        readme_lines.extend(summary_lines)
    else:
        readme_lines.append("- No analyzable low-confidence definitions.")

    readme_lines.extend(["", "Dual-marker checks:"])
    if marker_lines:
        readme_lines.extend(marker_lines)
    else:
        readme_lines.append("- No resolved dual-marker pairs for comparison.")

    readme_lines.extend(["", "Warnings:"])
    if warnings_log:
        for w in warnings_log:
            readme_lines.append(f"- {w}")
    else:
        readme_lines.append("- none")
    (outdir / "README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"embedding_key_used={embedding_key}")
    print(f"donor_key_used={donor_key_used if donor_key_used is not None else 'None'}")
    print(f"label_key_used={label_key_used}")
    print(f"confidence_score_key_used={score_key if score_key is not None else 'None'}")
    print(
        f"cl_match_type_key_used={matchtype_key if matchtype_key is not None else 'None'}"
    )
    print(f"expression_source_used={expr_source}")
    print(
        f"n_cells={int(adata.n_obs)} n_bins={int(args.n_bins)} n_perm={int(args.n_perm)} k={int(args.k)} random_reps={int(args.random_reps)}"
    )
    print(f"definitions_used={','.join([d.name for d in defs])}")
    print(f"confidence_keys_csv={(tables_dir / 'confidence_keys_used.csv').as_posix()}")
    print(
        f"lowconf_biorsp_csv={(tables_dir / 'lowconf_biorsp_summary.csv').as_posix()}"
    )
    print(
        f"boundary_enrichment_csv={(tables_dir / 'boundary_enrichment_summary.csv').as_posix()}"
    )
    print(
        f"marker_ambiguity_csv={(tables_dir / 'marker_ambiguity_summary.csv').as_posix()}"
    )
    print(f"results_root={outdir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
