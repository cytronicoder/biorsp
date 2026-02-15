#!/usr/bin/env python3
"""CM Experiment #2 (single donor): threshold sensitivity of BioRSP localization.

Hypothesis (pre-registered for this run):
Within one donor, genuine cardiomyocyte programs should show representation-conditional
BioRSP localization that is robust to foreground thresholding. Spurious localized patterns
arising from brittle threshold choices should fail to persist across threshold definitions.

Interpretation boundary:
Peak direction phi is embedding-conditional geometry and not physical tissue direction.
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

# Headless backend for script execution.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

# Allow execution via `python experiments/...`.
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
from biorsp.pipeline.hierarchy import _pct_mt_vector, _resolve_expr_matrix, _total_counts_vector
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

CM_MARKER_PANEL: dict[str, list[str]] = {
    "Contractile/core": ["MYH6", "MYH7", "TTN", "TNNT2", "TNNI3", "ACTC1"],
    "Calcium handling": ["RYR2", "PLN", "ATP2A2"],
}

CM_PANEL_PROVENANCE = (
    "Pre-registered CM panel from CM-1/CM-2 plan: contractile/core, calcium handling, stress."
)

QC_CANDIDATES = {
    "total_counts": ["total_counts", "n_counts", "n_genes_by_counts"],
    "pct_counts_mt": ["pct_counts_mt", "percent.mt", "pct_mt"],
    "pct_counts_ribo": ["pct_counts_ribo", "percent.ribo", "pct_ribo"],
}

CLASS_ORDER = ["Localized–unimodal", "Localized–multimodal", "Not-localized", "Underpowered"]
CLASS_COLORS = {
    "Localized–unimodal": "#1f77b4",
    "Localized–multimodal": "#ff7f0e",
    "Not-localized": "#8a8a8a",
    "Underpowered": "#d62728",
}

UNDERPOWERED_PREV = 0.005
UNDERPOWERED_MIN_FG = 50
Q_SIG = 0.05
QC_RISK_THRESH = 0.35


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


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    coords: np.ndarray
    params: dict[str, Any]


@dataclass(frozen=True)
class ThresholdScheme:
    threshold_id: str
    threshold_label: str
    family: str
    threshold_param: float | None
    enabled: bool
    reason: str


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
        description=(
            "CM Experiment #2: threshold sensitivity of BioRSP localization in a single donor's cardiomyocytes."
        )
    )
    p.add_argument("--h5ad", default="data/processed/HT_pca_umap.h5ad", help="Input .h5ad")
    p.add_argument(
        "--out",
        default="experiments/heart_single_donor_cardiomyocyte/results/cm_experiment2_threshold_sensitivity",
        help="Output directory",
    )
    p.add_argument("--seed", type=int, default=0, help="Global random seed")
    p.add_argument("--n_perm", type=int, default=300, help="Permutation count")
    p.add_argument("--n_bins", type=int, default=64, help="Angular bin count")
    p.add_argument("--k_pca", type=int, default=50, help="PCA dimensionality for embedding construction")
    p.add_argument(
        "--q_list",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20],
        help="Quantile thresholds for top-q foreground definitions",
    )
    p.add_argument(
        "--enable_gene_specific_threshold",
        type=_str2bool,
        default=False,
        help="Enable optional T4 per-gene threshold heuristic (default: False)",
    )
    p.add_argument("--layer", default=None, help="Optional layer override")
    p.add_argument("--use_raw", action="store_true", help="Use adata.raw as expression source")
    p.add_argument("--donor_key", default=None, help="Optional donor key override")
    p.add_argument("--label_key", default=None, help="Optional label key override")
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
        f"No {purpose} key found. Tried: {', '.join(candidates)}. This experiment requires {purpose}."
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
        warning = source in {"X", "raw"}
        return expr_matrix, adata_like, source, warning
    if "counts" in adata.layers:
        expr_matrix, adata_like, source = _resolve_expr_matrix(adata, layer="counts", use_raw=False)
        return expr_matrix, adata_like, source, False
    expr_matrix, adata_like, source = _resolve_expr_matrix(adata, layer=None, use_raw=False)
    return expr_matrix, adata_like, source, True


def _safe_numeric_obs(adata: ad.AnnData, keys: list[str]) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        if key not in adata.obs.columns:
            continue
        vals = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float)
        if int(np.isfinite(vals).sum()) == 0:
            continue
        if np.isnan(vals).any():
            fill = float(np.nanmedian(vals))
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

    ribo_counts = np.asarray(expr_matrix[:, ribo_mask].sum(axis=1)).ravel().astype(float)
    pct_ribo = np.divide(ribo_counts, np.maximum(total_counts, 1e-12)) * 100.0
    return pct_ribo, f"computed:{symbol_col}"


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


def _resolve_gene_panel(adata_like: Any) -> tuple[list[GeneStatus], pd.DataFrame]:
    statuses: list[GeneStatus] = []
    rows: list[dict[str, Any]] = []
    used_idx: set[int] = set()

    for group, genes in CM_MARKER_PANEL.items():
        for gene in genes:
            try:
                idx, label, symbol_col, source = resolve_feature_index(adata_like, gene)
                idx_i = int(idx)
                if idx_i in used_idx:
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
                    used_idx.add(idx_i)
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


def _is_integer_like_matrix(expr_matrix: Any, seed: int, sample_n: int = 200000) -> bool:
    rng = np.random.default_rng(int(seed))
    if hasattr(expr_matrix, "data"):
        data = np.asarray(expr_matrix.data, dtype=float)
    else:
        arr = np.asarray(expr_matrix, dtype=float)
        data = arr.ravel()
    if data.size == 0:
        return False
    if data.size > sample_n:
        idx = rng.choice(data.size, size=sample_n, replace=False)
        data = data[idx]
    data = data[np.isfinite(data)]
    if data.size == 0:
        return False
    return bool(np.all(np.isclose(data, np.round(data), atol=1e-8)))


def _prepare_embedding_input(
    adata_cm: ad.AnnData,
    expr_matrix_cm: Any,
    expr_source: str,
) -> tuple[ad.AnnData, str]:
    """Prepare matrix used to compute PCA/UMAP embeddings."""
    import scanpy as sc

    adata_embed = ad.AnnData(
        X=expr_matrix_cm.copy() if hasattr(expr_matrix_cm, "copy") else np.array(expr_matrix_cm),
        obs=adata_cm.obs.copy(),
    )

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


def _compute_fixed_embeddings(
    adata_embed: ad.AnnData,
    *,
    seed: int,
    k_pca: int,
) -> tuple[list[EmbeddingSpec], int]:
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


def _top_k_mask(expr: np.ndarray, q: float) -> np.ndarray:
    x = np.asarray(expr, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = int(max(1, round(float(q) * n)))
    order = np.argsort(x, kind="mergesort")
    keep = order[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[keep] = True
    return mask


def _gene_specific_threshold(expr: np.ndarray, seed: int) -> float | None:
    """Optional T4 threshold from simple two-component kmeans on log1p(expr)."""
    x = np.asarray(expr, dtype=float).ravel()
    if x.size == 0:
        return None
    x_log = np.log1p(np.maximum(x, 0.0))
    if np.nanstd(x_log) <= 1e-12:
        return None
    try:
        km = KMeans(n_clusters=2, random_state=int(seed), n_init=10)
        km.fit(x_log.reshape(-1, 1))
        centers = np.sort(km.cluster_centers_.ravel())
        cut_log = float(np.mean(centers))
        threshold = float(np.expm1(cut_log))
        if not np.isfinite(threshold):
            return None
        return max(0.0, threshold)
    except Exception:
        return None


def _build_threshold_schemes(
    *,
    q_list: list[float],
    counts_enabled: bool,
    enable_gene_specific: bool,
) -> tuple[list[ThresholdScheme], pd.DataFrame]:
    scored: list[ThresholdScheme] = []
    table_rows: list[dict[str, Any]] = []

    def add_scheme(s: ThresholdScheme) -> None:
        scored.append(s)
        table_rows.append(
            {
                "threshold_id": s.threshold_id,
                "threshold_label": s.threshold_label,
                "family": s.family,
                "threshold_param": "" if s.threshold_param is None else s.threshold_param,
                "enabled": s.enabled,
                "reason": s.reason,
            }
        )

    add_scheme(
        ThresholdScheme(
            threshold_id="T0_detection",
            threshold_label="T0: expr > 0",
            family="T0",
            threshold_param=0.0,
            enabled=True,
            reason="Baseline detection threshold",
        )
    )

    if counts_enabled:
        add_scheme(
            ThresholdScheme(
                threshold_id="T1_abs_ge1",
                threshold_label="T1a: expr >= 1",
                family="T1",
                threshold_param=1.0,
                enabled=True,
                reason="Absolute threshold valid on integer-like counts",
            )
        )
        add_scheme(
            ThresholdScheme(
                threshold_id="T1_abs_ge2",
                threshold_label="T1b: expr >= 2",
                family="T1",
                threshold_param=2.0,
                enabled=True,
                reason="Absolute threshold valid on integer-like counts",
            )
        )
    else:
        table_rows.append(
            {
                "threshold_id": "T1_abs_ge1",
                "threshold_label": "T1a: expr >= 1",
                "family": "T1",
                "threshold_param": 1.0,
                "enabled": False,
                "reason": "Disabled: counts layer unavailable or non-integer-like",
            }
        )
        table_rows.append(
            {
                "threshold_id": "T1_abs_ge2",
                "threshold_label": "T1b: expr >= 2",
                "family": "T1",
                "threshold_param": 2.0,
                "enabled": False,
                "reason": "Disabled: counts layer unavailable or non-integer-like",
            }
        )

    for q in q_list:
        qv = float(q)
        q_tag = f"{int(round(qv * 100)):02d}"
        add_scheme(
            ThresholdScheme(
                threshold_id=f"T2_top_q{q_tag}",
                threshold_label=f"T2: top {int(round(qv * 100))}%",
                family="T2",
                threshold_param=qv,
                enabled=True,
                reason="Global quantile threshold within CM subset",
            )
        )

    # Explicitly record T3 aliasing, but do not double-score redundant schemes.
    table_rows.append(
        {
            "threshold_id": "T3_local_quantile",
            "threshold_label": "T3: local quantile (alias of T2 in CM-only subset)",
            "family": "T3",
            "threshold_param": "same as T2",
            "enabled": False,
            "reason": "Conceptually explicit but redundant in CM-only subset; not separately scored",
        }
    )

    if enable_gene_specific:
        add_scheme(
            ThresholdScheme(
                threshold_id="T4_gene_specific",
                threshold_label="T4: per-gene heuristic threshold",
                family="T4",
                threshold_param=None,
                enabled=True,
                reason="Optional gene-specific cutoff from two-component log-expression kmeans",
            )
        )
    else:
        table_rows.append(
            {
                "threshold_id": "T4_gene_specific",
                "threshold_label": "T4: per-gene heuristic threshold",
                "family": "T4",
                "threshold_param": "",
                "enabled": False,
                "reason": "Disabled by CLI flag",
            }
        )

    table_df = pd.DataFrame(table_rows)
    return scored, table_df


def _build_foreground(
    *,
    expr: np.ndarray,
    scheme: ThresholdScheme,
    gene_specific_thr: float | None,
) -> tuple[np.ndarray, float | None]:
    x = np.asarray(expr, dtype=float).ravel()

    if scheme.threshold_id == "T0_detection":
        return x > 0.0, 0.0

    if scheme.threshold_id.startswith("T1_abs_ge"):
        thr = float(scheme.threshold_param if scheme.threshold_param is not None else 0.0)
        return x >= thr, thr

    if scheme.threshold_id.startswith("T2_top_q"):
        q = float(scheme.threshold_param if scheme.threshold_param is not None else 0.10)
        return _top_k_mask(x, q=q), q

    if scheme.threshold_id == "T4_gene_specific":
        if gene_specific_thr is None or not np.isfinite(float(gene_specific_thr)):
            return np.zeros(x.size, dtype=bool), float("nan")
        thr = float(gene_specific_thr)
        return x >= thr, thr

    raise ValueError(f"Unknown threshold scheme: {scheme.threshold_id}")


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


def _assign_bh_per_group(df: pd.DataFrame, p_col: str, group_cols: list[str], out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    for _, idx in out.groupby(group_cols).groups.items():
        p = out.loc[idx, p_col].to_numpy(dtype=float)
        finite = np.isfinite(p)
        if int(finite.sum()) == 0:
            continue
        q = np.full(p.shape, np.nan, dtype=float)
        q[finite] = bh_fdr(p[finite])
        out.loc[idx, out_col] = q
    return out


def _class_from_row(q_within: float, peaks_k: float, underpowered: bool) -> str:
    if bool(underpowered):
        return "Underpowered"
    if np.isfinite(float(q_within)) and float(q_within) <= Q_SIG:
        if int(peaks_k) >= 2:
            return "Localized–multimodal"
        return "Localized–unimodal"
    return "Not-localized"


def _circular_stats_deg(phi_deg: np.ndarray) -> tuple[float, float, float]:
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


def _threshold_sort_key(threshold_id: str) -> tuple[int, float]:
    tid = str(threshold_id)
    if tid == "T0_detection":
        return (0, 0.0)
    if tid == "T1_abs_ge1":
        return (1, 1.0)
    if tid == "T1_abs_ge2":
        return (2, 2.0)
    if tid.startswith("T2_top_q"):
        try:
            q = float(tid.split("q")[-1]) / 100.0
        except Exception:
            q = 0.0
        return (3, q)
    if tid == "T4_gene_specific":
        return (4, 0.0)
    return (99, 0.0)


def _score_all_tests(
    *,
    donor_star: str,
    gene_statuses: list[GeneStatus],
    expr_matrix_cm: Any,
    embeddings: list[EmbeddingSpec],
    thresholds_scored: list[ThresholdScheme],
    n_bins: int,
    n_perm: int,
    seed: int,
    qc_total_counts: np.ndarray | None,
    qc_pct_mt: np.ndarray | None,
    qc_pct_ribo: np.ndarray | None,
    out_tables_dir: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    cache_profiles: dict[tuple[str, str, str], dict[str, Any]] = {}
    expr_by_gene: dict[str, np.ndarray] = {}

    threshold_ids = [s.threshold_id for s in thresholds_scored]
    rep_threshold_ids = {"T0_detection", "T2_top_q10"}
    if "T1_abs_ge1" in threshold_ids:
        rep_threshold_ids.add("T1_abs_ge1")
    elif "T2_top_q05" in threshold_ids:
        rep_threshold_ids.add("T2_top_q05")

    test_counter = 0

    for emb_i, emb in enumerate(embeddings):
        center = compute_vantage_point(emb.coords, method="median")
        theta = compute_theta(emb.coords, center)
        _, bin_id = bin_theta(theta, bins=int(n_bins))
        bin_counts_total = np.bincount(bin_id, minlength=int(n_bins)).astype(float)

        for gene_i, status in enumerate(gene_statuses):
            if not status.present or status.gene_idx is None:
                continue
            expr = get_feature_vector(expr_matrix_cm, int(status.gene_idx))
            expr_by_gene[status.gene] = np.asarray(expr, dtype=float)

            gene_specific_thr = None
            if any(s.threshold_id == "T4_gene_specific" for s in thresholds_scored):
                gene_specific_thr = _gene_specific_threshold(expr, seed=seed + gene_i + emb_i * 100)

            for thr_i, scheme in enumerate(thresholds_scored):
                f, param_used = _build_foreground(
                    expr=expr,
                    scheme=scheme,
                    gene_specific_thr=gene_specific_thr,
                )
                n_cells = int(f.size)
                n_fg = int(f.sum())
                prev = float(n_fg / max(1, n_cells))
                underpowered = bool(prev < UNDERPOWERED_PREV or n_fg < UNDERPOWERED_MIN_FG)

                # QC audit on thresholded foreground.
                f_float = f.astype(float)
                rho_counts = _safe_spearman(f_float, qc_total_counts)
                rho_mt = _safe_spearman(f_float, qc_pct_mt)
                rho_ribo = _safe_spearman(f_float, qc_pct_ribo)
                qc_vals = np.array([rho_counts, rho_mt, rho_ribo], dtype=float)
                finite_qc = qc_vals[np.isfinite(qc_vals)]
                qc_risk = float(np.max(np.abs(finite_qc))) if finite_qc.size > 0 else 0.0
                qc_risky = bool(qc_risk >= QC_RISK_THRESH)

                # Observed profile.
                if n_fg == 0 or n_fg == n_cells:
                    e_obs = np.zeros(int(n_bins), dtype=float)
                    t_obs = 0.0
                    phi_hat_deg = float("nan")
                    p_t = float("nan")
                    z_t = float("nan")
                    coverage_c = float("nan")
                    peaks_k = float("nan")
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
                        p_t = float("nan")
                        z_t = float("nan")
                        coverage_c = float("nan")
                        peaks_k = float("nan")
                    else:
                        perm_seed = int(seed + emb_i * 100000 + gene_i * 1000 + thr_i * 19 + 13)
                        perm = perm_null_T_and_profile(
                            expr=f.astype(float),
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
                        peaks_k = float(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95))

                        cache_key = (status.gene, emb.key, scheme.threshold_id)
                        if scheme.threshold_id in rep_threshold_ids:
                            cache_profiles[cache_key] = {
                                "E_phi_obs": np.asarray(e_obs, dtype=float),
                                "null_E_phi": np.asarray(null_e, dtype=float),
                                "null_T": np.asarray(null_t, dtype=float),
                            }

                row = {
                    "donor_id": donor_star,
                    "gene": status.gene,
                    "resolved_gene": status.resolved_gene,
                    "marker_group": status.marker_group,
                    "embedding": emb.key,
                    "threshold_id": scheme.threshold_id,
                    "threshold_label": scheme.threshold_label,
                    "threshold_family": scheme.family,
                    "threshold_param": float(param_used) if param_used is not None and np.isfinite(float(param_used)) else np.nan,
                    "n_cells": n_cells,
                    "prev": prev,
                    "n_fg": n_fg,
                    "T_obs": float(t_obs),
                    "p_T": float(p_t),
                    "q_T_within_threshold": np.nan,
                    "q_T_global_per_embedding": np.nan,
                    "Z_T": float(z_t),
                    "coverage_C": float(coverage_c),
                    "peaks_K": float(peaks_k),
                    "phi_hat_deg": float(phi_hat_deg),
                    "underpowered_flag": bool(underpowered),
                    "rho_counts": float(rho_counts),
                    "rho_mt": float(rho_mt),
                    "rho_ribo": float(rho_ribo),
                    "qc_risk": float(qc_risk),
                    "qc_risky_flag": bool(qc_risky),
                    "params_json": json.dumps(emb.params, sort_keys=True),
                }
                rows.append(row)
                test_counter += 1

                if test_counter % 50 == 0:
                    tmp = pd.DataFrame(rows)
                    tmp.to_csv(out_tables_dir / "per_test_scores_long.intermediate.csv", index=False)
                    print(
                        f"[Progress] scored tests {test_counter}; "
                        f"intermediate -> {out_tables_dir / 'per_test_scores_long.intermediate.csv'}"
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, cache_profiles, expr_by_gene

    # BH per embedding x threshold_id.
    df = _assign_bh_per_group(
        df,
        p_col="p_T",
        group_cols=["embedding", "threshold_id"],
        out_col="q_T_within_threshold",
    )

    # BH global per embedding across gene x threshold tests.
    df = _assign_bh_per_group(
        df,
        p_col="p_T",
        group_cols=["embedding"],
        out_col="q_T_global_per_embedding",
    )

    df["class_label"] = [
        _class_from_row(float(q), float(k), bool(u))
        for q, k, u in zip(
            df["q_T_within_threshold"],
            df["peaks_K"],
            df["underpowered_flag"],
            strict=False,
        )
    ]

    return df, cache_profiles, expr_by_gene


def _summarize_gene_embedding(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (gene, emb), sub in long_df.groupby(["gene", "embedding"], sort=False):
        q = sub["q_T_within_threshold"].to_numpy(dtype=float)
        classes = sub["class_label"].astype(str)
        z = sub["Z_T"].to_numpy(dtype=float)

        sig_fraction = float(np.mean(np.isfinite(q) & (q <= Q_SIG)))
        class_counts = classes.value_counts(dropna=False)
        dominant_class = str(class_counts.index[0]) if len(class_counts) > 0 else "Not-localized"
        stable_class_fraction = float(class_counts.iloc[0] / max(1, len(classes)))

        sig_phi = sub.loc[
            (sub["q_T_within_threshold"] <= Q_SIG) & np.isfinite(sub["phi_hat_deg"]),
            "phi_hat_deg",
        ].to_numpy(dtype=float)
        if sig_phi.size > 0:
            mu_deg, R, circ_sd = _circular_stats_deg(sig_phi)
        else:
            mu_deg, R, circ_sd = float("nan"), float("nan"), float("nan")

        robust_localized = bool(
            sig_fraction >= 0.60
            and stable_class_fraction >= 0.60
            and np.isfinite(R)
            and R >= 0.60
        )

        z_fin = z[np.isfinite(z)]
        median_z = float(np.median(z_fin)) if z_fin.size > 0 else float("nan")

        qc_vals = sub["qc_risk"].to_numpy(dtype=float)
        qc_risky_fraction = float(np.mean(qc_vals >= QC_RISK_THRESH)) if qc_vals.size > 0 else float("nan")

        rows.append(
            {
                "gene": str(gene),
                "embedding": str(emb),
                "marker_group": str(sub["marker_group"].iloc[0]),
                "n_thresholds": int(len(sub)),
                "sig_fraction": sig_fraction,
                "stable_class_fraction": stable_class_fraction,
                "dominant_class": dominant_class,
                "median_Z": median_z,
                "phi_mean_deg": mu_deg,
                "R": R,
                "circ_sd": circ_sd,
                "robust_localized": robust_localized,
                "qc_risky_fraction": qc_risky_fraction,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["embedding", "robust_localized", "sig_fraction", "median_Z"], ascending=[True, False, False, False])
    return out


def _summarize_gene_overall(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    emb_pivot = summary_df.pivot(index="gene", columns="embedding", values="robust_localized")
    frac_pivot = summary_df.pivot(index="gene", columns="embedding", values="sig_fraction")
    R_pivot = summary_df.pivot(index="gene", columns="embedding", values="R")
    class_pivot = summary_df.pivot(index="gene", columns="embedding", values="dominant_class")

    genes = summary_df["gene"].astype(str).unique().tolist()
    rows: list[dict[str, Any]] = []
    for gene in genes:
        robust_pca = bool(emb_pivot.loc[gene, "pca2d"]) if "pca2d" in emb_pivot.columns and gene in emb_pivot.index and pd.notna(emb_pivot.loc[gene, "pca2d"]) else False
        robust_umap = bool(emb_pivot.loc[gene, "umap_repr"]) if "umap_repr" in emb_pivot.columns and gene in emb_pivot.index and pd.notna(emb_pivot.loc[gene, "umap_repr"]) else False
        sig_pca = float(frac_pivot.loc[gene, "pca2d"]) if "pca2d" in frac_pivot.columns and gene in frac_pivot.index and pd.notna(frac_pivot.loc[gene, "pca2d"]) else float("nan")
        sig_umap = float(frac_pivot.loc[gene, "umap_repr"]) if "umap_repr" in frac_pivot.columns and gene in frac_pivot.index and pd.notna(frac_pivot.loc[gene, "umap_repr"]) else float("nan")
        R_pca = float(R_pivot.loc[gene, "pca2d"]) if "pca2d" in R_pivot.columns and gene in R_pivot.index and pd.notna(R_pivot.loc[gene, "pca2d"]) else float("nan")
        R_umap = float(R_pivot.loc[gene, "umap_repr"]) if "umap_repr" in R_pivot.columns and gene in R_pivot.index and pd.notna(R_pivot.loc[gene, "umap_repr"]) else float("nan")
        class_pca = str(class_pivot.loc[gene, "pca2d"]) if "pca2d" in class_pivot.columns and gene in class_pivot.index and pd.notna(class_pivot.loc[gene, "pca2d"]) else ""
        class_umap = str(class_pivot.loc[gene, "umap_repr"]) if "umap_repr" in class_pivot.columns and gene in class_pivot.index and pd.notna(class_pivot.loc[gene, "umap_repr"]) else ""

        rows.append(
            {
                "gene": gene,
                "marker_group": str(summary_df.loc[summary_df["gene"] == gene, "marker_group"].iloc[0]),
                "robust_localized_pca2d": robust_pca,
                "robust_localized_umap_repr": robust_umap,
                "robust_both_embeddings": bool(robust_pca and robust_umap),
                "sig_fraction_pca2d": sig_pca,
                "sig_fraction_umap_repr": sig_umap,
                "R_pca2d": R_pca,
                "R_umap_repr": R_umap,
                "dominant_class_pca2d": class_pca,
                "dominant_class_umap_repr": class_umap,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(by=["robust_both_embeddings", "sig_fraction_umap_repr", "sig_fraction_pca2d"], ascending=[False, False, False])
    return out


def _plot_overview(
    *,
    out_dir: Path,
    donor_counts: pd.DataFrame,
    donor_star: str,
    umap_coords: np.ndarray,
    expr_by_gene: dict[str, np.ndarray],
    threshold_table: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for gene in ["TNNT2", "MYH7"]:
        expr = expr_by_gene.get(gene, None)
        if expr is None:
            _save_placeholder(
                out_dir / f"umap_{gene}.png",
                f"Representative UMAP: {gene}",
                f"{gene} missing in selected expression namespace.",
            )
            continue
        save_numeric_umap(
            umap_coords,
            np.log1p(np.maximum(np.asarray(expr, dtype=float), 0.0)),
            out_dir / f"umap_{gene}.png",
            title=f"Representative UMAP (umap_repr) colored by {gene}",
            cmap="Reds",
            colorbar_label=f"log1p({gene})",
        )

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
    ax1.set_title("Cardiomyocyte counts per donor (donor_star highlighted)")
    ax1.set_xlabel("Donor")
    ax1.set_ylabel("n_cm")
    ax1.tick_params(axis="x", rotation=70)
    fig1.tight_layout()
    fig1.savefig(out_dir / "cm_counts_per_donor.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    # Threshold schemes table plot.
    fig2, ax2 = plt.subplots(figsize=(13.5, 4.8))
    ax2.axis("off")
    show_cols = ["threshold_id", "threshold_label", "enabled", "reason"]
    tbl_df = threshold_table[show_cols].copy()
    tbl_df["enabled"] = tbl_df["enabled"].astype(str)
    tbl = ax2.table(
        cellText=tbl_df.values,
        colLabels=tbl_df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    ax2.set_title("Threshold schemes (T0..T4) and enablement status")
    fig2.tight_layout()
    fig2.savefig(out_dir / "threshold_schemes_table.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)


def _plot_threshold_trajectories(
    *,
    out_dir: Path,
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if long_df.empty:
        _save_placeholder(out_dir / "threshold_trajectories_empty.png", "Threshold trajectories", "No scored tests.")
        return

    for emb, sub_e in long_df.groupby("embedding", sort=False):
        ordered_ids = sorted(sub_e["threshold_id"].astype(str).unique().tolist(), key=_threshold_sort_key)
        x_map = {tid: i for i, tid in enumerate(ordered_ids)}

        robust_genes = set(
            summary_df.loc[
                (summary_df["embedding"] == emb) & (summary_df["robust_localized"]),
                "gene",
            ].astype(str)
        )

        # 1) Z_T threshold trajectories.
        fig1, ax1 = plt.subplots(figsize=(11.0, 6.0))
        for gene, sub_g in sub_e.groupby("gene", sort=False):
            sub_g = sub_g.sort_values(by="threshold_id", key=lambda s: s.map(x_map))
            xs = np.array([x_map[t] for t in sub_g["threshold_id"].astype(str)], dtype=float)
            ys = sub_g["Z_T"].to_numpy(dtype=float)
            lw = 2.5 if str(gene) in robust_genes else 1.2
            alpha = 0.95 if str(gene) in robust_genes else 0.65
            ax1.plot(xs, ys, marker="o", linewidth=lw, alpha=alpha, label=str(gene))
        ax1.set_xticks(np.arange(len(ordered_ids)))
        ax1.set_xticklabels(ordered_ids, rotation=35, ha="right")
        ax1.set_ylabel("Z_T")
        ax1.set_xlabel("Threshold scheme")
        ax1.set_title(f"{emb}: threshold trajectory of Z_T (robust genes emphasized)")
        ax1.axhline(0.0, color="#444444", linewidth=0.8)
        ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=True)
        fig1.tight_layout()
        fig1.savefig(out_dir / f"{emb}_trajectory_ZT.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig1)

        # 2) Score-space trajectories.
        fig2, ax2 = plt.subplots(figsize=(8.5, 6.5))
        cmap = plt.get_cmap("tab10")
        thr_colors = {tid: cmap(i % 10) for i, tid in enumerate(ordered_ids)}
        for gene, sub_g in sub_e.groupby("gene", sort=False):
            sub_g = sub_g.sort_values(by="threshold_id", key=lambda s: s.map(x_map))
            zx = sub_g["Z_T"].to_numpy(dtype=float)
            cy = sub_g["coverage_C"].to_numpy(dtype=float)
            ax2.plot(zx, cy, color="#999999", alpha=0.35, linewidth=1.0)
            for _, row in sub_g.iterrows():
                ax2.scatter(
                    float(row["Z_T"]),
                    float(row["coverage_C"]),
                    c=[thr_colors[str(row["threshold_id"]) ]],
                    s=70,
                    alpha=0.9,
                    edgecolors="black",
                    linewidths=0.3,
                )
            if str(gene) in robust_genes and np.isfinite(zx).any() and np.isfinite(cy).any():
                idx = np.nanargmax(np.nan_to_num(zx, nan=-np.inf))
                ax2.text(float(zx[idx]), float(cy[idx]) + 0.005, str(gene), fontsize=8)

        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=thr_colors[t], markeredgecolor="black", markersize=7, label=t)
            for t in ordered_ids
        ]
        ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=True)
        ax2.set_xlabel("Z_T")
        ax2.set_ylabel("coverage_C")
        ax2.set_title(f"{emb}: score-space trajectories across thresholds")
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{emb}_trajectory_score_space.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig2)

        # 3) Heatmap genes x thresholds for Z_T.
        pivot = sub_e.pivot(index="gene", columns="threshold_id", values="Z_T")
        pivot = pivot.reindex(columns=ordered_ids)
        mat = np.nan_to_num(pivot.to_numpy(dtype=float), nan=0.0)
        mat = np.clip(mat, -10.0, 10.0)

        fig3, ax3 = plt.subplots(figsize=(1.2 * len(ordered_ids) + 3.0, 0.55 * len(pivot.index) + 2.2))
        im = ax3.imshow(mat, aspect="auto", cmap="magma")
        ax3.set_xticks(np.arange(len(ordered_ids)))
        ax3.set_xticklabels(ordered_ids, rotation=35, ha="right")
        ax3.set_yticks(np.arange(len(pivot.index)))
        ax3.set_yticklabels(pivot.index.astype(str), fontsize=8)
        ax3.set_xlabel("Threshold")
        ax3.set_ylabel("Gene")
        ax3.set_title(f"{emb}: Z_T heatmap (capped to [-10,10])")
        cb = fig3.colorbar(im, ax=ax3)
        cb.set_label("Z_T")
        fig3.tight_layout()
        fig3.savefig(out_dir / f"{emb}_heatmap_ZT.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig3)


def _plot_class_stability(out_dir: Path, long_df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if long_df.empty:
        _save_placeholder(out_dir / "class_stability_empty.png", "Class stability", "No scored tests.")
        return

    # 1) Confusion-like transition matrix across ordered thresholds.
    classes = CLASS_ORDER
    class_to_idx = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=float)

    for (_, emb), sub in long_df.groupby(["gene", "embedding"], sort=False):
        sub = sub.sort_values(by="threshold_id", key=lambda s: s.map(lambda x: _threshold_sort_key(str(x))))
        labels = sub["class_label"].astype(str).tolist()
        for a, b in zip(labels[:-1], labels[1:], strict=False):
            ia = class_to_idx.get(a, None)
            ib = class_to_idx.get(b, None)
            if ia is not None and ib is not None:
                matrix[ia, ib] += 1.0

    fig1, ax1 = plt.subplots(figsize=(7.2, 6.4))
    im = ax1.imshow(matrix, cmap="Blues")
    ax1.set_xticks(np.arange(len(classes)))
    ax1.set_xticklabels(classes, rotation=35, ha="right")
    ax1.set_yticks(np.arange(len(classes)))
    ax1.set_yticklabels(classes)
    ax1.set_xlabel("Class at threshold t+1")
    ax1.set_ylabel("Class at threshold t")
    ax1.set_title("Class transition matrix across ordered thresholds")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax1.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=9)
    fig1.colorbar(im, ax=ax1, shrink=0.85)
    fig1.tight_layout()
    fig1.savefig(out_dir / "class_transition_matrix.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    # 2) Stacked bars: class fractions per gene, per embedding.
    for emb, sub_e in long_df.groupby("embedding", sort=False):
        frac = (
            sub_e.groupby(["gene", "class_label"]).size().unstack(fill_value=0)
        )
        for c in classes:
            if c not in frac.columns:
                frac[c] = 0
        frac = frac[classes]
        frac = frac.div(frac.sum(axis=1), axis=0)

        fig2, ax2 = plt.subplots(figsize=(max(8.0, 0.75 * len(frac.index)), 5.8))
        bottom = np.zeros(len(frac.index), dtype=float)
        x = np.arange(len(frac.index))
        for c in classes:
            vals = frac[c].to_numpy(dtype=float)
            ax2.bar(
                x,
                vals,
                bottom=bottom,
                color=CLASS_COLORS.get(c, "#999999"),
                edgecolor="white",
                linewidth=0.6,
                label=c,
            )
            bottom += vals
        ax2.set_xticks(x)
        ax2.set_xticklabels(frac.index.astype(str), rotation=45, ha="right")
        ax2.set_ylabel("Fraction of thresholds")
        ax2.set_xlabel("Gene")
        ax2.set_title(f"{emb}: class fraction across thresholds")
        ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=True)
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{emb}_class_fraction_per_gene.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig2)


def _plot_direction_stability(
    out_dir: Path,
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if long_df.empty or summary_df.empty:
        _save_placeholder(out_dir / "direction_stability_empty.png", "Direction stability", "No scored tests.")
        return

    for emb, sum_e in summary_df.groupby("embedding", sort=False):
        genes = sum_e["gene"].astype(str).tolist()
        n = len(genes)
        n_cols = 4
        n_rows = int(np.ceil(n / n_cols))

        fig = plt.figure(figsize=(4.0 * n_cols, 3.7 * n_rows))
        for i, gene in enumerate(genes):
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="polar")
            sub = long_df.loc[
                (long_df["embedding"] == emb)
                & (long_df["gene"] == gene)
                & (long_df["q_T_within_threshold"] <= Q_SIG)
            ]
            phi = sub["phi_hat_deg"].to_numpy(dtype=float)
            phi = phi[np.isfinite(phi)]
            if phi.size == 0:
                ax.text(0.5, 0.5, "No sig thresholds", transform=ax.transAxes, ha="center", va="center", fontsize=8)
            else:
                rad = np.deg2rad(phi)
                ax.scatter(rad, np.ones_like(rad), s=30, c="#1f77b4", alpha=0.85)
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

        fig.suptitle(f"{emb}: phi stability across significant thresholds", y=0.995)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
        fig.savefig(out_dir / f"{emb}_phi_circular_small_multiples.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig)

        # R vs sig_fraction and circ_sd vs sig_fraction.
        fig2, ax2 = plt.subplots(figsize=(7.6, 5.8))
        ax2.scatter(
            sum_e["R"].to_numpy(dtype=float),
            sum_e["sig_fraction"].to_numpy(dtype=float),
            s=100,
            c=[CLASS_COLORS.get(str(c), "#777777") for c in sum_e["dominant_class"].astype(str)],
            edgecolors="black",
            linewidths=0.4,
            alpha=0.9,
        )
        for _, row in sum_e.iterrows():
            ax2.text(float(row["R"]), float(row["sig_fraction"]) + 0.01, str(row["gene"]), fontsize=8)
        ax2.axvline(0.60, color="#444444", linestyle="--", linewidth=1.0)
        ax2.axhline(0.60, color="#444444", linestyle=":", linewidth=1.0)
        ax2.set_xlabel("R (direction stability)")
        ax2.set_ylabel("sig_fraction")
        ax2.set_title(f"{emb}: R vs sig_fraction")
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{emb}_R_vs_sig_fraction.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(7.6, 5.8))
        ax3.scatter(
            sum_e["circ_sd"].to_numpy(dtype=float),
            sum_e["sig_fraction"].to_numpy(dtype=float),
            s=100,
            c=[CLASS_COLORS.get(str(c), "#777777") for c in sum_e["dominant_class"].astype(str)],
            edgecolors="black",
            linewidths=0.4,
            alpha=0.9,
        )
        for _, row in sum_e.iterrows():
            ax3.text(float(row["circ_sd"]), float(row["sig_fraction"]) + 0.01, str(row["gene"]), fontsize=8)
        ax3.set_xlabel("circ_sd")
        ax3.set_ylabel("sig_fraction")
        ax3.set_title(f"{emb}: circ_sd vs sig_fraction")
        fig3.tight_layout()
        fig3.savefig(out_dir / f"{emb}_circsd_vs_sig_fraction.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig3)


def _plot_per_gene_panels(
    *,
    out_dir: Path,
    genes_present: list[str],
    long_df: pd.DataFrame,
    embeddings_map: dict[str, EmbeddingSpec],
    expr_by_gene: dict[str, np.ndarray],
    cache_profiles: dict[tuple[str, str, str], dict[str, Any]],
    threshold_ids_available: list[str],
    n_bins: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(genes_present) == 0:
        _save_placeholder(out_dir / "no_genes.png", "Per-gene panels", "No panel genes resolved.")
        return

    rep_thresholds = ["T0_detection", "T2_top_q10"]
    if "T1_abs_ge1" in threshold_ids_available:
        rep_thresholds.append("T1_abs_ge1")
    elif "T2_top_q05" in threshold_ids_available:
        rep_thresholds.append("T2_top_q05")

    for gene in genes_present:
        expr = expr_by_gene.get(gene, None)
        if expr is None:
            continue

        x_plot = np.log1p(np.maximum(np.asarray(expr, dtype=float), 0.0))
        if np.isfinite(x_plot).sum() > 0:
            vmin = float(np.quantile(x_plot, 0.01))
            vmax = float(np.quantile(x_plot, 0.99))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6
        else:
            vmin, vmax = 0.0, 1.0

        fig = plt.figure(figsize=(20.5, 8.6))
        gs = fig.add_gridspec(2, 5, wspace=0.20, hspace=0.28)

        for row_i, emb_key in enumerate(["pca2d", "umap_repr"]):
            if emb_key not in embeddings_map:
                continue
            coords = embeddings_map[emb_key].coords
            sub = long_df.loc[(long_df["gene"] == gene) & (long_df["embedding"] == emb_key)].copy()
            sub = sub.sort_values(by="threshold_id", key=lambda s: s.map(lambda x: _threshold_sort_key(str(x))))

            # A) Feature plot
            ax_feat = fig.add_subplot(gs[row_i, 0])
            order = np.argsort(x_plot, kind="mergesort")
            ax_feat.scatter(coords[:, 0], coords[:, 1], c="#dddddd", s=5, alpha=0.28, linewidths=0, rasterized=True)
            sc = ax_feat.scatter(
                coords[order, 0],
                coords[order, 1],
                c=x_plot[order],
                cmap="Reds",
                s=7,
                alpha=0.9,
                linewidths=0,
                rasterized=True,
                vmin=vmin,
                vmax=vmax,
            )
            ax_feat.set_title(f"{emb_key}: log1p expr")
            ax_feat.set_xticks([])
            ax_feat.set_yticks([])
            if row_i == 0:
                cax = fig.add_axes([0.91, 0.60, 0.012, 0.24])
                cb = fig.colorbar(sc, cax=cax)
                cb.set_label("log1p(expr)")

            # B) Three representative polar plots.
            for col_j, thr_id in enumerate(rep_thresholds):
                ax_pol = fig.add_subplot(gs[row_i, col_j + 1], projection="polar")
                key = (gene, emb_key, thr_id)
                row_match = sub.loc[sub["threshold_id"] == thr_id]
                if row_match.empty:
                    ax_pol.text(0.5, 0.5, f"{thr_id}\nnot scored", transform=ax_pol.transAxes, ha="center", va="center", fontsize=8)
                    ax_pol.set_xticks([])
                    ax_pol.set_yticks([])
                    continue

                r = row_match.iloc[0]
                cache = cache_profiles.get(key, None)
                if cache is None or cache.get("null_E_phi") is None:
                    ax_pol.text(0.5, 0.5, f"{thr_id}\nunderpowered/no null", transform=ax_pol.transAxes, ha="center", va="center", fontsize=8)
                    ax_pol.set_xticks([])
                    ax_pol.set_yticks([])
                    continue

                e_obs = np.asarray(cache["E_phi_obs"], dtype=float)
                null_e = np.asarray(cache["null_E_phi"], dtype=float)
                centers = theta_bin_centers(int(n_bins))
                th = np.concatenate([centers, centers[:1]])
                obs = np.concatenate([e_obs, e_obs[:1]])
                hi = np.quantile(null_e, 0.95, axis=0)
                lo = np.quantile(null_e, 0.05, axis=0)
                hi_c = np.concatenate([hi, hi[:1]])
                lo_c = np.concatenate([lo, lo[:1]])

                ax_pol.plot(th, obs, color="#8B0000", linewidth=2.0)
                ax_pol.plot(th, hi_c, color="#333333", linestyle="--", linewidth=1.1)
                ax_pol.plot(th, lo_c, color="#333333", linestyle="--", linewidth=0.9)
                ax_pol.fill_between(th, lo_c, hi_c, color="#999999", alpha=0.20)
                ax_pol.set_theta_zero_location("E")
                ax_pol.set_theta_direction(1)
                ax_pol.set_title(
                    f"{thr_id}\nZ={float(r['Z_T']):.2f}, q={float(r['q_T_within_threshold']):.2e}\n"
                    f"C={float(r['coverage_C']):.3f}, K={int(r['peaks_K']) if np.isfinite(r['peaks_K']) else -1}, "
                    f"phi={float(r['phi_hat_deg']):.1f}",
                    fontsize=8,
                )

            # C) Threshold summary table in last column.
            ax_tbl = fig.add_subplot(gs[row_i, 4])
            ax_tbl.axis("off")
            table_df = sub[["threshold_id", "Z_T", "q_T_within_threshold", "class_label"]].copy()
            table_df["Z_T"] = table_df["Z_T"].map(lambda x: "nan" if not np.isfinite(float(x)) else f"{float(x):.2f}")
            table_df["q_T_within_threshold"] = table_df["q_T_within_threshold"].map(
                lambda x: "nan" if not np.isfinite(float(x)) else f"{float(x):.2e}"
            )
            tbl = ax_tbl.table(
                cellText=table_df.values,
                colLabels=["threshold", "Z_T", "q", "class"],
                loc="center",
                cellLoc="left",
                colLoc="left",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1.0, 1.12)
            ax_tbl.set_title(f"{emb_key}: all thresholds", fontsize=9)

        fig.suptitle(f"{gene}: threshold sensitivity panels (PCA + UMAP)", y=0.995, fontsize=13)
        fig.tight_layout(rect=[0.0, 0.0, 0.90, 0.98])
        fig.savefig(out_dir / f"gene_{gene}.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig)


def _plot_qc_controls(
    *,
    out_dir: Path,
    long_df: pd.DataFrame,
    umap_coords: np.ndarray,
    qc_pct_mt: np.ndarray | None,
    expr_by_gene: dict[str, np.ndarray],
    thresholds_scored: list[ThresholdScheme],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if long_df.empty:
        _save_placeholder(out_dir / "qc_controls_empty.png", "QC controls", "No scored tests.")
        return

    # 1) qc_risk vs Z_T
    fig1, ax1 = plt.subplots(figsize=(8.6, 6.1))
    thr_ids = sorted(long_df["threshold_id"].astype(str).unique().tolist(), key=_threshold_sort_key)
    cmap = plt.get_cmap("tab10")
    color_map = {tid: cmap(i % 10) for i, tid in enumerate(thr_ids)}

    for tid in thr_ids:
        sub = long_df.loc[long_df["threshold_id"] == tid]
        ax1.scatter(
            sub["qc_risk"].to_numpy(dtype=float),
            sub["Z_T"].to_numpy(dtype=float),
            s=60,
            c=[color_map[tid]],
            alpha=0.72,
            edgecolors="black",
            linewidths=0.25,
            label=tid,
        )
    ax1.axvline(QC_RISK_THRESH, color="#444444", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("qc_risk = max(|rho_counts|, |rho_mt|, |rho_ribo|)")
    ax1.set_ylabel("Z_T")
    ax1.set_title("QC risk vs Z_T (points = gene x embedding x threshold)")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=True)
    fig1.tight_layout()
    fig1.savefig(out_dir / "qc_risk_vs_ZT.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    # 2) rho_mt vs rho_counts
    fig2, ax2 = plt.subplots(figsize=(8.0, 6.0))
    risky = long_df["qc_risky_flag"].to_numpy(dtype=bool)
    ax2.scatter(
        long_df["rho_counts"].to_numpy(dtype=float),
        long_df["rho_mt"].to_numpy(dtype=float),
        s=np.where(risky, 90, 55),
        c=np.where(risky, "#d62728", "#4c78a8"),
        alpha=0.78,
        edgecolors="black",
        linewidths=0.3,
    )
    ax2.axvline(0.0, color="#666666", linewidth=0.9)
    ax2.axhline(0.0, color="#666666", linewidth=0.9)
    ax2.set_xlabel("rho_counts")
    ax2.set_ylabel("rho_mt")
    ax2.set_title("QC audit: rho_mt vs rho_counts (red = qc_risk >= 0.35)")
    fig2.tight_layout()
    fig2.savefig(out_dir / "rho_mt_vs_rho_counts.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)

    # 3) UMAP pct_mt with overlay for most QC-risky call.
    if qc_pct_mt is None:
        _save_placeholder(
            out_dir / "umap_pctmt_overlay_most_qc_risky.png",
            "UMAP pct_mt + foreground overlay",
            "pct_counts_mt unavailable; overlay skipped.",
        )
        return

    idx = int(np.nanargmax(long_df["qc_risk"].to_numpy(dtype=float)))
    row = long_df.iloc[idx]
    gene = str(row["gene"])
    thr_id = str(row["threshold_id"])
    thr_param = float(row["threshold_param"]) if np.isfinite(float(row["threshold_param"])) else np.nan

    expr = expr_by_gene.get(gene, None)
    f_overlay = None
    if expr is not None:
        scheme_map = {s.threshold_id: s for s in thresholds_scored}
        scheme = scheme_map.get(thr_id, None)
        if scheme is not None:
            gene_thr = thr_param if thr_id == "T4_gene_specific" and np.isfinite(thr_param) else None
            f_overlay, _ = _build_foreground(expr=np.asarray(expr, dtype=float), scheme=scheme, gene_specific_thr=gene_thr)

    fig3, ax3 = plt.subplots(figsize=(7.6, 6.0))
    order = np.argsort(qc_pct_mt, kind="mergesort")
    sc = ax3.scatter(
        umap_coords[order, 0],
        umap_coords[order, 1],
        c=qc_pct_mt[order],
        cmap="magma",
        s=8,
        alpha=0.88,
        linewidths=0,
        rasterized=True,
    )
    if f_overlay is not None and int(np.sum(f_overlay)) > 0:
        idx_fg = np.flatnonzero(f_overlay)
        ax3.scatter(
            umap_coords[idx_fg, 0],
            umap_coords[idx_fg, 1],
            s=16,
            facecolors="none",
            edgecolors="cyan",
            linewidths=0.7,
            alpha=0.9,
            label=f"foreground: {gene} @ {thr_id}",
        )
        ax3.legend(loc="best", fontsize=8, frameon=True)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title(
        f"umap_repr: pct_counts_mt with overlay of most QC-risky call\n"
        f"gene={gene}, threshold={thr_id}, qc_risk={float(row['qc_risk']):.2f}"
    )
    cb = fig3.colorbar(sc, ax=ax3, fraction=0.046, pad=0.03)
    cb.set_label("pct_counts_mt")
    fig3.tight_layout()
    fig3.savefig(out_dir / "umap_pctmt_overlay_most_qc_risky.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig3)


def _write_readme(
    *,
    out_path: Path,
    seed: int,
    n_perm: int,
    n_bins: int,
    q_list: list[float],
    donor_key: str,
    label_key: str,
    donor_star: str,
    cm_labels: list[str],
    cm_label_counts: dict[str, int],
    expr_source: str,
    expr_warning: bool,
    counts_thresholds_enabled: bool,
    integer_like_counts: bool,
    enable_gene_specific_threshold: bool,
    embedding_prep_note: str,
    n_cells_donor: int,
    n_cells_cm: int,
    cm_underpowered: bool,
    qc_sources: dict[str, str],
    threshold_table: pd.DataFrame,
    per_gene_embedding_summary: pd.DataFrame,
    per_gene_overall_summary: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("CM Experiment #2 (Single-donor): Threshold sensitivity of BioRSP localization")
    lines.append("")
    lines.append("Hypothesis")
    lines.append(
        "Within one donor, genuine cardiomyocyte programs yield BioRSP localization that is robust to "
        "foreground threshold definition, while brittle threshold-driven artifacts will not persist across thresholds."
    )
    lines.append("")
    lines.append("Interpretation guardrail")
    lines.append("BioRSP direction phi is representation-conditional and not physical tissue direction.")
    lines.append("")
    lines.append("Run metadata")
    lines.append(f"- seed: {seed}")
    lines.append(f"- n_perm: {n_perm}")
    lines.append(f"- n_bins: {n_bins}")
    lines.append(f"- q_list: {', '.join([str(float(q)) for q in q_list])}")
    lines.append(f"- donor_key_used: {donor_key}")
    lines.append(f"- label_key_used: {label_key}")
    lines.append(f"- donor_star: {donor_star}")
    lines.append(f"- expression_source_used: {expr_source}")
    lines.append(f"- embedding_input_prep: {embedding_prep_note}")
    lines.append(f"- donor_star_total_cells: {n_cells_donor}")
    lines.append(f"- donor_star_cardiomyocytes: {n_cells_cm}")
    lines.append(f"- cm_underpowered: {cm_underpowered}")
    lines.append("")
    lines.append("Cardiomyocyte labels included")
    for label in cm_labels:
        lines.append(f"- {label}: {cm_label_counts.get(label, 0)}")
    lines.append("")
    lines.append("Thresholding notes")
    lines.append(f"- counts_thresholds_enabled: {counts_thresholds_enabled}")
    lines.append(f"- counts_integer_like: {integer_like_counts}")
    lines.append(f"- enable_gene_specific_threshold(T4): {enable_gene_specific_threshold}")
    lines.append("- T3 local quantile is explicitly documented but aliased to T2 in CM-only subset (not separately scored).")
    lines.append("")
    lines.append("QC covariate sources")
    for key, source in qc_sources.items():
        lines.append(f"- {key}: {source}")
    lines.append("")
    if expr_warning:
        lines.append("Warning")
        lines.append(
            "Counts layer was not selected; absolute count thresholds are not meaningful on log-normalized source. "
            "Only detection/quantile thresholds are interpretably used."
        )
        lines.append("")

    lines.append("Threshold schemes")
    for _, row in threshold_table.iterrows():
        lines.append(
            f"- {row['threshold_id']}: enabled={row['enabled']}, label={row['threshold_label']}, reason={row['reason']}"
        )
    lines.append("")

    if not per_gene_embedding_summary.empty:
        lines.append("Robust localized genes per embedding")
        for emb in ["pca2d", "umap_repr"]:
            sub = per_gene_embedding_summary.loc[
                (per_gene_embedding_summary["embedding"] == emb)
                & (per_gene_embedding_summary["robust_localized"])
            ]
            if sub.empty:
                lines.append(f"- {emb}: none")
            else:
                lines.append(f"- {emb}: " + ", ".join(sub["gene"].astype(str).tolist()))
        lines.append("")

    if not per_gene_overall_summary.empty:
        both = per_gene_overall_summary.loc[per_gene_overall_summary["robust_both_embeddings"]]
        lines.append("Robust in both embeddings")
        if both.empty:
            lines.append("- none")
        else:
            lines.append("- " + ", ".join(both["gene"].astype(str).tolist()))
        lines.append("")

    lines.append("Single-donor rigor note")
    lines.append(
        "No donor replication is available here. Evidence relies on threshold sensitivity analysis, null-calibrated "
        "permutation tests, and QC negative-control audits."
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
        plots_dir / "01_threshold_trajectories",
        plots_dir / "02_class_stability",
        plots_dir / "03_direction_stability",
        plots_dir / "04_per_gene_panels",
        plots_dir / "05_qc_controls",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad)

    donor_key = _resolve_key_required(adata, args.donor_key, DONOR_KEY_CANDIDATES, purpose="donor")
    label_key = _resolve_key_required(adata, args.label_key, LABEL_KEY_CANDIDATES, purpose="cell-type label")

    labels_all = adata.obs[label_key].astype("string").fillna("NA").astype(str)
    cm_mask_all = labels_all.map(_is_cm_label).to_numpy(dtype=bool)
    if int(cm_mask_all.sum()) == 0:
        raise RuntimeError(
            "Cardiomyocyte subset is empty with match rule containing ['cardio','cardiomyocyte','cm']."
        )

    donor_ids_all = adata.obs[donor_key].astype("string").fillna("NA").astype(str)
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
        raise RuntimeError("donor_star cardiomyocyte subset is empty after subsetting.")

    cm_underpowered = bool(int(adata_cm.n_obs) < 2000)
    if cm_underpowered:
        print(
            "WARNING: adata_cm.n_obs < 2000. Running with cm_underpowered=True "
            f"(n_cm={int(adata_cm.n_obs)})."
        )

    expr_matrix_cm, adata_like_cm, expr_source, expr_warning = _choose_expression_source(
        adata_cm,
        layer_arg=args.layer,
        use_raw_arg=bool(args.use_raw),
    )

    # Resolve CM gene panel.
    gene_statuses, gene_panel_df = _resolve_gene_panel(adata_like_cm)
    gene_panel_df.to_csv(tables_dir / "gene_panel_status.csv", index=False)

    # QC covariates in CM subset.
    qc_total_counts, total_key = _safe_numeric_obs(adata_cm, QC_CANDIDATES["total_counts"])
    if qc_total_counts is None:
        qc_total_counts = _total_counts_vector(adata_cm, expr_matrix_cm)
        total_key = "computed:expr_sum"

    qc_pct_mt, mt_key = _safe_numeric_obs(adata_cm, QC_CANDIDATES["pct_counts_mt"])
    if qc_pct_mt is None:
        qc_pct_mt, mt_source = _pct_mt_vector(adata_cm, expr_matrix_cm, adata_like_cm)
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

    # Build fixed embeddings (PCA2D + representative UMAP).
    adata_embed, embed_prep_note = _prepare_embedding_input(adata_cm, expr_matrix_cm, expr_source)
    embeddings, n_pcs_used = _compute_fixed_embeddings(
        adata_embed,
        seed=int(args.seed),
        k_pca=int(args.k_pca),
    )
    embeddings_map = {e.key: e for e in embeddings}

    # Threshold scheme setup.
    counts_integer_like = _is_integer_like_matrix(expr_matrix_cm, seed=int(args.seed))
    counts_thresholds_enabled = bool(expr_source.startswith("layer:counts") and counts_integer_like)

    thresholds_scored, threshold_table = _build_threshold_schemes(
        q_list=[float(q) for q in args.q_list],
        counts_enabled=counts_thresholds_enabled,
        enable_gene_specific=bool(args.enable_gene_specific_threshold),
    )

    # Core scoring loop.
    long_df, cache_profiles, expr_by_gene = _score_all_tests(
        donor_star=donor_star,
        gene_statuses=gene_statuses,
        expr_matrix_cm=expr_matrix_cm,
        embeddings=embeddings,
        thresholds_scored=thresholds_scored,
        n_bins=int(args.n_bins),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        qc_total_counts=np.asarray(qc_total_counts, dtype=float) if qc_total_counts is not None else None,
        qc_pct_mt=np.asarray(qc_pct_mt, dtype=float) if qc_pct_mt is not None else None,
        qc_pct_ribo=np.asarray(qc_pct_ribo, dtype=float) if qc_pct_ribo is not None else None,
        out_tables_dir=tables_dir,
    )

    long_df.to_csv(tables_dir / "per_test_scores_long.csv", index=False)

    per_gene_embedding_summary = _summarize_gene_embedding(long_df)
    per_gene_embedding_summary.to_csv(tables_dir / "per_gene_embedding_summary.csv", index=False)

    per_gene_overall_summary = _summarize_gene_overall(per_gene_embedding_summary)
    per_gene_overall_summary.to_csv(tables_dir / "per_gene_overall_summary.csv", index=False)

    qc_audit = long_df.loc[long_df["qc_risk"] >= QC_RISK_THRESH].copy() if not long_df.empty else pd.DataFrame()
    qc_audit.to_csv(tables_dir / "qc_audit_thresholds.csv", index=False)

    # Plot outputs.
    _plot_overview(
        out_dir=plots_dir / "00_overview",
        donor_counts=donor_choice,
        donor_star=donor_star,
        umap_coords=embeddings_map["umap_repr"].coords,
        expr_by_gene=expr_by_gene,
        threshold_table=threshold_table,
    )

    _plot_threshold_trajectories(
        out_dir=plots_dir / "01_threshold_trajectories",
        long_df=long_df,
        summary_df=per_gene_embedding_summary,
    )

    _plot_class_stability(out_dir=plots_dir / "02_class_stability", long_df=long_df)

    _plot_direction_stability(
        out_dir=plots_dir / "03_direction_stability",
        long_df=long_df,
        summary_df=per_gene_embedding_summary,
    )

    genes_present = [g.gene for g in gene_statuses if g.present and g.gene_idx is not None]
    _plot_per_gene_panels(
        out_dir=plots_dir / "04_per_gene_panels",
        genes_present=genes_present,
        long_df=long_df,
        embeddings_map=embeddings_map,
        expr_by_gene=expr_by_gene,
        cache_profiles=cache_profiles,
        threshold_ids_available=[s.threshold_id for s in thresholds_scored],
        n_bins=int(args.n_bins),
    )

    _plot_qc_controls(
        out_dir=plots_dir / "05_qc_controls",
        long_df=long_df,
        umap_coords=embeddings_map["umap_repr"].coords,
        qc_pct_mt=np.asarray(qc_pct_mt, dtype=float) if qc_pct_mt is not None else None,
        expr_by_gene=expr_by_gene,
        thresholds_scored=thresholds_scored,
    )

    cm_labels_included = sorted(labels_all.loc[cm_mask_all].unique().tolist())
    cm_label_counts = labels_all.loc[cm_mask_all].value_counts().to_dict()

    _write_readme(
        out_path=out_root / "README.txt",
        seed=int(args.seed),
        n_perm=int(args.n_perm),
        n_bins=int(args.n_bins),
        q_list=[float(q) for q in args.q_list],
        donor_key=donor_key,
        label_key=label_key,
        donor_star=donor_star,
        cm_labels=cm_labels_included,
        cm_label_counts={str(k): int(v) for k, v in cm_label_counts.items()},
        expr_source=expr_source,
        expr_warning=bool(expr_warning),
        counts_thresholds_enabled=counts_thresholds_enabled,
        integer_like_counts=counts_integer_like,
        enable_gene_specific_threshold=bool(args.enable_gene_specific_threshold),
        embedding_prep_note=f"{embed_prep_note}; n_pcs_used={n_pcs_used}",
        n_cells_donor=int(adata_donor.n_obs),
        n_cells_cm=int(adata_cm.n_obs),
        cm_underpowered=cm_underpowered,
        qc_sources=qc_sources,
        threshold_table=threshold_table,
        per_gene_embedding_summary=per_gene_embedding_summary,
        per_gene_overall_summary=per_gene_overall_summary,
    )

    # Required output checks.
    required = [
        tables_dir / "donor_choice.csv",
        tables_dir / "gene_panel_status.csv",
        tables_dir / "per_test_scores_long.csv",
        tables_dir / "per_gene_embedding_summary.csv",
        tables_dir / "per_gene_overall_summary.csv",
        tables_dir / "qc_audit_thresholds.csv",
        plots_dir / "00_overview",
        plots_dir / "01_threshold_trajectories",
        plots_dir / "02_class_stability",
        plots_dir / "03_direction_stability",
        plots_dir / "04_per_gene_panels",
        plots_dir / "05_qc_controls",
        out_root / "README.txt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing required outputs: " + ", ".join(missing))

    print(f"donor_key_used={donor_key}")
    print(f"label_key_used={label_key}")
    print(f"donor_star={donor_star}")
    print(f"expression_source_used={expr_source}")
    print(f"counts_integer_like={counts_integer_like}")
    print(f"counts_thresholds_enabled={counts_thresholds_enabled}")
    print(f"thresholds_scored={json.dumps([s.threshold_id for s in thresholds_scored])}")
    print(f"n_tests={int(len(long_df))}")
    print(f"cm_labels_included={json.dumps(cm_labels_included)}")
    print(f"results_root={out_root}")

    if not per_gene_overall_summary.empty:
        both = per_gene_overall_summary.loc[per_gene_overall_summary["robust_both_embeddings"]]
        print(f"robust_both_embeddings_genes={int(len(both))}")
        if len(both) > 0:
            print("robust_both_gene_list=" + ",".join(both["gene"].astype(str).tolist()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
