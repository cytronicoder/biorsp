#!/usr/bin/env python3
"""CM Experiment #3 (single donor): multimodality test for patchy islands vs single axis.

Hypothesis (pre-registered for this run):
Within one cardiomyocyte-heavy donor, some genes show multimodal spatial enrichment
(multiple angular peaks; patchy/multi-region programs) rather than a single gradient-like
axis. These multimodal patterns should be null-significant, reasonably robust across
foreground definitions, linked to coherent sector-level marker differences, and not primarily
QC-driven.

Interpretation guardrail:
Directional angles are representation-conditional geometry (embedding-dependent), not
anatomical/tissue direction.
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

# Non-interactive backend for scripts/CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import combine_pvalues, ranksums, spearmanr

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
from biorsp.pipeline.hierarchy import (
    _pct_mt_vector,
    _resolve_expr_matrix,
    _total_counts_vector,
)
from biorsp.plotting.qc import save_numeric_umap
from biorsp.plotting.styles import DEFAULT_PLOT_STYLE, apply_plot_style
from biorsp.stats.permutation import perm_null_T_and_profile
from biorsp.stats.scoring import bh_fdr, coverage_from_null, robust_z

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

# Candidate multimodality panel.
CANDIDATE_PANEL: dict[str, list[str]] = {
    "Immediate-early/stress": ["JUN", "FOS", "ATF3", "DDIT3", "HSPA1A", "HSPA1B"],
    "Metabolic/maturation": ["PPARGC1A", "CPT1B", "COX4I1"],
}

# Validation markers.
VALIDATION_MARKERS = [
    "TNNT2",
    "TNNI3",
    "ACTC1",
    "MYH6",
    "MYH7",
    "RYR2",
    "ATP2A2",
    "NPPA",
    "NPPB",
    "PPARGC1A",
    "CPT1B",
    "COX4I1",
]

MODULE_DEFS: dict[str, list[str]] = {
    "MODULE_STRESS": [
        "JUN",
        "FOS",
        "ATF3",
        "DDIT3",
        "HSPA1A",
        "HSPA1B",
        "NPPA",
        "NPPB",
    ],
    "MODULE_METABOLIC": ["PPARGC1A", "CPT1B", "COX4I1", "ATP2A2"],
}

UNDERPOWERED_PREV = 0.005
UNDERPOWERED_MIN_FG = 50
Q_SIG = 0.05
QC_RISK_THRESH = 0.35

CLASS_ORDER = [
    "Localized–unimodal",
    "Localized–multimodal",
    "Not-localized",
    "Underpowered",
]
CLASS_COLORS = {
    "Localized–unimodal": "#1f77b4",
    "Localized–multimodal": "#ff7f0e",
    "Not-localized": "#8a8a8a",
    "Underpowered": "#d62728",
}


@dataclass(frozen=True)
class GeneStatus:
    gene: str
    panel_role: str
    panel_group: str
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
        description="CM Experiment #3: multimodality test for patchy CM programs in a single donor."
    )
    p.add_argument(
        "--h5ad", default="data/processed/HT_pca_umap.h5ad", help="Input .h5ad"
    )
    p.add_argument(
        "--out",
        default="experiments/heart_single_donor_cardiomyocyte/results/cm_experiment3_multimodality_patchy_programs",
        help="Output root",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--n_perm", type=int, default=300, help="Permutation count")
    p.add_argument("--n_bins", type=int, default=64, help="Angular bin count")
    p.add_argument("--k_pca", type=int, default=50, help="PCA dimensions used for UMAP")
    p.add_argument(
        "--q", type=float, default=0.10, help="Top-quantile for F1 foreground mode"
    )
    p.add_argument(
        "--peak_width_bins",
        type=int,
        default=2,
        help="Peak distance parameter (bins) and +/- window width for in-peak-region mask",
    )
    p.add_argument(
        "--save_per_cell",
        type=_str2bool,
        default=False,
        help="Write per-cell sector assignments",
    )
    p.add_argument("--layer", default=None, help="Optional expression layer override")
    p.add_argument("--use_raw", action="store_true", help="Use adata.raw")
    p.add_argument("--donor_key", default=None, help="Optional donor key override")
    p.add_argument("--label_key", default=None, help="Optional label key override")
    return p.parse_args()


def _save_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
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
        raise KeyError(f"Requested {purpose} key '{requested}' not found in adata.obs")
    for k in candidates:
        if k in adata.obs.columns:
            return k
    raise KeyError(f"No {purpose} key found; tried: {', '.join(candidates)}")


def _choose_expression_source(
    adata: ad.AnnData,
    layer_arg: str | None,
    use_raw_arg: bool,
) -> tuple[Any, Any, str]:
    if layer_arg is not None or use_raw_arg:
        return _resolve_expr_matrix(adata, layer=layer_arg, use_raw=bool(use_raw_arg))
    if "counts" in adata.layers:
        return _resolve_expr_matrix(adata, layer="counts", use_raw=False)
    return _resolve_expr_matrix(adata, layer=None, use_raw=False)


def _safe_numeric_obs(
    adata: ad.AnnData, keys: list[str]
) -> tuple[np.ndarray | None, str | None]:
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
    arr, key = _safe_numeric_obs(adata, ["pct_counts_ribo", "percent.ribo", "pct_ribo"])
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
    tokens = (
        x.replace("/", " ")
        .replace("_", " ")
        .replace("-", " ")
        .replace("(", " ")
        .replace(")", " ")
        .split()
    )
    return "cm" in tokens


def _prepare_embedding_input(
    adata_cm: ad.AnnData,
    expr_matrix_cm: Any,
    expr_source: str,
) -> tuple[ad.AnnData, str]:
    import scanpy as sc

    adata_embed = ad.AnnData(
        X=(
            expr_matrix_cm.copy()
            if hasattr(expr_matrix_cm, "copy")
            else np.array(expr_matrix_cm)
        ),
        obs=adata_cm.obs.copy(),
    )
    if expr_source.startswith("layer:counts"):
        sc.pp.normalize_total(adata_embed, target_sum=1e4)
        sc.pp.log1p(adata_embed)
        note = "counts->normalize_total(1e4)->log1p"
    elif expr_source == "X":
        note = "X_as_is"
    elif expr_source == "raw":
        note = "raw_as_is"
    else:
        note = f"{expr_source}_as_is"
    return adata_embed, note


def _compute_fixed_embeddings(
    adata_embed: ad.AnnData, seed: int, k_pca: int
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
            params={
                "n_neighbors": 30,
                "min_dist": 0.1,
                "random_state": 0,
                "n_pcs": n_pcs,
            },
        ),
    ]
    return specs, n_pcs


def _resolve_gene_sets(
    adata_like: Any,
) -> tuple[list[GeneStatus], list[GeneStatus], pd.DataFrame]:
    candidate_set = [g for genes in CANDIDATE_PANEL.values() for g in genes]
    validation_set = list(dict.fromkeys(VALIDATION_MARKERS))

    rows: list[dict[str, Any]] = []
    cand_status: list[GeneStatus] = []
    val_status: list[GeneStatus] = []
    used = set()

    for role, genes in [("candidate", candidate_set), ("validation", validation_set)]:
        for gene in genes:
            panel_group = "candidate" if role == "candidate" else "validation"
            try:
                idx, label, sym_col, source = resolve_feature_index(adata_like, gene)
                idx_i = int(idx)
                duplicate = idx_i in used
                if not duplicate:
                    used.add(idx_i)
                status = GeneStatus(
                    gene=gene,
                    panel_role=role,
                    panel_group=panel_group,
                    present=not duplicate,
                    status="duplicate_index" if duplicate else "resolved",
                    resolved_gene="" if duplicate else str(label),
                    gene_idx=None if duplicate else idx_i,
                    resolution_source=source,
                    symbol_column=sym_col or "",
                )
            except KeyError:
                status = GeneStatus(
                    gene=gene,
                    panel_role=role,
                    panel_group=panel_group,
                    present=False,
                    status="missing",
                    resolved_gene="",
                    gene_idx=None,
                    resolution_source="",
                    symbol_column="",
                )

            rows.append(
                {
                    "gene": status.gene,
                    "panel_role": status.panel_role,
                    "panel_group": status.panel_group,
                    "present": status.present,
                    "status": status.status,
                    "resolved_gene": status.resolved_gene,
                    "gene_idx": status.gene_idx if status.gene_idx is not None else "",
                    "resolution_source": status.resolution_source,
                    "symbol_column": status.symbol_column,
                }
            )
            if role == "candidate":
                cand_status.append(status)
            else:
                val_status.append(status)

    return cand_status, val_status, pd.DataFrame(rows)


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


def _classify(q: float, peaks_k: int, underpowered: bool) -> str:
    if underpowered:
        return "Underpowered"
    if np.isfinite(float(q)) and float(q) <= Q_SIG:
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


def _top_quantile_mask(expr: np.ndarray, q: float) -> np.ndarray:
    x = np.asarray(expr, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = int(max(1, round(float(q) * n)))
    idx = np.argsort(x, kind="mergesort")
    keep = idx[-k:]
    out = np.zeros(n, dtype=bool)
    out[keep] = True
    return out


def _detect_peaks_null_calibrated(
    e_obs: np.ndarray,
    null_e: np.ndarray,
    *,
    peak_distance_bins: int,
    n_bins: int,
) -> tuple[list[int], list[float], list[float], list[float], float]:
    obs_abs = np.abs(np.asarray(e_obs, dtype=float).ravel())
    null_arr = np.asarray(null_e, dtype=float)

    dist = int(max(1, peak_distance_bins))

    null_prom_max = np.zeros(null_arr.shape[0], dtype=float)
    for i in range(null_arr.shape[0]):
        x = np.abs(null_arr[i, :])
        peaks_i, props_i = find_peaks(x, distance=dist, prominence=0.0)
        if len(peaks_i) == 0:
            null_prom_max[i] = 0.0
        else:
            null_prom_max[i] = float(
                np.max(np.asarray(props_i.get("prominences", [0.0]), dtype=float))
            )

    prom_thr = float(np.quantile(null_prom_max, 0.95))
    peaks, props = find_peaks(obs_abs, distance=dist, prominence=prom_thr)
    centers = theta_bin_centers(int(n_bins))

    peak_bins = [int(p) for p in np.asarray(peaks, dtype=int).tolist()]
    peak_angles_deg = [float(np.degrees(centers[p]) % 360.0) for p in peak_bins]
    peak_heights = [float(obs_abs[p]) for p in peak_bins]
    prom = np.asarray(
        props.get("prominences", np.zeros(len(peak_bins), dtype=float)), dtype=float
    )
    peak_prom = [float(v) for v in prom.tolist()]

    return peak_bins, peak_angles_deg, peak_heights, peak_prom, prom_thr


def _assign_qvals(long_df: pd.DataFrame) -> pd.DataFrame:
    out = long_df.copy()
    out["q_T_within"] = np.nan
    out["q_T_global_per_embedding"] = np.nan

    for _, idx in out.groupby(["embedding", "foreground_mode"]).groups.items():
        p = out.loc[idx, "p_T"].to_numpy(dtype=float)
        finite = np.isfinite(p)
        if int(finite.sum()) == 0:
            continue
        q = np.full(p.shape, np.nan, dtype=float)
        q[finite] = bh_fdr(p[finite])
        out.loc[idx, "q_T_within"] = q

    for emb, idx in out.groupby("embedding").groups.items():
        _ = emb
        p = out.loc[idx, "p_T"].to_numpy(dtype=float)
        finite = np.isfinite(p)
        if int(finite.sum()) == 0:
            continue
        q = np.full(p.shape, np.nan, dtype=float)
        q[finite] = bh_fdr(p[finite])
        out.loc[idx, "q_T_global_per_embedding"] = q

    out["class_label"] = [
        _classify(float(q), int(k) if np.isfinite(float(k)) else 0, bool(u))
        for q, k, u in zip(
            out["q_T_within"], out["peaks_K"], out["underpowered_flag"], strict=False
        )
    ]
    return out


def _score_tests(
    *,
    donor_star: str,
    candidate_status: list[GeneStatus],
    expr_matrix_cm: Any,
    embeddings: list[EmbeddingSpec],
    n_bins: int,
    n_perm: int,
    q_top: float,
    peak_width_bins: int,
    seed: int,
    qc_total_counts: np.ndarray | None,
    qc_pct_mt: np.ndarray | None,
    qc_pct_ribo: np.ndarray | None,
    out_tables_dir: Path,
) -> tuple[
    pd.DataFrame, dict[tuple[str, str, str], dict[str, Any]], dict[str, np.ndarray]
]:
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    expr_by_gene: dict[str, np.ndarray] = {}

    fg_modes = [
        ("F0_detection", None),
        ("F1_top_quantile", float(q_top)),
    ]

    tcount = 0
    for emb_i, emb in enumerate(embeddings):
        center = compute_vantage_point(emb.coords, method="median")
        theta = compute_theta(emb.coords, center)
        _, bin_id = bin_theta(theta, bins=int(n_bins))
        bin_counts_total = np.bincount(bin_id, minlength=int(n_bins)).astype(float)

        for gene_i, st in enumerate(candidate_status):
            if not st.present or st.gene_idx is None:
                continue

            expr = get_feature_vector(expr_matrix_cm, int(st.gene_idx))
            expr_by_gene[st.gene] = np.asarray(expr, dtype=float)

            for mode_i, (mode, q_param) in enumerate(fg_modes):
                if mode == "F0_detection":
                    f = np.asarray(expr, dtype=float) > 0.0
                    q_value = np.nan
                else:
                    f = _top_quantile_mask(
                        np.asarray(expr, dtype=float), q=float(q_param)
                    )
                    q_value = float(q_param)

                n_cells = int(f.size)
                n_fg = int(f.sum())
                prev = float(n_fg / max(1, n_cells))
                underpowered = bool(
                    prev < UNDERPOWERED_PREV or n_fg < UNDERPOWERED_MIN_FG
                )

                # QC foreground correlations.
                ff = f.astype(float)
                rho_counts = _safe_spearman(ff, qc_total_counts)
                rho_mt = _safe_spearman(ff, qc_pct_mt)
                rho_ribo = _safe_spearman(ff, qc_pct_ribo)
                qc_vals = np.array([rho_counts, rho_mt, rho_ribo], dtype=float)
                finite = qc_vals[np.isfinite(qc_vals)]
                qc_risk = float(np.max(np.abs(finite))) if finite.size > 0 else 0.0

                if n_fg == 0 or n_fg == n_cells:
                    e_obs = np.zeros(int(n_bins), dtype=float)
                    t_obs = 0.0
                    p_t = np.nan
                    z_t = np.nan
                    coverage_c = np.nan
                    peaks_k = np.nan
                    phi_hat_deg = np.nan
                    peak_bins = []
                    peak_angles_deg = []
                    peak_heights = []
                    peak_prominences = []
                    prom_thr = np.nan
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
                    phi_hat_deg = float(
                        np.degrees(theta_bin_centers(int(n_bins))[phi_idx]) % 360.0
                    )

                    if underpowered:
                        p_t = np.nan
                        z_t = np.nan
                        coverage_c = np.nan
                        peaks_k = np.nan
                        peak_bins = []
                        peak_angles_deg = []
                        peak_heights = []
                        peak_prominences = []
                        prom_thr = np.nan
                    else:
                        perm_seed = int(
                            seed + emb_i * 100000 + gene_i * 1000 + mode_i * 37 + 11
                        )
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

                        (
                            peak_bins,
                            peak_angles_deg,
                            peak_heights,
                            peak_prominences,
                            prom_thr,
                        ) = _detect_peaks_null_calibrated(
                            e_obs,
                            null_e,
                            peak_distance_bins=int(max(1, peak_width_bins)),
                            n_bins=int(n_bins),
                        )
                        peaks_k = float(len(peak_bins))

                        cache[(st.gene, emb.key, mode)] = {
                            "E_phi_obs": np.asarray(e_obs, dtype=float),
                            "null_E_phi": np.asarray(null_e, dtype=float),
                            "null_T": np.asarray(null_t, dtype=float),
                            "theta": np.asarray(theta, dtype=float),
                            "bin_id": np.asarray(bin_id, dtype=int),
                            "foreground": np.asarray(f, dtype=bool),
                            "peak_bins": np.asarray(peak_bins, dtype=int),
                            "peak_angles_deg": np.asarray(peak_angles_deg, dtype=float),
                            "prominence_threshold": float(prom_thr),
                        }

                row = {
                    "donor_id": donor_star,
                    "gene": st.gene,
                    "resolved_gene": st.resolved_gene,
                    "embedding": emb.key,
                    "foreground_mode": mode,
                    "q": q_value,
                    "prev": prev,
                    "n_fg": n_fg,
                    "n_cells": n_cells,
                    "T_obs": float(t_obs),
                    "p_T": float(p_t) if np.isfinite(p_t) else np.nan,
                    "Z_T": float(z_t) if np.isfinite(z_t) else np.nan,
                    "coverage_C": (
                        float(coverage_c) if np.isfinite(coverage_c) else np.nan
                    ),
                    "peaks_K": float(peaks_k) if np.isfinite(peaks_k) else np.nan,
                    "phi_hat_deg": (
                        float(phi_hat_deg) if np.isfinite(phi_hat_deg) else np.nan
                    ),
                    "peak_angles_deg": json.dumps([float(x) for x in peak_angles_deg]),
                    "peak_bins": json.dumps([int(x) for x in peak_bins]),
                    "peak_heights": json.dumps([float(x) for x in peak_heights]),
                    "peak_prominences": json.dumps(
                        [float(x) for x in peak_prominences]
                    ),
                    "peak_prom_threshold": (
                        float(prom_thr) if np.isfinite(prom_thr) else np.nan
                    ),
                    "underpowered_flag": bool(underpowered),
                    "rho_total_counts": float(rho_counts),
                    "rho_pct_mt": float(rho_mt),
                    "rho_pct_ribo": float(rho_ribo),
                    "qc_risk": float(qc_risk),
                }
                rows.append(row)
                tcount += 1

                if tcount % 50 == 0:
                    pd.DataFrame(rows).to_csv(
                        out_tables_dir / "per_test_scores_long.intermediate.csv",
                        index=False,
                    )
                    print(
                        f"[Progress] scored tests {tcount}; intermediate -> "
                        f"{out_tables_dir / 'per_test_scores_long.intermediate.csv'}"
                    )

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df, cache, expr_by_gene
    long_df = _assign_qvals(long_df)
    return long_df, cache, expr_by_gene


def _compute_fisher_map(long_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    fisher: dict[tuple[str, str], float] = {}
    if long_df.empty:
        return fisher

    for (gene, emb), sub in long_df.groupby(["gene", "embedding"], sort=False):
        sub_modes = sub.set_index("foreground_mode")
        p0 = (
            float(sub_modes.loc["F0_detection", "p_T"])
            if "F0_detection" in sub_modes.index
            else np.nan
        )
        p1 = (
            float(sub_modes.loc["F1_top_quantile", "p_T"])
            if "F1_top_quantile" in sub_modes.index
            else np.nan
        )
        if np.isfinite(p0) and np.isfinite(p1):
            stat, p = combine_pvalues([p0, p1], method="fisher")
            _ = stat
            fisher[(str(gene), str(emb))] = float(p)
        else:
            fisher[(str(gene), str(emb))] = np.nan
    return fisher


def _summarize_multimodality(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    fisher_map = _compute_fisher_map(long_df)
    rows: list[dict[str, Any]] = []

    for (gene, emb), sub in long_df.groupby(["gene", "embedding"], sort=False):
        q = sub["q_T_within"].to_numpy(dtype=float)
        cls = sub["class_label"].astype(str)
        sig_fraction = float(np.mean(np.isfinite(q) & (q <= Q_SIG)))

        cls_counts = cls.value_counts(dropna=False)
        dominant = str(cls_counts.index[0]) if len(cls_counts) > 0 else "Not-localized"
        stable_class_fraction = float(cls_counts.iloc[0] / max(1, len(sub)))

        sig_phi = sub.loc[
            (sub["q_T_within"] <= Q_SIG) & np.isfinite(sub["phi_hat_deg"]),
            "phi_hat_deg",
        ].to_numpy(dtype=float)
        if sig_phi.size > 0:
            mu_deg, R, circ_sd = _circular_stats_deg(sig_phi)
        else:
            mu_deg, R, circ_sd = np.nan, np.nan, np.nan

        multimodal_sig_any = bool(
            np.any(
                (sub["q_T_within"].to_numpy(dtype=float) <= Q_SIG)
                & (sub["peaks_K"].to_numpy(dtype=float) >= 2.0)
            )
        )

        robust_localized = bool(
            sig_fraction >= 0.60
            and stable_class_fraction >= 0.60
            and np.isfinite(R)
            and float(R) >= 0.60
        )

        qc_risk_max = float(np.nanmax(sub["qc_risk"].to_numpy(dtype=float)))
        qc_flag_basic = bool(
            np.any(
                (sub["q_T_within"].to_numpy(dtype=float) <= Q_SIG)
                & (
                    (
                        np.abs(sub["rho_total_counts"].to_numpy(dtype=float))
                        >= QC_RISK_THRESH
                    )
                    | (
                        np.abs(sub["rho_pct_mt"].to_numpy(dtype=float))
                        >= QC_RISK_THRESH
                    )
                )
            )
        )

        rows.append(
            {
                "gene": str(gene),
                "embedding": str(emb),
                "n_modes": int(len(sub)),
                "sig_fraction": sig_fraction,
                "stable_class_fraction": stable_class_fraction,
                "dominant_class": dominant,
                "phi_mean_deg": mu_deg,
                "R": R,
                "circ_sd": circ_sd,
                "multimodal_sig_any": multimodal_sig_any,
                "robust_localized": robust_localized,
                "qc_risk_max": qc_risk_max,
                "qc_flag_basic": qc_flag_basic,
                "p_fisher_F0_F1": fisher_map.get((str(gene), str(emb)), np.nan),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fisher_F0_F1"] = np.nan
        for emb, idx in out.groupby("embedding").groups.items():
            _ = emb
            p = out.loc[idx, "p_fisher_F0_F1"].to_numpy(dtype=float)
            finite = np.isfinite(p)
            if int(finite.sum()) == 0:
                continue
            q = np.full(p.shape, np.nan, dtype=float)
            q[finite] = bh_fdr(p[finite])
            out.loc[idx, "q_fisher_F0_F1"] = q

        out = out.sort_values(
            by=["embedding", "multimodal_sig_any", "sig_fraction", "R"],
            ascending=[True, False, False, False],
        )
    return out


def _primary_preference_order() -> list[tuple[str, str]]:
    return [
        ("umap_repr", "F1_top_quantile"),
        ("umap_repr", "F0_detection"),
        ("pca2d", "F1_top_quantile"),
        ("pca2d", "F0_detection"),
    ]


def _select_primary_condition(long_df: pd.DataFrame, gene: str) -> pd.Series | None:
    sub = long_df.loc[long_df["gene"] == gene].copy()
    if sub.empty:
        return None

    # Must be multimodal significant condition.
    is_mm = (sub["q_T_within"].to_numpy(dtype=float) <= Q_SIG) & (
        sub["peaks_K"].to_numpy(dtype=float) >= 2.0
    )
    mm = sub.loc[is_mm].copy()
    if mm.empty:
        return None

    pref = _primary_preference_order()
    for emb, mode in pref:
        hit = mm.loc[(mm["embedding"] == emb) & (mm["foreground_mode"] == mode)]
        if not hit.empty:
            return hit.sort_values(by="Z_T", ascending=False).iloc[0]

    return mm.sort_values(by="Z_T", ascending=False).iloc[0]


def _angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise absolute circular distance between angles in radians."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    return np.abs(np.angle(np.exp(1j * (aa - bb))))


def _assign_sectors(
    theta: np.ndarray, peak_bins: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each cell to nearest peak sector; return sector indices and sorted peak angles."""
    th = np.asarray(theta, dtype=float).ravel()
    if th.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    p = np.asarray(peak_bins, dtype=int).ravel()
    p = np.unique(p[(p >= 0) & (p < int(n_bins))])
    if p.size == 0:
        return np.zeros(th.size, dtype=int), np.zeros(0, dtype=float)

    centers = theta_bin_centers(int(n_bins))
    peak_angles = centers[p]
    order = np.argsort(peak_angles)
    peak_angles = peak_angles[order]

    # Nearest-peak assignment is equivalent to half-way boundaries on the circle.
    dmat = _angular_distance(th[:, None], peak_angles[None, :])
    sectors = np.argmin(dmat, axis=1).astype(int)
    return sectors, peak_angles


def _peak_window_mask(
    bin_id: np.ndarray, peak_bins: np.ndarray, n_bins: int, width: int
) -> np.ndarray:
    b = np.asarray(bin_id, dtype=int).ravel()
    p = np.asarray(peak_bins, dtype=int).ravel()
    w = int(max(0, width))
    keep_bins: set[int] = set()
    for pk in p:
        for dx in range(-w, w + 1):
            keep_bins.add(int((int(pk) + dx) % int(n_bins)))
    if len(keep_bins) == 0:
        return np.zeros(b.size, dtype=bool)
    return np.isin(b, np.array(sorted(keep_bins), dtype=int))


def _module_score(
    expr_by_gene: dict[str, np.ndarray], genes: list[str]
) -> np.ndarray | None:
    vecs = []
    for g in genes:
        if g in expr_by_gene:
            vecs.append(
                np.log1p(np.maximum(np.asarray(expr_by_gene[g], dtype=float), 0.0))
            )
    if len(vecs) == 0:
        return None
    stack = np.vstack(vecs)
    return np.mean(stack, axis=0)


def _compute_sector_validation(
    *,
    gene: str,
    exemplar_row: pd.Series,
    cache: dict[tuple[str, str, str], dict[str, Any]],
    expr_by_gene: dict[str, np.ndarray],
    validation_present: list[str],
    qc_total_counts: np.ndarray | None,
    qc_pct_mt: np.ndarray | None,
    qc_pct_ribo: np.ndarray | None,
    n_bins: int,
    peak_width_bins: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    emb = str(exemplar_row["embedding"])
    mode = str(exemplar_row["foreground_mode"])
    ck = (gene, emb, mode)
    if ck not in cache:
        return pd.DataFrame(), pd.DataFrame(), {"qc_flag": False}

    data = cache[ck]
    theta = np.asarray(data["theta"], dtype=float)
    bin_id = np.asarray(data["bin_id"], dtype=int)
    foreground = np.asarray(data["foreground"], dtype=bool)
    peak_bins = np.asarray(data["peak_bins"], dtype=int)
    peak_angles = np.asarray(data["peak_angles_deg"], dtype=float)

    sectors, peak_angles_rad = _assign_sectors(theta, peak_bins, n_bins=int(n_bins))
    n_sectors = int(np.max(sectors) + 1) if sectors.size > 0 else 0

    window_mask = _peak_window_mask(
        bin_id, peak_bins, n_bins=int(n_bins), width=int(max(1, peak_width_bins))
    )
    in_peak_region = foreground & window_mask

    # Per-cell assignment table.
    per_cell = pd.DataFrame(
        {
            "gene": gene,
            "embedding": emb,
            "foreground_mode": mode,
            "cell_idx": np.arange(theta.size, dtype=int),
            "theta_rad": theta,
            "theta_deg": np.degrees(theta) % 360.0,
            "sector_id": sectors,
            "is_foreground": foreground,
            "in_peak_region": in_peak_region,
        }
    )

    marker_rows: list[dict[str, Any]] = []

    # Build marker pool including module scores.
    marker_values: dict[str, np.ndarray] = {}
    for m in validation_present:
        if m in expr_by_gene:
            marker_values[m] = np.log1p(
                np.maximum(np.asarray(expr_by_gene[m], dtype=float), 0.0)
            )

    for module_name, genes in MODULE_DEFS.items():
        mod = _module_score(expr_by_gene, genes)
        if mod is not None:
            marker_values[module_name] = np.asarray(mod, dtype=float)

    # Sector summaries.
    for sid in range(n_sectors):
        mask = sectors == sid
        n_sid = int(mask.sum())
        if n_sid == 0:
            continue
        for marker, vals in marker_values.items():
            x = np.asarray(vals, dtype=float)
            marker_rows.append(
                {
                    "gene": gene,
                    "embedding": emb,
                    "foreground_mode": mode,
                    "analysis_type": "sector_summary",
                    "comparison": f"sector_{sid}",
                    "sector_id": sid,
                    "n_cells_sector": n_sid,
                    "marker": marker,
                    "mean_log1p": float(np.mean(x[mask])),
                    "median_log1p": float(np.median(x[mask])),
                    "frac_expr_gt0": float(np.mean(x[mask] > 0)),
                    "delta_median": np.nan,
                    "p_value": np.nan,
                    "q_value": np.nan,
                    "total_counts_median": (
                        float(np.median(qc_total_counts[mask]))
                        if qc_total_counts is not None
                        else np.nan
                    ),
                    "pct_mt_median": (
                        float(np.median(qc_pct_mt[mask]))
                        if qc_pct_mt is not None
                        else np.nan
                    ),
                    "pct_ribo_median": (
                        float(np.median(qc_pct_ribo[mask]))
                        if qc_pct_ribo is not None
                        else np.nan
                    ),
                }
            )

    # Pairwise Wilcoxon on two largest sectors.
    sector_sizes = np.bincount(sectors, minlength=max(1, n_sectors)).astype(int)
    top2 = (
        np.argsort(sector_sizes)[::-1][:2]
        if sector_sizes.size >= 2
        else np.array([], dtype=int)
    )
    pair_rows: list[dict[str, Any]] = []
    if top2.size == 2:
        a, b = int(top2[0]), int(top2[1])
        ma = sectors == a
        mb = sectors == b
        for marker, vals in marker_values.items():
            x = np.asarray(vals, dtype=float)
            xa = x[ma]
            xb = x[mb]
            if xa.size < 5 or xb.size < 5:
                p = np.nan
                d = np.nan
            else:
                stat, p = ranksums(xa, xb)
                _ = stat
                d = float(np.median(xa) - np.median(xb))
            pair_rows.append(
                {
                    "gene": gene,
                    "embedding": emb,
                    "foreground_mode": mode,
                    "analysis_type": "pairwise_wilcoxon",
                    "comparison": f"sector_{a}_vs_sector_{b}",
                    "sector_id": np.nan,
                    "n_cells_sector": int(xa.size + xb.size),
                    "marker": marker,
                    "mean_log1p": np.nan,
                    "median_log1p": np.nan,
                    "frac_expr_gt0": np.nan,
                    "delta_median": d,
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                    "q_value": np.nan,
                    "total_counts_median": np.nan,
                    "pct_mt_median": np.nan,
                    "pct_ribo_median": np.nan,
                }
            )

        if len(pair_rows) > 0:
            pvals = np.array([r["p_value"] for r in pair_rows], dtype=float)
            finite = np.isfinite(pvals)
            if int(finite.sum()) > 0:
                qvals = np.full(pvals.shape, np.nan, dtype=float)
                qvals[finite] = bh_fdr(pvals[finite])
                for i, qv in enumerate(qvals.tolist()):
                    pair_rows[i]["q_value"] = (
                        float(qv) if np.isfinite(float(qv)) else np.nan
                    )

    marker_df = pd.DataFrame(marker_rows + pair_rows)

    # QC flag for exemplar condition.
    overall_mt = float(np.median(qc_pct_mt)) if qc_pct_mt is not None else np.nan
    sector_mt = []
    for sid in range(n_sectors):
        mm = sectors == sid
        if qc_pct_mt is not None and int(mm.sum()) > 0:
            sector_mt.append(float(np.median(qc_pct_mt[mm])))
    extreme_sector_qc = False
    if np.isfinite(overall_mt) and overall_mt > 0 and len(sector_mt) > 0:
        extreme_sector_qc = bool(np.max(sector_mt) > 2.0 * overall_mt)

    qc_flag = bool(
        (float(exemplar_row.get("q_T_within", np.nan)) <= Q_SIG)
        and (
            (abs(float(exemplar_row.get("rho_total_counts", np.nan))) >= QC_RISK_THRESH)
            or (abs(float(exemplar_row.get("rho_pct_mt", np.nan))) >= QC_RISK_THRESH)
            or extreme_sector_qc
        )
    )

    meta = {
        "gene": gene,
        "embedding": emb,
        "foreground_mode": mode,
        "n_sectors": int(n_sectors),
        "sector_sizes": json.dumps([int(x) for x in sector_sizes.tolist()]),
        "peak_angles_deg": json.dumps([float(x) for x in peak_angles.tolist()]),
        "qc_flag": bool(qc_flag),
        "extreme_sector_qc": bool(extreme_sector_qc),
        "overall_pct_mt_median": overall_mt,
    }
    meta_df = pd.DataFrame([meta])

    return per_cell, marker_df, {"meta_df": meta_df, "qc_flag": qc_flag}


def _plot_overview(
    out_dir: Path,
    umap_coords: np.ndarray,
    expr_by_gene: dict[str, np.ndarray],
    qc_total_counts: np.ndarray | None,
    qc_pct_mt: np.ndarray | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for gene in ["TNNT2", "MYH7"]:
        expr = expr_by_gene.get(gene, None)
        if expr is None:
            _save_placeholder(
                out_dir / f"umap_{gene}.png", f"UMAP {gene}", f"{gene} missing"
            )
            continue
        save_numeric_umap(
            umap_coords,
            np.log1p(np.maximum(np.asarray(expr, dtype=float), 0.0)),
            out_dir / f"umap_{gene}.png",
            title=f"umap_repr colored by {gene}",
            cmap="Reds",
            colorbar_label=f"log1p({gene})",
        )

    if qc_total_counts is not None:
        save_numeric_umap(
            umap_coords,
            np.log1p(np.maximum(qc_total_counts, 0.0)),
            out_dir / "umap_total_counts.png",
            title="umap_repr log1p(total_counts)",
            cmap="viridis",
            colorbar_label="log1p(total_counts)",
        )
    else:
        _save_placeholder(
            out_dir / "umap_total_counts.png", "total_counts", "Unavailable"
        )

    if qc_pct_mt is not None:
        save_numeric_umap(
            umap_coords,
            qc_pct_mt,
            out_dir / "umap_pct_counts_mt.png",
            title="umap_repr pct_counts_mt",
            cmap="magma",
            colorbar_label="pct_counts_mt",
        )
    else:
        _save_placeholder(
            out_dir / "umap_pct_counts_mt.png", "pct_counts_mt", "Unavailable"
        )


def _plot_multimodality_score_space(out_dir: Path, long_df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if long_df.empty:
        _save_placeholder(
            out_dir / "empty.png", "Multimodality score space", "No tests"
        )
        return

    for emb, sub in long_df.groupby("embedding", sort=False):
        fig1, ax1 = plt.subplots(figsize=(8.8, 6.6))
        shapes = {"F0_detection": "o", "F1_top_quantile": "^"}
        cmap = plt.get_cmap("viridis")

        peaks = sub["peaks_K"].to_numpy(dtype=float)
        pmax = np.nanmax(peaks) if np.isfinite(peaks).any() else 1.0
        pmax = max(1.0, float(pmax))

        for mode, ssub in sub.groupby("foreground_mode", sort=False):
            cols = [
                (
                    cmap(min(1.0, float(k) / pmax))
                    if np.isfinite(float(k))
                    else (0.8, 0.8, 0.8, 1.0)
                )
                for k in ssub["peaks_K"].to_numpy(dtype=float)
            ]
            ax1.scatter(
                ssub["Z_T"].to_numpy(dtype=float),
                ssub["coverage_C"].to_numpy(dtype=float),
                s=85,
                marker=shapes.get(str(mode), "o"),
                c=cols,
                edgecolors="black",
                linewidths=0.4,
                alpha=0.9,
                label=str(mode),
            )

        mm = sub.loc[(sub["q_T_within"] <= Q_SIG) & (sub["peaks_K"] >= 2)]
        for _, r in mm.iterrows():
            ax1.text(
                float(r["Z_T"]),
                float(r["coverage_C"]) + 0.006,
                str(r["gene"]),
                fontsize=8,
            )

        ax1.set_xlabel("Z_T")
        ax1.set_ylabel("coverage_C")
        ax1.set_title(f"{emb}: Z_T vs coverage_C (color ~ peaks_K, shape ~ mode)")
        ax1.legend(loc="best", fontsize=8, frameon=True)
        fig1.tight_layout()
        fig1.savefig(out_dir / f"{emb}_score_space.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig1)

        # peaks_K bar per gene for two modes.
        pivot = sub.pivot(index="gene", columns="foreground_mode", values="peaks_K")
        genes = pivot.index.astype(str).tolist()
        x = np.arange(len(genes))
        width = 0.38

        fig2, ax2 = plt.subplots(figsize=(max(8.0, 0.72 * len(genes)), 5.6))
        f0 = (
            pivot["F0_detection"].to_numpy(dtype=float)
            if "F0_detection" in pivot.columns
            else np.zeros(len(genes))
        )
        f1 = (
            pivot["F1_top_quantile"].to_numpy(dtype=float)
            if "F1_top_quantile" in pivot.columns
            else np.zeros(len(genes))
        )
        ax2.bar(x - width / 2, f0, width=width, color="#1f77b4", label="F0_detection")
        ax2.bar(
            x + width / 2, f1, width=width, color="#ff7f0e", label="F1_top_quantile"
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(genes, rotation=45, ha="right")
        ax2.set_ylabel("peaks_K")
        ax2.set_title(f"{emb}: peaks_K by gene (F0 vs F1)")
        ax2.legend(loc="best", fontsize=8)
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{emb}_peaksK_bars.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig2)


def _choose_best_condition_for_polar(
    long_df: pd.DataFrame, gene: str
) -> tuple[str, str] | None:
    pref = [
        ("umap_repr", "F0_detection"),
        ("umap_repr", "F1_top_quantile"),
        ("pca2d", "F0_detection"),
        ("pca2d", "F1_top_quantile"),
    ]
    sub = long_df.loc[long_df["gene"] == gene]
    if sub.empty:
        return None
    for emb, mode in pref:
        hit = sub.loc[(sub["embedding"] == emb) & (sub["foreground_mode"] == mode)]
        if not hit.empty:
            return emb, mode
    return None


def _plot_polar_profiles_with_peaks(
    out_dir: Path,
    long_df: pd.DataFrame,
    cache: dict[tuple[str, str, str], dict[str, Any]],
    n_bins: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    genes = (
        sorted(long_df["gene"].astype(str).unique().tolist())
        if not long_df.empty
        else []
    )
    if len(genes) == 0:
        _save_placeholder(out_dir / "empty.png", "Polar profiles", "No genes scored")
        return

    for gene in genes:
        fig = plt.figure(figsize=(14.5, 8.0))
        gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.33)

        conditions = [("umap_repr", "F0_detection"), ("umap_repr", "F1_top_quantile")]
        for i, (emb, mode) in enumerate(conditions):
            row = long_df.loc[
                (long_df["gene"] == gene)
                & (long_df["embedding"] == emb)
                & (long_df["foreground_mode"] == mode)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            key = (gene, emb, mode)
            ax_pol = fig.add_subplot(gs[i, 0], projection="polar")
            ax_hist = fig.add_subplot(gs[i, 1])

            if key not in cache:
                ax_pol.text(
                    0.5,
                    0.5,
                    f"{emb}/{mode}\nunderpowered/no null",
                    transform=ax_pol.transAxes,
                    ha="center",
                    va="center",
                )
                ax_pol.set_xticks([])
                ax_pol.set_yticks([])
                ax_hist.axis("off")
                continue

            c = cache[key]
            e_obs = np.asarray(c["E_phi_obs"], dtype=float)
            null_e = np.asarray(c["null_E_phi"], dtype=float)
            null_t = np.asarray(c["null_T"], dtype=float)
            peak_bins = np.asarray(c["peak_bins"], dtype=int)

            centers = theta_bin_centers(int(n_bins))
            th = np.concatenate([centers, centers[:1]])
            obs = np.concatenate([e_obs, e_obs[:1]])
            hi = np.quantile(null_e, 0.95, axis=0)
            lo = np.quantile(null_e, 0.05, axis=0)
            hi_c = np.concatenate([hi, hi[:1]])
            lo_c = np.concatenate([lo, lo[:1]])

            ax_pol.plot(th, obs, color="#8B0000", linewidth=2.0)
            ax_pol.plot(th, hi_c, color="#333333", linestyle="--", linewidth=1.2)
            ax_pol.plot(th, lo_c, color="#333333", linestyle="--", linewidth=0.9)
            ax_pol.fill_between(th, lo_c, hi_c, color="#b5b5b5", alpha=0.2)

            # mark peaks on |E| arg bins.
            for p in peak_bins.tolist():
                ang = float(centers[int(p)])
                val = float(e_obs[int(p)])
                ax_pol.scatter(
                    [ang],
                    [val],
                    c="#1f77b4",
                    s=44,
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=5,
                )
                ax_pol.text(ang, val, f"{int(np.degrees(ang)%360):d}°", fontsize=7)

            ax_pol.set_theta_zero_location("E")
            ax_pol.set_theta_direction(1)
            ax_pol.set_title(
                f"{emb} / {mode}\nZ={float(r['Z_T']):.2f}, q={float(r['q_T_within']):.2e}, "
                f"C={float(r['coverage_C']):.3f}, K={int(r['peaks_K']) if np.isfinite(r['peaks_K']) else -1}",
                fontsize=9,
            )

            bins = int(min(40, max(10, np.ceil(np.sqrt(max(10, null_t.size))))))
            ax_hist.hist(
                null_t, bins=bins, color="#7aa6d6", edgecolor="white", alpha=0.95
            )
            ax_hist.axvline(
                float(r["T_obs"]), color="#8B0000", linestyle="--", linewidth=2.0
            )
            ax_hist.set_title(f"null_T ({emb}/{mode})")
            ax_hist.set_xlabel("T under null")
            ax_hist.set_ylabel("count")

        fig.suptitle(
            f"{gene}: polar profiles with null envelope and detected peaks", y=0.995
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
        fig.savefig(out_dir / f"polar_{gene}.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig)


def _plot_sector_maps(
    out_dir: Path,
    exemplars_df: pd.DataFrame,
    embeddings_map: dict[str, EmbeddingSpec],
    per_cell_assignments: dict[str, pd.DataFrame],
    cache: dict[tuple[str, str, str], dict[str, Any]],
    n_bins: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if exemplars_df.empty:
        _save_placeholder(
            out_dir / "no_exemplars.png", "Sector maps", "No multimodal exemplars"
        )
        return

    for _, ex in exemplars_df.iterrows():
        gene = str(ex["gene"])
        emb = str(ex["embedding"])
        mode = str(ex["foreground_mode"])
        key = (gene, emb, mode)

        if (
            emb not in embeddings_map
            or key not in cache
            or gene not in per_cell_assignments
        ):
            _save_placeholder(
                out_dir / f"sectors_{gene}.png",
                f"Sectors: {gene}",
                "Insufficient cached data",
            )
            continue

        coords = embeddings_map[emb].coords
        cell_df = per_cell_assignments[gene]
        sectors = cell_df["sector_id"].to_numpy(dtype=int)
        fg = cell_df["is_foreground"].to_numpy(dtype=bool)

        fig = plt.figure(figsize=(16.5, 5.3))
        gs = fig.add_gridspec(1, 3, wspace=0.24)

        # 1) sector assignment map.
        ax1 = fig.add_subplot(gs[0, 0])
        cmap = plt.get_cmap("tab20")
        cols = [cmap(int(s) % 20) for s in sectors.tolist()]
        ax1.scatter(
            coords[:, 0],
            coords[:, 1],
            c=cols,
            s=8,
            alpha=0.92,
            linewidths=0,
            rasterized=True,
        )
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f"{gene}: sectors ({emb}/{mode})")

        # 2) foreground overlay on sectors.
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(
            coords[:, 0],
            coords[:, 1],
            c=cols,
            s=7,
            alpha=0.25,
            linewidths=0,
            rasterized=True,
        )
        idx_fg = np.flatnonzero(fg)
        if idx_fg.size > 0:
            ax2.scatter(
                coords[idx_fg, 0],
                coords[idx_fg, 1],
                c=np.array(cols, dtype=object)[idx_fg],
                s=12,
                alpha=0.95,
                linewidths=0.2,
                edgecolors="black",
                rasterized=True,
            )
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("foreground highlighted")

        # 3) angle histogram with sector boundaries and peaks.
        ax3 = fig.add_subplot(gs[0, 2], projection="polar")
        theta = np.asarray(cache[key]["theta"], dtype=float)
        peaks = np.asarray(cache[key]["peak_bins"], dtype=int)
        centers = theta_bin_centers(int(n_bins))
        pk_ang = centers[peaks] if peaks.size > 0 else np.zeros(0, dtype=float)

        fg_theta = theta[fg]
        if fg_theta.size > 0:
            bins = np.linspace(0.0, 2.0 * np.pi, int(n_bins) + 1)
            hist, _ = np.histogram(fg_theta, bins=bins)
            th = centers
            hist = hist.astype(float)
            if np.max(hist) > 0:
                hist = hist / np.max(hist)
            ax3.plot(
                np.concatenate([th, th[:1]]),
                np.concatenate([hist, hist[:1]]),
                color="#8B0000",
                linewidth=2.0,
            )
            ax3.fill_between(
                np.concatenate([th, th[:1]]),
                0.0,
                np.concatenate([hist, hist[:1]]),
                color="#f08080",
                alpha=0.25,
            )

        # derive boundaries from sorted peak angles.
        if pk_ang.size >= 2:
            ord_idx = np.argsort(pk_ang)
            pa = pk_ang[ord_idx]
            bounds = []
            for i in range(pa.size):
                a = pa[i]
                b = pa[(i + 1) % pa.size]
                if i == pa.size - 1:
                    b = b + 2.0 * np.pi
                mid = 0.5 * (a + b)
                bounds.append(float(mid % (2.0 * np.pi)))
            for bd in bounds:
                ax3.plot(
                    [bd, bd],
                    [0.0, 1.05],
                    color="#333333",
                    linestyle="--",
                    linewidth=1.0,
                )

        for a in pk_ang.tolist():
            ax3.plot([a, a], [0.0, 1.15], color="#1f77b4", linewidth=2.0)
            ax3.text(
                a,
                1.18,
                f"{int(np.degrees(a)%360)}°",
                fontsize=8,
                ha="center",
                va="center",
            )

        ax3.set_theta_zero_location("E")
        ax3.set_theta_direction(1)
        ax3.set_rticks([])
        ax3.set_title("Foreground angle distribution")

        fig.suptitle(f"Sectorization map for {gene}", y=0.995)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
        fig.savefig(out_dir / f"sectors_{gene}.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig)


def _bootstrap_ci_mean(
    x: np.ndarray, rng: np.random.Generator, n_boot: int = 200
) -> tuple[float, float, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mu = float(np.mean(arr))
    if arr.size == 1:
        return mu, mu, mu
    boots = np.zeros(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        boots[i] = float(np.mean(arr[idx]))
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return mu, lo, hi


def _plot_sector_marker_validation(
    out_dir: Path,
    exemplars_df: pd.DataFrame,
    marker_stats_df: pd.DataFrame,
    per_cell_assignments: dict[str, pd.DataFrame],
    expr_by_gene: dict[str, np.ndarray],
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if exemplars_df.empty:
        _save_placeholder(
            out_dir / "no_exemplars.png",
            "Sector marker validation",
            "No multimodal exemplars",
        )
        return

    for ex_i, ex in enumerate(exemplars_df.itertuples(index=False), start=1):
        gene = str(ex.gene)
        sub = marker_stats_df.loc[
            (marker_stats_df["gene"] == gene)
            & (marker_stats_df["embedding"] == str(ex.embedding))
            & (marker_stats_df["foreground_mode"] == str(ex.foreground_mode))
        ].copy()
        if sub.empty:
            _save_placeholder(
                out_dir / f"sector_markers_{gene}.png",
                f"Sector markers: {gene}",
                "No marker stats available",
            )
            continue

        sec_sum = sub.loc[sub["analysis_type"] == "sector_summary"].copy()
        pair = sub.loc[sub["analysis_type"] == "pairwise_wilcoxon"].copy()

        fig = plt.figure(figsize=(17.0, 5.7))
        gs = fig.add_gridspec(1, 3, wspace=0.30)

        # 1) Heatmap sectors x markers.
        ax1 = fig.add_subplot(gs[0, 0])
        pivot = sec_sum.pivot(index="comparison", columns="marker", values="mean_log1p")
        if pivot.empty:
            ax1.axis("off")
            ax1.set_title("No sector summary")
        else:
            mat = pivot.to_numpy(dtype=float)
            im = ax1.imshow(mat, aspect="auto", cmap="viridis")
            ax1.set_xticks(np.arange(len(pivot.columns)))
            ax1.set_xticklabels(
                pivot.columns.astype(str), rotation=45, ha="right", fontsize=8
            )
            ax1.set_yticks(np.arange(len(pivot.index)))
            ax1.set_yticklabels(pivot.index.astype(str), fontsize=8)
            ax1.set_title("Mean(log1p expr): sector x marker")
            cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            cb.set_label("mean log1p")

        # 2) Selected marker bars with bootstrap CI.
        ax2 = fig.add_subplot(gs[0, 1])
        selected = [
            m for m in ["NPPA", "NPPB", "MYH7", "PPARGC1A"] if m in expr_by_gene
        ]
        cell_df = per_cell_assignments.get(gene, pd.DataFrame())
        if len(selected) == 0 or cell_df.empty:
            ax2.axis("off")
            ax2.set_title("Selected markers unavailable")
        else:
            sectors = cell_df["sector_id"].to_numpy(dtype=int)
            uniq_sec = np.unique(sectors)
            rng = np.random.default_rng(int(seed + ex_i * 101))
            bar_x = []
            bar_mu = []
            bar_lo = []
            bar_hi = []
            labels = []
            for sid in uniq_sec.tolist():
                msk = sectors == sid
                for m in selected:
                    vals = np.log1p(
                        np.maximum(np.asarray(expr_by_gene[m], dtype=float), 0.0)
                    )[msk]
                    mu, lo, hi = _bootstrap_ci_mean(vals, rng=rng, n_boot=200)
                    labels.append(f"s{sid}:{m}")
                    bar_x.append(len(bar_x))
                    bar_mu.append(mu)
                    bar_lo.append(mu - lo if np.isfinite(lo) else 0.0)
                    bar_hi.append(hi - mu if np.isfinite(hi) else 0.0)
            ax2.bar(bar_x, bar_mu, color="#4c78a8", edgecolor="white")
            ax2.errorbar(
                bar_x,
                bar_mu,
                yerr=[bar_lo, bar_hi],
                fmt="none",
                ecolor="black",
                capsize=2,
                linewidth=0.8,
            )
            ax2.set_xticks(bar_x)
            ax2.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
            ax2.set_ylabel("mean log1p")
            ax2.set_title("Selected markers per sector (bootstrap 95% CI)")

        # 3) Volcano-like pairwise marker differences.
        ax3 = fig.add_subplot(gs[0, 2])
        if pair.empty:
            ax3.axis("off")
            ax3.set_title("No pairwise Wilcoxon")
        else:
            delta = pair["delta_median"].to_numpy(dtype=float)
            qv = pair["q_value"].to_numpy(dtype=float)
            y = -np.log10(np.clip(qv, 1e-300, None))
            ax3.scatter(
                delta,
                y,
                s=70,
                c="#d62728",
                alpha=0.85,
                edgecolors="black",
                linewidths=0.3,
            )
            for _, r in pair.iterrows():
                ax3.text(
                    float(r["delta_median"]),
                    float(-np.log10(max(float(r["q_value"]), 1e-300))),
                    str(r["marker"]),
                    fontsize=8,
                )
            ax3.axvline(0.0, color="#666666", linewidth=0.9)
            ax3.axhline(-np.log10(0.05), color="#444444", linestyle="--", linewidth=1.0)
            ax3.set_xlabel("delta median (sectorA - sectorB)")
            ax3.set_ylabel("-log10(q)")
            ax3.set_title(str(pair["comparison"].iloc[0]))

        fig.suptitle(f"Sector marker validation: {gene}", y=0.995)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
        fig.savefig(out_dir / f"sector_markers_{gene}.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig)


def _plot_qc_controls(
    out_dir: Path,
    long_df: pd.DataFrame,
    exemplars_df: pd.DataFrame,
    marker_stats_df: pd.DataFrame,
    per_cell_assignments: dict[str, pd.DataFrame],
    qc_total_counts: np.ndarray | None,
    qc_pct_mt: np.ndarray | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if long_df.empty:
        _save_placeholder(out_dir / "empty.png", "QC controls", "No tests")
        return

    # 1) qc_risk vs Z_T all tests.
    fig1, ax1 = plt.subplots(figsize=(8.4, 6.1))
    mm_genes = (
        set(exemplars_df["gene"].astype(str).tolist())
        if not exemplars_df.empty
        else set()
    )

    for _, r in long_df.iterrows():
        g = str(r["gene"])
        is_mm = g in mm_genes
        ax1.scatter(
            float(r["qc_risk"]),
            float(r["Z_T"]) if np.isfinite(float(r["Z_T"])) else np.nan,
            s=90 if is_mm else 50,
            marker="*" if is_mm else "o",
            c="#d62728" if is_mm else "#4c78a8",
            alpha=0.78,
            edgecolors="black",
            linewidths=0.3,
        )
    ax1.axvline(QC_RISK_THRESH, color="#444444", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("qc_risk")
    ax1.set_ylabel("Z_T")
    ax1.set_title("qc_risk vs Z_T (stars = multimodal exemplars)")
    fig1.tight_layout()
    fig1.savefig(out_dir / "qc_risk_vs_ZT_all_tests.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    # 2) per-exemplar sector QC boxplots.
    if exemplars_df.empty:
        _save_placeholder(
            out_dir / "per_exemplar_sector_qc.png",
            "Sector QC",
            "No multimodal exemplars",
        )
    else:
        n = len(exemplars_df)
        fig2, axes = plt.subplots(n, 2, figsize=(11.5, 3.8 * n), squeeze=False)
        for i, ex in enumerate(exemplars_df.itertuples(index=False)):
            gene = str(ex.gene)
            cell_df = per_cell_assignments.get(gene, None)
            if cell_df is None or cell_df.empty:
                axes[i, 0].axis("off")
                axes[i, 1].axis("off")
                continue
            sectors = cell_df["sector_id"].to_numpy(dtype=int)
            uniq = np.unique(sectors)
            data_tc = []
            data_mt = []
            labels = []
            for sid in uniq.tolist():
                m = sectors == sid
                labels.append(f"s{sid}")
                data_tc.append(
                    qc_total_counts[m] if qc_total_counts is not None else np.array([])
                )
                data_mt.append(qc_pct_mt[m] if qc_pct_mt is not None else np.array([]))
            axes[i, 0].boxplot(data_tc, labels=labels, patch_artist=True)
            axes[i, 0].set_title(f"{gene}: total_counts by sector")
            axes[i, 0].tick_params(axis="x", rotation=0)
            axes[i, 1].boxplot(data_mt, labels=labels, patch_artist=True)
            axes[i, 1].set_title(f"{gene}: pct_mt by sector")
            axes[i, 1].tick_params(axis="x", rotation=0)
        fig2.tight_layout()
        fig2.savefig(
            out_dir / "per_exemplar_sector_qc_boxplots.png", dpi=DEFAULT_PLOT_STYLE.dpi
        )
        plt.close(fig2)

    # 3) optional polar overlay gene vs pct_mt profile for exemplars.
    # Here we provide one merged diagnostic figure using cached test rows.
    if exemplars_df.empty or qc_pct_mt is None:
        _save_placeholder(
            out_dir / "polar_overlay_gene_vs_pctmt.png",
            "Polar overlay gene vs pct_mt",
            "Skipped: no exemplars or pct_mt unavailable.",
        )
        return

    fig3 = plt.figure(figsize=(5.2 * len(exemplars_df), 5.0))
    for j, ex in enumerate(exemplars_df.itertuples(index=False), start=1):
        ax = fig3.add_subplot(1, len(exemplars_df), j, projection="polar")
        gene = str(ex.gene)
        emb = str(ex.embedding)
        mode = str(ex.foreground_mode)

        # We'll reconstruct from long_df condition row and cache-based E profile not available here;
        # for simplicity draw gene profile from best row already summarized in long_df via peak info only.
        sub = long_df.loc[
            (long_df["gene"] == gene)
            & (long_df["embedding"] == emb)
            & (long_df["foreground_mode"] == mode)
        ]
        if sub.empty:
            ax.axis("off")
            continue
        # Recompute quick profile for pct_mt using same embedding/mode's theta binning from per-cell assignment if available.
        cell_df = per_cell_assignments.get(gene, None)
        if cell_df is None or cell_df.empty:
            ax.axis("off")
            continue
        theta = np.deg2rad(cell_df["theta_deg"].to_numpy(dtype=float))
        _, b = bin_theta(theta, bins=64)
        counts = np.bincount(b, minlength=64).astype(float)

        # gene foreground profile
        fg = cell_df["is_foreground"].to_numpy(dtype=bool)
        if int(fg.sum()) == 0 or int((~fg).sum()) == 0:
            ax.axis("off")
            continue
        fg_counts = np.bincount(b[fg], minlength=64).astype(float)
        bg_counts = counts - fg_counts
        e_gene = fg_counts / float(fg.sum()) - bg_counts / float((~fg).sum())

        # pct_mt pseudo profile
        mt = np.asarray(qc_pct_mt, dtype=float)
        mt = np.maximum(mt, 0.0)
        mt_sum = float(np.sum(mt))
        if mt_sum <= 0:
            ax.axis("off")
            continue
        mt_bin = np.bincount(b, weights=mt, minlength=64).astype(float)
        p_mt = mt_bin / mt_sum
        p_bg = counts / float(counts.sum())
        e_mt = p_mt - p_bg

        centers = theta_bin_centers(64)
        th = np.concatenate([centers, centers[:1]])
        ax.plot(
            th, np.concatenate([e_gene, e_gene[:1]]), color="#8B0000", label=f"{gene}"
        )
        ax.plot(
            th,
            np.concatenate([e_mt, e_mt[:1]]),
            color="#1f77b4",
            linestyle="--",
            label="pct_mt",
        )
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_title(gene)
        ax.legend(loc="upper right", fontsize=7)

    fig3.suptitle("Polar overlay: exemplar gene vs pct_mt pseudo-profile", y=0.995)
    fig3.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig3.savefig(
        out_dir / "polar_overlay_gene_vs_pctmt.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig3)


def _write_readme(
    out_path: Path,
    *,
    seed: int,
    n_perm: int,
    n_bins: int,
    q: float,
    peak_width_bins: int,
    donor_key: str,
    label_key: str,
    donor_star: str,
    cm_labels: list[str],
    cm_counts: dict[str, int],
    expr_source: str,
    embed_note: str,
    n_cells_donor: int,
    n_cells_cm: int,
    cm_underpowered: bool,
    qc_sources: dict[str, str],
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    exemplars_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append(
        "CM Experiment #3 (Single-donor): Multimodality test — patchy islands vs single gradient"
    )
    lines.append("")
    lines.append("Hypothesis")
    lines.append(
        "Some CM genes show multimodal BioRSP enrichment (multiple angular peaks; patchy islands) rather than a single axis. "
        "These should be null-significant, stable across foreground definitions, linked to coherent sector-level marker differences, "
        "and not primarily QC-driven."
    )
    lines.append("")
    lines.append("Interpretation guardrail")
    lines.append(
        "Directions/angles are representation-conditional (embedding geometry), not anatomical direction."
    )
    lines.append("")
    lines.append("Run metadata")
    lines.append(f"- seed: {seed}")
    lines.append(f"- n_perm: {n_perm}")
    lines.append(f"- n_bins: {n_bins}")
    lines.append(f"- q_top_quantile: {q}")
    lines.append(f"- peak_width_bins: {peak_width_bins}")
    lines.append(f"- donor_key_used: {donor_key}")
    lines.append(f"- label_key_used: {label_key}")
    lines.append(f"- donor_star: {donor_star}")
    lines.append(f"- expression_source_used: {expr_source}")
    lines.append(f"- embedding_input_prep: {embed_note}")
    lines.append(f"- donor_star_total_cells: {n_cells_donor}")
    lines.append(f"- donor_star_cardiomyocytes: {n_cells_cm}")
    lines.append(f"- cm_underpowered: {cm_underpowered}")
    lines.append("")
    lines.append("Cardiomyocyte labels included")
    for label in cm_labels:
        lines.append(f"- {label}: {cm_counts.get(label, 0)}")
    lines.append("")
    lines.append("QC covariate sources")
    for k, v in qc_sources.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Reference notes")
    lines.append(
        "- t-SNE perplexity guidance (5-50; relatively insensitive) motivated keeping embedding robustness mild."
    )
    lines.append(
        "- CM maturation/metabolic program literature motivated including PPARGC1A/CPT1B/COX4I1 validation signals."
    )
    lines.append("")

    n_tests = int(len(long_df))
    n_sig = (
        int(
            np.sum(
                (long_df["q_T_within"].to_numpy(dtype=float) <= Q_SIG)
                & np.isfinite(long_df["q_T_within"].to_numpy(dtype=float))
            )
        )
        if not long_df.empty
        else 0
    )
    n_mm = (
        int(
            np.sum(
                (long_df["q_T_within"].to_numpy(dtype=float) <= Q_SIG)
                & (long_df["peaks_K"].to_numpy(dtype=float) >= 2.0)
            )
        )
        if not long_df.empty
        else 0
    )
    lines.append("Summary counts")
    lines.append(f"- n_tests: {n_tests}")
    lines.append(f"- n_significant_tests(q<=0.05): {n_sig}")
    lines.append(f"- n_multimodal_significant_tests(q<=0.05 & peaks>=2): {n_mm}")
    lines.append(f"- n_multimodal_exemplar_genes: {int(len(exemplars_df))}")
    if not exemplars_df.empty:
        lines.append(
            "- exemplar_genes: " + ", ".join(exemplars_df["gene"].astype(str).tolist())
        )
    lines.append("")

    if not summary_df.empty:
        robust = summary_df.loc[summary_df["robust_localized"]]
        lines.append(f"- robust_localized_gene_embedding_pairs: {int(len(robust))}")
    lines.append("")

    lines.append("Single-donor rigor note")
    lines.append(
        "No donor replication exists here. Evidence is based on null-calibrated multimodality statistics, "
        "foreground robustness (F0 vs F1), internal sector-level validation, and QC control audits."
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
        plots_dir / "01_multimodality_score_space",
        plots_dir / "02_polar_profiles_with_peaks",
        plots_dir / "03_sector_maps",
        plots_dir / "04_sector_marker_validation",
        plots_dir / "05_qc_controls",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad)

    donor_key = _resolve_key_required(
        adata, args.donor_key, DONOR_KEY_CANDIDATES, purpose="donor"
    )
    label_key = _resolve_key_required(
        adata, args.label_key, LABEL_KEY_CANDIDATES, purpose="label"
    )

    labels_all = adata.obs[label_key].astype("string").fillna("NA").astype(str)
    cm_mask_all = labels_all.map(_is_cm_label).to_numpy(dtype=bool)
    if int(cm_mask_all.sum()) == 0:
        raise RuntimeError("No cardiomyocyte cells detected by substring rule.")

    donor_ids_all = adata.obs[donor_key].astype("string").fillna("NA").astype(str)
    donor_choice = (
        pd.DataFrame({"donor_id": donor_ids_all.to_numpy(), "is_cm": cm_mask_all})
        .groupby("donor_id", as_index=False)
        .agg(n_cells_total=("is_cm", "size"), n_cm=("is_cm", "sum"))
        .sort_values(
            by=["n_cm", "n_cells_total", "donor_id"], ascending=[False, False, True]
        )
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
        raise RuntimeError("donor_star cardiomyocyte subset is empty")

    cm_underpowered = bool(int(adata_cm.n_obs) < 2000)
    if cm_underpowered:
        print(
            f"WARNING: adata_cm.n_obs={int(adata_cm.n_obs)} < 2000; cm_underpowered=True"
        )

    expr_matrix_cm, adata_like_cm, expr_source = _choose_expression_source(
        adata_cm, layer_arg=args.layer, use_raw_arg=bool(args.use_raw)
    )

    # QC covariates.
    qc_total_counts, key_total = _safe_numeric_obs(
        adata_cm, ["total_counts", "n_counts", "n_genes_by_counts"]
    )
    if qc_total_counts is None:
        qc_total_counts = _total_counts_vector(adata_cm, expr_matrix_cm)
        key_total = "computed:expr_sum"

    qc_pct_mt, key_mt = _safe_numeric_obs(
        adata_cm, ["pct_counts_mt", "percent.mt", "pct_mt"]
    )
    if qc_pct_mt is None:
        qc_pct_mt, key_mt2 = _pct_mt_vector(adata_cm, expr_matrix_cm, adata_like_cm)
        key_mt = key_mt2

    qc_pct_ribo, key_ribo = _compute_pct_counts_ribo(
        adata_cm,
        expr_matrix_cm,
        adata_like_cm,
        np.asarray(qc_total_counts, dtype=float),
    )

    qc_sources = {
        "total_counts": str(key_total),
        "pct_counts_mt": str(key_mt),
        "pct_counts_ribo": str(key_ribo),
    }

    # Embeddings.
    adata_embed, embed_note = _prepare_embedding_input(
        adata_cm, expr_matrix_cm, expr_source
    )
    embeddings, n_pcs_used = _compute_fixed_embeddings(
        adata_embed, seed=int(args.seed), k_pca=int(args.k_pca)
    )
    embeddings_map = {e.key: e for e in embeddings}

    # Gene sets.
    candidate_status, validation_status, gene_panel_df = _resolve_gene_sets(
        adata_like_cm
    )
    gene_panel_df.to_csv(tables_dir / "gene_panel_status.csv", index=False)

    # Main scoring.
    long_df, cache, expr_by_gene = _score_tests(
        donor_star=donor_star,
        candidate_status=candidate_status,
        expr_matrix_cm=expr_matrix_cm,
        embeddings=embeddings,
        n_bins=int(args.n_bins),
        n_perm=int(args.n_perm),
        q_top=float(args.q),
        peak_width_bins=int(args.peak_width_bins),
        seed=int(args.seed),
        qc_total_counts=(
            np.asarray(qc_total_counts, dtype=float)
            if qc_total_counts is not None
            else None
        ),
        qc_pct_mt=np.asarray(qc_pct_mt, dtype=float) if qc_pct_mt is not None else None,
        qc_pct_ribo=(
            np.asarray(qc_pct_ribo, dtype=float) if qc_pct_ribo is not None else None
        ),
        out_tables_dir=tables_dir,
    )
    long_df.to_csv(tables_dir / "per_test_scores_long.csv", index=False)

    # Summary per gene x embedding.
    summary_df = _summarize_multimodality(long_df)
    summary_df.to_csv(tables_dir / "per_gene_multimodality_summary.csv", index=False)

    # Select multimodal exemplars.
    exemplars: list[pd.Series] = []
    for gene in (
        sorted(long_df["gene"].astype(str).unique().tolist())
        if not long_df.empty
        else []
    ):
        ex = _select_primary_condition(long_df, gene)
        if ex is not None:
            exemplars.append(ex)

    if len(exemplars) > 0:
        exemplars_df = pd.DataFrame(exemplars)
    else:
        exemplars_df = pd.DataFrame(
            columns=(
                long_df.columns
                if not long_df.empty
                else ["gene", "embedding", "foreground_mode"]
            )
        )

    # Sector assignments + marker validation.
    per_cell_tables: list[pd.DataFrame] = []
    marker_stats_tables: list[pd.DataFrame] = []
    qc_meta_rows: list[pd.DataFrame] = []
    per_cell_assignments_map: dict[str, pd.DataFrame] = {}

    validation_present = [
        s.gene
        for s in validation_status
        if s.present and s.gene_idx is not None and s.gene in expr_by_gene
    ]

    for ex_i, ex in enumerate(exemplars_df.itertuples(index=False), start=1):
        gene = str(ex.gene)
        per_cell_df, marker_df, meta = _compute_sector_validation(
            gene=gene,
            exemplar_row=pd.Series(ex._asdict()),
            cache=cache,
            expr_by_gene=expr_by_gene,
            validation_present=validation_present,
            qc_total_counts=(
                np.asarray(qc_total_counts, dtype=float)
                if qc_total_counts is not None
                else None
            ),
            qc_pct_mt=(
                np.asarray(qc_pct_mt, dtype=float) if qc_pct_mt is not None else None
            ),
            qc_pct_ribo=(
                np.asarray(qc_pct_ribo, dtype=float)
                if qc_pct_ribo is not None
                else None
            ),
            n_bins=int(args.n_bins),
            peak_width_bins=int(args.peak_width_bins),
            seed=int(args.seed + ex_i * 103),
        )
        if not per_cell_df.empty:
            per_cell_assignments_map[gene] = per_cell_df
            per_cell_tables.append(per_cell_df)
        if not marker_df.empty:
            marker_stats_tables.append(marker_df)
        if isinstance(meta, dict) and "meta_df" in meta:
            qc_meta_rows.append(meta["meta_df"])

    # Write exemplar table with qc_flag metadata merged when available.
    if not exemplars_df.empty and len(qc_meta_rows) > 0:
        meta_all = pd.concat(qc_meta_rows, axis=0, ignore_index=True)
        exemplars_df = exemplars_df.merge(
            meta_all[
                [
                    "gene",
                    "embedding",
                    "foreground_mode",
                    "qc_flag",
                    "extreme_sector_qc",
                    "overall_pct_mt_median",
                ]
            ],
            on=["gene", "embedding", "foreground_mode"],
            how="left",
        )
    else:
        exemplars_df["qc_flag"] = False
        exemplars_df["extreme_sector_qc"] = False
        exemplars_df["overall_pct_mt_median"] = np.nan

    exemplars_df.to_csv(tables_dir / "multimodal_exemplars.csv", index=False)

    if len(marker_stats_tables) > 0:
        marker_stats_df = pd.concat(marker_stats_tables, axis=0, ignore_index=True)
    else:
        marker_stats_df = pd.DataFrame(
            columns=[
                "gene",
                "embedding",
                "foreground_mode",
                "analysis_type",
                "comparison",
                "sector_id",
                "n_cells_sector",
                "marker",
                "mean_log1p",
                "median_log1p",
                "frac_expr_gt0",
                "delta_median",
                "p_value",
                "q_value",
                "total_counts_median",
                "pct_mt_median",
                "pct_ribo_median",
            ]
        )
    marker_stats_df.to_csv(tables_dir / "per_gene_sector_marker_stats.csv", index=False)

    if bool(args.save_per_cell):
        if len(per_cell_tables) > 0:
            per_cell_all = pd.concat(per_cell_tables, axis=0, ignore_index=True)
        else:
            per_cell_all = pd.DataFrame(
                columns=[
                    "gene",
                    "embedding",
                    "foreground_mode",
                    "cell_idx",
                    "theta_rad",
                    "theta_deg",
                    "sector_id",
                    "is_foreground",
                    "in_peak_region",
                ]
            )
        per_cell_all.to_csv(tables_dir / "per_gene_sector_assignments.csv", index=False)

    # QC audit table (all tests + exemplar sector QC flags).
    qc_audit = long_df[
        [
            "gene",
            "embedding",
            "foreground_mode",
            "q_T_within",
            "Z_T",
            "peaks_K",
            "rho_total_counts",
            "rho_pct_mt",
            "rho_pct_ribo",
            "qc_risk",
        ]
    ].copy()
    if not exemplars_df.empty:
        qc_audit = qc_audit.merge(
            exemplars_df[
                ["gene", "embedding", "foreground_mode", "qc_flag", "extreme_sector_qc"]
            ],
            on=["gene", "embedding", "foreground_mode"],
            how="left",
        )
    else:
        qc_audit["qc_flag"] = False
        qc_audit["extreme_sector_qc"] = False
    qc_audit["qc_flag"] = qc_audit["qc_flag"].fillna(False)
    qc_audit["extreme_sector_qc"] = qc_audit["extreme_sector_qc"].fillna(False)
    qc_audit.to_csv(tables_dir / "qc_audit.csv", index=False)

    # Plots.
    _plot_overview(
        out_dir=plots_dir / "00_overview",
        umap_coords=embeddings_map["umap_repr"].coords,
        expr_by_gene=expr_by_gene,
        qc_total_counts=(
            np.asarray(qc_total_counts, dtype=float)
            if qc_total_counts is not None
            else None
        ),
        qc_pct_mt=np.asarray(qc_pct_mt, dtype=float) if qc_pct_mt is not None else None,
    )
    _plot_multimodality_score_space(plots_dir / "01_multimodality_score_space", long_df)
    _plot_polar_profiles_with_peaks(
        plots_dir / "02_polar_profiles_with_peaks",
        long_df,
        cache,
        n_bins=int(args.n_bins),
    )
    _plot_sector_maps(
        plots_dir / "03_sector_maps",
        exemplars_df,
        embeddings_map,
        per_cell_assignments_map,
        cache,
        n_bins=int(args.n_bins),
    )
    _plot_sector_marker_validation(
        plots_dir / "04_sector_marker_validation",
        exemplars_df,
        marker_stats_df,
        per_cell_assignments_map,
        expr_by_gene,
        seed=int(args.seed),
    )
    _plot_qc_controls(
        plots_dir / "05_qc_controls",
        long_df,
        exemplars_df,
        marker_stats_df,
        per_cell_assignments_map,
        qc_total_counts=(
            np.asarray(qc_total_counts, dtype=float)
            if qc_total_counts is not None
            else None
        ),
        qc_pct_mt=np.asarray(qc_pct_mt, dtype=float) if qc_pct_mt is not None else None,
    )

    # README.
    cm_labels = sorted(labels_all.loc[cm_mask_all].unique().tolist())
    cm_counts = labels_all.loc[cm_mask_all].value_counts().to_dict()
    _write_readme(
        out_root / "README.txt",
        seed=int(args.seed),
        n_perm=int(args.n_perm),
        n_bins=int(args.n_bins),
        q=float(args.q),
        peak_width_bins=int(args.peak_width_bins),
        donor_key=donor_key,
        label_key=label_key,
        donor_star=donor_star,
        cm_labels=cm_labels,
        cm_counts={str(k): int(v) for k, v in cm_counts.items()},
        expr_source=expr_source,
        embed_note=f"{embed_note}; n_pcs_used={n_pcs_used}",
        n_cells_donor=int(adata_donor.n_obs),
        n_cells_cm=int(adata_cm.n_obs),
        cm_underpowered=cm_underpowered,
        qc_sources=qc_sources,
        long_df=long_df,
        summary_df=summary_df,
        exemplars_df=exemplars_df,
    )

    # Required output checks.
    required = [
        tables_dir / "donor_choice.csv",
        tables_dir / "gene_panel_status.csv",
        tables_dir / "per_test_scores_long.csv",
        tables_dir / "per_gene_multimodality_summary.csv",
        tables_dir / "multimodal_exemplars.csv",
        tables_dir / "per_gene_sector_marker_stats.csv",
        tables_dir / "qc_audit.csv",
        plots_dir / "00_overview",
        plots_dir / "01_multimodality_score_space",
        plots_dir / "02_polar_profiles_with_peaks",
        plots_dir / "03_sector_maps",
        plots_dir / "04_sector_marker_validation",
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
    print(f"n_tests={int(len(long_df))}")
    print(f"n_multimodal_exemplars={int(len(exemplars_df))}")
    print(f"cm_labels_included={json.dumps(cm_labels)}")
    print(f"results_root={out_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
