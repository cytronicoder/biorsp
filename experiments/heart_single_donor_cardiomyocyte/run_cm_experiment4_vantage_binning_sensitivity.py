#!/usr/bin/env python3
"""CM Experiment #4 (single donor): BioRSP-internal robustness to vantage + binning.

Hypothesis (pre-registered):
For genuine cardiomyocyte programs, BioRSP localization calls (significance, class,
peak direction) are robust to reasonable internal choices of vantage/origin and
angular bin resolution. If localization appears only under narrow internal settings,
it is treated as method-induced instability.

Interpretation guardrail:
Angles are representation-conditional (embedding geometry) and do not imply anatomy.
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

# Headless backend for deterministic script runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

# Support direct execution from repo root.
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

CM_PANEL: dict[str, list[str]] = {
    "Core CM": ["TNNT2", "MYH6", "MYH7", "RYR2", "ATP2A2"],
}

UNDERPOWERED_PREV = 0.005
UNDERPOWERED_MIN_FG = 50
Q_SIG = 0.05

CLASS_COLORS = {
    "Localized–unimodal": "#1f77b4",
    "Localized–multimodal": "#ff7f0e",
    "Not-localized": "#8a8a8a",
    "Underpowered": "#d62728",
}


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
        description="CM Experiment #4: BioRSP-internal robustness to vantage + angular binning."
    )
    p.add_argument(
        "--h5ad", default="data/processed/HT_pca_umap.h5ad", help="Input .h5ad"
    )
    p.add_argument(
        "--out",
        default="experiments/heart_single_donor_cardiomyocyte/results/cm_experiment4_vantage_binning_sensitivity",
        help="Output root",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--n_perm", type=int, default=300, help="Permutation count")
    p.add_argument(
        "--bins", type=int, nargs="+", default=[32, 64, 128], help="Angular bin sweep"
    )
    p.add_argument(
        "--foregrounds",
        nargs="+",
        default=["detect", "topq"],
        help="Foreground modes from {detect, topq}",
    )
    p.add_argument(
        "--q", type=float, default=0.10, help="Top-quantile for topq foreground"
    )
    p.add_argument(
        "--k_pca", type=int, default=50, help="PCA dimensionality for UMAP construction"
    )
    p.add_argument("--layer", default=None, help="Optional expression layer override")
    p.add_argument("--use_raw", action="store_true", help="Use adata.raw")
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
        raise KeyError(f"Requested {purpose} key '{requested}' not found in adata.obs")
    for c in candidates:
        if c in adata.obs.columns:
            return c
    raise KeyError(f"No {purpose} key found. Tried: {', '.join(candidates)}")


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


def _resolve_panel(adata_like: Any) -> tuple[list[GeneStatus], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    statuses: list[GeneStatus] = []
    used_idx = set()

    for group, genes in CM_PANEL.items():
        for gene in genes:
            try:
                idx, label, sym_col, source = resolve_feature_index(adata_like, gene)
                idx_i = int(idx)
                dup = idx_i in used_idx
                if not dup:
                    used_idx.add(idx_i)
                st = GeneStatus(
                    gene=gene,
                    marker_group=group,
                    present=not dup,
                    status="duplicate_index" if dup else "resolved",
                    resolved_gene="" if dup else str(label),
                    gene_idx=None if dup else idx_i,
                    resolution_source=source,
                    symbol_column=sym_col or "",
                )
            except KeyError:
                st = GeneStatus(
                    gene=gene,
                    marker_group=group,
                    present=False,
                    status="missing",
                    resolved_gene="",
                    gene_idx=None,
                    resolution_source="",
                    symbol_column="",
                )

            rows.append(
                {
                    "gene": st.gene,
                    "marker_group": st.marker_group,
                    "present": st.present,
                    "status": st.status,
                    "resolved_gene": st.resolved_gene,
                    "gene_idx": st.gene_idx if st.gene_idx is not None else "",
                    "resolution_source": st.resolution_source,
                    "symbol_column": st.symbol_column,
                }
            )
            statuses.append(st)

    return statuses, pd.DataFrame(rows)


def _safe_spearman(x: np.ndarray, y: np.ndarray | None) -> float:
    if y is None:
        return float("nan")
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(xv) & np.isfinite(yv)
    if int(m.sum()) < 3:
        return float("nan")
    xs = xv[m]
    ys = yv[m]
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return float("nan")
    rho = spearmanr(xs, ys, nan_policy="omit").correlation
    if rho is None or not np.isfinite(float(rho)):
        return float("nan")
    return float(rho)


def _classify(q: float, peaks_k: float, underpowered: bool) -> str:
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
        return np.nan, np.nan, np.nan
    ang = np.deg2rad(arr)
    z = np.exp(1j * ang)
    mean_vec = np.mean(z)
    mu = float(np.mod(np.angle(mean_vec), 2.0 * np.pi))
    R = float(np.abs(mean_vec))
    circ_sd = float(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12)))))
    return float(np.rad2deg(mu)), R, circ_sd


def _top_q_mask(expr: np.ndarray, q: float) -> np.ndarray:
    x = np.asarray(expr, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = int(max(1, round(float(q) * n)))
    ord_idx = np.argsort(x, kind="mergesort")
    keep = ord_idx[-k:]
    out = np.zeros(n, dtype=bool)
    out[keep] = True
    return out


def _approx_medoid_index(
    coords: np.ndarray, seed: int, max_candidates: int = 1000
) -> int:
    xy = np.asarray(coords, dtype=float)
    n = xy.shape[0]
    if n == 0:
        return 0
    if n <= max_candidates:
        candidates = np.arange(n, dtype=int)
    else:
        rng = np.random.default_rng(int(seed))
        candidates = np.sort(
            rng.choice(n, size=int(max_candidates), replace=False).astype(int)
        )

    best_idx = int(candidates[0])
    best_score = np.inf
    for idx in candidates.tolist():
        d = np.linalg.norm(xy - xy[idx], axis=1)
        s = float(np.sum(d))
        if s < best_score:
            best_score = s
            best_idx = int(idx)
    return best_idx


def _density_mode_index(coords: np.ndarray, k: int = 30) -> int:
    xy = np.asarray(coords, dtype=float)
    n = xy.shape[0]
    if n <= 2:
        return 0
    kk = int(max(2, min(k, n - 1)))
    nn = NearestNeighbors(n_neighbors=kk, metric="euclidean")
    nn.fit(xy)
    d, _ = nn.kneighbors(xy)
    # Exclude self-distance column when present.
    if d.shape[1] > 1:
        score = np.mean(d[:, 1:], axis=1)
    else:
        score = np.mean(d, axis=1)
    return int(np.argmin(score))


def _compute_vantages(coords: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    xy = np.asarray(coords, dtype=float)
    centroid = compute_vantage_point(xy, method="mean")

    medoid_idx = _approx_medoid_index(xy, seed=seed)
    medoid = xy[medoid_idx].astype(float)

    dens_idx = _density_mode_index(xy, k=30)
    density = xy[dens_idx].astype(float)

    centered = xy - centroid[None, :]
    if (
        centered.shape[0] >= 2
        and np.nanstd(centered[:, 0]) + np.nanstd(centered[:, 1]) > 0
    ):
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = vt[0, :]
        norm = float(np.linalg.norm(pc1))
        if norm > 0:
            pc1 = pc1 / norm
        alpha = 0.5 * float(np.std(np.linalg.norm(centered, axis=1)))
        anchor = centroid + alpha * pc1
    else:
        anchor = centroid.copy()

    return {
        "V0_centroid": np.asarray(centroid, dtype=float),
        "V1_medoid": np.asarray(medoid, dtype=float),
        "V2_density_mode": np.asarray(density, dtype=float),
        "V3_pc1_anchor": np.asarray(anchor, dtype=float),
    }


def _combo_label(vantage_id: str, n_bins: int, smoothing_id: str) -> str:
    vid_short = vantage_id.split("_")[0] if "_" in vantage_id else vantage_id
    sid_short = smoothing_id.replace("S", "S")
    return f"{vid_short}_B{int(n_bins)}_{sid_short}"


def _score_all(
    *,
    donor_star: str,
    panel_status: list[GeneStatus],
    expr_matrix_cm: Any,
    embeddings: list[EmbeddingSpec],
    bins_grid: list[int],
    fg_modes: list[str],
    q_top: float,
    n_perm: int,
    seed: int,
    qc_total_counts: np.ndarray | None,
    qc_pct_mt: np.ndarray | None,
    qc_pct_ribo: np.ndarray | None,
    out_tables_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test_counter = 0

    # Smoothing auto-detect: not supported in current rsp/scoring APIs; keep explicit fixed level.
    smoothing_levels = ["S0_none"]

    for emb_i, emb in enumerate(embeddings):
        coords = np.asarray(emb.coords, dtype=float)
        vantages = _compute_vantages(coords, seed=seed + emb_i * 101)

        # Precompute theta/bin assignments per vantage x bins.
        geom_map: dict[tuple[str, int], dict[str, Any]] = {}
        for v_id, center in vantages.items():
            theta = compute_theta(coords, center)
            for b in bins_grid:
                _, bin_id = bin_theta(theta, bins=int(b))
                counts = np.bincount(bin_id, minlength=int(b)).astype(float)
                geom_map[(v_id, int(b))] = {
                    "theta": theta,
                    "bin_id": bin_id,
                    "bin_counts_total": counts,
                    "center": center,
                }

        for g_i, st in enumerate(panel_status):
            if not st.present or st.gene_idx is None:
                continue
            expr = get_feature_vector(expr_matrix_cm, int(st.gene_idx))

            fg_map: dict[str, np.ndarray] = {}
            if "detect" in fg_modes:
                fg_map["F0_detect"] = np.asarray(expr, dtype=float) > 0.0
            if "topq" in fg_modes:
                fg_map["F1_topq"] = _top_q_mask(
                    np.asarray(expr, dtype=float), q=float(q_top)
                )

            for fg_name, fg in fg_map.items():
                ff = fg.astype(float)
                rho_tc = _safe_spearman(ff, qc_total_counts)
                rho_mt = _safe_spearman(ff, qc_pct_mt)
                rho_rb = _safe_spearman(ff, qc_pct_ribo)
                qvals = np.array([rho_tc, rho_mt, rho_rb], dtype=float)
                finite = qvals[np.isfinite(qvals)]
                qc_risk = float(np.max(np.abs(finite))) if finite.size > 0 else 0.0

                n_cells = int(fg.size)
                n_fg = int(fg.sum())
                prev = float(n_fg / max(1, n_cells))

                for v_i, (v_id, _) in enumerate(vantages.items()):
                    for b_i, b in enumerate(bins_grid):
                        g = geom_map[(v_id, int(b))]
                        theta = np.asarray(g["theta"], dtype=float)
                        bin_id = np.asarray(g["bin_id"], dtype=int)
                        bin_counts_total = np.asarray(
                            g["bin_counts_total"], dtype=float
                        )

                        for s_id in smoothing_levels:
                            combo = _combo_label(v_id, int(b), s_id)
                            underpowered = bool(
                                prev < UNDERPOWERED_PREV or n_fg < UNDERPOWERED_MIN_FG
                            )

                            if n_fg == 0 or n_fg == n_cells:
                                e_obs = np.zeros(int(b), dtype=float)
                                t_obs = 0.0
                                p_t = np.nan
                                z_t = np.nan
                                coverage = np.nan
                                peaks = np.nan
                                phi_hat = np.nan
                                underpowered = True
                            else:
                                e_obs, _, _, _ = compute_rsp_profile_from_boolean(
                                    fg,
                                    theta,
                                    int(b),
                                    bin_id=bin_id,
                                    bin_counts_total=bin_counts_total,
                                )
                                t_obs = float(np.max(np.abs(e_obs)))
                                phi_idx = int(np.argmax(np.abs(e_obs)))
                                phi_hat = float(
                                    np.degrees(theta_bin_centers(int(b))[phi_idx])
                                    % 360.0
                                )

                                if underpowered:
                                    p_t = np.nan
                                    z_t = np.nan
                                    coverage = np.nan
                                    peaks = np.nan
                                else:
                                    perm_seed = int(
                                        seed
                                        + emb_i * 100000
                                        + g_i * 10000
                                        + v_i * 1000
                                        + b_i * 100
                                        + (0 if fg_name == "F0_detect" else 1)
                                    )
                                    perm = perm_null_T_and_profile(
                                        expr=fg.astype(float),
                                        theta=theta,
                                        donor_ids=None,
                                        n_bins=int(b),
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
                                    coverage = float(
                                        coverage_from_null(e_obs, null_e, q=0.95)
                                    )
                                    peaks = float(
                                        peak_count(
                                            e_obs, null_e, smooth_w=3, q_prom=0.95
                                        )
                                    )

                            row = {
                                "donor_id": donor_star,
                                "gene": st.gene,
                                "resolved_gene": st.resolved_gene,
                                "marker_group": st.marker_group,
                                "embedding": emb.key,
                                "foreground_mode": fg_name,
                                "vantage_id": v_id,
                                "n_bins": int(b),
                                "smoothing_id": s_id,
                                "combo_id": combo,
                                "prev": prev,
                                "n_fg": n_fg,
                                "n_cells": n_cells,
                                "T_obs": float(t_obs),
                                "p_T": (
                                    float(p_t) if np.isfinite(float(p_t)) else np.nan
                                ),
                                "Z_T": (
                                    float(z_t) if np.isfinite(float(z_t)) else np.nan
                                ),
                                "coverage_C": (
                                    float(coverage)
                                    if np.isfinite(float(coverage))
                                    else np.nan
                                ),
                                "peaks_K": (
                                    float(peaks)
                                    if np.isfinite(float(peaks))
                                    else np.nan
                                ),
                                "phi_hat_deg": (
                                    float(phi_hat)
                                    if np.isfinite(float(phi_hat))
                                    else np.nan
                                ),
                                "underpowered_flag": bool(underpowered),
                                "rho_total_counts": float(rho_tc),
                                "rho_pct_mt": float(rho_mt),
                                "rho_pct_ribo": float(rho_rb),
                                "qc_risk": float(qc_risk),
                            }
                            rows.append(row)
                            test_counter += 1

                            if test_counter % 50 == 0:
                                pd.DataFrame(rows).to_csv(
                                    out_tables_dir
                                    / "per_test_scores_long.intermediate.csv",
                                    index=False,
                                )
                                print(
                                    f"[Progress] scored tests {test_counter}; intermediate -> "
                                    f"{out_tables_dir / 'per_test_scores_long.intermediate.csv'}"
                                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH within embedding x foreground x (vantage,bins,smoothing) across genes.
    df["q_T_within_condition"] = np.nan
    for _, idx in df.groupby(
        ["embedding", "foreground_mode", "vantage_id", "n_bins", "smoothing_id"]
    ).groups.items():
        p = df.loc[idx, "p_T"].to_numpy(dtype=float)
        finite = np.isfinite(p)
        if int(finite.sum()) == 0:
            continue
        q = np.full(p.shape, np.nan, dtype=float)
        q[finite] = bh_fdr(p[finite])
        df.loc[idx, "q_T_within_condition"] = q

    # Conservative BH across all tests per embedding+foreground.
    df["q_T_global_embedding_foreground"] = np.nan
    for _, idx in df.groupby(["embedding", "foreground_mode"]).groups.items():
        p = df.loc[idx, "p_T"].to_numpy(dtype=float)
        finite = np.isfinite(p)
        if int(finite.sum()) == 0:
            continue
        q = np.full(p.shape, np.nan, dtype=float)
        q[finite] = bh_fdr(p[finite])
        df.loc[idx, "q_T_global_embedding_foreground"] = q

    df["class_label"] = [
        _classify(float(q), float(k), bool(u))
        for q, k, u in zip(
            df["q_T_within_condition"],
            df["peaks_K"],
            df["underpowered_flag"],
            strict=False,
        )
    ]

    return df


def _summarize_stability(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    rows = []
    for (gene, emb, fg), sub in long_df.groupby(
        ["gene", "embedding", "foreground_mode"], sort=False
    ):
        q = sub["q_T_within_condition"].to_numpy(dtype=float)
        sig = np.isfinite(q) & (q <= Q_SIG)
        frac_sig = float(np.mean(sig))

        cls = sub["class_label"].astype(str)
        cc = cls.value_counts(dropna=False)
        class_mode = str(cc.index[0]) if len(cc) > 0 else "Not-localized"
        class_stability = float(cc.iloc[0] / max(1, len(sub)))

        phi = sub.loc[sig & np.isfinite(sub["phi_hat_deg"]), "phi_hat_deg"].to_numpy(
            dtype=float
        )
        if phi.size > 0:
            phi_mean, R, circ_sd = _circular_stats_deg(phi)
        else:
            phi_mean, R, circ_sd = np.nan, np.nan, np.nan

        robust = bool(
            frac_sig >= 0.70
            and class_stability >= 0.70
            and np.isfinite(R)
            and float(R) >= 0.60
        )

        rows.append(
            {
                "gene": str(gene),
                "embedding": str(emb),
                "foreground_mode": str(fg),
                "n_tests": int(len(sub)),
                "frac_sig": frac_sig,
                "class_mode": class_mode,
                "class_stability": class_stability,
                "phi_mean_deg": phi_mean,
                "R": R,
                "circ_sd": circ_sd,
                "robust_internal": robust,
                "median_Z": float(np.nanmedian(sub["Z_T"].to_numpy(dtype=float))),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["embedding", "foreground_mode", "robust_internal", "frac_sig", "R"],
        ascending=[True, True, False, False, False],
    )
    return out


def _sensitivity_attribution(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["gene", "embedding", "foreground_mode"]
    for keys, sub in long_df.groupby(group_cols, sort=False):
        gene, emb, fg = keys
        z = sub["Z_T"].to_numpy(dtype=float)
        finite = np.isfinite(z)
        if int(finite.sum()) < 2:
            rows.append(
                {
                    "gene": str(gene),
                    "embedding": str(emb),
                    "foreground_mode": str(fg),
                    "n_tests": int(len(sub)),
                    "var_total": np.nan,
                    "var_vantage_frac": np.nan,
                    "var_bins_frac": np.nan,
                    "var_smoothing_frac": np.nan,
                    "dominant_sensitivity": "insufficient",
                }
            )
            continue

        subf = sub.loc[finite].copy()
        zz = subf["Z_T"].to_numpy(dtype=float)
        mu = float(np.mean(zz))
        ss_total = float(np.sum((zz - mu) ** 2))
        if ss_total <= 1e-12:
            rows.append(
                {
                    "gene": str(gene),
                    "embedding": str(emb),
                    "foreground_mode": str(fg),
                    "n_tests": int(len(subf)),
                    "var_total": 0.0,
                    "var_vantage_frac": 0.0,
                    "var_bins_frac": 0.0,
                    "var_smoothing_frac": 0.0,
                    "dominant_sensitivity": "none",
                }
            )
            continue

        def ss_factor(col: str) -> float:
            ss = 0.0
            for _, g in subf.groupby(col):
                m = float(np.mean(g["Z_T"].to_numpy(dtype=float)))
                ss += float(len(g)) * (m - mu) ** 2
            return float(ss)

        ss_v = ss_factor("vantage_id")
        ss_b = ss_factor("n_bins")
        ss_s = ss_factor("smoothing_id")

        fv = float(max(0.0, min(1.0, ss_v / ss_total)))
        fb = float(max(0.0, min(1.0, ss_b / ss_total)))
        fs = float(max(0.0, min(1.0, ss_s / ss_total)))

        dom = max(
            [("vantage", fv), ("bins", fb), ("smoothing", fs)], key=lambda x: x[1]
        )[0]

        rows.append(
            {
                "gene": str(gene),
                "embedding": str(emb),
                "foreground_mode": str(fg),
                "n_tests": int(len(subf)),
                "var_total": ss_total / max(1, len(subf)),
                "var_vantage_frac": fv,
                "var_bins_frac": fb,
                "var_smoothing_frac": fs,
                "dominant_sensitivity": dom,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["embedding", "foreground_mode", "var_total"], ascending=[True, True, False]
    )
    return out


def _plot_overview(
    out_dir: Path,
    umap_coords: np.ndarray,
    expr_by_gene: dict[str, np.ndarray],
    parameter_grid_df: pd.DataFrame,
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

    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    ax.axis("off")
    show = parameter_grid_df[
        ["combo_id", "vantage_id", "n_bins", "smoothing_id"]
    ].copy()
    tbl = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    ax.set_title("BioRSP internal parameter grid (vantage x bins x smoothing)")
    fig.tight_layout()
    fig.savefig(out_dir / "parameter_grid_table.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)


def _plot_stability_heatmaps(
    out_dir: Path,
    long_df: pd.DataFrame,
    gene_order: list[str],
    combo_order: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if long_df.empty:
        _save_placeholder(out_dir / "empty.png", "Stability heatmaps", "No tests")
        return

    for (emb, fg), sub in long_df.groupby(["embedding", "foreground_mode"], sort=False):
        piv_z = sub.pivot(index="gene", columns="combo_id", values="Z_T").reindex(
            index=gene_order, columns=combo_order
        )
        piv_q = sub.pivot(
            index="gene", columns="combo_id", values="q_T_within_condition"
        ).reindex(index=gene_order, columns=combo_order)
        piv_k = sub.pivot(index="gene", columns="combo_id", values="peaks_K").reindex(
            index=gene_order, columns=combo_order
        )

        mat_z = np.clip(
            np.nan_to_num(piv_z.to_numpy(dtype=float), nan=0.0), -10.0, 10.0
        )
        mat_q = np.clip(
            -np.log10(
                np.clip(
                    np.nan_to_num(piv_q.to_numpy(dtype=float), nan=1.0), 1e-300, None
                )
            ),
            0.0,
            5.0,
        )
        mat_k = np.nan_to_num(piv_k.to_numpy(dtype=float), nan=-1.0)

        # Z heatmap
        fig1, ax1 = plt.subplots(
            figsize=(1.0 * len(combo_order) + 3.6, 0.6 * len(gene_order) + 2.2)
        )
        im1 = ax1.imshow(mat_z, aspect="auto", cmap="magma", vmin=-10, vmax=10)
        ax1.set_xticks(np.arange(len(combo_order)))
        ax1.set_xticklabels(combo_order, rotation=60, ha="right", fontsize=7)
        ax1.set_yticks(np.arange(len(gene_order)))
        ax1.set_yticklabels(gene_order, fontsize=8)
        ax1.set_title(f"{emb} / {fg}: Z_T heatmap")
        cb1 = fig1.colorbar(im1, ax=ax1)
        cb1.set_label("Z_T (capped ±10)")
        fig1.tight_layout()
        fig1.savefig(out_dir / f"{emb}_{fg}_heatmap_ZT.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig1)

        # -log10 q heatmap
        fig2, ax2 = plt.subplots(
            figsize=(1.0 * len(combo_order) + 3.6, 0.6 * len(gene_order) + 2.2)
        )
        im2 = ax2.imshow(mat_q, aspect="auto", cmap="viridis", vmin=0, vmax=5)
        ax2.set_xticks(np.arange(len(combo_order)))
        ax2.set_xticklabels(combo_order, rotation=60, ha="right", fontsize=7)
        ax2.set_yticks(np.arange(len(gene_order)))
        ax2.set_yticklabels(gene_order, fontsize=8)
        ax2.set_title(f"{emb} / {fg}: -log10(q_T_within_condition)")
        cb2 = fig2.colorbar(im2, ax=ax2)
        cb2.set_label("-log10(q), cap=5")
        fig2.tight_layout()
        fig2.savefig(
            out_dir / f"{emb}_{fg}_heatmap_neglog10q.png", dpi=DEFAULT_PLOT_STYLE.dpi
        )
        plt.close(fig2)

        # peaks_K heatmap
        fig3, ax3 = plt.subplots(
            figsize=(1.0 * len(combo_order) + 3.6, 0.6 * len(gene_order) + 2.2)
        )
        im3 = ax3.imshow(mat_k, aspect="auto", cmap="cividis")
        ax3.set_xticks(np.arange(len(combo_order)))
        ax3.set_xticklabels(combo_order, rotation=60, ha="right", fontsize=7)
        ax3.set_yticks(np.arange(len(gene_order)))
        ax3.set_yticklabels(gene_order, fontsize=8)
        ax3.set_title(f"{emb} / {fg}: peaks_K")
        cb3 = fig3.colorbar(im3, ax=ax3)
        cb3.set_label("peaks_K")
        fig3.tight_layout()
        fig3.savefig(
            out_dir / f"{emb}_{fg}_heatmap_peaksK.png", dpi=DEFAULT_PLOT_STYLE.dpi
        )
        plt.close(fig3)


def _plot_direction_stability(
    out_dir: Path, long_df: pd.DataFrame, summary_df: pd.DataFrame
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if long_df.empty or summary_df.empty:
        _save_placeholder(out_dir / "empty.png", "Direction stability", "No tests")
        return

    # 1) Circular scatter per embedding with genes x foreground columns.
    for emb in sorted(summary_df["embedding"].astype(str).unique().tolist()):
        genes = sorted(summary_df["gene"].astype(str).unique().tolist())
        fgs = sorted(summary_df["foreground_mode"].astype(str).unique().tolist())
        n_rows = len(genes)
        n_cols = len(fgs)
        fig = plt.figure(figsize=(4.0 * n_cols, 3.2 * n_rows))
        for i, gene in enumerate(genes):
            for j, fg in enumerate(fgs):
                ax = fig.add_subplot(
                    n_rows, n_cols, i * n_cols + j + 1, projection="polar"
                )
                sub = long_df.loc[
                    (long_df["embedding"] == emb)
                    & (long_df["gene"] == gene)
                    & (long_df["foreground_mode"] == fg)
                    & (long_df["q_T_within_condition"] <= Q_SIG)
                ]
                phi = sub["phi_hat_deg"].to_numpy(dtype=float)
                phi = phi[np.isfinite(phi)]
                if phi.size == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "no sig",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                else:
                    rad = np.deg2rad(phi)
                    ax.scatter(rad, np.ones_like(rad), s=24, c="#1f77b4", alpha=0.85)
                    mu, R, csd = _circular_stats_deg(phi)
                    ax.plot(
                        [np.deg2rad(mu), np.deg2rad(mu)],
                        [0.0, 1.1],
                        color="#d62728",
                        linewidth=2.0,
                    )
                    ax.text(
                        0.02,
                        0.02,
                        f"R={R:.2f}\ncsd={csd:.2f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=7,
                        bbox={"facecolor": "white", "edgecolor": "#999", "alpha": 0.8},
                    )
                ax.set_rticks([])
                ax.set_theta_zero_location("E")
                ax.set_theta_direction(1)
                ax.set_title(f"{gene} / {fg}", fontsize=8)
        fig.suptitle(f"{emb}: phi stability over significant parameter combos", y=0.995)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
        fig.savefig(
            out_dir / f"{emb}_phi_circular_grid.png", dpi=DEFAULT_PLOT_STYLE.dpi
        )
        plt.close(fig)

        # 2) R vs frac_sig (color by foreground)
        fig2, ax2 = plt.subplots(figsize=(7.6, 5.8))
        for fg, sub in summary_df.loc[summary_df["embedding"] == emb].groupby(
            "foreground_mode", sort=False
        ):
            ax2.scatter(
                sub["R"].to_numpy(dtype=float),
                sub["frac_sig"].to_numpy(dtype=float),
                s=95,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.4,
                label=str(fg),
            )
            for _, r in sub.iterrows():
                ax2.text(
                    float(r["R"]),
                    float(r["frac_sig"]) + 0.01,
                    str(r["gene"]),
                    fontsize=8,
                )
        ax2.axvline(0.60, color="#444", linestyle="--", linewidth=1.0)
        ax2.axhline(0.70, color="#444", linestyle=":", linewidth=1.0)
        ax2.set_xlabel("R")
        ax2.set_ylabel("frac_sig")
        ax2.set_title(f"{emb}: R vs frac_sig")
        ax2.legend(loc="best", fontsize=8)
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{emb}_R_vs_frac_sig.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig2)

    # 3) circ_sd vs frac_sig colored by embedding.
    fig3, ax3 = plt.subplots(figsize=(7.8, 6.0))
    emb_colors = {"pca2d": "#4c78a8", "umap_repr": "#f58518"}
    fg_mark = {"F0_detect": "o", "F1_topq": "^"}
    for _, r in summary_df.iterrows():
        emb = str(r["embedding"])
        fg = str(r["foreground_mode"])
        ax3.scatter(
            float(r["circ_sd"]),
            float(r["frac_sig"]),
            s=90,
            marker=fg_mark.get(fg, "o"),
            c=emb_colors.get(emb, "#777777"),
            edgecolors="black",
            linewidths=0.4,
            alpha=0.88,
        )
        ax3.text(
            float(r["circ_sd"]), float(r["frac_sig"]) + 0.01, str(r["gene"]), fontsize=8
        )
    ax3.set_xlabel("circ_sd")
    ax3.set_ylabel("frac_sig")
    ax3.set_title("circ_sd vs frac_sig (color=embedding, marker=foreground)")
    fig3.tight_layout()
    fig3.savefig(
        out_dir / "circsd_vs_fracsig_by_embedding.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig3)


def _plot_parameter_effects(
    out_dir: Path, long_df: pd.DataFrame, bins_grid: list[int]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if long_df.empty:
        _save_placeholder(out_dir / "empty.png", "Parameter effects", "No tests")
        return

    # 1) Z grouped by vantage per n_bins (per emb/fg).
    for (emb, fg), sub in long_df.groupby(["embedding", "foreground_mode"], sort=False):
        fig, axes = plt.subplots(
            1, len(bins_grid), figsize=(5.0 * len(bins_grid), 4.8), squeeze=False
        )
        for i, b in enumerate(bins_grid):
            ax = axes[0, i]
            sb = sub.loc[sub["n_bins"] == int(b)]
            groups = []
            labels = []
            for v_id, gv in sb.groupby("vantage_id", sort=False):
                groups.append(gv["Z_T"].to_numpy(dtype=float))
                labels.append(str(v_id))
            if len(groups) == 0:
                ax.axis("off")
                continue
            ax.boxplot(groups, tick_labels=labels, patch_artist=True)
            ax.tick_params(axis="x", rotation=35)
            ax.set_title(f"n_bins={b}")
            ax.set_ylabel("Z_T")
        fig.suptitle(
            f"{emb} / {fg}: Z_T grouped by vantage (subplots by n_bins)", y=0.995
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        fig.savefig(
            out_dir / f"{emb}_{fg}_box_Z_by_vantage_per_bins.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
        )
        plt.close(fig)

        # 2) Z grouped by n_bins with points colored by vantage.
        fig2, ax2 = plt.subplots(figsize=(8.0, 5.6))
        groups = [
            sub.loc[sub["n_bins"] == int(b), "Z_T"].to_numpy(dtype=float)
            for b in bins_grid
        ]
        ax2.boxplot(
            groups, tick_labels=[str(int(b)) for b in bins_grid], patch_artist=True
        )
        cmap = plt.get_cmap("tab10")
        vid_map = {
            v: cmap(i % 10)
            for i, v in enumerate(
                sorted(sub["vantage_id"].astype(str).unique().tolist())
            )
        }
        x_pos = {int(b): i + 1 for i, b in enumerate(bins_grid)}
        for _, r in sub.iterrows():
            b = int(r["n_bins"])
            jitter = 0.06 * ((hash((str(r["gene"]), str(r["vantage_id"]))) % 11) - 5)
            ax2.scatter(
                x_pos[b] + jitter,
                float(r["Z_T"]) if np.isfinite(float(r["Z_T"])) else np.nan,
                s=45,
                c=[vid_map.get(str(r["vantage_id"]), "#777777")],
                alpha=0.72,
                edgecolors="black",
                linewidths=0.25,
            )
        ax2.set_xlabel("n_bins")
        ax2.set_ylabel("Z_T")
        ax2.set_title(
            f"{emb} / {fg}: Z_T grouped by n_bins (points colored by vantage)"
        )
        fig2.tight_layout()
        fig2.savefig(
            out_dir / f"{emb}_{fg}_box_Z_by_bins.png", dpi=DEFAULT_PLOT_STYLE.dpi
        )
        plt.close(fig2)

    # 3) Smoothing effects placeholder (not applicable).
    _save_placeholder(
        out_dir / "smoothing_effects.png",
        "Smoothing effects",
        "Smoothing sweep not applicable: current BioRSP API has no smoothing parameter.",
    )


def _recompute_condition(
    *,
    expr: np.ndarray,
    fg: np.ndarray,
    coords: np.ndarray,
    vantage: np.ndarray,
    n_bins: int,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    n_fg = int(fg.sum())
    n_cells = int(fg.size)
    prev = float(n_fg / max(1, n_cells))
    underpowered = bool(prev < UNDERPOWERED_PREV or n_fg < UNDERPOWERED_MIN_FG)

    theta = compute_theta(coords, vantage)
    _, bin_id = bin_theta(theta, bins=int(n_bins))
    bin_counts_total = np.bincount(bin_id, minlength=int(n_bins)).astype(float)

    if n_fg == 0 or n_fg == n_cells:
        return {
            "E_obs": np.zeros(int(n_bins), dtype=float),
            "null_E": None,
            "null_T": None,
            "T_obs": 0.0,
            "p_T": np.nan,
            "Z_T": np.nan,
            "coverage_C": np.nan,
            "peaks_K": np.nan,
            "phi_hat_deg": np.nan,
            "underpowered": True,
        }

    e_obs, _, _, _ = compute_rsp_profile_from_boolean(
        fg,
        theta,
        int(n_bins),
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )
    t_obs = float(np.max(np.abs(e_obs)))
    phi_idx = int(np.argmax(np.abs(e_obs)))
    phi_hat = float(np.degrees(theta_bin_centers(int(n_bins))[phi_idx]) % 360.0)

    if underpowered:
        return {
            "E_obs": np.asarray(e_obs, dtype=float),
            "null_E": None,
            "null_T": None,
            "T_obs": t_obs,
            "p_T": np.nan,
            "Z_T": np.nan,
            "coverage_C": np.nan,
            "peaks_K": np.nan,
            "phi_hat_deg": phi_hat,
            "underpowered": True,
        }

    perm = perm_null_T_and_profile(
        expr=fg.astype(float),
        theta=theta,
        donor_ids=None,
        n_bins=int(n_bins),
        n_perm=int(n_perm),
        seed=int(seed),
        donor_stratified=False,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )
    null_e = np.asarray(perm["null_E_phi"], dtype=float)
    null_t = np.asarray(perm["null_T"], dtype=float)

    return {
        "E_obs": np.asarray(e_obs, dtype=float),
        "null_E": null_e,
        "null_T": null_t,
        "T_obs": float(perm["T_obs"]),
        "p_T": float(perm["p_T"]),
        "Z_T": float(robust_z(float(perm["T_obs"]), null_t)),
        "coverage_C": float(coverage_from_null(e_obs, null_e, q=0.95)),
        "peaks_K": float(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95)),
        "phi_hat_deg": phi_hat,
        "underpowered": False,
    }


def _plot_exemplar_panels(
    out_dir: Path,
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    embeddings_map: dict[str, EmbeddingSpec],
    expr_by_gene: dict[str, np.ndarray],
    q_top: float,
    bins_grid: list[int],
    n_perm: int,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        _save_placeholder(
            out_dir / "no_exemplars.png", "Exemplar panels", "No summary rows"
        )
        return

    # Use UMAP/F1 when available for exemplar choice.
    sub_ref = summary_df.loc[
        (summary_df["embedding"] == "umap_repr")
        & (summary_df["foreground_mode"] == "F1_topq")
    ]
    if sub_ref.empty:
        sub_ref = summary_df.copy()

    robust_row = sub_ref.sort_values(
        by=["frac_sig", "R"], ascending=[False, False]
    ).iloc[0]
    unstable_candidates = sub_ref.loc[sub_ref["frac_sig"] > 0.0]
    if unstable_candidates.empty:
        unstable_row = sub_ref.sort_values(by="frac_sig", ascending=True).iloc[0]
    else:
        unstable_row = unstable_candidates.sort_values(
            by=["frac_sig", "R"], ascending=[True, True]
        ).iloc[0]

    exemplar_genes = [str(robust_row["gene"])]
    ug = str(unstable_row["gene"])
    if ug not in exemplar_genes:
        exemplar_genes.append(ug)

    # Precompute vantages on UMAP embedding.
    if "umap_repr" not in embeddings_map:
        _save_placeholder(
            out_dir / "no_umap.png", "Exemplar panels", "umap_repr missing"
        )
        return

    coords = embeddings_map["umap_repr"].coords
    vantages = _compute_vantages(coords, seed=seed + 409)
    b_default = 64 if 64 in bins_grid else int(bins_grid[len(bins_grid) // 2])
    fg_name = (
        "F1_topq"
        if "F1_topq" in long_df["foreground_mode"].astype(str).unique()
        else "F0_detect"
    )

    for ex_i, gene in enumerate(exemplar_genes, start=1):
        expr = expr_by_gene.get(gene, None)
        if expr is None:
            continue

        fg = (
            _top_q_mask(expr, q=q_top)
            if fg_name == "F1_topq"
            else (np.asarray(expr, dtype=float) > 0.0)
        )

        n_cols = max(len(vantages), len(bins_grid))
        fig = plt.figure(figsize=(4.1 * n_cols, 8.2))
        gs = fig.add_gridspec(2, n_cols, wspace=0.28, hspace=0.32)

        # Row1: each vantage at n_bins=64.
        for j, (v_id, center) in enumerate(vantages.items()):
            ax = fig.add_subplot(gs[0, j], projection="polar")
            res = _recompute_condition(
                expr=np.asarray(expr, dtype=float),
                fg=fg,
                coords=coords,
                vantage=np.asarray(center, dtype=float),
                n_bins=int(b_default),
                n_perm=int(n_perm),
                seed=int(seed + ex_i * 1000 + j * 17 + 1),
            )
            e_obs = np.asarray(res["E_obs"], dtype=float)
            th = theta_bin_centers(int(b_default))
            th_c = np.concatenate([th, th[:1]])
            obs_c = np.concatenate([e_obs, e_obs[:1]])
            ax.plot(th_c, obs_c, color="#8B0000", linewidth=2.0)
            if res["null_E"] is not None:
                null_e = np.asarray(res["null_E"], dtype=float)
                hi = np.quantile(null_e, 0.95, axis=0)
                lo = np.quantile(null_e, 0.05, axis=0)
                ax.plot(
                    th_c,
                    np.concatenate([hi, hi[:1]]),
                    color="#333",
                    linestyle="--",
                    linewidth=1.1,
                )
                ax.plot(
                    th_c,
                    np.concatenate([lo, lo[:1]]),
                    color="#333",
                    linestyle="--",
                    linewidth=0.9,
                )
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_title(
                f"{v_id}\nZ={float(res['Z_T']):.2f}, q={float(res['p_T']):.2e}\n"
                f"C={float(res['coverage_C']):.3f}, K={int(res['peaks_K']) if np.isfinite(float(res['peaks_K'])) else -1}, "
                f"phi={float(res['phi_hat_deg']):.1f}",
                fontsize=8,
            )

        for j in range(len(vantages), n_cols):
            ax = fig.add_subplot(gs[0, j])
            ax.axis("off")

        # Row2: n_bins sweep at centroid.
        centroid = vantages["V0_centroid"]
        for j, b in enumerate(bins_grid):
            ax = fig.add_subplot(gs[1, j], projection="polar")
            res = _recompute_condition(
                expr=np.asarray(expr, dtype=float),
                fg=fg,
                coords=coords,
                vantage=np.asarray(centroid, dtype=float),
                n_bins=int(b),
                n_perm=int(n_perm),
                seed=int(seed + ex_i * 2000 + j * 19 + 3),
            )
            e_obs = np.asarray(res["E_obs"], dtype=float)
            th = theta_bin_centers(int(b))
            th_c = np.concatenate([th, th[:1]])
            obs_c = np.concatenate([e_obs, e_obs[:1]])
            ax.plot(th_c, obs_c, color="#8B0000", linewidth=2.0)
            if res["null_E"] is not None:
                null_e = np.asarray(res["null_E"], dtype=float)
                hi = np.quantile(null_e, 0.95, axis=0)
                lo = np.quantile(null_e, 0.05, axis=0)
                ax.plot(
                    th_c,
                    np.concatenate([hi, hi[:1]]),
                    color="#333",
                    linestyle="--",
                    linewidth=1.1,
                )
                ax.plot(
                    th_c,
                    np.concatenate([lo, lo[:1]]),
                    color="#333",
                    linestyle="--",
                    linewidth=0.9,
                )
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_title(
                f"V0_centroid, B={int(b)}\nZ={float(res['Z_T']):.2f}, q={float(res['p_T']):.2e}\n"
                f"C={float(res['coverage_C']):.3f}, K={int(res['peaks_K']) if np.isfinite(float(res['peaks_K'])) else -1}, "
                f"phi={float(res['phi_hat_deg']):.1f}",
                fontsize=8,
            )

        for j in range(len(bins_grid), n_cols):
            ax = fig.add_subplot(gs[1, j])
            ax.axis("off")

        fig.suptitle(
            f"Exemplar {gene} ({fg_name}, umap_repr): row1 vantage sweep @ B={b_default}, row2 bin sweep @ V0",
            y=0.995,
            fontsize=12,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
        fig.savefig(out_dir / f"exemplar_{gene}.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig)


def _write_readme(
    out_path: Path,
    *,
    seed: int,
    n_perm: int,
    bins_grid: list[int],
    fg_modes: list[str],
    q_top: float,
    donor_key: str,
    label_key: str,
    donor_star: str,
    cm_labels: list[str],
    cm_counts: dict[str, int],
    expr_source: str,
    embed_note: str,
    n_cells_donor: int,
    n_cells_cm: int,
    summary_df: pd.DataFrame,
    smoothing_supported: bool,
) -> None:
    lines: list[str] = []
    lines.append(
        "CM Experiment #4 (Single-donor): BioRSP-internal robustness — vantage + angular binning"
    )
    lines.append("")
    lines.append("Hypothesis")
    lines.append(
        "For genuine cardiomyocyte programs, BioRSP localization calls should remain stable across reasonable "
        "vantage/origin and angular-binning choices; narrow-parameter-only positives indicate method-induced instability."
    )
    lines.append("")
    lines.append("Interpretation guardrails")
    lines.append(
        "- Single donor only: robustness is via internal sensitivity + calibrated nulls, not donor replication."
    )
    lines.append("- Directionality is representation-conditional, not anatomy.")
    lines.append("- Circular stability reported using R and circ_sd = sqrt(-2 ln R).")
    lines.append("")
    lines.append("Run metadata")
    lines.append(f"- seed: {seed}")
    lines.append(f"- n_perm: {n_perm}")
    lines.append(f"- bins_grid: {', '.join([str(int(b)) for b in bins_grid])}")
    lines.append(f"- foreground_modes: {', '.join(fg_modes)}")
    lines.append(f"- q_top: {q_top}")
    lines.append(f"- donor_key_used: {donor_key}")
    lines.append(f"- label_key_used: {label_key}")
    lines.append(f"- donor_star: {donor_star}")
    lines.append(f"- expression_source_used: {expr_source}")
    lines.append(f"- embedding_input_prep: {embed_note}")
    lines.append(f"- donor_star_total_cells: {n_cells_donor}")
    lines.append(f"- donor_star_cardiomyocytes: {n_cells_cm}")
    lines.append(f"- smoothing_supported: {smoothing_supported}")
    lines.append("")
    lines.append("Cardiomyocyte labels included")
    for lab in cm_labels:
        lines.append(f"- {lab}: {cm_counts.get(lab, 0)}")
    lines.append("")

    if not summary_df.empty:
        lines.append("Robust-internal hits")
        for (emb, fg), sub in summary_df.groupby(
            ["embedding", "foreground_mode"], sort=False
        ):
            hits = sub.loc[sub["robust_internal"], "gene"].astype(str).tolist()
            if len(hits) == 0:
                lines.append(f"- {emb}/{fg}: none")
            else:
                lines.append(f"- {emb}/{fg}: " + ", ".join(hits))
        lines.append("")

    lines.append("Smoothing note")
    if smoothing_supported:
        lines.append("- Smoothing sweep enabled.")
    else:
        lines.append(
            "- Smoothing not applicable in current BioRSP API; fixed at S0_none."
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
        plots_dir / "01_stability_heatmaps",
        plots_dir / "02_direction_stability",
        plots_dir / "03_parameter_effects",
        plots_dir / "04_exemplar_panels",
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
        raise RuntimeError("No CM cells detected by substring match rule")

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
        raise RuntimeError("donor_star CM subset is empty")

    expr_matrix_cm, adata_like_cm, expr_source = _choose_expression_source(
        adata_cm,
        layer_arg=args.layer,
        use_raw_arg=bool(args.use_raw),
    )

    # QC covariates for audit.
    qc_total_counts, key_total = _safe_numeric_obs(
        adata_cm, ["total_counts", "n_counts", "n_genes_by_counts"]
    )
    if qc_total_counts is None:
        qc_total_counts = _total_counts_vector(adata_cm, expr_matrix_cm)
        _ = "computed:expr_sum"

    qc_pct_mt, key_mt = _safe_numeric_obs(
        adata_cm, ["pct_counts_mt", "percent.mt", "pct_mt"]
    )
    if qc_pct_mt is None:
        qc_pct_mt, key_mt2 = _pct_mt_vector(adata_cm, expr_matrix_cm, adata_like_cm)
        _ = key_mt2

    qc_pct_ribo, key_ribo = _safe_numeric_obs(
        adata_cm, ["pct_counts_ribo", "percent.ribo", "pct_ribo"]
    )

    # Panel resolution.
    panel_status, panel_df = _resolve_panel(adata_like_cm)
    panel_df.to_csv(tables_dir / "gene_panel_status.csv", index=False)

    # Embeddings.
    adata_embed, embed_note = _prepare_embedding_input(
        adata_cm, expr_matrix_cm, expr_source
    )
    embeddings, n_pcs_used = _compute_fixed_embeddings(
        adata_embed,
        seed=int(args.seed),
        k_pca=int(args.k_pca),
    )
    embeddings_map = {e.key: e for e in embeddings}

    # Foreground settings.
    fg_modes_cli = [str(x).strip().lower() for x in args.foregrounds]
    fg_modes = []
    if "detect" in fg_modes_cli:
        fg_modes.append("detect")
    if "topq" in fg_modes_cli:
        fg_modes.append("topq")
    if len(fg_modes) == 0:
        raise ValueError("No valid foreground modes selected. Use detect and/or topq.")

    bins_grid = sorted(list(dict.fromkeys([int(b) for b in args.bins if int(b) > 0])))
    if len(bins_grid) == 0:
        raise ValueError("bins grid is empty after parsing")

    # Score all tests.
    long_df = _score_all(
        donor_star=donor_star,
        panel_status=panel_status,
        expr_matrix_cm=expr_matrix_cm,
        embeddings=embeddings,
        bins_grid=bins_grid,
        fg_modes=fg_modes,
        q_top=float(args.q),
        n_perm=int(args.n_perm),
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

    # Summary tables.
    summary_df = _summarize_stability(long_df)
    summary_df.to_csv(
        tables_dir / "per_gene_embedding_foreground_summary.csv", index=False
    )

    sens_df = _sensitivity_attribution(long_df)
    sens_df.to_csv(tables_dir / "sensitivity_attribution.csv", index=False)

    # Parameter grid table.
    combos = (
        long_df[["combo_id", "vantage_id", "n_bins", "smoothing_id"]]
        .drop_duplicates()
        .sort_values(by=["vantage_id", "n_bins", "smoothing_id"])
        .reset_index(drop=True)
    )

    # Build expression dict for overview/exemplars.
    expr_by_gene: dict[str, np.ndarray] = {}
    for st in panel_status:
        if st.present and st.gene_idx is not None:
            expr_by_gene[st.gene] = get_feature_vector(expr_matrix_cm, int(st.gene_idx))

    # Plots.
    _plot_overview(
        plots_dir / "00_overview",
        umap_coords=embeddings_map["umap_repr"].coords,
        expr_by_gene=expr_by_gene,
        parameter_grid_df=combos,
    )

    gene_order = [
        st.gene for st in panel_status if st.present and st.gene_idx is not None
    ]
    combo_order = [
        _combo_label(v, b, "S0_none")
        for v in ["V0_centroid", "V1_medoid", "V2_density_mode", "V3_pc1_anchor"]
        for b in bins_grid
    ]
    combo_order = [
        c for c in combo_order if c in set(long_df["combo_id"].astype(str).tolist())
    ]
    _plot_stability_heatmaps(
        plots_dir / "01_stability_heatmaps",
        long_df=long_df,
        gene_order=gene_order,
        combo_order=combo_order,
    )

    _plot_direction_stability(
        plots_dir / "02_direction_stability", long_df=long_df, summary_df=summary_df
    )
    _plot_parameter_effects(
        plots_dir / "03_parameter_effects", long_df=long_df, bins_grid=bins_grid
    )
    _plot_exemplar_panels(
        plots_dir / "04_exemplar_panels",
        long_df=long_df,
        summary_df=summary_df,
        embeddings_map=embeddings_map,
        expr_by_gene=expr_by_gene,
        q_top=float(args.q),
        bins_grid=bins_grid,
        n_perm=int(args.n_perm),
        seed=int(args.seed),
    )

    cm_labels = sorted(labels_all.loc[cm_mask_all].unique().tolist())
    cm_counts = labels_all.loc[cm_mask_all].value_counts().to_dict()
    _write_readme(
        out_root / "README.txt",
        seed=int(args.seed),
        n_perm=int(args.n_perm),
        bins_grid=bins_grid,
        fg_modes=fg_modes,
        q_top=float(args.q),
        donor_key=donor_key,
        label_key=label_key,
        donor_star=donor_star,
        cm_labels=cm_labels,
        cm_counts={str(k): int(v) for k, v in cm_counts.items()},
        expr_source=expr_source,
        embed_note=f"{embed_note}; n_pcs_used={n_pcs_used}",
        n_cells_donor=int(adata_donor.n_obs),
        n_cells_cm=int(adata_cm.n_obs),
        summary_df=summary_df,
        smoothing_supported=False,
    )

    # Output checks.
    required = [
        tables_dir / "donor_choice.csv",
        tables_dir / "gene_panel_status.csv",
        tables_dir / "per_test_scores_long.csv",
        tables_dir / "per_gene_embedding_foreground_summary.csv",
        tables_dir / "sensitivity_attribution.csv",
        plots_dir / "00_overview",
        plots_dir / "01_stability_heatmaps",
        plots_dir / "02_direction_stability",
        plots_dir / "03_parameter_effects",
        plots_dir / "04_exemplar_panels",
        out_root / "README.txt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing required outputs: " + ", ".join(missing))

    print(f"donor_key_used={donor_key}")
    print(f"label_key_used={label_key}")
    print(f"donor_star={donor_star}")
    print(f"expression_source_used={expr_source}")
    print(f"bins_grid={json.dumps(bins_grid)}")
    print(f"foreground_modes={json.dumps(fg_modes)}")
    print(f"n_tests={int(len(long_df))}")
    print(f"results_root={out_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
