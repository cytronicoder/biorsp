#!/usr/bin/env python3
"""CM Experiment #5 (single donor): split-half + bootstrap reproducibility of BioRSP.

Hypothesis (pre-registered):
Within one cardiomyocyte-heavy donor, true localized programs produce BioRSP signals
that are stable under within-donor resampling (split-half and bootstrap), while unstable
signals (high variance in Z_T/phi) are likely noise, threshold artifacts, or QC-driven.

Interpretation guardrail:
This is within-donor reproducibility, not donor replication.
Angles are representation-conditional geometry, not anatomy.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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

GENE_PANEL = {
    "Core CM": ["TNNT2", "TNNI3", "MYH6", "MYH7", "RYR2", "ATP2A2"],
    "Stress": ["NPPA", "NPPB"],
    "Optional": ["PPARGC1A", "CPT1B"],
}

UNDERPOWERED_PREV = 0.005
UNDERPOWERED_MIN_FG = 50
Q_SIG = 0.05
QC_RISK_THRESH = 0.35


@dataclass(frozen=True)
class GeneStatus:
    gene: str
    group: str
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
        description="CM Experiment #5: split-half + bootstrap reproducibility (single donor CM)."
    )
    p.add_argument("--h5ad", default="data/processed/HT_pca_umap.h5ad", help="Input .h5ad")
    p.add_argument(
        "--out",
        default="experiments/heart_single_donor_cardiomyocyte/results/cm_experiment5_resampling_reproducibility",
        help="Output root",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--q", type=float, default=0.10, help="Top-quantile foreground for F1")
    p.add_argument("--n_bins", type=int, default=64)
    p.add_argument("--n_perm_split", type=int, default=300)
    p.add_argument("--n_perm_boot", type=int, default=200)
    p.add_argument("--boot_reps", type=int, default=200)
    p.add_argument("--boot_frac", type=float, default=0.80)
    p.add_argument("--save_boot_long", type=_str2bool, default=False)
    p.add_argument("--k_pca", type=int, default=50)
    p.add_argument("--layer", default=None)
    p.add_argument("--use_raw", action="store_true")
    p.add_argument("--donor_key", default=None)
    p.add_argument("--label_key", default=None)
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


def _resolve_key_required(adata: ad.AnnData, requested: str | None, candidates: list[str], purpose: str) -> str:
    if requested is not None:
        if requested in adata.obs.columns:
            return str(requested)
        raise KeyError(f"Requested {purpose} key '{requested}' not in adata.obs")
    for c in candidates:
        if c in adata.obs.columns:
            return c
    raise KeyError(f"No {purpose} key found. Tried: {', '.join(candidates)}")


def _choose_expression_source(adata: ad.AnnData, layer_arg: str | None, use_raw_arg: bool) -> tuple[Any, Any, str]:
    if layer_arg is not None or use_raw_arg:
        return _resolve_expr_matrix(adata, layer=layer_arg, use_raw=bool(use_raw_arg))
    if "counts" in adata.layers:
        return _resolve_expr_matrix(adata, layer="counts", use_raw=False)
    return _resolve_expr_matrix(adata, layer=None, use_raw=False)


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


def _prepare_embedding_input(adata_cm: ad.AnnData, expr_matrix_cm: Any, expr_source: str) -> tuple[ad.AnnData, str]:
    import scanpy as sc

    adata_embed = ad.AnnData(
        X=expr_matrix_cm.copy() if hasattr(expr_matrix_cm, "copy") else np.array(expr_matrix_cm),
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


def _compute_fixed_embeddings(adata_embed: ad.AnnData, seed: int, k_pca: int) -> tuple[list[EmbeddingSpec], int]:
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
        EmbeddingSpec(key="pca2d", coords=pca[:, :2].copy(), params={"n_pcs": n_pcs, "seed": int(seed)}),
        EmbeddingSpec(
            key="umap_repr",
            coords=umap[:, :2].copy(),
            params={"n_neighbors": 30, "min_dist": 0.1, "random_state": 0, "n_pcs": n_pcs},
        ),
    ]
    return specs, n_pcs


def _resolve_panel(adata_like: Any) -> tuple[list[GeneStatus], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    stats: list[GeneStatus] = []
    used_idx = set()

    for group, genes in GENE_PANEL.items():
        for gene in genes:
            try:
                idx, label, sym_col, source = resolve_feature_index(adata_like, gene)
                idx_i = int(idx)
                dup = idx_i in used_idx
                if not dup:
                    used_idx.add(idx_i)
                st = GeneStatus(
                    gene=gene,
                    group=group,
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
                    group=group,
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
                    "group": st.group,
                    "present": st.present,
                    "status": st.status,
                    "resolved_gene": st.resolved_gene,
                    "gene_idx": st.gene_idx if st.gene_idx is not None else "",
                    "resolution_source": st.resolution_source,
                    "symbol_column": st.symbol_column,
                }
            )
            stats.append(st)

    return stats, pd.DataFrame(rows)


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


def _circular_distance_deg(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))


def _circular_stats_deg(phi_deg: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(phi_deg, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    rad = np.deg2rad(arr)
    z = np.exp(1j * rad)
    mean_vec = np.mean(z)
    mu = float(np.mod(np.angle(mean_vec), 2.0 * np.pi))
    R = float(np.abs(mean_vec))
    circ_sd = float(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12)))))
    return float(np.rad2deg(mu)), R, circ_sd


def _classify(q: float, peaks_k: float, underpowered: bool) -> str:
    if underpowered:
        return "Underpowered"
    if np.isfinite(float(q)) and float(q) <= Q_SIG:
        if int(peaks_k) >= 2:
            return "Localized–multimodal"
        return "Localized–unimodal"
    return "Not-localized"


def _top_q_mask(expr: np.ndarray, q: float) -> np.ndarray:
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


def _score_condition_from_arrays(
    *,
    expr: np.ndarray,
    coords: np.ndarray,
    fg_mode: str,
    q_top: float,
    n_bins: int,
    n_perm: int,
    seed: int,
    keep_profiles: bool,
) -> dict[str, Any]:
    x = np.asarray(expr, dtype=float).ravel()
    xy = np.asarray(coords, dtype=float)
    if fg_mode == "F0_detect":
        fg = x > 0.0
        q_val = np.nan
    elif fg_mode == "F1_topq":
        fg = _top_q_mask(x, q=float(q_top))
        q_val = float(q_top)
    else:
        raise ValueError(f"Unknown fg_mode: {fg_mode}")

    n_cells = int(fg.size)
    n_fg = int(fg.sum())
    prev = float(n_fg / max(1, n_cells))
    underpowered = bool(prev < UNDERPOWERED_PREV or n_fg < UNDERPOWERED_MIN_FG)

    center = compute_vantage_point(xy, method="mean")
    theta = compute_theta(xy, center)
    _, bin_id = bin_theta(theta, bins=int(n_bins))
    bin_counts = np.bincount(bin_id, minlength=int(n_bins)).astype(float)

    if n_fg == 0 or n_fg == n_cells:
        return {
            "q_param": q_val,
            "prev": prev,
            "n_fg": n_fg,
            "n_cells": n_cells,
            "T_obs": 0.0,
            "p_T": np.nan,
            "Z_T": np.nan,
            "coverage_C": np.nan,
            "peaks_K": np.nan,
            "phi_hat_deg": np.nan,
            "underpowered_flag": True,
            "foreground": fg if keep_profiles else None,
            "E_obs": np.zeros(int(n_bins), dtype=float) if keep_profiles else None,
            "null_E": None,
            "null_T": None,
        }

    e_obs, _, _, _ = compute_rsp_profile_from_boolean(
        fg,
        theta,
        int(n_bins),
        bin_id=bin_id,
        bin_counts_total=bin_counts,
    )
    t_obs = float(np.max(np.abs(e_obs)))
    phi_idx = int(np.argmax(np.abs(e_obs)))
    phi_hat = float(np.degrees(theta_bin_centers(int(n_bins))[phi_idx]) % 360.0)

    if underpowered:
        return {
            "q_param": q_val,
            "prev": prev,
            "n_fg": n_fg,
            "n_cells": n_cells,
            "T_obs": t_obs,
            "p_T": np.nan,
            "Z_T": np.nan,
            "coverage_C": np.nan,
            "peaks_K": np.nan,
            "phi_hat_deg": phi_hat,
            "underpowered_flag": True,
            "foreground": fg if keep_profiles else None,
            "E_obs": e_obs if keep_profiles else None,
            "null_E": None,
            "null_T": None,
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
        bin_counts_total=bin_counts,
    )

    null_e = np.asarray(perm["null_E_phi"], dtype=float)
    null_t = np.asarray(perm["null_T"], dtype=float)

    return {
        "q_param": q_val,
        "prev": prev,
        "n_fg": n_fg,
        "n_cells": n_cells,
        "T_obs": float(perm["T_obs"]),
        "p_T": float(perm["p_T"]),
        "Z_T": float(robust_z(float(perm["T_obs"]), null_t)),
        "coverage_C": float(coverage_from_null(e_obs, null_e, q=0.95)),
        "peaks_K": float(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95)),
        "phi_hat_deg": phi_hat,
        "underpowered_flag": False,
        "foreground": fg if keep_profiles else None,
        "E_obs": e_obs if keep_profiles else None,
        "null_E": null_e if keep_profiles else None,
        "null_T": null_t if keep_profiles else None,
    }


def _make_split_halves(
    n_cells: int,
    seed: int,
    strat_values: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    rng = np.random.default_rng(int(seed))

    if strat_values is None:
        idx = np.arange(n_cells, dtype=int)
        perm = rng.permutation(idx)
        cut = n_cells // 2
        h1 = np.sort(perm[:cut])
        h2 = np.sort(perm[cut:])
        assign = np.zeros(n_cells, dtype=int)
        assign[h1] = 1
        assign[h2] = 2
        return h1, h2, "random_50_50", assign

    ranks = pd.Series(np.asarray(strat_values, dtype=float)).rank(method="first")
    bins = pd.qcut(ranks, q=10, labels=False, duplicates="drop")
    bins_arr = np.asarray(bins, dtype=int)

    half1 = []
    half2 = []
    for b in np.unique(bins_arr):
        idx_b = np.flatnonzero(bins_arr == b)
        perm = rng.permutation(idx_b)
        cut = len(perm) // 2
        half1.extend(perm[:cut].tolist())
        half2.extend(perm[cut:].tolist())

    h1 = np.sort(np.asarray(half1, dtype=int))
    h2 = np.sort(np.asarray(half2, dtype=int))
    assign = np.zeros(n_cells, dtype=int)
    assign[h1] = 1
    assign[h2] = 2
    return h1, h2, "stratified_total_counts_decile", assign


def _score_sample_set(
    *,
    sample_name: str,
    sample_idx: np.ndarray,
    embeddings: list[EmbeddingSpec],
    genes_present: list[GeneStatus],
    expr_by_gene: dict[str, np.ndarray],
    q_top: float,
    n_bins: int,
    n_perm: int,
    seed_base: int,
    keep_profiles: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    detail: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    fg_modes = ["F0_detect", "F1_topq"]

    for emb_i, emb in enumerate(embeddings):
        coords_sub = emb.coords[sample_idx]
        for fg_i, fg_mode in enumerate(fg_modes):
            block_rows = []
            for g_i, st in enumerate(genes_present):
                expr_sub = expr_by_gene[st.gene][sample_idx]
                score = _score_condition_from_arrays(
                    expr=expr_sub,
                    coords=coords_sub,
                    fg_mode=fg_mode,
                    q_top=float(q_top),
                    n_bins=int(n_bins),
                    n_perm=int(n_perm),
                    seed=int(seed_base + emb_i * 10000 + fg_i * 1000 + g_i * 23 + 17),
                    keep_profiles=bool(keep_profiles),
                )
                row = {
                    "sample": sample_name,
                    "gene": st.gene,
                    "embedding": emb.key,
                    "foreground_mode": fg_mode,
                    "q": score["q_param"],
                    "prev": score["prev"],
                    "n_fg": score["n_fg"],
                    "n_cells": score["n_cells"],
                    "T_obs": score["T_obs"],
                    "p_T": score["p_T"],
                    "Z_T": score["Z_T"],
                    "coverage_C": score["coverage_C"],
                    "peaks_K": score["peaks_K"],
                    "phi_hat_deg": score["phi_hat_deg"],
                    "underpowered_flag": score["underpowered_flag"],
                }
                block_rows.append(row)

                if keep_profiles:
                    detail[(sample_name, st.gene, emb.key, fg_mode)] = {
                        "E_obs": score["E_obs"],
                        "null_E": score["null_E"],
                        "null_T": score["null_T"],
                        "fg": score["foreground"],
                        "coords": coords_sub,
                    }

            block_df = pd.DataFrame(block_rows)
            if not block_df.empty:
                p = block_df["p_T"].to_numpy(dtype=float)
                finite = np.isfinite(p)
                qvals = np.full(p.shape, np.nan, dtype=float)
                if int(finite.sum()) > 0:
                    qvals[finite] = bh_fdr(p[finite])
                block_df["q_T"] = qvals
                block_df["class_label"] = [
                    _classify(float(q), float(k), bool(u))
                    for q, k, u in zip(block_df["q_T"], block_df["peaks_K"], block_df["underpowered_flag"], strict=False)
                ]
                rows.extend(block_df.to_dict(orient="records"))

    return rows, detail


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    m = np.isfinite(xx) & np.isfinite(yy)
    if int(m.sum()) < 3:
        return np.nan
    xs = xx[m]
    ys = yy[m]
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    m = np.isfinite(xx) & np.isfinite(yy)
    if int(m.sum()) < 3:
        return np.nan
    xs = xx[m]
    ys = yy[m]
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return np.nan
    rho = spearmanr(xs, ys, nan_policy="omit").correlation
    return float(rho) if rho is not None else np.nan


def _summarize_split_half(split_df: pd.DataFrame) -> pd.DataFrame:
    if split_df.empty:
        return pd.DataFrame()

    h1 = split_df.loc[split_df["sample"] == "half1"].copy()
    h2 = split_df.loc[split_df["sample"] == "half2"].copy()

    merged = h1.merge(
        h2,
        on=["gene", "embedding", "foreground_mode"],
        suffixes=("_half1", "_half2"),
        how="outer",
    )

    rows = []
    for (emb, fg), sub in merged.groupby(["embedding", "foreground_mode"], sort=False):
        pear = _pearson(sub["Z_T_half1"].to_numpy(dtype=float), sub["Z_T_half2"].to_numpy(dtype=float))
        spear = _spearman(sub["Z_T_half1"].to_numpy(dtype=float), sub["Z_T_half2"].to_numpy(dtype=float))

        for _, r in sub.iterrows():
            z1 = float(r["Z_T_half1"]) if np.isfinite(float(r["Z_T_half1"])) else np.nan
            z2 = float(r["Z_T_half2"]) if np.isfinite(float(r["Z_T_half2"])) else np.nan
            p1 = float(r["phi_hat_deg_half1"]) if np.isfinite(float(r["phi_hat_deg_half1"])) else np.nan
            p2 = float(r["phi_hat_deg_half2"]) if np.isfinite(float(r["phi_hat_deg_half2"])) else np.nan
            dZ = float(abs(z1 - z2)) if np.isfinite(z1) and np.isfinite(z2) else np.nan
            dPhi = _circular_distance_deg(p1, p2) if np.isfinite(p1) and np.isfinite(p2) else np.nan
            if np.isfinite(p1) and np.isfinite(p2):
                _, r_pair, circ_sd_pair = _circular_stats_deg(np.asarray([p1, p2], dtype=float))
            else:
                r_pair, circ_sd_pair = np.nan, np.nan
            both_sig = bool(
                np.isfinite(float(r["q_T_half1"]))
                and np.isfinite(float(r["q_T_half2"]))
                and float(r["q_T_half1"]) <= Q_SIG
                and float(r["q_T_half2"]) <= Q_SIG
            )

            rows.append(
                {
                    "gene": str(r["gene"]),
                    "embedding": str(emb),
                    "foreground_mode": str(fg),
                    "Z_T_half1": z1,
                    "Z_T_half2": z2,
                    "q_T_half1": float(r["q_T_half1"]) if np.isfinite(float(r["q_T_half1"])) else np.nan,
                    "q_T_half2": float(r["q_T_half2"]) if np.isfinite(float(r["q_T_half2"])) else np.nan,
                    "phi_half1_deg": p1,
                    "phi_half2_deg": p2,
                    "delta_Z": dZ,
                    "delta_phi_deg": dPhi,
                    "R_half_pair": r_pair,
                    "circ_sd_half_pair": circ_sd_pair,
                    "half_consistent_significant": both_sig,
                    "pearson_Z_global": pear,
                    "spearman_Z_global": spear,
                }
            )

    out = pd.DataFrame(rows)
    return out


def _init_boot_acc(keys: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], dict[str, list[float]]]:
    acc = {}
    for k in keys:
        acc[k] = {
            "Z": [],
            "coverage": [],
            "peaks": [],
            "phi": [],
            "q": [],
            "underpowered": [],
        }
    return acc


def _boot_progress_table(acc: dict[tuple[str, str, str], dict[str, list[float]]], reps_done: int) -> pd.DataFrame:
    rows = []
    for key, d in acc.items():
        gene, emb, fg = key
        z = np.asarray(d["Z"], dtype=float)
        q = np.asarray(d["q"], dtype=float)
        qv = q[np.isfinite(q)]
        frac_sig = float(np.mean(qv <= Q_SIG)) if qv.size > 0 else np.nan
        zmed = float(np.nanmedian(z)) if np.isfinite(z).sum() > 0 else np.nan
        rows.append(
            {
                "gene": gene,
                "embedding": emb,
                "foreground_mode": fg,
                "reps_done": int(reps_done),
                "z_median_so_far": zmed,
                "frac_sig_so_far": frac_sig,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_repro(
    *,
    embeddings: list[EmbeddingSpec],
    genes_present: list[GeneStatus],
    expr_by_gene: dict[str, np.ndarray],
    n_cells: int,
    q_top: float,
    n_bins: int,
    n_perm_boot: int,
    boot_reps: int,
    boot_frac: float,
    seed: int,
    save_boot_long: bool,
    out_tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[tuple[str, str, str], dict[str, list[float]]]]:
    fg_modes = ["F0_detect", "F1_topq"]
    keys = [(st.gene, emb.key, fg) for st in genes_present for emb in embeddings for fg in fg_modes]
    acc = _init_boot_acc(keys)

    rng = np.random.default_rng(int(seed))
    m = int(max(10, min(n_cells, np.floor(float(boot_frac) * n_cells))))
    boot_rows: list[dict[str, Any]] = []

    for b in range(int(boot_reps)):
        idx = rng.integers(0, n_cells, size=m, endpoint=False)

        for emb_i, emb in enumerate(embeddings):
            coords_sub = emb.coords[idx]
            for fg_i, fg in enumerate(fg_modes):
                block_rows = []
                for g_i, st in enumerate(genes_present):
                    expr_sub = expr_by_gene[st.gene][idx]
                    score = _score_condition_from_arrays(
                        expr=expr_sub,
                        coords=coords_sub,
                        fg_mode=fg,
                        q_top=float(q_top),
                        n_bins=int(n_bins),
                        n_perm=int(n_perm_boot),
                        seed=int(seed + 1000000 + b * 10000 + emb_i * 1000 + fg_i * 100 + g_i * 13),
                        keep_profiles=False,
                    )
                    block_rows.append(
                        {
                            "bootstrap_id": int(b),
                            "gene": st.gene,
                            "embedding": emb.key,
                            "foreground_mode": fg,
                            "q": score["q_param"],
                            "prev": score["prev"],
                            "n_fg": score["n_fg"],
                            "n_cells": score["n_cells"],
                            "T_obs": score["T_obs"],
                            "p_T": score["p_T"],
                            "Z_T": score["Z_T"],
                            "coverage_C": score["coverage_C"],
                            "peaks_K": score["peaks_K"],
                            "phi_hat_deg": score["phi_hat_deg"],
                            "underpowered_flag": score["underpowered_flag"],
                        }
                    )

                block = pd.DataFrame(block_rows)
                p = block["p_T"].to_numpy(dtype=float)
                finite = np.isfinite(p)
                qvals = np.full(p.shape, np.nan, dtype=float)
                if int(finite.sum()) > 0:
                    qvals[finite] = bh_fdr(p[finite])
                block["q_T"] = qvals

                for _, r in block.iterrows():
                    key = (str(r["gene"]), str(r["embedding"]), str(r["foreground_mode"]))
                    acc[key]["Z"].append(float(r["Z_T"]) if np.isfinite(float(r["Z_T"])) else np.nan)
                    acc[key]["coverage"].append(
                        float(r["coverage_C"]) if np.isfinite(float(r["coverage_C"])) else np.nan
                    )
                    acc[key]["peaks"].append(float(r["peaks_K"]) if np.isfinite(float(r["peaks_K"])) else np.nan)
                    acc[key]["phi"].append(
                        float(r["phi_hat_deg"]) if np.isfinite(float(r["phi_hat_deg"])) else np.nan
                    )
                    acc[key]["q"].append(float(r["q_T"]) if np.isfinite(float(r["q_T"])) else np.nan)
                    acc[key]["underpowered"].append(bool(r["underpowered_flag"]))

                if bool(save_boot_long):
                    boot_rows.extend(block.to_dict(orient="records"))

        if (b + 1) % 20 == 0 or (b + 1) == int(boot_reps):
            print(f"[Bootstrap] completed {b + 1}/{int(boot_reps)} replicates")
            prog = _boot_progress_table(acc, reps_done=b + 1)
            prog.to_csv(out_tables_dir / "bootstrap_summary.intermediate.csv", index=False)
            if bool(save_boot_long):
                pd.DataFrame(boot_rows).to_csv(
                    out_tables_dir / "bootstrap_scores_long.intermediate.csv", index=False
                )

    # Summarize.
    summary_rows = []
    for key in keys:
        gene, emb, fg = key
        d = acc[key]

        z = np.asarray(d["Z"], dtype=float)
        cov = np.asarray(d["coverage"], dtype=float)
        pk = np.asarray(d["peaks"], dtype=float)
        phi = np.asarray(d["phi"], dtype=float)
        q = np.asarray(d["q"], dtype=float)
        under = np.asarray(d["underpowered"], dtype=bool)

        qv = q[np.isfinite(q)]
        n_valid = int(qv.size)
        frac_sig = float(np.mean(qv <= Q_SIG)) if n_valid > 0 else np.nan
        median_q = float(np.median(qv)) if n_valid > 0 else np.nan

        zv = z[np.isfinite(z)]
        if zv.size > 0:
            z_med = float(np.median(zv))
            z_lo = float(np.quantile(zv, 0.025))
            z_hi = float(np.quantile(zv, 0.975))
        else:
            z_med = np.nan
            z_lo = np.nan
            z_hi = np.nan

        cv = cov[np.isfinite(cov)]
        if cv.size > 0:
            c_med = float(np.median(cv))
            c_lo = float(np.quantile(cv, 0.025))
            c_hi = float(np.quantile(cv, 0.975))
        else:
            c_med = np.nan
            c_lo = np.nan
            c_hi = np.nan

        pkv = pk[np.isfinite(pk)]
        if pkv.size > 0:
            pki = np.rint(pkv).astype(int)
            unique, counts = np.unique(pki, return_counts=True)
            mode_k = int(unique[int(np.argmax(counts))])
            frac_k2 = float(np.mean(pki >= 2))
        else:
            mode_k = -1
            frac_k2 = np.nan

        sig_mask = np.isfinite(q) & (q <= Q_SIG) & np.isfinite(phi)
        phi_sig = phi[sig_mask]
        if phi_sig.size > 0:
            phi_mean, R, circ_sd = _circular_stats_deg(phi_sig)
        else:
            phi_mean, R, circ_sd = np.nan, np.nan, np.nan

        cond_sig = bool((np.isfinite(median_q) and median_q <= Q_SIG) or (np.isfinite(frac_sig) and frac_sig >= 0.60))
        bootstrap_repro = bool(
            cond_sig
            and np.isfinite(z_lo)
            and (z_lo > 0.0)
            and np.isfinite(R)
            and (float(R) >= 0.60)
            and np.isfinite(frac_sig)
            and (float(frac_sig) >= 0.60)
        )

        summary_rows.append(
            {
                "gene": gene,
                "embedding": emb,
                "foreground_mode": fg,
                "boot_reps": int(boot_reps),
                "n_valid": n_valid,
                "fraction_significant": frac_sig,
                "median_q": median_q,
                "Z_median": z_med,
                "Z_ci_low": z_lo,
                "Z_ci_high": z_hi,
                "coverage_median": c_med,
                "coverage_ci_low": c_lo,
                "coverage_ci_high": c_hi,
                "peaks_mode": mode_k,
                "peaks_frac_ge2": frac_k2,
                "phi_mean_deg": phi_mean,
                "R": R,
                "circ_sd": circ_sd,
                "bootstrap_reproducible": bootstrap_repro,
                "underpowered_fraction": float(np.mean(under)) if under.size > 0 else np.nan,
            }
        )

    boot_summary = pd.DataFrame(summary_rows)

    if bool(save_boot_long):
        boot_long_df = pd.DataFrame(boot_rows)
    else:
        boot_long_df = None

    return boot_summary, boot_long_df, acc


def _plot_overview(
    out_dir: Path,
    umap_coords: np.ndarray,
    expr_by_gene: dict[str, np.ndarray],
    total_counts: np.ndarray | None,
    split_assign: np.ndarray,
    split_method: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for gene in ["TNNT2", "NPPA"]:
        expr = expr_by_gene.get(gene, None)
        if expr is None:
            _save_placeholder(out_dir / f"umap_{gene}.png", f"UMAP {gene}", f"{gene} missing")
            continue
        save_numeric_umap(
            umap_coords,
            np.log1p(np.maximum(np.asarray(expr, dtype=float), 0.0)),
            out_dir / f"umap_{gene}.png",
            title=f"umap_repr colored by {gene}",
            cmap="Reds",
            colorbar_label=f"log1p({gene})",
        )

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    if total_counts is None:
        ax.axis("off")
        ax.set_title("Split strategy histogram")
        ax.text(0.5, 0.5, f"total_counts unavailable\nmethod={split_method}", ha="center", va="center")
    else:
        h1 = total_counts[split_assign == 1]
        h2 = total_counts[split_assign == 2]
        bins = int(min(60, max(20, np.ceil(np.sqrt(total_counts.size)))))
        ax.hist(
            np.log1p(np.maximum(h1, 0.0)),
            bins=bins,
            alpha=0.6,
            color="#1f77b4",
            label="half1",
            density=True,
        )
        ax.hist(
            np.log1p(np.maximum(h2, 0.0)),
            bins=bins,
            alpha=0.6,
            color="#ff7f0e",
            label="half2",
            density=True,
        )
        ax.set_title(f"Split strategy ({split_method}): log1p(total_counts)")
        ax.set_xlabel("log1p(total_counts)")
        ax.set_ylabel("density")
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "split_strategy_hist_total_counts.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)


def _plot_split_half(
    out_dir: Path,
    split_scores: pd.DataFrame,
    split_summary: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if split_scores.empty or split_summary.empty:
        _save_placeholder(out_dir / "empty.png", "Split-half", "No split-half scores")
        return

    for (emb, fg), sub in split_summary.groupby(["embedding", "foreground_mode"], sort=False):
        # 1) Scatter Z1 vs Z2.
        fig1, ax1 = plt.subplots(figsize=(7.0, 6.0))
        x = sub["Z_T_half1"].to_numpy(dtype=float)
        y = sub["Z_T_half2"].to_numpy(dtype=float)
        ax1.scatter(x, y, s=90, c="#4c78a8", edgecolors="black", linewidths=0.4, alpha=0.88)
        for _, r in sub.iterrows():
            ax1.text(float(r["Z_T_half1"]), float(r["Z_T_half2"]) + 0.02, str(r["gene"]), fontsize=8)
        lo = np.nanmin(np.concatenate([x, y])) if np.isfinite(np.concatenate([x, y])).any() else 0.0
        hi = np.nanmax(np.concatenate([x, y])) if np.isfinite(np.concatenate([x, y])).any() else 1.0
        ax1.plot([lo, hi], [lo, hi], linestyle="--", color="#444444")
        ax1.set_xlabel("Z_T half1")
        ax1.set_ylabel("Z_T half2")
        ax1.set_title(f"{emb} / {fg}: split-half Z concordance")
        fig1.tight_layout()
        fig1.savefig(out_dir / f"{emb}_{fg}_scatter_Z1_vs_Z2.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig1)

        # 2) Bland-Altman.
        fig2, ax2 = plt.subplots(figsize=(7.4, 6.0))
        mean_z = 0.5 * (sub["Z_T_half1"].to_numpy(dtype=float) + sub["Z_T_half2"].to_numpy(dtype=float))
        diff_z = sub["Z_T_half1"].to_numpy(dtype=float) - sub["Z_T_half2"].to_numpy(dtype=float)
        ax2.scatter(mean_z, diff_z, s=90, c="#f58518", edgecolors="black", linewidths=0.4, alpha=0.88)
        for _, r in sub.iterrows():
            mz = 0.5 * (float(r["Z_T_half1"]) + float(r["Z_T_half2"]))
            dz = float(r["Z_T_half1"]) - float(r["Z_T_half2"])
            ax2.text(mz, dz + 0.02, str(r["gene"]), fontsize=8)
        mu = float(np.nanmean(diff_z)) if np.isfinite(diff_z).any() else 0.0
        sd = float(np.nanstd(diff_z)) if np.isfinite(diff_z).any() else 0.0
        ax2.axhline(mu, color="#333333", linestyle="--")
        ax2.axhline(mu + 1.96 * sd, color="#999999", linestyle=":")
        ax2.axhline(mu - 1.96 * sd, color="#999999", linestyle=":")
        ax2.set_xlabel("mean(Z1,Z2)")
        ax2.set_ylabel("Z1-Z2")
        ax2.set_title(f"{emb} / {fg}: Bland–Altman")
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{emb}_{fg}_bland_altman.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig2)

        # 3) Side-by-side bars Z1/Z2 with significance markers.
        fig3, ax3 = plt.subplots(figsize=(max(8.0, 0.8 * len(sub)), 5.8))
        sub2 = sub.sort_values(by="gene")
        genes = sub2["gene"].astype(str).tolist()
        x = np.arange(len(genes))
        w = 0.38
        z1 = sub2["Z_T_half1"].to_numpy(dtype=float)
        z2 = sub2["Z_T_half2"].to_numpy(dtype=float)
        b1 = ax3.bar(x - w / 2, z1, width=w, color="#1f77b4", label="half1")
        b2 = ax3.bar(x + w / 2, z2, width=w, color="#ff7f0e", label="half2")
        sig = sub2["half_consistent_significant"].to_numpy(dtype=bool)
        for i, s in enumerate(sig.tolist()):
            if not s:
                continue
            ymax = np.nanmax([z1[i], z2[i]]) if np.isfinite([z1[i], z2[i]]).any() else 0.0
            ax3.text(i, ymax + 0.15, "*", ha="center", va="bottom", fontsize=14, color="#d62728")
        ax3.set_xticks(x)
        ax3.set_xticklabels(genes, rotation=45, ha="right")
        ax3.set_ylabel("Z_T")
        ax3.set_title(f"{emb} / {fg}: split-half Z per gene (* both halves q<=0.05)")
        ax3.legend(loc="best")
        fig3.tight_layout()
        fig3.savefig(out_dir / f"{emb}_{fg}_bars_Z_half1_half2.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig3)


def _plot_bootstrap_distributions(
    out_dir: Path,
    boot_summary: pd.DataFrame,
    boot_long_df: pd.DataFrame | None,
    acc: dict[tuple[str, str, str], dict[str, list[float]]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if boot_summary.empty:
        _save_placeholder(out_dir / "empty.png", "Bootstrap", "No bootstrap summary")
        return

    # Build plot-long Z table from accumulator for violin.
    rows = []
    for key, d in acc.items():
        gene, emb, fg = key
        z = np.asarray(d["Z"], dtype=float)
        for i, v in enumerate(z.tolist()):
            rows.append(
                {
                    "bootstrap_id": i,
                    "gene": gene,
                    "embedding": emb,
                    "foreground_mode": fg,
                    "Z_T": float(v) if np.isfinite(float(v)) else np.nan,
                }
            )
    boot_plot_df = pd.DataFrame(rows)

    for (emb, fg), sub_sum in boot_summary.groupby(["embedding", "foreground_mode"], sort=False):
        sub_z = boot_plot_df.loc[(boot_plot_df["embedding"] == emb) & (boot_plot_df["foreground_mode"] == fg)]
        genes_sorted = (
            sub_sum.sort_values(by="Z_median", ascending=False)["gene"].astype(str).tolist()
        )

        # 1) Box/violin-ish (boxplot for robustness).
        fig1, ax1 = plt.subplots(figsize=(max(8.0, 0.85 * len(genes_sorted)), 5.8))
        data = [
            sub_z.loc[sub_z["gene"] == g, "Z_T"].dropna().to_numpy(dtype=float)
            for g in genes_sorted
        ]
        ax1.boxplot(data, tick_labels=genes_sorted, patch_artist=True)
        ax1.tick_params(axis="x", rotation=45)
        ax1.set_ylabel("Bootstrap Z_T")
        ax1.set_title(f"{emb} / {fg}: bootstrap Z_T distributions")
        fig1.tight_layout()
        fig1.savefig(out_dir / f"{emb}_{fg}_bootstrap_Z_boxplot.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig1)

        # 2) Median + CI errorbars.
        fig2, ax2 = plt.subplots(figsize=(max(8.0, 0.85 * len(genes_sorted)), 5.8))
        s2 = sub_sum.set_index("gene").reindex(genes_sorted)
        x = np.arange(len(genes_sorted))
        med = s2["Z_median"].to_numpy(dtype=float)
        lo = s2["Z_ci_low"].to_numpy(dtype=float)
        hi = s2["Z_ci_high"].to_numpy(dtype=float)
        ax2.errorbar(
            x,
            med,
            yerr=[med - lo, hi - med],
            fmt="o",
            color="#4c78a8",
            ecolor="#333333",
            capsize=3,
            linewidth=1.0,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(genes_sorted, rotation=45, ha="right")
        ax2.set_ylabel("Z_T median ± 95% CI")
        ax2.set_title(f"{emb} / {fg}: bootstrap median Z_T with CI")
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{emb}_{fg}_bootstrap_Z_CI.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig2)

        # 3) Histogram of fraction significant across genes.
        fig3, ax3 = plt.subplots(figsize=(7.2, 5.2))
        vals = sub_sum["fraction_significant"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        bins = np.linspace(0.0, 1.0, 11)
        ax3.hist(vals, bins=bins, color="#f58518", alpha=0.85, edgecolor="white")
        ax3.axvline(0.60, color="#333333", linestyle="--", linewidth=1.0)
        ax3.set_xlabel("fraction_significant")
        ax3.set_ylabel("# genes")
        ax3.set_title(f"{emb} / {fg}: fraction_significant distribution")
        fig3.tight_layout()
        fig3.savefig(out_dir / f"{emb}_{fg}_fraction_significant_hist.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig3)


def _plot_repro_maps(out_dir: Path, boot_summary: pd.DataFrame, calls_df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if boot_summary.empty or calls_df.empty:
        _save_placeholder(out_dir / "empty.png", "Reproducibility maps", "No summaries")
        return

    boot_summary = boot_summary.copy()
    boot_summary["emb_fg"] = boot_summary["embedding"].astype(str) + "|" + boot_summary["foreground_mode"].astype(str)
    calls_df = calls_df.copy()
    calls_df["emb_fg"] = calls_df["embedding"].astype(str) + "|" + calls_df["foreground_mode"].astype(str)

    genes = sorted(boot_summary["gene"].astype(str).unique().tolist())
    combos = sorted(boot_summary["emb_fg"].astype(str).unique().tolist())

    p_frac = boot_summary.pivot(index="gene", columns="emb_fg", values="fraction_significant").reindex(index=genes, columns=combos)
    p_low = boot_summary.pivot(index="gene", columns="emb_fg", values="Z_ci_low").reindex(index=genes, columns=combos)
    p_flag = calls_df.pivot(index="gene", columns="emb_fg", values="reproducible_localized").reindex(index=genes, columns=combos)

    mat_frac = np.nan_to_num(p_frac.to_numpy(dtype=float), nan=0.0)
    mat_low = np.nan_to_num(p_low.to_numpy(dtype=float), nan=0.0)
    mat_flag = np.nan_to_num(p_flag.to_numpy(dtype=float), nan=0.0)

    fig1, ax1 = plt.subplots(figsize=(1.1 * len(combos) + 3.0, 0.6 * len(genes) + 2.0))
    im1 = ax1.imshow(mat_frac, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax1.set_xticks(np.arange(len(combos)))
    ax1.set_xticklabels(combos, rotation=45, ha="right", fontsize=8)
    ax1.set_yticks(np.arange(len(genes)))
    ax1.set_yticklabels(genes, fontsize=8)
    ax1.set_title("fraction_significant")
    fig1.colorbar(im1, ax=ax1)
    fig1.tight_layout()
    fig1.savefig(out_dir / "heatmap_fraction_significant.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(1.1 * len(combos) + 3.0, 0.6 * len(genes) + 2.0))
    im2 = ax2.imshow(mat_low, aspect="auto", cmap="magma")
    ax2.set_xticks(np.arange(len(combos)))
    ax2.set_xticklabels(combos, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(np.arange(len(genes)))
    ax2.set_yticklabels(genes, fontsize=8)
    ax2.set_title("Z_ci_low")
    fig2.colorbar(im2, ax=ax2)
    fig2.tight_layout()
    fig2.savefig(out_dir / "heatmap_Z_ci_low.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(1.1 * len(combos) + 3.0, 0.6 * len(genes) + 2.0))
    im3 = ax3.imshow(mat_flag, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax3.set_xticks(np.arange(len(combos)))
    ax3.set_xticklabels(combos, rotation=45, ha="right", fontsize=8)
    ax3.set_yticks(np.arange(len(genes)))
    ax3.set_yticklabels(genes, fontsize=8)
    ax3.set_title("reproducible_localized (0/1)")
    fig3.colorbar(im3, ax=ax3)
    fig3.tight_layout()
    fig3.savefig(out_dir / "heatmap_reproducible_flag.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig3)


def _plot_direction_stability(
    out_dir: Path,
    boot_summary: pd.DataFrame,
    acc: dict[tuple[str, str, str], dict[str, list[float]]],
    split_summary: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if boot_summary.empty:
        _save_placeholder(out_dir / "empty.png", "Direction stability", "No bootstrap summary")
        return

    # Circular plots per embedding/foreground.
    for (emb, fg), sub in boot_summary.groupby(["embedding", "foreground_mode"], sort=False):
        genes = sub["gene"].astype(str).tolist()
        n_cols = 4
        n_rows = int(np.ceil(len(genes) / n_cols)) if len(genes) > 0 else 1
        fig = plt.figure(figsize=(4.0 * n_cols, 3.6 * n_rows))

        for i, gene in enumerate(genes):
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="polar")
            key = (gene, str(emb), str(fg))
            d = acc.get(key, None)
            if d is None:
                ax.axis("off")
                continue
            phi = np.asarray(d["phi"], dtype=float)
            q = np.asarray(d["q"], dtype=float)
            mask = np.isfinite(phi) & np.isfinite(q) & (q <= Q_SIG)
            ph = phi[mask]
            if ph.size == 0:
                ax.text(0.5, 0.5, "no sig", transform=ax.transAxes, ha="center", va="center", fontsize=8)
            else:
                rad = np.deg2rad(ph)
                ax.scatter(rad, np.ones_like(rad), s=24, c="#1f77b4", alpha=0.85)
                mu, R, csd = _circular_stats_deg(ph)
                ax.plot([np.deg2rad(mu), np.deg2rad(mu)], [0.0, 1.1], color="#d62728", linewidth=2.0)
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
            ax.set_title(gene, fontsize=8)

        fig.suptitle(f"{emb} / {fg}: bootstrap phi (significant reps)", y=0.995)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
        fig.savefig(out_dir / f"{emb}_{fg}_bootstrap_phi_grid.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig)

        # R vs fraction significant.
        fig2, ax2 = plt.subplots(figsize=(7.2, 5.8))
        ax2.scatter(
            sub["R"].to_numpy(dtype=float),
            sub["fraction_significant"].to_numpy(dtype=float),
            s=95,
            c="#4c78a8",
            edgecolors="black",
            linewidths=0.4,
            alpha=0.9,
        )
        for _, r in sub.iterrows():
            ax2.text(float(r["R"]), float(r["fraction_significant"]) + 0.01, str(r["gene"]), fontsize=8)
        ax2.axvline(0.60, color="#444", linestyle="--")
        ax2.axhline(0.60, color="#444", linestyle=":")
        ax2.set_xlabel("R")
        ax2.set_ylabel("fraction_significant")
        ax2.set_title(f"{emb} / {fg}: R vs fraction_significant")
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{emb}_{fg}_R_vs_fraction_significant.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig2)

        # Split-half phi comparison on circle + arc.
        sub_split = split_summary.loc[
            (split_summary["embedding"] == emb) & (split_summary["foreground_mode"] == fg)
        ]
        fig3 = plt.figure(figsize=(7.6, 7.0))
        ax3 = fig3.add_subplot(111, projection="polar")
        for _, r in sub_split.iterrows():
            p1 = float(r["phi_half1_deg"]) if np.isfinite(float(r["phi_half1_deg"])) else np.nan
            p2 = float(r["phi_half2_deg"]) if np.isfinite(float(r["phi_half2_deg"])) else np.nan
            if not (np.isfinite(p1) and np.isfinite(p2)):
                continue
            a1 = np.deg2rad(p1)
            a2 = np.deg2rad(p2)
            ax3.scatter([a1], [1.0], c="#1f77b4", s=40)
            ax3.scatter([a2], [1.1], c="#ff7f0e", s=40)
            ax3.plot([a1, a2], [1.0, 1.1], color="#555", linewidth=1.0)
            mid = (a1 + a2) / 2.0
            ax3.text(mid, 1.16, str(r["gene"]), fontsize=8, ha="center", va="center")
        ax3.set_theta_zero_location("E")
        ax3.set_theta_direction(1)
        ax3.set_rticks([])
        ax3.set_title(f"{emb} / {fg}: split-half phi1/phi2 with arc")
        fig3.tight_layout()
        fig3.savefig(out_dir / f"{emb}_{fg}_split_half_phi_arc.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig3)


def _plot_qc_controls(out_dir: Path, calls_df: pd.DataFrame, qc_audit_df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if calls_df.empty or qc_audit_df.empty:
        _save_placeholder(out_dir / "empty.png", "QC controls", "No data")
        return

    # 1) qc_risk vs bootstrap median Z.
    fig1, ax1 = plt.subplots(figsize=(7.6, 5.8))
    x = calls_df["qc_risk"].to_numpy(dtype=float)
    y = calls_df["boot_Z_median"].to_numpy(dtype=float)
    flag = calls_df["reproducible_localized"].to_numpy(dtype=bool)
    base_mask = ~flag
    if int(np.sum(base_mask)) > 0:
        ax1.scatter(
            x[base_mask],
            y[base_mask],
            s=75,
            c="#4c78a8",
            marker="o",
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
            label="non-reproducible",
        )
    if int(np.sum(flag)) > 0:
        ax1.scatter(
            x[flag],
            y[flag],
            s=130,
            c="#2ca02c",
            marker="*",
            alpha=0.90,
            edgecolors="black",
            linewidths=0.5,
            label="reproducible",
        )
    for _, r in calls_df.iterrows():
        ax1.text(float(r["qc_risk"]), float(r["boot_Z_median"]) + 0.01, f"{r['gene']}|{r['embedding']}|{r['foreground_mode']}", fontsize=7)
    ax1.axvline(QC_RISK_THRESH, color="#444", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("qc_risk")
    ax1.set_ylabel("bootstrap median Z_T")
    ax1.set_title("QC risk vs bootstrap median Z_T (stars=reproducible calls)")
    if int(np.sum(flag)) > 0:
        ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig(out_dir / "qc_risk_vs_bootstrap_medianZ.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    # 2) rho_mt vs rho_counts on full data QC audit.
    fig2, ax2 = plt.subplots(figsize=(7.2, 5.8))
    ax2.scatter(
        qc_audit_df["rho_total_counts"].to_numpy(dtype=float),
        qc_audit_df["rho_pct_mt"].to_numpy(dtype=float),
        s=90,
        c="#f58518",
        alpha=0.82,
        edgecolors="black",
        linewidths=0.4,
    )
    for _, r in qc_audit_df.iterrows():
        ax2.text(float(r["rho_total_counts"]), float(r["rho_pct_mt"]) + 0.01, f"{r['gene']}|{r['foreground_mode']}", fontsize=8)
    ax2.axvline(0.0, color="#666", linewidth=0.9)
    ax2.axhline(0.0, color="#666", linewidth=0.9)
    ax2.set_xlabel("rho_total_counts")
    ax2.set_ylabel("rho_pct_mt")
    ax2.set_title("Full-data QC audit: rho_mt vs rho_counts")
    fig2.tight_layout()
    fig2.savefig(out_dir / "rho_mt_vs_rho_counts.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)


def _plot_polar(ax: plt.Axes, detail: dict[str, Any] | None, title: str, stats_text: str = "") -> None:
    if detail is None or detail.get("E_obs") is None:
        ax.text(0.5, 0.5, "no profile", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        return

    e_obs = np.asarray(detail["E_obs"], dtype=float)
    n_bins = int(e_obs.size)
    centers = theta_bin_centers(n_bins)
    th = np.concatenate([centers, centers[:1]])
    obs = np.concatenate([e_obs, e_obs[:1]])
    ax.plot(th, obs, color="#8B0000", linewidth=2.0)

    null_e = detail.get("null_E", None)
    if null_e is not None:
        ne = np.asarray(null_e, dtype=float)
        hi = np.quantile(ne, 0.95, axis=0)
        lo = np.quantile(ne, 0.05, axis=0)
        ax.plot(th, np.concatenate([hi, hi[:1]]), color="#333", linestyle="--", linewidth=1.0)
        ax.plot(th, np.concatenate([lo, lo[:1]]), color="#333", linestyle="--", linewidth=0.9)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(title, fontsize=9)
    if stats_text:
        ax.text(
            0.02,
            0.02,
            stats_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#999", "alpha": 0.85},
        )


def _plot_exemplar_panels(
    out_dir: Path,
    calls_df: pd.DataFrame,
    split_scores_df: pd.DataFrame,
    boot_summary_df: pd.DataFrame,
    acc: dict[tuple[str, str, str], dict[str, list[float]]],
    detail_map: dict[tuple[str, str, str, str], dict[str, Any]],
    embeddings_map: dict[str, EmbeddingSpec],
    expr_by_gene: dict[str, np.ndarray],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if calls_df.empty:
        _save_placeholder(out_dir / "empty.png", "Exemplar panels", "No calls")
        return

    repro = calls_df.loc[calls_df["reproducible_localized"]].copy()
    unstable = calls_df.loc[(calls_df["full_localized"]) & (~calls_df["reproducible_localized"])].copy()

    sel = []
    if not repro.empty:
        sel.extend(repro.sort_values(by=["fraction_significant", "boot_Z_median"], ascending=[False, False]).head(2).to_dict(orient="records"))
    if not unstable.empty:
        sel.extend(unstable.sort_values(by=["full_Z_T", "fraction_significant"], ascending=[False, True]).head(2).to_dict(orient="records"))

    if len(sel) == 0:
        _save_placeholder(out_dir / "no_exemplars.png", "Exemplar panels", "No exemplar calls found")
        return

    used = set()
    final_sel = []
    for r in sel:
        key = (r["gene"], r["embedding"], r["foreground_mode"])
        if key in used:
            continue
        used.add(key)
        final_sel.append(r)

    for r in final_sel:
        gene = str(r["gene"])
        emb = str(r["embedding"])
        fg = str(r["foreground_mode"])

        expr = expr_by_gene.get(gene, None)
        if expr is None or emb not in embeddings_map:
            continue

        coords = embeddings_map[emb].coords
        x_plot = np.log1p(np.maximum(np.asarray(expr, dtype=float), 0.0))
        vmin = float(np.quantile(x_plot, 0.01)) if np.isfinite(x_plot).sum() > 0 else 0.0
        vmax = float(np.quantile(x_plot, 0.99)) if np.isfinite(x_plot).sum() > 0 else 1.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6

        d_full = detail_map.get(("full", gene, emb, fg), None)
        d_h1 = detail_map.get(("half1", gene, emb, fg), None)
        d_h2 = detail_map.get(("half2", gene, emb, fg), None)

        row_full = split_scores_df.loc[
            (split_scores_df["sample"] == "full")
            & (split_scores_df["gene"] == gene)
            & (split_scores_df["embedding"] == emb)
            & (split_scores_df["foreground_mode"] == fg)
        ]
        row_h1 = split_scores_df.loc[
            (split_scores_df["sample"] == "half1")
            & (split_scores_df["gene"] == gene)
            & (split_scores_df["embedding"] == emb)
            & (split_scores_df["foreground_mode"] == fg)
        ]
        row_h2 = split_scores_df.loc[
            (split_scores_df["sample"] == "half2")
            & (split_scores_df["gene"] == gene)
            & (split_scores_df["embedding"] == emb)
            & (split_scores_df["foreground_mode"] == fg)
        ]
        row_boot = boot_summary_df.loc[
            (boot_summary_df["gene"] == gene)
            & (boot_summary_df["embedding"] == emb)
            & (boot_summary_df["foreground_mode"] == fg)
        ]

        fig = plt.figure(figsize=(16.5, 9.0))
        gs = fig.add_gridspec(2, 3, wspace=0.28, hspace=0.30)

        # A) feature map
        axA = fig.add_subplot(gs[0, 0])
        ord_idx = np.argsort(x_plot, kind="mergesort")
        axA.scatter(coords[:, 0], coords[:, 1], c="#dddddd", s=5, alpha=0.25, linewidths=0, rasterized=True)
        sc = axA.scatter(
            coords[ord_idx, 0],
            coords[ord_idx, 1],
            c=x_plot[ord_idx],
            cmap="Reds",
            s=8,
            alpha=0.9,
            linewidths=0,
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
        )
        axA.set_xticks([])
        axA.set_yticks([])
        axA.set_title(f"{gene} feature map ({emb})")
        cb = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.03)
        cb.set_label("log1p(expr)")

        # B/C/D) Full, half1, half2 polars
        axB = fig.add_subplot(gs[0, 1], projection="polar")
        txt_full = ""
        if not row_full.empty:
            rr = row_full.iloc[0]
            txt_full = f"Z={float(rr['Z_T']):.2f}\nq={float(rr['q_T']):.2e}\nC={float(rr['coverage_C']):.3f}\nK={int(rr['peaks_K']) if np.isfinite(rr['peaks_K']) else -1}\nphi={float(rr['phi_hat_deg']):.1f}"
        _plot_polar(axB, d_full, "Full-data polar", txt_full)

        axC = fig.add_subplot(gs[0, 2], projection="polar")
        txt_h1 = ""
        if not row_h1.empty:
            rr = row_h1.iloc[0]
            txt_h1 = f"Z={float(rr['Z_T']):.2f}\nq={float(rr['q_T']):.2e}\nphi={float(rr['phi_hat_deg']):.1f}"
        _plot_polar(axC, d_h1, "Half1 polar", txt_h1)

        axD = fig.add_subplot(gs[1, 0], projection="polar")
        txt_h2 = ""
        if not row_h2.empty:
            rr = row_h2.iloc[0]
            txt_h2 = f"Z={float(rr['Z_T']):.2f}\nq={float(rr['q_T']):.2e}\nphi={float(rr['phi_hat_deg']):.1f}"
        _plot_polar(axD, d_h2, "Half2 polar", txt_h2)

        # E) bootstrap Z distribution + CI
        axE = fig.add_subplot(gs[1, 1])
        key = (gene, emb, fg)
        dacc = acc.get(key, None)
        if dacc is None:
            axE.axis("off")
        else:
            z = np.asarray(dacc["Z"], dtype=float)
            z = z[np.isfinite(z)]
            if z.size > 0:
                bins = int(min(40, max(10, np.ceil(np.sqrt(z.size)))))
                axE.hist(z, bins=bins, color="#7aa6d6", edgecolor="white", alpha=0.95)
                if not row_boot.empty:
                    rb = row_boot.iloc[0]
                    lo = float(rb["Z_ci_low"]) if np.isfinite(float(rb["Z_ci_low"])) else np.nan
                    hi = float(rb["Z_ci_high"]) if np.isfinite(float(rb["Z_ci_high"])) else np.nan
                    med = float(rb["Z_median"]) if np.isfinite(float(rb["Z_median"])) else np.nan
                    if np.isfinite(lo):
                        axE.axvline(lo, color="#d62728", linestyle=":", linewidth=1.2)
                    if np.isfinite(hi):
                        axE.axvline(hi, color="#d62728", linestyle=":", linewidth=1.2)
                    if np.isfinite(med):
                        axE.axvline(med, color="#8B0000", linestyle="--", linewidth=2.0)
                axE.set_title("Bootstrap Z_T distribution")
                axE.set_xlabel("Z_T")
                axE.set_ylabel("count")
            else:
                axE.axis("off")

        # F) bootstrap phi circular + R
        axF = fig.add_subplot(gs[1, 2], projection="polar")
        if dacc is None:
            axF.axis("off")
        else:
            phi = np.asarray(dacc["phi"], dtype=float)
            q = np.asarray(dacc["q"], dtype=float)
            mask = np.isfinite(phi) & np.isfinite(q) & (q <= Q_SIG)
            ph = phi[mask]
            if ph.size == 0:
                axF.text(0.5, 0.5, "no sig bootstrap phis", transform=axF.transAxes, ha="center", va="center", fontsize=8)
                axF.set_xticks([])
                axF.set_yticks([])
            else:
                rad = np.deg2rad(ph)
                axF.scatter(rad, np.ones_like(rad), s=24, c="#1f77b4", alpha=0.85)
                mu, R, csd = _circular_stats_deg(ph)
                axF.plot([np.deg2rad(mu), np.deg2rad(mu)], [0.0, 1.1], color="#d62728", linewidth=2.0)
                axF.set_theta_zero_location("E")
                axF.set_theta_direction(1)
                axF.set_rticks([])
                axF.set_title(f"Bootstrap phi (R={R:.2f}, csd={csd:.2f})")

        fig.suptitle(
            f"Exemplar {gene} | {emb} | {fg}\n"
            f"reproducible_localized={bool(r['reproducible_localized'])}, full_localized={bool(r['full_localized'])}",
            y=0.995,
            fontsize=12,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
        fig.savefig(out_dir / f"exemplar_{gene}_{emb}_{fg}.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig)


def _write_readme(
    out_path: Path,
    *,
    seed: int,
    q: float,
    n_bins: int,
    n_perm_split: int,
    n_perm_boot: int,
    boot_reps: int,
    boot_frac: float,
    save_boot_long: bool,
    donor_key: str,
    label_key: str,
    donor_star: str,
    cm_labels: list[str],
    cm_counts: dict[str, int],
    expr_source: str,
    embed_note: str,
    n_cells_donor: int,
    n_cells_cm: int,
    split_method: str,
    calls_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("CM Experiment #5 (Single-donor): Split-half + bootstrap reproducibility of BioRSP")
    lines.append("")
    lines.append("Hypothesis")
    lines.append(
        "Within a single donor, true localized programs should remain stable under within-donor resampling "
        "(split-half and bootstrap), while unstable patterns indicate noise, threshold artifacts, or QC-driven structure."
    )
    lines.append("")
    lines.append("Single-donor rigor")
    lines.append("- No donor replication claim; this is within-donor reproducibility only.")
    lines.append("- Split-half + bootstrap with per-resample permutation null calibration.")
    lines.append("- Direction stability summarized by mean resultant length R and circular SD.")
    lines.append("")
    lines.append("Run metadata")
    lines.append(f"- seed: {seed}")
    lines.append(f"- q: {q}")
    lines.append(f"- n_bins: {n_bins}")
    lines.append(f"- n_perm_split: {n_perm_split}")
    lines.append(f"- n_perm_boot: {n_perm_boot}")
    lines.append(f"- boot_reps: {boot_reps}")
    lines.append(f"- boot_frac: {boot_frac}")
    lines.append(f"- save_boot_long: {save_boot_long}")
    lines.append(f"- donor_key_used: {donor_key}")
    lines.append(f"- label_key_used: {label_key}")
    lines.append(f"- donor_star: {donor_star}")
    lines.append(f"- expression_source_used: {expr_source}")
    lines.append(f"- embedding_input_prep: {embed_note}")
    lines.append(f"- split_method: {split_method}")
    lines.append(f"- donor_star_total_cells: {n_cells_donor}")
    lines.append(f"- donor_star_cardiomyocytes: {n_cells_cm}")
    lines.append("")
    lines.append("Cardiomyocyte labels included")
    for lab in cm_labels:
        lines.append(f"- {lab}: {cm_counts.get(lab, 0)}")
    lines.append("")

    lines.append("Final call rule")
    lines.append("reproducible_localized = split_repro AND bootstrap_repro AND (qc_risk < 0.35)")
    lines.append("where split_repro requires both halves significant + delta_phi<=45° + delta_Z<=2")
    lines.append("and bootstrap_repro requires lowerCI(Z)>0 + R>=0.6 + fraction_significant>=0.6")
    lines.append("")

    if not calls_df.empty:
        n_repro = int(np.sum(calls_df["reproducible_localized"].to_numpy(dtype=bool)))
        lines.append(f"- reproducible_localized calls: {n_repro}/{len(calls_df)}")
        if n_repro > 0:
            lines.append(
                "- reproducible calls: "
                + ", ".join(
                    calls_df.loc[calls_df["reproducible_localized"], ["gene", "embedding", "foreground_mode"]]
                    .apply(lambda r: f"{r['gene']}|{r['embedding']}|{r['foreground_mode']}", axis=1)
                    .tolist()
                )
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
        plots_dir / "01_split_half",
        plots_dir / "02_bootstrap_distributions",
        plots_dir / "03_reproducibility_maps",
        plots_dir / "04_direction_stability",
        plots_dir / "05_qc_controls",
        plots_dir / "06_exemplar_panels",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad)

    donor_key = _resolve_key_required(adata, args.donor_key, DONOR_KEY_CANDIDATES, purpose="donor")
    label_key = _resolve_key_required(adata, args.label_key, LABEL_KEY_CANDIDATES, purpose="label")

    labels_all = adata.obs[label_key].astype("string").fillna("NA").astype(str)
    cm_mask_all = labels_all.map(_is_cm_label).to_numpy(dtype=bool)
    if int(cm_mask_all.sum()) == 0:
        raise RuntimeError("No cardiomyocyte cells detected by substring matching")

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
        raise RuntimeError("donor_star cardiomyocyte subset is empty")

    cm_label_counts_donor = labels_donor.loc[cm_mask_donor].value_counts().sort_index()
    cm_labels_included = cm_label_counts_donor.index.astype(str).tolist()
    print("cm_labels_included=" + ", ".join(cm_labels_included))
    for label, count in cm_label_counts_donor.items():
        print(f"cm_label_count[{label}]={int(count)}")

    if int(adata_cm.n_obs) < 2000:
        print(f"WARNING: adata_cm.n_obs={int(adata_cm.n_obs)} < 2000 (cm_underpowered=True)")

    expr_matrix_cm, adata_like_cm, expr_source = _choose_expression_source(
        adata_cm,
        layer_arg=args.layer,
        use_raw_arg=bool(args.use_raw),
    )

    # QC covariates.
    total_counts, key_total = _safe_numeric_obs(adata_cm, ["total_counts", "n_counts", "n_genes_by_counts"])
    if total_counts is None:
        total_counts = _total_counts_vector(adata_cm, expr_matrix_cm)
        key_total = "computed:expr_sum"

    pct_mt, key_mt = _safe_numeric_obs(adata_cm, ["pct_counts_mt", "percent.mt", "pct_mt"])
    if pct_mt is None:
        pct_mt, key_mt2 = _pct_mt_vector(adata_cm, expr_matrix_cm, adata_like_cm)
        key_mt = key_mt2

    pct_ribo, key_ribo = _safe_numeric_obs(adata_cm, ["pct_counts_ribo", "percent.ribo", "pct_ribo"])

    # Panel.
    panel_status, panel_df = _resolve_panel(adata_like_cm)
    panel_df.to_csv(tables_dir / "gene_panel_status.csv", index=False)
    genes_present = [st for st in panel_status if st.present and st.gene_idx is not None]
    if len(genes_present) == 0:
        raise RuntimeError("No panel genes resolved in selected expression namespace")

    expr_by_gene = {st.gene: get_feature_vector(expr_matrix_cm, int(st.gene_idx)) for st in genes_present}

    # Embeddings.
    adata_embed, embed_note = _prepare_embedding_input(adata_cm, expr_matrix_cm, expr_source)
    embeddings, n_pcs_used = _compute_fixed_embeddings(
        adata_embed,
        seed=int(args.seed),
        k_pca=int(args.k_pca),
    )
    emb_map = {e.key: e for e in embeddings}

    n_cells = int(adata_cm.n_obs)

    # Split halves.
    h1_idx, h2_idx, split_method, split_assign = _make_split_halves(
        n_cells=n_cells,
        seed=int(args.seed),
        strat_values=np.asarray(total_counts, dtype=float) if total_counts is not None else None,
    )

    split_rows_h1, detail_h1 = _score_sample_set(
        sample_name="half1",
        sample_idx=h1_idx,
        embeddings=embeddings,
        genes_present=genes_present,
        expr_by_gene=expr_by_gene,
        q_top=float(args.q),
        n_bins=int(args.n_bins),
        n_perm=int(args.n_perm_split),
        seed_base=int(args.seed + 1000),
        keep_profiles=True,
    )
    split_rows_h2, detail_h2 = _score_sample_set(
        sample_name="half2",
        sample_idx=h2_idx,
        embeddings=embeddings,
        genes_present=genes_present,
        expr_by_gene=expr_by_gene,
        q_top=float(args.q),
        n_bins=int(args.n_bins),
        n_perm=int(args.n_perm_split),
        seed_base=int(args.seed + 2000),
        keep_profiles=True,
    )

    full_idx = np.arange(n_cells, dtype=int)
    full_rows, detail_full = _score_sample_set(
        sample_name="full",
        sample_idx=full_idx,
        embeddings=embeddings,
        genes_present=genes_present,
        expr_by_gene=expr_by_gene,
        q_top=float(args.q),
        n_bins=int(args.n_bins),
        n_perm=int(args.n_perm_split),
        seed_base=int(args.seed + 3000),
        keep_profiles=True,
    )

    split_scores_df = pd.DataFrame(split_rows_h1 + split_rows_h2 + full_rows)
    split_scores_df.to_csv(tables_dir / "split_half_scores.csv", index=False)

    split_summary_df = _summarize_split_half(split_scores_df.loc[split_scores_df["sample"].isin(["half1", "half2"])])
    split_summary_df.to_csv(tables_dir / "split_half_repro_summary.csv", index=False)

    detail_map = {}
    detail_map.update(detail_h1)
    detail_map.update(detail_h2)
    detail_map.update(detail_full)

    # Bootstrap scoring.
    boot_summary_df, boot_long_df, boot_acc = _bootstrap_repro(
        embeddings=embeddings,
        genes_present=genes_present,
        expr_by_gene=expr_by_gene,
        n_cells=n_cells,
        q_top=float(args.q),
        n_bins=int(args.n_bins),
        n_perm_boot=int(args.n_perm_boot),
        boot_reps=int(args.boot_reps),
        boot_frac=float(args.boot_frac),
        seed=int(args.seed + 5000),
        save_boot_long=bool(args.save_boot_long),
        out_tables_dir=tables_dir,
    )
    boot_summary_df.to_csv(tables_dir / "bootstrap_summary.csv", index=False)
    if boot_long_df is not None:
        boot_long_df.to_csv(tables_dir / "bootstrap_scores_long.csv", index=False)

    # Full-data QC audit by gene x foreground.
    qc_rows = []
    for st in genes_present:
        expr = np.asarray(expr_by_gene[st.gene], dtype=float)
        for fg_mode in ["F0_detect", "F1_topq"]:
            if fg_mode == "F0_detect":
                fg = expr > 0.0
            else:
                fg = _top_q_mask(expr, q=float(args.q))
            ff = fg.astype(float)
            rho_tc = _safe_spearman(ff, np.asarray(total_counts, dtype=float) if total_counts is not None else None)
            rho_mt = _safe_spearman(ff, np.asarray(pct_mt, dtype=float) if pct_mt is not None else None)
            rho_rb = _safe_spearman(ff, np.asarray(pct_ribo, dtype=float) if pct_ribo is not None else None)
            vals = np.array([rho_tc, rho_mt, rho_rb], dtype=float)
            fin = vals[np.isfinite(vals)]
            qc_risk = float(np.max(np.abs(fin))) if fin.size > 0 else 0.0
            qc_rows.append(
                {
                    "gene": st.gene,
                    "foreground_mode": fg_mode,
                    "rho_total_counts": rho_tc,
                    "rho_pct_mt": rho_mt,
                    "rho_pct_ribo": rho_rb,
                    "qc_risk": qc_risk,
                    "qc_driven": bool(qc_risk >= QC_RISK_THRESH),
                }
            )
    qc_audit_df = pd.DataFrame(qc_rows)
    qc_audit_df.to_csv(tables_dir / "qc_audit.csv", index=False)

    # Final reproducible calls.
    full_df = split_scores_df.loc[split_scores_df["sample"] == "full"].copy()

    calls_rows = []
    for _, srow in split_summary_df.iterrows():
        gene = str(srow["gene"])
        emb = str(srow["embedding"])
        fg = str(srow["foreground_mode"])

        b = boot_summary_df.loc[
            (boot_summary_df["gene"] == gene)
            & (boot_summary_df["embedding"] == emb)
            & (boot_summary_df["foreground_mode"] == fg)
        ]
        if b.empty:
            continue
        brow = b.iloc[0]

        q = qc_audit_df.loc[
            (qc_audit_df["gene"] == gene)
            & (qc_audit_df["foreground_mode"] == fg)
        ]
        qrow = q.iloc[0] if not q.empty else pd.Series({"qc_risk": np.nan, "qc_driven": False})

        f = full_df.loc[
            (full_df["gene"] == gene)
            & (full_df["embedding"] == emb)
            & (full_df["foreground_mode"] == fg)
        ]
        frow = f.iloc[0] if not f.empty else pd.Series({"q_T": np.nan, "Z_T": np.nan})

        split_repro = bool(
            bool(srow["half_consistent_significant"])
            and np.isfinite(float(srow["delta_phi_deg"]))
            and float(srow["delta_phi_deg"]) <= 45.0
            and np.isfinite(float(srow["delta_Z"]))
            and float(srow["delta_Z"]) <= 2.0
        )

        boot_repro = bool(brow["bootstrap_reproducible"])
        qc_ok = bool(np.isfinite(float(qrow["qc_risk"])) and float(qrow["qc_risk"]) < QC_RISK_THRESH)
        if not np.isfinite(float(qrow["qc_risk"])):
            qc_ok = True

        full_localized = bool(np.isfinite(float(frow.get("q_T", np.nan))) and float(frow.get("q_T", np.nan)) <= Q_SIG)

        final = bool(split_repro and boot_repro and qc_ok)

        calls_rows.append(
            {
                "gene": gene,
                "embedding": emb,
                "foreground_mode": fg,
                "split_repro": split_repro,
                "bootstrap_repro": boot_repro,
                "qc_risk": float(qrow["qc_risk"]) if np.isfinite(float(qrow["qc_risk"])) else np.nan,
                "qc_ok": qc_ok,
                "reproducible_localized": final,
                "full_localized": full_localized,
                "half_consistent_significant": bool(srow["half_consistent_significant"]),
                "delta_Z": float(srow["delta_Z"]) if np.isfinite(float(srow["delta_Z"])) else np.nan,
                "delta_phi_deg": float(srow["delta_phi_deg"]) if np.isfinite(float(srow["delta_phi_deg"])) else np.nan,
                "fraction_significant": float(brow["fraction_significant"]) if np.isfinite(float(brow["fraction_significant"])) else np.nan,
                "boot_Z_median": float(brow["Z_median"]) if np.isfinite(float(brow["Z_median"])) else np.nan,
                "boot_Z_ci_low": float(brow["Z_ci_low"]) if np.isfinite(float(brow["Z_ci_low"])) else np.nan,
                "boot_Z_ci_high": float(brow["Z_ci_high"]) if np.isfinite(float(brow["Z_ci_high"])) else np.nan,
                "boot_R": float(brow["R"]) if np.isfinite(float(brow["R"])) else np.nan,
                "full_q_T": float(frow.get("q_T", np.nan)) if np.isfinite(float(frow.get("q_T", np.nan))) else np.nan,
                "full_Z_T": float(frow.get("Z_T", np.nan)) if np.isfinite(float(frow.get("Z_T", np.nan))) else np.nan,
            }
        )

    calls_df = pd.DataFrame(calls_rows)
    calls_df.to_csv(tables_dir / "reproducible_gene_calls.csv", index=False)

    # Plots.
    _plot_overview(
        plots_dir / "00_overview",
        umap_coords=emb_map["umap_repr"].coords,
        expr_by_gene=expr_by_gene,
        total_counts=np.asarray(total_counts, dtype=float) if total_counts is not None else None,
        split_assign=split_assign,
        split_method=split_method,
    )

    _plot_split_half(
        plots_dir / "01_split_half",
        split_scores=split_scores_df.loc[split_scores_df["sample"].isin(["half1", "half2"])],
        split_summary=split_summary_df,
    )

    _plot_bootstrap_distributions(
        plots_dir / "02_bootstrap_distributions",
        boot_summary=boot_summary_df,
        boot_long_df=boot_long_df,
        acc=boot_acc,
    )

    _plot_repro_maps(
        plots_dir / "03_reproducibility_maps",
        boot_summary=boot_summary_df,
        calls_df=calls_df,
    )

    _plot_direction_stability(
        plots_dir / "04_direction_stability",
        boot_summary=boot_summary_df,
        acc=boot_acc,
        split_summary=split_summary_df,
    )

    _plot_qc_controls(
        plots_dir / "05_qc_controls",
        calls_df=calls_df,
        qc_audit_df=qc_audit_df,
    )

    _plot_exemplar_panels(
        plots_dir / "06_exemplar_panels",
        calls_df=calls_df,
        split_scores_df=split_scores_df,
        boot_summary_df=boot_summary_df,
        acc=boot_acc,
        detail_map=detail_map,
        embeddings_map=emb_map,
        expr_by_gene=expr_by_gene,
    )

    # README.
    cm_labels = sorted(labels_donor.loc[cm_mask_donor].unique().tolist())
    cm_counts = labels_donor.loc[cm_mask_donor].value_counts().to_dict()
    _write_readme(
        out_root / "README.txt",
        seed=int(args.seed),
        q=float(args.q),
        n_bins=int(args.n_bins),
        n_perm_split=int(args.n_perm_split),
        n_perm_boot=int(args.n_perm_boot),
        boot_reps=int(args.boot_reps),
        boot_frac=float(args.boot_frac),
        save_boot_long=bool(args.save_boot_long),
        donor_key=donor_key,
        label_key=label_key,
        donor_star=donor_star,
        cm_labels=cm_labels,
        cm_counts={str(k): int(v) for k, v in cm_counts.items()},
        expr_source=expr_source,
        embed_note=f"{embed_note}; n_pcs_used={n_pcs_used}",
        n_cells_donor=int(adata_donor.n_obs),
        n_cells_cm=int(adata_cm.n_obs),
        split_method=split_method,
        calls_df=calls_df,
    )

    # Required outputs verification.
    required = [
        tables_dir / "donor_choice.csv",
        tables_dir / "gene_panel_status.csv",
        tables_dir / "split_half_scores.csv",
        tables_dir / "split_half_repro_summary.csv",
        tables_dir / "bootstrap_summary.csv",
        tables_dir / "reproducible_gene_calls.csv",
        tables_dir / "qc_audit.csv",
        plots_dir / "00_overview",
        plots_dir / "01_split_half",
        plots_dir / "02_bootstrap_distributions",
        plots_dir / "03_reproducibility_maps",
        plots_dir / "04_direction_stability",
        plots_dir / "05_qc_controls",
        plots_dir / "06_exemplar_panels",
        out_root / "README.txt",
    ]
    if bool(args.save_boot_long):
        required.append(tables_dir / "bootstrap_scores_long.csv")

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing required outputs: " + ", ".join(missing))

    print(f"donor_key_used={donor_key}")
    print(f"label_key_used={label_key}")
    print(f"donor_star={donor_star}")
    print(f"expression_source_used={expr_source}")
    print(f"split_method={split_method}")
    print(f"boot_reps={int(args.boot_reps)}")
    print(f"n_tests_split={int(len(split_scores_df))}")
    print(f"n_calls={int(len(calls_df))}")
    print(f"reproducible_calls={int(np.sum(calls_df['reproducible_localized'].to_numpy(dtype=bool)) if not calls_df.empty else 0)}")
    print(f"results_root={out_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
