#!/usr/bin/env python3
"""CM Experiment #8 (single donor): BioRSP vs graph autocorrelation baselines.

Why this experiment exists:
Reviewers commonly ask why angular localization (BioRSP) is needed if graph
spatial-autocorrelation baselines (Moran's I / Geary's C) are available.
This script quantifies agreement and disagreement regimes between methods.

Scientific notes (for README/comments; no runtime fetching):
- Moran's I and Geary's C are standard spatial autocorrelation baselines and are
  widely used in spatial omics pipelines (e.g., Squidpy context).
- Geary's C has null expectation near 1, with C<1 indicating positive autocorrelation.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
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
from biorsp.pipeline.hierarchy import _resolve_expr_matrix
from biorsp.plotting.styles import DEFAULT_PLOT_STYLE, apply_plot_style
from biorsp.stats.moran import morans_i
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

BASE_GENE_PANEL = [
    "TNNT2",
    "TNNI3",
    "ACTC1",
    "MYH6",
    "MYH7",
    "RYR2",
    "ATP2A2",
    "PLN",
    "NPPA",
    "NPPB",
]

UNDERPOWERED_PREV = 0.005
UNDERPOWERED_MIN_FG = 50
Q_SIG = 0.05


@dataclass(frozen=True)
class GeneStatus:
    gene: str
    present: bool
    status: str
    resolved_gene: str
    gene_idx: int | None
    resolution_source: str
    symbol_column: str


@dataclass(frozen=True)
class EmbeddingSpec:
    family: str
    name: str
    params: dict[str, Any]
    coords: np.ndarray


@dataclass(frozen=True)
class EmbeddingGeom:
    center_xy: np.ndarray
    theta: np.ndarray
    bin_id: np.ndarray
    bin_counts_total: np.ndarray


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
        description="CM Experiment #8: cross-validate BioRSP localization against graph Moran/Geary baselines."
    )
    p.add_argument("--h5ad", default="data/processed/HT_pca_umap.h5ad")
    p.add_argument(
        "--out",
        default="experiments/heart_single_donor_cardiomyocyte/results/cm_experiment8_graph_baselines_moran_geary",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--q", type=float, default=0.10)
    p.add_argument("--n_bins", type=int, default=64)
    p.add_argument("--n_perm_biorsp", type=int, default=300)
    p.add_argument("--n_perm_graph", type=int, default=300)
    p.add_argument("--knn_list", type=int, nargs="+", default=[15, 30, 50])
    p.add_argument("--k_pca", type=int, default=50)
    p.add_argument("--compute_moran_binary", type=_str2bool, default=True)
    p.add_argument("--include_detect", type=_str2bool, default=False)
    p.add_argument("--extra_genes_csv", default="")
    p.add_argument("--top_extra", type=int, default=50)
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
    for key in candidates:
        if key in adata.obs.columns:
            return key
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


def _read_extra_genes(path: str, top_n: int) -> list[str]:
    if str(path).strip() == "":
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"extra_genes_csv not found: {p}")
    df = pd.read_csv(p)
    if "gene" in df.columns:
        col = df["gene"].astype(str)
    else:
        col = df.iloc[:, 0].astype(str)
    out: list[str] = []
    seen: set[str] = set()
    for g in col.tolist():
        gg = g.strip()
        if gg == "" or gg in seen:
            continue
        seen.add(gg)
        out.append(gg)
        if len(out) >= int(top_n):
            break
    return out


def _resolve_gene(gene: str, adata_like: Any) -> GeneStatus:
    try:
        idx, label, symbol_col, source = resolve_feature_index(adata_like, gene)
        return GeneStatus(
            gene=gene,
            present=True,
            status="resolved",
            resolved_gene=str(label),
            gene_idx=int(idx),
            resolution_source=str(source),
            symbol_column=str(symbol_col or ""),
        )
    except KeyError:
        return GeneStatus(
            gene=gene,
            present=False,
            status="missing",
            resolved_gene="",
            gene_idx=None,
            resolution_source="",
            symbol_column="",
        )


def _top_q_mask(values: np.ndarray, q: float) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = int(max(1, round(float(q) * n)))
    order = np.argsort(x, kind="mergesort")
    keep = order[-k:]
    out = np.zeros(n, dtype=bool)
    out[keep] = True
    return out


def _prepare_proc_matrix(
    expr_matrix_cm: Any,
    *,
    expr_source: str,
) -> tuple[np.ndarray, str]:
    """Prepare matrix for graph baselines and embeddings.

    If counts layer is used, normalize_total + log1p; otherwise pass through.
    """
    import scanpy as sc

    x = (
        expr_matrix_cm.toarray().astype(float)
        if sp.issparse(expr_matrix_cm)
        else np.asarray(expr_matrix_cm, dtype=float)
    )
    ad_tmp = ad.AnnData(X=x)

    if expr_source.startswith("layer:counts"):
        sc.pp.normalize_total(ad_tmp, target_sum=1e4)
        sc.pp.log1p(ad_tmp)
        note = "counts->normalize_total(1e4)->log1p"
    else:
        note = f"{expr_source}_as_is"

    return np.asarray(ad_tmp.X, dtype=float), note


def _build_embeddings_and_graphs(
    *,
    x_proc: np.ndarray,
    obs_cm: pd.DataFrame,
    knn_list: list[int],
    seed: int,
    k_pca: int,
) -> tuple[list[EmbeddingSpec], dict[int, sp.csr_matrix], dict[int, np.ndarray], int]:
    import scanpy as sc

    ad_graph = ad.AnnData(X=np.asarray(x_proc, dtype=float), obs=obs_cm.copy())
    n_cells, n_vars = ad_graph.n_obs, ad_graph.n_vars
    n_pcs = int(max(2, min(int(k_pca), 50, n_vars - 1, n_cells - 1)))

    sc.pp.pca(ad_graph, n_comps=n_pcs, svd_solver="arpack", random_state=int(seed))
    pca_all = np.asarray(ad_graph.obsm["X_pca"], dtype=float)

    # Representative UMAP at n_neighbors=30
    sc.pp.neighbors(
        ad_graph,
        n_neighbors=30,
        n_pcs=n_pcs,
        use_rep="X_pca",
        random_state=int(seed),
    )
    sc.tl.umap(ad_graph, min_dist=0.1, random_state=0)
    umap_xy = np.asarray(ad_graph.obsm["X_umap"], dtype=float)[:, :2].copy()

    embeddings = [
        EmbeddingSpec(
            family="PCA",
            name="pca2d",
            params={"n_pcs": n_pcs, "seed": int(seed)},
            coords=pca_all[:, :2].copy(),
        ),
        EmbeddingSpec(
            family="UMAP",
            name="umap_repr",
            params={
                "n_neighbors": 30,
                "min_dist": 0.1,
                "random_state": 0,
                "n_pcs": n_pcs,
            },
            coords=umap_xy,
        ),
    ]

    # Explicitly build graph per requested k.
    graphs: dict[int, sp.csr_matrix] = {}
    degree_dict: dict[int, np.ndarray] = {}
    for k in [int(v) for v in knn_list]:
        sc.pp.neighbors(
            ad_graph,
            n_neighbors=int(k),
            n_pcs=n_pcs,
            use_rep="X_pca",
            random_state=int(seed),
        )
        w = ad_graph.obsp["connectivities"].tocsr(copy=True)
        # enforce symmetric, zero-diagonal weights
        w = ((w + w.T) * 0.5).tocsr()
        w.setdiag(0.0)
        w.eliminate_zeros()
        graphs[int(k)] = w
        degree_dict[int(k)] = np.asarray((w > 0).sum(axis=1)).ravel().astype(float)

    return embeddings, graphs, degree_dict, n_pcs


def _compute_geom(spec: EmbeddingSpec, n_bins: int) -> EmbeddingGeom:
    center = compute_vantage_point(spec.coords, method="mean")
    theta = compute_theta(spec.coords, center)
    _, bin_id = bin_theta(theta, bins=int(n_bins))
    counts = np.bincount(bin_id, minlength=int(n_bins)).astype(float)
    return EmbeddingGeom(
        center_xy=center,
        theta=theta,
        bin_id=bin_id,
        bin_counts_total=counts,
    )


def _score_biorsp(
    *,
    expr: np.ndarray,
    geom: EmbeddingGeom,
    mode: str,
    q: float,
    n_bins: int,
    n_perm: int,
    seed: int,
    with_profiles: bool,
) -> dict[str, Any]:
    x = np.asarray(expr, dtype=float).ravel()
    if mode == "topq":
        f = _top_q_mask(x, q=float(q))
        q_param = float(q)
    elif mode == "detect":
        f = x > 0.0
        q_param = np.nan
    else:
        raise ValueError(f"Unknown mode: {mode}")

    n_cells = int(f.size)
    n_fg = int(f.sum())
    prev = float(n_fg / max(1, n_cells))
    underpowered = bool(prev < UNDERPOWERED_PREV or n_fg < UNDERPOWERED_MIN_FG)

    if n_fg == 0 or n_fg == n_cells:
        return {
            "q_param": q_param,
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
            "E_obs": np.zeros(int(n_bins), dtype=float) if with_profiles else None,
            "null_E": None,
            "null_T": None,
            "fg": f,
        }

    e_obs, _, _, _ = compute_rsp_profile_from_boolean(
        f,
        geom.theta,
        int(n_bins),
        bin_id=geom.bin_id,
        bin_counts_total=geom.bin_counts_total,
    )
    phi_idx = int(np.argmax(np.abs(e_obs)))
    phi_hat = float(np.degrees(theta_bin_centers(int(n_bins))[phi_idx]) % 360.0)
    t_obs = float(np.max(np.abs(e_obs)))

    if underpowered:
        return {
            "q_param": q_param,
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
            "E_obs": e_obs if with_profiles else None,
            "null_E": None,
            "null_T": None,
            "fg": f,
        }

    perm = perm_null_T_and_profile(
        expr=f.astype(float),
        theta=geom.theta,
        donor_ids=None,
        n_bins=int(n_bins),
        n_perm=int(n_perm),
        seed=int(seed),
        donor_stratified=False,
        bin_id=geom.bin_id,
        bin_counts_total=geom.bin_counts_total,
    )

    null_e = np.asarray(perm["null_E_phi"], dtype=float)
    null_t = np.asarray(perm["null_T"], dtype=float)

    return {
        "q_param": q_param,
        "prev": prev,
        "n_fg": n_fg,
        "n_cells": n_cells,
        "T_obs": float(perm["T_obs"]),
        "p_T": float(perm["p_T"]),
        "Z_T": float(robust_z(float(perm["T_obs"]), null_t)),
        "coverage_C": float(coverage_from_null(e_obs, null_e, q=0.95)),
        "peaks_K": int(peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95)),
        "phi_hat_deg": phi_hat,
        "underpowered_flag": False,
        "E_obs": e_obs if with_profiles else None,
        "null_E": null_e if with_profiles else None,
        "null_T": null_t if with_profiles else None,
        "fg": f,
    }


def _row_standardize(w: sp.csr_matrix) -> sp.csr_matrix:
    w_csr = w.tocsr(copy=True)
    row_sum = np.asarray(w_csr.sum(axis=1)).ravel()
    scale = np.zeros_like(row_sum, dtype=float)
    nz = row_sum > 0
    scale[nz] = 1.0 / row_sum[nz]
    if np.any(nz):
        w_csr = sp.diags(scale).dot(w_csr).tocsr()
    return w_csr


def _gearys_c(x: np.ndarray, w: sp.csr_matrix, row_standardize: bool = True) -> float:
    x_arr = np.asarray(x, dtype=float).ravel()
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError("x must be non-empty 1D")
    if not np.isfinite(x_arr).all():
        raise ValueError("x contains NaN/inf")
    if not sp.issparse(w):
        raise TypeError("w must be sparse")

    w_use = _row_standardize(w) if row_standardize else w.tocsr(copy=True)
    n = int(x_arr.size)
    if w_use.shape != (n, n):
        raise ValueError("w shape mismatch")

    xbar = float(np.mean(x_arr))
    den = float(np.sum((x_arr - xbar) ** 2))
    if den <= 0:
        raise ValueError("Variance zero; Geary's C undefined")

    w_coo = w_use.tocoo(copy=False)
    diff = x_arr[w_coo.row] - x_arr[w_coo.col]
    num = float(np.sum(w_coo.data * (diff**2)))
    s0 = float(w_use.sum())
    if s0 <= 0:
        raise ValueError("Sum weights zero; Geary's C undefined")

    c = ((n - 1.0) * num) / (2.0 * s0 * den)
    return float(c)


def _score_graph_continuous(
    *,
    x: np.ndarray,
    w: sp.csr_matrix,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    x_arr = np.asarray(x, dtype=float).ravel()
    if x_arr.size == 0 or not np.isfinite(x_arr).all():
        return {
            "I_obs": np.nan,
            "p_I": np.nan,
            "Z_I": np.nan,
            "C_obs": np.nan,
            "A_C": np.nan,
            "p_C": np.nan,
            "Z_C": np.nan,
        }

    try:
        i_obs = float(morans_i(x_arr, w, row_standardize=True))
    except Exception:
        i_obs = np.nan

    try:
        c_obs = float(_gearys_c(x_arr, w, row_standardize=True))
    except Exception:
        c_obs = np.nan

    a_obs = float(1.0 - c_obs) if np.isfinite(c_obs) else np.nan

    rng = np.random.default_rng(int(seed))
    null_i = np.zeros(int(n_perm), dtype=float)
    null_c = np.zeros(int(n_perm), dtype=float)
    null_a = np.zeros(int(n_perm), dtype=float)

    for i in range(int(n_perm)):
        xp = x_arr[rng.permutation(x_arr.size)]
        try:
            ii = float(morans_i(xp, w, row_standardize=True))
        except Exception:
            ii = np.nan
        try:
            cc = float(_gearys_c(xp, w, row_standardize=True))
        except Exception:
            cc = np.nan
        null_i[i] = ii
        null_c[i] = cc
        null_a[i] = float(1.0 - cc) if np.isfinite(cc) else np.nan

    i_valid = null_i[np.isfinite(null_i)]
    c_valid = null_c[np.isfinite(null_c)]
    a_valid = null_a[np.isfinite(null_a)]

    if np.isfinite(i_obs) and i_valid.size > 0:
        p_i = float((1.0 + np.sum(i_valid >= i_obs)) / (1.0 + i_valid.size))
        z_i = float(robust_z(i_obs, i_valid))
    else:
        p_i, z_i = np.nan, np.nan

    if np.isfinite(c_obs) and c_valid.size > 0:
        p_c = float(
            (1.0 + np.sum(np.abs(c_valid - 1.0) >= np.abs(c_obs - 1.0)))
            / (1.0 + c_valid.size)
        )
    else:
        p_c = np.nan

    if np.isfinite(a_obs) and a_valid.size > 0:
        z_c = float(robust_z(a_obs, a_valid))
    else:
        z_c = np.nan

    return {
        "I_obs": i_obs,
        "p_I": p_i,
        "Z_I": z_i,
        "C_obs": c_obs,
        "A_C": a_obs,
        "p_C": p_c,
        "Z_C": z_c,
    }


def _score_graph_binary(
    *,
    y: np.ndarray,
    w: sp.csr_matrix,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size == 0 or not np.isfinite(yy).all() or np.allclose(yy, yy[0]):
        return {"I_bin": np.nan, "p_I_bin": np.nan, "Z_I_bin": np.nan}

    try:
        i_obs = float(morans_i(yy, w, row_standardize=True))
    except Exception:
        return {"I_bin": np.nan, "p_I_bin": np.nan, "Z_I_bin": np.nan}

    rng = np.random.default_rng(int(seed))
    null_i = np.zeros(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        yp = yy[rng.permutation(yy.size)]
        try:
            null_i[i] = float(morans_i(yp, w, row_standardize=True))
        except Exception:
            null_i[i] = np.nan

    valid = null_i[np.isfinite(null_i)]
    if valid.size == 0:
        return {"I_bin": i_obs, "p_I_bin": np.nan, "Z_I_bin": np.nan}

    p = float((1.0 + np.sum(valid >= i_obs)) / (1.0 + valid.size))
    z = float(robust_z(i_obs, valid))
    return {"I_bin": i_obs, "p_I_bin": p, "Z_I_bin": z}


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(xv) & np.isfinite(yv)
    if int(m.sum()) < 3:
        return np.nan
    xs = xv[m]
    ys = yv[m]
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return np.nan
    rho = spearmanr(xs, ys, nan_policy="omit").correlation
    return float(rho) if rho is not None else np.nan


def _plot_polar(
    ax: plt.Axes, e_obs: np.ndarray, null_e: np.ndarray | None, title: str, stats: str
) -> None:
    n_bins = int(e_obs.size)
    th_c = theta_bin_centers(n_bins)
    th = np.concatenate([th_c, th_c[:1]])
    obs = np.concatenate([e_obs, e_obs[:1]])
    ax.plot(th, obs, color="#8B0000", linewidth=2.0)
    if null_e is not None:
        hi = np.quantile(null_e, 0.95, axis=0)
        lo = np.quantile(null_e, 0.05, axis=0)
        ax.plot(
            th,
            np.concatenate([hi, hi[:1]]),
            color="#333",
            linestyle="--",
            linewidth=1.0,
        )
        ax.plot(
            th,
            np.concatenate([lo, lo[:1]]),
            color="#333",
            linestyle="--",
            linewidth=1.0,
        )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rticks([])
    ax.set_title(title, fontsize=9)
    ax.text(
        0.02,
        0.02,
        stats,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#999", "alpha": 0.85},
    )


def _write_readme(
    out_path: Path,
    *,
    args: argparse.Namespace,
    donor_key: str,
    label_key: str,
    donor_star: str,
    expr_source: str,
    prep_note: str,
    n_cells_donor: int,
    n_cells_cm: int,
    cm_label_counts: dict[str, int],
    n_genes_present: int,
    warnings_log: list[str],
) -> None:
    lines: list[str] = []
    lines.append(
        "CM Experiment #8 (Single-donor): BioRSP vs graph baselines (Moran's I / Geary's C)"
    )
    lines.append("")
    lines.append("Why this experiment exists")
    lines.append(
        "BioRSP is an angular localization-mode detector on 2D embeddings. This benchmark tests whether BioRSP "
        "is merely rediscovering generic graph autocorrelation or provides complementary structure information."
    )
    lines.append("")
    lines.append("Method references (conceptual)")
    lines.append(
        "- Moran's I and Geary's C are standard graph/spatial autocorrelation baselines."
    )
    lines.append(
        "- Geary's C null expectation is near 1; C<1 indicates positive autocorrelation."
    )
    lines.append("")
    lines.append("Run metadata")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- q: {args.q}")
    lines.append(f"- n_bins: {args.n_bins}")
    lines.append(f"- n_perm_biorsp: {args.n_perm_biorsp}")
    lines.append(f"- n_perm_graph: {args.n_perm_graph}")
    lines.append(f"- knn_list: {', '.join(map(str, args.knn_list))}")
    lines.append(f"- compute_moran_binary: {bool(args.compute_moran_binary)}")
    lines.append("- foreground mode primary: topq")
    lines.append(f"- include_detect: {bool(args.include_detect)}")
    lines.append(f"- donor_key_used: {donor_key}")
    lines.append(f"- label_key_used: {label_key}")
    lines.append(f"- donor_star: {donor_star}")
    lines.append(f"- expression_source_used: {expr_source}")
    lines.append(f"- preprocessing_for_graph_and_embedding: {prep_note}")
    lines.append(f"- donor_star_total_cells: {n_cells_donor}")
    lines.append(f"- donor_star_cardiomyocytes: {n_cells_cm}")
    lines.append(f"- n_genes_present: {n_genes_present}")
    lines.append("")
    lines.append("Cardiomyocyte labels included")
    for k, v in cm_label_counts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Interpretation guardrails")
    lines.append(
        "- BioRSP directions are representation-conditional (phi is not anatomy)."
    )
    lines.append(
        "- Agreement between methods suggests shared structure; disagreement regimes are diagnostic and not errors."
    )

    if warnings_log:
        lines.append("")
        lines.append("Warnings")
        for w in warnings_log:
            lines.append(f"- {w}")

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
        plots_dir / "01_method_strength_scatter",
        plots_dir / "02_agreement_regimes",
        plots_dir / "03_rank_concordance",
        plots_dir / "04_exemplar_panels",
        plots_dir / "05_k_sensitivity",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    warnings_log: list[str] = []

    adata = ad.read_h5ad(args.h5ad)

    donor_key = _resolve_key_required(
        adata, args.donor_key, DONOR_KEY_CANDIDATES, purpose="donor"
    )
    label_key = _resolve_key_required(
        adata, args.label_key, LABEL_KEY_CANDIDATES, purpose="label"
    )

    labels_all = adata.obs[label_key].astype("string").fillna("NA").astype(str)
    donor_ids_all = adata.obs[donor_key].astype("string").fillna("NA").astype(str)
    cm_mask_all = labels_all.map(_is_cm_label).to_numpy(dtype=bool)
    if int(cm_mask_all.sum()) == 0:
        raise RuntimeError("No cardiomyocyte cells detected by substring matching")

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

    if int(adata_cm.n_obs) < 2000:
        msg = f"CM subset has {int(adata_cm.n_obs)} cells (<2000); proceeding with warning."
        warnings_log.append(msg)
        print(f"WARNING: {msg}")

    cm_label_counts = (
        labels_donor.loc[cm_mask_donor]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    print("cm_labels_included=" + ", ".join(cm_label_counts.keys()))

    expr_matrix_cm, adata_like_cm, expr_source = _choose_expression_source(
        adata_cm,
        layer_arg=args.layer,
        use_raw_arg=bool(args.use_raw),
    )

    extra_genes = _read_extra_genes(
        str(args.extra_genes_csv), top_n=int(args.top_extra)
    )
    gene_candidates = list(dict.fromkeys(BASE_GENE_PANEL + extra_genes))
    gene_statuses = [_resolve_gene(g, adata_like_cm) for g in gene_candidates]
    gene_panel_df = pd.DataFrame(
        [
            {
                "gene": st.gene,
                "present": st.present,
                "status": st.status,
                "resolved_gene": st.resolved_gene,
                "gene_idx": st.gene_idx if st.gene_idx is not None else "",
                "resolution_source": st.resolution_source,
                "symbol_column": st.symbol_column,
            }
            for st in gene_statuses
        ]
    )
    gene_panel_df.to_csv(tables_dir / "gene_panel_status.csv", index=False)

    genes_present = [
        st for st in gene_statuses if st.present and st.gene_idx is not None
    ]
    if len(genes_present) == 0:
        raise RuntimeError("No genes from panel resolved")

    expr_by_gene = {
        st.gene: get_feature_vector(expr_matrix_cm, int(st.gene_idx))
        for st in genes_present
    }

    # Prepare matrix for graph baselines and embedding construction.
    x_proc, prep_note = _prepare_proc_matrix(expr_matrix_cm, expr_source=expr_source)

    embeddings, graphs, degrees_by_k, n_pcs = _build_embeddings_and_graphs(
        x_proc=x_proc,
        obs_cm=adata_cm.obs,
        knn_list=[int(k) for k in args.knn_list],
        seed=int(args.seed),
        k_pca=int(args.k_pca),
    )
    emb_map = {e.name: e for e in embeddings}

    # BioRSP scoring.
    modes = ["topq"] + (["detect"] if bool(args.include_detect) else [])
    biorsp_rows: list[dict[str, Any]] = []

    total_biorsp_tests = len(embeddings) * len(genes_present) * len(modes)
    done = 0
    for emb_i, emb in enumerate(embeddings):
        geom = _compute_geom(emb, int(args.n_bins))
        for gene_i, st in enumerate(genes_present):
            expr = np.asarray(expr_by_gene[st.gene], dtype=float)
            for mode_i, mode in enumerate(modes):
                sc = _score_biorsp(
                    expr=expr,
                    geom=geom,
                    mode=mode,
                    q=float(args.q),
                    n_bins=int(args.n_bins),
                    n_perm=int(args.n_perm_biorsp),
                    seed=int(
                        args.seed + emb_i * 100000 + gene_i * 123 + mode_i * 17 + 3
                    ),
                    with_profiles=False,
                )
                biorsp_rows.append(
                    {
                        "gene": st.gene,
                        "embedding": emb.name,
                        "embedding_family": emb.family,
                        "foreground_mode": mode,
                        "q_param": sc["q_param"],
                        "prev": sc["prev"],
                        "n_fg": sc["n_fg"],
                        "n_cells": sc["n_cells"],
                        "T_obs": sc["T_obs"],
                        "p_T": sc["p_T"],
                        "q_T": np.nan,
                        "q_T_global": np.nan,
                        "Z_T": sc["Z_T"],
                        "coverage_C": sc["coverage_C"],
                        "peaks_K": sc["peaks_K"],
                        "phi_hat_deg": sc["phi_hat_deg"],
                        "underpowered_flag": sc["underpowered_flag"],
                    }
                )
                done += 1
                if done % 40 == 0 or done == total_biorsp_tests:
                    print(f"[BioRSP] {done}/{total_biorsp_tests} tests")
                    pd.DataFrame(biorsp_rows).to_csv(
                        tables_dir / "biorsp_scores.intermediate.csv", index=False
                    )

    biorsp_df = pd.DataFrame(biorsp_rows)

    # BH for BioRSP.
    q_within = np.full(len(biorsp_df), np.nan, dtype=float)
    for _, idx in biorsp_df.groupby(
        ["embedding", "foreground_mode"], sort=False
    ).groups.items():
        ids = np.asarray(list(idx), dtype=int)
        p = biorsp_df.loc[ids, "p_T"].to_numpy(dtype=float)
        fin = np.isfinite(p)
        if int(fin.sum()) == 0:
            continue
        qq = np.full_like(p, np.nan, dtype=float)
        qq[fin] = bh_fdr(p[fin])
        q_within[ids] = qq
    biorsp_df["q_T"] = q_within

    q_global = np.full(len(biorsp_df), np.nan, dtype=float)
    for _, idx in biorsp_df.groupby(["foreground_mode"], sort=False).groups.items():
        ids = np.asarray(list(idx), dtype=int)
        p = biorsp_df.loc[ids, "p_T"].to_numpy(dtype=float)
        fin = np.isfinite(p)
        if int(fin.sum()) == 0:
            continue
        qq = np.full_like(p, np.nan, dtype=float)
        qq[fin] = bh_fdr(p[fin])
        q_global[ids] = qq
    biorsp_df["q_T_global"] = q_global

    biorsp_df["localized"] = (
        np.isfinite(biorsp_df["q_T"].to_numpy(dtype=float))
        & (biorsp_df["q_T"].to_numpy(dtype=float) <= Q_SIG)
        & (~biorsp_df["underpowered_flag"].to_numpy(dtype=bool))
    )

    biorsp_df.to_csv(tables_dir / "biorsp_scores.csv", index=False)

    # Graph baseline scoring.
    moran_rows: list[dict[str, Any]] = []
    geary_rows: list[dict[str, Any]] = []

    total_graph_tests = len(genes_present) * len(graphs)
    g_done = 0
    for gene_i, st in enumerate(genes_present):
        expr_proc = np.asarray(x_proc[:, int(st.gene_idx)], dtype=float)
        fg_topq = _top_q_mask(
            np.asarray(expr_by_gene[st.gene], dtype=float), q=float(args.q)
        ).astype(float)

        for k_i, (k, w) in enumerate(sorted(graphs.items(), key=lambda kv: kv[0])):
            cont = _score_graph_continuous(
                x=expr_proc,
                w=w,
                n_perm=int(args.n_perm_graph),
                seed=int(args.seed + gene_i * 20000 + k_i * 123 + 7),
            )

            if bool(args.compute_moran_binary):
                bin_out = _score_graph_binary(
                    y=fg_topq,
                    w=w,
                    n_perm=int(args.n_perm_graph),
                    seed=int(args.seed + 500000 + gene_i * 20000 + k_i * 123 + 11),
                )
            else:
                bin_out = {"I_bin": np.nan, "p_I_bin": np.nan, "Z_I_bin": np.nan}

            moran_rows.append(
                {
                    "gene": st.gene,
                    "k": int(k),
                    "I_obs": cont["I_obs"],
                    "p_I": cont["p_I"],
                    "q_I": np.nan,
                    "q_I_global": np.nan,
                    "Z_I": cont["Z_I"],
                    "I_bin": bin_out["I_bin"],
                    "p_I_bin": bin_out["p_I_bin"],
                    "q_I_bin": np.nan,
                    "q_I_bin_global": np.nan,
                    "Z_I_bin": bin_out["Z_I_bin"],
                }
            )

            geary_rows.append(
                {
                    "gene": st.gene,
                    "k": int(k),
                    "C_obs": cont["C_obs"],
                    "A_C": cont["A_C"],
                    "p_C": cont["p_C"],
                    "q_C": np.nan,
                    "q_C_global": np.nan,
                    "Z_C": cont["Z_C"],
                }
            )

            g_done += 1
            if g_done % 20 == 0 or g_done == total_graph_tests:
                print(f"[Graph] {g_done}/{total_graph_tests} gene-k tests")
                pd.DataFrame(moran_rows).to_csv(
                    tables_dir / "moran_scores.intermediate.csv", index=False
                )
                pd.DataFrame(geary_rows).to_csv(
                    tables_dir / "geary_scores.intermediate.csv", index=False
                )

    moran_df = pd.DataFrame(moran_rows)
    geary_df = pd.DataFrame(geary_rows)

    # BH corrections for graph baselines.
    for col_p, col_q, grp in [("p_I", "q_I", ["k"]), ("p_I_bin", "q_I_bin", ["k"])]:
        qarr = np.full(len(moran_df), np.nan, dtype=float)
        for _, idx in moran_df.groupby(grp, sort=False).groups.items():
            ids = np.asarray(list(idx), dtype=int)
            p = moran_df.loc[ids, col_p].to_numpy(dtype=float)
            fin = np.isfinite(p)
            if int(fin.sum()) == 0:
                continue
            qq = np.full_like(p, np.nan, dtype=float)
            qq[fin] = bh_fdr(p[fin])
            qarr[ids] = qq
        moran_df[col_q] = qarr

    # global moran q
    for col_p, col_q in [("p_I", "q_I_global"), ("p_I_bin", "q_I_bin_global")]:
        p = moran_df[col_p].to_numpy(dtype=float)
        fin = np.isfinite(p)
        qq = np.full_like(p, np.nan, dtype=float)
        if int(fin.sum()) > 0:
            qq[fin] = bh_fdr(p[fin])
        moran_df[col_q] = qq

    # geary q by k and global
    qk = np.full(len(geary_df), np.nan, dtype=float)
    for _, idx in geary_df.groupby(["k"], sort=False).groups.items():
        ids = np.asarray(list(idx), dtype=int)
        p = geary_df.loc[ids, "p_C"].to_numpy(dtype=float)
        fin = np.isfinite(p)
        if int(fin.sum()) == 0:
            continue
        qq = np.full_like(p, np.nan, dtype=float)
        qq[fin] = bh_fdr(p[fin])
        qk[ids] = qq
    geary_df["q_C"] = qk

    p = geary_df["p_C"].to_numpy(dtype=float)
    fin = np.isfinite(p)
    qq = np.full_like(p, np.nan, dtype=float)
    if int(fin.sum()) > 0:
        qq[fin] = bh_fdr(p[fin])
    geary_df["q_C_global"] = qq

    moran_df.to_csv(tables_dir / "moran_scores.csv", index=False)
    geary_df.to_csv(tables_dir / "geary_scores.csv", index=False)

    # Method concordance table.
    # Primary BioRSP selection: topq only
    btop = biorsp_df.loc[biorsp_df["foreground_mode"] == "topq"].copy()

    concord_rows: list[dict[str, Any]] = []
    for st in genes_present:
        g = st.gene

        gb = btop.loc[btop["gene"] == g]
        gm = moran_df.loc[moran_df["gene"] == g]
        gc = geary_df.loc[geary_df["gene"] == g]

        if gb.empty:
            continue

        # BioRSP summaries
        z_umap = gb.loc[gb["embedding"] == "umap_repr", "Z_T"].to_numpy(dtype=float)
        q_umap = gb.loc[gb["embedding"] == "umap_repr", "q_T"].to_numpy(dtype=float)
        z_pca = gb.loc[gb["embedding"] == "pca2d", "Z_T"].to_numpy(dtype=float)
        q_pca = gb.loc[gb["embedding"] == "pca2d", "q_T"].to_numpy(dtype=float)
        peaks_umap = gb.loc[gb["embedding"] == "umap_repr", "peaks_K"].to_numpy(
            dtype=float
        )

        z_umap_val = (
            float(z_umap[0]) if z_umap.size > 0 and np.isfinite(z_umap[0]) else np.nan
        )
        q_umap_val = (
            float(q_umap[0]) if q_umap.size > 0 and np.isfinite(q_umap[0]) else np.nan
        )
        z_pca_val = (
            float(z_pca[0]) if z_pca.size > 0 and np.isfinite(z_pca[0]) else np.nan
        )
        q_pca_val = (
            float(q_pca[0]) if q_pca.size > 0 and np.isfinite(q_pca[0]) else np.nan
        )
        peaks_umap_val = (
            float(peaks_umap[0])
            if peaks_umap.size > 0 and np.isfinite(peaks_umap[0])
            else np.nan
        )

        biorsp_localized_any = bool(
            np.any(
                np.isfinite(gb["q_T"].to_numpy(dtype=float))
                & (gb["q_T"].to_numpy(dtype=float) <= Q_SIG)
                & (~gb["underpowered_flag"].to_numpy(dtype=bool))
            )
        )

        # Moran/Geary summaries across k.
        q_i_vals = gm["q_I"].to_numpy(dtype=float)
        q_c_vals = gc["q_C"].to_numpy(dtype=float)
        moran_sig_any = bool(np.any(np.isfinite(q_i_vals) & (q_i_vals <= Q_SIG)))
        geary_sig_any = bool(np.any(np.isfinite(q_c_vals) & (q_c_vals <= Q_SIG)))
        graph_sig_any = bool(moran_sig_any or geary_sig_any)

        z_i = gm["Z_I"].to_numpy(dtype=float)
        z_c = gc["Z_C"].to_numpy(dtype=float)

        med_z_i = float(np.nanmedian(z_i)) if np.isfinite(z_i).any() else np.nan
        iqr_z_i = (
            float(np.nanpercentile(z_i, 75) - np.nanpercentile(z_i, 25))
            if np.isfinite(z_i).any()
            else np.nan
        )
        med_z_c = float(np.nanmedian(z_c)) if np.isfinite(z_c).any() else np.nan
        iqr_z_c = (
            float(np.nanpercentile(z_c, 75) - np.nanpercentile(z_c, 25))
            if np.isfinite(z_c).any()
            else np.nan
        )

        if biorsp_localized_any and graph_sig_any:
            regime = "Both"
        elif biorsp_localized_any and (not graph_sig_any):
            regime = "BioRSP-only"
        elif (not biorsp_localized_any) and graph_sig_any:
            regime = "Graph-only"
        else:
            regime = "None"

        # Diagnostic buckets
        high_i_low_t = bool(
            np.isfinite(med_z_i)
            and med_z_i >= 3.0
            and np.isfinite(z_umap_val)
            and z_umap_val < 3.0
        )
        high_t_low_i = bool(
            np.isfinite(z_umap_val)
            and z_umap_val >= 3.0
            and np.isfinite(med_z_i)
            and med_z_i < 3.0
        )

        concord_rows.append(
            {
                "gene": g,
                "biorsp_Z_umap": z_umap_val,
                "biorsp_q_umap": q_umap_val,
                "biorsp_peaks_umap": peaks_umap_val,
                "biorsp_Z_pca": z_pca_val,
                "biorsp_q_pca": q_pca_val,
                "moran_Z_median": med_z_i,
                "moran_Z_iqr": iqr_z_i,
                "geary_Z_median": med_z_c,
                "geary_Z_iqr": iqr_z_c,
                "biorsp_localized_any": biorsp_localized_any,
                "moran_sig_any": moran_sig_any,
                "geary_sig_any": geary_sig_any,
                "graph_sig_any": graph_sig_any,
                "agreement_regime": regime,
                "high_I_low_T": high_i_low_t,
                "high_T_low_I": high_t_low_i,
            }
        )

    concord_df = pd.DataFrame(concord_rows)
    concord_df.to_csv(tables_dir / "method_concordance.csv", index=False)

    # Correlation matrix (embedding x k)
    corr_rows: list[dict[str, Any]] = []
    for emb in ["pca2d", "umap_repr"]:
        b = btop.loc[btop["embedding"] == emb, ["gene", "Z_T"]].rename(
            columns={"Z_T": "Z_B"}
        )
        for k in sorted(graphs.keys()):
            m = moran_df.loc[moran_df["k"] == int(k), ["gene", "Z_I"]].rename(
                columns={"Z_I": "Z_M"}
            )
            c = geary_df.loc[geary_df["k"] == int(k), ["gene", "Z_C"]].rename(
                columns={"Z_C": "Z_C"}
            )
            bm = b.merge(m, on="gene", how="inner")
            bc = b.merge(c, on="gene", how="inner")
            rho_bm = _safe_spearman(
                bm["Z_B"].to_numpy(dtype=float), bm["Z_M"].to_numpy(dtype=float)
            )
            rho_bc = _safe_spearman(
                bc["Z_B"].to_numpy(dtype=float), bc["Z_C"].to_numpy(dtype=float)
            )
            corr_rows.append(
                {
                    "embedding": emb,
                    "k": int(k),
                    "rho_biorsp_vs_moran": rho_bm,
                    "rho_biorsp_vs_geary": rho_bc,
                    "n_genes_moran": int(len(bm)),
                    "n_genes_geary": int(len(bc)),
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(tables_dir / "correlation_matrix.csv", index=False)

    # Exemplar selection.
    ex_rows: list[dict[str, Any]] = []
    if not concord_df.empty:
        both = concord_df.loc[concord_df["agreement_regime"] == "Both"].copy()
        both["joint_score"] = both["biorsp_Z_umap"].fillna(0.0) + both[
            "moran_Z_median"
        ].fillna(0.0)
        both = both.sort_values(by="joint_score", ascending=False).head(3)
        for rank, (_, r) in enumerate(both.iterrows(), start=1):
            ex_rows.append(
                {
                    "gene": r["gene"],
                    "group": "Both",
                    "rank": rank,
                    "score": float(r["joint_score"]),
                }
            )

        bo = concord_df.loc[concord_df["agreement_regime"] == "BioRSP-only"].copy()
        bo = bo.sort_values(by="biorsp_Z_umap", ascending=False).head(3)
        for rank, (_, r) in enumerate(bo.iterrows(), start=1):
            ex_rows.append(
                {
                    "gene": r["gene"],
                    "group": "BioRSP-only",
                    "rank": rank,
                    "score": (
                        float(r["biorsp_Z_umap"])
                        if np.isfinite(r["biorsp_Z_umap"])
                        else np.nan
                    ),
                }
            )

        go = concord_df.loc[concord_df["agreement_regime"] == "Graph-only"].copy()
        go["graph_score"] = np.maximum(
            go["moran_Z_median"].fillna(-np.inf), go["geary_Z_median"].fillna(-np.inf)
        )
        go = go.sort_values(by="graph_score", ascending=False).head(3)
        for rank, (_, r) in enumerate(go.iterrows(), start=1):
            ex_rows.append(
                {
                    "gene": r["gene"],
                    "group": "Graph-only",
                    "rank": rank,
                    "score": (
                        float(r["graph_score"])
                        if np.isfinite(r["graph_score"])
                        else np.nan
                    ),
                }
            )

    exemplar_df = pd.DataFrame(ex_rows)
    if not exemplar_df.empty:
        exemplar_df = exemplar_df.drop_duplicates(subset=["gene"], keep="first")
    exemplar_df.to_csv(tables_dir / "exemplar_genes.csv", index=False)

    # ============================
    # Plots 00: overview
    # ============================
    umap_xy = emb_map["umap_repr"].coords
    fig0, ax0 = plt.subplots(figsize=(6.0, 5.3))
    ax0.scatter(
        umap_xy[:, 0],
        umap_xy[:, 1],
        s=5,
        c="#4c78a8",
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("CM subset UMAP geometry")
    fig0.tight_layout()
    fig0.savefig(
        plots_dir / "00_overview" / "cm_umap_geometry.png", dpi=DEFAULT_PLOT_STYLE.dpi
    )
    plt.close(fig0)

    fig1, axes1 = plt.subplots(1, len(graphs), figsize=(5.2 * len(graphs), 4.2))
    axes1_arr = np.atleast_1d(axes1).ravel()
    for i, k in enumerate(sorted(graphs.keys())):
        ax = axes1_arr[i]
        deg = degrees_by_k[int(k)]
        bins = int(min(50, max(10, np.ceil(np.sqrt(deg.size)))))
        ax.hist(deg, bins=bins, color="#729FCF", alpha=0.9, edgecolor="white")
        ax.set_title(f"k={int(k)} degree")
        ax.set_xlabel("degree")
        ax.set_ylabel("count")
    fig1.tight_layout()
    fig1.savefig(
        plots_dir / "00_overview" / "knn_degree_distributions.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
    )
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9.8, 3.6))
    ax2.axis("off")
    table_rows = [
        [
            "BioRSP",
            int(args.n_perm_biorsp),
            int(args.n_bins),
            f"q={float(args.q):.2f}",
            "topq",
        ],
        [
            "Moran/Geary",
            int(args.n_perm_graph),
            "graph",
            f"k={','.join(map(str, sorted(graphs.keys())))}",
            "continuous",
        ],
        [
            "Moran binary",
            int(args.n_perm_graph),
            "graph",
            f"k={','.join(map(str, sorted(graphs.keys())))}",
            str(bool(args.compute_moran_binary)),
        ],
    ]
    tbl = ax2.table(
        cellText=table_rows,
        colLabels=["Method", "n_perm", "bins/type", "neighborhood", "mode"],
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.25)
    ax2.set_title(f"Method parameters (n_cells={int(adata_cm.n_obs)})")
    fig2.tight_layout()
    fig2.savefig(
        plots_dir / "00_overview" / "method_parameter_table.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
    )
    plt.close(fig2)

    # ============================
    # Plots 01: method strength scatter
    # ============================
    k_ref = 30 if 30 in graphs else int(sorted(graphs.keys())[0])
    b_um = btop.loc[
        btop["embedding"] == "umap_repr", ["gene", "Z_T", "peaks_K"]
    ].rename(columns={"Z_T": "Z_B", "peaks_K": "peaks_K"})
    m_ref = moran_df.loc[moran_df["k"] == int(k_ref), ["gene", "Z_I"]]
    g_ref = geary_df.loc[geary_df["k"] == int(k_ref), ["gene", "Z_C"]]
    bm = b_um.merge(m_ref, on="gene", how="inner")
    bg = b_um.merge(g_ref, on="gene", how="inner")

    fig3, axes3 = plt.subplots(1, 3, figsize=(16.5, 5.2))

    ax = axes3[0]
    if not bm.empty:
        ax.scatter(
            bm["Z_B"],
            bm["Z_I"],
            s=90,
            c="#4c78a8",
            edgecolors="black",
            linewidths=0.4,
            alpha=0.9,
        )
        for _, r in bm.iterrows():
            ax.text(float(r["Z_B"]), float(r["Z_I"]) + 0.02, str(r["gene"]), fontsize=8)
        ax.set_xlabel("BioRSP Z_T (UMAP)")
        ax.set_ylabel(f"Moran Z_I (k={k_ref})")
    else:
        ax.axis("off")
    ax.set_title("BioRSP vs Moran")

    ax = axes3[1]
    if not bg.empty:
        ax.scatter(
            bg["Z_B"],
            bg["Z_C"],
            s=90,
            c="#ff7f0e",
            edgecolors="black",
            linewidths=0.4,
            alpha=0.9,
        )
        for _, r in bg.iterrows():
            ax.text(float(r["Z_B"]), float(r["Z_C"]) + 0.02, str(r["gene"]), fontsize=8)
        ax.set_xlabel("BioRSP Z_T (UMAP)")
        ax.set_ylabel(f"Geary Z_C (k={k_ref})")
    else:
        ax.axis("off")
    ax.set_title("BioRSP vs Geary")

    ax = axes3[2]
    if not bm.empty:
        sc = ax.scatter(
            bm["Z_B"],
            bm["Z_I"],
            c=bm["peaks_K"],
            cmap="viridis",
            s=110,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.9,
        )
        for _, r in bm.iterrows():
            ax.text(float(r["Z_B"]), float(r["Z_I"]) + 0.02, str(r["gene"]), fontsize=8)
        ax.set_xlabel("BioRSP Z_T (UMAP)")
        ax.set_ylabel(f"Moran Z_I (k={k_ref})")
        fig3.colorbar(sc, ax=ax, shrink=0.85, label="peaks_K")
    else:
        ax.axis("off")
    ax.set_title("BioRSP vs Moran colored by peaks_K")

    fig3.tight_layout()
    fig3.savefig(
        plots_dir / "01_method_strength_scatter" / "strength_scatter_triptych.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
    )
    plt.close(fig3)

    # ============================
    # Plots 02: agreement regimes
    # ============================
    fig4, ax4 = plt.subplots(figsize=(7.0, 5.2))
    reg_order = ["Both", "BioRSP-only", "Graph-only", "None"]
    reg_counts = (
        {k: int(np.sum(concord_df["agreement_regime"] == k)) for k in reg_order}
        if not concord_df.empty
        else {k: 0 for k in reg_order}
    )
    ax4.bar(
        np.arange(len(reg_order)),
        [reg_counts[k] for k in reg_order],
        color=["#2ca02c", "#1f77b4", "#ff7f0e", "#9e9e9e"],
    )
    ax4.set_xticks(np.arange(len(reg_order)))
    ax4.set_xticklabels(reg_order, rotation=25, ha="right")
    ax4.set_ylabel("# genes")
    ax4.set_title("Agreement regimes")
    fig4.tight_layout()
    fig4.savefig(
        plots_dir / "02_agreement_regimes" / "agreement_regime_counts.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
    )
    plt.close(fig4)

    fig5, axes5 = plt.subplots(2, 2, figsize=(12.0, 9.0))
    ax_arr = axes5.ravel()
    for i, reg in enumerate(reg_order):
        ax = ax_arr[i]
        sub = concord_df.loc[concord_df["agreement_regime"] == reg]
        if sub.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{reg}: none", ha="center", va="center")
            continue
        ax.scatter(
            sub["biorsp_Z_umap"],
            sub["moran_Z_median"],
            s=100,
            c="#4c78a8",
            alpha=0.88,
            edgecolors="black",
            linewidths=0.4,
        )
        for _, r in sub.iterrows():
            ax.text(
                float(r["biorsp_Z_umap"]),
                float(r["moran_Z_median"]) + 0.02,
                str(r["gene"]),
                fontsize=8,
            )
        ax.set_xlabel("BioRSP Z_T (UMAP)")
        ax.set_ylabel("Moran median Z_I")
        ax.set_title(reg)
    fig5.tight_layout()
    fig5.savefig(
        plots_dir / "02_agreement_regimes" / "regime_faceted_scatter.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
    )
    plt.close(fig5)

    # heatmap genes x methods of -log10(q)
    methods = (
        ["BioRSP-UMAP", "BioRSP-PCA"]
        + [f"Moran-k{k}" for k in sorted(graphs.keys())]
        + [f"Geary-k{k}" for k in sorted(graphs.keys())]
    )
    genes = concord_df["gene"].astype(str).tolist() if not concord_df.empty else []
    mat_q = np.full((len(genes), len(methods)), np.nan, dtype=float)
    for i, g in enumerate(genes):
        bu = btop.loc[
            (btop["gene"] == g) & (btop["embedding"] == "umap_repr"), "q_T"
        ].to_numpy(dtype=float)
        bp = btop.loc[
            (btop["gene"] == g) & (btop["embedding"] == "pca2d"), "q_T"
        ].to_numpy(dtype=float)
        mat_q[i, 0] = bu[0] if bu.size > 0 else np.nan
        mat_q[i, 1] = bp[0] if bp.size > 0 else np.nan
        for j, k in enumerate(sorted(graphs.keys())):
            q_i = moran_df.loc[
                (moran_df["gene"] == g) & (moran_df["k"] == int(k)), "q_I"
            ].to_numpy(dtype=float)
            q_c = geary_df.loc[
                (geary_df["gene"] == g) & (geary_df["k"] == int(k)), "q_C"
            ].to_numpy(dtype=float)
            mat_q[i, 2 + j] = q_i[0] if q_i.size > 0 else np.nan
            mat_q[i, 2 + len(graphs) + j] = q_c[0] if q_c.size > 0 else np.nan

    if len(genes) > 0:
        mat_plot = -np.log10(np.clip(mat_q, 1e-300, 1.0))
        mat_plot = np.nan_to_num(mat_plot, nan=0.0)
        mat_plot = np.clip(mat_plot, 0.0, 8.0)

        fig6, ax6 = plt.subplots(
            figsize=(0.9 * len(methods) + 3.0, 0.5 * len(genes) + 2.2)
        )
        im6 = ax6.imshow(mat_plot, aspect="auto", cmap="magma", vmin=0, vmax=8)
        ax6.set_xticks(np.arange(len(methods)))
        ax6.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax6.set_yticks(np.arange(len(genes)))
        ax6.set_yticklabels(genes, fontsize=8)
        ax6.set_title("-log10(q) across methods")
        fig6.colorbar(im6, ax=ax6)
        fig6.tight_layout()
        fig6.savefig(
            plots_dir / "02_agreement_regimes" / "gene_method_q_heatmap.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
        )
        plt.close(fig6)
    else:
        _save_placeholder(
            plots_dir / "02_agreement_regimes" / "gene_method_q_heatmap.png",
            "Method heatmap",
            "No genes",
        )

    # ============================
    # Plots 03: rank concordance
    # ============================
    embs = ["pca2d", "umap_repr"]
    ks = sorted(graphs.keys())

    mat_m = np.full((len(embs), len(ks)), np.nan, dtype=float)
    mat_g = np.full((len(embs), len(ks)), np.nan, dtype=float)
    for i, emb in enumerate(embs):
        b = btop.loc[btop["embedding"] == emb, ["gene", "Z_T"]].rename(
            columns={"Z_T": "Z_B"}
        )
        for j, k in enumerate(ks):
            m = moran_df.loc[moran_df["k"] == int(k), ["gene", "Z_I"]].rename(
                columns={"Z_I": "Z_M"}
            )
            c = geary_df.loc[geary_df["k"] == int(k), ["gene", "Z_C"]].rename(
                columns={"Z_C": "Z_C"}
            )
            bm = b.merge(m, on="gene", how="inner")
            bc = b.merge(c, on="gene", how="inner")
            mat_m[i, j] = _safe_spearman(
                bm["Z_B"].to_numpy(dtype=float), bm["Z_M"].to_numpy(dtype=float)
            )
            mat_g[i, j] = _safe_spearman(
                bc["Z_B"].to_numpy(dtype=float), bc["Z_C"].to_numpy(dtype=float)
            )

    def _plot_corr_heat(mat: np.ndarray, title: str, out: Path) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(ks)))
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_yticks(np.arange(len(embs)))
        ax.set_yticklabels(embs)
        ax.set_xlabel("k")
        ax.set_ylabel("BioRSP embedding")
        ax.set_title(title)
        for ii in range(mat.shape[0]):
            for jj in range(mat.shape[1]):
                v = mat[ii, jj]
                txt = "nan" if not np.isfinite(v) else f"{v:.2f}"
                ax.text(jj, ii, txt, ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out, dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig)

    _plot_corr_heat(
        mat_m,
        "Spearman: BioRSP Z_T vs Moran Z_I",
        plots_dir / "03_rank_concordance" / "corr_heatmap_moran.png",
    )
    _plot_corr_heat(
        mat_g,
        "Spearman: BioRSP Z_T vs Geary Z_C",
        plots_dir / "03_rank_concordance" / "corr_heatmap_geary.png",
    )

    # boxplot per-k distribution of correlations (across embeddings) for Moran/Geary
    fig7, ax7 = plt.subplots(figsize=(8.0, 5.2))
    data = []
    labels = []
    for j, k in enumerate(ks):
        vals_m = mat_m[:, j]
        vals_m = vals_m[np.isfinite(vals_m)]
        vals_g = mat_g[:, j]
        vals_g = vals_g[np.isfinite(vals_g)]
        if vals_m.size > 0:
            data.append(vals_m)
            labels.append(f"Moran-k{k}")
        if vals_g.size > 0:
            data.append(vals_g)
            labels.append(f"Geary-k{k}")
    if len(data) > 0:
        ax7.boxplot(data, tick_labels=labels, patch_artist=True)
        ax7.tick_params(axis="x", rotation=45)
        ax7.set_ylabel("Spearman correlation")
    else:
        ax7.axis("off")
        ax7.text(0.5, 0.5, "No correlation values", ha="center", va="center")
    ax7.set_title("Correlation distributions by k and baseline")
    fig7.tight_layout()
    fig7.savefig(
        plots_dir / "03_rank_concordance" / "corr_boxplots_by_k.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
    )
    plt.close(fig7)

    # ============================
    # Plots 04: exemplar panels
    # ============================
    if exemplar_df.empty:
        _save_placeholder(
            plots_dir / "04_exemplar_panels" / "no_exemplars.png",
            "Exemplars",
            "No exemplars selected",
        )
    else:
        w30 = graphs[k_ref]
        w30_row = _row_standardize(w30)

        for _, ex in exemplar_df.iterrows():
            gene = str(ex["gene"])
            grp = str(ex["group"])
            if gene not in expr_by_gene:
                continue

            expr_raw = np.asarray(expr_by_gene[gene], dtype=float)
            expr_log = np.log1p(np.maximum(expr_raw, 0.0))

            # Recompute UMAP BioRSP with profiles for panel.
            geom_u = _compute_geom(emb_map["umap_repr"], int(args.n_bins))
            sc_u = _score_biorsp(
                expr=expr_raw,
                geom=geom_u,
                mode="topq",
                q=float(args.q),
                n_bins=int(args.n_bins),
                n_perm=int(args.n_perm_biorsp),
                seed=int(
                    args.seed
                    + 900000
                    + int(np.sum(np.frombuffer(gene.encode("utf-8"), dtype=np.uint8)))
                ),
                with_profiles=True,
            )

            x_cont = np.asarray(
                x_proc[
                    :, int([st.gene_idx for st in genes_present if st.gene == gene][0])
                ],
                dtype=float,
            )
            wx = np.asarray(w30_row.dot(x_cont)).ravel()
            rho_lag = _safe_spearman(x_cont, wx)

            # summary rows for k-sensitivity
            mk = moran_df.loc[moran_df["gene"] == gene].sort_values(by="k")
            gk = geary_df.loc[geary_df["gene"] == gene].sort_values(by="k")

            fig8 = plt.figure(figsize=(15.8, 9.4))
            gs = fig8.add_gridspec(2, 3, wspace=0.28, hspace=0.30)

            # A) UMAP feature
            axA = fig8.add_subplot(gs[0, 0])
            ord_idx = np.argsort(expr_log, kind="mergesort")
            axA.scatter(
                umap_xy[:, 0],
                umap_xy[:, 1],
                c="#dddddd",
                s=4,
                alpha=0.25,
                linewidths=0,
                rasterized=True,
            )
            scA = axA.scatter(
                umap_xy[ord_idx, 0],
                umap_xy[ord_idx, 1],
                c=expr_log[ord_idx],
                cmap="Reds",
                s=8,
                alpha=0.9,
                linewidths=0,
                rasterized=True,
            )
            axA.set_xticks([])
            axA.set_yticks([])
            axA.set_title(f"{gene} feature on UMAP")
            fig8.colorbar(scA, ax=axA, fraction=0.046, pad=0.03)

            # B) BioRSP polar
            axB = fig8.add_subplot(gs[0, 1], projection="polar")
            stats_txt = (
                f"Z={float(sc_u['Z_T']):.2f}\n"
                f"p={float(sc_u['p_T']):.2e}\n"
                f"C={float(sc_u['coverage_C']):.3f}\n"
                f"K={int(sc_u['peaks_K']) if np.isfinite(sc_u['peaks_K']) else -1}\n"
                f"phi={float(sc_u['phi_hat_deg']):.1f}"
            )
            _plot_polar(
                axB,
                np.asarray(sc_u["E_obs"], dtype=float),
                (
                    np.asarray(sc_u["null_E"], dtype=float)
                    if sc_u["null_E"] is not None
                    else None
                ),
                "BioRSP polar (UMAP)",
                stats_txt,
            )

            # C) Moran scatter x vs W x
            axC = fig8.add_subplot(gs[0, 2])
            axC.scatter(
                x_cont,
                wx,
                s=12,
                alpha=0.45,
                color="#4c78a8",
                edgecolors="none",
                rasterized=True,
            )
            if np.isfinite(rho_lag):
                axC.set_title(f"Moran scatter (k={k_ref})\nSpearman={rho_lag:.2f}")
            else:
                axC.set_title(f"Moran scatter (k={k_ref})")
            axC.set_xlabel("x")
            axC.set_ylabel("W x")

            # D) Geary summary text
            axD = fig8.add_subplot(gs[1, 0])
            axD.axis("off")
            g30 = gk.loc[gk["k"] == int(k_ref)]
            if not g30.empty:
                r = g30.iloc[0]
                txt = (
                    f"Geary summary (k={k_ref})\n"
                    f"C_obs={float(r['C_obs']):.4f}\n"
                    f"A_C=1-C={float(r['A_C']):.4f}\n"
                    f"Z_C={float(r['Z_C']):.2f}\n"
                    f"q_C={float(r['q_C']):.2e}"
                )
            else:
                txt = f"No Geary row for k={k_ref}"
            axD.text(0.05, 0.95, txt, ha="left", va="top", fontsize=10)
            axD.set_title("Geary baseline")

            # E) k-sensitivity mini-plot
            axE = fig8.add_subplot(gs[1, 1:3])
            if not mk.empty:
                axE.plot(
                    mk["k"], mk["Z_I"], marker="o", color="#1f77b4", label="Moran Z_I"
                )
            if not gk.empty:
                axE.plot(
                    gk["k"], gk["Z_C"], marker="s", color="#ff7f0e", label="Geary Z_C"
                )
            axE.axhline(0.0, color="#666", linewidth=0.9)
            axE.set_xlabel("k")
            axE.set_ylabel("Z baseline")
            axE.set_title("k-sensitivity")
            axE.legend(loc="best")

            fig8.suptitle(f"Exemplar: {gene} [{grp}]", y=0.995)
            fig8.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
            fig8.savefig(
                plots_dir / "04_exemplar_panels" / f"exemplar_{gene}.png",
                dpi=DEFAULT_PLOT_STYLE.dpi,
            )
            plt.close(fig8)

    # ============================
    # Plots 05: k sensitivity
    # ============================
    fig9, ax9 = plt.subplots(figsize=(8.2, 5.8))
    for st in genes_present:
        sub = moran_df.loc[moran_df["gene"] == st.gene].sort_values(by="k")
        if sub.empty:
            continue
        ax9.plot(sub["k"], sub["Z_I"], marker="o", alpha=0.8, label=st.gene)
    ax9.axhline(0.0, color="#666", linewidth=0.9)
    ax9.set_xlabel("k")
    ax9.set_ylabel("Moran Z_I")
    ax9.set_title("Moran Z_I vs k by gene")
    ax9.legend(loc="best", fontsize=7, ncol=2)
    fig9.tight_layout()
    fig9.savefig(
        plots_dir / "05_k_sensitivity" / "moran_Z_vs_k_lines.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
    )
    plt.close(fig9)

    fig10, ax10 = plt.subplots(figsize=(8.2, 5.8))
    for st in genes_present:
        sub = geary_df.loc[geary_df["gene"] == st.gene].sort_values(by="k")
        if sub.empty:
            continue
        ax10.plot(sub["k"], sub["Z_C"], marker="s", alpha=0.8, label=st.gene)
    ax10.axhline(0.0, color="#666", linewidth=0.9)
    ax10.set_xlabel("k")
    ax10.set_ylabel("Geary Z_C")
    ax10.set_title("Geary Z_C vs k by gene")
    ax10.legend(loc="best", fontsize=7, ncol=2)
    fig10.tight_layout()
    fig10.savefig(
        plots_dir / "05_k_sensitivity" / "geary_Z_vs_k_lines.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
    )
    plt.close(fig10)

    # IQR vs median scatter
    sens_rows = []
    for st in genes_present:
        z = moran_df.loc[moran_df["gene"] == st.gene, "Z_I"].to_numpy(dtype=float)
        z = z[np.isfinite(z)]
        if z.size == 0:
            continue
        sens_rows.append(
            {
                "gene": st.gene,
                "median_Z_I": float(np.median(z)),
                "iqr_Z_I": float(np.percentile(z, 75) - np.percentile(z, 25)),
            }
        )
    sens_df = pd.DataFrame(sens_rows)
    fig11, ax11 = plt.subplots(figsize=(7.2, 5.8))
    if not sens_df.empty:
        ax11.scatter(
            sens_df["median_Z_I"],
            sens_df["iqr_Z_I"],
            s=95,
            c="#4c78a8",
            alpha=0.9,
            edgecolors="black",
            linewidths=0.4,
        )
        for _, r in sens_df.iterrows():
            ax11.text(
                float(r["median_Z_I"]),
                float(r["iqr_Z_I"]) + 0.02,
                str(r["gene"]),
                fontsize=8,
            )
        ax11.set_xlabel("median_k(Z_I)")
        ax11.set_ylabel("IQR_k(Z_I)")
    else:
        ax11.axis("off")
    ax11.set_title("k-sensitivity diagnostic (Moran)")
    fig11.tight_layout()
    fig11.savefig(
        plots_dir / "05_k_sensitivity" / "iqr_vs_median_moran.png",
        dpi=DEFAULT_PLOT_STYLE.dpi,
    )
    plt.close(fig11)

    # README
    _write_readme(
        out_root / "README.txt",
        args=args,
        donor_key=donor_key,
        label_key=label_key,
        donor_star=donor_star,
        expr_source=expr_source,
        prep_note=f"{prep_note}; n_pcs_used={n_pcs}",
        n_cells_donor=int(adata_donor.n_obs),
        n_cells_cm=int(adata_cm.n_obs),
        cm_label_counts={str(k): int(v) for k, v in cm_label_counts.items()},
        n_genes_present=len(genes_present),
        warnings_log=warnings_log,
    )

    # Verify required outputs.
    required = [
        tables_dir / "donor_choice.csv",
        tables_dir / "gene_panel_status.csv",
        tables_dir / "biorsp_scores.csv",
        tables_dir / "moran_scores.csv",
        tables_dir / "geary_scores.csv",
        tables_dir / "method_concordance.csv",
        tables_dir / "correlation_matrix.csv",
        tables_dir / "exemplar_genes.csv",
        plots_dir / "00_overview",
        plots_dir / "01_method_strength_scatter",
        plots_dir / "02_agreement_regimes",
        plots_dir / "03_rank_concordance",
        plots_dir / "04_exemplar_panels",
        plots_dir / "05_k_sensitivity",
        out_root / "README.txt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing required outputs: " + ", ".join(missing))

    print(f"donor_key_used={donor_key}")
    print(f"label_key_used={label_key}")
    print(f"donor_star={donor_star}")
    print(f"expression_source_used={expr_source}")
    print(f"knn_list={','.join(map(str, sorted(graphs.keys())))}")
    print(f"n_genes_present={len(genes_present)}")
    print(f"n_biorsp_rows={len(biorsp_df)}")
    print(f"n_moran_rows={len(moran_df)}")
    print(f"n_geary_rows={len(geary_df)}")
    if not concord_df.empty:
        print(
            "agreement_counts="
            + json.dumps(
                concord_df["agreement_regime"].value_counts().to_dict(), sort_keys=True
            )
        )
    print(f"results_root={out_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
