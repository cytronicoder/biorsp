#!/usr/bin/env python3
"""Experiment #6: donor mixing / batch-geometry stress test for heart embedding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

# Force headless backend for deterministic local/CI figure generation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors

# Allow direct script execution from repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biorsp.core.geometry import (
    bin_theta,
    compute_theta,
    compute_vantage_point,
    theta_bin_centers,
)
from biorsp.pipeline.hierarchy import _ensure_umap, _resolve_expr_matrix
from biorsp.plotting.qc import plot_categorical_umap
from biorsp.plotting.styles import DEFAULT_PLOT_STYLE, apply_plot_style
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

Q_SIG = 0.05
Z_STRONG = 4.0
Z_MODERATE = 3.0
COVERAGE_STRONG = 0.20
POOR_MIXING_SCALED = 0.30
POOR_MIXING_ILISI = 2.0


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
        description="Run Experiment #6 donor mixing / batch-geometry stress test."
    )
    p.add_argument(
        "--h5ad",
        default="data/processed/HT_pca_umap.h5ad",
        help="Input .h5ad path.",
    )
    p.add_argument(
        "--out",
        default="experiments/heart_smoketest/results/experiment6_donor_mixing_stress_test",
        help="Output directory.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--n_perm", type=int, default=300, help="Permutation count.")
    p.add_argument("--n_bins", type=int, default=64, help="Angular bin count.")
    p.add_argument("--k", type=int, default=30, help="kNN neighborhood size.")
    p.add_argument(
        "--top_donors", type=int, default=6, help="Top donor panels to render by Z_T."
    )
    p.add_argument(
        "--top_celltypes",
        type=int,
        default=5,
        help="Top cell types by abundance for within-celltype diagnostics.",
    )
    p.add_argument(
        "--save_per_cell",
        type=_str2bool,
        default=False,
        help="Write per_cell_mixing_metrics.csv (can be large).",
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
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
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
            raise KeyError(f"Requested embedding key '{requested_key}' missing.")
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
    # Metadata only in this experiment; we still resolve and report source consistently.
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


def _resolve_donor_ids_required(
    adata: ad.AnnData,
    requested_key: str | None,
) -> tuple[np.ndarray, str]:
    key = _resolve_key(adata, requested_key, DONOR_CANDIDATES)
    if key is None:
        raise RuntimeError(
            "Experiment #6 requires donor IDs, but no donor key was found in adata.obs."
        )
    donor_ids = adata.obs[key].astype("string").fillna("NA").astype(str).to_numpy()
    uniq = np.unique(donor_ids)
    if uniq.size < 2:
        raise RuntimeError(
            f"Experiment #6 requires >=2 donors; found {int(uniq.size)} using key '{key}'."
        )
    return donor_ids, key


def _resolve_label_values(
    adata: ad.AnnData,
    requested_key: str | None,
) -> tuple[np.ndarray | None, str | None]:
    key = _resolve_key(adata, requested_key, LABEL_KEY_CANDIDATES)
    if key is None:
        return None, None
    labels = adata.obs[key].astype("string").fillna("NA").astype(str).to_numpy()
    return labels, key


def _collapse_rare_labels(
    labels: np.ndarray,
    *,
    min_fraction: float = 0.01,
) -> tuple[np.ndarray, list[str]]:
    s = pd.Series(np.asarray(labels).astype(str))
    freq = s.value_counts(normalize=True, dropna=False)
    rare = sorted(freq.loc[freq < float(min_fraction)].index.astype(str).tolist())
    if len(rare) == 0:
        return s.to_numpy(dtype=str), rare
    collapsed = s.where(~s.isin(rare), "Other").astype(str).to_numpy()
    return collapsed, rare


def _build_shuffle_blocks(group_labels: np.ndarray) -> list[np.ndarray]:
    groups = pd.Categorical(np.asarray(group_labels).astype(str))
    codes = groups.codes.astype(int)
    blocks: list[np.ndarray] = []
    for g in range(int(len(groups.categories))):
        idx = np.flatnonzero(codes == g).astype(int)
        if idx.size > 0:
            blocks.append(idx)
    return blocks


def _permute_codes_within_blocks(
    base_codes: np.ndarray,
    blocks: list[np.ndarray],
    rng: np.random.Generator,
    out_codes: np.ndarray,
) -> None:
    out_codes[:] = base_codes
    for idx in blocks:
        if idx.size <= 1:
            continue
        out_codes[idx] = base_codes[idx][rng.permutation(idx.size)]


def _profiles_by_donor(
    donor_codes: np.ndarray,
    *,
    donor_counts: np.ndarray,
    bin_id: np.ndarray,
    n_donors: int,
    n_bins: int,
    p_bg: np.ndarray,
    counts_buf: np.ndarray | None = None,
) -> np.ndarray:
    if counts_buf is None:
        counts = np.zeros((int(n_donors), int(n_bins)), dtype=float)
    else:
        counts = counts_buf
        counts.fill(0.0)
    np.add.at(counts, (donor_codes, bin_id), 1.0)
    return counts / np.maximum(donor_counts[:, None], 1.0) - p_bg[None, :]


def _analyze_donor_directionality(
    *,
    xy: np.ndarray,
    donor_ids: np.ndarray,
    n_bins: int,
    n_perm: int,
    seed: int,
    center_xy: np.ndarray | None,
    null_group_labels: np.ndarray | None,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    xy_arr = np.asarray(xy, dtype=float)
    donors = np.asarray(donor_ids).astype(str)
    n_cells = int(xy_arr.shape[0])
    if n_cells != int(donors.size):
        raise ValueError("xy and donor_ids length mismatch.")
    if n_cells < 3:
        raise ValueError("Need at least 3 cells for donor directionality analysis.")

    donor_cat = pd.Categorical(donors, categories=sorted(pd.Index(donors).unique().tolist()))
    donor_levels = donor_cat.categories.astype(str).tolist()
    donor_codes = donor_cat.codes.astype(int)
    n_donors = int(len(donor_levels))
    if n_donors < 2:
        raise ValueError("Need >=2 donors.")

    if center_xy is None:
        center = compute_vantage_point(xy_arr, method="median")
    else:
        center = np.asarray(center_xy, dtype=float).ravel()[:2]
    theta = compute_theta(xy_arr, center)
    _, bin_id = bin_theta(theta, int(n_bins))
    bin_counts_total = np.bincount(bin_id, minlength=int(n_bins)).astype(float)
    p_bg = bin_counts_total / float(n_cells)

    donor_counts = np.bincount(donor_codes, minlength=n_donors).astype(float)
    obs_counts = np.zeros((n_donors, int(n_bins)), dtype=float)
    obs_profiles = _profiles_by_donor(
        donor_codes,
        donor_counts=donor_counts,
        bin_id=bin_id,
        n_donors=n_donors,
        n_bins=int(n_bins),
        p_bg=p_bg,
        counts_buf=obs_counts,
    )
    t_obs = np.max(np.abs(obs_profiles), axis=1)
    phi_idx = np.argmax(np.abs(obs_profiles), axis=1)
    theta_centers = theta_bin_centers(int(n_bins))
    phi_rad = theta_centers[phi_idx]
    phi_deg = np.degrees(phi_rad) % 360.0

    use_blocked_null = bool(null_group_labels is not None)
    if use_blocked_null:
        blocks = _build_shuffle_blocks(np.asarray(null_group_labels))
        null_mode = "within_celltype_label_permutation"
    else:
        blocks = []
        null_mode = "global_label_permutation"

    rng = np.random.default_rng(int(seed))
    perm_codes = np.empty_like(donor_codes)
    perm_counts = np.zeros((n_donors, int(n_bins)), dtype=float)
    null_e = np.zeros((n_donors, int(n_perm), int(n_bins)), dtype=float)
    null_t = np.zeros((n_donors, int(n_perm)), dtype=float)
    for i in range(int(n_perm)):
        if use_blocked_null:
            _permute_codes_within_blocks(donor_codes, blocks, rng, perm_codes)
        else:
            perm_codes[:] = donor_codes[rng.permutation(n_cells)]
        prof = _profiles_by_donor(
            perm_codes,
            donor_counts=donor_counts,
            bin_id=bin_id,
            n_donors=n_donors,
            n_bins=int(n_bins),
            p_bg=p_bg,
            counts_buf=perm_counts,
        )
        null_e[:, i, :] = prof
        null_t[:, i] = np.max(np.abs(prof), axis=1)

    p_t = (1.0 + np.sum(null_t >= t_obs[:, None], axis=1)) / (1.0 + float(int(n_perm)))
    q_t = bh_fdr(np.asarray(p_t, dtype=float))

    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    for d_idx, donor_id in enumerate(donor_levels):
        e_d = np.asarray(obs_profiles[d_idx], dtype=float)
        ne_d = np.asarray(null_e[d_idx], dtype=float)
        nt_d = np.asarray(null_t[d_idx], dtype=float)
        z_d = float(robust_z(float(t_obs[d_idx]), nt_d))
        c_d = float(coverage_from_null(e_d, ne_d, q=0.95))
        k_d = int(peak_count(e_d, ne_d, smooth_w=3, q_prom=0.95))
        rows.append(
            {
                "donor_id": donor_id,
                "n_cells": int(donor_counts[d_idx]),
                "T_obs": float(t_obs[d_idx]),
                "p_T": float(p_t[d_idx]),
                "q_T": float(q_t[d_idx]),
                "Z_T": z_d,
                "coverage_C": c_d,
                "peaks_K": k_d,
                "phi_hat_rad": float(phi_rad[d_idx]),
                "phi_hat_deg": float(phi_deg[d_idx]),
                "null_mode": null_mode,
            }
        )
        artifacts[donor_id] = {
            "E_phi_obs": e_d,
            "null_E_phi": ne_d,
            "null_T": nt_d,
            "T_obs": np.asarray([float(t_obs[d_idx])], dtype=float),
            "xy": xy_arr,
            "center": np.asarray(center, dtype=float),
            "mask": (donor_codes == d_idx),
        }

    out_df = pd.DataFrame(rows).sort_values(
        by=["q_T", "Z_T"], ascending=[True, False], kind="mergesort"
    )
    meta = {
        "n_cells": int(n_cells),
        "n_donors": int(n_donors),
        "n_bins": int(n_bins),
        "n_perm": int(n_perm),
        "null_mode": null_mode,
        "center_xy": np.asarray(center, dtype=float),
    }
    return out_df.reset_index(drop=True), artifacts, meta


def _compute_mixing_metrics(
    *,
    xy: np.ndarray,
    donor_ids: np.ndarray,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    xy_arr = np.asarray(xy, dtype=float)
    donors = np.asarray(donor_ids).astype(str)
    n = int(xy_arr.shape[0])
    if n != int(donors.size):
        raise ValueError("xy and donor_ids length mismatch.")
    if n < 3:
        raise ValueError("Need at least 3 cells for neighborhood mixing metrics.")

    donor_cat = pd.Categorical(donors, categories=sorted(pd.Index(donors).unique().tolist()))
    donor_levels = donor_cat.categories.astype(str).tolist()
    donor_codes = donor_cat.codes.astype(int)
    n_donors = int(len(donor_levels))
    if n_donors < 2:
        raise ValueError("Need >=2 donors.")

    k_use = int(min(max(2, int(k)), max(2, n - 1)))
    nn = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean")
    nn.fit(xy_arr)
    nbr_idx = nn.kneighbors(xy_arr, return_distance=False)[:, 1:]
    nbr_codes = donor_codes[nbr_idx]

    counts = np.zeros((n, n_donors), dtype=float)
    row_idx = np.repeat(np.arange(n, dtype=int), k_use)
    col_idx = nbr_codes.ravel().astype(int)
    np.add.at(counts, (row_idx, col_idx), 1.0)

    p = counts / float(k_use)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(p > 0.0, p * np.log(p), 0.0), axis=1)
    diversity = np.exp(entropy)
    ilisi = 1.0 / np.maximum(np.sum(p**2, axis=1), 1e-12)

    global_p = np.bincount(donor_codes, minlength=n_donors).astype(float) / float(n)
    expected = np.maximum(global_p * float(k_use), 1e-12)
    chi2_stat = np.sum(((counts - expected[None, :]) ** 2) / expected[None, :], axis=1)
    dof = max(1, n_donors - 1)
    kbet_p = chi2.sf(chi2_stat, dof)
    kbet_accept = kbet_p >= 0.05

    per_cell_df = pd.DataFrame(
        {
            "cell_index": np.arange(n, dtype=int),
            "donor_id": donors,
            "iLISI": ilisi,
            "diversity": diversity,
            "diversity_scaled": diversity / float(n_donors),
            "entropy": entropy,
            "kbet_chi2": chi2_stat,
            "kbet_pvalue": kbet_p,
            "kbet_accept": kbet_accept,
            "k_neighbors": int(k_use),
        }
    )

    donor_rows: list[dict[str, Any]] = []
    for donor_id in donor_levels:
        mask = per_cell_df["donor_id"] == donor_id
        if int(mask.sum()) == 0:
            continue
        donor_rows.append(
            {
                "donor_id": donor_id,
                "mixing_median_iLISI": float(per_cell_df.loc[mask, "iLISI"].median()),
                "mixing_median_diversity": float(
                    per_cell_df.loc[mask, "diversity"].median()
                ),
                "mixing_scaled": float(
                    per_cell_df.loc[mask, "diversity"].median() / float(n_donors)
                ),
                "kbet_accept_rate": float(
                    per_cell_df.loc[mask, "kbet_accept"].mean()
                ),
                "n_cells": int(mask.sum()),
            }
        )
    donor_mix_df = pd.DataFrame(donor_rows)
    overall = {
        "n_donors": float(n_donors),
        "k_neighbors": float(k_use),
        "mixing_median_iLISI": float(np.median(ilisi)),
        "mixing_median_diversity": float(np.median(diversity)),
        "mixing_scaled_overall": float(np.median(diversity) / float(n_donors)),
        "kbet_accept_rate_overall": float(np.mean(kbet_accept)),
    }
    return per_cell_df, donor_mix_df, overall


def _compute_donor_celltype_composition(
    donor_ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    donors = pd.Series(np.asarray(donor_ids).astype(str), name="donor_id")
    celltypes = pd.Series(np.asarray(labels).astype(str), name="celltype")
    counts = pd.crosstab(donors, celltypes)
    props = counts.div(np.maximum(counts.sum(axis=1), 1), axis=0)
    global_props = counts.sum(axis=0) / float(max(1, counts.values.sum()))
    skew = (props - global_props).abs().sum(axis=1)
    skew.name = "composition_skew"

    long_df = (
        counts.reset_index()
        .melt(id_vars="donor_id", var_name="celltype", value_name="n_cells")
        .merge(
            props.reset_index().melt(
                id_vars="donor_id",
                var_name="celltype",
                value_name="prop_within_donor",
            ),
            on=["donor_id", "celltype"],
            how="left",
        )
    )
    long_df["global_prop"] = long_df["celltype"].map(global_props).astype(float)
    return long_df, props, skew


def _plot_overview(
    *,
    umap_xy: np.ndarray,
    donor_ids: np.ndarray,
    label_values: np.ndarray | None,
    donor_key: str,
    label_key: str | None,
    center_xy: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_categorical_umap(
        umap_xy=umap_xy,
        labels=pd.Series(np.asarray(donor_ids).astype(str)),
        title=f"UMAP by donor ({donor_key})",
        outpath=out_dir / "umap_by_donor.png",
        vantage_point=(float(center_xy[0]), float(center_xy[1])),
        annotate_cluster_medians=False,
    )
    if label_values is not None and label_key is not None:
        plot_categorical_umap(
            umap_xy=umap_xy,
            labels=pd.Series(np.asarray(label_values).astype(str)),
            title=f"UMAP by label ({label_key})",
            outpath=out_dir / "umap_by_labels.png",
            vantage_point=(float(center_xy[0]), float(center_xy[1])),
            annotate_cluster_medians=False,
        )
    else:
        _save_placeholder(
            out_dir / "umap_by_labels.png",
            "UMAP by labels",
            "No cell-type label key found.",
        )

    counts = pd.Series(np.asarray(donor_ids).astype(str)).value_counts().sort_values(
        ascending=False
    )
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    x = np.arange(counts.shape[0], dtype=float)
    ax.bar(x, counts.to_numpy(dtype=float), color="#5DA5DA", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index.tolist(), rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("# cells")
    ax.set_title("Cells per donor")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "cells_per_donor.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)


def _plot_donor_directionality(
    *,
    donor_summary: pd.DataFrame,
    donor_artifacts: dict[str, dict[str, np.ndarray]],
    n_bins: int,
    top_donors: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if donor_summary.empty:
        _save_placeholder(
            out_dir / "donor_score_space.png",
            "Donor directionality",
            "No donor rows to plot.",
        )
        return

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    sc = ax.scatter(
        donor_summary["Z_T"].to_numpy(dtype=float),
        donor_summary["coverage_C"].to_numpy(dtype=float),
        c=donor_summary["mixing_scaled"].to_numpy(dtype=float),
        cmap="viridis",
        s=90,
        edgecolors="black",
        linewidths=0.6,
        alpha=0.9,
    )
    top_annot = donor_summary.sort_values(by="Z_T", ascending=False).head(5)
    for _, row in top_annot.iterrows():
        ax.text(
            float(row["Z_T"]) + 0.04,
            float(row["coverage_C"]) + 0.004,
            str(row["donor_id"]),
            fontsize=8,
        )
    ax.axvline(Z_STRONG, color="#8B0000", linestyle="--", linewidth=1.3)
    ax.axhline(COVERAGE_STRONG, color="#8B0000", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Z_T")
    ax.set_ylabel("coverage_C")
    ax.set_title("Donor-directionality score space")
    ax.grid(alpha=0.25, linewidth=0.6)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("mixing_scaled")
    fig.tight_layout()
    fig.savefig(out_dir / "donor_score_space.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)

    top_d = donor_summary.sort_values(by="Z_T", ascending=False).head(int(top_donors))
    for _, row in top_d.iterrows():
        donor_id = str(row["donor_id"])
        if donor_id not in donor_artifacts:
            continue
        art = donor_artifacts[donor_id]
        xy = np.asarray(art["xy"], dtype=float)
        mask = np.asarray(art["mask"], dtype=bool)
        center = np.asarray(art["center"], dtype=float)
        e_obs = np.asarray(art["E_phi_obs"], dtype=float)
        null_e = np.asarray(art["null_E_phi"], dtype=float)
        null_t = np.asarray(art["null_T"], dtype=float)
        t_obs = float(row["T_obs"])

        figp = plt.figure(figsize=(14.6, 4.8))
        ax1 = figp.add_subplot(1, 3, 1)
        ax2 = figp.add_subplot(1, 3, 2, projection="polar")
        ax3 = figp.add_subplot(1, 3, 3)

        ax1.scatter(
            xy[~mask, 0],
            xy[~mask, 1],
            c="#D2D2D2",
            s=4.0,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            label="others",
        )
        ax1.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c="#1F77B4",
            s=8.0,
            alpha=0.90,
            linewidths=0,
            rasterized=True,
            label=donor_id,
        )
        ax1.scatter(
            [float(center[0])],
            [float(center[1])],
            marker="X",
            s=85,
            c="black",
            edgecolors="white",
            linewidths=0.8,
        )
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f"Donor {donor_id} on UMAP")
        ax1.legend(loc="upper left", fontsize=7, frameon=True)

        th = theta_bin_centers(int(n_bins))
        th_c = np.concatenate([th, th[:1]])
        obs_c = np.concatenate([e_obs, e_obs[:1]])
        q_hi = np.quantile(null_e, 0.95, axis=0)
        q_lo = np.quantile(null_e, 0.05, axis=0)
        ax2.plot(th_c, obs_c, color="#8B0000", linewidth=2.0, label="obs")
        ax2.plot(
            th_c,
            np.concatenate([q_hi, q_hi[:1]]),
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            label="null95",
        )
        ax2.plot(
            th_c,
            np.concatenate([q_lo, q_lo[:1]]),
            color="#444444",
            linestyle="--",
            linewidth=1.0,
            label="null5",
        )
        ax2.set_theta_zero_location("E")
        ax2.set_theta_direction(1)
        ax2.set_thetagrids(np.arange(0, 360, 90))
        ann = (
            f"Z={float(row['Z_T']):.2f}\n"
            f"q={float(row['q_T']):.2e}\n"
            f"C={float(row['coverage_C']):.3f}\n"
            f"K={int(row['peaks_K'])}\n"
            f"phi={float(row['phi_hat_deg']):.1f}°"
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
        ax2.set_title("RSP profile + null envelope")
        ax2.legend(loc="upper right", bbox_to_anchor=(1.18, 1.2), fontsize=8, frameon=True)

        bins = int(min(45, max(12, np.ceil(np.sqrt(null_t.size)))))
        ax3.hist(null_t, bins=bins, color="#779ECB", edgecolor="white", alpha=0.9)
        ax3.axvline(t_obs, color="#8B0000", linestyle="--", linewidth=2.0)
        ax3.set_xlabel("null_T")
        ax3.set_ylabel("count")
        ax3.set_title("Null T distribution")

        figp.suptitle(f"Donor panel: {donor_id}", y=1.02)
        figp.tight_layout()
        figp.savefig(
            out_dir / f"donor_{_sanitize_name(donor_id)}_panel.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
            bbox_inches="tight",
        )
        plt.close(figp)


def _plot_mixing_metrics(
    *,
    per_cell_mix: pd.DataFrame,
    donor_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if per_cell_mix.empty or donor_summary.empty:
        _save_placeholder(
            out_dir / "per_cell_iLISI_hist.png",
            "Mixing metrics",
            "No mixing metrics available.",
        )
        return

    fig1, ax1 = plt.subplots(figsize=(8.0, 5.2))
    vals = per_cell_mix["iLISI"].to_numpy(dtype=float)
    ax1.hist(vals, bins=45, color="#4C78A8", edgecolor="white", alpha=0.9)
    med = float(np.median(vals))
    ax1.axvline(med, color="#8B0000", linestyle="--", linewidth=1.8, label=f"median={med:.2f}")
    ax1.set_xlabel("iLISI")
    ax1.set_ylabel("count")
    ax1.set_title("Per-cell iLISI distribution")
    ax1.legend(loc="best", fontsize=8, frameon=True)
    ax1.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig1.tight_layout()
    fig1.savefig(out_dir / "per_cell_iLISI_hist.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    donors_order = donor_summary.sort_values(by="n_cells", ascending=False)["donor_id"].astype(str).tolist()
    data = [
        per_cell_mix.loc[per_cell_mix["donor_id"].astype(str) == d, "iLISI"].to_numpy(dtype=float)
        for d in donors_order
    ]
    fig2, ax2 = plt.subplots(figsize=(max(8.5, 0.45 * len(donors_order) + 4.0), 5.2))
    ax2.boxplot(
        data,
        tick_labels=donors_order,
        patch_artist=True,
        boxprops={"facecolor": "#9ecae1"},
        medianprops={"color": "#8B0000", "linewidth": 1.4},
    )
    ax2.set_ylabel("iLISI")
    ax2.set_title("Per-donor iLISI distributions")
    ax2.set_xticklabels(donors_order, rotation=40, ha="right", fontsize=8)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig2.tight_layout()
    fig2.savefig(out_dir / "per_donor_iLISI_boxplot.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(7.6, 5.4))
    ax3.scatter(
        donor_summary["n_cells"].to_numpy(dtype=float),
        donor_summary["mixing_scaled"].to_numpy(dtype=float),
        c=donor_summary["Z_T"].to_numpy(dtype=float),
        cmap="magma",
        s=90,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
    )
    for _, row in donor_summary.sort_values(by="Z_T", ascending=False).head(5).iterrows():
        ax3.text(
            float(row["n_cells"]) + 12.0,
            float(row["mixing_scaled"]) + 0.004,
            str(row["donor_id"]),
            fontsize=8,
        )
    ax3.axhline(POOR_MIXING_SCALED, color="#8B0000", linestyle="--", linewidth=1.2)
    ax3.set_xlabel("donor cell count")
    ax3.set_ylabel("mixing_scaled")
    ax3.set_title("Donor cell count vs mixing_scaled")
    ax3.grid(alpha=0.25, linewidth=0.6)
    fig3.tight_layout()
    fig3.savefig(out_dir / "donor_size_vs_mixing_scaled.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(max(8.2, 0.42 * len(donors_order) + 4.0), 4.8))
    rates = (
        donor_summary.set_index("donor_id")
        .loc[donors_order, "kbet_accept_rate"]
        .to_numpy(dtype=float)
    )
    x = np.arange(len(donors_order), dtype=float)
    ax4.bar(x, rates, color="#72B7B2", edgecolor="black", linewidth=0.5)
    ax4.axhline(0.50, color="#8B0000", linestyle="--", linewidth=1.2)
    ax4.set_ylim(0.0, 1.0)
    ax4.set_xticks(x)
    ax4.set_xticklabels(donors_order, rotation=40, ha="right", fontsize=8)
    ax4.set_ylabel("kBET-like acceptance rate")
    ax4.set_title("Per-donor kBET-style neighborhood acceptance")
    ax4.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig4.tight_layout()
    fig4.savefig(out_dir / "kbet_accept_rate_by_donor.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig4)


def _plot_composition_controls(
    *,
    donor_summary: pd.DataFrame,
    composition_props: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if composition_props.empty:
        _save_placeholder(
            out_dir / "donor_celltype_stacked.png",
            "Composition controls",
            "No label information available.",
        )
        _save_placeholder(
            out_dir / "donor_celltype_heatmap.png",
            "Composition controls",
            "No label information available.",
        )
        _save_placeholder(
            out_dir / "composition_skew_vs_ZT.png",
            "Composition controls",
            "No label information available.",
        )
        return

    donors_order = donor_summary.sort_values(by="n_cells", ascending=False)["donor_id"].astype(str).tolist()
    props = composition_props.copy()
    props = props.loc[[d for d in donors_order if d in props.index]]
    celltypes = props.columns.astype(str).tolist()
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(celltypes))]

    fig1, ax1 = plt.subplots(figsize=(max(9.0, 0.45 * props.shape[0] + 4.0), 5.8))
    bottom = np.zeros(props.shape[0], dtype=float)
    x = np.arange(props.shape[0], dtype=float)
    for i, ct in enumerate(celltypes):
        vals = props[ct].to_numpy(dtype=float)
        ax1.bar(
            x,
            vals,
            bottom=bottom,
            color=colors[i],
            width=0.82,
            edgecolor="white",
            linewidth=0.3,
            label=ct,
        )
        bottom += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(props.index.tolist(), rotation=40, ha="right", fontsize=8)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Proportion within donor")
    ax1.set_title("Donor-by-celltype composition (stacked)")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=True)
    fig1.tight_layout()
    fig1.savefig(out_dir / "donor_celltype_stacked.png", dpi=DEFAULT_PLOT_STYLE.dpi, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(max(8.0, 0.42 * props.shape[1] + 3.0), max(5.0, 0.28 * props.shape[0] + 2.8)))
    im = ax2.imshow(props.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=max(1e-6, float(np.nanmax(props.to_numpy(dtype=float)))))
    ax2.set_xticks(np.arange(props.shape[1], dtype=int))
    ax2.set_xticklabels(celltypes, rotation=40, ha="right", fontsize=8)
    ax2.set_yticks(np.arange(props.shape[0], dtype=int))
    ax2.set_yticklabels(props.index.tolist(), fontsize=8)
    ax2.set_title("Donor-by-celltype proportion heatmap")
    cbar = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    cbar.set_label("proportion")
    fig2.tight_layout()
    fig2.savefig(out_dir / "donor_celltype_heatmap.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)

    skew_col = "composition_skew"
    if skew_col in donor_summary.columns:
        fig3, ax3 = plt.subplots(figsize=(7.5, 5.4))
        ax3.scatter(
            donor_summary[skew_col].to_numpy(dtype=float),
            donor_summary["Z_T"].to_numpy(dtype=float),
            c=donor_summary["mixing_scaled"].to_numpy(dtype=float),
            cmap="cividis",
            s=90,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.9,
        )
        for _, row in donor_summary.sort_values(by="Z_T", ascending=False).head(5).iterrows():
            ax3.text(
                float(row[skew_col]) + 0.005,
                float(row["Z_T"]) + 0.04,
                str(row["donor_id"]),
                fontsize=8,
            )
        ax3.axhline(Z_STRONG, color="#8B0000", linestyle="--", linewidth=1.2)
        ax3.set_xlabel("composition_skew (L1 distance vs global)")
        ax3.set_ylabel("Z_T")
        ax3.set_title("Donor composition skew vs donor directionality")
        ax3.grid(alpha=0.25, linewidth=0.6)
        fig3.tight_layout()
        fig3.savefig(out_dir / "composition_skew_vs_ZT.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig3)
    else:
        _save_placeholder(
            out_dir / "composition_skew_vs_ZT.png",
            "Composition skew vs Z",
            "composition_skew unavailable.",
        )


def _plot_within_celltype(
    *,
    umap_xy: np.ndarray,
    donor_ids: np.ndarray,
    within_df: pd.DataFrame,
    top_celltypes: list[str],
    label_values: np.ndarray,
    center_xy: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if within_df.empty:
        _save_placeholder(
            out_dir / "within_celltype_ZT_heatmap.png",
            "Within-celltype diagnostics",
            "No within-celltype donor-directionality rows.",
        )
        return

    donor_palette_levels = sorted(pd.Index(np.asarray(donor_ids).astype(str)).unique().tolist())
    donor_color_map = {d: plt.get_cmap("tab20")(i % 20) for i, d in enumerate(donor_palette_levels)}

    labels = np.asarray(label_values).astype(str)
    donors = np.asarray(donor_ids).astype(str)

    for ct in top_celltypes:
        sub = within_df.loc[within_df["celltype"] == ct].copy()
        if sub.empty:
            continue
        mask = labels == str(ct)
        xy_ct = np.asarray(umap_xy[mask], dtype=float)
        donors_ct = donors[mask]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.0))
        for donor in sorted(pd.Index(donors_ct).unique().tolist()):
            dm = donors_ct == donor
            ax1.scatter(
                xy_ct[dm, 0],
                xy_ct[dm, 1],
                s=7.0,
                alpha=0.88,
                linewidths=0,
                rasterized=True,
                color=donor_color_map.get(donor, "#666666"),
                label=donor,
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
        ax1.set_title(f"{ct}: subset UMAP by donor")
        handles, labels_l = ax1.get_legend_handles_labels()
        if len(handles) > 0:
            ax1.legend(
                handles[: min(12, len(handles))],
                labels_l[: min(12, len(labels_l))],
                loc="upper left",
                fontsize=7,
                frameon=True,
            )

        sc = ax2.scatter(
            sub["Z_T"].to_numpy(dtype=float),
            sub["coverage_C"].to_numpy(dtype=float),
            c=sub["mixing_scaled_in_celltype"].to_numpy(dtype=float),
            cmap="viridis",
            s=85,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
        )
        ax2.axvline(Z_STRONG, color="#8B0000", linestyle="--", linewidth=1.1)
        ax2.axhline(COVERAGE_STRONG, color="#8B0000", linestyle="--", linewidth=1.1)
        ax2.set_xlabel("Z_T")
        ax2.set_ylabel("coverage_C")
        ax2.set_title(f"{ct}: donor Z_T vs coverage")
        ax2.grid(alpha=0.25, linewidth=0.6)
        cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.03)
        cbar.set_label("mixing_scaled_in_celltype")
        fig.tight_layout()
        fig.savefig(
            out_dir / f"celltype_{_sanitize_name(ct)}_summary.png",
            dpi=DEFAULT_PLOT_STYLE.dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

    heat = within_df.pivot_table(
        index="donor_id",
        columns="celltype",
        values="Z_T",
        aggfunc="first",
    )
    for ct in heat.columns.tolist():
        low_mask = within_df.loc[within_df["celltype"] == ct, ["donor_id", "n_cells_in_celltype_donor"]].set_index("donor_id")["n_cells_in_celltype_donor"]
        low_mask = low_mask.reindex(heat.index)
        heat.loc[low_mask < 20, ct] = np.nan

    fig_h, ax_h = plt.subplots(
        figsize=(
            max(8.0, 1.0 + 0.95 * max(1, heat.shape[1])),
            max(4.8, 1.0 + 0.35 * max(1, heat.shape[0])),
        )
    )
    arr = heat.to_numpy(dtype=float)
    if np.all(~np.isfinite(arr)):
        _save_placeholder(
            out_dir / "within_celltype_ZT_heatmap.png",
            "Within-celltype Z heatmap",
            "No finite Z_T values after masking low-count cells.",
        )
        plt.close(fig_h)
    else:
        vlim = float(np.nanmax(np.abs(arr)))
        vlim = max(vlim, 1e-6)
        im = ax_h.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-vlim, vmax=vlim)
        ax_h.set_xticks(np.arange(heat.shape[1], dtype=int))
        ax_h.set_xticklabels(heat.columns.tolist(), rotation=35, ha="right", fontsize=8)
        ax_h.set_yticks(np.arange(heat.shape[0], dtype=int))
        ax_h.set_yticklabels(heat.index.tolist(), fontsize=8)
        ax_h.set_title("Within-celltype donor-directionality Z_T heatmap")
        cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.03)
        cbar.set_label("Z_T")
        fig_h.tight_layout()
        fig_h.savefig(out_dir / "within_celltype_ZT_heatmap.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig_h)


def _build_within_celltype_table(
    *,
    umap_xy: np.ndarray,
    donor_ids: np.ndarray,
    label_values: np.ndarray,
    top_celltypes: int,
    center_xy: np.ndarray,
    n_bins: int,
    n_perm: int,
    k: int,
    seed: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    labels = np.asarray(label_values).astype(str)
    donors = np.asarray(donor_ids).astype(str)
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    celltypes = counts.index.astype(str).tolist()[: int(top_celltypes)]

    warnings_log: list[str] = []
    rows: list[pd.DataFrame] = []
    for i, ct in enumerate(celltypes):
        mask = labels == ct
        n_ct = int(mask.sum())
        if n_ct < 100:
            warnings_log.append(f"Skipped celltype '{ct}' (n={n_ct}) due to low cell count.")
            continue
        donors_ct = donors[mask]
        if np.unique(donors_ct).size < 2:
            warnings_log.append(
                f"Skipped celltype '{ct}' because <2 donors are present in subset."
            )
            continue
        xy_ct = umap_xy[mask]
        scores_ct, _, _ = _analyze_donor_directionality(
            xy=xy_ct,
            donor_ids=donors_ct,
            n_bins=int(n_bins),
            n_perm=int(n_perm),
            seed=int(seed + 100_000 + i * 1000),
            center_xy=center_xy,
            null_group_labels=None,
        )
        _, donor_mix_ct, _ = _compute_mixing_metrics(
            xy=xy_ct,
            donor_ids=donors_ct,
            k=int(k),
        )
        merged = scores_ct.merge(
            donor_mix_ct[
                [
                    "donor_id",
                    "mixing_median_iLISI",
                    "mixing_median_diversity",
                    "mixing_scaled",
                ]
            ],
            on="donor_id",
            how="left",
        )
        merged = merged.rename(
            columns={
                "n_cells": "n_cells_in_celltype_donor",
                "mixing_scaled": "mixing_scaled_in_celltype",
            }
        )
        merged["celltype"] = str(ct)
        merged["n_cells_in_celltype_total"] = int(n_ct)
        rows.append(merged)

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out = out[
            [
                "celltype",
                "donor_id",
                "n_cells_in_celltype_donor",
                "Z_T",
                "q_T",
                "coverage_C",
                "peaks_K",
                "phi_hat_deg",
                "mixing_scaled_in_celltype",
                "mixing_median_iLISI",
                "mixing_median_diversity",
                "n_cells_in_celltype_total",
            ]
        ].copy()
        return out, celltypes, warnings_log
    return pd.DataFrame(), celltypes, warnings_log


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
    p_donor = plots_dir / "01_donor_directionality"
    p_mix = plots_dir / "02_mixing_metrics"
    p_comp = plots_dir / "03_composition_controls"
    p_within = plots_dir / "04_within_celltype"
    for d in [tables_dir, p_overview, p_donor, p_mix, p_comp, p_within]:
        d.mkdir(parents=True, exist_ok=True)

    warnings_log: list[str] = []

    adata = ad.read_h5ad(h5ad_path)
    _ensure_umap(adata, seed=int(args.seed), recompute_if_missing=False)
    embedding_key, umap_xy = _resolve_embedding(adata, args.embedding_key)
    center_xy = compute_vantage_point(umap_xy, method="median")

    # Resolve expression source for provenance consistency across experiments.
    _, _, expr_source = _choose_expression_source(
        adata, layer_arg=args.layer, use_raw_arg=bool(args.use_raw)
    )

    donor_ids, donor_key_used = _resolve_donor_ids_required(adata, args.donor_key)
    label_values_raw, label_key_used = _resolve_label_values(adata, args.label_key)
    if label_key_used is None:
        warnings_log.append(
            "Label key unavailable; composition-preserving null and composition diagnostics skipped."
        )

    null_group_labels = label_values_raw if label_values_raw is not None else None
    donor_scores, donor_artifacts, donor_meta = _analyze_donor_directionality(
        xy=umap_xy,
        donor_ids=donor_ids,
        n_bins=int(args.n_bins),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        center_xy=np.asarray(center_xy, dtype=float),
        null_group_labels=null_group_labels,
    )

    per_cell_mix, donor_mix, mix_overall = _compute_mixing_metrics(
        xy=umap_xy, donor_ids=donor_ids, k=int(args.k)
    )

    donor_summary = donor_scores.merge(
        donor_mix[
            [
                "donor_id",
                "mixing_median_iLISI",
                "mixing_median_diversity",
                "mixing_scaled",
                "kbet_accept_rate",
            ]
        ],
        on="donor_id",
        how="left",
    )
    donor_summary["strong_donor_directionality"] = (
        ((donor_summary["q_T"] <= Q_SIG) & (donor_summary["Z_T"] >= Z_STRONG))
        | (
            (donor_summary["coverage_C"] >= COVERAGE_STRONG)
            & (donor_summary["Z_T"] >= Z_MODERATE)
        )
    )
    donor_summary["poor_mixing"] = (
        (donor_summary["mixing_scaled"] <= POOR_MIXING_SCALED)
        | (donor_summary["mixing_median_iLISI"] <= POOR_MIXING_ILISI)
    )
    donor_summary["confounding_flag"] = (
        donor_summary["strong_donor_directionality"] | donor_summary["poor_mixing"]
    )
    donor_summary = donor_summary.sort_values(
        by=["confounding_flag", "Z_T", "q_T"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    composition_long = pd.DataFrame()
    composition_props = pd.DataFrame()
    within_df = pd.DataFrame()
    within_top_celltypes: list[str] = []
    collapsed_labels = None
    rare_labels: list[str] = []
    if label_values_raw is not None and label_key_used is not None:
        collapsed_labels, rare_labels = _collapse_rare_labels(
            label_values_raw, min_fraction=0.01
        )
        composition_long, composition_props, composition_skew = (
            _compute_donor_celltype_composition(donor_ids, collapsed_labels)
        )
        donor_summary["composition_skew"] = donor_summary["donor_id"].map(
            composition_skew.to_dict()
        )
        within_df, within_top_celltypes, within_warn = _build_within_celltype_table(
            umap_xy=umap_xy,
            donor_ids=donor_ids,
            label_values=label_values_raw,
            top_celltypes=int(args.top_celltypes),
            center_xy=np.asarray(center_xy, dtype=float),
            n_bins=int(args.n_bins),
            n_perm=int(args.n_perm),
            k=int(args.k),
            seed=int(args.seed),
        )
        warnings_log.extend(within_warn)

    # Donor-level summary table with requested columns.
    donor_summary_out = donor_summary[
        [
            "donor_id",
            "n_cells",
            "Z_T",
            "p_T",
            "q_T",
            "coverage_C",
            "peaks_K",
            "phi_hat_deg",
            "mixing_median_iLISI",
            "mixing_median_diversity",
            "mixing_scaled",
            "confounding_flag",
            "strong_donor_directionality",
            "poor_mixing",
            "kbet_accept_rate",
            "null_mode",
        ]
    ].copy()
    if "composition_skew" in donor_summary.columns:
        donor_summary_out["composition_skew"] = donor_summary["composition_skew"].to_numpy(
            dtype=float
        )

    donor_summary_csv = tables_dir / "donor_summary.csv"
    donor_summary_out.to_csv(donor_summary_csv, index=False)

    per_cell_csv = tables_dir / "per_cell_mixing_metrics.csv"
    if bool(args.save_per_cell):
        # Keep full per-cell metrics only when requested.
        per_cell_out = per_cell_mix.copy()
        per_cell_out["cell_id"] = adata.obs_names.to_numpy(dtype=str)
        per_cell_out.to_csv(per_cell_csv, index=False)

    composition_csv = tables_dir / "donor_celltype_composition.csv"
    within_csv = tables_dir / "donor_directionality_by_celltype.csv"
    if not composition_long.empty:
        composition_long.to_csv(composition_csv, index=False)
    if not within_df.empty:
        within_df.to_csv(within_csv, index=False)

    _plot_overview(
        umap_xy=umap_xy,
        donor_ids=donor_ids,
        label_values=label_values_raw,
        donor_key=donor_key_used,
        label_key=label_key_used,
        center_xy=np.asarray(center_xy, dtype=float),
        out_dir=p_overview,
    )
    _plot_donor_directionality(
        donor_summary=donor_summary,
        donor_artifacts=donor_artifacts,
        n_bins=int(args.n_bins),
        top_donors=int(args.top_donors),
        out_dir=p_donor,
    )
    _plot_mixing_metrics(
        per_cell_mix=per_cell_mix,
        donor_summary=donor_summary,
        out_dir=p_mix,
    )
    _plot_composition_controls(
        donor_summary=donor_summary,
        composition_props=composition_props,
        out_dir=p_comp,
    )
    if label_values_raw is not None and not within_df.empty:
        _plot_within_celltype(
            umap_xy=umap_xy,
            donor_ids=donor_ids,
            within_df=within_df,
            top_celltypes=within_top_celltypes,
            label_values=label_values_raw,
            center_xy=np.asarray(center_xy, dtype=float),
            out_dir=p_within,
        )
    else:
        _save_placeholder(
            p_within / "within_celltype_ZT_heatmap.png",
            "Within-celltype diagnostics",
            "No label key or no analyzable celltype subsets.",
        )

    strong_n = int(donor_summary["strong_donor_directionality"].sum())
    confounded_n = int(donor_summary["confounding_flag"].sum())
    poor_mix_n = int(donor_summary["poor_mixing"].sum())
    if strong_n > 0:
        warnings_log.append(
            f"{strong_n} donor(s) show strong donor-directionality (q<=0.05 & Z>=4 or C>=0.20 & Z>=3)."
        )
    if poor_mix_n > 0:
        warnings_log.append(
            f"{poor_mix_n} donor(s) show poor mixing (mixing_scaled<=0.30 or median iLISI<=2.0)."
        )

    readme_lines = [
        "Experiment #6: Donor mixing / batch-geometry stress test",
        "",
        "This diagnostic quantifies donor-directionality and neighborhood mixing in embedding space.",
        "Directions are representation-conditional (embedding direction is not tissue direction).",
        "",
        "Metadata:",
        f"- embedding_key_used: {embedding_key}",
        f"- donor_key_used: {donor_key_used}",
        f"- label_key_used: {label_key_used if label_key_used is not None else 'None'}",
        f"- expression_source_used: {expr_source}",
        f"- n_cells: {int(adata.n_obs)}",
        f"- n_donors: {int(np.unique(donor_ids).size)}",
        f"- n_bins: {int(args.n_bins)}",
        f"- n_perm: {int(args.n_perm)}",
        f"- k_neighbors: {int(args.k)}",
        f"- donor_null_mode: {donor_meta.get('null_mode', 'unknown')}",
        f"- save_per_cell: {bool(args.save_per_cell)}",
        "",
        "Overall mixing summary:",
        f"- overall_median_iLISI: {float(mix_overall['mixing_median_iLISI']):.4f}",
        f"- overall_median_diversity: {float(mix_overall['mixing_median_diversity']):.4f}",
        f"- overall_mixing_scaled: {float(mix_overall['mixing_scaled_overall']):.4f}",
        f"- overall_kbet_accept_rate: {float(mix_overall['kbet_accept_rate_overall']):.4f}",
        "",
        "Interpretation guidelines:",
        "- If donor-directionality is strong or mixing is poor, gene-directionality claims need stronger controls.",
        "- Recommended controls: donor-stratified nulls with donor x depth-decile strata and per-donor replication checks.",
        "- Donor-directionality within a single cell type is stronger evidence of embedding batch structure than across-all-cells donor-directionality.",
        "",
        "Thresholds used for confounding flags:",
        "- strong_donor_directionality: (q_T<=0.05 and Z_T>=4) or (coverage_C>=0.20 and Z_T>=3)",
        "- poor_mixing: (mixing_scaled<=0.30) or (median_iLISI<=2.0)",
        "- confounding_flag: strong_donor_directionality OR poor_mixing",
        "",
        "Summary counts:",
        f"- donors_with_strong_directionality: {strong_n}",
        f"- donors_with_poor_mixing: {poor_mix_n}",
        f"- donors_with_confounding_flag: {confounded_n}",
    ]
    if label_values_raw is not None:
        readme_lines.extend(
            [
                f"- composition_labels_collapsed_rare_under_1pct: {len(rare_labels)}",
                f"- within_celltypes_attempted: {int(args.top_celltypes)}",
                f"- within_celltypes_analyzed: {int(len(pd.Index(within_top_celltypes).unique()))}",
            ]
        )
    readme_lines.extend(["", "Warnings:"])
    if warnings_log:
        for w in warnings_log:
            readme_lines.append(f"- {w}")
    else:
        readme_lines.append("- none")

    (outdir / "README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"embedding_key_used={embedding_key}")
    print(f"donor_key_used={donor_key_used}")
    print(f"label_key_used={label_key_used if label_key_used is not None else 'None'}")
    print(f"expression_source_used={expr_source}")
    print(
        f"n_cells={int(adata.n_obs)} n_donors={int(np.unique(donor_ids).size)} n_bins={int(args.n_bins)} n_perm={int(args.n_perm)} k={int(args.k)}"
    )
    print(f"donor_summary_csv={donor_summary_csv.as_posix()}")
    if bool(args.save_per_cell):
        print(f"per_cell_mixing_csv={per_cell_csv.as_posix()}")
    else:
        print("per_cell_mixing_csv=skipped (save_per_cell=False)")
    if not composition_long.empty:
        print(f"donor_celltype_composition_csv={composition_csv.as_posix()}")
    if not within_df.empty:
        print(f"donor_directionality_by_celltype_csv={within_csv.as_posix()}")
    print(f"results_root={outdir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
