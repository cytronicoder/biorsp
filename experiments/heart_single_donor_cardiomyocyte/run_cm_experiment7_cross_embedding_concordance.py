#!/usr/bin/env python3
"""CM Experiment #7 (single donor): cross-embedding concordance as robustness instrument.

Core hypothesis:
Within a single cardiomyocyte-heavy donor, real cardiomyocyte programs should produce
BioRSP localization signals that are detectable in PCA-2D, not only non-linear embeddings,
and broadly concordant in ranking/classification across PCA, UMAP, and t-SNE families.

Interpretation guardrails:
- Directions are representation-conditional (phi is embedding geometry, not anatomy).
- We do not claim non-linear embeddings "discover" biology; we test cross-representation
  generalization of BioRSP scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

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
    "TTN",
    "RYR2",
    "ATP2A2",
    "PLN",
    "NPPA",
    "NPPB",
]

UNDERPOWERED_PREV = 0.005
UNDERPOWERED_MIN_FG = 50
Q_SIG = 0.05

CLASS_ORDER = ["Localized-unimodal", "Localized-multimodal", "Not-localized"]
CLASS_TO_INT = {"Not-localized": 0, "Localized-unimodal": 1, "Localized-multimodal": 2}
INT_TO_CLASS = {0: "Not-localized", 1: "Localized-unimodal", 2: "Localized-multimodal"}


@dataclass(frozen=True)
class EmbeddingSpec:
    family: str
    name: str
    params: dict[str, Any]
    coords: np.ndarray


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
class EmbeddingGeom:
    center_xy: np.ndarray
    theta: np.ndarray
    bin_id: np.ndarray
    bin_counts_total: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CM Experiment #7: cross-embedding concordance for single-donor cardiomyocyte BioRSP signals."
    )
    p.add_argument("--h5ad", default="data/processed/HT_pca_umap.h5ad")
    p.add_argument(
        "--out",
        default="experiments/heart_single_donor_cardiomyocyte/results/cm_experiment7_cross_embedding_concordance",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_perm", type=int, default=300)
    p.add_argument("--n_bins", type=int, default=64)
    p.add_argument("--q", type=float, default=0.10)
    p.add_argument("--k_pca", type=int, default=50)

    p.add_argument("--umap_neighbors", type=int, nargs="+", default=[15, 30, 50])
    p.add_argument("--umap_min_dist", type=float, nargs="+", default=[0.1, 0.5])
    p.add_argument("--umap_seeds", type=int, nargs="+", default=[0, 1])

    p.add_argument("--tsne_perplexity", type=int, nargs="+", default=[30, 50])
    p.add_argument("--tsne_seeds", type=int, nargs="+", default=[0, 1])

    p.add_argument("--foreground_modes", nargs="+", choices=["topq", "detect"], default=["topq"])
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


def _prepare_embedding_input(
    adata_cm: ad.AnnData,
    expr_matrix_cm: Any,
    expr_source: str,
) -> tuple[ad.AnnData, str]:
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


def _fmt_float(x: float) -> str:
    s = f"{float(x):.6g}"
    return s


def _compute_embedding_suite(
    adata_embed: ad.AnnData,
    *,
    seed: int,
    k_pca: int,
    umap_neighbors: list[int],
    umap_min_dist: list[float],
    umap_seeds: list[int],
    tsne_perplexities: list[int],
    tsne_seeds: list[int],
    warnings_log: list[str],
) -> tuple[list[EmbeddingSpec], int]:
    import scanpy as sc

    n_cells, n_vars = adata_embed.n_obs, adata_embed.n_vars
    n_pcs = int(max(2, min(int(k_pca), 50, n_vars - 1, n_cells - 1)))

    sc.pp.pca(adata_embed, n_comps=n_pcs, svd_solver="arpack", random_state=int(seed))
    pca_all = np.asarray(adata_embed.obsm["X_pca"], dtype=float)

    specs: list[EmbeddingSpec] = [
        EmbeddingSpec(
            family="PCA",
            name="pca2d",
            params={"n_pcs": n_pcs, "seed": int(seed)},
            coords=pca_all[:, :2].copy(),
        )
    ]

    # UMAP family.
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
                nm = f"umap_nn{int(nn)}_md{_fmt_float(float(md))}_s{int(rs)}"
                specs.append(
                    EmbeddingSpec(
                        family="UMAP",
                        name=nm,
                        params={
                            "n_neighbors": int(nn),
                            "min_dist": float(md),
                            "random_state": int(rs),
                            "n_pcs": n_pcs,
                        },
                        coords=np.asarray(adata_embed.obsm["X_umap"], dtype=float)[:, :2].copy(),
                    )
                )

    # t-SNE family.
    max_perp = float((int(n_cells) - 1) / 3.0)
    for perp in tsne_perplexities:
        if float(perp) >= max_perp:
            msg = (
                f"Skipping t-SNE perplexity={perp}: requires perplexity < (n_obs-1)/3 "
                f"= {max_perp:.2f}."
            )
            warnings_log.append(msg)
            print(f"WARNING: {msg}")
            continue
        for rs in tsne_seeds:
            try:
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
            except Exception as exc:  # pragma: no cover - runtime guard
                msg = f"t-SNE failed for perplexity={perp}, seed={rs}: {exc}"
                warnings_log.append(msg)
                print(f"WARNING: {msg}")
                continue

            nm = f"tsne_p{int(perp)}_s{int(rs)}"
            specs.append(
                EmbeddingSpec(
                    family="TSNE",
                    name=nm,
                    params={"perplexity": float(perp), "random_state": int(rs), "n_pcs": n_pcs},
                    coords=coords,
                )
            )

    return specs, n_pcs


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


def _circular_stats_deg(phi_deg: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(phi_deg, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    rad = np.deg2rad(arr)
    mean_vec = np.mean(np.exp(1j * rad))
    mu = float(np.mod(np.angle(mean_vec), 2.0 * np.pi))
    r = float(np.abs(mean_vec))
    circ_sd = float(np.sqrt(max(0.0, -2.0 * np.log(max(r, 1e-12)))))
    return float(np.rad2deg(mu)), r, circ_sd


def _circ_diff_deg(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))


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


def _score_one(
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
    }


def _classify_test(q_within_embedding: float, peaks_k: float, underpowered: bool) -> str:
    if underpowered:
        return "Not-localized"
    if np.isfinite(float(q_within_embedding)) and float(q_within_embedding) <= Q_SIG:
        if int(peaks_k) >= 2:
            return "Localized-multimodal"
        return "Localized-unimodal"
    return "Not-localized"


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


def _pair_label(a: str, b: str) -> str:
    parts = sorted([str(a), str(b)])
    return f"{parts[0]}-vs-{parts[1]}"


def _stable_token(text: str) -> int:
    s = str(text)
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(s)))


def _mode_class(values: list[str]) -> str:
    if len(values) == 0:
        return "Not-localized"
    ser = pd.Series(values, dtype="string")
    counts = ser.value_counts()
    return str(counts.index[0])


def _plot_feature(ax: plt.Axes, coords: np.ndarray, values_log: np.ndarray, title: str, vmin: float, vmax: float) -> None:
    ord_idx = np.argsort(values_log, kind="mergesort")
    ax.scatter(coords[:, 0], coords[:, 1], c="#dddddd", s=4, alpha=0.25, linewidths=0, rasterized=True)
    ax.scatter(
        coords[ord_idx, 0],
        coords[ord_idx, 1],
        c=values_log[ord_idx],
        cmap="Reds",
        s=7,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def _plot_polar_family(
    ax: plt.Axes,
    title: str,
    e_obs: np.ndarray,
    null_e: np.ndarray | None,
    stats_text: str,
) -> None:
    n_bins = int(e_obs.size)
    th_c = theta_bin_centers(n_bins)
    th = np.concatenate([th_c, th_c[:1]])
    obs = np.concatenate([e_obs, e_obs[:1]])
    ax.plot(th, obs, color="#8B0000", linewidth=2.0)
    if null_e is not None:
        hi = np.quantile(null_e, 0.95, axis=0)
        lo = np.quantile(null_e, 0.05, axis=0)
        ax.plot(th, np.concatenate([hi, hi[:1]]), color="#333", linestyle="--", linewidth=1.0)
        ax.plot(th, np.concatenate([lo, lo[:1]]), color="#333", linestyle="--", linewidth=1.0)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rticks([])
    ax.set_title(title, fontsize=8)
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


def _write_readme(
    out_path: Path,
    *,
    args: argparse.Namespace,
    donor_key: str,
    label_key: str,
    donor_star: str,
    expr_source: str,
    embed_note: str,
    n_cells_donor: int,
    n_cells_cm: int,
    cm_label_counts: dict[str, int],
    n_embeddings: int,
    n_genes_present: int,
    warnings_log: list[str],
) -> None:
    lines: list[str] = []
    lines.append("CM Experiment #7 (Single-donor): Cross-embedding concordance as robustness instrument")
    lines.append("")
    lines.append("Core hypothesis")
    lines.append(
        "If a cardiomyocyte program is real within a single donor, BioRSP localization should be detectable in PCA-2D "
        "and not be exclusive to non-linear embeddings, with consistent ranking/classification across PCA/UMAP/t-SNE."
    )
    lines.append("")
    lines.append("Interpretation guardrails")
    lines.append("- Embedding directions are representation-conditional; phi does not map to anatomy.")
    lines.append("- We do not claim UMAP/t-SNE discovers biology; we test score generalization across representations.")
    lines.append("")
    lines.append("Run metadata")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- n_perm: {args.n_perm}")
    lines.append(f"- n_bins: {args.n_bins}")
    lines.append(f"- q: {args.q}")
    lines.append(f"- foreground_modes: {', '.join(args.foreground_modes)}")
    lines.append(f"- donor_key_used: {donor_key}")
    lines.append(f"- label_key_used: {label_key}")
    lines.append(f"- donor_star: {donor_star}")
    lines.append(f"- expression_source_used: {expr_source}")
    lines.append(f"- embedding_input_prep: {embed_note}")
    lines.append(f"- donor_star_total_cells: {n_cells_donor}")
    lines.append(f"- donor_star_cardiomyocytes: {n_cells_cm}")
    lines.append(f"- n_embeddings_scored: {n_embeddings}")
    lines.append(f"- n_genes_present: {n_genes_present}")
    lines.append("")
    lines.append("Cardiomyocyte labels included")
    for k, v in cm_label_counts.items():
        lines.append(f"- {k}: {v}")
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
        plots_dir / "01_rank_concordance",
        plots_dir / "02_class_concordance",
        plots_dir / "03_direction_concordance",
        plots_dir / "04_exemplar_panels",
        plots_dir / "05_embedding_gallery",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    warnings_log: list[str] = []

    adata = ad.read_h5ad(args.h5ad)

    donor_key = _resolve_key_required(adata, args.donor_key, DONOR_KEY_CANDIDATES, purpose="donor")
    label_key = _resolve_key_required(adata, args.label_key, LABEL_KEY_CANDIDATES, purpose="label")

    labels_all = adata.obs[label_key].astype("string").fillna("NA").astype(str)
    donor_ids_all = adata.obs[donor_key].astype("string").fillna("NA").astype(str)

    cm_mask_all = labels_all.map(_is_cm_label).to_numpy(dtype=bool)
    if int(cm_mask_all.sum()) == 0:
        raise RuntimeError("No cardiomyocyte cells detected by substring matching")

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

    if int(adata_cm.n_obs) < 2000:
        msg = f"CM subset has {int(adata_cm.n_obs)} cells (<2000); proceeding with warning."
        warnings_log.append(msg)
        print(f"WARNING: {msg}")

    cm_label_counts = (
        labels_donor.loc[cm_mask_donor].value_counts().sort_index().astype(int).to_dict()
    )
    print("cm_labels_included=" + ", ".join(cm_label_counts.keys()))

    expr_matrix_cm, adata_like_cm, expr_source = _choose_expression_source(
        adata_cm,
        layer_arg=args.layer,
        use_raw_arg=bool(args.use_raw),
    )

    extra_genes = _read_extra_genes(str(args.extra_genes_csv), top_n=int(args.top_extra))
    all_genes = list(dict.fromkeys(BASE_GENE_PANEL + extra_genes))
    gene_statuses = [_resolve_gene(g, adata_like_cm) for g in all_genes]
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

    genes_present = [st for st in gene_statuses if st.present and st.gene_idx is not None]
    if len(genes_present) == 0:
        raise RuntimeError("No genes from panel resolved in this expression namespace")

    expr_by_gene = {st.gene: get_feature_vector(expr_matrix_cm, int(st.gene_idx)) for st in genes_present}

    adata_embed, embed_note = _prepare_embedding_input(adata_cm, expr_matrix_cm, expr_source)
    embedding_specs, n_pcs_used = _compute_embedding_suite(
        adata_embed,
        seed=int(args.seed),
        k_pca=int(args.k_pca),
        umap_neighbors=[int(x) for x in args.umap_neighbors],
        umap_min_dist=[float(x) for x in args.umap_min_dist],
        umap_seeds=[int(x) for x in args.umap_seeds],
        tsne_perplexities=[int(x) for x in args.tsne_perplexity],
        tsne_seeds=[int(x) for x in args.tsne_seeds],
        warnings_log=warnings_log,
    )
    if len(embedding_specs) == 0:
        raise RuntimeError("No embeddings were successfully constructed")

    emb_map = {e.name: e for e in embedding_specs}
    embedding_order = [e.name for e in embedding_specs]

    # Score all tests.
    rows: list[dict[str, Any]] = []
    test_counter = 0
    total_tests = len(embedding_specs) * len(genes_present) * len(args.foreground_modes)

    for emb_i, emb in enumerate(embedding_specs):
        geom = _compute_geom(emb, int(args.n_bins))
        for gene_i, st in enumerate(genes_present):
            expr = np.asarray(expr_by_gene[st.gene], dtype=float)
            for mode_i, mode in enumerate(args.foreground_modes):
                sc = _score_one(
                    expr=expr,
                    geom=geom,
                    mode=str(mode),
                    q=float(args.q),
                    n_bins=int(args.n_bins),
                    n_perm=int(args.n_perm),
                    seed=int(args.seed + emb_i * 100000 + gene_i * 997 + mode_i * 31 + 13),
                    with_profiles=False,
                )

                rows.append(
                    {
                        "donor_id": donor_star,
                        "gene": st.gene,
                        "resolved_gene": st.resolved_gene,
                        "gene_idx": int(st.gene_idx),
                        "embedding_family": emb.family,
                        "embedding_name": emb.name,
                        "params_json": json.dumps(emb.params, sort_keys=True),
                        "foreground_mode": str(mode),
                        "q_param": sc["q_param"],
                        "prev": sc["prev"],
                        "n_fg": sc["n_fg"],
                        "n_cells": sc["n_cells"],
                        "T_obs": sc["T_obs"],
                        "p_T": sc["p_T"],
                        "q_T_within_embedding": np.nan,
                        "q_T_within_family": np.nan,
                        "Z_T": sc["Z_T"],
                        "coverage_C": sc["coverage_C"],
                        "peaks_K": sc["peaks_K"],
                        "phi_hat_deg": sc["phi_hat_deg"],
                        "underpowered_flag": bool(sc["underpowered_flag"]),
                        "class_label": "",
                    }
                )

                test_counter += 1
                if test_counter % 50 == 0 or test_counter == total_tests:
                    print(f"[Scoring] {test_counter}/{total_tests} tests completed")
                    pd.DataFrame(rows).to_csv(
                        tables_dir / "per_embedding_gene_scores.intermediate.csv",
                        index=False,
                    )

    scores = pd.DataFrame(rows)
    if scores.empty:
        raise RuntimeError("No scoring rows produced")

    # BH within embedding (primary) and within family.
    q_within_embed = np.full(len(scores), np.nan, dtype=float)
    for _, idx in scores.groupby(["embedding_name", "foreground_mode"], sort=False).groups.items():
        ids = np.asarray(list(idx), dtype=int)
        p = scores.loc[ids, "p_T"].to_numpy(dtype=float)
        fin = np.isfinite(p)
        if int(fin.sum()) == 0:
            continue
        qq = np.full_like(p, np.nan, dtype=float)
        qq[fin] = bh_fdr(p[fin])
        q_within_embed[ids] = qq
    scores["q_T_within_embedding"] = q_within_embed

    q_within_family = np.full(len(scores), np.nan, dtype=float)
    for _, idx in scores.groupby(["embedding_family", "foreground_mode"], sort=False).groups.items():
        ids = np.asarray(list(idx), dtype=int)
        p = scores.loc[ids, "p_T"].to_numpy(dtype=float)
        fin = np.isfinite(p)
        if int(fin.sum()) == 0:
            continue
        qq = np.full_like(p, np.nan, dtype=float)
        qq[fin] = bh_fdr(p[fin])
        q_within_family[ids] = qq
    scores["q_T_within_family"] = q_within_family

    scores["class_label"] = [
        _classify_test(float(qe), float(pk), bool(up))
        for qe, pk, up in zip(
            scores["q_T_within_embedding"].to_numpy(dtype=float),
            scores["peaks_K"].to_numpy(dtype=float),
            scores["underpowered_flag"].to_numpy(dtype=bool),
            strict=False,
        )
    ]
    scores.to_csv(tables_dir / "per_embedding_gene_scores.csv", index=False)

    # Primary mode for concordance summaries.
    primary_mode = "topq" if "topq" in list(args.foreground_modes) else list(args.foreground_modes)[0]
    s0 = scores.loc[scores["foreground_mode"] == primary_mode].copy()

    # Pairwise embedding concordance table (Spearman Z_T across genes).
    pair_rows: list[dict[str, Any]] = []
    for e1, e2 in combinations(embedding_order, 2):
        a = s0.loc[s0["embedding_name"] == e1, ["gene", "Z_T", "underpowered_flag", "embedding_family"]].rename(
            columns={"Z_T": "Z1", "underpowered_flag": "up1", "embedding_family": "family1"}
        )
        b = s0.loc[s0["embedding_name"] == e2, ["gene", "Z_T", "underpowered_flag", "embedding_family"]].rename(
            columns={"Z_T": "Z2", "underpowered_flag": "up2", "embedding_family": "family2"}
        )
        m = a.merge(b, on="gene", how="inner")
        valid = (
            np.isfinite(m["Z1"].to_numpy(dtype=float))
            & np.isfinite(m["Z2"].to_numpy(dtype=float))
            & (~m["up1"].to_numpy(dtype=bool))
            & (~m["up2"].to_numpy(dtype=bool))
        )
        n_valid = int(np.sum(valid))
        rho = _safe_spearman(m.loc[valid, "Z1"].to_numpy(dtype=float), m.loc[valid, "Z2"].to_numpy(dtype=float))
        fam1 = str(m["family1"].iloc[0]) if not m.empty else emb_map[e1].family
        fam2 = str(m["family2"].iloc[0]) if not m.empty else emb_map[e2].family
        pair_rows.append(
            {
                "embedding_a": e1,
                "embedding_b": e2,
                "family_a": fam1,
                "family_b": fam2,
                "family_pair": _pair_label(fam1, fam2),
                "n_genes": n_valid,
                "spearman_Z_T": rho,
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(tables_dir / "pairwise_embedding_concordance.csv", index=False)

    fam_summary = (
        pair_df.loc[pair_df["family_a"] != pair_df["family_b"]]
        .groupby("family_pair", as_index=False)
        .agg(
            n_pairs=("spearman_Z_T", "size"),
            mean_spearman_Z_T=("spearman_Z_T", "mean"),
            sd_spearman_Z_T=("spearman_Z_T", "std"),
        )
        .sort_values(by="family_pair", kind="mergesort")
        .reset_index(drop=True)
    )
    fam_summary.to_csv(tables_dir / "family_concordance_summary.csv", index=False)

    # Per-gene concordance summary.
    sum_rows: list[dict[str, Any]] = []
    for gene, sub in s0.groupby("gene", sort=False):
        sub = sub.copy().sort_values(by="embedding_name", kind="mergesort")

        localized = (
            np.isfinite(sub["q_T_within_embedding"].to_numpy(dtype=float))
            & (sub["q_T_within_embedding"].to_numpy(dtype=float) <= Q_SIG)
            & (~sub["underpowered_flag"].to_numpy(dtype=bool))
        )
        frac_localized = float(np.mean(localized)) if localized.size > 0 else np.nan

        loc_classes = sub.loc[localized, "class_label"].astype(str).tolist()
        frac_unimodal = (
            float(np.mean(np.array(loc_classes, dtype=object) == "Localized-unimodal"))
            if len(loc_classes) > 0
            else np.nan
        )

        pca_mask = sub["embedding_family"].astype(str).to_numpy() == "PCA"
        umap_mask = sub["embedding_family"].astype(str).to_numpy() == "UMAP"
        tsne_mask = sub["embedding_family"].astype(str).to_numpy() == "TSNE"

        loc_pca = bool(np.any(localized[pca_mask])) if int(np.sum(pca_mask)) > 0 else False
        loc_umap_maj = bool(np.mean(localized[umap_mask]) >= 0.5) if int(np.sum(umap_mask)) > 0 else False
        loc_tsne_maj = bool(np.mean(localized[tsne_mask]) >= 0.5) if int(np.sum(tsne_mask)) > 0 else False

        z_vals = sub["Z_T"].to_numpy(dtype=float)
        median_z = float(np.nanmedian(z_vals)) if np.isfinite(z_vals).any() else np.nan
        var_z = float(np.nanvar(z_vals)) if np.isfinite(z_vals).any() else np.nan

        z_umap = sub.loc[umap_mask, "Z_T"].to_numpy(dtype=float)
        mean_z_umap = float(np.nanmean(z_umap)) if np.isfinite(z_umap).any() else np.nan
        z_pca = sub.loc[pca_mask, "Z_T"].to_numpy(dtype=float)
        z_pca_val = float(z_pca[0]) if z_pca.size > 0 and np.isfinite(z_pca[0]) else np.nan

        pca_class = (
            str(sub.loc[pca_mask, "class_label"].iloc[0])
            if int(np.sum(pca_mask)) > 0
            else "Not-localized"
        )
        umap_maj_class = _mode_class(sub.loc[umap_mask, "class_label"].astype(str).tolist())
        tsne_maj_class = _mode_class(sub.loc[tsne_mask, "class_label"].astype(str).tolist())

        # Direction stability per family (localized calls only).
        def _fam_stats(mask: np.ndarray) -> tuple[float, float, float, int, float]:
            ph = sub.loc[mask & localized, "phi_hat_deg"].to_numpy(dtype=float)
            ph = ph[np.isfinite(ph)]
            if ph.size == 0:
                return np.nan, np.nan, np.nan, 0, np.nan
            mu, R, csd = _circular_stats_deg(ph)
            return mu, R, csd, int(ph.size), float(np.mean(ph))

        mu_pca, R_pca, csd_pca, nloc_pca, _ = _fam_stats(pca_mask)
        mu_umap, R_umap, csd_umap, nloc_umap, _ = _fam_stats(umap_mask)
        mu_tsne, R_tsne, csd_tsne, nloc_tsne, _ = _fam_stats(tsne_mask)

        if nloc_pca > 0 and nloc_umap > 0 and nloc_tsne > 0:
            _, R_cross, csd_cross = _circular_stats_deg(np.array([mu_pca, mu_umap, mu_tsne], dtype=float))
        else:
            R_cross, csd_cross = np.nan, np.nan

        if np.isfinite(frac_localized) and frac_localized == 0.0:
            category = "Never localized"
        elif loc_pca and loc_umap_maj and loc_tsne_maj:
            category = "PCA-confirmed"
        elif (not loc_pca) and (loc_umap_maj or loc_tsne_maj):
            category = "Nonlinear-only"
        else:
            category = "Inconsistent"

        robust_flag = bool(
            loc_pca
            and loc_umap_maj
            and loc_tsne_maj
            and np.isfinite(median_z)
            and (median_z >= 3.0)
            and np.isfinite(R_umap)
            and (R_umap >= 0.60)
            and np.isfinite(R_tsne)
            and (R_tsne >= 0.60)
        )

        sum_rows.append(
            {
                "gene": gene,
                "frac_localized": frac_localized,
                "frac_unimodal": frac_unimodal,
                "localized_in_PCA": loc_pca,
                "localized_in_UMAP_majority": loc_umap_maj,
                "localized_in_TSNE_majority": loc_tsne_maj,
                "PCA_class": pca_class,
                "UMAP_majority_class": umap_maj_class,
                "TSNE_majority_class": tsne_maj_class,
                "median_Z_T": median_z,
                "mean_Z_UMAP": mean_z_umap,
                "Z_T_PCA": z_pca_val,
                "var_Z_T": var_z,
                "R_PCA": R_pca,
                "circ_sd_PCA": csd_pca,
                "R_UMAP": R_umap,
                "circ_sd_UMAP": csd_umap,
                "R_TSNE": R_tsne,
                "circ_sd_TSNE": csd_tsne,
                "R_cross_family": R_cross,
                "circ_sd_cross_family": csd_cross,
                "n_localized_PCA": nloc_pca,
                "n_localized_UMAP": nloc_umap,
                "n_localized_TSNE": nloc_tsne,
                "category": category,
                "robust_embedding_concordant": robust_flag,
            }
        )

    gene_summary = pd.DataFrame(sum_rows).sort_values(by=["median_Z_T", "gene"], ascending=[False, True], kind="mergesort")
    gene_summary.to_csv(tables_dir / "per_gene_concordance_summary.csv", index=False)

    # Exemplar selection.
    ex_rows: list[dict[str, Any]] = []
    robust_top = gene_summary.loc[gene_summary["robust_embedding_concordant"]].sort_values(by="median_Z_T", ascending=False).head(3)
    nonlinear_top = gene_summary.loc[gene_summary["category"] == "Nonlinear-only"].sort_values(by="mean_Z_UMAP", ascending=False).head(3)
    inconsistent_top = gene_summary.loc[gene_summary["category"] == "Inconsistent"].sort_values(by="var_Z_T", ascending=False).head(2)

    for rank, (_, r) in enumerate(robust_top.iterrows(), start=1):
        ex_rows.append({"gene": r["gene"], "exemplar_group": "PCA-confirmed robust", "rank_metric": float(r["median_Z_T"]), "rank": rank})
    for rank, (_, r) in enumerate(nonlinear_top.iterrows(), start=1):
        ex_rows.append({"gene": r["gene"], "exemplar_group": "Nonlinear-only", "rank_metric": float(r["mean_Z_UMAP"]), "rank": rank})
    for rank, (_, r) in enumerate(inconsistent_top.iterrows(), start=1):
        ex_rows.append({"gene": r["gene"], "exemplar_group": "Inconsistent", "rank_metric": float(r["var_Z_T"]), "rank": rank})

    exemplar_df = pd.DataFrame(ex_rows)
    if not exemplar_df.empty:
        exemplar_df = exemplar_df.drop_duplicates(subset=["gene"], keep="first").reset_index(drop=True)
    exemplar_df.to_csv(tables_dir / "exemplar_selection.csv", index=False)

    # -----------------------------
    # Plots 00: overview
    # -----------------------------
    fig0, ax0 = plt.subplots(figsize=(10.0, 4.6))
    dc = donor_choice.sort_values(by="n_cm", ascending=False).reset_index(drop=True)
    colors = ["#d62728" if str(d) == donor_star else "#4c78a8" for d in dc["donor_id"].astype(str).tolist()]
    ax0.bar(np.arange(len(dc)), dc["n_cm"].to_numpy(dtype=float), color=colors)
    ax0.set_xticks(np.arange(len(dc)))
    ax0.set_xticklabels(dc["donor_id"].astype(str).tolist(), rotation=45, ha="right", fontsize=8)
    ax0.set_ylabel("Cardiomyocyte cells")
    ax0.set_title("Cardiomyocyte counts per donor (donor_star highlighted)")
    fig0.tight_layout()
    fig0.savefig(plots_dir / "00_overview" / "cm_counts_per_donor.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig0)

    rep_umap = "umap_nn30_md0.1_s0" if "umap_nn30_md0.1_s0" in emb_map else next((e.name for e in embedding_specs if e.family == "UMAP"), "pca2d")
    rep_tsne = "tsne_p30_s0" if "tsne_p30_s0" in emb_map else next((e.name for e in embedding_specs if e.family == "TSNE"), "pca2d")

    fig1, axes1 = plt.subplots(1, 3, figsize=(13.5, 4.3))
    reps = ["pca2d", rep_umap, rep_tsne]
    titles = ["PCA-2D", "UMAP representative", "t-SNE representative"]
    for ax, nm, ttl in zip(axes1, reps, titles, strict=False):
        coords = emb_map[nm].coords
        ax.scatter(coords[:, 0], coords[:, 1], c="#4c78a8", s=5, alpha=0.70, linewidths=0, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{ttl}\n{nm}", fontsize=9)
    fig1.tight_layout()
    fig1.savefig(plots_dir / "00_overview" / "representative_embedding_gallery.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig1)

    # -----------------------------
    # Plots 01: rank concordance
    # -----------------------------
    # Pairwise heatmap matrix
    n_emb = len(embedding_order)
    mat = np.full((n_emb, n_emb), np.nan, dtype=float)
    for i in range(n_emb):
        mat[i, i] = 1.0
    for _, r in pair_df.iterrows():
        a = embedding_order.index(str(r["embedding_a"]))
        b = embedding_order.index(str(r["embedding_b"]))
        mat[a, b] = float(r["spearman_Z_T"]) if np.isfinite(float(r["spearman_Z_T"])) else np.nan
        mat[b, a] = mat[a, b]

    fig2, ax2 = plt.subplots(figsize=(0.45 * n_emb + 4.6, 0.45 * n_emb + 3.8))
    im2 = ax2.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(np.arange(n_emb))
    ax2.set_xticklabels(embedding_order, rotation=90, fontsize=7)
    ax2.set_yticks(np.arange(n_emb))
    ax2.set_yticklabels(embedding_order, fontsize=7)
    ax2.set_title("Pairwise Spearman correlation of Z_T across embeddings")
    fig2.colorbar(im2, ax=ax2)
    fig2.tight_layout()
    fig2.savefig(plots_dir / "01_rank_concordance" / "pairwise_spearman_heatmap.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig2)

    # Boxplots by family pair.
    fig3, ax3 = plt.subplots(figsize=(8.2, 5.6))
    fam_order = ["PCA-vs-TSNE", "PCA-vs-UMAP", "TSNE-vs-UMAP"]
    data = []
    labels = []
    for fp in fam_order:
        vals = pair_df.loc[pair_df["family_pair"] == fp, "spearman_Z_T"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            data.append(vals)
            labels.append(fp)
    if len(data) > 0:
        ax3.boxplot(data, tick_labels=labels, patch_artist=True)
        ax3.set_ylabel("Spearman(Z_T)")
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "No valid pairwise correlations", ha="center", va="center")
    ax3.set_title("Rank concordance grouped by embedding-family pair")
    fig3.tight_layout()
    fig3.savefig(plots_dir / "01_rank_concordance" / "family_pair_boxplots.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig3)

    # PCA vs mean UMAP scatter
    fig4, ax4 = plt.subplots(figsize=(7.1, 5.8))
    if not gene_summary.empty:
        x = gene_summary["Z_T_PCA"].to_numpy(dtype=float)
        y = gene_summary["mean_Z_UMAP"].to_numpy(dtype=float)
        ax4.scatter(x, y, s=90, c="#4c78a8", alpha=0.88, edgecolors="black", linewidths=0.4)
        for _, r in gene_summary.iterrows():
            if np.isfinite(float(r["Z_T_PCA"])) and np.isfinite(float(r["mean_Z_UMAP"])):
                ax4.text(float(r["Z_T_PCA"]), float(r["mean_Z_UMAP"]) + 0.03, str(r["gene"]), fontsize=8)
        lo = np.nanmin(np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])) if (np.isfinite(x).any() and np.isfinite(y).any()) else 0.0
        hi = np.nanmax(np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])) if (np.isfinite(x).any() and np.isfinite(y).any()) else 1.0
        ax4.plot([lo, hi], [lo, hi], linestyle="--", color="#444")
        ax4.set_xlabel("Z_T (PCA-2D)")
        ax4.set_ylabel("mean Z_T (UMAP family)")
    else:
        ax4.axis("off")
    ax4.set_title("PCA baseline vs UMAP-family mean Z_T")
    fig4.tight_layout()
    fig4.savefig(plots_dir / "01_rank_concordance" / "PCA_vs_mean_UMAP_scatter.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig4)

    # -----------------------------
    # Plots 02: class concordance
    # -----------------------------
    # 1) stacked class frequency across embeddings per gene
    if s0.empty:
        _save_placeholder(plots_dir / "02_class_concordance" / "stacked_class_frequency.png", "Class frequency", "No scores")
    else:
        pivot_counts = (
            s0.groupby(["gene", "class_label"]).size().unstack(fill_value=0).reindex(columns=CLASS_ORDER, fill_value=0)
        )
        frac = pivot_counts.div(np.maximum(1, pivot_counts.sum(axis=1)), axis=0)
        genes_plot = frac.index.astype(str).tolist()
        fig5, ax5 = plt.subplots(figsize=(max(8.0, 0.7 * len(genes_plot)), 5.8))
        bottom = np.zeros(len(genes_plot), dtype=float)
        class_colors = {
            "Localized-unimodal": "#1f77b4",
            "Localized-multimodal": "#ff7f0e",
            "Not-localized": "#9e9e9e",
        }
        xloc = np.arange(len(genes_plot))
        for cls in CLASS_ORDER:
            vals = frac[cls].to_numpy(dtype=float)
            ax5.bar(xloc, vals, bottom=bottom, color=class_colors[cls], label=cls)
            bottom += vals
        ax5.set_xticks(xloc)
        ax5.set_xticklabels(genes_plot, rotation=45, ha="right")
        ax5.set_ylabel("Fraction across embeddings")
        ax5.set_title("Class frequency across embeddings by gene")
        ax5.legend(loc="upper right", fontsize=8)
        fig5.tight_layout()
        fig5.savefig(plots_dir / "02_class_concordance" / "stacked_class_frequency.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig5)

    # 2) confusion PCA class vs UMAP-majority class
    conf_labels = CLASS_ORDER
    conf_mat = np.zeros((len(conf_labels), len(conf_labels)), dtype=float)
    for _, r in gene_summary.iterrows():
        pca_c = str(r["PCA_class"])
        umap_c = str(r["UMAP_majority_class"])
        if pca_c in conf_labels and umap_c in conf_labels:
            i = conf_labels.index(pca_c)
            j = conf_labels.index(umap_c)
            conf_mat[i, j] += 1
    fig6, ax6 = plt.subplots(figsize=(7.0, 6.0))
    im6 = ax6.imshow(conf_mat, cmap="Blues", aspect="auto")
    ax6.set_xticks(np.arange(len(conf_labels)))
    ax6.set_xticklabels(conf_labels, rotation=45, ha="right")
    ax6.set_yticks(np.arange(len(conf_labels)))
    ax6.set_yticklabels(conf_labels)
    ax6.set_xlabel("UMAP-majority class")
    ax6.set_ylabel("PCA class")
    ax6.set_title("PCA class vs UMAP-majority class")
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax6.text(j, i, int(conf_mat[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig6.colorbar(im6, ax=ax6)
    fig6.tight_layout()
    fig6.savefig(plots_dir / "02_class_concordance" / "confusion_PCA_vs_UMAP_majority.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig6)

    # 3) overlap/set bars for localized calls across families
    comb_counts: dict[str, int] = {}
    for _, r in gene_summary.iterrows():
        key = (
            f"PCA:{int(bool(r['localized_in_PCA']))}|"
            f"UMAPmaj:{int(bool(r['localized_in_UMAP_majority']))}|"
            f"TSNEmaj:{int(bool(r['localized_in_TSNE_majority']))}"
        )
        comb_counts[key] = comb_counts.get(key, 0) + 1
    comb_items = sorted(comb_counts.items(), key=lambda kv: kv[1], reverse=True)
    fig7, ax7 = plt.subplots(figsize=(10.0, 4.8))
    if len(comb_items) > 0:
        labs = [k for k, _ in comb_items]
        vals = [v for _, v in comb_items]
        ax7.bar(np.arange(len(labs)), vals, color="#4c78a8")
        ax7.set_xticks(np.arange(len(labs)))
        ax7.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
        ax7.set_ylabel("# genes")
    else:
        ax7.axis("off")
    ax7.set_title("Overlap of localized calls across PCA / UMAP-majority / TSNE-majority")
    fig7.tight_layout()
    fig7.savefig(plots_dir / "02_class_concordance" / "localized_overlap_setbars.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig7)

    # -----------------------------
    # Plots 03: direction concordance
    # -----------------------------
    # 1) R_UMAP vs R_TSNE
    fig8, ax8 = plt.subplots(figsize=(7.2, 5.8))
    dsub = gene_summary.loc[
        np.isfinite(gene_summary["R_UMAP"].to_numpy(dtype=float))
        & np.isfinite(gene_summary["R_TSNE"].to_numpy(dtype=float))
    ]
    if not dsub.empty:
        x = dsub["R_UMAP"].to_numpy(dtype=float)
        y = dsub["R_TSNE"].to_numpy(dtype=float)
        col = np.where(dsub["robust_embedding_concordant"].to_numpy(dtype=bool), "#2ca02c", "#4c78a8")
        ax8.scatter(x, y, s=95, c=col, alpha=0.88, edgecolors="black", linewidths=0.4)
        for _, r in dsub.iterrows():
            ax8.text(float(r["R_UMAP"]), float(r["R_TSNE"]) + 0.01, str(r["gene"]), fontsize=8)
        ax8.axvline(0.60, color="#333", linestyle="--")
        ax8.axhline(0.60, color="#333", linestyle="--")
        ax8.set_xlabel("R_UMAP")
        ax8.set_ylabel("R_TSNE")
    else:
        ax8.axis("off")
    ax8.set_title("Direction concentration concordance across non-linear families")
    fig8.tight_layout()
    fig8.savefig(plots_dir / "03_direction_concordance" / "R_UMAP_vs_R_TSNE.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig8)

    # 2) mini circular plots for UMAP phi points
    umap_names = [e.name for e in embedding_specs if e.family == "UMAP"]
    genes_plot = gene_summary["gene"].astype(str).tolist()
    n_cols = 4
    n_rows = int(np.ceil(max(1, len(genes_plot)) / n_cols))
    fig9 = plt.figure(figsize=(4.0 * n_cols, 3.4 * n_rows))
    for i, gene in enumerate(genes_plot):
        ax = fig9.add_subplot(n_rows, n_cols, i + 1, projection="polar")
        sub = s0.loc[(s0["gene"] == gene) & (s0["embedding_name"].isin(umap_names))]
        loc = (
            np.isfinite(sub["q_T_within_embedding"].to_numpy(dtype=float))
            & (sub["q_T_within_embedding"].to_numpy(dtype=float) <= Q_SIG)
        )
        phi = sub.loc[loc, "phi_hat_deg"].to_numpy(dtype=float)
        phi = phi[np.isfinite(phi)]
        if phi.size == 0:
            ax.text(0.5, 0.5, "no localized", transform=ax.transAxes, ha="center", va="center", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            rad = np.deg2rad(phi)
            ax.scatter(rad, np.ones_like(rad), s=24, c="#1f77b4", alpha=0.85)
            mu, R, csd = _circular_stats_deg(phi)
            ax.plot([np.deg2rad(mu), np.deg2rad(mu)], [0.0, 1.15], color="#d62728", linewidth=1.8)
            ax.text(0.02, 0.02, f"R={R:.2f}", transform=ax.transAxes, fontsize=7)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_rticks([])
        ax.set_title(gene, fontsize=8)
    fig9.suptitle("UMAP-family localized phi points per gene", y=0.995)
    fig9.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig9.savefig(plots_dir / "03_direction_concordance" / "umap_phi_small_multiples.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig9)

    # 3) cross-family direction drift arrows (PCA/UMAP/TSNE means)
    fig10 = plt.figure(figsize=(4.0 * n_cols, 3.4 * n_rows))
    for i, gene in enumerate(genes_plot):
        ax = fig10.add_subplot(n_rows, n_cols, i + 1, projection="polar")
        row = gene_summary.loc[gene_summary["gene"] == gene]
        if row.empty:
            ax.axis("off")
            continue
        rr = row.iloc[0]
        fam_mu = []
        fam_lbl = []
        for fam, col_mu in [("PCA", "R_PCA"), ("UMAP", "R_UMAP"), ("TSNE", "R_TSNE")]:
            sub = s0.loc[(s0["gene"] == gene) & (s0["embedding_family"] == fam)]
            loc = (
                np.isfinite(sub["q_T_within_embedding"].to_numpy(dtype=float))
                & (sub["q_T_within_embedding"].to_numpy(dtype=float) <= Q_SIG)
            )
            phi = sub.loc[loc, "phi_hat_deg"].to_numpy(dtype=float)
            phi = phi[np.isfinite(phi)]
            if phi.size > 0:
                mu, _, _ = _circular_stats_deg(phi)
                fam_mu.append(mu)
                fam_lbl.append(fam)
        if len(fam_mu) == 0:
            ax.text(0.5, 0.5, "no localized", transform=ax.transAxes, ha="center", va="center", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            colors = {"PCA": "#2ca02c", "UMAP": "#1f77b4", "TSNE": "#ff7f0e"}
            for mu, fam in zip(fam_mu, fam_lbl, strict=False):
                ang = np.deg2rad(mu)
                ax.plot([ang, ang], [0.0, 1.0], color=colors[fam], linewidth=1.8, label=fam)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_rticks([])
            ax.legend(loc="upper right", fontsize=6, frameon=True)
        ax.set_title(gene, fontsize=8)
    fig10.suptitle("Cross-family direction means (localized calls)", y=0.995)
    fig10.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig10.savefig(plots_dir / "03_direction_concordance" / "cross_family_direction_arrows.png", dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig10)

    # -----------------------------
    # Plots 04: exemplar panels
    # -----------------------------
    if exemplar_df.empty:
        _save_placeholder(plots_dir / "04_exemplar_panels" / "no_exemplars.png", "Exemplar panels", "No exemplars selected")
    else:
        rep_umap2 = "umap_nn50_md0.5_s1" if "umap_nn50_md0.5_s1" in emb_map else rep_umap
        rep_tsne2 = "tsne_p50_s1" if "tsne_p50_s1" in emb_map else rep_tsne

        for _, ex in exemplar_df.iterrows():
            gene = str(ex["gene"])
            grp = str(ex["exemplar_group"])
            if gene not in expr_by_gene:
                continue

            expr = np.asarray(expr_by_gene[gene], dtype=float)
            expr_log = np.log1p(np.maximum(expr, 0.0))
            vmin = float(np.quantile(expr_log, 0.01)) if np.isfinite(expr_log).any() else 0.0
            vmax = float(np.quantile(expr_log, 0.99)) if np.isfinite(expr_log).any() else 1.0
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6

            sub_gene = s0.loc[s0["gene"] == gene].sort_values(by="Z_T", ascending=True)
            ordered = [
                "pca2d",
                rep_umap,
                rep_umap2,
                rep_tsne,
                rep_tsne2,
            ]
            worst = None
            for nm in sub_gene["embedding_name"].astype(str).tolist():
                if nm not in ordered:
                    worst = nm
                    break
            if worst is None:
                worst = str(sub_gene["embedding_name"].iloc[0]) if not sub_gene.empty else "pca2d"
            ordered.append(worst)

            # ensure unique and existing
            seen = set()
            plot_names: list[str] = []
            for nm in ordered:
                if nm in emb_map and nm not in seen:
                    seen.add(nm)
                    plot_names.append(nm)
                if len(plot_names) >= 6:
                    break
            if len(plot_names) < 6:
                for nm in embedding_order:
                    if nm not in seen:
                        seen.add(nm)
                        plot_names.append(nm)
                    if len(plot_names) >= 6:
                        break

            fig = plt.figure(figsize=(16.0, 11.0))
            gs = fig.add_gridspec(3, 4, wspace=0.28, hspace=0.30)

            feat_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
            for (r, c), nm in zip(feat_positions, plot_names, strict=False):
                ax = fig.add_subplot(gs[r, c])
                _plot_feature(ax, emb_map[nm].coords, expr_log, nm, vmin, vmax)

            # polar per family (representative)
            family_rep = {
                "PCA": "pca2d",
                "UMAP": rep_umap,
                "TSNE": rep_tsne,
            }
            polar_pos = {"PCA": (0, 3), "UMAP": (1, 3), "TSNE": (2, 0)}
            for fam in ["PCA", "UMAP", "TSNE"]:
                nm = family_rep[fam]
                axp = fig.add_subplot(gs[polar_pos[fam][0], polar_pos[fam][1]], projection="polar")
                if nm not in emb_map:
                    axp.axis("off")
                    continue
                geom = _compute_geom(emb_map[nm], int(args.n_bins))
                sc = _score_one(
                    expr=expr,
                    geom=geom,
                    mode=primary_mode,
                    q=float(args.q),
                    n_bins=int(args.n_bins),
                    n_perm=int(args.n_perm),
                    seed=int(args.seed + 900000 + (_stable_token(gene) * 131 + _stable_token(nm)) % 100000),
                    with_profiles=True,
                )
                row = s0.loc[(s0["gene"] == gene) & (s0["embedding_name"] == nm)]
                if row.empty:
                    stats_txt = "missing"
                else:
                    rr = row.iloc[0]
                    stats_txt = (
                        f"Z={float(rr['Z_T']):.2f}\n"
                        f"q={float(rr['q_T_within_embedding']):.2e}\n"
                        f"C={float(rr['coverage_C']):.3f}\n"
                        f"K={int(rr['peaks_K']) if np.isfinite(rr['peaks_K']) else -1}"
                    )
                _plot_polar_family(
                    axp,
                    title=f"{fam}: {nm}",
                    e_obs=np.asarray(sc["E_obs"], dtype=float),
                    null_e=np.asarray(sc["null_E"], dtype=float) if sc["null_E"] is not None else None,
                    stats_text=stats_txt,
                )

            # table inset of all embeddings
            ax_tbl = fig.add_subplot(gs[2, 1:4])
            ax_tbl.axis("off")
            tab = s0.loc[s0["gene"] == gene, ["embedding_name", "embedding_family", "Z_T", "q_T_within_embedding", "class_label"]].copy()
            tab = tab.sort_values(by=["embedding_family", "embedding_name"], kind="mergesort")
            if tab.empty:
                ax_tbl.text(0.5, 0.5, "No per-embedding rows", ha="center", va="center")
            else:
                tab["Z_T"] = tab["Z_T"].map(lambda x: f"{float(x):.2f}" if np.isfinite(float(x)) else "nan")
                tab["q_T_within_embedding"] = tab["q_T_within_embedding"].map(
                    lambda x: f"{float(x):.2e}" if np.isfinite(float(x)) else "nan"
                )
                table_obj = ax_tbl.table(cellText=tab.values, colLabels=tab.columns, loc="center")
                table_obj.auto_set_font_size(False)
                table_obj.set_fontsize(7)
                table_obj.scale(1.0, 1.15)
            ax_tbl.set_title("Per-embedding Z/q/class summary", fontsize=9)

            fig.suptitle(f"Exemplar: {gene} [{grp}]", y=0.995)
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
            fig.savefig(plots_dir / "04_exemplar_panels" / f"exemplar_{gene}.png", dpi=DEFAULT_PLOT_STYLE.dpi)
            plt.close(fig)

    # -----------------------------
    # Plots 05: embedding gallery summaries
    # -----------------------------
    # Z heatmap genes x embeddings
    if s0.empty:
        _save_placeholder(plots_dir / "05_embedding_gallery" / "heatmap_ZT.png", "Embedding gallery", "No scores")
        _save_placeholder(plots_dir / "05_embedding_gallery" / "heatmap_class.png", "Embedding gallery", "No scores")
    else:
        g_order = gene_summary["gene"].astype(str).tolist()
        z_mat = (
            s0.pivot(index="gene", columns="embedding_name", values="Z_T")
            .reindex(index=g_order, columns=embedding_order)
            .to_numpy(dtype=float)
        )
        z_plot = np.nan_to_num(z_mat, nan=0.0)

        fig11, ax11 = plt.subplots(figsize=(0.5 * len(embedding_order) + 4.0, 0.4 * len(g_order) + 3.0))
        im11 = ax11.imshow(z_plot, aspect="auto", cmap="magma", vmin=0, vmax=max(6.0, float(np.nanpercentile(z_plot, 95))))
        ax11.set_xticks(np.arange(len(embedding_order)))
        ax11.set_xticklabels(embedding_order, rotation=90, fontsize=7)
        ax11.set_yticks(np.arange(len(g_order)))
        ax11.set_yticklabels(g_order, fontsize=8)
        ax11.set_title("Z_T across embeddings")
        fig11.colorbar(im11, ax=ax11)
        fig11.tight_layout()
        fig11.savefig(plots_dir / "05_embedding_gallery" / "heatmap_ZT.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig11)

        cls_mat_raw = (
            s0.assign(class_int=s0["class_label"].map(CLASS_TO_INT).fillna(0).astype(int))
            .pivot(index="gene", columns="embedding_name", values="class_int")
            .reindex(index=g_order, columns=embedding_order)
            .to_numpy(dtype=float)
        )
        cls_plot = np.nan_to_num(cls_mat_raw, nan=0.0)

        fig12, ax12 = plt.subplots(figsize=(0.5 * len(embedding_order) + 4.0, 0.4 * len(g_order) + 3.0))
        im12 = ax12.imshow(cls_plot, aspect="auto", cmap="viridis", vmin=0, vmax=2)
        ax12.set_xticks(np.arange(len(embedding_order)))
        ax12.set_xticklabels(embedding_order, rotation=90, fontsize=7)
        ax12.set_yticks(np.arange(len(g_order)))
        ax12.set_yticklabels(g_order, fontsize=8)
        ax12.set_title("Class labels across embeddings (0=Not,1=Uni,2=Multi)")
        fig12.colorbar(im12, ax=ax12)
        fig12.tight_layout()
        fig12.savefig(plots_dir / "05_embedding_gallery" / "heatmap_class.png", dpi=DEFAULT_PLOT_STYLE.dpi)
        plt.close(fig12)

    # README
    _write_readme(
        out_root / "README.txt",
        args=args,
        donor_key=donor_key,
        label_key=label_key,
        donor_star=donor_star,
        expr_source=expr_source,
        embed_note=f"{embed_note}; n_pcs_used={n_pcs_used}",
        n_cells_donor=int(adata_donor.n_obs),
        n_cells_cm=int(adata_cm.n_obs),
        cm_label_counts={str(k): int(v) for k, v in cm_label_counts.items()},
        n_embeddings=len(embedding_specs),
        n_genes_present=len(genes_present),
        warnings_log=warnings_log,
    )

    # Verify required outputs.
    required = [
        tables_dir / "donor_choice.csv",
        tables_dir / "gene_panel_status.csv",
        tables_dir / "per_embedding_gene_scores.csv",
        tables_dir / "pairwise_embedding_concordance.csv",
        tables_dir / "family_concordance_summary.csv",
        tables_dir / "per_gene_concordance_summary.csv",
        tables_dir / "exemplar_selection.csv",
        plots_dir / "00_overview",
        plots_dir / "01_rank_concordance",
        plots_dir / "02_class_concordance",
        plots_dir / "03_direction_concordance",
        plots_dir / "04_exemplar_panels",
        plots_dir / "05_embedding_gallery",
        out_root / "README.txt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing required outputs: " + ", ".join(missing))

    print(f"donor_key_used={donor_key}")
    print(f"label_key_used={label_key}")
    print(f"donor_star={donor_star}")
    print(f"expression_source_used={expr_source}")
    print(f"n_embeddings={len(embedding_specs)}")
    print(f"n_genes_present={len(genes_present)}")
    print(f"foreground_modes={','.join(args.foreground_modes)}")
    print(f"primary_mode={primary_mode}")
    print(f"n_tests={len(scores)}")
    print(
        "robust_embedding_concordant_genes="
        f"{int(np.sum(gene_summary['robust_embedding_concordant'].to_numpy(dtype=bool)) if not gene_summary.empty else 0)}"
    )
    print(f"results_root={out_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
