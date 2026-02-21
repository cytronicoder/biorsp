"""Genome-wide BioRSP screening pipeline (donor-aware, within-strata)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from biorsp.config import load_json_config
from biorsp.geometry import compute_angles, compute_vantage
from biorsp.moran import extract_weights, morans_i
from biorsp.pipeline_utils import (
    bh_fdr,
    circular_sd,
)
from biorsp.pipeline_utils import (
    detect_obs_col as _detect_obs_col,
)
from biorsp.pipeline_utils import (
    normalize_label as _normalize_label,
)
from biorsp.pipeline_utils import (
    plot_null_hist as _plot_null_hist,
)
from biorsp.pipeline_utils import (
    plot_qq as _plot_qq,
)
from biorsp.pipeline_utils import (
    setup_logger as _setup_logger,
)
from biorsp.rsp import compute_rsp_profile_from_boolean, plot_rsp_polar
from biorsp.staged_pipeline import run_scope_staged
from biorsp.utils import ensure_dir


def _get_scanpy():
    import scanpy as sc

    return sc


def load_config(path: str) -> dict[str, Any]:
    """Load genome-wide pipeline config from strict JSON."""
    return load_json_config(path)


def permute_values_within_donor(
    values: np.ndarray, donor_to_idx: dict[str, np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Shuffle continuous values within donor blocks."""
    values = np.asarray(values).ravel()
    out = values.copy()
    for idx in donor_to_idx.values():
        idx_arr = np.asarray(idx, dtype=int)
        if idx_arr.size <= 1:
            continue
        out[idx_arr] = values[idx_arr][rng.permutation(idx_arr.size)]
    return out


def compute_rsp_profile_continuous(
    weights: np.ndarray, angles: np.ndarray, n_bins: int
) -> tuple[np.ndarray, float, float]:
    """
    Continuous-weight RSP: compare weighted angular mass to uniform cell distribution.

    E_phi[b] = (sum_w_b / sum_w) - (n_b / N)
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != angles.size:
        raise ValueError("weights and angles must have same length")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    if np.all(w == 0):
        raise ValueError("weights all zero")

    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1, endpoint=True)
    bin_idx = np.digitize(angles, bin_edges, right=False) - 1
    bin_idx = np.where(bin_idx == n_bins, n_bins - 1, bin_idx)

    w_b = np.bincount(bin_idx, weights=w, minlength=n_bins)
    n_b = np.bincount(bin_idx, minlength=n_bins)
    pW = w_b / np.sum(w)
    pC = n_b / float(w.size)
    E_phi = pW - pC

    E_max = float(np.max(E_phi))
    b_max = int(np.argmax(E_phi))
    phi_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    phi_max = float(phi_centers[b_max] % (2 * np.pi))
    return E_phi, phi_max, E_max


def _compute_embeddings(
    adata,
    sc_module,
    seeds: list[int],
    n_pcs: int,
    n_neighbors: int,
    umap_min_dist: float,
    umap_spread: float,
) -> dict[str, np.ndarray]:
    n_comps = min(n_pcs, max(2, min(adata.n_obs - 1, adata.n_vars - 1)))
    if "X_pca" not in adata.obsm:
        sc_module.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")
    sc_module.pp.neighbors(
        adata, n_neighbors=n_neighbors, n_pcs=n_comps, use_rep="X_pca"
    )
    embeddings: dict[str, np.ndarray] = {}
    for seed in seeds:
        sc_module.tl.umap(
            adata,
            random_state=int(seed),
            min_dist=umap_min_dist,
            spread=umap_spread,
        )
        key = f"umap_seed{seed}"
        embeddings[key] = adata.obsm["X_umap"].copy()
        adata.obsm[f"X_umap_seed{seed}"] = embeddings[key]
    embeddings["pca2d"] = adata.obsm["X_pca"][:, :2].copy()
    adata.obsm["X_pca2d"] = embeddings["pca2d"]
    return embeddings


def _subset_stratum(
    adata,
    celltype_col: str,
    donor_col: str,
    labels: list[str],
    min_donors: int,
    min_cells_per_donor: int,
) -> tuple[object | None, int, int, bool]:
    obs_labels = adata.obs[celltype_col].astype(str)
    norm = {_normalize_label(x) for x in labels}
    mask = obs_labels.str.lower().str.strip().isin(norm)
    if mask.sum() == 0:
        return None, 0, 0, False
    adata_s = adata[mask].copy()
    donor_counts = adata_s.obs[donor_col].value_counts()
    keep = donor_counts[donor_counts >= min_cells_per_donor].index
    adata_s = adata_s[adata_s.obs[donor_col].isin(keep)].copy()
    n_donors = int(adata_s.obs[donor_col].nunique())
    n_cells = int(adata_s.n_obs)
    inferential = n_donors >= min_donors
    return adata_s, n_cells, n_donors, inferential


def _get_matrix(adata, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers.")
    return adata.layers[layer]


def _ambient_flags(var_names: Iterable[str]) -> set[str]:
    flags = set()
    for g in var_names:
        g_upper = str(g).upper()
        if (
            g_upper.startswith("MT-")
            or g_upper.startswith("RPL")
            or g_upper.startswith("RPS")
        ):
            flags.add(str(g))
        if g_upper in {"MALAT1", "FOS", "JUN", "JUNB", "FOSB", "FOSL1", "FOSL2"}:
            flags.add(str(g))
    return flags


def _plot_umap_with_arrow(
    emb: np.ndarray,
    values: np.ndarray,
    phi_max: float,
    out_png: Path,
    title: str,
    subtitle: str,
    is_ambient: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    sca = ax.scatter(emb[:, 0], emb[:, 1], c=values, s=5, cmap="viridis", linewidths=0)
    ax.set_title(title + (" (ambient)" if is_ambient else ""))
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    # arrow from vantage
    v = compute_vantage(emb)
    span = max(np.ptp(emb[:, 0]), np.ptp(emb[:, 1]))
    length = 0.25 * span
    dx = length * math.cos(phi_max)
    dy = length * math.sin(phi_max)
    ax.annotate(
        "",
        xy=(v[0] + dx, v[1] + dy),
        xytext=(v[0], v[1]),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png.as_posix(), dpi=150)
    plt.close(fig)


def _plot_phi_compass(phi_deg: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    angles = np.deg2rad(phi_deg)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="polar")
    ax.hist(angles, bins=16)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png.as_posix(), dpi=150)
    plt.close(fig)


def _plot_phi_stability_bar(
    phi_sd_deg: np.ndarray, labels: list[str], out_png: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(phi_sd_deg)), phi_sd_deg)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("phi_sd_deg")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png.as_posix(), dpi=150)
    plt.close(fig)


def _plot_scatter(
    x,
    y,
    labels,
    out_png: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    highlight: set[str],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        x, y, c=["red" if label in highlight else "black" for label in labels], s=10
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for xi, yi, label in zip(x, y, labels):
        if label in highlight:
            ax.text(xi, yi, label, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png.as_posix(), dpi=150)
    plt.close(fig)


def _spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _safe_moran(
    *,
    expr: np.ndarray,
    W,
    logger: logging.Logger,
    stratum_name: str,
    feature: str,
) -> float:
    """Compute Moran's I with explicit warning for expected data issues."""
    try:
        return float(morans_i(expr.astype(float), W, row_standardize=False))
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Moran skipped: stratum=%s feature=%s reason=%s",
            stratum_name,
            feature,
            exc,
        )
        return float("nan")


def _prepare_genomewide_dirs(outdir: Path) -> tuple[Path, Path, Path]:
    results_dir = outdir / "results"
    figures_dir = outdir / "figures" / "genomewide"
    logs_dir = outdir / "logs"
    ensure_dir(results_dir.as_posix())
    ensure_dir(figures_dir.as_posix())
    ensure_dir(logs_dir.as_posix())
    return results_dir, figures_dir, logs_dir


def _write_genomewide_outputs(
    *,
    results_dir: Path,
    results_rows: list[pd.DataFrame],
    top_hits_rows: list[dict[str, Any]],
    gene_filtering_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    if results_rows:
        results_df = pd.concat(results_rows, ignore_index=True)
    else:
        results_df = pd.DataFrame()
    results_df.to_csv(results_dir / "biorsp_genomewide_results.csv", index=False)
    pd.DataFrame(top_hits_rows).to_csv(
        results_dir / "biorsp_genomewide_top_hits.csv", index=False
    )
    pd.DataFrame(gene_filtering_rows).to_csv(
        results_dir / "gene_filtering.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        results_dir / "genomewide_summary.csv", index=False
    )


def _gene_filtering(
    mat,
    var_names: list[str],
    min_detect_frac: float,
    min_detect_n: int,
    min_var: float,
) -> tuple[list[str], dict[str, int], np.ndarray]:
    if sp.issparse(mat):
        detect_n = np.asarray(mat.getnnz(axis=0)).ravel()
        mean = np.asarray(mat.mean(axis=0)).ravel()
        mean_sq = np.asarray(mat.power(2).mean(axis=0)).ravel()
    else:
        detect_n = np.sum(mat > 0, axis=0)
        mean = np.mean(mat, axis=0)
        mean_sq = np.mean(mat * mat, axis=0)
    var = mean_sq - mean * mean
    n_cells = mat.shape[0]
    detect_frac = detect_n / max(1, n_cells)

    keep = (
        (detect_frac >= min_detect_frac) & (detect_n >= min_detect_n) & (var >= min_var)
    )
    kept_genes = [g for g, k in zip(var_names, keep) if k]

    counts = {
        "total": int(len(var_names)),
        "filtered_low_detect": int(np.sum(detect_frac < min_detect_frac)),
        "filtered_low_n": int(np.sum(detect_n < min_detect_n)),
        "filtered_low_var": int(np.sum(var < min_var)),
        "kept": int(np.sum(keep)),
    }
    return kept_genes, counts, detect_frac


def run_genomewide_pipeline(config_path: str) -> None:
    cfg = load_config(config_path)
    sc = _get_scanpy()

    outdir = Path(cfg.get("outdir", "."))
    results_dir, figures_dir, logs_dir = _prepare_genomewide_dirs(outdir)

    logger = _setup_logger(logs_dir / "biorsp_genomewide.log", "biorsp_genomewide")

    h5ad_path = cfg.get("h5ad_path", "adata_embed_graph.h5ad")
    if not Path(h5ad_path).exists():
        raise FileNotFoundError(
            f"Input file not found: {h5ad_path}. Update configs/biorsp_genomewide.json."
        )

    donor_col = cfg.get("donor_col")
    celltype_col = cfg.get("celltype_col")
    batch_col = cfg.get("batch_col")

    seeds = [int(s) for s in cfg.get("umap_seeds", [0, 1, 2, 3, 4])]
    reference_embedding = cfg.get("reference_embedding_name", "X_umap_seed0")
    bins = int(cfg.get("bins", 72))
    stage1_perms = int(cfg.get("stage1_perms", 100))
    stage2_perms = int(cfg.get("stage2_perms", 1000))
    stage1_p_cutoff = cfg.get("stage1_p_cutoff", 0.05)
    stage1_top_k = cfg.get("stage1_top_k", None)
    min_detect_frac = float(cfg.get("min_detect_frac", 0.02))
    min_detect_n = int(cfg.get("min_detect_n", 50))
    min_var = float(cfg.get("min_var", 0.0))
    feature_mode = cfg.get("feature_mode", "continuous")
    thresholds_to_validate = cfg.get("thresholds_to_validate", ["t0", "q90", "q95"])
    topn_plots = int(cfg.get("topN_plots_per_stratum", 20))
    min_donors = int(cfg.get("min_donors", 3))
    min_cells_per_donor = int(cfg.get("min_cells_per_donor", 200))
    n_pcs = int(cfg.get("pca_n_comps", 50))
    n_neighbors = int(cfg.get("neighbors_k", 15))
    umap_min_dist = float(cfg.get("umap_min_dist", 0.3))
    umap_spread = float(cfg.get("umap_spread", 1.0))
    seed = int(cfg.get("seed", 0))
    resume = bool(cfg.get("resume", True))
    include_modules = bool(cfg.get("include_modules", False))

    # staged/discovery mode defaults
    staged_pipeline_enabled = bool(cfg.get("staged_pipeline_enabled", True))
    discovery_mode = bool(cfg.get("discovery_mode", True))
    pipeline_mode_default = "compute" if discovery_mode else "full"
    pipeline_mode = str(cfg.get("pipeline_mode", pipeline_mode_default)).strip().lower()
    if pipeline_mode not in {"compute", "plot", "full"}:
        pipeline_mode = pipeline_mode_default
    plot_top_k = int(cfg.get("plot_top_k", 20))
    bins_screen = int(cfg.get("bins_screen", 36))
    bins_confirm = int(cfg.get("bins_confirm", 72))
    perm_init = int(cfg.get("perm_init", 100))
    perm_mid = int(cfg.get("perm_mid", 300))
    perm_final = int(cfg.get("perm_final", 1000))
    p_escalate_1 = float(cfg.get("p_escalate_1", 0.2))
    p_escalate_2 = float(cfg.get("p_escalate_2", 0.05))
    stage1_top_k_global = int(
        cfg.get("stage1_top_k_global", cfg.get("stage1_top_k", 2000))
    )
    stage1_top_k_mega = int(cfg.get("stage1_top_k_mega", 1000))
    stage1_top_k_cluster = int(cfg.get("stage1_top_k_cluster", 300))
    stage2_top_k_global = int(cfg.get("stage2_top_k_global", 200))
    stage2_top_k_mega = int(cfg.get("stage2_top_k_mega", 80))
    stage2_top_k_cluster = int(cfg.get("stage2_top_k_cluster", 30))
    min_prev_global = float(cfg.get("min_prev_global", 0.01))
    min_prev_cluster = float(cfg.get("min_prev_cluster", 0.03))
    min_fg_global = int(cfg.get("min_fg_global", 50))
    min_fg_cluster = int(cfg.get("min_fg_cluster", 30))
    skip_umap_plots = bool(cfg.get("skip_umap_plots", discovery_mode))
    skip_pair_plots = bool(cfg.get("skip_pair_plots", discovery_mode))
    skip_rsp_plots = bool(cfg.get("skip_rsp_plots", discovery_mode))

    all_genes_scopes = cfg.get("all_genes_scopes", [])
    if (
        isinstance(all_genes_scopes, list)
        and len([s for s in all_genes_scopes if s]) > 1
    ):
        raise ValueError("Only one scope may run --all_genes at a time.")

    expr_layer_binary = cfg.get("binary_expr_layer", None)
    expr_layer_continuous = cfg.get("continuous_expr_layer", "lognorm")

    strata = cfg.get("strata", {})
    if not strata:
        raise ValueError("Config missing strata mapping.")

    adata = sc.read_h5ad(h5ad_path)
    donor_col = _detect_obs_col(
        adata,
        donor_col,
        ["donor", "hubmap_id", "donor_id", "sample", "subject", "individual"],
    )
    celltype_col = _detect_obs_col(
        adata,
        celltype_col,
        [
            "cell_type",
            "celltype",
            "cell_type_l2",
            "cell_type_l1",
            "celltype_l2",
            "celltype_l1",
            "azimuth_label",
            "predicted_label",
        ],
    )
    if batch_col is None:
        for c in ["batch", "batch_id", "library", "dataset", "sample_id"]:
            if c in adata.obs.columns:
                batch_col = c
                break

    # ensure lognorm layer for continuous
    if expr_layer_continuous == "lognorm":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.layers["lognorm"] = adata.X.copy()

    ambient_flags = _ambient_flags(adata.var_names)

    results_rows = []
    top_hits_rows = []
    gene_filtering_rows = []
    summary_rows = []

    rng = np.random.default_rng(seed)

    for stratum_name, labels in strata.items():
        adata_s, n_cells, n_donors, inferential = _subset_stratum(
            adata, celltype_col, donor_col, labels, min_donors, min_cells_per_donor
        )
        if adata_s is None:
            continue

        logger.info(
            "Stratum %s: n_cells=%s n_donors=%s inferential=%s",
            stratum_name,
            n_cells,
            n_donors,
            inferential,
        )

        stratum_dir = results_dir / "genomewide" / stratum_name.replace(" ", "_")
        ensure_dir(stratum_dir.as_posix())
        fig_dir = figures_dir / stratum_name.replace(" ", "_")
        ensure_dir(fig_dir.as_posix())

        embeddings = _compute_embeddings(
            adata_s,
            sc,
            seeds,
            n_pcs,
            n_neighbors,
            umap_min_dist,
            umap_spread,
        )
        angles_by_embed = {
            k: compute_angles(v, compute_vantage(v)) for k, v in embeddings.items()
        }

        # reference embedding
        ref_key = reference_embedding.replace("X_", "")
        if ref_key not in embeddings:
            ref_key = f"umap_seed{seeds[0]}"
        angles_ref = angles_by_embed[ref_key]

        # neighbors and Moran
        if "connectivities" not in adata_s.obsp:
            sc.pp.neighbors(
                adata_s, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep="X_pca"
            )
        W = extract_weights(adata_s)

        donor_ids = np.asarray(adata_s.obs[donor_col])
        unique_donors = np.unique(donor_ids)
        donor_to_idx = {
            str(d): np.nonzero(donor_ids == d)[0].astype(int) for d in unique_donors
        }

        # gene filtering
        mat_cont = _get_matrix(adata_s, expr_layer_continuous)
        var_names = list(adata_s.var_names)
        kept_genes, counts, detect_frac = _gene_filtering(
            mat_cont, var_names, min_detect_frac, min_detect_n, min_var
        )
        gene_filtering_rows.append(
            {
                "stratum": stratum_name,
                "total_genes": counts["total"],
                "filtered_low_detect": counts["filtered_low_detect"],
                "filtered_low_n": counts["filtered_low_n"],
                "filtered_low_var": counts["filtered_low_var"],
                "kept": counts["kept"],
                "min_detect_frac": min_detect_frac,
                "min_detect_n": min_detect_n,
                "min_var": min_var,
            }
        )

        # optional module scores
        module_scores = []
        if include_modules:
            for module_name, gene_list in cfg.get("module_sets", {}).items():
                present = [g for g in gene_list if g in adata_s.var_names]
                if len(present) < 2:
                    continue
                score_name = f"score_{module_name}"
                sc.tl.score_genes(
                    adata_s, gene_list=present, score_name=score_name, use_raw=False
                )
                module_scores.append(score_name)

        if staged_pipeline_enabled:
            kept_idx = np.array(
                [adata_s.var_names.get_loc(g) for g in kept_genes], dtype=int
            )
            X_scope = mat_cont[:, kept_idx] if kept_idx.size else mat_cont[:, :0]
            scope_level = str(
                cfg.get("scope_level_by_stratum", {}).get(
                    stratum_name,
                    cfg.get("scope_level_default", "global"),
                )
            )
            scope_ctx = {
                "scope_id": stratum_name.replace(" ", "_"),
                "scope_name": stratum_name,
                "scope_level": scope_level,
                "out_dir": stratum_dir,
                "figure_dir": fig_dir,
                "angles": angles_ref,
                "umap_xy": embeddings[ref_key],
                "donor_ids": donor_ids,
                "logger": logger,
            }
            staged_params: dict[str, Any] = {
                "discovery_mode": discovery_mode,
                "pipeline_mode": pipeline_mode,
                "plot_top_k": plot_top_k,
                "bins_screen": bins_screen,
                "bins_confirm": bins_confirm,
                "perm_init": perm_init,
                "perm_mid": perm_mid,
                "perm_final": perm_final,
                "p_escalate_1": p_escalate_1,
                "p_escalate_2": p_escalate_2,
                "stage1_top_k_global": stage1_top_k_global,
                "stage1_top_k_mega": stage1_top_k_mega,
                "stage1_top_k_cluster": stage1_top_k_cluster,
                "stage2_top_k_global": stage2_top_k_global,
                "stage2_top_k_mega": stage2_top_k_mega,
                "stage2_top_k_cluster": stage2_top_k_cluster,
                "min_prev_global": min_prev_global,
                "min_prev_cluster": min_prev_cluster,
                "min_fg_global": min_fg_global,
                "min_fg_cluster": min_fg_cluster,
                "skip_umap_plots": skip_umap_plots,
                "skip_pair_plots": skip_pair_plots,
                "skip_rsp_plots": skip_rsp_plots,
                "q_threshold": float(cfg.get("fdr_alpha", 0.05)),
                "seed": seed,
            }
            scope_results = run_scope_staged(
                scope_ctx=scope_ctx,
                X=X_scope,
                genes=kept_genes,
                labels=None,
                qc_covariates=adata_s.obs,
                cache_dir=results_dir
                / "cache"
                / "genomewide"
                / stratum_name.replace(" ", "_"),
                params=staged_params,
            )

            stage3_df = scope_results.get("stage3", pd.DataFrame()).copy()
            if not stage3_df.empty:
                stage3_df["stratum"] = stratum_name
                stage3_df["mode"] = "binary"
                stage3_df["threshold"] = "t0"
                stage3_df["embedding"] = ref_key
                stage3_df["p_perm_stage1"] = float("nan")
                stage3_df["p_perm_stage2"] = stage3_df["p_T"]
                stage3_df["FDR_stage2"] = stage3_df["q_T"]
                stage3_df["moran_I"] = float("nan")
                stage3_df["phi_sd_deg"] = float("nan")
                stage3_df["jackknife_Emax_sd"] = float("nan")
                stage3_df["n_cells"] = int(n_cells)
                stage3_df["n_donors"] = int(n_donors)
                stage3_df["detect_frac"] = stage3_df["prevalence"]
                stage3_df["ambient_prone"] = stage3_df["gene"].isin(ambient_flags)
                stage3_df["robust_hit"] = stage3_df["q_T"] <= float(
                    cfg.get("fdr_alpha", 0.05)
                )
                results_rows.append(stage3_df)

                top_hits = stage3_df.sort_values(
                    ["FDR_stage2", "E_max"], ascending=[True, False]
                ).head(plot_top_k)
                for _, row in top_hits.iterrows():
                    top_hits_rows.append(row.to_dict())

                summary_rows.append(
                    {
                        "stratum": stratum_name,
                        "genes_tested": int(len(kept_genes)),
                        "candidates": (
                            int(
                                scope_results["stage1"]
                                .get("stage1_selected", pd.Series([], dtype=bool))
                                .sum()
                            )
                            if "stage1" in scope_results
                            else 0
                        ),
                        "robust_hits": int(
                            np.sum(
                                stage3_df["FDR_stage2"]
                                <= float(cfg.get("fdr_alpha", 0.05))
                            )
                        ),
                        "median_phi_sd": float("nan"),
                        "ambient_frac_in_hits": float(
                            np.mean(top_hits["ambient_prone"].values)
                            if not top_hits.empty
                            else float("nan")
                        ),
                    }
                )
            else:
                summary_rows.append(
                    {
                        "stratum": stratum_name,
                        "genes_tested": int(len(kept_genes)),
                        "candidates": (
                            int(
                                scope_results["stage1"]
                                .get("stage1_selected", pd.Series([], dtype=bool))
                                .sum()
                            )
                            if "stage1" in scope_results
                            else 0
                        ),
                        "robust_hits": 0,
                        "median_phi_sd": float("nan"),
                        "ambient_frac_in_hits": float("nan"),
                    }
                )
            continue

        # Stage 1 cache
        stage1_path = stratum_dir / "stage1.csv"
        if resume and stage1_path.exists():
            stage1_df = pd.read_csv(stage1_path)
        else:
            stage1_rows = []
            for gene in kept_genes:
                expr_cont = _get_matrix(adata_s, expr_layer_continuous)[
                    :, adata_s.var_names.get_loc(gene)
                ]
                if sp.issparse(expr_cont):
                    expr_cont = expr_cont.toarray().ravel()
                else:
                    expr_cont = np.asarray(expr_cont).ravel()
                try:
                    _, phi_max, emax = compute_rsp_profile_continuous(
                        expr_cont, angles_ref, n_bins=bins
                    )
                except ValueError:
                    continue

                p_perm = float("nan")
                if inferential:
                    null_emax = np.zeros(stage1_perms, dtype=float)
                    for i in range(stage1_perms):
                        perm_vals = permute_values_within_donor(
                            expr_cont, donor_to_idx, rng
                        )
                        try:
                            _, _, emax_p = compute_rsp_profile_continuous(
                                perm_vals, angles_ref, n_bins=bins
                            )
                        except ValueError:
                            emax_p = float("nan")
                        null_emax[i] = emax_p
                    null_emax = null_emax[np.isfinite(null_emax)]
                    if null_emax.size > 0:
                        p_perm = (1.0 + np.sum(null_emax >= emax)) / (
                            1.0 + null_emax.size
                        )

                stage1_rows.append(
                    {
                        "stratum": stratum_name,
                        "gene": gene,
                        "mode": "continuous",
                        "E_max": float(emax),
                        "phi_max_deg": float(phi_max * 180.0 / math.pi),
                        "p_perm_stage1": float(p_perm),
                        "detect_frac": float(detect_frac[var_names.index(gene)]),
                        "ambient_prone": gene in ambient_flags,
                    }
                )
            stage1_df = pd.DataFrame(stage1_rows)
            stage1_df.to_csv(stage1_path, index=False)

        # Candidate selection
        candidates = set()
        if stage1_p_cutoff is not None:
            candidates |= set(
                stage1_df.loc[stage1_df["p_perm_stage1"] <= stage1_p_cutoff, "gene"]
            )
        if stage1_top_k is not None:
            stage1_sorted = stage1_df.sort_values("p_perm_stage1", ascending=True)
            candidates |= set(stage1_sorted.head(int(stage1_top_k))["gene"].values)
        candidates = list(candidates)
        if not inferential:
            candidates = []

        # Stage 2 for candidates
        stage2_path = stratum_dir / "stage2.csv"
        if resume and stage2_path.exists():
            stage2_df = pd.read_csv(stage2_path)
        else:
            stage2_rows = []
            for gene in candidates:
                expr_cont = _get_matrix(adata_s, expr_layer_continuous)[
                    :, adata_s.var_names.get_loc(gene)
                ]
                if sp.issparse(expr_cont):
                    expr_cont = expr_cont.toarray().ravel()
                else:
                    expr_cont = np.asarray(expr_cont).ravel()

                # p_perm_stage2 on reference embedding
                null_emax = np.zeros(stage2_perms, dtype=float)
                try:
                    _, phi_ref, emax_ref = compute_rsp_profile_continuous(
                        expr_cont, angles_ref, n_bins=bins
                    )
                except ValueError:
                    continue
                for i in range(stage2_perms):
                    perm_vals = permute_values_within_donor(
                        expr_cont, donor_to_idx, rng
                    )
                    try:
                        _, _, emax_p = compute_rsp_profile_continuous(
                            perm_vals, angles_ref, n_bins=bins
                        )
                    except ValueError:
                        emax_p = float("nan")
                    null_emax[i] = emax_p
                null_emax = null_emax[np.isfinite(null_emax)]
                p_perm2 = (1.0 + np.sum(null_emax >= emax_ref)) / (1.0 + null_emax.size)

                # Moran's I on lognorm
                moran_val = _safe_moran(
                    expr=expr_cont,
                    W=W,
                    logger=logger,
                    stratum_name=stratum_name,
                    feature=gene,
                )

                # phi stability across embeddings
                phi_by_embed = {}
                for emb_key, ang in angles_by_embed.items():
                    try:
                        _, phi_k, _ = compute_rsp_profile_continuous(
                            expr_cont, ang, n_bins=bins
                        )
                        phi_by_embed[emb_key] = phi_k
                    except ValueError:
                        phi_by_embed[emb_key] = float("nan")
                phi_vals = np.array(
                    [v for v in phi_by_embed.values() if np.isfinite(v)], dtype=float
                )
                phi_sd_deg = float(circular_sd(phi_vals) * 180.0 / math.pi)

                # jackknife
                jk = []
                for donor in unique_donors:
                    mask = donor_ids != donor
                    emb = embeddings[ref_key][mask]
                    ang = compute_angles(emb, compute_vantage(emb))
                    expr_sub = expr_cont[mask]
                    try:
                        _, _, emax_sub = compute_rsp_profile_continuous(
                            expr_sub, ang, n_bins=bins
                        )
                        jk.append(emax_sub)
                    except ValueError:
                        continue
                jk_sd = float(np.std(jk, ddof=1)) if len(jk) > 1 else float("nan")

                for emb_key, ang in angles_by_embed.items():
                    try:
                        _, phi_k, emax_k = compute_rsp_profile_continuous(
                            expr_cont, ang, n_bins=bins
                        )
                    except ValueError:
                        phi_k, emax_k = float("nan"), float("nan")
                    stage2_rows.append(
                        {
                            "stratum": stratum_name,
                            "gene": gene,
                            "mode": "continuous",
                            "threshold": "continuous",
                            "embedding": emb_key,
                            "E_max": float(emax_k),
                            "phi_max_deg": (
                                float(phi_k * 180.0 / math.pi)
                                if np.isfinite(phi_k)
                                else float("nan")
                            ),
                            "p_perm_stage1": (
                                float(
                                    stage1_df.loc[
                                        stage1_df["gene"] == gene, "p_perm_stage1"
                                    ].values[0]
                                )
                                if gene in stage1_df["gene"].values
                                else float("nan")
                            ),
                            "p_perm_stage2": (
                                float(p_perm2) if emb_key == ref_key else float("nan")
                            ),
                            "FDR_stage2": float("nan"),
                            "moran_I": float(moran_val),
                            "phi_sd_deg": phi_sd_deg,
                            "jackknife_Emax_sd": jk_sd,
                            "n_cells": int(n_cells),
                            "n_donors": int(n_donors),
                            "detect_frac": float(detect_frac[var_names.index(gene)]),
                            "ambient_prone": gene in ambient_flags,
                        }
                    )

                # binary validation if requested
                if feature_mode in {"binary", "both"}:
                    expr_bin = _get_matrix(adata_s, expr_layer_binary)[
                        :, adata_s.var_names.get_loc(gene)
                    ]
                    if sp.issparse(expr_bin):
                        expr_bin = expr_bin.toarray().ravel()
                    else:
                        expr_bin = np.asarray(expr_bin).ravel()
                    for thr_name in thresholds_to_validate:
                        if thr_name == "t0":
                            f_mask = expr_bin > 0
                        else:
                            q = 0.9 if thr_name == "q90" else 0.95
                            f_mask = expr_bin > np.quantile(expr_bin, q)
                        try:
                            _, phi_k, emax_k = compute_rsp_profile_from_boolean(
                                f_mask, angles_ref, n_bins=bins
                            )
                        except ValueError:
                            phi_k, emax_k = float("nan"), float("nan")
                        stage2_rows.append(
                            {
                                "stratum": stratum_name,
                                "gene": gene,
                                "mode": "binary",
                                "threshold": thr_name,
                                "embedding": ref_key,
                                "E_max": float(emax_k),
                                "phi_max_deg": (
                                    float(phi_k * 180.0 / math.pi)
                                    if np.isfinite(phi_k)
                                    else float("nan")
                                ),
                                "p_perm_stage1": float("nan"),
                                "p_perm_stage2": float("nan"),
                                "FDR_stage2": float("nan"),
                                "moran_I": float(moran_val),
                                "phi_sd_deg": phi_sd_deg,
                                "jackknife_Emax_sd": jk_sd,
                                "n_cells": int(n_cells),
                                "n_donors": int(n_donors),
                                "detect_frac": float(
                                    detect_frac[var_names.index(gene)]
                                ),
                                "ambient_prone": gene in ambient_flags,
                            }
                        )

            stage2_df = pd.DataFrame(stage2_rows)
            stage2_df.to_csv(stage2_path, index=False)

        # FDR within stratum for continuous ref embedding
        mask_ref = (stage2_df["mode"] == "continuous") & (
            stage2_df["embedding"] == ref_key
        )
        pvals = stage2_df.loc[mask_ref, "p_perm_stage2"].values
        fdrs = bh_fdr(np.asarray(pvals, dtype=float))
        stage2_df.loc[mask_ref, "FDR_stage2"] = fdrs
        # propagate FDR to all rows for same gene
        fdr_map = dict(
            zip(stage2_df.loc[mask_ref, "gene"], stage2_df.loc[mask_ref, "FDR_stage2"])
        )
        stage2_df["FDR_stage2"] = stage2_df["gene"].map(fdr_map)

        # Moran baseline from random genes
        rand_genes = rng.choice(
            list(kept_genes), size=min(200, len(kept_genes)), replace=False
        )
        moran_random = []
        for g in rand_genes:
            expr = _get_matrix(adata_s, expr_layer_continuous)[
                :, adata_s.var_names.get_loc(g)
            ]
            if sp.issparse(expr):
                expr = expr.toarray().ravel()
            else:
                expr = np.asarray(expr).ravel()
            moran_val = _safe_moran(
                expr=expr,
                W=W,
                logger=logger,
                stratum_name=stratum_name,
                feature=str(g),
            )
            if np.isfinite(moran_val):
                moran_random.append(moran_val)
        moran_random = np.asarray(moran_random, dtype=float)

        # robust hit flags
        robust = []
        for gene, group in stage2_df[mask_ref].groupby("gene"):
            row = group.iloc[0]
            phi_sd_deg = float(row["phi_sd_deg"])
            moran_val = float(row["moran_I"])
            moran_pct = (
                float(np.mean(moran_random <= moran_val))
                if moran_random.size
                else float("nan")
            )
            fdr = (
                float(row["FDR_stage2"])
                if np.isfinite(row["FDR_stage2"])
                else float("nan")
            )
            jack_sd = float(row["jackknife_Emax_sd"])
            robust_hit = (
                np.isfinite(fdr)
                and fdr <= 0.05
                and phi_sd_deg <= float(cfg.get("embedding_phi_sd_deg", 30.0))
                and moran_pct >= 0.95
                and jack_sd <= float(cfg.get("donor_jackknife_sd", 0.1))
            )
            robust.append(
                {
                    "gene": gene,
                    "robust_hit": bool(robust_hit),
                    "moran_I_percentile": moran_pct,
                }
            )
        robust_df = pd.DataFrame(robust)
        stage2_df = stage2_df.merge(robust_df, on="gene", how="left")

        # store results
        results_rows.append(stage2_df)

        # top hits
        top_hits = (
            stage2_df[mask_ref]
            .sort_values(["FDR_stage2", "E_max"], ascending=[True, False])
            .head(topn_plots)
        )
        for _, row in top_hits.iterrows():
            top_hits_rows.append(row.to_dict())

        # plots per stratum
        if not top_hits.empty:
            # volcano
            x = top_hits["E_max"].values
            fdr_vals = top_hits["FDR_stage2"].values
            y = -np.log10(np.clip(fdr_vals, 1e-12, 1.0))
            _plot_scatter(
                x,
                y,
                top_hits["gene"].tolist(),
                fig_dir / "volcano.png",
                f"{stratum_name}: E_max vs FDR",
                "E_max",
                "-log10(FDR)",
                set(
                    top_hits.loc[top_hits["robust_hit"].fillna(False), "gene"].tolist()
                ),
            )

            # Moran vs E_max
            _plot_scatter(
                top_hits["E_max"].values,
                top_hits["moran_I"].values,
                top_hits["gene"].tolist(),
                fig_dir / "moran_vs_emax.png",
                f"{stratum_name}: Moran vs E_max",
                "E_max",
                "Moran's I",
                set(
                    top_hits.loc[top_hits["robust_hit"].fillna(False), "gene"].tolist()
                ),
            )

            # phi compass
            _plot_phi_compass(
                top_hits["phi_max_deg"].values,
                fig_dir / "phi_compass.png",
                f"{stratum_name}: phi_max distribution",
            )

            # phi stability bars
            _plot_phi_stability_bar(
                top_hits["phi_sd_deg"].values,
                top_hits["gene"].tolist(),
                fig_dir / "phi_stability.png",
                f"{stratum_name}: phi_sd_deg",
            )

            # per-hit plots
            for _, row in top_hits.iterrows():
                gene = row["gene"]
                expr = _get_matrix(adata_s, expr_layer_continuous)[
                    :, adata_s.var_names.get_loc(gene)
                ]
                if sp.issparse(expr):
                    expr = expr.toarray().ravel()
                else:
                    expr = np.asarray(expr).ravel()

                # recompute E_phi for polar
                E_phi, phi_max, emax = compute_rsp_profile_continuous(
                    expr, angles_ref, n_bins=bins
                )
                is_ambient = bool(row.get("ambient_prone", False))
                subtitle = f"FDR={row['FDR_stage2']:.3g} | Moran={row['moran_I']:.3g}"
                _plot_umap_with_arrow(
                    embeddings[ref_key],
                    expr,
                    phi_max,
                    fig_dir / f"{gene}_umap_arrow.png",
                    f"{stratum_name}: {gene}",
                    subtitle,
                    is_ambient,
                )
                plot_rsp_polar(
                    E_phi,
                    (fig_dir / f"{gene}_rsp_polar.png").as_posix(),
                    f"RSP: {gene}",
                )

                # null histogram (stage2)
                null_emax = np.zeros(stage2_perms, dtype=float)
                for i in range(stage2_perms):
                    perm_vals = permute_values_within_donor(expr, donor_to_idx, rng)
                    try:
                        _, _, emax_p = compute_rsp_profile_continuous(
                            perm_vals, angles_ref, n_bins=bins
                        )
                    except ValueError:
                        emax_p = float("nan")
                    null_emax[i] = emax_p
                null_emax = null_emax[np.isfinite(null_emax)]
                if null_emax.size > 0:
                    _plot_null_hist(
                        null_emax,
                        emax,
                        fig_dir / f"{gene}_null_hist.png",
                        f"Null E_max: {gene}",
                    )

        # null calibration QQ (reuse if prereg exists)
        prereg_qq = results_dir / "qq" / f"{stratum_name.replace(' ', '_')}_qq.csv"
        if prereg_qq.exists():
            qq_df = pd.read_csv(prereg_qq)
            _plot_qq(
                qq_df["observed"].values,
                fig_dir / "null_calibration_qq.png",
                "Null calibration",
            )

        # summary row
        robust_count = (
            int(top_hits["robust_hit"].sum()) if "robust_hit" in top_hits.columns else 0
        )
        summary_rows.append(
            {
                "stratum": stratum_name,
                "genes_tested": int(len(kept_genes)),
                "candidates": int(len(candidates)),
                "robust_hits": robust_count,
                "median_phi_sd": (
                    float(np.nanmedian(top_hits["phi_sd_deg"].values))
                    if not top_hits.empty
                    else float("nan")
                ),
                "ambient_frac_in_hits": float(
                    np.mean(top_hits["ambient_prone"].values)
                    if not top_hits.empty
                    else float("nan")
                ),
            }
        )

    _write_genomewide_outputs(
        results_dir=results_dir,
        results_rows=results_rows,
        top_hits_rows=top_hits_rows,
        gene_filtering_rows=gene_filtering_rows,
        summary_rows=summary_rows,
    )

    logger.info("Genome-wide pipeline complete. Results in %s", results_dir.as_posix())
