"""Preregistered BioRSP evaluation pipeline (donor-aware, within-strata)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from biorsp.config import load_json_config
from biorsp.geometry import compute_angles, compute_vantage
from biorsp.moran import extract_weights, morans_i
from biorsp.permutation import perm_null_T_and_profile, permute_foreground_within_donor
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
from biorsp.rsp import (
    compute_rsp_profile,
    compute_rsp_profile_from_boolean,
    plot_rsp_polar,
)
from biorsp.utils import ensure_dir

DEFAULT_STRATA = {
    "Ventricular cardiomyocytes": [
        "Ventricular cardiomyocyte",
        "Ventricular cardiomyocytes",
        "Ventricular CM",
    ],
    "Fibroblasts": ["Fibroblast", "Fibroblasts"],
    "Vascular mural": ["Pericyte", "Smooth muscle", "Smooth muscle cell"],
    "Capillary EC": ["Capillary EC", "Capillary endothelial", "Capillary"],
    "Arterial EC": ["Arterial EC", "Arterial endothelial", "Arterial"],
    "Venous EC": ["Venous EC", "Venous endothelial", "Venous"],
    "Endocardial": ["Endocardial", "Endocardial EC"],
    "Lymphatic EC": ["Lymphatic EC", "Lymphatic endothelial"],
    "Myeloid": ["Macrophage", "Monocyte/cDC", "Monocyte", "cDC"],
    "Lymphoid": ["T", "NK", "B", "Mast"],
}

SECONDARY_STRATA = {
    "Mesothelial": ["Mesothelial"],
    "Adipocyte": ["Adipocyte", "Adipocytes"],
    "Neuronal": ["Neuronal", "Neuron", "Neurons"],
    "Atrial CM": ["Atrial CM", "Atrial cardiomyocyte", "Atrial cardiomyocytes"],
}

POSITIVE_PANELS = {
    "Ventricular cardiomyocytes": ["TTN", "TNNT2", "MYBPC3", "MYH7", "MYL2", "MYL3"],
    "Fibroblasts": ["COL1A1", "COL1A2", "COL3A1", "DCN", "POSTN", "LUM"],
    "Capillary EC": ["CA4", "RGCC"],
    "Pericyte": ["RGS5", "PDGFRB", "CSPG4", "ABCC9", "KCNJ8"],
}

HOUSEKEEPING = ["ACTB", "GAPDH", "RPLP0"]

MODULE_SETS = {
    "fibrosis_ecm": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "POSTN"],
    "inflammation": ["IL1B", "TNF", "CCL2", "CCL3", "CCL5", "CXCL8"],
    "hypoxia_stress": ["HIF1A", "VEGFA", "DDIT3", "ATF4", "SLC2A1"],
    "cell_cycle": ["MKI67", "TOP2A", "PCNA", "MCM5", "MCM6"],
    "oxphos": ["MT-CO1", "MT-CO2", "NDUFA1", "NDUFB8", "ATP5F1A"],
}

AMBIENT_PATTERNS = ["MT-", "RPL", "RPS", "MALAT1", "FOS", "JUN"]


def _perm_null_emax_from_canonical(
    expr: np.ndarray,
    angles: np.ndarray,
    donor_ids: np.ndarray,
    n_bins: int,
    n_perm: int,
    seed: int,
) -> tuple[np.ndarray, float, float, float]:
    f_obs = np.asarray(expr, dtype=float).ravel() > 0
    out = perm_null_T_and_profile(
        f=f_obs,
        angles=np.asarray(angles, dtype=float).ravel(),
        donor_ids=np.asarray(donor_ids),
        n_bins=int(n_bins),
        n_perm=int(n_perm),
        seed=int(seed),
        donor_stratified=True,
        return_null_profiles=True,
        mode="raw",
        smooth_w=1,
    )

    e_obs = np.asarray(
        out.get("E_phi_obs", np.zeros(int(n_bins), dtype=float)), dtype=float
    )
    if e_obs.size == 0:
        return np.zeros(int(n_perm), dtype=float), 0.0, 0.0, 1.0

    b_max = int(np.argmax(e_obs))
    edges = np.linspace(0.0, 2.0 * np.pi, int(n_bins) + 1, endpoint=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    phi_max_obs = float(centers[b_max] % (2.0 * np.pi))
    emax_obs = float(np.max(e_obs))

    null_e = np.asarray(
        out.get("null_E_phi", np.zeros((0, int(n_bins)), dtype=float)), dtype=float
    )
    if null_e.ndim != 2 or null_e.shape[0] == 0:
        return np.zeros(0, dtype=float), emax_obs, phi_max_obs, 1.0
    null_emax = np.max(null_e, axis=1)
    p = float((1.0 + np.sum(null_emax >= emax_obs)) / (1.0 + null_emax.size))
    return null_emax, emax_obs, phi_max_obs, p


def _get_scanpy():
    import scanpy as sc

    return sc


@dataclass
class QCThresholds:
    min_genes: int
    min_counts: int
    max_mito_frac: float


def load_config(path: str) -> dict[str, Any]:
    """Load prereg pipeline config from strict JSON."""
    return load_json_config(path)


def _get_expr_vector(adata, gene: str, layer: str | None) -> np.ndarray:
    if gene not in adata.var_names:
        raise KeyError(f"Gene '{gene}' not found in adata.var_names.")
    if layer:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        mat = adata[:, gene].layers[layer]
    else:
        mat = adata[:, gene].X
    if sp.issparse(mat):
        vec = mat.toarray().ravel()
    else:
        vec = np.asarray(mat).ravel()
    if vec.size != adata.n_obs:
        raise ValueError("Expression vector length mismatch.")
    return vec


def _apply_qc(
    adata,
    thresholds: QCThresholds,
    logger: logging.Logger,
    doublet_col: str | None,
    doublet_score_max: float | None,
    auto_filter_doublets: bool,
) -> tuple[object, dict[str, Any]]:
    obs = adata.obs
    report = {
        "n_cells_before": int(adata.n_obs),
        "n_cells_after": int(adata.n_obs),
    }
    required_cols = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    missing = [c for c in required_cols if c not in obs.columns]
    if missing:
        logger.warning(
            "QC metrics missing (%s). Skipping QC filtering.", ", ".join(missing)
        )
        return adata, report

    mask = (
        (obs["n_genes_by_counts"] >= thresholds.min_genes)
        & (obs["total_counts"] >= thresholds.min_counts)
        & (obs["pct_counts_mt"] <= thresholds.max_mito_frac)
    )
    # Optional doublet filtering
    auto_cols = ["doublet", "predicted_doublet", "is_doublet"]
    col = doublet_col
    if col is None and auto_filter_doublets:
        for c in auto_cols:
            if c in obs.columns:
                col = c
                break
    if col is not None:
        if col not in obs.columns:
            raise KeyError(f"Doublet column '{col}' not found in adata.obs.")
        if obs[col].dtype == bool:
            mask &= ~obs[col]
        else:
            if doublet_score_max is None:
                logger.warning(
                    "Doublet column '%s' is numeric but no doublet_score_max set; skipping.",
                    col,
                )
            else:
                mask &= obs[col] <= doublet_score_max
    adata_f = adata[mask].copy()
    report["n_cells_after"] = int(adata_f.n_obs)
    return adata_f, report


def _preprocessing_report(
    adata_before,
    adata_after,
    donor_col: str,
    qc_mode: str,
) -> pd.DataFrame:
    rows = []
    donors = sorted(set(adata_before.obs[donor_col].astype(str)))
    for donor in donors:
        before_mask = adata_before.obs[donor_col].astype(str) == donor
        after_mask = adata_after.obs[donor_col].astype(str) == donor
        row = {
            "qc_mode": qc_mode,
            "donor": donor,
            "n_cells_before": int(before_mask.sum()),
            "n_cells_after": int(after_mask.sum()),
        }
        for col in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]:
            if col in adata_before.obs.columns:
                row[f"{col}_median_before"] = float(
                    np.median(adata_before.obs.loc[before_mask, col].values)
                )
                row[f"{col}_median_after"] = float(
                    np.median(adata_after.obs.loc[after_mask, col].values)
                )
            else:
                row[f"{col}_median_before"] = float("nan")
                row[f"{col}_median_after"] = float("nan")
        if "n_genes_by_counts" in adata_before.obs.columns:
            n_vars = max(1, adata_before.n_vars)
            before_rate = (
                adata_before.obs.loc[before_mask, "n_genes_by_counts"] / n_vars
            )
            after_rate = adata_after.obs.loc[after_mask, "n_genes_by_counts"] / n_vars
            row["detection_rate_median_before"] = (
                float(np.median(before_rate.values))
                if before_rate.size
                else float("nan")
            )
            row["detection_rate_median_after"] = (
                float(np.median(after_rate.values)) if after_rate.size else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _ambient_flag_genes(var_names: Iterable[str]) -> list[str]:
    flags = []
    for g in var_names:
        g_upper = str(g).upper()
        if any(g_upper.startswith(p) for p in AMBIENT_PATTERNS if p.endswith("-")):
            flags.append(str(g))
            continue
        if any(g_upper == p for p in AMBIENT_PATTERNS if not p.endswith("-")):
            flags.append(str(g))
    return sorted(set(flags))


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


def _plot_umap_overlay(
    emb: np.ndarray, values: np.ndarray, out_png: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    sca = ax.scatter(emb[:, 0], emb[:, 1], c=values, s=5, cmap="viridis", linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png.as_posix(), dpi=150)
    plt.close(fig)


def _plot_phi_stability(
    phi_by_embed: dict[str, float], out_png: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    labels = list(phi_by_embed.keys())
    angles = np.array([phi_by_embed[k] for k in labels], dtype=float)
    radii = np.ones_like(angles)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="polar")
    ax.scatter(angles, radii, c="tab:blue")
    for label, angle in zip(labels, angles):
        ax.text(angle, 1.05, label, ha="center", va="center", fontsize=8)
    ax.set_title(title)
    ax.set_rticks([])
    fig.tight_layout()
    fig.savefig(out_png.as_posix(), dpi=150)
    plt.close(fig)


def _detect_multimodal(E_phi: np.ndarray, frac: float = 0.5) -> tuple[bool, list[int]]:
    if E_phi.size < 3:
        return False, []
    thresh = frac * float(np.max(E_phi))
    peaks = []
    for i in range(1, E_phi.size - 1):
        if E_phi[i] > E_phi[i - 1] and E_phi[i] > E_phi[i + 1] and E_phi[i] >= thresh:
            peaks.append(i)
    return len(peaks) > 1, peaks


def _spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _safe_moran(
    *,
    adata,
    feature: str,
    layer: str | None,
    W,
    logger: logging.Logger,
    qc_mode: str,
    stratum_name: str,
) -> float:
    """Compute Moran's I with explicit warning for expected data issues."""
    try:
        expr = _get_expr_vector(adata, feature, layer)
        return float(morans_i(expr, W, row_standardize=False))
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "Moran skipped: qc_mode=%s stratum=%s feature=%s reason=%s",
            qc_mode,
            stratum_name,
            feature,
            exc,
        )
        return float("nan")


def _prepare_prereg_dirs(outdir: Path) -> tuple[Path, Path, Path]:
    results_dir = outdir / "results"
    figures_dir = outdir / "figures"
    logs_dir = outdir / "logs"
    ensure_dir(results_dir.as_posix())
    ensure_dir(figures_dir.as_posix())
    ensure_dir(logs_dir.as_posix())
    return results_dir, figures_dir, logs_dir


def _write_prereg_outputs(
    *,
    results_dir: Path,
    prep_rows: list[pd.DataFrame],
    all_results: list[dict[str, Any]],
    null_rows: list[dict[str, Any]],
    jackknife_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    donor_artifact_rows: list[dict[str, Any]],
    moran_baseline_rows: list[dict[str, Any]],
    de_baseline_rows: list[dict[str, Any]],
    embedding_stability_rows: list[dict[str, Any]],
    ablation_global_rows: list[dict[str, Any]],
    multimodal_rows: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]],
    qq_rows: list[dict[str, Any]],
) -> None:
    if prep_rows:
        prep_df = pd.concat(prep_rows, ignore_index=True)
        prep_df.to_csv(results_dir / "preprocessing_report.csv", index=False)
        with open(results_dir / "preprocessing_report.md", "w", encoding="utf-8") as fh:
            fh.write("# Preprocessing report\n\n")
            fh.write(prep_df.to_markdown(index=False))

    pd.DataFrame(all_results).to_csv(
        results_dir / "biorsp_stratum_results.csv", index=False
    )
    pd.DataFrame(null_rows).to_csv(results_dir / "null_calibration.csv", index=False)
    pd.DataFrame(jackknife_rows).to_csv(
        results_dir / "jackknife_details.csv", index=False
    )
    pd.DataFrame(threshold_rows).to_csv(
        results_dir / "threshold_sensitivity.csv", index=False
    )
    pd.DataFrame(donor_artifact_rows).to_csv(
        results_dir / "donor_artifact_test.csv", index=False
    )
    pd.DataFrame(moran_baseline_rows).to_csv(
        results_dir / "moran_baseline.csv", index=False
    )
    pd.DataFrame(de_baseline_rows).to_csv(results_dir / "de_baseline.csv", index=False)
    pd.DataFrame(embedding_stability_rows).to_csv(
        results_dir / "embedding_stability.csv", index=False
    )
    pd.DataFrame(ablation_global_rows).to_csv(
        results_dir / "ablation_global_shuffle.csv", index=False
    )
    pd.DataFrame(multimodal_rows).to_csv(
        results_dir / "multimodal_peaks.csv", index=False
    )
    if batch_rows:
        pd.DataFrame(batch_rows).to_csv(results_dir / "batch_summary.csv", index=False)
    if qq_rows:
        pd.DataFrame(qq_rows).to_csv(
            results_dir / "null_calibration_qq_index.csv", index=False
        )


def run_prereg_pipeline(config_path: str) -> None:
    cfg = load_config(config_path)
    sc = _get_scanpy()

    outdir = Path(cfg.get("outdir", "."))
    results_dir, figures_dir, logs_dir = _prepare_prereg_dirs(outdir)

    logger = _setup_logger(logs_dir / "biorsp_prereg.log", "biorsp_prereg")

    h5ad_path = cfg.get("h5ad_path", "adata_embed_graph.h5ad")
    if not Path(h5ad_path).exists():
        raise FileNotFoundError(
            f"Input file not found: {h5ad_path}. Update configs/biorsp_prereg.json."
        )

    donor_col = cfg.get("donor_col")
    celltype_col = cfg.get("celltype_col")
    batch_col = cfg.get("batch_col")
    doublet_col = cfg.get("doublet_col")
    doublet_score_max = cfg.get("doublet_score_max")
    auto_filter_doublets = bool(cfg.get("auto_filter_doublets", True))
    seeds = [int(s) for s in cfg.get("umap_seeds", [0, 1, 2, 3, 4])]
    bins = int(cfg.get("bins", 72))
    n_perm = int(cfg.get("n_perm", 1000))
    n_perm_null = int(cfg.get("n_perm_null", 200))
    random_genes = int(cfg.get("random_genes", 50))
    min_donors = int(cfg.get("min_donors", 3))
    min_cells_per_donor = int(cfg.get("min_cells_per_donor", 200))
    n_pcs = int(cfg.get("pca_n_comps", 50))
    n_neighbors = int(cfg.get("neighbors_k", 15))
    umap_min_dist = float(cfg.get("umap_min_dist", 0.3))
    umap_spread = float(cfg.get("umap_spread", 1.0))
    include_secondary = bool(cfg.get("include_secondary", False))
    split_mural = bool(cfg.get("split_mural", False))
    include_modules = bool(cfg.get("include_modules", False))
    seed = int(cfg.get("seed", 0))
    expr_layer_rsp = cfg.get("rsp_expr_layer", None)
    lognorm_layer = cfg.get("moran_expr_layer", "lognorm")

    qc_clean = QCThresholds(
        **cfg.get(
            "qc_clean", {"min_genes": 200, "min_counts": 500, "max_mito_frac": 20.0}
        )
    )
    qc_stress = QCThresholds(
        **cfg.get(
            "qc_stress", {"min_genes": 100, "min_counts": 200, "max_mito_frac": 30.0}
        )
    )
    qc_modes = (
        ["cleaned", "stress"] if cfg.get("run_stress_mode", True) else ["cleaned"]
    )

    strata = cfg.get("strata", DEFAULT_STRATA)
    if include_secondary:
        strata = {**strata, **SECONDARY_STRATA}
    if split_mural and "Vascular mural" in strata:
        strata.pop("Vascular mural", None)
        strata["Pericyte"] = ["Pericyte"]
        strata["Smooth muscle"] = ["Smooth muscle", "Smooth muscle cell"]

    all_results = []
    null_rows = []
    qq_rows = []
    jackknife_rows = []
    threshold_rows = []
    donor_artifact_rows = []
    moran_baseline_rows = []
    de_baseline_rows = []
    embedding_stability_rows = []
    ablation_global_rows = []
    multimodal_rows = []
    prep_rows = []

    batch_rows: list[dict[str, Any]] = []

    for qc_mode in qc_modes:
        logger.info("QC mode: %s", qc_mode)
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

        qc = qc_clean if qc_mode == "cleaned" else qc_stress
        adata_qc, qc_report = _apply_qc(
            adata, qc, logger, doublet_col, doublet_score_max, auto_filter_doublets
        )
        prep_rows.append(_preprocessing_report(adata, adata_qc, donor_col, qc_mode))

        # Ambient flag list
        ambient_flags = _ambient_flag_genes(adata_qc.var_names)
        pd.DataFrame({"ambient_flag_gene": ambient_flags}).to_csv(
            results_dir / "ambient_flag_genes.csv", index=False
        )

        # log-normalize for Moran and modules
        adata_qc = adata_qc.copy()
        sc.pp.normalize_total(adata_qc, target_sum=1e4)
        sc.pp.log1p(adata_qc)
        adata_qc.layers["lognorm"] = adata_qc.X.copy()

        rng = np.random.default_rng(seed)

        for stratum_name, labels in strata.items():
            adata_s, n_cells, n_donors, inferential = _subset_stratum(
                adata_qc,
                celltype_col,
                donor_col,
                labels,
                min_donors,
                min_cells_per_donor,
            )
            if adata_s is None:
                logger.info("Stratum %s: no cells", stratum_name)
                continue

            if batch_col and batch_col in adata_s.obs.columns:
                batch_counts = (
                    adata_s.obs.groupby([batch_col, donor_col], observed=True)
                    .size()
                    .reset_index(name="n_cells")
                )
                for _, row in batch_counts.iterrows():
                    batch_rows.append(
                        {
                            "qc_mode": qc_mode,
                            "stratum": stratum_name,
                            "batch": str(row[batch_col]),
                            "donor": str(row[donor_col]),
                            "n_cells": int(row["n_cells"]),
                        }
                    )

            embeddings = _compute_embeddings(
                adata_s,
                sc,
                seeds,
                n_pcs,
                n_neighbors,
                umap_min_dist,
                umap_spread,
            )
            angles_by_embed = {}
            for key, emb in embeddings.items():
                angles_by_embed[key] = compute_angles(emb, compute_vantage(emb))

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

            # feature lists
            panel_genes = POSITIVE_PANELS.get(stratum_name, [])
            if stratum_name == "Vascular mural":
                panel_genes = POSITIVE_PANELS.get("Pericyte", [])
            features: list[tuple[str, str]] = []
            for g in panel_genes:
                if g in adata_s.var_names:
                    features.append(("gene", g))
            for g in HOUSEKEEPING:
                if g in adata_s.var_names:
                    features.append(("gene", g))

            # random genes for null calibration and Moran baseline
            expressed = []
            for g in adata_s.var_names:
                vec = _get_expr_vector(adata_s, g, expr_layer_rsp)
                if np.any(vec > 0):
                    expressed.append(g)
            expressed = np.array(expressed)
            rng.shuffle(expressed)
            rand_genes = expressed[: min(random_genes, expressed.size)]

            # module scores
            if include_modules:
                for module_name, gene_list in MODULE_SETS.items():
                    present = [g for g in gene_list if g in adata_s.var_names]
                    if len(present) < 2:
                        continue
                    score_name = f"score_{module_name}"
                    sc.tl.score_genes(
                        adata_s, gene_list=present, score_name=score_name, use_raw=False
                    )
                    features.append(("module", score_name))

            # ambient probes
            ambient_genes = [g for g in ambient_flags if g in adata_s.var_names]
            for g in ambient_genes:
                features.append(("ambient", g))

            # Precompute Moran random baseline
            moran_random = []
            for g in rand_genes:
                try:
                    x = _get_expr_vector(adata_s, g, lognorm_layer)
                    moran_random.append(
                        morans_i(x.astype(float), W, row_standardize=False)
                    )
                except ValueError:
                    continue
            moran_random = np.asarray(moran_random, dtype=float)

            # Embedding stability: E_max ranks per seed
            emax_by_seed = {k: {} for k in embeddings}

            seed0_key = f"umap_seed{seeds[0]}"
            thresholds = ["t0", "q90", "q95"]

            per_stratum_rows = []
            for ftype, feat in features:
                if ftype == "module":
                    expr = np.asarray(adata_s.obs[feat]).ravel()
                    display = feat
                else:
                    expr = _get_expr_vector(adata_s, feat, expr_layer_rsp)
                    display = feat

                # RSP on seed0, t0 (for p_perm)
                angles0 = angles_by_embed[seed0_key]
                try:
                    E_phi, phi_max0, E_max0 = compute_rsp_profile(
                        expr, angles0, n_bins=bins
                    )
                except ValueError:
                    continue

                # Donor-aware permutations (inferential only)
                p_perm = float("nan")
                null_emax = None
                if inferential:
                    null_emax, E_max_obs, phi_max_obs, p_perm = (
                        _perm_null_emax_from_canonical(
                            expr, angles0, donor_ids, bins, n_perm, seed=seed
                        )
                    )
                else:
                    E_max_obs = E_max0

                # Moran's I
                moran_val = _safe_moran(
                    adata=adata_s,
                    feature=feat,
                    layer=lognorm_layer,
                    W=W,
                    logger=logger,
                    qc_mode=qc_mode,
                    stratum_name=stratum_name,
                )

                # phi stability across embeddings
                phi_by_embed = {}
                for emb_key, ang in angles_by_embed.items():
                    try:
                        _, phi_k, emax_k = compute_rsp_profile(expr, ang, n_bins=bins)
                        phi_by_embed[emb_key] = phi_k
                        emax_by_seed[emb_key][display] = emax_k
                    except ValueError:
                        phi_by_embed[emb_key] = float("nan")
                phi_sd = circular_sd(
                    np.array([v for v in phi_by_embed.values() if np.isfinite(v)])
                )
                phi_sd_deg = float(phi_sd * 180.0 / math.pi)

                # donor jackknife (seed0 only)
                jk_emax = []
                for donor in unique_donors:
                    mask = donor_ids != donor
                    emb = embeddings[seed0_key][mask]
                    ang = compute_angles(emb, compute_vantage(emb))
                    expr_sub = expr[mask]
                    try:
                        _, _, emax_sub = compute_rsp_profile(expr_sub, ang, n_bins=bins)
                        jk_emax.append(emax_sub)
                        jackknife_rows.append(
                            {
                                "qc_mode": qc_mode,
                                "stratum": stratum_name,
                                "gene_or_signature": display,
                                "donor_left_out": str(donor),
                                "E_max": float(emax_sub),
                            }
                        )
                    except ValueError:
                        continue
                jackknife_sd = (
                    float(np.std(jk_emax, ddof=1)) if len(jk_emax) > 1 else float("nan")
                )

                # threshold sensitivity (seed0 only)
                threshold_emax = {}
                threshold_phi = {}
                for t_name in thresholds:
                    if t_name == "t0":
                        f_mask = expr > 0
                    else:
                        q = 0.9 if t_name == "q90" else 0.95
                        thr = np.quantile(expr, q)
                        f_mask = expr > thr
                    try:
                        _, phi_t, emax_t = compute_rsp_profile_from_boolean(
                            f_mask, angles0, n_bins=bins
                        )
                        threshold_emax[t_name] = emax_t
                        threshold_phi[t_name] = phi_t
                    except ValueError:
                        threshold_emax[t_name] = float("nan")
                        threshold_phi[t_name] = float("nan")
                    threshold_rows.append(
                        {
                            "qc_mode": qc_mode,
                            "stratum": stratum_name,
                            "gene_or_signature": display,
                            "threshold": t_name,
                            "E_max": threshold_emax[t_name],
                        }
                    )

                # multimodality detection on seed0 t0
                multimodal, peak_idx = _detect_multimodal(E_phi)
                if multimodal:
                    phi_centers = np.linspace(0, 2 * np.pi, bins, endpoint=False)
                    peaks = [
                        float(phi_centers[i] * 180.0 / math.pi) for i in peak_idx[:2]
                    ]
                    multimodal_rows.append(
                        {
                            "qc_mode": qc_mode,
                            "stratum": stratum_name,
                            "gene_or_signature": display,
                            "phi_peak1_deg": (
                                peaks[0] if len(peaks) > 0 else float("nan")
                            ),
                            "phi_peak2_deg": (
                                peaks[1] if len(peaks) > 1 else float("nan")
                            ),
                        }
                    )

                # flags
                threshold_phi_sd = circular_sd(
                    np.array([v for v in threshold_phi.values() if np.isfinite(v)])
                )
                threshold_sensitive = bool(
                    np.nanmax(list(threshold_emax.values()))
                    - np.nanmin(list(threshold_emax.values()))
                    > float(cfg.get("threshold_emax_delta", 0.1))
                    or (threshold_phi_sd * 180.0 / math.pi)
                    > float(cfg.get("threshold_phi_sd_deg", 30.0))
                )
                embedding_sensitive = bool(
                    phi_sd_deg > float(cfg.get("embedding_phi_sd_deg", 30.0))
                )
                donor_sensitive = bool(
                    jackknife_sd > float(cfg.get("donor_jackknife_sd", 0.1))
                )

                moran_pct = (
                    float(np.mean(moran_random <= moran_val))
                    if moran_random.size > 0
                    else float("nan")
                )

                # store rows per embedding + threshold
                for emb_key in embeddings.keys():
                    for t_name in thresholds:
                        row = {
                            "qc_mode": qc_mode,
                            "stratum": stratum_name,
                            "gene_or_signature": display,
                            "embedding": emb_key,
                            "threshold": t_name,
                            "E_max": float("nan"),
                            "phi_max_deg": float("nan"),
                            "p_perm": float("nan"),
                            "FDR": float("nan"),
                            "moran_I": float(moran_val),
                            "moran_I_percentile": moran_pct,
                            "phi_sd_deg": phi_sd_deg,
                            "jackknife_Emax_sd": jackknife_sd,
                            "n_cells": int(n_cells),
                            "n_donors": int(n_donors),
                            "threshold_sensitive": threshold_sensitive,
                            "embedding_sensitive": embedding_sensitive,
                            "donor_sensitive": donor_sensitive,
                            "multimodal": multimodal,
                        }
                        # fill E_max/phi_max for this embedding + threshold
                        if t_name == "t0":
                            try:
                                _, phi_k, emax_k = compute_rsp_profile(
                                    expr, angles_by_embed[emb_key], n_bins=bins
                                )
                                row["E_max"] = float(emax_k)
                                row["phi_max_deg"] = float(phi_k * 180.0 / math.pi)
                            except ValueError:
                                pass
                        else:
                            f_mask = expr > (
                                np.quantile(expr, 0.9)
                                if t_name == "q90"
                                else np.quantile(expr, 0.95)
                            )
                            try:
                                _, phi_k, emax_k = compute_rsp_profile_from_boolean(
                                    f_mask, angles_by_embed[emb_key], n_bins=bins
                                )
                                row["E_max"] = float(emax_k)
                                row["phi_max_deg"] = float(phi_k * 180.0 / math.pi)
                            except ValueError:
                                pass

                        if emb_key == seed0_key and t_name == "t0" and inferential:
                            row["p_perm"] = float(p_perm)
                        per_stratum_rows.append(row)

                # figures for exemplar strata (cleaned only)
                if qc_mode == "cleaned" and stratum_name in {
                    "Ventricular cardiomyocytes",
                    "Fibroblasts",
                    "Capillary EC",
                }:
                    stratum_dir = figures_dir / stratum_name.replace(" ", "_")
                    ensure_dir(stratum_dir.as_posix())
                    _plot_umap_overlay(
                        embeddings[seed0_key],
                        expr,
                        stratum_dir / f"{display}_umap.png",
                        f"{stratum_name}: {display}",
                    )
                    plot_rsp_polar(
                        E_phi,
                        (stratum_dir / f"{display}_rsp_polar.png").as_posix(),
                        f"RSP: {display}",
                    )
                    if inferential and null_emax is not None:
                        _plot_null_hist(
                            null_emax,
                            E_max_obs,
                            stratum_dir / f"{display}_null_hist.png",
                            f"Null E_max: {display}",
                        )
                    _plot_phi_stability(
                        {k: v for k, v in phi_by_embed.items() if np.isfinite(v)},
                        stratum_dir / f"{display}_phi_stability.png",
                        "phi_max stability",
                    )

                # store Moran baseline
                moran_baseline_rows.append(
                    {
                        "qc_mode": qc_mode,
                        "stratum": stratum_name,
                        "gene_or_signature": display,
                        "moran_I": float(moran_val),
                    }
                )

            # FDR within stratum (seed0 + t0 only)
            primary_rows = [
                r
                for r in per_stratum_rows
                if r["embedding"] == seed0_key
                and r["threshold"] == "t0"
                and np.isfinite(r["p_perm"])
            ]
            pvals = np.array([r["p_perm"] for r in primary_rows], dtype=float)
            fdrs = bh_fdr(pvals)
            for row, fdr in zip(primary_rows, fdrs):
                row["FDR"] = float(fdr)
            all_results.extend(per_stratum_rows)

            # Embedding stability: rank correlations of E_max across seeds
            ref_scores = emax_by_seed.get(seed0_key, {})
            for emb_key, score_map in emax_by_seed.items():
                if emb_key == seed0_key:
                    continue
                common = list(set(ref_scores.keys()) & set(score_map.keys()))
                if len(common) < 2:
                    continue
                x = np.array([ref_scores[g] for g in common], dtype=float)
                y = np.array([score_map[g] for g in common], dtype=float)
                rho = _spearman_rank_corr(x, y)
                embedding_stability_rows.append(
                    {
                        "qc_mode": qc_mode,
                        "stratum": stratum_name,
                        "embedding": emb_key,
                        "spearman_Emax_rank": rho,
                        "n_features": len(common),
                    }
                )

            # Null calibration
            null_pvals = []
            for gene in rand_genes:
                expr = _get_expr_vector(adata_s, gene, expr_layer_rsp)
                f_template = expr > 0
                if f_template.sum() == 0 or f_template.sum() == f_template.size:
                    continue
                f_synth = permute_foreground_within_donor(f_template, donor_to_idx, rng)
                try:
                    _, _, emax_obs = compute_rsp_profile_from_boolean(
                        f_synth, angles_by_embed[seed0_key], n_bins=bins
                    )
                except ValueError:
                    continue
                null_emax = np.zeros(n_perm_null, dtype=float)
                for i in range(n_perm_null):
                    f_perm = permute_foreground_within_donor(f_synth, donor_to_idx, rng)
                    try:
                        _, _, emax = compute_rsp_profile_from_boolean(
                            f_perm, angles_by_embed[seed0_key], n_bins=bins
                        )
                        null_emax[i] = emax
                    except ValueError:
                        null_emax[i] = float("nan")
                null_emax = null_emax[np.isfinite(null_emax)]
                if null_emax.size == 0:
                    continue
                p_val = (1.0 + np.sum(null_emax >= emax_obs)) / (1.0 + n_perm_null)
                null_pvals.append(p_val)

            null_pvals = np.asarray(null_pvals, dtype=float)
            frac_sig = (
                float(np.mean(null_pvals <= 0.05)) if null_pvals.size else float("nan")
            )
            fdr_null = bh_fdr(null_pvals) if null_pvals.size else np.array([])
            frac_sig_fdr = (
                float(np.mean(fdr_null <= 0.05)) if fdr_null.size else float("nan")
            )
            null_rows.append(
                {
                    "qc_mode": qc_mode,
                    "stratum": stratum_name,
                    "n_synth": int(null_pvals.size),
                    "frac_sig_p05": frac_sig,
                    "frac_sig_fdr05": frac_sig_fdr,
                    "p_median": (
                        float(np.median(null_pvals))
                        if null_pvals.size
                        else float("nan")
                    ),
                    "p_mean": (
                        float(np.mean(null_pvals)) if null_pvals.size else float("nan")
                    ),
                }
            )

            # QQ data and plot
            if null_pvals.size:
                qq_dir = results_dir / "qq"
                ensure_dir(qq_dir.as_posix())
                qq_csv = qq_dir / f"{stratum_name.replace(' ', '_')}_qq.csv"
                obs = np.sort(null_pvals)
                exp = np.arange(1, obs.size + 1) / (obs.size + 1)
                pd.DataFrame({"expected": exp, "observed": obs}).to_csv(
                    qq_csv, index=False
                )
                qq_rows.append(
                    {
                        "qc_mode": qc_mode,
                        "stratum": stratum_name,
                        "qq_csv": qq_csv.as_posix(),
                    }
                )
                if qc_mode == "cleaned" and stratum_name in {
                    "Ventricular cardiomyocytes",
                    "Fibroblasts",
                    "Capillary EC",
                }:
                    _plot_qq(
                        null_pvals,
                        figures_dir
                        / stratum_name.replace(" ", "_")
                        / "null_calibration_qq.png",
                        f"Null calibration: {stratum_name}",
                    )

            # Donor artifact simulation
            donor_counts = adata_s.obs[donor_col].value_counts()
            if donor_counts.size > 0:
                donor_a = donor_counts.index[0]
                f_artifact = donor_ids == donor_a
                null_emax, E_max_obs, _, p_perm_art = _perm_null_emax_from_canonical(
                    f_artifact.astype(float),
                    angles_by_embed[seed0_key],
                    donor_ids,
                    bins,
                    n_perm,
                    seed=seed,
                )
                # Global shuffle ablation
                rng_local = np.random.default_rng(seed)
                null_global = np.zeros(n_perm, dtype=float)
                for i in range(n_perm):
                    f_perm = rng_local.permutation(f_artifact)
                    _, _, emax = compute_rsp_profile_from_boolean(
                        f_perm, angles_by_embed[seed0_key], n_bins=bins
                    )
                    null_global[i] = emax
                p_global = (1.0 + np.sum(null_global >= E_max_obs)) / (1.0 + n_perm)
                donor_artifact_rows.append(
                    {
                        "qc_mode": qc_mode,
                        "stratum": stratum_name,
                        "donor": str(donor_a),
                        "p_perm_donor_strat": float(p_perm_art),
                        "p_perm_global": float(p_global),
                    }
                )

            # DE baseline (minimal): leiden if not present
            if "leiden" not in adata_s.obs:
                sc.tl.leiden(adata_s, resolution=0.5, key_added="leiden")
            try:
                sc.tl.rank_genes_groups(adata_s, "leiden", method="t-test")
                de = sc.get.rank_genes_groups_df(adata_s, group=None)
                de = de.head(100)
                for _, row in de.iterrows():
                    de_baseline_rows.append(
                        {
                            "qc_mode": qc_mode,
                            "stratum": stratum_name,
                            "gene": row.get("names"),
                            "score": row.get("scores"),
                            "logfoldchanges": row.get("logfoldchanges"),
                        }
                    )
            except (ValueError, KeyError, RuntimeError) as exc:
                logger.warning(
                    "DE baseline skipped: qc_mode=%s stratum=%s reason=%s",
                    qc_mode,
                    stratum_name,
                    exc,
                )

            # Global shuffle ablation for panel genes
            for g in panel_genes + HOUSEKEEPING:
                if g not in adata_s.var_names:
                    continue
                expr = _get_expr_vector(adata_s, g, expr_layer_rsp)
                f_obs = expr > 0
                if f_obs.sum() == 0 or f_obs.sum() == f_obs.size:
                    continue
                try:
                    _, _, emax_obs = compute_rsp_profile(
                        expr, angles_by_embed[seed0_key], n_bins=bins
                    )
                except ValueError:
                    continue
                rng_local = np.random.default_rng(seed)
                null_global = np.zeros(n_perm, dtype=float)
                for i in range(n_perm):
                    f_perm = rng_local.permutation(f_obs)
                    _, _, emax = compute_rsp_profile_from_boolean(
                        f_perm, angles_by_embed[seed0_key], n_bins=bins
                    )
                    null_global[i] = emax
                p_global = (1.0 + np.sum(null_global >= emax_obs)) / (1.0 + n_perm)
                ablation_global_rows.append(
                    {
                        "qc_mode": qc_mode,
                        "stratum": stratum_name,
                        "gene": g,
                        "p_perm_global": float(p_global),
                    }
                )

    _write_prereg_outputs(
        results_dir=results_dir,
        prep_rows=prep_rows,
        all_results=all_results,
        null_rows=null_rows,
        jackknife_rows=jackknife_rows,
        threshold_rows=threshold_rows,
        donor_artifact_rows=donor_artifact_rows,
        moran_baseline_rows=moran_baseline_rows,
        de_baseline_rows=de_baseline_rows,
        embedding_stability_rows=embedding_stability_rows,
        ablation_global_rows=ablation_global_rows,
        multimodal_rows=multimodal_rows,
        batch_rows=batch_rows,
        qq_rows=qq_rows,
    )

    # Null calibration gate
    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        max_frac = float(cfg.get("null_calibration_max_frac", 0.1))
        failed = null_df[null_df["frac_sig_p05"] > max_frac]
        if not failed.empty:
            msg = (
                "Synthetic null calibration failed (inflated significant fraction). "
                "Check permutation or thresholds."
            )
            logger.error(msg)
            raise RuntimeError(msg)

    logger.info("Pipeline complete. Results in %s", results_dir.as_posix())
