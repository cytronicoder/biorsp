from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)

from biorsp import evaluation, genomewide


def _make_tiny_adata(path: Path) -> None:
    X = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [2.0, 0.0, 1.0, 1.0],
            [0.0, 2.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {
            "donor": ["d1", "d1", "d1", "d2", "d2", "d2"],
            "cell_type": [
                "Fibroblast",
                "Fibroblast",
                "Fibroblast",
                "Fibroblast",
                "Fibroblast",
                "Fibroblast",
            ],
            "n_genes_by_counts": [300, 320, 310, 305, 315, 325],
            "total_counts": [1200, 1100, 1000, 1150, 1180, 1120],
            "pct_counts_mt": [5.0, 4.0, 6.0, 5.5, 4.5, 5.2],
        },
        index=[f"c{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["ACTB", "GAPDH", "RPLP0", "COL1A1"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = np.array(
        [[0.0, 0.0], [1.0, 0.2], [0.2, 1.0], [1.0, 1.0], [0.6, 1.2], [1.2, 0.6]]
    )
    adata.obsp["connectivities"] = sp.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
    )
    adata.write_h5ad(path)


def _scanpy_stub():
    def _read_h5ad(path: str):
        return ad.read_h5ad(path)

    def _normalize_total(_adata, target_sum=1e4):
        return None

    def _log1p(_adata):
        return None

    def _neighbors(_adata, **_kwargs):
        return None

    def _score_genes(adata, gene_list, score_name, use_raw=False):
        adata.obs[score_name] = 0.0

    def _leiden(adata, resolution=0.5, key_added="leiden"):
        adata.obs[key_added] = "0"

    def _rank_genes_groups(_adata, groupby, method="t-test"):
        return None

    def _rank_genes_groups_df(_adata, group=None):
        return pd.DataFrame(
            {"names": ["ACTB"], "scores": [1.0], "logfoldchanges": [0.0]}
        )

    return SimpleNamespace(
        read_h5ad=_read_h5ad,
        pp=SimpleNamespace(
            normalize_total=_normalize_total, log1p=_log1p, neighbors=_neighbors
        ),
        tl=SimpleNamespace(
            score_genes=_score_genes,
            leiden=_leiden,
            rank_genes_groups=_rank_genes_groups,
        ),
        get=SimpleNamespace(rank_genes_groups_df=_rank_genes_groups_df),
    )


def _fake_embeddings(adata, _sc_module, *_args, **_kwargs):
    emb = np.asarray(adata.obsm["X_umap"], dtype=float)
    adata.obsm["X_umap_seed0"] = emb.copy()
    adata.obsm["X_pca2d"] = emb.copy()
    return {"umap_seed0": emb.copy(), "pca2d": emb.copy()}


def test_prereg_pipeline_tiny_integration(tmp_path: Path, monkeypatch):
    h5ad_path = tmp_path / "tiny.h5ad"
    _make_tiny_adata(h5ad_path)
    outdir = tmp_path / "out_prereg"
    cfg = {
        "h5ad_path": str(h5ad_path),
        "outdir": str(outdir),
        "donor_col": "donor",
        "celltype_col": "cell_type",
        "umap_seeds": [0],
        "bins": 12,
        "seed": 0,
        "n_perm": 2,
        "n_perm_null": 2,
        "random_genes": 2,
        "min_donors": 2,
        "min_cells_per_donor": 1,
        "pca_n_comps": 2,
        "neighbors_k": 2,
        "umap_min_dist": 0.3,
        "umap_spread": 1.0,
        "include_secondary": False,
        "split_mural": False,
        "include_modules": False,
        "run_stress_mode": False,
        "rsp_expr_layer": None,
        "moran_expr_layer": None,
        "null_calibration_max_frac": 1.0,
        "strata": {"Fibroblast": ["Fibroblast"]},
    }
    cfg_path = tmp_path / "prereg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    monkeypatch.setattr(evaluation, "_get_scanpy", _scanpy_stub)
    monkeypatch.setattr(evaluation, "_compute_embeddings", _fake_embeddings)

    evaluation.run_prereg_pipeline(str(cfg_path))
    assert (outdir / "results" / "preprocessing_report.csv").exists()
    assert (outdir / "results" / "biorsp_stratum_results.csv").exists()


def test_genomewide_pipeline_tiny_integration(tmp_path: Path, monkeypatch):
    h5ad_path = tmp_path / "tiny.h5ad"
    _make_tiny_adata(h5ad_path)
    outdir = tmp_path / "out_genomewide"
    cfg = {
        "h5ad_path": str(h5ad_path),
        "outdir": str(outdir),
        "donor_col": "donor",
        "celltype_col": "cell_type",
        "umap_seeds": [0],
        "reference_embedding_name": "X_umap_seed0",
        "bins": 12,
        "seed": 0,
        "stage1_perms": 2,
        "stage2_perms": 2,
        "stage1_p_cutoff": None,
        "stage1_top_k": 2,
        "min_detect_frac": 0.0,
        "min_detect_n": 1,
        "min_var": 0.0,
        "feature_mode": "continuous",
        "thresholds_to_validate": ["t0"],
        "topN_plots_per_stratum": 2,
        "min_donors": 2,
        "min_cells_per_donor": 1,
        "pca_n_comps": 2,
        "neighbors_k": 2,
        "umap_min_dist": 0.3,
        "umap_spread": 1.0,
        "resume": False,
        "include_modules": False,
        "binary_expr_layer": None,
        "continuous_expr_layer": None,
        "embedding_phi_sd_deg": 180.0,
        "donor_jackknife_sd": 1.0,
        "strata": {"Fibroblast": ["Fibroblast"]},
    }
    cfg_path = tmp_path / "genomewide.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    monkeypatch.setattr(genomewide, "_get_scanpy", _scanpy_stub)
    monkeypatch.setattr(genomewide, "_compute_embeddings", _fake_embeddings)

    genomewide.run_genomewide_pipeline(str(cfg_path))
    assert (outdir / "results" / "biorsp_genomewide_results.csv").exists()
    assert (outdir / "results" / "biorsp_genomewide_top_hits.csv").exists()
