import json

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from biorsp import __all__ as biorsp_public
from biorsp.pipeline import __all__ as pipeline_public
from biorsp.pipeline.hierarchy import run_case_study

matplotlib.use("Agg")


def _make_connectivities(n_cells: int) -> sp.csr_matrix:
    rows = []
    cols = []
    vals = []
    for i in range(n_cells):
        for j in [(i - 1) % n_cells, (i + 1) % n_cells]:
            rows.append(i)
            cols.append(j)
            vals.append(1.0)
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells), dtype=float)
    return mat


def _make_adata(*, include_donor: bool, n_cells: int = 80, n_genes: int = 30) -> ad.AnnData:
    rng = np.random.default_rng(0)
    x = rng.poisson(1.2, size=(n_cells, n_genes)).astype(float)
    x[rng.uniform(size=x.shape) < 0.35] = 0.0

    symbols = [f"GENE{i}" for i in range(n_genes)]
    symbols[0] = "TNNT2"
    symbols[1] = "VWF"
    symbols[2] = "PTPRC"
    symbols[3] = "RGS5"

    var = pd.DataFrame(
        {
            "hugo_symbol": symbols,
            "mt": [s.startswith("MT-") for s in symbols],
        },
        index=[f"ENSG{i:09d}" for i in range(n_genes)],
    )

    cluster_ids = np.array([f"AZ:{i:07d}" for i in np.repeat([1, 2, 3, 4], n_cells // 4)], dtype=object)
    if cluster_ids.size < n_cells:
        cluster_ids = np.concatenate([cluster_ids, np.array(["AZ:0000001"] * (n_cells - cluster_ids.size))])
    cluster_ids = cluster_ids[:n_cells]

    obs = pd.DataFrame(
        {
            "azimuth_id": cluster_ids,
            "azimuth_label": np.where(cluster_ids == "AZ:0000001", "Fibroblast", "Cardiomyocyte"),
            "total_counts": np.maximum(1.0, x.sum(axis=1)),
            "pct_counts_mt": rng.uniform(0.0, 15.0, size=n_cells),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    if include_donor:
        obs["hubmap_id"] = np.array([f"D{(i % 4) + 1}" for i in range(n_cells)], dtype=object)

    centers = {
        "AZ:0000001": np.array([0.0, 0.0]),
        "AZ:0000002": np.array([4.0, 0.0]),
        "AZ:0000003": np.array([0.0, 4.0]),
        "AZ:0000004": np.array([4.0, 4.0]),
    }
    umap = np.vstack([centers[cid] + rng.normal(scale=0.2, size=2) for cid in cluster_ids]).astype(float)

    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.obsm["X_umap"] = umap
    adata.obsp["connectivities"] = _make_connectivities(n_cells)
    adata.obsp["distances"] = _make_connectivities(n_cells)
    return adata


def test_case_study_marker_panel_resolution_and_fallback(tmp_path) -> None:
    adata = _make_adata(include_donor=True)
    outdir = tmp_path / "run"

    run_case_study(
        adata=adata,
        outdir=outdir,
        do_hierarchy=False,
        donor_key="hubmap_id",
        cluster_key="azimuth_id",
        celltype_key="azimuth_label",
        bins=24,
        n_perm=8,
        seed=0,
        min_cells_per_cluster=5,
        min_cells_per_mega=10,
    )

    marker_csv = outdir / "tables" / "marker_panel_found_missing.csv"
    assert marker_csv.exists()
    marker_df = pd.read_csv(marker_csv)
    found_markers = marker_df[(marker_df["status"] == "marker_found") & (~marker_df["auto_gene"].astype(bool))]
    auto_genes = marker_df[marker_df["auto_gene"].astype(bool)]

    assert found_markers.shape[0] < 12
    assert (found_markers.shape[0] + auto_genes.shape[0]) >= 12


def test_case_study_donor_missing_fallback_to_library_quantiles(tmp_path) -> None:
    adata = _make_adata(include_donor=False)
    outdir = tmp_path / "run"

    run_case_study(
        adata=adata,
        outdir=outdir,
        do_hierarchy=False,
        donor_key=None,
        cluster_key="azimuth_id",
        celltype_key="azimuth_label",
        bins=24,
        n_perm=6,
        seed=0,
        min_cells_per_cluster=5,
        min_cells_per_mega=10,
    )

    scope_meta = json.loads((outdir / "hierarchy" / "global" / "metadata.json").read_text())
    assert scope_meta["inference"]["mode"] == "library_quantile"
    assert scope_meta["inference"]["inference_limited"] is True


def test_case_study_mega_split_deterministic(tmp_path) -> None:
    adata = _make_adata(include_donor=True)
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    kwargs = {
        "adata": adata,
        "do_hierarchy": True,
        "donor_key": "hubmap_id",
        "cluster_key": "azimuth_id",
        "celltype_key": "azimuth_label",
        "bins": 24,
        "n_perm": 5,
        "seed": 7,
        "min_cells_per_cluster": 5,
        "min_cells_per_mega": 10,
    }
    run_case_study(outdir=out_a, **kwargs)
    run_case_study(outdir=out_b, **kwargs)

    map_a = pd.read_csv(out_a / "hierarchy" / "mega" / "cluster_to_mega.csv")
    map_b = pd.read_csv(out_b / "hierarchy" / "mega" / "cluster_to_mega.csv")
    pd.testing.assert_frame_equal(map_a, map_b)


def test_public_api_exports_case_study_only() -> None:
    assert "run_case_study" in biorsp_public
    legacy_name = "run_" + "hierarchy"
    assert legacy_name not in biorsp_public
    assert pipeline_public == ["run_case_study"]
