import numpy as np
from anndata import AnnData

from biorsp.core.scoring import score_genes_impl
from biorsp.utils.config import BioRSPConfig


def test_abstention_contract_invariants():
    rng = np.random.default_rng(7)
    n_cells = 120
    coords = rng.normal(size=(n_cells, 2))
    x = np.zeros((n_cells, 1))
    x[:30, 0] = 10.0

    adata = AnnData(x, obsm={"X_test": coords})
    adata.var_names = ["gene0"]

    config = BioRSPConfig(
        B=16,
        delta_deg=30.0,
        foreground_mode="absolute",
        foreground_threshold=5.0,
        min_fg_total=1,
        min_fg_sector=15,
        min_bg_sector=6,
        min_valid_sectors=12,
        n_permutations=50,
        perm_mode_scoring="global",
    )

    df = score_genes_impl(
        adata,
        genes=["gene0"],
        embedding_key="X_test",
        subset=None,
        config=config,
    )

    row = df.iloc[0]
    assert row["p_value"] is None or np.isnan(row["p_value"])
    assert np.isnan(row["spatial_score"])
    assert np.isnan(row["r_mean"])
    assert "abstain" in str(row["warnings"]).lower()
