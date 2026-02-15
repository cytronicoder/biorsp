import anndata as ad
import numpy as np
import pandas as pd

from biorsp.core.features import resolve_feature_index


def test_resolve_feature_from_hugo_symbol() -> None:
    x = np.zeros((3, 3), dtype=float)
    var = pd.DataFrame(
        {
            "hugo_symbol": ["TNNT2", "VWF", "COL1A1"],
        },
        index=["ENSG00000118194", "ENSG00000110799", "ENSG00000108821"],
    )
    obs = pd.DataFrame(index=["c0", "c1", "c2"])
    adata = ad.AnnData(X=x, obs=obs, var=var)

    idx, label, symbol_col, source = resolve_feature_index(adata, "TNNT2")
    assert idx == 0
    assert label == "TNNT2"
    assert symbol_col == "hugo_symbol"
    assert source == "symbol"
