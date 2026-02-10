from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biorsp.config import load_json_config
from biorsp.pipeline_utils import bh_fdr, circular_sd, detect_obs_col, normalize_label


class _DummyAdata:
    def __init__(self, columns: dict[str, list[str]]):
        self.obs = pd.DataFrame(columns)


def test_normalize_label():
    assert normalize_label("  FibroBlast  ") == "fibroblast"


def test_detect_obs_col_uses_provided_or_candidate():
    adata = _DummyAdata({"donor_id": ["d1", "d2"], "cell_type": ["a", "b"]})
    assert detect_obs_col(adata, "donor_id", ["donor"]) == "donor_id"
    assert detect_obs_col(adata, None, ["donor", "donor_id"]) == "donor_id"
    with pytest.raises(KeyError):
        detect_obs_col(adata, "missing", ["donor"])


def test_circular_sd_and_bh_fdr():
    assert circular_sd(np.zeros(10)) < 1e-6
    pvals = np.array([0.01, 0.02, 0.2, 0.9])
    fdr = bh_fdr(pvals)
    assert fdr.shape == pvals.shape
    assert np.all((fdr >= 0) & (fdr <= 1))
    assert np.all(np.diff(np.sort(fdr)) >= 0)


def test_config_loader_used_by_shared_pipeline_paths(tmp_path):
    cfg_path = tmp_path / "ok.json"
    cfg_path.write_text('{"h5ad_path":"x.h5ad","strata":{}}', encoding="utf-8")
    cfg = load_json_config(cfg_path)
    assert cfg["h5ad_path"] == "x.h5ad"
