"""Public API regression tests for schema and semantics.

These tests enforce the current public specification:
- Coverage is defined by biological thresholding, independent of foreground quantiles.
- Empty foreground sectors with empty_fg_policy='zero' contribute zero to Spatial_Bias_Score and stay finite.
"""

import numpy as np
import pytest

from biorsp.api import BioRSPConfig, score_genes
from biorsp.core.engine import compute_rsp_radar


def test_coverage_not_affected_by_foreground_quantile():
    """Coverage must be threshold-based and invariant to foreground quantile choice."""
    pytest.importorskip("anndata")
    from anndata import AnnData

    rng = np.random.default_rng(123)
    coords = rng.normal(size=(200, 2))
    expr = rng.integers(low=0, high=3, size=(200, 1))

    adata = AnnData(X=expr, obsm={"X_spatial": coords})
    adata.var_names = ["gene"]

    cfg_lo = BioRSPConfig(
        foreground_quantile=0.1, foreground_mode="quantile", expr_threshold_mode="detect"
    )
    cfg_hi = BioRSPConfig(
        foreground_quantile=0.99, foreground_mode="quantile", expr_threshold_mode="detect"
    )

    res_lo = score_genes(adata, ["gene"], embedding_key="X_spatial", config=cfg_lo)
    res_hi = score_genes(adata, ["gene"], embedding_key="X_spatial", config=cfg_hi)

    true_coverage = float(np.mean(expr[:, 0] >= 1))

    assert res_lo.loc[0, "Coverage"] == pytest.approx(true_coverage, rel=0, abs=1e-9)
    assert res_hi.loc[0, "Coverage"] == pytest.approx(true_coverage, rel=0, abs=1e-9)


def test_empty_foreground_zero_policy_keeps_spatial_score_zero():
    """When the foreground is empty, Spatial_Bias_Score should be zero and finite under zero policy."""
    rng = np.random.default_rng(2024)
    theta = rng.uniform(-np.pi, np.pi, 120)
    r = rng.uniform(0.1, 1.0, 120)
    fg_mask = np.zeros_like(theta)

    config = BioRSPConfig(
        B=12, delta_deg=30, empty_fg_policy="zero", min_fg_sector=0, min_fg_total=0
    )
    radar = compute_rsp_radar(r, theta, fg_mask, config=config)

    valid_mask = radar.geom_supported_mask
    if valid_mask is None:
        valid_mask = np.isfinite(radar.rsp)

    assert np.all(np.isfinite(radar.rsp[valid_mask])), "Supported sectors must stay finite"

    weights = (
        radar.sector_weights[valid_mask]
        if radar.sector_weights is not None
        else np.ones_like(radar.rsp[valid_mask])
    )
    rsp = radar.rsp[valid_mask]
    if weights.size and np.sum(weights) > 0:
        spatial_score = float(np.sqrt(np.sum(weights * rsp**2) / np.sum(weights)))
    else:
        spatial_score = float(np.sqrt(np.mean(rsp**2))) if rsp.size else 0.0

    assert spatial_score == pytest.approx(0.0, abs=1e-12)
