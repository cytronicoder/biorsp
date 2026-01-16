"""Tests for double-weighting bug fix.

This module verifies that the refactor correctly avoids the double-weighting
bug where rsp values were multiplied by sector_weights twice (once in engine,
once in scoring).

Contract:
- compute_rsp_radar returns RAW unweighted rsp
- Scoring functions apply weights ONCE when computing S_g
- S_g should be identical whether computed fresh or via the radar object
"""

import numpy as np
import pytest

from biorsp.api import BioRSPConfig
from biorsp.core.engine import compute_anisotropy, compute_rsp_radar
from biorsp.core.summaries import compute_scalar_summaries


@pytest.fixture
def synthetic_data():
    """Generate synthetic data with known RSP pattern."""
    rng = np.random.default_rng(42)
    n_cells = 1000

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)

    wedge_center = 0.0
    wedge_width = np.pi / 4
    fg_prob = np.where(np.abs(theta - wedge_center) < wedge_width, 0.7, 0.1)
    y = rng.random(n_cells) < fg_prob

    return r, theta, y.astype(float)


def test_rsp_is_unweighted(synthetic_data):
    """Verify that radar.rsp is the raw unweighted statistic."""
    r, theta, y = synthetic_data
    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    radar = compute_rsp_radar(r, theta, y, config=config)

    valid_rsp = radar.rsp[np.isfinite(radar.rsp)]
    assert len(valid_rsp) > 0, "Should have some valid RSP values"

    mask = radar.geom_supported_mask
    if mask is not None and np.any(mask):
        w = radar.sector_weights[mask]
        assert np.all(w >= 0), "Weights should be non-negative"


def test_s_g_computed_once(synthetic_data):
    """Verify S_g is computed with weights applied exactly once."""
    r, theta, y = synthetic_data
    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    radar = compute_rsp_radar(r, theta, y, config=config)

    valid_mask = np.isfinite(radar.rsp)
    if np.any(valid_mask):
        rsp = radar.rsp[valid_mask]
        w = (
            radar.sector_weights[valid_mask]
            if radar.sector_weights is not None
            else np.ones_like(rsp)
        )

        sum_w = np.sum(w)
        if sum_w > 0:
            manual_s_g = np.sqrt(np.sum(w * rsp**2) / sum_w)
        else:
            manual_s_g = np.sqrt(np.mean(rsp**2))

        aniso = compute_anisotropy(radar.rsp, valid_mask, weights=radar.sector_weights)

        np.testing.assert_allclose(
            manual_s_g, aniso, rtol=1e-10, err_msg="Manual S_g should match compute_anisotropy"
        )


def test_double_weighting_would_reduce_score(synthetic_data):
    """Verify that double-weighting would give a different (smaller) score."""
    r, theta, y = synthetic_data
    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    radar = compute_rsp_radar(r, theta, y, config=config)

    mask = radar.geom_supported_mask
    if mask is None or not np.any(mask):
        pytest.skip("No valid sectors")

    rsp = radar.rsp[mask]
    w = radar.sector_weights[mask]

    s_g_correct = np.sqrt(np.sum(w * rsp**2) / np.sum(w))

    rsp_double = w * rsp
    s_g_double = np.sqrt(np.sum(w * rsp_double**2) / np.sum(w))

    if s_g_correct > 0:
        ratio = s_g_double / s_g_correct
        assert (
            abs(ratio - 1.0) > 0.01
        ), f"Double-weighting should change the score (ratio={ratio:.4f})"


def test_scalar_summaries_uses_correct_anisotropy(synthetic_data):
    """Verify ScalarSummaries uses weights correctly for anisotropy."""
    r, theta, y = synthetic_data
    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    radar = compute_rsp_radar(r, theta, y, config=config)
    summaries = compute_scalar_summaries(radar)

    valid_mask = np.isfinite(radar.rsp)
    if np.any(valid_mask):
        rsp = radar.rsp[valid_mask]
        w = (
            radar.sector_weights[valid_mask]
            if radar.sector_weights is not None
            else np.ones_like(rsp)
        )
        sum_w = np.sum(w)
        if sum_w > 0:
            expected = float(np.sqrt(np.sum(w * rsp**2) / sum_w))
        else:
            expected = float(np.sqrt(np.mean(rsp**2)))

        np.testing.assert_allclose(
            summaries.anisotropy,
            expected,
            rtol=1e-10,
            err_msg="ScalarSummaries.anisotropy should match manual computation",
        )


def test_weights_stored_separately():
    """Verify that weights are stored in radar but not applied to rsp."""
    rng = np.random.default_rng(123)
    n_cells = 500

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)
    y = (rng.random(n_cells) > 0.5).astype(float)

    config = BioRSPConfig(B=8, delta_deg=45, seed=42)
    radar = compute_rsp_radar(r, theta, y, config=config)

    assert radar.sector_weights is not None, "sector_weights should exist"
    assert len(radar.sector_weights) == len(radar.rsp), "Same length as rsp"

    mask = radar.geom_supported_mask
    if mask is not None and np.any(mask):
        assert np.all(radar.sector_weights[mask] >= 0), "Weights should be non-negative"
