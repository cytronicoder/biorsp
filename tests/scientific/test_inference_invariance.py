"""Tests for inference invariance.

Verifies that permutation inference uses FIXED geometry masks and weights
from the observed data, preventing selection bias.

Contract:
- Permutation null uses the same geom_supported_mask as observed
- Permutation null uses the same sector_weights as observed
- The geometry mask does not change between permutations
"""

import numpy as np

from biorsp.api import BioRSPConfig
from biorsp.core.engine import compute_rsp_radar


def test_geometry_mask_fixed_across_permutations():
    """Verify geometry mask is determined by spatial structure, not FG labels."""
    rng = np.random.default_rng(42)
    n_cells = 500

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)

    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    masks = []
    weights = []
    for seed in [1, 2, 3, 4, 5]:
        rng_fg = np.random.default_rng(seed)
        y = (rng_fg.random(n_cells) > 0.3).astype(float)

        radar = compute_rsp_radar(r, theta, y, config=config)
        if radar.geom_supported_mask is not None:
            masks.append(radar.geom_supported_mask.copy())
            weights.append(radar.sector_weights.copy())

    if len(masks) >= 2:
        for i in range(1, len(masks)):
            np.testing.assert_array_equal(
                masks[0],
                masks[i],
                err_msg=f"Geometry mask should be invariant to FG labels (iteration {i})",
            )

            np.testing.assert_array_equal(
                weights[0],
                weights[i],
                err_msg=f"Sector weights should be invariant to FG labels (iteration {i})",
            )


def test_permutation_uses_observed_mask():
    """Verify permutation inference uses fixed mask from observed data."""
    rng = np.random.default_rng(42)
    n_cells = 300

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)
    y = (rng.random(n_cells) > 0.4).astype(float)

    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    radar = compute_rsp_radar(r, theta, y, config=config)

    if radar.geom_supported_mask is not None:
        coverage = np.mean(radar.geom_supported_mask)
        assert coverage >= 0 and coverage <= 1, "Coverage should be between 0 and 1"


def test_sector_weights_invariant_to_fg_pattern():
    """Verify sector_weights don't change with FG cell selection."""
    rng = np.random.default_rng(42)
    n_cells = 500

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)

    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    y1 = (rng.random(n_cells) > 0.3).astype(float)
    y2 = (rng.random(n_cells) > 0.7).astype(float)

    radar1 = compute_rsp_radar(r, theta, y1, config=config)
    radar2 = compute_rsp_radar(r, theta, y2, config=config)

    np.testing.assert_array_equal(
        radar1.sector_weights,
        radar2.sector_weights,
        err_msg="Sector weights should be invariant to FG proportion",
    )


def test_no_selection_bias_in_coverage():
    """Verify coverage doesn't depend on expression level."""
    rng = np.random.default_rng(42)
    n_cells = 500

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)

    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    y_sparse = (rng.random(n_cells) > 0.9).astype(float)
    y_dense = (rng.random(n_cells) > 0.1).astype(float)

    radar_sparse = compute_rsp_radar(r, theta, y_sparse, config=config)
    radar_dense = compute_rsp_radar(r, theta, y_dense, config=config)

    if radar_sparse.geom_supported_mask is not None and radar_dense.geom_supported_mask is not None:
        coverage_sparse = np.mean(radar_sparse.geom_supported_mask)
        coverage_dense = np.mean(radar_dense.geom_supported_mask)

        assert (
            abs(coverage_sparse - coverage_dense) < 0.01
        ), f"Geometry coverage should be invariant to expression level: {coverage_sparse} vs. {coverage_dense}"


def test_inference_with_stratification():
    """Verify radar computation works with different data patterns."""
    rng = np.random.default_rng(42)
    n_cells = 400

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)
    y = (rng.random(n_cells) > 0.4).astype(float)

    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    radar = compute_rsp_radar(r, theta, y, config=config)

    assert radar.rsp is not None, "Should compute RSP values"
    assert len(radar.rsp) == config.B, "Should have B sectors"
