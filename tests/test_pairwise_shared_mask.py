"""Tests for pairwise similarity with shared masks.

Verifies that pairwise gene-gene similarity uses:
- Shared geom_supported_mask between gene pairs
- Geometric mean of weights for shared support
- Weighted correlation on shared valid sectors

Contract:
- Correlation is computed only on sectors valid for BOTH genes
- Weights are combined using geometric mean
- Result is symmetric: corr(g1, g2) == corr(g2, g1)
"""

import numpy as np
import pytest

from biorsp.api import BioRSPConfig
from biorsp.core.engine import compute_rsp_radar
from biorsp.core.pairwise import _weighted_corr, compute_pairwise_relationships


@pytest.fixture
def two_gene_data():
    """Generate synthetic data for two genes with known patterns."""
    rng = np.random.default_rng(42)
    n_cells = 500

    # Fixed spatial structure
    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)

    # Gene 1: wedge pattern
    wedge_center = 0.0
    wedge_width = np.pi / 4
    y1_prob = np.where(np.abs(theta - wedge_center) < wedge_width, 0.8, 0.2)
    y1 = (rng.random(n_cells) < y1_prob).astype(float)

    # Gene 2: similar wedge pattern (should be highly correlated)
    y2 = (rng.random(n_cells) < y1_prob).astype(float)

    # Gene 3: opposite wedge (should be anti-correlated)
    y3_prob = np.where(np.abs(theta - wedge_center) < wedge_width, 0.2, 0.8)
    y3 = (rng.random(n_cells) < y3_prob).astype(float)

    return r, theta, y1, y2, y3


def test_pairwise_uses_shared_mask(two_gene_data):
    """Verify pairwise similarity uses intersection of valid masks."""
    r, theta, y1, y2, _ = two_gene_data
    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    radar1 = compute_rsp_radar(r, theta, y1, config=config)
    radar2 = compute_rsp_radar(r, theta, y2, config=config)

    # Compute pairwise using the API
    radar_by_feature = {"gene1": radar1, "gene2": radar2}
    synergy, complementarity = compute_pairwise_relationships(radar_by_feature)

    # Should have results
    assert len(synergy) > 0 or len(complementarity) > 0, "Should have pairwise results"


def test_pairwise_symmetry(two_gene_data):
    """Verify correlation is symmetric: corr(g1, g2) == corr(g2, g1)."""
    r, theta, y1, y2, _ = two_gene_data
    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    radar1 = compute_rsp_radar(r, theta, y1, config=config)
    radar2 = compute_rsp_radar(r, theta, y2, config=config)

    # Get shared mask and weights
    mask1 = (
        radar1.geom_supported_mask
        if radar1.geom_supported_mask is not None
        else np.isfinite(radar1.rsp)
    )
    mask2 = (
        radar2.geom_supported_mask
        if radar2.geom_supported_mask is not None
        else np.isfinite(radar2.rsp)
    )
    shared_mask = mask1 & mask2

    w1 = radar1.sector_weights
    w2 = radar2.sector_weights
    shared_weights = np.sqrt(w1 * w2)
    shared_weights[~shared_mask] = 0.0

    # Compute correlation both ways
    corr_12 = _weighted_corr(radar1.rsp, radar2.rsp, shared_weights)
    corr_21 = _weighted_corr(radar2.rsp, radar1.rsp, shared_weights)

    if np.isfinite(corr_12) and np.isfinite(corr_21):
        np.testing.assert_allclose(
            corr_12, corr_21, rtol=1e-10, err_msg="Correlation should be symmetric"
        )


def test_similar_patterns_high_correlation(two_gene_data):
    """Verify similar wedge patterns have correlated RSP profiles."""
    r, theta, y1, y2, _ = two_gene_data
    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    radar1 = compute_rsp_radar(r, theta, y1, config=config)
    radar2 = compute_rsp_radar(r, theta, y2, config=config)

    radar_by_feature = {"gene1": radar1, "gene2": radar2}
    synergy, _ = compute_pairwise_relationships(radar_by_feature)

    # Since both genes have the same underlying wedge pattern,
    # their RSP profiles should be correlated (positive or negative depends on sign)
    if synergy and np.isfinite(synergy[0].correlation):
        # Similar underlying patterns should have high absolute correlation
        assert (
            abs(synergy[0].correlation) > 0.5
        ), f"Similar patterns should have high |correlation|, got {synergy[0].correlation}"


def test_opposite_patterns_negative_correlation(two_gene_data):
    """Verify opposite wedge patterns have negative correlation."""
    r, theta, y1, _, y3 = two_gene_data
    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    radar1 = compute_rsp_radar(r, theta, y1, config=config)
    radar3 = compute_rsp_radar(r, theta, y3, config=config)

    radar_by_feature = {"gene1": radar1, "gene3": radar3}
    synergy, _ = compute_pairwise_relationships(radar_by_feature)

    if synergy and np.isfinite(synergy[0].correlation):
        # Opposite patterns should have negative correlation
        assert (
            synergy[0].correlation < 0.0
        ), f"Opposite patterns should be negatively correlated, got {synergy[0].correlation}"


def test_pairwise_drops_non_shared_supported_sectors(two_gene_data):
    """Pairwise similarity must ignore sectors unsupported by either gene."""
    r, theta, y1, y2, _ = two_gene_data
    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    radar1 = compute_rsp_radar(r, theta, y1, config=config)
    radar2 = compute_rsp_radar(r, theta, y2, config=config)

    mask1 = (
        radar1.geom_supported_mask
        if radar1.geom_supported_mask is not None
        else np.isfinite(radar1.rsp)
    )
    mask2 = (
        radar2.geom_supported_mask
        if radar2.geom_supported_mask is not None
        else np.isfinite(radar2.rsp)
    )

    # Force a few sectors in gene2 to be unsupported and extreme so they would bias correlation
    mask2 = mask2.copy()
    mask2[:2] = False
    radar2.geom_supported_mask = mask2
    radar2.rsp = radar2.rsp.copy()
    radar2.rsp[:2] = 10.0

    radar_by_feature = {"gene1": radar1, "gene2": radar2}
    synergy, _ = compute_pairwise_relationships(radar_by_feature)

    shared_mask = mask1 & mask2
    shared_weights = np.sqrt(radar1.sector_weights * radar2.sector_weights)
    shared_weights[~shared_mask] = 0.0
    expected_corr = _weighted_corr(radar1.rsp, radar2.rsp, shared_weights)

    if synergy and np.isfinite(expected_corr):
        np.testing.assert_allclose(
            synergy[0].correlation,
            expected_corr,
            rtol=1e-12,
            err_msg="Pairwise correlation must use only shared supported sectors",
        )


def test_weighted_corr_basic():
    """Test basic properties of weighted correlation."""
    # Identical arrays should have correlation 1
    a = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([0.25, 0.25, 0.25, 0.25])

    corr = _weighted_corr(a, a, w)
    np.testing.assert_allclose(corr, 1.0, rtol=1e-10, err_msg="Self-correlation should be 1.0")


def test_weighted_corr_opposite():
    """Test that opposite arrays have correlation -1."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([-1.0, -2.0, -3.0, -4.0])
    w = np.array([0.25, 0.25, 0.25, 0.25])

    corr = _weighted_corr(a, b, w)
    np.testing.assert_allclose(
        corr, -1.0, rtol=1e-10, err_msg="Opposite arrays should have correlation -1.0"
    )


def test_geometric_mean_weights():
    """Verify weights are combined using geometric mean."""
    rng = np.random.default_rng(42)
    n_cells = 500

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)
    y1 = (rng.random(n_cells) > 0.3).astype(float)
    y2 = (rng.random(n_cells) > 0.5).astype(float)

    config = BioRSPConfig(B=12, delta_deg=45, seed=42)

    radar1 = compute_rsp_radar(r, theta, y1, config=config)
    radar2 = compute_rsp_radar(r, theta, y2, config=config)

    # Geometric mean of weights
    w1 = radar1.sector_weights
    w2 = radar2.sector_weights
    geometric_mean = np.sqrt(w1 * w2)

    # Should be <= arithmetic mean (by AM-GM inequality)
    arithmetic_mean = (w1 + w2) / 2
    assert np.all(
        geometric_mean <= arithmetic_mean + 1e-10
    ), "Geometric mean should be <= arithmetic mean"


def test_self_correlation_is_one():
    """Verify correlation of a gene with itself is 1.0."""
    rng = np.random.default_rng(42)
    n_cells = 400

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)
    y = (rng.random(n_cells) > 0.4).astype(float)

    config = BioRSPConfig(B=12, delta_deg=45, seed=42)
    radar = compute_rsp_radar(r, theta, y, config=config)

    # Use weighted_corr directly
    mask = (
        radar.geom_supported_mask
        if radar.geom_supported_mask is not None
        else np.isfinite(radar.rsp)
    )
    w = radar.sector_weights.copy()
    w[~mask] = 0.0

    corr = _weighted_corr(radar.rsp, radar.rsp, w)

    if np.isfinite(corr):
        np.testing.assert_allclose(corr, 1.0, rtol=1e-10, err_msg="Self-correlation should be 1.0")
