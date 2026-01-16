"""Tests for mask semantics.

Verifies the semantic correctness of the three mask types:
- geom_supported_mask: Sector has sufficient total cells and robust scale
- contrast_supported_mask: Sector has valid FG/BG contrast (enough of each)
- forced_zero_mask: Sector had zero FG cells and rsp was forced to 0

Contract:
- forced_zero sectors have rsp == 0 (not NaN)
- geom_supported = total >= min_total AND robust_scale >= threshold
- contrast_supported = FG >= min_fg AND BG >= min_bg
- All three masks are disjoint in their "exclusion reasons"
"""

import numpy as np

from biorsp.api import BioRSPConfig
from biorsp.core.engine import compute_rsp_radar


def test_forced_zero_sectors_are_numeric_zeros():
    """Verify that forced_zero sectors have rsp = 0.0, not NaN."""
    rng = np.random.default_rng(42)
    n_cells = 200

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)

    y = np.ones(n_cells)
    empty_mask = (theta > np.pi / 2) & (theta < np.pi)
    y[empty_mask] = 0

    config = BioRSPConfig(B=8, delta_deg=45, seed=42, empty_fg_policy="zero")
    radar = compute_rsp_radar(r, theta, y, config=config)

    if radar.forced_zero_mask is not None and np.any(radar.forced_zero_mask):
        forced_zero_rsp = radar.rsp[radar.forced_zero_mask]

        assert np.all(
            forced_zero_rsp == 0.0
        ), f"Forced-zero sectors should have rsp=0.0, got {forced_zero_rsp}"
        assert not np.any(np.isnan(forced_zero_rsp)), "Forced-zero sectors should not have NaN"


def test_geom_supported_reflects_total_count():
    """Verify geom_supported_mask is False when total cells < threshold."""
    rng = np.random.default_rng(42)
    n_cells = 100

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)
    y = (rng.random(n_cells) > 0.3).astype(float)

    config = BioRSPConfig(B=16, delta_deg=30, seed=42, min_total_per_sector=50)
    radar = compute_rsp_radar(r, theta, y, config=config)

    total_per_sector = radar.counts_fg + radar.counts_bg

    low_total_sectors = total_per_sector < 10
    if np.any(low_total_sectors) and radar.geom_supported_mask is not None:
        unsupported = ~radar.geom_supported_mask
        assert np.all(
            unsupported[low_total_sectors]
        ), "Very low-count sectors should not be geom_supported"


def test_contrast_supported_requires_fg_and_bg():
    """Verify contrast_supported_mask requires both FG and BG cells."""
    rng = np.random.default_rng(42)
    n_cells = 500

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)

    y = np.ones(n_cells)
    bg_mask = theta < -np.pi / 2
    y[bg_mask] = 0

    config = BioRSPConfig(B=8, delta_deg=45, seed=42, min_fg_sector=5, min_bg_sector=5)
    radar = compute_rsp_radar(r, theta, y, config=config)

    if radar.contrast_supported_mask is not None:
        no_bg_sectors = radar.counts_bg == 0
        contrast_unsupported = ~radar.contrast_supported_mask

        assert np.all(
            contrast_unsupported[no_bg_sectors]
        ), "Sectors with no BG should not be contrast_supported"


def test_masks_are_boolean_arrays():
    """Verify all masks are proper boolean arrays."""
    rng = np.random.default_rng(42)
    n_cells = 300

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)
    y = (rng.random(n_cells) > 0.4).astype(float)

    config = BioRSPConfig(B=12, delta_deg=45, seed=42)
    radar = compute_rsp_radar(r, theta, y, config=config)

    if radar.geom_supported_mask is not None:
        assert radar.geom_supported_mask.dtype == bool, "geom_supported_mask should be bool"
        assert len(radar.geom_supported_mask) == config.B, "Mask length should match B"

    if radar.contrast_supported_mask is not None:
        assert radar.contrast_supported_mask.dtype == bool, "contrast_supported_mask should be bool"

    if radar.forced_zero_mask is not None:
        assert radar.forced_zero_mask.dtype == bool, "forced_zero_mask should be bool"


def test_invalid_reason_populated():
    """Verify invalid_reason is populated for unsupported sectors."""
    rng = np.random.default_rng(42)
    n_cells = 100

    theta = rng.uniform(-np.pi, np.pi, n_cells)
    r = rng.uniform(0.1, 1.0, n_cells)
    y = (rng.random(n_cells) > 0.8).astype(float)

    config = BioRSPConfig(B=12, delta_deg=30, seed=42)
    radar = compute_rsp_radar(r, theta, y, config=config)

    if radar.invalid_reason is not None:
        assert len(radar.invalid_reason) == config.B, "invalid_reason should have B entries"

        has_reason = np.array([bool(r) for r in radar.invalid_reason])

        if radar.geom_supported_mask is not None:
            assert np.all(
                ~radar.geom_supported_mask[has_reason] | radar.forced_zero_mask[has_reason]
                if radar.forced_zero_mask is not None
                else ~radar.geom_supported_mask[has_reason]
            ), "Sectors with invalid_reason should be unsupported or forced-zero"
