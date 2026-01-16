import numpy as np

from biorsp.api import BioRSPConfig
from biorsp.core.engine import compute_rsp_radar
from biorsp.core.inference import compute_p_value


def test_mask_stability_across_permutations():
    """Verify that the valid_mask is identical for observed and all permutations."""
    rng = np.random.default_rng(42)
    n = 100
    r = rng.uniform(0, 1, n)
    theta = rng.uniform(-np.pi, np.pi, n)
    y = rng.choice([0, 1], n)

    config = BioRSPConfig(
        B=10, delta_deg=45.0, min_fg_sector=5, min_bg_sector=5, empty_fg_policy="nan"
    )

    res = compute_p_value(r, theta, y, n_perm=5, seed=42, config=config)

    radar_obs = compute_rsp_radar(r, theta, y, config=config)
    assert np.array_equal(res.valid_mask, ~np.isnan(radar_obs.rsp))


def test_foreground_tie_breaking_determinism():
    """Verify that foreground tie-breaking (if any) is deterministic with seed."""

    n = 100
    r = np.zeros(n)
    theta = np.linspace(-np.pi, np.pi, n)
    y = np.zeros(n)
    y[:50] = 1

    config = BioRSPConfig(B=1, delta_deg=360.0, min_fg_sector=1, min_bg_sector=1)

    res1 = compute_rsp_radar(r, theta, y, config=config)
    res2 = compute_rsp_radar(r, theta, y, config=config)

    assert np.allclose(res1.rsp, res2.rsp, equal_nan=True)


def test_zero_inflation_handling():
    """Verify that zero-inflated foreground (common in scRNA-seq) is handled gracefully."""
    n = 1000
    r = np.random.uniform(0, 1, n)
    theta = np.random.uniform(-np.pi, np.pi, n)

    y = np.zeros(n)
    y[:100] = np.random.uniform(0.1, 10, 100)

    config = BioRSPConfig(B=10, delta_deg=45.0, min_fg_sector=1, min_bg_sector=1)

    res = compute_rsp_radar(r, theta, y, config=config)
    assert np.any(np.isfinite(res.rsp))
