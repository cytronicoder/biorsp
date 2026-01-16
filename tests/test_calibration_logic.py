import numpy as np

from biorsp.api import BioRSPConfig
from biorsp.core.adequacy import assess_adequacy
from biorsp.core.engine import compute_rsp_radar
from biorsp.core.inference import compute_p_value
from biorsp.preprocess.stratification import get_strata_indices


def test_strata_merging():
    """Test that small strata are merged correctly."""
    n = 100
    r = np.linspace(0, 1, n)
    theta = np.zeros(n)

    indices = get_strata_indices(r, theta, n_r_bins=2, min_stratum_size=40, mode="radial")
    assert len(indices) == 2
    assert len(indices[0]) == 50
    assert len(indices[1]) == 50

    indices = get_strata_indices(r, theta, n_r_bins=3, min_stratum_size=40, mode="radial")

    assert len(indices) == 1
    assert len(indices[0]) == 100


def test_p_value_finite_correction():
    """Test that p-value uses (count + 1) / (n_perm + 1)."""
    n = 1000
    r = np.random.uniform(0, 1, n)
    theta = np.random.uniform(0, 2 * np.pi, n)
    y = np.zeros(n)
    y[:200] = 1.0

    config = BioRSPConfig(B=4, n_permutations=9, perm_mode="none", min_fg_sector=5, min_bg_sector=5)
    res = compute_p_value(r, theta, y, config=config, n_perm=9, seed=42)

    assert np.isfinite(res.p_value)
    assert res.p_value >= 0.1
    assert res.p_value <= 1.0

    assert np.isclose(res.p_value * 10, np.round(res.p_value * 10))


def test_reproducibility():
    """Test that same seed gives same p-value."""
    n = 1000
    r = np.random.uniform(0, 1, n)
    theta = np.random.uniform(0, 2 * np.pi, n)
    y = np.zeros(n)
    y[:200] = 1.0

    config = BioRSPConfig(
        B=4, n_permutations=10, perm_mode="radial", min_fg_sector=5, min_bg_sector=5
    )
    res1 = compute_p_value(r, theta, y, config=config, n_perm=10, seed=42)
    res2 = compute_p_value(r, theta, y, config=config, n_perm=10, seed=42)

    assert np.isfinite(res1.p_value)
    assert res1.p_value == res2.p_value
    assert np.array_equal(res1.null_stats, res2.null_stats)


def test_mask_freezing():
    """Test that the adequacy mask is frozen across permutations."""
    n = 200
    r = np.random.uniform(0, 1, n)
    theta = np.random.uniform(0, 2 * np.pi, n)
    y = np.zeros(n)
    y[:20] = 1.0

    config = BioRSPConfig(B=10, min_fg_sector=2)
    adequacy = assess_adequacy(r, theta, y, config=config)
    valid_mask = adequacy.sector_mask

    y_perm = np.random.permutation(y)
    radar_perm = compute_rsp_radar(r, theta, y_perm, config=config, frozen_mask=valid_mask)

    assert np.all(~np.isnan(radar_perm.rsp[valid_mask]))
    assert np.all(np.isnan(radar_perm.rsp[~valid_mask]))
