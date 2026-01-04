import numpy as np

from biorsp.core import compute_rsp_radar
from biorsp.stats import compute_p_value
from biorsp.typing import BioRSPConfig


def test_inference_result_contains_seeds():
    """Verify that InferenceResult contains the seeds used for permutations."""
    rng = np.random.default_rng(42)
    n = 100
    r = rng.uniform(0, 1, n)
    theta = rng.uniform(-np.pi, np.pi, n)
    y = rng.choice([0, 1], n)

    config = BioRSPConfig(B=1, delta_deg=360.0, min_fg_sector=1, min_bg_sector=1)
    res = compute_p_value(r, theta, y, n_perm=10, seed=123, config=config)
    assert res.seeds is not None
    assert len(res.seeds) == 10
    assert res.seeds.dtype == np.int64


def test_permutation_determinism():
    """Verify that same seed leads to same null statistics and seeds."""
    rng = np.random.default_rng(42)
    n = 100
    r = rng.uniform(0, 1, n)
    theta = rng.uniform(-np.pi, np.pi, n)
    y = rng.choice([0, 1], n)

    config = BioRSPConfig(B=1, delta_deg=360.0, min_fg_sector=1, min_bg_sector=1)
    res1 = compute_p_value(r, theta, y, n_perm=10, seed=42, config=config)
    res2 = compute_p_value(r, theta, y, n_perm=10, seed=42, config=config)

    np.testing.assert_array_equal(res1.null_stats, res2.null_stats)
    np.testing.assert_array_equal(res1.seeds, res2.seeds)
    assert res1.p_value == res2.p_value


def test_finite_permutation_correction():
    """Verify p = (count_ge + 1) / (n_perm + 1)."""
    # Mock observed and nulls
    # If obs = 0.5 and nulls = [0.4, 0.5, 0.6], count_ge = 2.
    # p = (2 + 1) / (3 + 1) = 0.75.

    # We can't easily mock the internal loop without monkeypatching,
    # but we can check the result of a known small case.
    n = 50
    r = np.linspace(0, 1, n)
    theta = np.linspace(-np.pi, np.pi, n)
    y = np.zeros(n)
    y[n // 2 :] = 1  # Some signal

    n_perm = 9
    config = BioRSPConfig(B=1, delta_deg=360.0, min_fg_sector=1, min_bg_sector=1)
    res = compute_p_value(r, theta, y, n_perm=n_perm, seed=42, config=config)

    # p-value must be a multiple of 1/(n_perm + 1) = 0.1
    assert np.isclose((res.p_value * (n_perm + 1)) % 1, 0) or np.isclose(
        (res.p_value * (n_perm + 1)) % 1, 1
    )


def test_weighted_quantile_binary_match():
    """Verify that weighted_quantile matches np.quantile for binary weights."""
    from biorsp.utils import weighted_quantile

    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([1, 0, 1, 0, 1])  # Use 1, 3, 5

    q = 0.5
    wq = weighted_quantile(values, weights, q)
    nq = np.quantile(np.array([1.0, 3.0, 5.0]), q)

    assert wq == nq

    q = 0.25
    wq = weighted_quantile(values, weights, q)
    nq = np.quantile(np.array([1.0, 3.0, 5.0]), q)
    assert wq == nq


def test_compute_rsp_radar_kwargs_compatibility():
    """Verify that compute_rsp_radar accepts B and delta_deg as kwargs."""
    rng = np.random.default_rng(42)
    n = 100
    r = rng.uniform(0, 1, n)
    theta = rng.uniform(-np.pi, np.pi, n)
    y = rng.choice([0, 1], n)

    # This should not raise TypeError
    res = compute_rsp_radar(r, theta, y, B=10, delta_deg=45.0)
    assert len(res.rsp) == 10
