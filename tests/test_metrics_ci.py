import numpy as np
import pytest

from analysis.benchmarks.simlib.metrics_ci import binomial_wilson_ci, bootstrap_ci


def test_binomial_wilson_ci_bounds_and_order():
    low, high = binomial_wilson_ci(5, 10, alpha=0.05)
    assert 0.0 <= low <= high <= 1.0


@pytest.mark.parametrize("k,n", [(0, 0), (-1, 10), (11, 10)])
def test_binomial_wilson_ci_invalid_inputs(k, n):
    with pytest.raises(ValueError):
        binomial_wilson_ci(k, n)


def test_bootstrap_ci_reproducible_mean():
    arr = np.array([1, 2, 3, 4, 5], dtype=float)
    low, high = bootstrap_ci(arr, func=np.mean, n_boot=200, alpha=0.10, seed=123)
    point_est = arr.mean()
    assert low <= point_est <= high


def test_bootstrap_ci_raises_on_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([], seed=0)
