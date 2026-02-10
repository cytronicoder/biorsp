import numpy as np
import pytest
import scipy.sparse as sp

from biorsp.moran import morans_i


def test_moran_finite():
    W = sp.csr_matrix(
        np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=float,
        )
    )
    x = np.array([1.0, 2.0, 3.0])
    moran_i = morans_i(x, W, row_standardize=True)
    assert np.isfinite(moran_i)


def test_moran_zero_variance_raises():
    W = sp.csr_matrix(np.eye(3))
    x = np.array([5.0, 5.0, 5.0])
    with pytest.raises(ValueError):
        morans_i(x, W, row_standardize=True)


def test_moran_random_near_zero():
    rng = np.random.default_rng(0)
    W = sp.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=float,
        )
    )
    x = rng.normal(size=4)
    moran_i = morans_i(x, W, row_standardize=True)
    assert abs(moran_i) < 1.0
