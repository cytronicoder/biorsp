import numpy as np

from biorsp.smoothing import circular_moving_average


def test_circular_moving_average_identity_w1():
    x = np.array([0.2, -0.5, 1.1, 0.0, 3.0], dtype=float)
    y = circular_moving_average(x, 1)
    assert np.array_equal(y, x)
    assert y is not x


def test_circular_moving_average_preserves_constant_signal():
    x = np.full(17, 2.5, dtype=float)
    for w in [1, 3, 5, 7, 17]:
        y = circular_moving_average(x, w)
        assert np.allclose(y, x)


def test_circular_moving_average_roll_equivariance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=31)
    for w in [1, 3, 5, 9]:
        y = circular_moving_average(x, w)
        for shift in [1, 4, 11]:
            rolled = np.roll(x, shift)
            y_rolled = circular_moving_average(rolled, w)
            assert np.allclose(y_rolled, np.roll(y, shift), atol=1e-12, rtol=0.0)
