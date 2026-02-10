import numpy as np

from biorsp.evaluation import circular_sd


def test_circular_sd_zero_for_constant():
    ang = np.zeros(10)
    sd = circular_sd(ang)
    assert sd >= 0.0
    assert sd < 1e-6


def test_circular_sd_nonzero_for_spread():
    ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    sd = circular_sd(ang)
    assert sd > 0.1
