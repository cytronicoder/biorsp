import numpy as np

from biorsp.geometry import compute_angles, compute_vantage


def test_angles_in_range():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 2))
    v = compute_vantage(pts)
    ang = compute_angles(pts, v)
    assert ang.min() >= -1e-12
    assert ang.max() < 2 * np.pi + 1e-12
