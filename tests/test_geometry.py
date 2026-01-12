import numpy as np

from biorsp.core.geometry import geometric_median, wrapped_circular_distance


def test_geometric_median_simple():
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    v, _, _ = geometric_median(points)
    assert np.allclose(v, [0.5, 0.5], atol=1e-2)


def test_angular_distance():
    d = wrapped_circular_distance(np.array([0.0]), np.pi)
    assert np.isclose(d[0], np.pi)

    d = wrapped_circular_distance(np.array([0.1]), 2 * np.pi + 0.2)
    assert np.isclose(d[0], 0.1)

    d = wrapped_circular_distance(np.array([-np.pi + 0.05]), np.pi - 0.05)
    assert np.isclose(d[0], 0.1)
