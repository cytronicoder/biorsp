import numpy as np

from biorsp.geometry import compute_vantage, geometric_median, wrapped_circular_distance


def test_geometric_median_simple():
    # Square: (0,0), (1,0), (0,1), (1,1) -> median approx (0.5, 0.5)
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    v, _, _ = geometric_median(points)
    assert np.allclose(v, [0.5, 0.5], atol=1e-2)


def test_angular_distance():
    # 0 and pi -> pi
    d = wrapped_circular_distance(np.array([0.0]), np.pi)
    assert np.isclose(d[0], np.pi)

    # 0.1 and 2pi + 0.2 -> 0.1
    d = wrapped_circular_distance(np.array([0.1]), 2 * np.pi + 0.2)
    assert np.isclose(d[0], 0.1)

    # near -pi and pi should wrap to small distance
    d = wrapped_circular_distance(np.array([-np.pi + 0.05]), np.pi - 0.05)
    assert np.isclose(d[0], 0.1)


def test_vantage_snaps_to_medoid_in_low_density_region():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=[-10.0, 0.0], scale=0.2, size=(30, 2))
    cluster_b = rng.normal(loc=[10.0, 0.0], scale=0.2, size=(30, 2))
    points = np.vstack([cluster_a, cluster_b])
    v = compute_vantage(points, method="geometric_median")
    assert np.linalg.norm(v) > 5.0
    assert np.any(np.all(np.isclose(points, v, atol=1e-6), axis=1))
