import numpy as np

from biorsp.core.geometry import compute_theta


def test_theta_umap_convention_cardinal_points() -> None:
    center = np.array([0.0, 0.0], dtype=float)
    xy = np.array(
        [
            [1.0, 0.0],   # east
            [0.0, 1.0],   # north
            [-1.0, 0.0],  # west
            [0.0, -1.0],  # south
        ],
        dtype=float,
    )
    theta = compute_theta(xy, center)
    expected = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0], dtype=float)
    np.testing.assert_allclose(theta, expected, atol=1e-12)
