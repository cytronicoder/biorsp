from __future__ import annotations

import numpy as np

from biorsp.rsp import compute_theta_from_embedding


def test_theta_orientation_cardinal_points() -> None:
    center = np.array([0.0, 0.0], dtype=float)
    points = np.array(
        [
            [1.0, 0.0],  # East
            [0.0, 1.0],  # North
            [-1.0, 0.0],  # West
            [0.0, -1.0],  # South
        ],
        dtype=float,
    )
    theta = compute_theta_from_embedding(points, center, debug=True)
    expected = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0], dtype=float)
    np.testing.assert_allclose(theta, expected, atol=1e-8, rtol=0.0)
