"""Tests for utils module."""

import numpy as np
import pytest
from src import utils


class TestUtils:
    """Test utility functions."""

    def test_compute_distances(self):
        """Test distance computation."""
        coords = np.array([[0, 0], [1, 0], [0, 1]])
        distances = utils.compute_distances(coords)

        assert distances.shape == (3, 3)
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(distances), 0)
        # Should be symmetric
        np.testing.assert_array_almost_equal(distances, distances.T)

    def test_compute_angles(self):
        """Test angle computation."""
        coords = np.array([[0, 0], [1, 0], [0, 1], [-1, 0]])
        angles = utils.compute_angles(coords)

        assert angles.shape == (4, 4)
        # Angles should be in [0, 2*pi)
        assert np.all(angles >= 0)
        assert np.all(angles < 2 * np.pi)
