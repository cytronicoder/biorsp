"""Tests for preprocessing module."""

import numpy as np
import pytest
from src import preprocessing


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_standardize_rank(self):
        """Test rank-based standardization."""
        data = np.array([1, 2, 3, 4, 5])
        result = preprocessing.standardize(data, method="rank")
        assert len(result) == len(data)
        # Rank standardization should give values roughly in [-1, 1]
        assert np.abs(result).max() <= 3.0  # Allow some tolerance

    def test_standardize_zscore(self):
        """Test z-score standardization."""
        data = np.array([1, 2, 3, 4, 5])
        result = preprocessing.standardize(data, method="zscore")
        # Z-score should have mean ~0 and std ~1
        assert np.abs(np.mean(result)) < 1e-10
        assert np.abs(np.std(result) - 1.0) < 1e-10

    def test_standardize_none(self):
        """Test no standardization."""
        data = np.array([1, 2, 3, 4, 5])
        result = preprocessing.standardize(data, method="none")
        np.testing.assert_array_equal(result, data)
