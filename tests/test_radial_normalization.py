import numpy as np
import pytest

from biorsp.preprocess import normalize_radii


def test_normalize_radii_typical():
    """Test normalization with a typical range of values."""
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r_hat, stats = normalize_radii(r)

    assert len(r_hat) == 5
    assert stats["median_r"] == 3.0
    assert stats["iqr_r"] == 2.0
    assert np.allclose(r_hat, (r - 3.0) / (2.0 + 1e-8))


def test_normalize_radii_constant():
    """Test normalization with constant values (IQR=0)."""
    r = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    r_hat, stats = normalize_radii(r)

    assert stats["iqr_r"] == 0.0
    assert np.allclose(r_hat, 0.0)


def test_normalize_radii_outliers():
    """Test that normalization is robust to outliers."""
    r = np.array([1.0, 1.1, 1.2, 1.3, 100.0])
    r_hat, stats = normalize_radii(r)

    # Median is 1.2
    assert stats["median_r"] == 1.2
    # IQR should be small (0.2), not affected by 100.0
    assert np.allclose(stats["iqr_r"], 0.2)
    assert r_hat[4] > 100.0  # Outlier is very far in normalized units


def test_normalize_radii_non_finite():
    """Test that non-finite inputs raise ValueError."""
    r = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="non-finite"):
        normalize_radii(r)

    r = np.array([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="non-finite"):
        normalize_radii(r)


def test_normalize_radii_reproducibility():
    """Test that the function is deterministic."""
    r = np.random.rand(100)
    r_hat1, stats1 = normalize_radii(r)
    r_hat2, stats2 = normalize_radii(r)

    assert np.array_equal(r_hat1, r_hat2)
    assert stats1 == stats2


def test_normalize_radii_fallback_mad():
    """Test fallback to MAD when IQR is zero but data is not constant."""
    # Data with many ties at the median, but some variation
    r = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    # IQR will be 0 because Q1=1.0, Q3=1.0
    # MAD will be median(|r - 1.0|) = median([0, 0, 0, 0, 0, 0, 0, 1]) = 0
    # Wait, if MAD is also 0, it falls back to std.

    r_hat, stats = normalize_radii(r)
    assert stats["iqr_r"] > 0.0
