"""Tests for radar_scan module."""

import numpy as np
import pytest
from src.radar_scan import ScanParams, RadarScanner


class TestScanParams:
    """Test ScanParams configuration."""

    def test_default_params(self):
        """Test that default parameters are created correctly."""
        params = ScanParams()
        assert params.B == 180
        assert params.random_state == 0

    def test_custom_params(self):
        """Test custom parameters."""
        params = ScanParams(B=90, random_state=42)
        assert params.B == 90
        assert params.random_state == 42


class TestRadarScanner:
    """Test RadarScanner functionality."""

    @pytest.fixture
    def simple_coords(self):
        """Create simple 2D coordinates for testing."""
        np.random.seed(42)
        return np.random.randn(100, 2)

    @pytest.fixture
    def simple_scanner(self):
        """Create a simple scanner for testing."""
        params = ScanParams(
            B=36,  # Fewer angles for faster testing
            widths_deg=(30, 60),
            n_bands=2,
            R=10,  # Fewer permutations for faster testing
            random_state=42,
        )
        return RadarScanner(params)

    def test_scanner_initialization(self, simple_scanner):
        """Test that scanner initializes correctly."""
        assert simple_scanner.params.B == 36
        assert simple_scanner.params.R == 10

    def test_fit_basic(self, simple_scanner, simple_coords):
        """Test basic fit without covariates or batches."""
        scanner = simple_scanner.fit(simple_coords)
        assert scanner.coords is not None
        assert scanner.coords.shape == simple_coords.shape

    def test_fit_with_covariates(self, simple_scanner, simple_coords):
        """Test fit with covariates."""
        covariates = np.random.randn(100, 2)
        scanner = simple_scanner.fit(simple_coords, covariates=covariates)
        assert scanner.coords is not None

    def test_scan_feature_binary(self, simple_scanner, simple_coords):
        """Test scanning a binary feature."""
        scanner = simple_scanner.fit(simple_coords)
        feature = np.random.randint(0, 2, size=100).astype(float)
        result = scanner.scan_feature(feature, name="test_feature")

        assert result is not None
        assert hasattr(result, "name") or isinstance(result, dict)

    def test_scan_feature_continuous(self, simple_scanner, simple_coords):
        """Test scanning a continuous feature."""
        scanner = simple_scanner.fit(simple_coords)
        feature = np.random.randn(100)
        result = scanner.scan_feature(feature, name="test_continuous")

        assert result is not None

    def test_invalid_feature_length(self, simple_scanner, simple_coords):
        """Test that invalid feature length raises error."""
        scanner = simple_scanner.fit(simple_coords)
        feature = np.random.randn(50)  # Wrong length

        with pytest.raises((ValueError, AssertionError)):
            scanner.scan_feature(feature, name="invalid")
