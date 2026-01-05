import numpy as np
import pytest
from biorsp.stats import compute_localization

def test_localization_uniform():
    """Uniform profile: R = constant over M sectors => L ≈ 0"""
    M = 10
    R = np.ones(M)
    L, info = compute_localization(R)
    assert info["status"] == "ok"
    assert L == pytest.approx(0.0, abs=1e-7)
    assert info["M"] == M
    assert info["sum_abs"] == M

def test_localization_single_peak():
    """Single-peak profile: one sector has value 1, others 0 => L ≈ 1"""
    M = 10
    R = np.zeros(M)
    R[0] = 1.0
    L, info = compute_localization(R)
    assert info["status"] == "ok"
    assert L == pytest.approx(1.0, abs=1e-7)
    assert info["M"] == M
    assert info["sum_abs"] == 1.0

def test_localization_two_peaks():
    """Two-peak profile: two equal peaks => L intermediate, higher than uniform"""
    M = 10
    R = np.zeros(M)
    R[0] = 1.0
    R[5] = 1.0
    L, info = compute_localization(R)
    assert info["status"] == "ok"
    # H = - (0.5 * log(0.5) + 0.5 * log(0.5)) = log(2)
    # L = 1 - log(2) / log(10)
    expected_L = 1.0 - np.log(2) / np.log(10)
    assert L == pytest.approx(expected_L, abs=1e-7)

def test_localization_zero_profile():
    """Zero profile: all zeros => L NaN or 0 with status 'no_signal'"""
    M = 10
    R = np.zeros(M)
    L, info = compute_localization(R)
    assert info["status"] == "no_signal"
    assert L == 0.0

def test_localization_missing_sectors():
    """Missing sectors: valid_mask excludes half; metric uses only valid ones and normalizes by log(M_valid)"""
    M_total = 20
    M_valid = 10
    R = np.ones(M_total)
    valid_mask = np.zeros(M_total, dtype=bool)
    valid_mask[:M_valid] = True
    
    L, info = compute_localization(R, valid_mask=valid_mask)
    assert info["status"] == "ok"
    assert info["M"] == M_valid
    assert L == pytest.approx(0.0, abs=1e-7)

def test_localization_insufficient_sectors():
    """M <= 1 => L NaN with status 'insufficient_sectors'"""
    R = np.array([1.0])
    L, info = compute_localization(R)
    assert info["status"] == "insufficient_sectors"
    assert np.isnan(L)

def test_localization_gini():
    """Test Gini method"""
    M = 10
    R = np.zeros(M)
    R[0] = 1.0
    L, info = compute_localization(R, method="gini")
    # For single peak: G = (M-1)/M
    expected_G = (M - 1) / M
    assert L == pytest.approx(expected_G, abs=1e-7)
    assert info["gini"] == pytest.approx(expected_G, abs=1e-7)

def test_localization_robustness_to_sign():
    """Ensure localization is independent of sign."""
    M = 10
    R1 = np.zeros(M)
    R1[0] = 1.0
    R2 = np.zeros(M)
    R2[0] = -1.0
    
    L1, _ = compute_localization(R1)
    L2, _ = compute_localization(R2)
    assert L1 == L2
