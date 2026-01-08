import numpy as np
import pytest

from biorsp.utils.stats import compute_signed_summaries


def test_signed_all_positive():
    """All-positive constant => polarity ~ +1"""
    M = 10
    R = np.ones(M)
    res = compute_signed_summaries(R)
    assert res["status"] == "ok"
    assert res["R_mean"] == pytest.approx(1.0)
    assert res["polarity"] == pytest.approx(1.0)
    assert res["frac_pos"] == 1.0
    assert res["frac_neg"] == 0.0
    assert res["A_signed"] == pytest.approx(1.0)


def test_signed_all_negative():
    """All-negative constant profile => R_mean < 0, polarity ~ -1"""
    M = 10
    R = -np.ones(M)
    res = compute_signed_summaries(R)
    assert res["status"] == "ok"
    assert res["R_mean"] == pytest.approx(-1.0)
    assert res["polarity"] == pytest.approx(-1.0)
    assert res["frac_pos"] == 0.0
    assert res["frac_neg"] == 1.0
    assert res["A_signed"] == pytest.approx(-1.0)


def test_signed_symmetric_mixed():
    """Symmetric mixed (+1 half, -1 half) => R_mean ~ 0, polarity ~ 0"""
    R = np.array([1.0] * 5 + [-1.0] * 5)
    res = compute_signed_summaries(R)
    assert res["status"] == "ok"
    assert res["R_mean"] == pytest.approx(0.0)
    assert res["polarity"] == pytest.approx(0.0)
    assert res["frac_pos"] == 0.5
    assert res["frac_neg"] == 0.5
    assert res["A_signed"] == pytest.approx(0.0)


def test_signed_localized_wedge():
    """Localized wedge (one positive spike, rest 0) => polarity ~ +1 but R_mean small"""
    M = 10
    R = np.zeros(M)
    R[0] = 1.0
    res = compute_signed_summaries(R)
    assert res["status"] == "ok"
    assert res["R_mean"] == pytest.approx(0.1)
    assert res["polarity"] == pytest.approx(1.0)
    assert res["frac_pos"] == 0.1
    assert res["frac_neg"] == 0.0


def test_signed_no_signal():
    """No-signal (all zeros) => polarity = 0, status no_signal"""
    M = 10
    R = np.zeros(M)
    res = compute_signed_summaries(R)
    assert res["status"] == "no_signal"
    assert res["polarity"] == 0.0
    assert res["R_mean"] == 0.0


def test_signed_insufficient_sectors():
    """M < 2 => status insufficient_sectors but still compute mean"""
    R = np.array([1.0])
    res = compute_signed_summaries(R)
    assert res["status"] == "insufficient_sectors"
    assert res["R_mean"] == 1.0
    assert res["polarity"] == pytest.approx(1.0)


def test_signed_missing_sectors():
    """Missing sectors: valid_mask excludes half"""
    M_total = 20
    M_valid = 10
    R = np.ones(M_total)
    R[M_valid:] = -1.0
    valid_mask = np.zeros(M_total, dtype=bool)
    valid_mask[:M_valid] = True

    res = compute_signed_summaries(R, valid_mask=valid_mask)
    assert res["status"] == "ok"
    assert res["M_valid"] == M_valid
    assert res["R_mean"] == pytest.approx(1.0)
    assert res["polarity"] == pytest.approx(1.0)
