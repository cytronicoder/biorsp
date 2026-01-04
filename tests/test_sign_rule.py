import numpy as np

from biorsp.core import sector_signed_stat


def test_sign_rule_proximal():
    # Case A: R_F all smaller than R_B -> sign +1 and stat > 0
    r = np.array([1.0, 2.0, 10.0, 11.0])
    y = np.array([1.0, 1.0, 0.0, 0.0])  # 1, 2 are FG; 10, 11 are BG
    idx = np.arange(4)

    res = sector_signed_stat(r, y, idx)
    assert res["sign"] == 1
    assert res["stat"] > 0
    assert res["medF"] < res["medB"]


def test_sign_rule_distal():
    # Case B: R_F all larger -> sign -1 and stat < 0
    r = np.array([1.0, 2.0, 10.0, 11.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])  # 10, 11 are FG; 1, 2 are BG
    idx = np.arange(4)

    res = sector_signed_stat(r, y, idx)
    assert res["sign"] == -1
    assert res["stat"] < 0
    assert res["medF"] > res["medB"]


def test_sign_rule_equal():
    # Case C: equal medians -> sign 0 and stat == 0
    r = np.array([1.0, 2.0, 1.0, 2.0])
    y = np.array([1.0, 0.0, 0.0, 1.0])  # FG: 1, 2; BG: 1, 2
    idx = np.arange(4)

    res = sector_signed_stat(r, y, idx, sign_tol=1e-5)
    assert res["sign"] == 0
    assert res["stat"] == 0
    assert res["medF"] == res["medB"]


def test_sign_rule_empty():
    # Case D: empty foreground/background
    r = np.array([1.0, 2.0])
    y = np.array([1.0, 1.0])
    idx = np.arange(2)

    res = sector_signed_stat(r, y, idx)
    assert res["status"] == "empty_fg_or_bg"
    assert np.isnan(res["stat"])


def test_sign_rule_weighted():
    # Weighted case
    r = np.array([1.0, 10.0])
    y = np.array([0.9, 0.1])  # Mostly proximal
    idx = np.arange(2)

    res = sector_signed_stat(r, y, idx)
    assert res["sign"] == 1
    assert res["stat"] > 0

    y_rev = np.array([0.1, 0.9])  # Mostly distal
    res_rev = sector_signed_stat(r, y_rev, idx)
    assert res_rev["sign"] == -1
    assert res_rev["stat"] < 0


def test_sign_rule_tolerance():
    # Test sign_tol
    r = np.array([1.0, 1.1])
    y = np.array([1.0, 0.0])  # medF=1.0, medB=1.1, diff=0.1
    idx = np.arange(2)

    res_no_tol = sector_signed_stat(r, y, idx, sign_tol=0.0)
    assert res_no_tol["sign"] == 1

    res_with_tol = sector_signed_stat(r, y, idx, sign_tol=0.2)
    assert res_with_tol["sign"] == 0
    assert res_with_tol["stat"] == 0
