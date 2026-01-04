import numpy as np
import pytest

from biorsp.config import BioRSPConfig
from biorsp.core import compute_rsp_radar, sector_signed_stat


def test_pooled_vs_bg_scale_stability():
    # Construct a sector where background radii are nearly constant (tiny IQR)
    # but foreground differs slightly.
    # R_B = [5.0, 5.0, 5.01, 4.99] -> IQR is very small
    # R_F = [5.5, 5.6] -> Shifted but small magnitude

    r = np.array([5.0, 5.0, 5.01, 4.99, 5.5, 5.6])
    y = np.array([0, 0, 0, 0, 1, 1])
    idx = np.arange(6)

    # Under bg_iqr, stat should be large because denom is tiny
    res_bg = sector_signed_stat(r, y, idx, scale_mode="bg_iqr", eps=1e-8)

    # Under pooled_iqr, stat should be smaller because denom includes FG spread
    res_pooled = sector_signed_stat(r, y, idx, scale_mode="pooled_iqr", eps=1e-8)

    assert res_bg["denom"] < res_pooled["denom"]
    assert abs(res_bg["stat"]) > abs(res_pooled["stat"])
    # Sign and w1 should be the same
    assert res_bg["sign"] == res_pooled["sign"]
    assert pytest.approx(res_bg["w1"]) == res_pooled["w1"]


def test_degeneracy_guard():
    # Both fg and bg are tightly concentrated
    r = np.array([5.0, 5.0, 5.0, 5.0001])
    y = np.array([0, 0, 1, 1])
    idx = np.arange(4)

    # min_scale = 0.1 should trigger degeneracy
    res = sector_signed_stat(r, y, idx, min_scale=0.1)
    assert res["status"] == "degenerate_scale"
    assert res["stat"] == 0.0


def test_pooled_mad():
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0, 0, 1, 1, 1])
    idx = np.arange(5)

    res_mad = sector_signed_stat(r, y, idx, scale_mode="pooled_mad")
    # Unweighted MAD of [1,2,3,4,5] is 1.4826 * median(|1-3|, |2-3|, |3-3|, |4-3|, |5-3|)
    # = 1.4826 * median(2, 1, 0, 1, 2) = 1.4826 * 1 = 1.4826
    assert pytest.approx(res_mad["denom"]) == 1.4826


def test_radar_uses_pooled_iqr_by_default():
    r = np.array([1.0, 2.0, 10.0, 11.0])
    theta = np.zeros(4)
    y = np.array([1, 1, 0, 0])

    config = BioRSPConfig(B=1, delta_deg=360, min_fg_sector=1, min_bg_sector=1)
    assert config.scale_mode == "pooled_iqr"

    res = compute_rsp_radar(r, theta, y, config=config)
    # pooled IQR of [1, 2, 10, 11] is 8.5
    # global_iqr of background [10, 11] is 0.5 (using weighted_quantile_sorted)
    # iqr_floor = 0.1 * 0.5 = 0.05
    # w1 = 9.0
    # stat = 9.0 / (8.5 + 0.05) = 9.0 / 8.55 approx 1.05263
    assert res.rsp[0] == pytest.approx(9.0 / 8.55, rel=1e-4)


def test_adequacy_scale_guard():
    r = np.array([5.0, 5.0, 5.0, 5.0001])
    theta = np.zeros(4)
    y = np.array([0, 0, 1, 1])

    from biorsp.adequacy import assess_adequacy

    # min_scale = 0.1 should make the sector inadequate
    report = assess_adequacy(
        r, theta, y, B=1, delta_deg=360, min_scale=0.1, min_fg_sector=1, min_bg_sector=1
    )
    assert not report.sector_mask[0]
    assert not report.is_adequate
