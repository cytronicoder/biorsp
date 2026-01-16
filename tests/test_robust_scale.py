import numpy as np
import pytest

from biorsp.api import BioRSPConfig
from biorsp.core.engine import compute_rsp_radar, sector_signed_stat


def test_pooled_vs_bg_scale_stability():
    r = np.array([5.0, 5.0, 5.01, 4.99, 5.5, 5.6])
    y = np.array([0, 0, 0, 0, 1, 1])
    idx = np.arange(6)

    res_bg = sector_signed_stat(r, y, idx, scale_mode="bg_iqr", eps=1e-8)

    res_pooled = sector_signed_stat(r, y, idx, scale_mode="pooled_iqr", eps=1e-8)

    assert res_bg["denom"] < res_pooled["denom"]
    assert abs(res_bg["stat"]) > abs(res_pooled["stat"])

    assert res_bg["sign"] == res_pooled["sign"]
    assert pytest.approx(res_bg["w1"]) == res_pooled["w1"]


def test_degeneracy_guard():
    r = np.array([5.0, 5.0, 5.0, 5.0001])
    y = np.array([0, 0, 1, 1])
    idx = np.arange(4)

    res = sector_signed_stat(r, y, idx, min_scale=0.1, scale_mode="pooled_iqr")
    assert res["status"] == "degenerate_scale"
    assert res["stat"] == 0.0


def test_pooled_mad():
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0, 0, 1, 1, 1])
    idx = np.arange(5)

    res_mad = sector_signed_stat(r, y, idx, scale_mode="pooled_mad")

    assert pytest.approx(res_mad["denom"], abs=1e-3) == 1.4826


def test_radar_uses_u_space_by_default():
    r = np.array([1.0, 2.0, 10.0, 11.0])
    theta = np.zeros(4)
    y = np.array([1, 1, 0, 0])

    # With only 4 data points, we need to lower min_total_per_sector
    config = BioRSPConfig(
        B=1, delta_deg=360, min_fg_sector=1, min_bg_sector=1, min_total_per_sector=1
    )
    assert config.scale_mode == "u_space"

    res = compute_rsp_radar(r, theta, y, config=config)

    assert res.rsp[0] == pytest.approx(0.47619, rel=1e-3)


def test_adequacy_scale_guard():
    r = np.array([5.0, 5.0, 5.0, 5.0001])
    theta = np.zeros(4)
    y = np.array([0, 0, 1, 1])

    from biorsp.core.adequacy import assess_adequacy

    config = BioRSPConfig(scale_mode="pooled_iqr", min_scale=0.1, min_fg_sector=1, min_bg_sector=1)
    report = assess_adequacy(r, theta, y, B=1, delta_deg=360, config=config)
    assert not report.sector_mask[0]
    assert not report.is_adequate
