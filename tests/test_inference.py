import numpy as np

from biorsp.inference import compute_p_value
from biorsp.radar import RadarResult


def test_permutation_missing_sectors_treated_as_zero(monkeypatch):
    call_count = {"n": 0}

    def fake_compute_rsp_radar(*args, **kwargs):
        call_count["n"] += 1
        centers = np.array([0.0, 1.0, 2.0])
        if call_count["n"] == 1:
            rsp = np.array([1.0, 2.0, np.nan])
        else:
            rsp = np.array([np.nan, 2.0, np.nan])
        counts = np.array([1, 1, 1])
        return RadarResult(
            rsp=rsp,
            counts_fg=counts,
            counts_bg=counts,
            centers=centers,
            iqr_floor=1.0,
            iqr_floor_hits=np.array([False, False, False]),
        )

    monkeypatch.setattr("biorsp.inference.compute_rsp_radar", fake_compute_rsp_radar)

    r = np.ones(5)
    theta = np.linspace(-np.pi, np.pi, 5, endpoint=False)
    y = np.array([True, False, True, False, True])

    p_val, nulls, obs, valid_mask = compute_p_value(
        r, theta, y, B=3, delta_deg=360.0, n_perm=1, seed=0, min_fg_sector=1, min_bg_sector=1
    )

    assert np.array_equal(valid_mask, np.array([True, True, False]))
    expected_perm = np.sqrt(np.mean(np.array([0.0, 2.0]) ** 2))
    assert np.isclose(nulls[0], expected_perm)
    assert np.isfinite(obs)
    assert np.isfinite(p_val)
