import numpy as np

from biorsp.inference import _build_strata_indices, compute_p_value
from biorsp.radar import RadarResult


def test_permutation_resamples_missing_sectors(monkeypatch):
    call_count = {"n": 0}

    def fake_compute_rsp_radar(*args, **kwargs):
        call_count["n"] += 1
        centers = np.array([0.0, 1.0, 2.0])
        if call_count["n"] == 1:
            rsp = np.array([1.0, 2.0, np.nan])
        elif call_count["n"] == 2:
            rsp = np.array([np.nan, 2.0, np.nan])
        else:
            rsp = np.array([1.0, 2.0, np.nan])
        counts = np.array([1, 1, 1])
        return RadarResult(
            rsp=rsp,
            counts_fg=counts,
            counts_bg=counts,
            centers=centers,
            iqr_floor=np.nan,
            iqr_floor_hits=np.array([False, False, False]),
        )

    monkeypatch.setattr("biorsp.inference.compute_rsp_radar", fake_compute_rsp_radar)
    monkeypatch.setattr("biorsp.inference._estimate_anisotropy_se", lambda *args, **kwargs: 1.0)

    r = np.ones(5)
    theta = np.linspace(-np.pi, np.pi, 5, endpoint=False)
    y = np.array([True, False, True, False, True])

    p_val, nulls, obs, valid_mask, rejected = compute_p_value(
        r, theta, y, B=3, delta_deg=360.0, n_perm=1, seed=0, min_fg_sector=1, min_bg_sector=1
    )

    assert np.array_equal(valid_mask, np.array([True, True, False]))
    expected_perm = np.sqrt(np.mean(np.array([1.0, 2.0]) ** 2))
    assert np.isclose(nulls[0], expected_perm)
    assert np.isfinite(obs)
    assert np.isfinite(p_val)
    assert rejected == 1


def test_donor_stratification_keeps_labels_within_donor():
    n_cells = 6
    donor_ids = np.array(["d1", "d1", "d1", "d2", "d2", "d2"])
    umi_counts = np.array([10, 11, 12, 20, 21, 22])
    strata = _build_strata_indices(n_cells, umi_counts, umi_bins=2, donor_ids=donor_ids)
    for idx in strata:
        assert len(set(donor_ids[idx])) == 1


def test_studentized_statistic_uses_se(monkeypatch):
    y_obs = np.array([True, False, True, False])

    def fake_compute_rsp_radar(*args, **kwargs):
        centers = np.array([0.0, 1.0])
        rsp = np.array([2.0, 2.0])
        counts = np.array([2, 2])
        return RadarResult(
            rsp=rsp,
            counts_fg=counts,
            counts_bg=counts,
            centers=centers,
            iqr_floor=np.nan,
            iqr_floor_hits=np.array([False, False]),
        )

    def fake_se(r, theta, y, *args, **kwargs):
        return 2.0 if np.array_equal(y, y_obs) else 1.0

    monkeypatch.setattr("biorsp.inference.compute_rsp_radar", fake_compute_rsp_radar)
    monkeypatch.setattr("biorsp.inference._estimate_anisotropy_se", fake_se)

    r = np.ones(4)
    theta = np.linspace(-np.pi, np.pi, 4, endpoint=False)

    p_val, nulls, obs, _, _ = compute_p_value(
        r, theta, y_obs, B=2, delta_deg=360.0, n_perm=1, seed=0, min_fg_sector=1, min_bg_sector=1
    )

    assert np.isclose(obs, 1.0)
    assert np.isclose(nulls[0], 2.0)
    assert np.isfinite(p_val)
