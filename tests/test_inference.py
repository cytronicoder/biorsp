import numpy as np

from biorsp.inference import compute_p_value
from biorsp.radar import RadarResult


def test_compute_p_value_resamples_until_enough_valid(monkeypatch):
    # Construct simple geometry
    n = 20
    r = np.linspace(1, 10, n)
    theta = np.linspace(-np.pi, np.pi, n)
    # half of cells foreground randomly
    rng = np.random.default_rng(0)
    y = rng.choice([False, True], size=n)

    # Make observed radar finite by letting compute_rsp_radar be normal for observed
    # but replace _compute_permutation_stat to be deterministic: even seeds => NaN, odd => 0.5
    def fake_perm(
        r_arg, theta_arg, y_arg, strata_indices, B, delta_deg, min_fg_sector, min_bg_sector, seed, valid_mask
    ):
        if seed % 2 == 0:
            return np.nan
        return 0.5

    monkeypatch.setattr("biorsp.inference._compute_permutation_stat", fake_perm)

    p_val, nulls, obs = compute_p_value(
        r, theta, y, B=4, delta_deg=180.0, n_perm=3, seed=1, min_fg_sector=1, min_bg_sector=1
    )

    # We should collect exactly 3 valid nulls (not NaN)
    assert np.sum(np.isfinite(nulls)) == 3
    # p_val should be finite
    assert np.isfinite(p_val)
    assert np.isfinite(obs)


def test_compute_p_value_returns_nan_if_all_perms_invalid(monkeypatch):
    n = 10
    r = np.linspace(1, 5, n)
    theta = np.linspace(-np.pi, np.pi, n)
    y = np.zeros(n, dtype=bool)

    def fake_perm_all_nan(*args, **kwargs):
        return np.nan

    monkeypatch.setattr("biorsp.inference._compute_permutation_stat", fake_perm_all_nan)

    p_val, nulls, obs = compute_p_value(r, theta, y, B=36, delta_deg=20.0, n_perm=3, seed=2)

    assert np.isnan(p_val)
    assert np.sum(np.isfinite(nulls)) == 0
    assert np.isfinite(obs) or np.isnan(obs)  # observed may be NaN depending on adequacy
