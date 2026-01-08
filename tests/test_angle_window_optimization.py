import numpy as np

from biorsp.core.adequacy import assess_adequacy
from biorsp.core.engine import compute_rsp_radar


def naive_sector_counts(theta, y, n_sectors, delta_deg):
    centers = np.linspace(-np.pi, np.pi, n_sectors, endpoint=False)
    delta_rad = np.deg2rad(delta_deg)
    half_width = delta_rad / 2.0
    counts_fg = np.zeros(n_sectors, dtype=int)
    counts_bg = np.zeros(n_sectors, dtype=int)
    y_bool = np.asarray(y).astype(bool)
    for b, phi in enumerate(centers):
        rel_theta = theta - phi
        rel_theta = (rel_theta + np.pi) % (2 * np.pi) - np.pi
        mask_all = np.abs(rel_theta) <= half_width
        counts_fg[b] = int(np.sum(mask_all & y_bool))
        counts_bg[b] = int(np.sum(mask_all & ~y_bool))
    return counts_fg, counts_bg


def naive_rsp(r, theta, y, B, delta_deg, min_fg_sector, min_bg_sector):
    centers = np.linspace(-np.pi, np.pi, B, endpoint=False)
    delta_rad = np.deg2rad(delta_deg)
    half_width = delta_rad / 2.0
    rsp = np.full(B, np.nan)
    for b, phi in enumerate(centers):
        rel_theta = theta - phi
        rel_theta = (rel_theta + np.pi) % (2 * np.pi) - np.pi
        mask_all = np.abs(rel_theta) <= half_width
        mask_fg = mask_all & (np.asarray(y).astype(bool))
        r_fg = r[mask_fg]
        r_bg = r[mask_all & ~np.asarray(y).astype(bool)]
        if len(r_fg) < min_fg_sector or len(r_bg) < min_bg_sector:
            continue

        r_bg_sorted = np.sort(r_bg)
        n_bg = len(r_bg_sorted)
        u_f = (
            np.searchsorted(r_bg_sorted, r_fg, side="left")
            + np.searchsorted(r_bg_sorted, r_fg, side="right")
        ) / (2.0 * n_bg)
        u_b = (
            np.searchsorted(r_bg_sorted, r_bg, side="left")
            + np.searchsorted(r_bg_sorted, r_bg, side="right")
        ) / (2.0 * n_bg)

        from scipy.stats import iqr, wasserstein_distance

        w1 = wasserstein_distance(u_f, u_b)
        medF = np.median(r_fg)
        medB = np.median(r_bg)
        sign = 1.0 if medB > medF else -1.0

        global_iqr = iqr(r[~np.asarray(y).astype(bool)])
        iqr_floor = max(0.1 * global_iqr, 1e-8)

        rsp[b] = sign * w1 / (1.0 + iqr_floor)
    return rsp


def test_assess_adequacy_counts_match_naive():
    rng = np.random.default_rng(1)
    n = 500
    theta = rng.uniform(-np.pi, np.pi, size=n)
    y = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    counts_fg_naive, counts_bg_naive = naive_sector_counts(theta, y, n_sectors=180, delta_deg=20.0)
    r = np.zeros_like(theta)
    report = assess_adequacy(
        r, theta, y, n_sectors=180, delta_deg=20.0, min_fg_sector=1, min_bg_sector=1
    )
    assert np.all(counts_fg_naive == report.counts_fg)
    assert np.all(counts_bg_naive == report.counts_bg)


def test_compute_rsp_radar_matches_naive():
    rng = np.random.default_rng(2)
    n = 800
    r = rng.normal(loc=5.0, scale=1.0, size=n)
    theta = rng.uniform(-np.pi, np.pi, size=n)
    y = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    rsp_opt = compute_rsp_radar(
        r,
        theta,
        y,
        B=180,
        delta_deg=20.0,
        min_fg_sector=5,
        min_bg_sector=20,
    )
    rsp_naive = naive_rsp(r, theta, y, B=180, delta_deg=20.0, min_fg_sector=5, min_bg_sector=20)

    mask = np.isfinite(rsp_opt.rsp) | np.isfinite(rsp_naive)

    for a, b in zip(rsp_opt.rsp[mask], rsp_naive[mask]):
        if np.isfinite(a) and np.isfinite(b):
            assert np.isclose(a, b, rtol=1e-6, atol=1e-8)
        else:

            assert np.isnan(a) or np.isnan(b)
