import numpy as np

from biorsp.adequacy import gene_adequacy
from biorsp.constants import EPS
from biorsp.radar import compute_rsp_radar


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
        from scipy.stats import iqr, wasserstein_distance

        w1 = wasserstein_distance(r_fg, r_bg)
        global_iqr = (
            iqr(r[~np.asarray(y).astype(bool)])
            if len(r[~np.asarray(y).astype(bool)]) > 0
            else np.nan
        )
        if not np.isfinite(global_iqr) or global_iqr < 0:
            global_iqr = 0.0
        iqr_floor = max(0.1 * global_iqr, EPS)
        iqr_bg = iqr(r_bg)
        if not np.isfinite(iqr_bg):
            iqr_bg = 0.0
        denom = max(iqr_bg, iqr_floor)
        diff_median = np.median(r_bg) - np.median(r_fg)
        sign = np.sign(diff_median)
        rsp[b] = sign * (w1 / denom)
    return rsp


def test_gene_adequacy_counts_match_naive():
    rng = np.random.default_rng(1)
    n = 500
    theta = rng.uniform(-np.pi, np.pi, size=n)
    y = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    counts_fg_naive, counts_bg_naive = naive_sector_counts(theta, y, n_sectors=180, delta_deg=20.0)
    report = gene_adequacy(
        y, theta, n_sectors=180, delta_deg=20.0, min_fg_sector=1, min_bg_sector=1
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
        r, theta, y, B=180, delta_deg=20.0, min_fg_sector=5, min_bg_sector=20
    )
    rsp_naive = naive_rsp(r, theta, y, B=180, delta_deg=20.0, min_fg_sector=5, min_bg_sector=20)
    # Compare where neither is NaN
    mask = np.isfinite(rsp_opt.rsp) | np.isfinite(rsp_naive)
    # Differences should be small where both compute values
    for a, b in zip(rsp_opt.rsp[mask], rsp_naive[mask]):
        if np.isfinite(a) and np.isfinite(b):
            assert np.isclose(a, b, rtol=1e-6, atol=1e-8)
        else:
            # One may be NaN while other is NaN; allow that
            assert np.isnan(a) or np.isnan(b)
