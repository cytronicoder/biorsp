import numpy as np

from biorsp.adequacy import gene_adequacy
from biorsp.radar import compute_rsp_radar


def test_compute_rsp_with_precomputed_indices_matches():
    rng = np.random.default_rng(3)
    n = 600
    r = rng.normal(loc=5.0, scale=1.0, size=n)
    theta = rng.uniform(-np.pi, np.pi, size=n)
    y = rng.choice([0, 1], size=n, p=[0.85, 0.15])

    adequacy = gene_adequacy(y, theta, n_sectors=180, delta_deg=20.0, min_fg_sector=3, min_bg_sector=20)

    # Compute radar without reuse
    radar_a = compute_rsp_radar(r, theta, y, B=180, delta_deg=20.0, min_fg_sector=3, min_bg_sector=20)

    # Compute radar with reuse of sector indices
    radar_b = compute_rsp_radar(
        r, theta, y, B=180, delta_deg=20.0, min_fg_sector=3, min_bg_sector=20, sector_indices=adequacy.sector_indices
    )

    # Counts should match
    assert np.all(radar_a.counts_fg == radar_b.counts_fg)
    assert np.all(radar_a.counts_bg == radar_b.counts_bg)

    # RSP values should match where either is finite
    mask = np.isfinite(radar_a.rsp) | np.isfinite(radar_b.rsp)
    for a, b in zip(radar_a.rsp[mask], radar_b.rsp[mask]):
        if np.isfinite(a) and np.isfinite(b):
            assert np.isclose(a, b, rtol=1e-8, atol=1e-12)
        else:
            assert np.isnan(a) or np.isnan(b)
