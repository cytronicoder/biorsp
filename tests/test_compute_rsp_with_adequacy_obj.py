import numpy as np

from biorsp.core import assess_adequacy, compute_rsp_radar
from biorsp.typing import BioRSPConfig


def test_compute_rsp_accepts_adequacy_object():
    rng = np.random.default_rng(7)
    n = 400
    r = rng.normal(loc=5.0, scale=1.0, size=n)
    theta = rng.uniform(-np.pi, np.pi, size=n)
    y = rng.choice([0, 1], size=n, p=[0.8, 0.2])

    config = BioRSPConfig(B=180, delta_deg=20.0, min_fg_sector=2, min_bg_sector=10)

    adequacy = assess_adequacy(y, theta, config=config)

    radar_direct = compute_rsp_radar(r, theta, y, config=config)
    radar_from_adequacy = compute_rsp_radar(r, theta, y, config=config, adequacy=adequacy)

    assert np.all(radar_direct.counts_fg == radar_from_adequacy.counts_fg)
    assert np.all(radar_direct.counts_bg == radar_from_adequacy.counts_bg)
    mask = np.isfinite(radar_direct.rsp) | np.isfinite(radar_from_adequacy.rsp)
    for a, b in zip(radar_direct.rsp[mask], radar_from_adequacy.rsp[mask]):
        if np.isfinite(a) and np.isfinite(b):
            assert np.isclose(a, b, rtol=1e-8, atol=1e-12)
        else:
            assert np.isnan(a) or np.isnan(b)
