import numpy as np

from biorsp.radar import compute_rsp_radar


def test_radar_simple():
    # 100 points
    theta = np.linspace(-np.pi, np.pi, 100, endpoint=False)
    y = np.zeros(100, dtype=bool)
    y[:50] = True  # First half foreground

    # simple radial distances (unit radius)
    r = np.ones_like(theta)

    # Should run without error using full API (r, theta, y)
    result = compute_rsp_radar(r, theta, y, B=360, delta_deg=20.0)
    assert len(result.rsp) == 360
    assert len(result.counts_fg) == 360
    assert len(result.counts_bg) == 360
