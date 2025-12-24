import numpy as np

from biorsp.radar import compute_rsp_radar


def test_radar_simple():
    # 100 points
    theta = np.linspace(-np.pi, np.pi, 100, endpoint=False)
    y = np.zeros(100, dtype=bool)
    y[:50] = True  # First half foreground

    theta_fg = theta[y]

    # Should run without error
    result = compute_rsp_radar(theta_fg, B=360, delta_deg=20.0)
    assert len(result.rsp) == 360
    assert len(result.counts) == 360
