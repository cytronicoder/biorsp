import numpy as np

from biorsp.core.engine import compute_rsp_radar
from biorsp.core.typing import BioRSPConfig


def test_radar_simple():
    theta = np.linspace(-np.pi, np.pi, 100, endpoint=False)
    y = np.zeros(100, dtype=bool)
    y[:50] = True

    r = np.ones_like(theta)

    # Should run without error using full API (r, theta, y)
    config = BioRSPConfig(B=360, delta_deg=20.0)
    result = compute_rsp_radar(r, theta, y, config=config)
    assert len(result.rsp) == 360
    assert len(result.counts_fg) == 360
    assert len(result.counts_bg) == 360
