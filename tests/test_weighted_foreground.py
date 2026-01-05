import numpy as np
import pytest

from biorsp.core.adequacy import assess_adequacy
from biorsp.core.engine import compute_rsp_radar
from biorsp.preprocess.foreground import define_foreground_weights
from biorsp.utils.helpers import weighted_wasserstein_1d


def test_weighted_wasserstein_1d_correctness():
    # Hand-computed example
    # A: values [1, 2], weights [0.5, 0.5] -> CDF: 0.5 at [1, 2), 1.0 at [2, inf)
    # B: values [1.5, 2.5], weights [0.5, 0.5] -> CDF: 0.5 at [1.5, 2.5), 1.0 at [2.5, inf)
    # |CDF_A - CDF_B|:
    # [1, 1.5): |0.5 - 0| = 0.5. Width 0.5. Area = 0.25
    # [1.5, 2): |0.5 - 0.5| = 0. Width 0.5. Area = 0
    # [2, 2.5): |1.0 - 0.5| = 0.5. Width 0.5. Area = 0.25
    # Total Area = 0.5

    values_a = np.array([1.0, 2.0])
    weights_a = np.array([0.5, 0.5])
    values_b = np.array([1.5, 2.5])
    weights_b = np.array([0.5, 0.5])

    w1 = weighted_wasserstein_1d(values_a, weights_a, values_b, weights_b)
    assert pytest.approx(w1) == 0.5


def test_define_foreground_weights_logistic():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    # tau=5, scale=1, sharpness=10
    w, info = define_foreground_weights(x, tau=5.0, scale=1.0, sharpness=10.0, min_effective_fg=5.0)

    assert len(w) == len(x)
    assert w[5] == 0.5
    assert w[0] < 0.01
    assert w[10] > 0.99
    assert info["status"] == "ok"
    assert info["mode"] == "weights"
    assert info["tau"] == 5.0


def test_weighted_convergence_to_binary():
    # As sharpness -> infinity, weighted mode should match binary mode
    x = np.linspace(0, 10, 100)
    tau = 5.0

    # Binary: x >= 5
    y_binary = (x >= tau).astype(float)

    # Weighted with high sharpness
    y_weighted, _ = define_foreground_weights(x, tau=tau, scale=1.0, sharpness=100.0)

    # Check that they are very close (except exactly at tau where binary is 1 and weighted is 0.5)
    # We exclude the point exactly at tau
    mask = np.abs(x - tau) > 0.1
    np.testing.assert_allclose(y_binary[mask], y_weighted[mask], atol=1e-3)


def test_radar_weighted_mode():
    # Simple case: foreground is distal
    r = np.array([1, 1, 1, 5, 5, 5], dtype=float)
    theta = np.zeros(6)
    # Weights: first 3 are background (0), last 3 are foreground (1)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    # Binary mode
    res_bin = compute_rsp_radar(
        r, theta, y.astype(bool), B=1, delta_deg=360, min_fg_sector=1, min_bg_sector=1
    )
    # Weighted mode
    res_weight = compute_rsp_radar(
        r, theta, y, B=1, delta_deg=360, min_fg_sector=1, min_bg_sector=1
    )

    # Should be identical for binary weights
    np.testing.assert_allclose(res_bin.rsp, res_weight.rsp)
    assert res_bin.rsp[0] < 0  # Distal is negative


def test_adequacy_effective_mass():
    y = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # Total mass = 0.5
    theta = np.zeros(5)
    r = np.zeros(5)

    # min_fg_total = 1.0 should fail
    report = assess_adequacy(
        r, theta, y, n_sectors=1, delta_deg=360, min_fg_total=1.0, min_bg_sector=1
    )
    assert not report.is_adequate
    assert report.n_foreground == 0.5

    # min_fg_total = 0.4 should pass
    report = assess_adequacy(
        r,
        theta,
        y,
        n_sectors=1,
        delta_deg=360,
        min_fg_total=0.4,
        min_fg_sector=0.4,
        min_bg_sector=1,
        min_scale=0.0,
    )
    assert report.is_adequate


if __name__ == "__main__":
    pytest.main([__file__])
