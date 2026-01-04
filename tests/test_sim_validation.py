import numpy as np

from biorsp.core import compute_rsp_radar
from biorsp.inference import compute_p_value
from biorsp.simulations.generator import simulate_dataset
from biorsp.summaries import compute_scalar_summaries


def test_monotonic_power_with_N():
    """Verify that detection power increases with N for a fixed signal."""
    q = 0.1
    alpha = 0.05
    n_reps = 10

    powers = []
    for N in [500, 5000]:
        detections = 0
        for rep in range(n_reps):
            data = simulate_dataset(
                shape="disk", enrichment_type="wedge", n_points=N, quantile=q, seed=rep
            )
            coords = data["coords"]
            y = data["labels"]
            r = np.linalg.norm(coords, axis=1)
            theta = np.arctan2(coords[:, 1], coords[:, 0])
            res = compute_p_value(r, theta, y, n_perm=50)
            p_val = res.p_value
            if p_val <= alpha:
                detections += 1
        powers.append(detections / n_reps)

    # Power at N=5000 should be >= power at N=500
    assert powers[1] >= powers[0]


def test_rotation_invariance():
    """Verify that anisotropy score is invariant to rotation."""
    N = 2000
    q = 0.1
    data = simulate_dataset(shape="disk", enrichment_type="rim", n_points=N, quantile=q, seed=42)
    coords = data["coords"]
    y = data["labels"]
    r_base = np.linalg.norm(coords, axis=1)
    theta_base = np.arctan2(coords[:, 1], coords[:, 0])
    radar_base = compute_rsp_radar(r_base, theta_base, y)
    summ_base = compute_scalar_summaries(radar_base)

    # Rotate 90 degrees
    angle = np.pi / 2
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    coords_rot = coords @ R.T
    r_rot = np.linalg.norm(coords_rot, axis=1)
    theta_rot = np.arctan2(coords_rot[:, 1], coords_rot[:, 0])
    radar_rot = compute_rsp_radar(r_rot, theta_rot, y)
    summ_rot = compute_scalar_summaries(radar_rot)

    # Anisotropy should be very close
    assert np.isclose(summ_base.anisotropy, summ_rot.anisotropy, rtol=1e-2)

    # Peak angle should shift by ~90 degrees (modulo 2pi)
    # Use peak_distal_angle for rim pattern
    diff = (summ_rot.peak_distal_angle - summ_base.peak_distal_angle) % (2 * np.pi)
    # 90 deg is pi/2
    assert np.isclose(diff, np.pi / 2, atol=0.2)  # Allow some noise due to binning


if __name__ == "__main__":
    test_monotonic_power_with_N()
    test_rotation_invariance()
