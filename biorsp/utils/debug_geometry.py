import numpy as np

from biorsp.core.geometry import (
    get_sector_indices,
    polar_coordinates,
    wrapped_circular_distance,
)


def debug_polar_sanity(z, v):
    """Task 1: Geometric sanity checks."""
    print("--- Debug Polar Sanity ---")
    print(f"Vantage point v: {v}")

    p_right = v + np.array([1.0, 0.0])
    r, theta = polar_coordinates(np.array([p_right]), v)
    print(f"Right point {p_right}: r={r[0]:.4f}, theta={theta[0]:.4f} (expected ~0)")
    assert np.isclose(theta[0], 0, atol=1e-5), f"Right point theta {theta[0]} != 0"

    p_above = v + np.array([0.0, 1.0])
    r, theta = polar_coordinates(np.array([p_above]), v)
    print(f"Above point {p_above}: r={r[0]:.4f}, theta={theta[0]:.4f} (expected ~pi/2)")
    assert np.isclose(theta[0], np.pi / 2, atol=1e-5), f"Above point theta {theta[0]} != pi/2"

    p_left_up = v + np.array([-1.0, 0.0001])
    p_left_down = v + np.array([-1.0, -0.0001])
    _, theta_up = polar_coordinates(np.array([p_left_up]), v)
    _, theta_down = polar_coordinates(np.array([p_left_down]), v)
    print(f"Left-Up point: theta={theta_up[0]:.4f} (expected ~pi)")
    print(f"Left-Down point: theta={theta_down[0]:.4f} (expected ~-pi)")

    assert np.isclose(np.abs(theta_up[0]), np.pi, atol=1e-3), "Left-Up not near pi"
    assert np.isclose(np.abs(theta_down[0]), np.pi, atol=1e-3), "Left-Down not near pi"

    print("Polar sanity checks passed.")


def debug_sector_membership(z, v, theta_grid, delta_deg, theta_star_idx):
    """Task 2: Verify sector membership."""
    print("\n--- Debug Sector Membership ---")
    r, theta = polar_coordinates(z, v)
    B = len(theta_grid)

    sector_indices_list = get_sector_indices(theta, B, delta_deg)
    indices_in_sector = sector_indices_list[theta_star_idx]

    theta_star = theta_grid[theta_star_idx]
    delta_rad = np.deg2rad(delta_deg)

    print(f"Theta* index: {theta_star_idx}, Theta*: {theta_star:.4f} rad, Delta: {delta_deg} deg")

    dists = wrapped_circular_distance(theta, theta_star)
    expected_mask = dists <= (delta_rad / 2.0 + 1e-9)

    actual_mask = np.zeros(len(z), dtype=bool)
    actual_mask[indices_in_sector] = True

    discrepancies = np.sum(expected_mask != actual_mask)
    print(f"Discrepancies between manual distance check and get_sector_indices: {discrepancies}")

    if discrepancies > 0:
        print("Indices in manual but not in library:", np.where(expected_mask & ~actual_mask)[0])
        print("Indices in library but not in manual:", np.where(~expected_mask & actual_mask)[0])

    test_angles = [
        theta_star,
        theta_star + delta_rad / 4,
        theta_star - delta_rad / 4,
        theta_star + np.pi,
    ]
    test_points = v + np.array([[np.cos(a), np.sin(a)] for a in test_angles])
    _, test_thetas = polar_coordinates(test_points, v)
    test_indices = get_sector_indices(test_thetas, B, delta_deg)[theta_star_idx]

    print("Test points check (0=center, 1=inside+, 2=inside-, 3=opposite):")
    print(f"  Center included: {0 in test_indices}")
    print(f"  Inside+ included: {1 in test_indices}")
    print(f"  Inside- included: {2 in test_indices}")
    print(f"  Opposite included: {3 in test_indices}")

    if delta_deg >= 180:
        print("WARNING: Delta >= 180 degrees. Directionality may be lost.")

    if 3 in test_indices and delta_deg < 360:
        print("WARNING: Opposite point included in sector!")

    n_in = len(indices_in_sector)
    print(f"Total points in sector: {n_in} / {len(z)}")
