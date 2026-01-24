"""Micro-benchmark for angle-window reuse vs. recompute.

Usage: python scripts/benchmark_angle_window.py
"""

import time

import numpy as np

from biorsp.core.geometry import get_sector_indices
from biorsp.core.radar import compute_rsp_radar


def time_trial(n_cells, B, reps=5):
    rng = np.random.default_rng(0)
    r = rng.normal(loc=5.0, scale=1.0, size=n_cells)
    theta = rng.uniform(-np.pi, np.pi, size=n_cells)
    y = rng.choice([0, 1], size=n_cells, p=[0.85, 0.15])

    sector_indices = get_sector_indices(theta, n_sectors=B, delta_deg=180.0)

    t_a = []
    t_b = []
    for _ in range(reps):
        t0 = time.perf_counter()

        _ = compute_rsp_radar(r, theta, y, B=B, delta_deg=180.0)
        t_a.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = compute_rsp_radar(r, theta, y, B=B, delta_deg=180.0, sector_indices=sector_indices)
        t_b.append(time.perf_counter() - t0)

    return np.mean(t_a), np.mean(t_b)


if __name__ == "__main__":
    sizes = [(1000, 180), (5000, 360), (20000, 360)]
    print("n_cells, B, time_recompute, time_reuse, speedup")
    for n_cells, B in sizes:
        ta, tb = time_trial(n_cells, B, reps=3)
        print(f"{n_cells}, {B}, {ta:.6f}, {tb:.6f}, {ta / tb:.2f}")
