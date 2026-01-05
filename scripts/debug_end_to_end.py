import os
import sys

import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from biorsp.plotting.workflow import make_end_to_end_figure
from biorsp.preprocess.geometry import angle_grid
from biorsp.utils.debug_geometry import debug_polar_sanity


def make_synthetic_data(scenario="wedge"):
    np.random.seed(42)
    n_bg = 2000
    n_fg = 500

    # Background: Uniform disk
    r_bg = np.sqrt(np.random.uniform(0, 1, n_bg))
    theta_bg = np.random.uniform(-np.pi, np.pi, n_bg)
    bg_x = r_bg * np.cos(theta_bg)
    bg_y = r_bg * np.sin(theta_bg)

    if scenario == "wedge_core":
        # Wedge at theta=0, concentrated near center (Core)
        # Core means r is smaller than background.
        r_fg = np.sqrt(np.random.uniform(0, 0.2, n_fg))  # Core
        theta_fg = np.random.normal(0, 0.2, n_fg)  # Wedge at 0

    elif scenario == "wedge_rim":
        # Wedge at theta=pi/2, concentrated near edge (Rim)
        r_fg = np.sqrt(np.random.uniform(0.8, 1.0, n_fg))  # Rim
        theta_fg = np.random.normal(np.pi / 2, 0.2, n_fg)  # Wedge at pi/2

    elif scenario == "global_rim":
        # Ring at edge, all angles
        r_fg = np.sqrt(np.random.uniform(0.8, 1.0, n_fg))
        theta_fg = np.random.uniform(-np.pi, np.pi, n_fg)

    elif scenario == "null":
        # Same as background
        r_fg = np.sqrt(np.random.uniform(0, 1, n_fg))
        theta_fg = np.random.uniform(-np.pi, np.pi, n_fg)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    fg_x = r_fg * np.cos(theta_fg)
    fg_y = r_fg * np.sin(theta_fg)

    z = np.column_stack([np.concatenate([bg_x, fg_x]), np.concatenate([bg_y, fg_y])])
    is_fg = np.concatenate([np.zeros(n_bg, dtype=bool), np.ones(n_fg, dtype=bool)])

    return z, is_fg


def run_debug_session():
    print("Running Debug Session...")

    # Task 1: Sanity Check
    z_dummy = np.random.rand(10, 2)
    v_dummy = np.array([0.0, 0.0])
    debug_polar_sanity(z_dummy, v_dummy)

    scenarios = ["wedge_core", "wedge_rim", "global_rim", "null"]

    for sc in scenarios:
        print(f"\n--- Running Scenario: {sc} ---")
        z, y = make_synthetic_data(sc)
        v = np.array([0.0, 0.0])

        B = 36
        delta_deg = 180
        theta_grid = angle_grid(B)

        outpath = f"debug_figure_{sc}.png"
        make_end_to_end_figure(z, y, v, theta_grid, delta_deg, outpath, feature_name=sc)


if __name__ == "__main__":
    run_debug_session()
