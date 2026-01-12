"""Debug script for end-to-end workflow visualization.

Generates end-to-end figures for synthetic scenarios to verify the workflow
is functioning correctly. This is a developer/debugging tool.

Usage:
    python scripts/debug_end_to_end.py

Requires:
    Package installation: pip install -e .
"""

from pathlib import Path

import numpy as np

try:
    from biorsp.plotting.workflow import make_end_to_end_figure
except ImportError as e:
    print("ERROR: Cannot import biorsp. Please install the package first:")
    print("  pip install -e .")
    print(f"Details: {e}")
    exit(1)


def make_synthetic_data(scenario="wedge"):
    np.random.seed(42)
    n_bg = 2000
    n_fg = 500

    r_bg = np.sqrt(np.random.uniform(0, 1, n_bg))
    theta_bg = np.random.uniform(-np.pi, np.pi, n_bg)
    bg_x = r_bg * np.cos(theta_bg)
    bg_y = r_bg * np.sin(theta_bg)

    if scenario == "wedge_core":
        r_fg = np.sqrt(np.random.uniform(0, 0.2, n_fg))
        theta_fg = np.random.normal(0, 0.2, n_fg)

    elif scenario == "wedge_rim":
        r_fg = np.sqrt(np.random.uniform(0.8, 1.0, n_fg))
        theta_fg = np.random.normal(np.pi / 2, 0.2, n_fg)

    elif scenario == "global_rim":
        r_fg = np.sqrt(np.random.uniform(0.8, 1.0, n_fg))
        theta_fg = np.random.uniform(-np.pi, np.pi, n_fg)

    elif scenario == "null":
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
    """Run debug session for all synthetic scenarios."""
    print("Running Debug Session for End-to-End Workflow...")

    debug_dir = Path("scripts") / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ["wedge_core", "wedge_rim", "global_rim", "null"]

    for sc in scenarios:
        print(f"\n--- Running Scenario: {sc} ---")
        z, y = make_synthetic_data(sc)
        v = np.array([0.0, 0.0])

        B = 36
        delta_deg = 180
        theta_grid = np.linspace(-np.pi, np.pi, B, endpoint=False)

        outpath = debug_dir / f"debug_figure_{sc}.png"
        print(f"  Generating figure: {outpath}")
        make_end_to_end_figure(z, y, v, theta_grid, delta_deg, str(outpath), feature_name=sc)


if __name__ == "__main__":
    run_debug_session()
