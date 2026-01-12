#!/usr/bin/env python
"""Smoke test for all RSP plotting examples.

This script ensures that all plotting code examples can run without errors.
It does NOT validate correctness of plots, only that they execute cleanly.

Usage:
    python scripts/smoke_plot_all_rsp_examples.py

Exit codes:
    0: All plots generated successfully
    1: One or more plots failed
"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend
matplotlib.use("Agg")

from biorsp import BioRSPConfig, compute_rsp_radar, polar_coordinates
from biorsp.core.summaries import compute_scalar_summaries
from biorsp.core.typing import RadarResult
from biorsp.plotting.radar import plot_radar, plot_radar_absolute


def create_synthetic_data(n_cells=500, seed=42):
    """Create synthetic spatial data for testing."""
    rng = np.random.default_rng(seed)

    # Polar embedding
    r = np.sqrt(rng.random(n_cells))
    theta = rng.uniform(-np.pi, np.pi, n_cells)
    coords = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Wedge pattern (niche)
    wedge_mask = np.abs(theta) < 0.6
    y_niche = np.zeros(n_cells)
    y_niche[wedge_mask] = 1.0

    # Rim pattern
    rim_mask = r > 0.7
    y_rim = np.zeros(n_cells)
    y_rim[rim_mask] = 1.0

    # Core pattern
    core_mask = r < 0.3
    y_core = np.zeros(n_cells)
    y_core[core_mask] = 1.0

    return coords, {"niche": y_niche, "rim": y_rim, "core": y_core}


def test_basic_radar_plot():
    """Test basic plot_radar with signed mode."""
    print("  Testing basic radar plot (signed mode)...")

    coords, patterns = create_synthetic_data()
    r, theta = polar_coordinates(coords, v=np.array([0.0, 0.0]))
    y = patterns["niche"]

    config = BioRSPConfig(delta_deg=60, B=36)
    radar = compute_rsp_radar(r, theta, y, config=config)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_radar(radar, ax=ax, title="Basic Test", mode="signed", theta_convention="math")
    plt.close(fig)
    print("    ✓ Basic radar plot passed")


def test_theta_conventions():
    """Test both math and compass conventions."""
    print("  Testing theta conventions...")

    coords, patterns = create_synthetic_data()
    r, theta = polar_coordinates(coords, v=np.array([0.0, 0.0]))
    y = patterns["niche"]

    config = BioRSPConfig(delta_deg=60, B=36)
    radar = compute_rsp_radar(r, theta, y, config=config)

    # Math convention
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "polar"})
    plot_radar(radar, ax=ax1, title="Math Conv", mode="signed", theta_convention="math")
    plt.close(fig1)

    # Compass convention
    fig2, ax2 = plt.subplots(subplot_kw={"projection": "polar"})
    plot_radar(radar, ax=ax2, title="Compass Conv", mode="signed", theta_convention="compass")
    plt.close(fig2)

    print("    ✓ Theta conventions passed")


def test_plot_modes():
    """Test all plotting modes (signed, proximal, distal)."""
    print("  Testing all plot modes...")

    coords, patterns = create_synthetic_data()
    r, theta = polar_coordinates(coords, v=np.array([0.0, 0.0]))
    y = patterns["rim"]

    config = BioRSPConfig(delta_deg=60, B=36)
    radar = compute_rsp_radar(r, theta, y, config=config)

    for mode in ["signed", "proximal", "distal"]:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        plot_radar(radar, ax=ax, title=f"Mode: {mode}", mode=mode, theta_convention="math")
        plt.close(fig)

    print("    ✓ All plot modes passed")


def test_radial_max_options():
    """Test different radial_max settings."""
    print("  Testing radial_max options...")

    coords, patterns = create_synthetic_data()
    r, theta = polar_coordinates(coords, v=np.array([0.0, 0.0]))
    y = patterns["core"]

    config = BioRSPConfig(delta_deg=60, B=36)
    radar = compute_rsp_radar(r, theta, y, config=config)

    # Add outlier
    radar.rsp[0] = 10.0

    for radial_setting in [None, "max", 2.0]:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        plot_radar(
            radar,
            ax=ax,
            title=f"Radial: {radial_setting}",
            mode="signed",
            radial_max=radial_setting,
            theta_convention="math",
        )
        plt.close(fig)

    print("    ✓ Radial max options passed")


def test_absolute_split_plot():
    """Test plot_radar_absolute (side-by-side proximal/distal)."""
    print("  Testing absolute (split) plot...")

    coords, patterns = create_synthetic_data()
    r, theta = polar_coordinates(coords, v=np.array([0.0, 0.0]))
    y = patterns["niche"]

    config = BioRSPConfig(delta_deg=60, B=36)
    radar = compute_rsp_radar(r, theta, y, config=config)

    fig = plot_radar_absolute(
        radar, title="Split View", color="b", alpha=0.3, theta_convention="math"
    )
    plt.close(fig)

    print("    ✓ Absolute/split plot passed")


def test_with_summaries():
    """Test plot with summary annotations."""
    print("  Testing plot with summaries...")

    coords, patterns = create_synthetic_data()
    r, theta = polar_coordinates(coords, v=np.array([0.0, 0.0]))
    y = patterns["niche"]

    config = BioRSPConfig(delta_deg=60, B=36)
    radar = compute_rsp_radar(r, theta, y, config=config)
    summaries = compute_scalar_summaries(radar)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_radar(
        radar,
        ax=ax,
        title="With Summaries",
        mode="signed",
        summaries=summaries,
        show_anchors=True,
        theta_convention="math",
    )
    plt.close(fig)

    print("    ✓ Plot with summaries passed")


def test_nan_handling():
    """Test plotting with NaN values."""
    print("  Testing NaN handling...")

    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    rsp = np.sin(3 * theta)
    rsp[5:8] = np.nan  # Add NaN gap

    radar = RadarResult(
        rsp=rsp,
        counts_fg=np.ones(20) * 10,
        counts_bg=np.ones(20) * 10,
        centers=theta,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(20, dtype=bool),
    )

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_radar(radar, ax=ax, title="With NaN", mode="signed", theta_convention="math")
    plt.close(fig)

    print("    ✓ NaN handling passed")


def test_forced_zero_mask():
    """Test explicit forced-zero sector visualization."""
    print("  Testing forced-zero mask...")

    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    rsp = np.random.randn(20)
    rsp[10] = 0.0  # Zero value

    zero_mask = np.zeros(20, dtype=bool)
    zero_mask[10] = True  # Mark as forced-zero

    radar = RadarResult(
        rsp=rsp,
        counts_fg=np.ones(20) * 10,
        counts_bg=np.ones(20) * 10,
        centers=theta,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(20, dtype=bool),
        forced_zero_mask=zero_mask,
    )

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_radar(radar, ax=ax, title="Forced-Zero", mode="signed", theta_convention="math")
    plt.close(fig)

    print("    ✓ Forced-zero mask passed")


def test_wraparound_continuity():
    """Test circular wrap-around at 0/2π boundary."""
    print("  Testing wrap-around continuity...")

    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    # Pattern centered at theta=0 (crosses boundary)
    rsp = np.cos(theta) * 2.0 + 1.0
    # Add NaN gap in middle to test segmentation
    rsp[9:12] = np.nan

    radar = RadarResult(
        rsp=rsp,
        counts_fg=np.ones(20) * 10,
        counts_bg=np.ones(20) * 10,
        centers=theta,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(20, dtype=bool),
    )

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_radar(radar, ax=ax, title="Wrap-Around", mode="signed", theta_convention="math")
    plt.close(fig)

    print("    ✓ Wrap-around continuity passed")


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("RSP Plotting Smoke Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_basic_radar_plot,
        test_theta_conventions,
        test_plot_modes,
        test_radial_max_options,
        test_absolute_split_plot,
        test_with_summaries,
        test_nan_handling,
        test_forced_zero_mask,
        test_wraparound_continuity,
    ]

    failed = []

    for test_func in tests:
        try:
            test_func()
        except Exception as e:  # noqa: PERF203
            print(f"    ✗ FAILED: {e}")
            failed.append(test_func.__name__)

    print("\n" + "=" * 60)
    if not failed:
        print("✓ All smoke tests PASSED")
        print("=" * 60 + "\n")
        return 0
    else:
        print(f"✗ {len(failed)} test(s) FAILED:")
        for name in failed:
            print(f"  - {name}")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
