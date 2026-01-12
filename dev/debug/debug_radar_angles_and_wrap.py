"""Debug script to validate radar plotting angle conventions and wrap-around.

This script generates synthetic data with known geometry to verify:
1. Theta convention transformations (math vs compass)
2. Circular wrap-around continuity
3. Zero-filled sector visualization
4. Peak anchor alignment

Run with: python scripts/debug_radar_angles_and_wrap.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from biorsp.core.typing import RadarResult
from biorsp.plotting.radar import plot_radar


def create_wedge_at_angle(
    center_angle: float, n_sectors: int = 36, width: float = np.pi / 6
) -> RadarResult:
    """Create synthetic RSP with a wedge centered at a specific angle.

    Args:
        center_angle: Center of wedge in radians (math convention).
        n_sectors: Number of angular sectors.
        width: Angular width of the wedge in radians.

    Returns:
        RadarResult with a Gaussian-like wedge.
    """
    theta = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)

    angular_dist = np.abs(np.angle(np.exp(1j * (theta - center_angle))))
    rsp = 2.0 * np.exp(-0.5 * (angular_dist / (width / 2)) ** 2)

    rsp += 0.1 * np.random.randn(n_sectors)

    rsp = np.maximum(rsp, 0.1)

    return RadarResult(
        rsp=rsp,
        counts_fg=np.ones(n_sectors) * 100,
        counts_bg=np.ones(n_sectors) * 100,
        centers=theta,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(n_sectors, dtype=bool),
        zero_filled_mask=np.zeros(n_sectors, dtype=bool),
    )


def create_wraparound_pattern(n_sectors: int = 36) -> RadarResult:
    """Create RSP pattern that crosses the 0/2pi boundary.

    Returns:
        RadarResult with high values at theta ~= 0 and ~= 2pi.
    """
    theta = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)

    rsp = np.cos(theta) * 1.5 + 0.5

    rsp[n_sectors // 2 - 2 : n_sectors // 2 + 2] = np.nan

    zero_mask = np.zeros(n_sectors, dtype=bool)
    zero_mask[n_sectors // 4] = True

    return RadarResult(
        rsp=rsp,
        counts_fg=np.ones(n_sectors) * 50,
        counts_bg=np.ones(n_sectors) * 50,
        centers=theta,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(n_sectors, dtype=bool),
        forced_zero_mask=zero_mask,
    )


def create_signed_pattern(n_sectors: int = 36) -> RadarResult:
    """Create pattern with both proximal and distal shifts.

    Returns:
        RadarResult with positive and negative values.
    """
    theta = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)

    rsp = 2.0 * np.cos(theta)

    return RadarResult(
        rsp=rsp,
        counts_fg=np.ones(n_sectors) * 100,
        counts_bg=np.ones(n_sectors) * 100,
        centers=theta,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(n_sectors, dtype=bool),
        forced_zero_mask=np.zeros(n_sectors, dtype=bool),
    )


def main():
    """Generate debug plots."""
    output_dir = Path("scripts/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating debug plots for radar angle conventions and wrap-around...")

    print("\n1. Testing wedge at theta=0 (east)...")
    radar_east = create_wedge_at_angle(center_angle=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": "polar"})

    plot_radar(
        radar_east,
        ax=axes[0],
        title="Math Convention\n(0 at East, CCW)",
        mode="signed",
        theta_convention="math",
    )

    plot_radar(
        radar_east,
        ax=axes[1],
        title="Compass Convention\n(0 at North, CW)",
        mode="signed",
        theta_convention="compass",
    )

    fig.suptitle("Wedge at θ=0: Convention Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "debug_wedge_conventions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {output_dir / 'debug_wedge_conventions.png'}")

    print("\n2. Testing wrap-around continuity...")
    radar_wrap = create_wraparound_pattern()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    plot_radar(
        radar_wrap,
        ax=ax,
        title="Wrap-Around Pattern\n(Should be continuous at 0/2π)",
        mode="signed",
        theta_convention="math",
    )

    plt.savefig(output_dir / "debug_wraparound.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {output_dir / 'debug_wraparound.png'}")

    print("\n3. Testing signed pattern with proximal/distal labels...")
    radar_signed = create_signed_pattern()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "polar"})

    plot_radar(
        radar_signed,
        ax=axes[0],
        title="Signed Mode",
        mode="signed",
        theta_convention="math",
        radial_max="max",
    )

    plot_radar(
        radar_signed,
        ax=axes[0],
        title="Signed Mode (Robust Radial)",
        mode="signed",
        theta_convention="math",
        radial_max=None,
    )

    fig.suptitle("Signed RSP: Proximal vs Distal Shift", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "debug_signed_labels.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {output_dir / 'debug_signed_labels.png'}")

    print("\n4. Testing robust radial scaling...")
    radar_outlier = create_wedge_at_angle(center_angle=np.pi / 2)
    radar_outlier.rsp[0] = 20.0  # Add outlier

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": "polar"})

    plot_radar(
        radar_outlier,
        ax=axes[0],
        title="Radial Max = 'max'\n(Includes outlier)",
        mode="signed",
        radial_max="max",
    )

    plot_radar(
        radar_outlier,
        ax=axes[1],
        title="Radial Max = None\n(Robust 99th percentile)",
        mode="signed",
        radial_max=None,
    )

    fig.suptitle("Robust Radial Scaling", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "debug_robust_scaling.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {output_dir / 'debug_robust_scaling.png'}")

    print("\n✓ All debug plots generated successfully!")
    print(f"  Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
