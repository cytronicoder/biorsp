#!/usr/bin/env python3
"""Create a visual comparison showing the difference in point density."""

import matplotlib.pyplot as plt
import numpy as np

from biorsp.simulations import metrics

np.random.seed(42)
p_values = np.random.uniform(0, 1, 1000)

expected_new, observed_new = metrics.qq_quantiles(p_values)

n_total = 100
expected_old = np.linspace(0, 1, n_total)
observed_old = np.sort(p_values)[:n_total]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(expected_old, observed_old, alpha=0.5, s=20)
ax.plot([0, 1], [0, 1], "r--", alpha=0.7, label="Perfect calibration")
ax.set_xlim(-0.01, 0.06)
ax.set_ylim(-0.01, 0.06)
ax.set_xlabel("Expected p-value")
ax.set_ylabel("Observed p-value")
ax.set_title("OLD: Uniform Spacing\n(~5 points in [0, 0.05])")
ax.grid(True, alpha=0.3)
ax.legend()

old_sig_points = np.sum(expected_old <= 0.05)
ax.text(
    0.03,
    0.001,
    f"{old_sig_points} points",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
)

ax = axes[1]
ax.scatter(expected_new, observed_new, alpha=0.5, s=20)
ax.plot([0, 1], [0, 1], "r--", alpha=0.7, label="Perfect calibration")
ax.set_xlim(-0.01, 0.06)
ax.set_ylim(-0.01, 0.06)
ax.set_xlabel("Expected p-value")
ax.set_ylabel("Observed p-value")
ax.set_title("NEW: Adaptive Spacing\n(60 points in [0, 0.05])")
ax.grid(True, alpha=0.3)
ax.legend()

new_sig_points = np.sum(expected_new <= 0.05)
ax.text(
    0.03,
    0.001,
    f"{new_sig_points} points",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
)

plt.suptitle(
    "QQ Plot Enhancement: Point Density Comparison (Zoomed to p ≤ 0.05)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("outputs/qq_density_comparison.png", dpi=150, bbox_inches="tight")
print("✓ Comparison saved to: outputs/qq_density_comparison.png")
print(
    f"  Old approach: {old_sig_points} points in [0, 0.05] ({100 * old_sig_points / n_total:.0f}%)"
)
print(
    f"  New approach: {new_sig_points} points in [0, 0.05] ({100 * new_sig_points / n_total:.0f}%)"
)
print(f"  Improvement: {new_sig_points / old_sig_points:.1f}× more points where it matters!")
