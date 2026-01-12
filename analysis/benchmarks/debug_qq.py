"""Debug QQ quantile generation to see what's actually happening."""

import matplotlib.pyplot as plt
import numpy as np

from biorsp.simulations import metrics

np.random.seed(42)
p_values = np.random.uniform(0, 1, 1000)

print("=" * 70)
print("DEBUGGING QQ QUANTILE GENERATION")
print("=" * 70)

expected, observed = metrics.qq_quantiles(p_values, alpha=0.05)

print(f"\nTotal quantile points: {len(expected)}")
print(f"Expected quantiles range: [{expected.min():.4f}, {expected.max():.4f}]")
print(f"Observed quantiles range: [{observed.min():.4f}, {observed.max():.4f}]")

mask_sig = expected <= 0.05
n_sig = np.sum(mask_sig)
print(f"\nPoints with expected <= 0.05: {n_sig} ({100 * n_sig / len(expected):.0f}%)")
print(f"Expected values in sig region: {expected[mask_sig]}")
print(f"Observed values in sig region: {observed[mask_sig]}")

print(f"\nFirst 10 expected quantiles: {expected[:10]}")
print(f"Last 10 expected quantiles: {expected[-10:]}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.hist(expected, bins=50, alpha=0.7, edgecolor="black")
ax.axvline(0.05, color="red", linestyle="--", label="α=0.05")
ax.set_xlabel("Expected quantile value")
ax.set_ylabel("Count")
ax.set_title("Distribution of Expected Quantiles")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.hist(expected[expected <= 0.05], bins=30, alpha=0.7, edgecolor="black", color="orange")
ax.set_xlabel("Expected quantile value")
ax.set_ylabel("Count")
ax.set_title(f"Zoomed: Expected Quantiles ≤ 0.05 (n={n_sig})")
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.scatter(expected, observed, alpha=0.6, s=10)
ax.plot([0, 1], [0, 1], "r--", alpha=0.5)
ax.set_xlabel("Expected p-value")
ax.set_ylabel("Observed p-value")
ax.set_title("QQ Plot - Full Range")
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.scatter(
    expected[mask_sig], observed[mask_sig], alpha=0.7, s=30, edgecolors="black", linewidth=0.5
)
ax.plot([0, 0.05], [0, 0.05], "r--", alpha=0.5)
ax.set_xlabel("Expected p-value")
ax.set_ylabel("Observed p-value")
ax.set_title(f"QQ Plot - Zoomed (n={n_sig} points)")
ax.set_xlim(-0.001, 0.052)
ax.set_ylim(-0.001, 0.052)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/debug_qq_quantiles.png", dpi=150, bbox_inches="tight")
print("\n✓ Debug plot saved to: outputs/debug_qq_quantiles.png")
print("=" * 70)
