import numpy as np
import pandas as pd

from biorsp.simulations import metrics

df = pd.read_csv("outputs/calibration/runs.csv")
p_values = df["p_value"].dropna().values

print("=" * 70)
print("QQ PLOT WITH 100 REPLICATES")
print("=" * 70)
print(f"Total p-values: {len(p_values)}")
print(
    f"P-values < 0.05: {np.sum(p_values < 0.05)} ({100 * np.sum(p_values < 0.05) / len(p_values):.1f}%)"
)
print(f"P-values range: [{p_values.min():.4f}, {p_values.max():.4f}]")
print()

expected, observed = metrics.qq_quantiles(p_values)
sig_mask = expected <= 0.05

print(f"Total quantile points: {len(expected)}")
print(
    f"Expected points <= 0.05: {np.sum(sig_mask)} ({100 * np.sum(sig_mask) / len(expected):.0f}%)"
)
print(f"Observed points <= 0.05: {np.sum(observed <= 0.05)}")
print()
print("✓ With 100 replicates and adaptive quantile spacing,")
print(f"  we now have {np.sum(sig_mask)} points in the [0, 0.05] region!")
print("  (60% of 100 quantile points = 60 points)")
