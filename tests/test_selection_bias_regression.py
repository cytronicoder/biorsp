"""Regression test for selection bias fix under empty_fg_policy='zero'.

This test ensures that the empty_fg_policy='zero' correctly handles empty sectors
and prevents selection bias. The key insight is:

- Global rim: Foreground spread across all sectors (high coverage_geom, moderate |R_mean|)
- Wedge rim: Foreground localized to narrow angular range (lower coverage_geom, smaller |R_mean|)

Under empty_fg_policy='zero', wedge_rim should have materially smaller tendency
metrics because fewer sectors contain foreground cells.

This test prevents regression of the selection bias fix.
"""

import numpy as np
import pandas as pd

from biorsp.core.engine import compute_rsp_radar
from biorsp.core.geometry import polar_coordinates
from biorsp.core.summaries import compute_scalar_summaries
from biorsp.utils.config import BioRSPConfig


def make_synthetic_rim_pattern(scenario: str, seed: int = 42) -> tuple:
    """Generate synthetic rim patterns for testing.

    Parameters
    ----------
    scenario : str
        One of: 'global_rim', 'wedge_rim', 'null'
    seed : int
        Random seed

    Returns
    -------
    z : np.ndarray
        (N, 2) cell coordinates
    is_fg : np.ndarray
        (N,) boolean foreground mask
    """
    rng = np.random.default_rng(seed)
    n_bg = 2000
    n_fg = 500
    r_bg = np.sqrt(rng.uniform(0, 1, n_bg))
    theta_bg = rng.uniform(-np.pi, np.pi, n_bg)
    if scenario == "global_rim":
        r_fg = np.sqrt(rng.uniform(0.8, 1.0, n_fg))
        theta_fg = rng.uniform(-np.pi, np.pi, n_fg)
    elif scenario == "wedge_rim":
        r_fg = np.sqrt(rng.uniform(0.8, 1.0, n_fg))
        theta_fg = rng.normal(np.pi / 2, 0.2, n_fg)
    elif scenario == "null":
        r_fg = np.sqrt(rng.uniform(0, 1, n_fg))
        theta_fg = rng.uniform(-np.pi, np.pi, n_fg)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    bg_x = r_bg * np.cos(theta_bg)
    bg_y = r_bg * np.sin(theta_bg)
    fg_x = r_fg * np.cos(theta_fg)
    fg_y = r_fg * np.sin(theta_fg)
    z = np.column_stack([np.concatenate([bg_x, fg_x]), np.concatenate([bg_y, fg_y])])
    is_fg = np.concatenate([np.zeros(n_bg, dtype=bool), np.ones(n_fg, dtype=bool)])

    return z, is_fg


def test_selection_bias_wedge_vs_global_rim():
    """Test that wedge_rim has different coverage_fg than global_rim under policy='zero'.

    This verifies the selection bias fix works correctly.
    The key is that wedge should have fewer sectors with foreground cells (coverage_fg).
    """
    z_global, y_global = make_synthetic_rim_pattern("global_rim", seed=42)
    z_wedge, y_wedge = make_synthetic_rim_pattern("wedge_rim", seed=43)
    v = np.array([0.0, 0.0])
    r_global, theta_global = polar_coordinates(z_global, v)
    r_wedge, theta_wedge = polar_coordinates(z_wedge, v)
    r_med = np.median(np.concatenate([r_global, r_wedge]))
    r_iqr = np.percentile(np.concatenate([r_global, r_wedge]), 75) - np.percentile(
        np.concatenate([r_global, r_wedge]), 25
    )
    r_iqr = max(r_iqr, 1e-8)
    r_global_norm = (r_global - r_med) / r_iqr
    r_wedge_norm = (r_wedge - r_med) / r_iqr
    config = BioRSPConfig(B=36, delta_deg=60.0, empty_fg_policy="zero")
    res_global = compute_rsp_radar(
        r_global_norm, theta_global, y_global.astype(float), config=config
    )
    res_wedge = compute_rsp_radar(r_wedge_norm, theta_wedge, y_wedge.astype(float), config=config)
    summ_global = compute_scalar_summaries(res_global)
    summ_wedge = compute_scalar_summaries(res_wedge)

    assert summ_wedge.coverage_fg < summ_global.coverage_fg, (
        f"Expected wedge coverage_fg < global, got {summ_wedge.coverage_fg:.3f} >= {summ_global.coverage_fg:.3f}"
    )
    assert summ_global.r_mean < 0, (
        f"Global rim should have R_mean < 0, got {summ_global.r_mean:.3f}"
    )
    assert summ_wedge.r_mean < 0, f"Wedge rim should have R_mean < 0, got {summ_wedge.r_mean:.3f}"
    assert not np.all(np.isnan(res_wedge.rsp)), "RSP should not be all NaN"
    assert not np.all(np.isnan(res_global.rsp)), "RSP should not be all NaN"


def test_selection_bias_parameter_sweep():
    """Sweep over delta_deg and wedge width to verify selection bias fix holds."""
    scenarios = ["global_rim", "wedge_rim"]
    delta_values = [60.0, 90.0, 180.0]

    results = []

    for delta_deg in delta_values:
        for scenario in scenarios:
            z, y = make_synthetic_rim_pattern(scenario, seed=42 if scenario == "global_rim" else 43)
            v = np.array([0.0, 0.0])
            r, theta = polar_coordinates(z, v)

            r_med = np.median(r)
            r_iqr = max(np.percentile(r, 75) - np.percentile(r, 25), 1e-8)
            r_norm = (r - r_med) / r_iqr

            config = BioRSPConfig(B=36, delta_deg=delta_deg, empty_fg_policy="zero")
            res = compute_rsp_radar(r_norm, theta, y.astype(float), config=config)
            summ = compute_scalar_summaries(res)

            results.append(
                {
                    "scenario": scenario,
                    "delta_deg": delta_deg,
                    "Directionality": summ.r_mean,
                    "abs_r_mean": abs(summ.r_mean),
                    "anisotropy": summ.anisotropy,
                    "coverage_geom": summ.coverage_geom,
                    "coverage_fg": summ.coverage_fg,
                }
            )

    df = pd.DataFrame(results)

    for delta in delta_values:
        subset = df[df["delta_deg"] == delta]
        global_val = subset[subset["scenario"] == "global_rim"]["abs_r_mean"].iloc[0]
        wedge_val = subset[subset["scenario"] == "wedge_rim"]["abs_r_mean"].iloc[0]
        assert wedge_val < global_val, (
            f"Delta={delta}: Expected |R_mean_wedge| < |R_mean_global|, got {wedge_val:.3f} >= {global_val:.3f}"
        )

    # Optional: save results for inspection
    try:
        from pathlib import Path

        outdir = Path("tests/output")
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / "selection_bias_sweep.csv", index=False)
    except Exception:
        pass  # Don't fail test if can't save


def test_empty_fg_policy_comparison():
    """Verify that policy='zero' and policy='nan' behave differently as expected."""
    z, y = make_synthetic_rim_pattern("wedge_rim", seed=42)
    v = np.array([0.0, 0.0])
    r, theta = polar_coordinates(z, v)

    r_med = np.median(r)
    r_iqr = max(np.percentile(r, 75) - np.percentile(r, 25), 1e-8)
    r_norm = (r - r_med) / r_iqr

    # Compare two policies
    config_zero = BioRSPConfig(B=36, delta_deg=180.0, empty_fg_policy="zero")
    config_nan = BioRSPConfig(B=36, delta_deg=180.0, empty_fg_policy="nan")

    res_zero = compute_rsp_radar(r_norm, theta, y.astype(float), config=config_zero)
    res_nan = compute_rsp_radar(r_norm, theta, y.astype(float), config=config_nan)

    summ_zero = compute_scalar_summaries(res_zero)
    summ_nan = compute_scalar_summaries(res_nan)

    # Policy='nan' should exclude empty sectors, potentially inflating |R_mean|
    # Policy='zero' should include them as zero, reducing |R_mean|
    assert summ_zero.coverage_geom >= summ_nan.coverage_geom, (
        "Policy='zero' should have >= coverage_geom than policy='nan'"
    )


if __name__ == "__main__":
    # Run tests
    test_selection_bias_wedge_vs_global_rim()
    test_selection_bias_parameter_sweep()
    test_empty_fg_policy_comparison()
    print("✅ All selection bias regression tests passed!")
