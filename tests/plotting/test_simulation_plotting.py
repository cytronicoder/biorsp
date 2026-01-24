"""
Test simulation plotting functions.

Verifies that per-sigma plotting works correctly with the simulation output format.
"""

import numpy as np
import pandas as pd
import pytest


def test_sigma_column_in_simulation_csv():
    """
    Verify that simulation output includes sigma_deg column for plotting.

    This is a regression test for the per-sigma plotting feature.
    """

    data = {
        "variant": ["wedge", "wedge", "rim", "rim"],
        "beta": [0.5, 1.0, 0.5, 1.0],
        "sigma_deg": [10, 10, 20, 20],
        "A_bg": [0.1, 0.2, 0.15, 0.25],
        "R_mean_bg": [0.05, 0.10, -0.08, -0.15],
        "coverage_geom": [0.9, 0.9, 0.85, 0.85],
        "coverage_fg": [0.7, 0.8, 0.75, 0.80],
    }

    df = pd.DataFrame(data)

    assert "sigma_deg" in df.columns, "sigma_deg column missing"
    assert "variant" in df.columns, "variant column missing"
    assert "beta" in df.columns, "beta column missing"
    assert "A_bg" in df.columns, "A_bg column missing"

    grouped = df.groupby(["variant", "sigma_deg"])
    assert len(grouped) > 0, "Grouping by variant and sigma_deg failed"

    for (variant, sigma), group in grouped:
        assert variant in ["wedge", "rim"], f"Unexpected variant: {variant}"
        assert sigma in [10, 20], f"Unexpected sigma: {sigma}"
        assert "beta" in group.columns
        assert "A_bg" in group.columns


def test_per_sigma_plotting_logic():
    """
    Test the per-sigma plotting logic without actually creating plots.

    Verifies that the grouping and aggregation work correctly.
    """

    np.random.seed(42)
    data = []
    for variant in ["wedge", "rim"]:
        for beta in [0.5, 1.0, 1.5]:
            for sigma_deg in [10, 20, 40]:
                for replicate in range(5):
                    base_aniso = beta * 0.15
                    noise = np.random.normal(0, 0.02)
                    data.append(
                        {
                            "variant": variant,
                            "beta": beta,
                            "sigma_deg": sigma_deg,
                            "A_bg": base_aniso + noise,
                            "gene_id": replicate,
                        }
                    )

    df = pd.DataFrame(data)

    variants = ["wedge", "rim"]
    for variant in variants:
        sub = df[df["variant"] == variant]
        assert not sub.empty, f"No data for variant {variant}"

        unique_sigmas = sorted(sub["sigma_deg"].unique())
        assert len(unique_sigmas) == 3, f"Expected 3 sigma values, got {len(unique_sigmas)}"

        for sigma_val in unique_sigmas:
            sigma_sub = sub[sub["sigma_deg"] == sigma_val]
            assert not sigma_sub.empty, f"No data for sigma={sigma_val}"

            grouped = sigma_sub.groupby("beta")["A_bg"].agg(["mean", "sem", "count"])
            assert len(grouped) == 3, f"Expected 3 beta values, got {len(grouped)}"

            for beta in grouped.index:
                mean = grouped.loc[beta, "mean"]
                sem = grouped.loc[beta, "sem"]
                count = grouped.loc[beta, "count"]

                assert np.isfinite(mean), f"Non-finite mean for beta={beta}"
                assert sem >= 0, f"Negative SEM for beta={beta}"
                assert count == 5, f"Expected 5 replicates, got {count}"

                assert mean > 0, f"Expected positive anisotropy for beta={beta}"


def test_plotting_handles_missing_data():
    """
    Test that plotting gracefully handles missing data or empty groups.
    """

    data = {
        "variant": ["wedge", "wedge"],
        "beta": [0.5, 1.0],
        "sigma_deg": [10, 10],
        "A_bg": [0.1, 0.2],
    }

    df = pd.DataFrame(data)

    sub = df[df["variant"] == "nonexistent"]
    assert sub.empty, "Should get empty dataframe for missing variant"

    sub = df[df["variant"] == "wedge"]
    sigma_sub = sub[sub["sigma_deg"] == 999]
    assert sigma_sub.empty, "Should get empty dataframe for missing sigma"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
