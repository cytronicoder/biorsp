"""
Tests for plot standardization (PlotSpec and panels).

Validates that:
1. PlotSpec classification matches cutoff logic
2. Colors are assigned consistently
3. Panel generators produce valid outputs
"""

import numpy as np
import pandas as pd

from biorsp.plotting.spec import PlotSpec


def test_plotspec_classification_logic():
    """Verify that PlotSpec.classify() matches quadrant cutoffs exactly."""
    spec = PlotSpec(c_cut=0.30, s_cut=0.15)

    test_cases = [
        (0.50, 0.10, "Ubiquitous"),
        (0.50, 0.20, "Gradient"),
        (0.10, 0.20, "Patchy"),
        (0.10, 0.10, "Basal"),
        (0.30, 0.10, "Ubiquitous"),
        (0.50, 0.15, "Gradient"),
        (0.30, 0.15, "Gradient"),
        (0.29, 0.14, "Basal"),
    ]

    for coverage, spatial_score, expected in test_cases:
        result = spec.classify(coverage, spatial_score)
        assert result == expected, (
            f"Classification failed for C={coverage}, S={spatial_score}: "
            f"expected {expected}, got {result}"
        )


def test_plotspec_abstention():
    """Test that abstention flags are respected."""
    spec = PlotSpec(c_cut=0.30, s_cut=0.15, min_expr_cells=10)

    result = spec.classify(0.50, 0.20, abstain_flag=True)
    assert result == "Abstention"

    result = spec.classify(0.50, 0.20, n_expr_cells=5)
    assert result == "Abstention"

    result = spec.classify(np.nan, 0.20)
    assert result == "Abstention"

    result = spec.classify(0.50, np.nan)
    assert result == "Abstention"

    result = spec.classify(0.50, 0.20, n_expr_cells=50, abstain_flag=False)
    assert result == "Gradient"


def test_plotspec_dataframe_classification():
    """Test batch classification of a DataFrame."""
    spec = PlotSpec(c_cut=0.30, s_cut=0.15)

    df = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D"],
            "Coverage": [0.50, 0.10, 0.50, 0.10],
            "Spatial_Bias_Score": [0.10, 0.10, 0.20, 0.20],
        }
    )

    df = spec.classify_dataframe(df)

    assert "Archetype" in df.columns
    assert df.loc[df["gene"] == "A", "Archetype"].values[0] == "Ubiquitous"
    assert df.loc[df["gene"] == "B", "Archetype"].values[0] == "Basal"
    assert df.loc[df["gene"] == "C", "Archetype"].values[0] == "Gradient"
    assert df.loc[df["gene"] == "D", "Archetype"].values[0] == "Patchy"


def test_plotspec_colors_consistent():
    """Verify that all archetypes have defined colors."""
    spec = PlotSpec()

    archetypes = ["Ubiquitous", "Gradient", "Patchy", "Basal"]
    for archetype in archetypes:
        color = spec.get_color(archetype)
        assert color is not None
        assert color.startswith("#")
        assert len(color) == 7  # Hex color format


def test_plotspec_dataframe_validation():
    """Test that validation catches missing columns."""
    spec = PlotSpec()

    df_valid = pd.DataFrame({"Coverage": [0.5], "Spatial_Bias_Score": [0.2]})
    result = spec.validate_dataframe(df_valid)
    assert result["status"] in ["PASS", "WARNING"]
    assert len(result["issues"]) == 0

    df_invalid = pd.DataFrame({"Spatial_Bias_Score": [0.2]})
    result = spec.validate_dataframe(df_invalid)
    assert result["status"] == "FAIL"
    assert any("Coverage" in issue for issue in result["issues"])

    df_invalid = pd.DataFrame({"Coverage": [0.5]})
    result = spec.validate_dataframe(df_invalid)
    assert result["status"] == "FAIL"
    assert any("Spatial_Bias_Score" in issue for issue in result["issues"])


def test_plotspec_to_from_dict():
    """Test serialization and deserialization."""
    spec1 = PlotSpec(c_cut=0.25, s_cut=0.18, min_expr_cells=20)
    d = spec1.to_dict()

    spec2 = PlotSpec.from_dict(d)

    assert spec2.c_cut == 0.25
    assert spec2.s_cut == 0.18
    assert spec2.min_expr_cells == 20


def test_plotspec_cutoff_consistency():
    """
    CRITICAL TEST: Verify that classification boundaries match plotting cutoffs.

    This ensures no mismatch between quadrant lines and archetype colors.
    """
    spec = PlotSpec(c_cut=0.30, s_cut=0.15)
    c_cut, s_cut = spec.get_quadrant_bounds()

    assert spec.classify(c_cut, 0.10) == "Ubiquitous"
    assert spec.classify(c_cut, 0.20) == "Gradient"

    assert spec.classify(0.10, s_cut) == "Patchy"
    assert spec.classify(0.50, s_cut) == "Gradient"

    epsilon = 1e-6
    assert spec.classify(c_cut - epsilon, 0.10) == "Basal"
    assert spec.classify(0.10, s_cut - epsilon) == "Basal"


def test_panel_pairwise_with_truth():
    """Test Panel D pairwise plot with ground truth."""
    from biorsp.plotting.panels import plot_pairwise_panel

    spec = PlotSpec()

    np.random.seed(42)
    n_pairs = 100
    pairs_df = pd.DataFrame(
        {
            "gene_a": [f"gene_{i}" for i in range(n_pairs)],
            "gene_b": [f"gene_{i + 100}" for i in range(n_pairs)],
            "similarity_profile": np.random.rand(n_pairs) * 0.8,
            "is_true_edge": np.random.choice([True, False], n_pairs, p=[0.2, 0.8]),
        }
    )

    pairs_df.loc[pairs_df["is_true_edge"], "similarity_profile"] += 0.3

    fig = plot_pairwise_panel(pairs_df, spec)
    assert fig is not None

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_panel_marker_recovery():
    """Test Panel C marker recovery plot."""
    from biorsp.plotting.panels import plot_marker_recovery_panel

    precision_df = pd.DataFrame(
        {
            "k": [10, 20, 50, 100],
            "precision_at_k": [0.8, 0.7, 0.6, 0.5],
        }
    )

    fig = plot_marker_recovery_panel(precision_df)
    assert fig is not None

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_panel_examples_with_data():
    """Test Panel C examples plot with spatial data."""
    from biorsp.plotting.panels import plot_examples_panel

    spec = PlotSpec(c_cut=0.30, s_cut=0.15)

    np.random.seed(42)
    n_cells = 100
    n_genes = 20

    coords = np.random.rand(n_cells, 2) * 100
    expression = np.random.rand(n_cells, n_genes)
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    df = pd.DataFrame(
        {
            "gene": gene_names,
            "Coverage": np.random.rand(n_genes),
            "Spatial_Bias_Score": np.random.rand(n_genes),
        }
    )
    df = spec.classify_dataframe(df)

    fig = plot_examples_panel(coords, expression, gene_names, df, spec, n_examples_per_archetype=1)
    assert fig is not None

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plotspec_classification_matches_archetype_colors():
    """
    CRITICAL TEST: Verify that all classify() outputs exist in ARCHETYPE_COLORS.

    This ensures no mismatch between classification logic and plotting colors.
    """
    from biorsp.plotting.spec import ARCHETYPE_COLORS

    spec = PlotSpec(c_cut=0.30, s_cut=0.15)

    # Test all quadrant centers
    test_points = [
        (0.50, 0.05),  # High C, low S -> Ubiquitous
        (0.50, 0.30),  # High C, high S -> Gradient
        (0.10, 0.05),  # Low C, low S -> Basal
        (0.10, 0.30),  # Low C, high S -> Patchy
    ]

    for c, s in test_points:
        archetype = spec.classify(c, s)
        assert archetype in ARCHETYPE_COLORS, (
            f"Archetype '{archetype}' from classify(C={c}, S={s}) "
            f"not found in ARCHETYPE_COLORS: {list(ARCHETYPE_COLORS.keys())}"
        )

    # Test boundary points (exact cutoffs)
    boundary_points = [
        (0.30, 0.15),  # At both cutoffs -> should be Gradient
        (0.30, 0.05),  # At c_cut, below s_cut -> Ubiquitous
        (0.10, 0.15),  # Below c_cut, at s_cut -> Patchy
    ]

    for c, s in boundary_points:
        archetype = spec.classify(c, s)
        assert (
            archetype in ARCHETYPE_COLORS
        ), f"Boundary archetype '{archetype}' not found in ARCHETYPE_COLORS"


def test_story_generate_onepager_smoke():
    """Smoke test for onepager generation with minimal data."""
    import os
    import tempfile

    from biorsp.plotting.story import generate_onepager

    spec = PlotSpec(c_cut=0.30, s_cut=0.15)

    # Minimal DataFrame
    df = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "Coverage": [0.5, 0.5, 0.1, 0.1, 0.4, 0.3, 0.2, 0.15],
            "Spatial_Bias_Score": [0.05, 0.25, 0.05, 0.25, 0.1, 0.2, 0.08, 0.18],
        }
    )
    df = spec.classify_dataframe(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save runs.csv for generate_onepager
        runs_csv_path = os.path.join(tmpdir, "runs.csv")
        df.to_csv(runs_csv_path, index=False)

        # Save manifest.json
        import json

        manifest = {"benchmark": "smoke_test", "plot_spec": spec.to_dict()}
        manifest_path = os.path.join(tmpdir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Generate onepager
        fig, caption = generate_onepager(runs_csv_path, manifest_json=manifest_path, outdir=tmpdir)

        assert fig is not None
        assert caption is not None
        assert len(caption) > 0

        import matplotlib.pyplot as plt

        plt.close(fig)


def test_make_figures_smoke():
    """Smoke test for make_figures CLI components."""
    import json
    import os
    import tempfile

    from biorsp.plotting.make_figures import generate_panel_a

    spec = PlotSpec(c_cut=0.30, s_cut=0.15)

    # Create minimal runs.csv
    df = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D"],
            "Coverage": [0.5, 0.5, 0.1, 0.1],
            "Spatial_Bias_Score": [0.05, 0.25, 0.05, 0.25],
        }
    )
    df = spec.classify_dataframe(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path

        tmpdir = Path(tmpdir)

        # Save runs.csv
        df.to_csv(tmpdir / "runs.csv", index=False)

        # Save manifest
        manifest = {
            "benchmark": "test",
            "plot_spec": spec.to_dict(),
        }
        with open(tmpdir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        # Generate Panel A (kidney mode since no true_archetype column)
        generate_panel_a(df, spec, tmpdir, run_type="kidney")

        # Check file was saved
        assert os.path.exists(tmpdir / "A_archetype_scatter.png")


def test_kidney_standardized_plotting_smoke():
    """Smoke test for kidney standardized plotting adapter."""
    import os
    import tempfile

    # Skip if kidney adapter doesn't exist
    try:
        from analysis.kidney_atlas.utils.standardized_plotting import (
            generate_kidney_panels,
            save_kidney_manifest,
        )
    except ImportError:
        import pytest

        pytest.skip("Kidney standardized_plotting module not available")

    spec = PlotSpec(c_cut=0.30, s_cut=0.15, spatial_col="Spatial_Score")

    # Create minimal kidney-like DataFrame
    df = pd.DataFrame(
        {
            "gene": ["NPHS1", "UMOD", "PODXL", "SLC12A1"],
            "Coverage": [0.4, 0.35, 0.45, 0.25],
            "Spatial_Score": [0.1, 0.2, 0.08, 0.3],
            "condition": ["Control", "DKD", "Control", "AKI"],
        }
    )
    df = spec.classify_dataframe(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path

        tmpdir = Path(tmpdir)

        # Generate panels
        generate_kidney_panels(df, tmpdir, c_cut=0.30, s_cut=0.15)

        # Check standard plots were created
        assert (tmpdir / "fig_cs_scatter.png").exists()
        assert (tmpdir / "fig_cs_marginals.png").exists()

        # Save manifest with required arguments
        save_kidney_manifest(
            tmpdir,
            params={"test_param": "value"},
            n_genes=len(df),
            n_cells=100,
            c_cut=0.30,
            s_cut=0.15,
            runtime_seconds=1.0,
        )
        assert os.path.exists(tmpdir / "manifest.json")
