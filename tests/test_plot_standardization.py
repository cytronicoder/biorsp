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
