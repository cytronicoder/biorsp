import pandas as pd
import pytest

from analysis.benchmarks.simlib.runner_harness import (
    normalize_scores_df,
    split_train_test,
)
from biorsp.plotting.standard import make_standard_plot_set


def test_normalize_scores_df_renames_bias_column():
    df = pd.DataFrame({"Coverage": [0.5], "Spatial_Bias_Score": [0.2]})
    out = normalize_scores_df(df)
    assert "Spatial_Score" in out.columns
    assert "Spatial_Bias_Score" not in out.columns
    assert out.loc[0, "Spatial_Score"] == pytest.approx(0.2)


def test_normalize_scores_df_missing_raises():
    df = pd.DataFrame({"Coverage": [0.5]})
    with pytest.raises(AssertionError):
        normalize_scores_df(df)


def test_split_train_test_deterministic():
    df = pd.DataFrame({"case_id": [f"c{i}" for i in range(10)], "Coverage": [0.1] * 10})
    res1 = split_train_test(df, ["case_id"], test_frac=0.2, seed=123)
    res2 = split_train_test(df, ["case_id"], test_frac=0.2, seed=123)
    assert res1.test_idx.equals(res2.test_idx)
    assert res1.train_idx.equals(res2.train_idx)


def test_make_standard_plot_set_creates_outputs(tmp_path):
    scores_df = pd.DataFrame(
        {
            "Coverage": [0.6, 0.8],
            "Spatial_Score": [0.2, 0.4],
            "gene": ["g1", "g2"],
            "Archetype_pred": ["Ubiquitous", "Gradient"],
            "Archetype_true": ["Ubiquitous", "Gradient"],
        }
    )
    fig_paths = make_standard_plot_set(
        scores_df,
        outdir=tmp_path,
        thresholds={"C_cut": 0.3, "S_cut": 0.15},
        truth_col="Archetype_true",
        pred_col="Archetype_pred",
        gene_col="gene",
    )
    expected_files = [
        "fig_cs_scatter.png",
        "fig_cs_marginals.png",
        "fig_confusion_or_composition.png",
        "fig_archetype_examples.png",
        "fig_top_tables.png",
    ]
    for fname in expected_files:
        fpath = tmp_path / fname
        assert fpath.exists(), f"Missing figure {fname}"
        assert fpath.stat().st_size > 0
    assert fig_paths, "Returned mapping should not be empty"
