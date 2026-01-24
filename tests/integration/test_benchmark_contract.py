import pandas as pd
import pytest

from analysis.benchmarks.simlib.io_contract import validate_runs_df


def _base_runs_df():
    return pd.DataFrame(
        {
            "run_id": ["r1"],
            "benchmark": ["archetypes"],
            "mode": ["quick"],
            "seed": [1],
            "replicate_id": [0],
            "status": ["ok"],
            "abstain_flag": [False],
            "abstain_reason": ["ok"],
            "shape": ["disk"],
            "n_cells": [100],
            "timestamp": ["2020-01-01T00:00:00"],
            "center_x": [50.0],
            "center_y": [50.0],
            "pattern_family": ["ubiquitous"],
            "pattern_variant": ["uniform"],
            "target_prevalence": [1.0],
            "prevalence_empirical": [1.0],
            "n_fg": [100],
            "Coverage": [0.5],
            "Spatial_Score": [0.1],
            "Directionality": [0.0],
            "Archetype_true": ["Ubiquitous"],
            "Archetype_pred": ["Ubiquitous"],
            "C_cut": [0.3],
            "S_cut": [0.15],
            "thresholds_source": ["fixed_default"],
        }
    )


def test_validate_runs_df_passes_with_required_columns():
    df = _base_runs_df()
    validate_runs_df(df, "archetypes")


def test_validate_runs_df_raises_on_missing_column():
    df = _base_runs_df().drop(columns=["Coverage"])
    with pytest.raises(ValueError):
        validate_runs_df(df, "archetypes")
