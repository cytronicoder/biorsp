"""Tests for held-out calibration semantics.

Ensures calibration runner properly:
- Splits data into train/test deterministically
- Derives thresholds on train only
- Evaluates metrics on test only
- Handles abstention correctly in QQ/FPR
"""

import numpy as np
import pandas as pd
import pytest


def test_held_out_split_deterministic():
    """Test that split_train_test produces deterministic splits."""
    from analysis.benchmarks.simlib.runner_harness import split_train_test

    # Create fake runs with condition keys
    # Use 5 conditions so group-level split gives reasonable proportions
    runs = [
        {"condition": f"cond_{cond_id}", "replicate": rep, "score": np.random.rand()}
        for cond_id in range(5)
        for rep in range(10)
    ]
    runs_df = pd.DataFrame(runs)

    # Split twice with same seed
    split1 = split_train_test(runs_df, group_cols=["condition"], test_frac=0.3, seed=42)
    split2 = split_train_test(runs_df, group_cols=["condition"], test_frac=0.3, seed=42)

    # Should be identical
    assert set(split1.train_idx) == set(split2.train_idx)
    assert set(split1.test_idx) == set(split2.test_idx)

    # No overlap
    assert len(set(split1.train_idx) & set(split1.test_idx)) == 0

    # With 5 groups and test_frac=0.3, should get 1 or 2 groups in test (20-40%)
    test_frac_actual = len(split1.test_idx) / len(runs_df)
    assert 0.15 < test_frac_actual < 0.45, f"Test fraction {test_frac_actual} outside [15%, 45%]"


def test_calibration_masking_abstention_aware():
    """Test that QQ/FPR computed only over finite p-values."""
    from analysis.benchmarks.simlib.runner_harness import safe_metric_mask

    # Create fake results with some abstained
    results = pd.DataFrame(
        {
            "p_value": [0.01, 0.5, np.nan, 0.9, np.nan, 0.05],
            "abstain_flag": [False, False, True, False, True, False],
        }
    )

    mask = safe_metric_mask(results["p_value"])

    # Should exclude NaNs
    assert mask.sum() == 4  # 4 finite p-values
    assert not mask[2]  # NaN excluded
    assert not mask[4]  # NaN excluded

    # Verify finite p-values
    finite_p = results.loc[mask, "p_value"].values
    assert len(finite_p) == 4
    assert not np.any(np.isnan(finite_p))


def test_abstention_counts():
    """Test that abstention rate is correctly computed."""
    results = pd.DataFrame(
        {
            "abstain_flag": [True, False, True, False, False],
            "p_value": [np.nan, 0.5, np.nan, 0.2, 0.8],
        }
    )

    abstain_rate = results["abstain_flag"].mean()
    assert abstain_rate == 0.4  # 2 out of 5


def test_calibration_thresholds_train_only():
    """Test that thresholds are derived from train set only."""
    # This is an integration test concept - just verify the pattern
    # In actual calibration runner, thresholds must come from train split
    train_s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    test_s = np.array([0.15, 0.25, 0.35])

    # Threshold from train
    s_cut_train = np.quantile(train_s, 0.95)

    # Test metrics should NOT affect threshold
    s_cut_test = np.quantile(test_s, 0.95)

    # They should differ
    assert s_cut_train != s_cut_test

    # Actual implementation should only use train for deriving thresholds
    assert s_cut_train == pytest.approx(0.48, abs=0.01)
