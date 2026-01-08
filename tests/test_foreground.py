import numpy as np

from biorsp.preprocess.foreground import define_foreground


def test_define_foreground_dense_quantile():
    """Test dense continuous x: quantile_ge path."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000) + 10
    q = 0.9
    target_frac = 0.1

    fg_mask, info = define_foreground(x, mode="quantile", q=q, rng=np.random.default_rng(42))

    assert fg_mask is not None
    assert info["status"] == "ok"
    assert info["rule"] == "quantile_ge"

    assert np.isclose(info["realized_frac"], target_frac, atol=0.05)
    assert np.sum(fg_mask) == info["n_fg"]


def test_define_foreground_ties_quantile():
    """Test many ties at tau>0: quantile_tie_subsample hits target_count exactly."""
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    x = np.array([1.0] * 8 + [2.0] * 2)

    fg_mask, info = define_foreground(
        x, mode="quantile", q=0.5, rng=np.random.default_rng(42), min_fg=1, min_nonzero=1
    )

    assert info["rule"] == "quantile_tie_subsample"
    assert info["target_count"] == 5
    assert info["n_fg"] == 5
    assert info["n_high"] == 2
    assert info["n_tie"] == 8
    assert info["sampled_k"] == 3
    assert np.sum(fg_mask) == 5

    fg_mask2, info2 = define_foreground(
        x, mode="quantile", q=0.5, rng=np.random.default_rng(42), min_fg=1, min_nonzero=1
    )
    assert np.array_equal(fg_mask, fg_mask2)


def test_define_foreground_kept_only_s_high():
    """When strictly-higher values already exceed the target, we keep them and log it."""

    x = np.array([4.0] * 6 + [3.0] * 4)
    fg_mask, info = define_foreground(
        x,
        mode="quantile",
        q=0.35,
        rng=np.random.default_rng(0),
        overshoot_tol=0.0,
        min_fg=1,
        min_nonzero=1,
    )
    assert info["rule"] == "quantile_tie_subsample"
    assert info.get("kept_only_s_high", False) is True
    assert info["n_high"] == 6
    assert info["n_fg"] == 6


def test_define_foreground_zero_inflated():
    """Test zero-inflated x where tau==0: topk_nonzero_with_ties path."""

    x = np.array([0.0] * 950 + [1.0] * 50)

    fg_mask, info = define_foreground(
        x, mode="quantile", q=0.9, rng=np.random.default_rng(42), min_fg=1, min_nonzero=1
    )
    assert info["rule"] == "all_nonzero"
    assert info["n_fg"] == 50

    x = np.concatenate([np.zeros(1000), np.ones(50)])

    fg_mask, info = define_foreground(
        x, mode="quantile", q=0.9, rng=np.random.default_rng(42), min_fg=1, min_nonzero=1
    )
    assert info["rule"] == "all_nonzero"

    x = np.concatenate([np.zeros(1000), np.ones(200)])

    x = np.concatenate([np.zeros(1000), np.ones(100)])

    x = np.array([0.0] * 80 + [1.0] * 20)

    fg_mask, info = define_foreground(
        x, mode="quantile", q=0.9, rng=np.random.default_rng(42), min_fg=1, min_nonzero=1
    )
    assert info["rule"] == "quantile_tie_subsample"
    assert info["n_fg"] == 10


def test_define_foreground_underpowered():
    """Test underpowered: n_nonzero < min_nonzero returns None."""
    x = np.array([0.0] * 95 + [1.0] * 5)
    fg_mask, info = define_foreground(x, min_nonzero=10)
    assert fg_mask is None
    assert info["status"] == "underpowered_nonzero"


def test_define_foreground_absolute():
    """Test absolute mode."""
    x = np.array([0.0] * 50 + [1.0] * 50)

    fg_mask, info = define_foreground(
        x, mode="absolute", abs_threshold=0.5, min_fg=10, min_nonzero=1
    )
    assert info["status"] == "ok"
    assert info["rule"] == "absolute_ge"
    assert info["n_fg"] == 50

    fg_mask, info = define_foreground(
        x, mode="absolute", abs_threshold=2.0, min_fg=10, min_nonzero=1
    )
    assert fg_mask is None
    assert info["status"] == "underpowered_absolute_fg"


def test_define_foreground_auto():
    """Test auto mode."""
    x = np.array([0.0] * 85 + [1.0] * 15)

    fg_mask, info = define_foreground(
        x, mode="auto", abs_threshold=0.5, frac_bounds=(0.02, 0.20), min_fg=1, min_nonzero=1
    )
    assert info["rule"] == "auto_absolute_ge"
    assert info["n_fg"] == 15

    x = np.array([0.0] * 70 + [1.0] * 30)

    fg_mask, info = define_foreground(
        x, mode="auto", abs_threshold=0.5, frac_bounds=(0.02, 0.20), min_fg=1, min_nonzero=1
    )

    assert info["rule"] == "auto_fallback_quantile_tie_subsample"
    assert info["n_fg"] == 10
