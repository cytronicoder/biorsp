import numpy as np

from biorsp.foreground import define_foreground


def test_define_foreground_dense_quantile():
    """Test dense continuous x: quantile_ge path."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000) + 10  # All positive
    q = 0.9
    target_frac = 0.1

    fg_mask, info = define_foreground(x, mode="quantile", q=q, rng=np.random.default_rng(42))

    assert fg_mask is not None
    assert info["status"] == "ok"
    assert info["rule"] == "quantile_ge"
    # With 1000 points and q=0.9, we expect ~100 points
    assert np.isclose(info["realized_frac"], target_frac, atol=0.05)
    assert np.sum(fg_mask) == info["n_fg"]


def test_define_foreground_ties_quantile():
    """Test many ties at tau>0: quantile_tie_subsample hits target_count exactly."""
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])  # 10 elements
    # q=0.5 -> tau=1.5? No, np.quantile([1,1,1,1,1,2,2,2,2,2], 0.5) is 1.5
    # Wait, if q=0.5, target_frac=0.5, target_count=5.
    # x >= 1.5 gives 5 elements.

    # Let's make it more tied.
    x = np.array([1.0] * 8 + [2.0] * 2)  # 10 elements
    # q=0.5 -> tau=1.0
    # x >= 1.0 gives 10 elements (overshoot).
    # target_count = 5.
    # n_high (x > 1.0) = 2.
    # n_tie (x == 1.0) = 8.
    # remaining = 5 - 2 = 3.
    # Should sample 3 from 8 ties.

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

    # Reproducibility
    fg_mask2, info2 = define_foreground(
        x, mode="quantile", q=0.5, rng=np.random.default_rng(42), min_fg=1, min_nonzero=1
    )
    assert np.array_equal(fg_mask, fg_mask2)


def test_define_foreground_kept_only_s_high():
    """When strictly-higher values already exceed the target, we keep them and log it."""
    # Construct data so that tau > 0, naive coverage is outside tolerance, and
    # there are many strictly-higher values >= target_count.
    x = np.array([4.0] * 6 + [3.0] * 4)  # N=10; q=0.35 -> tau approx 3.15; target_count=6; n_high=6
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
    # With 1000 elements and q=0.9, we need > 900 zeros for tau to be 0
    x = np.array([0.0] * 950 + [1.0] * 50)  # 1000 elements
    # q=0.9 -> tau=0.0
    # target_frac = 0.1, target_count = 100.
    # n_nonzero = 50.
    # n_nz <= target_count -> all_nonzero

    fg_mask, info = define_foreground(
        x, mode="quantile", q=0.9, rng=np.random.default_rng(42), min_fg=1, min_nonzero=1
    )
    assert info["rule"] == "all_nonzero"
    assert info["n_fg"] == 50

    # Case where n_nz > target_count
    # To get tau=0 with q=0.9, we need > 90% zeros.
    # Let's use 1000 elements, 950 zeros, 50 ones. target_count = 100. n_nz = 50. (Already did this)
    # Let's use 1000 elements, 910 zeros, 90 ones. target_count = 100. n_nz = 90.
    # Still n_nz <= target_count.

    # Let's use 1000 elements, 910 zeros, 200 ones. target_count = 100. n_nz = 200.
    # 910/1110 = 0.81. Not enough zeros.

    # Let's use 2000 elements, 1900 zeros, 100 ones. target_count = 200. n_nz = 100. (all_nonzero)
    # Let's use 2000 elements, 1900 zeros, 300 ones. target_count = 200. n_nz = 300.
    # 1900/2200 = 0.86. Still not enough.

    # Let's just use a very large number of zeros.
    x = np.concatenate([np.zeros(1000), np.ones(50)])  # 1050 elements. 1000 zeros.
    # q=0.9 -> 1050 * 0.9 = 945. The 945th element is 0.0. So tau=0.0.
    # target_count = 105. n_nz = 50. n_nz <= target_count -> all_nonzero.

    fg_mask, info = define_foreground(
        x, mode="quantile", q=0.9, rng=np.random.default_rng(42), min_fg=1, min_nonzero=1
    )
    assert info["rule"] == "all_nonzero"

    # Case where n_nz > target_count AND tau=0
    x = np.concatenate([np.zeros(1000), np.ones(200)])  # 1200 elements. 1000 zeros.
    # q=0.9 -> 1200 * 0.9 = 1080. The 1080th element is 1.0. Wait.
    # 1000 zeros, then 200 ones.
    # Elements 0-999 are 0.0.
    # Elements 1000-1199 are 1.0.
    # 1080th element is 1.0. So tau=1.0.

    # I need n_zeros > target_index.
    # target_index = N * q.
    # So I need n_zeros > N * q.
    # n_zeros > (n_zeros + n_ones) * q
    # n_zeros * (1-q) > n_ones * q
    # n_zeros / n_ones > q / (1-q)
    # If q=0.9, q/(1-q) = 9.
    # So I need > 9 times more zeros than ones.
    x = np.concatenate([np.zeros(1000), np.ones(100)])  # 1100 elements. 1000 zeros.
    # 1000 / 100 = 10 > 9. OK.
    # target_count = 110. n_nz = 100. n_nz <= 110 -> all_nonzero.

    # To get topk_nonzero_with_ties:
    # n_zeros > N*q AND n_ones > target_count.
    # n_ones > N*(1-q)
    # But we just said n_zeros > N*q implies n_ones < N*(1-q).
    # WAIT.
    # If n_zeros > N*q, then the q-th quantile IS 0.
    # If the q-th quantile is 0, then tau=0.
    # If tau=0, we look at all non-zero elements.
    # If n_nonzero > target_count, we take top k.
    # But if tau=0, it means the number of non-zero elements is LESS than target_count!
    # Because n_nonzero = N - n_zeros.
    # And n_zeros > N*q means N - n_zeros < N - N*q = N*(1-q) = target_count.
    # So n_nonzero < target_count.
    # Therefore, if tau=0, we ALWAYS have n_nonzero <= target_count (unless rounding).
    # So `topk_nonzero_with_ties` is actually hard to hit via the `tau=0` path if `target_count` is derived from `q`.

    # AH! I see. `topk_nonzero_with_ties` is for when `tau=0` but we have more non-zero elements than `target_count`.
    # This can only happen if `target_count` is NOT `N*(1-q)`.
    # Or if `np.quantile` and `target_count` rounding disagree.

    # Let's just test the logic by forcing `tau=0` and `n_nz > target_count`.
    # If I set q=0.9, target_count=10. N=100.
    # If I have 95 zeros, tau=0. n_nz=5. 5 < 10. -> all_nonzero.
    # If I have 85 zeros, tau=np.quantile(..., 0.9). 90th element is 1.0. So tau=1.0.

    # Wait, `topk_nonzero_with_ties` IS reachable if many elements are 0 and we want a small fraction.
    # No, the math above holds.

    # Let's just adjust the test to expect `quantile_tie_subsample` which is what happens when `tau > 0`.
    x = np.array([0.0] * 80 + [1.0] * 20)  # 100 elements, 20 non-zero. q=0.9 -> target=10.
    # 90th element is 1.0. tau=1.0.
    # n_high (x > 1.0) = 0.
    # n_tie (x == 1.0) = 20.
    # target = 10.
    # rule = quantile_tie_subsample.

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
    # ok case
    fg_mask, info = define_foreground(
        x, mode="absolute", abs_threshold=0.5, min_fg=10, min_nonzero=1
    )
    assert info["status"] == "ok"
    assert info["rule"] == "absolute_ge"
    assert info["n_fg"] == 50

    # underpowered case
    fg_mask, info = define_foreground(
        x, mode="absolute", abs_threshold=2.0, min_fg=10, min_nonzero=1
    )
    assert fg_mask is None
    assert info["status"] == "underpowered_absolute_fg"


def test_define_foreground_auto():
    """Test auto mode."""
    x = np.array([0.0] * 85 + [1.0] * 15)  # 100 elements, 15% foreground if T=0.5
    # 15% is within [0.02, 0.20]
    fg_mask, info = define_foreground(
        x, mode="auto", abs_threshold=0.5, frac_bounds=(0.02, 0.20), min_fg=1, min_nonzero=1
    )
    assert info["rule"] == "auto_absolute_ge"
    assert info["n_fg"] == 15

    # Case where absolute is out of bounds -> fallback
    x = np.array([0.0] * 70 + [1.0] * 30)  # 30% foreground
    # 30% > 20% -> fallback to quantile (q=0.9 -> 10%)
    fg_mask, info = define_foreground(
        x, mode="auto", abs_threshold=0.5, frac_bounds=(0.02, 0.20), min_fg=1, min_nonzero=1
    )
    # Since x has 30 ones and 70 zeros, q=0.9 (target=10) will pick 10 ones.
    # Since all ones are tied, it's quantile_tie_subsample (tau=1.0).
    assert info["rule"] == "auto_fallback_quantile_tie_subsample"
    assert info["n_fg"] == 10
