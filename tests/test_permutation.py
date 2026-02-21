import numpy as np

from biorsp.permutation import (
    check_mode_consistency,
    perm_null_T,
    perm_null_T_and_profile,
    permute_foreground_within_donor,
)


def test_permute_preserves_donor_counts():
    f = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=bool)
    donor_to_idx = {
        "d1": np.array([0, 1, 2, 3]),
        "d2": np.array([4, 5, 6, 7]),
    }
    rng = np.random.default_rng(0)
    f_perm = permute_foreground_within_donor(f, donor_to_idx, rng)
    assert f_perm[donor_to_idx["d1"]].sum() == f[donor_to_idx["d1"]].sum()
    assert f_perm[donor_to_idx["d2"]].sum() == f[donor_to_idx["d2"]].sum()


def test_perm_null_t_and_profile_shapes():
    expr = np.array([1, 0, 2, 0, 3, 0, 1, 0], dtype=float)
    angles = np.linspace(0, 2 * np.pi, expr.size, endpoint=False)
    donor_ids = np.array(["d1", "d1", "d1", "d1", "d2", "d2", "d2", "d2"])

    out = perm_null_T_and_profile(
        expr=expr,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=6,
        n_perm=20,
        seed=0,
        donor_stratified=True,
    )
    assert out["null_T"].shape == (20,)
    assert out["null_E_phi"].shape == (20, 6)
    assert out["E_phi_obs"].shape == (6,)
    assert 0.0 <= float(out["p_T"]) <= 1.0
    assert bool(out["used_donor_stratified"]) is True
    assert "stratified_counts_signature" in out


def test_perm_null_t_and_profile_global_fallback_without_donors():
    expr = np.array([1, 0, 2, 0, 3, 0, 1, 0], dtype=float)
    angles = np.linspace(0, 2 * np.pi, expr.size, endpoint=False)

    out = perm_null_T_and_profile(
        expr=expr,
        angles=angles,
        donor_ids=None,
        n_bins=6,
        n_perm=15,
        seed=1,
        donor_stratified=True,
    )
    assert out["null_T"].shape == (15,)
    assert out["null_E_phi"].shape == (15, 6)
    assert bool(out["used_donor_stratified"]) is False
    assert "stratified_counts_signature" in out


def test_perm_null_t_mode_consistency_w1():
    rng = np.random.default_rng(7)
    n = 120
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    donor_ids = np.repeat(np.array(["d1", "d2", "d3"]), n // 3)
    f = rng.random(n) < 0.2

    out_raw = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=24,
        n_perm=50,
        seed=11,
        mode="raw",
        smooth_w=1,
        donor_stratified=True,
        return_null_T=True,
        return_obs_profile=True,
    )
    out_sm1 = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=24,
        n_perm=50,
        seed=11,
        mode="smoothed",
        smooth_w=1,
        donor_stratified=True,
        return_null_T=True,
        return_obs_profile=True,
    )
    assert np.isclose(
        float(out_raw["T_obs"]), float(out_sm1["T_obs"]), atol=1e-12, rtol=0.0
    )
    assert float(out_raw["p_T"]) == float(out_sm1["p_T"])
    assert np.array_equal(np.asarray(out_raw["null_T"]), np.asarray(out_sm1["null_T"]))

    chk = check_mode_consistency(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=24,
        n_perm=50,
        seed=11,
        donor_stratified=True,
    )
    assert chk["T_abs_diff"] < 1e-10
    assert chk["p_abs_diff"] == 0.0


def test_max_operator_applied_inside_each_permutation():
    rng = np.random.default_rng(23)
    n = 180
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    donor_ids = np.repeat(np.array(["d1", "d2", "d3"]), n // 3)
    f = rng.random(n) < 0.3

    out = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=30,
        n_perm=70,
        seed=99,
        donor_stratified=True,
        return_null_T=True,
        return_obs_profile=True,
        return_null_profiles=True,
    )
    e_obs = np.asarray(out["E_phi_obs"], dtype=float)
    null_e = np.asarray(out["null_E_phi"], dtype=float)
    null_t = np.asarray(out["null_T"], dtype=float)

    assert np.isclose(
        float(out["T_obs"]), float(np.max(np.abs(e_obs))), atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        null_t, np.max(np.abs(null_e), axis=1), atol=1e-12, rtol=0.0
    )


def test_no_cross_gene_null_reuse_when_input_changes():
    n = 160
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    donor_ids = np.repeat(np.array(["d1", "d2"]), n // 2)
    f1 = np.zeros(n, dtype=bool)
    f1[:32] = True
    f2 = np.zeros(n, dtype=bool)
    f2[20:76] = True

    out1 = perm_null_T(
        f=f1,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=24,
        n_perm=80,
        seed=11,
        donor_stratified=True,
        return_null_T=True,
    )
    out2 = perm_null_T(
        f=f2,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=24,
        n_perm=80,
        seed=11,
        donor_stratified=True,
        return_null_T=True,
    )

    assert str(out1["stratified_counts_signature_hash"]) != str(
        out2["stratified_counts_signature_hash"]
    )
    assert not np.array_equal(np.asarray(out1["null_T"]), np.asarray(out2["null_T"]))


def test_recompute_same_gene_with_different_seed_changes_null():
    rng = np.random.default_rng(101)
    n = 150
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    donor_ids = np.repeat(np.array(["d1", "d2", "d3"]), n // 3)
    f = rng.random(n) < 0.25

    out_a = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=20,
        n_perm=60,
        seed=100,
        donor_stratified=True,
        return_null_T=True,
    )
    out_b = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=20,
        n_perm=60,
        seed=101,
        donor_stratified=True,
        return_null_T=True,
    )

    assert not np.array_equal(np.asarray(out_a["null_T"]), np.asarray(out_b["null_T"]))


def test_plus_one_pvalue_positive_and_lower_bound():
    rng = np.random.default_rng(303)
    n = 120
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    donor_ids = np.repeat(np.array(["d1", "d2"]), n // 2)
    f = rng.random(n) < 0.2
    n_perm = 37

    out = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=18,
        n_perm=n_perm,
        seed=7,
        donor_stratified=True,
        return_null_T=True,
    )
    p_t = float(out["p_T"])
    p_min = 1.0 / (n_perm + 1)
    assert p_t > 0.0
    assert p_t >= (p_min - 1e-12)
