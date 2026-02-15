import numpy as np

from biorsp.permutation import perm_null_T_and_profile, permute_foreground_within_donor


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
