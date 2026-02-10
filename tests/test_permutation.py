import numpy as np

from biorsp.permutation import permute_foreground_within_donor


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
