import numpy as np

from biorsp.genomewide import permute_values_within_donor


def test_permute_values_within_donor_preserves_multiset():
    values = np.array([1, 2, 3, 4, 10, 20, 30, 40], dtype=float)
    donor_to_idx = {"d1": np.array([0, 1, 2, 3]), "d2": np.array([4, 5, 6, 7])}
    rng = np.random.default_rng(0)
    perm = permute_values_within_donor(values, donor_to_idx, rng)
    assert sorted(perm[donor_to_idx["d1"]]) == sorted(values[donor_to_idx["d1"]])
    assert sorted(perm[donor_to_idx["d2"]]) == sorted(values[donor_to_idx["d2"]])
