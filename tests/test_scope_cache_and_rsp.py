import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-test")

import numpy as np

from biorsp.permutation import perm_null_T_and_profile
from biorsp.rsp import compute_rsp_profile_from_boolean
from biorsp.scope_cache import build_or_load_scope_cache


def test_rsp_profile_bin_cache_parity():
    rng = np.random.default_rng(7)
    n_cells = 512
    bins = 36
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_cells)
    f = rng.random(n_cells) < 0.2

    E_ref, phi_ref, emax_ref = compute_rsp_profile_from_boolean(f, angles, bins)

    bin_width = (2.0 * np.pi) / bins
    bin_id = np.floor((angles % (2.0 * np.pi)) / bin_width).astype(int)
    bin_id = np.clip(bin_id, 0, bins - 1)
    bin_counts_total = np.bincount(bin_id, minlength=bins)
    E_new, phi_new, emax_new = compute_rsp_profile_from_boolean(
        f,
        angles,
        bins,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )

    np.testing.assert_allclose(E_ref, E_new, atol=1e-12, rtol=0.0)
    assert float(phi_ref) == float(phi_new)
    assert float(emax_ref) == float(emax_new)


def test_scope_cache_reuse_and_invalidation(tmp_path):
    rng = np.random.default_rng(11)
    n_cells = 120
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_cells)
    donor_ids = np.array(["d1"] * 60 + ["d2"] * 60)

    cache1 = build_or_load_scope_cache(
        scope_id="scopeA",
        angles=angles,
        donor_ids=donor_ids,
        bins=24,
        n_perm=50,
        seed=123,
        cache_dir=tmp_path,
    )
    assert cache1.loaded_from_disk is False

    cache2 = build_or_load_scope_cache(
        scope_id="scopeA",
        angles=angles,
        donor_ids=donor_ids,
        bins=24,
        n_perm=50,
        seed=123,
        cache_dir=tmp_path,
    )
    assert cache2.loaded_from_disk is True
    np.testing.assert_array_equal(cache1.bin_id, cache2.bin_id)
    np.testing.assert_array_equal(cache1.bin_counts_total, cache2.bin_counts_total)

    # Parameter change should invalidate cache.
    cache3 = build_or_load_scope_cache(
        scope_id="scopeA",
        angles=angles,
        donor_ids=donor_ids,
        bins=25,
        n_perm=50,
        seed=123,
        cache_dir=tmp_path,
    )
    assert cache3.loaded_from_disk is False


def test_perm_observed_stat_parity_with_precomputed_scope_cache(tmp_path):
    rng = np.random.default_rng(19)
    n_cells = 200
    bins = 30
    angles = np.linspace(0.0, 2.0 * np.pi, n_cells, endpoint=False)
    donor_ids = np.array(["d1"] * 100 + ["d2"] * 100)
    expr = (rng.random(n_cells) < 0.2).astype(float)

    legacy = perm_null_T_and_profile(
        expr=expr,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=bins,
        n_perm=40,
        seed=5,
        donor_stratified=True,
    )

    cache = build_or_load_scope_cache(
        scope_id="scopeB",
        angles=angles,
        donor_ids=donor_ids,
        bins=bins,
        n_perm=40,
        seed=5,
        cache_dir=tmp_path,
    )
    cached = perm_null_T_and_profile(
        expr=expr,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=bins,
        n_perm=40,
        seed=5,
        donor_stratified=True,
        perm_indices=cache.perm_indices,
        perm_start=0,
        perm_end=40,
        bin_id=cache.bin_id,
        bin_counts_total=cache.bin_counts_total,
    )

    np.testing.assert_allclose(
        np.asarray(legacy["E_phi_obs"], dtype=float),
        np.asarray(cached["E_phi_obs"], dtype=float),
        atol=1e-12,
        rtol=0.0,
    )
    assert float(legacy["T_obs"]) == float(cached["T_obs"])
