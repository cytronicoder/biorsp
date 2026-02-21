import numpy as np

from biorsp.permutation import mode_max_stat_from_profiles, perm_null_T
from biorsp.smoothing import circular_moving_average
from experiments.simulations.expB_maxstat_sensitivity.cell_runner_b import (
    EdgeRuleConfigB,
    apply_foreground_edge_rule_b,
    simulate_run_context_b,
)


def test_max_stat_applied_inside_each_permutation_b():
    rng = np.random.default_rng(77)
    n = 180
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    donor_ids = np.repeat(np.array(["d1", "d2", "d3"]), n // 3)
    f = rng.random(n) < 0.28

    out = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=36,
        n_perm=64,
        seed=1234,
        mode="smoothed",
        smooth_w=5,
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


def test_smoothed_pipeline_matches_raw_profiles_plus_mode_transform():
    rng = np.random.default_rng(9)
    n = 150
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    donor_ids = np.repeat(np.array(["d1", "d2", "d3"]), n // 3)
    f = rng.random(n) < 0.33

    raw = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=30,
        n_perm=50,
        seed=222,
        mode="raw",
        smooth_w=1,
        donor_stratified=True,
        return_null_T=True,
        return_obs_profile=True,
        return_null_profiles=True,
    )
    transformed = mode_max_stat_from_profiles(
        E_obs_raw=np.asarray(raw["E_phi_obs"], dtype=float),
        null_E_raw=np.asarray(raw["null_E_phi"], dtype=float),
        mode="smoothed",
        smooth_w=5,
    )
    direct = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=30,
        n_perm=50,
        seed=222,
        mode="smoothed",
        smooth_w=5,
        donor_stratified=True,
        return_null_T=True,
        return_obs_profile=True,
        return_null_profiles=True,
    )
    np.testing.assert_allclose(
        np.asarray(transformed["E_phi_obs"], dtype=float),
        np.asarray(direct["E_phi_obs"], dtype=float),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(transformed["null_T"], dtype=float),
        np.asarray(direct["null_T"], dtype=float),
        atol=1e-12,
        rtol=0.0,
    )
    assert float(transformed["p_T"]) == float(direct["p_T"])


def test_circular_smoothing_wraparound_and_window_effect():
    x = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    y = circular_moving_average(x, 3)
    expected = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0], dtype=float)
    np.testing.assert_allclose(y, expected, atol=1e-12, rtol=0.0)

    rng = np.random.default_rng(14)
    n = 120
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    donor_ids = np.repeat(np.array(["d1", "d2"]), n // 2)
    f = rng.random(n) < 0.25
    out_w1 = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=24,
        n_perm=40,
        seed=44,
        mode="smoothed",
        smooth_w=1,
        donor_stratified=True,
        return_null_T=True,
    )
    out_w5 = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=24,
        n_perm=40,
        seed=44,
        mode="smoothed",
        smooth_w=5,
        donor_stratified=True,
        return_null_T=True,
    )
    assert not np.array_equal(
        np.asarray(out_w1["null_T"]), np.asarray(out_w5["null_T"])
    )


def test_no_cross_gene_null_reuse_b():
    n = 140
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    donor_ids = np.repeat(np.array(["d1", "d2"]), n // 2)
    f_a = np.zeros(n, dtype=bool)
    f_b = np.zeros(n, dtype=bool)
    f_a[:30] = True
    f_b[20:70] = True

    out_a = perm_null_T(
        f=f_a,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=24,
        n_perm=60,
        seed=101,
        donor_stratified=True,
        return_null_T=True,
    )
    out_b = perm_null_T(
        f=f_b,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=24,
        n_perm=60,
        seed=101,
        donor_stratified=True,
        return_null_T=True,
    )
    assert not np.array_equal(np.asarray(out_a["null_T"]), np.asarray(out_b["null_T"]))


def test_same_gene_different_seed_changes_null_b():
    rng = np.random.default_rng(48)
    n = 132
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    donor_ids = np.repeat(np.array(["d1", "d2", "d3"]), n // 3)
    f = rng.random(n) < 0.3

    out_1 = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=28,
        n_perm=55,
        seed=500,
        donor_stratified=True,
        return_null_T=True,
    )
    out_2 = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=28,
        n_perm=55,
        seed=501,
        donor_stratified=True,
        return_null_T=True,
    )
    assert not np.array_equal(np.asarray(out_1["null_T"]), np.asarray(out_2["null_T"]))


def test_plus_one_lower_bound_b():
    rng = np.random.default_rng(88)
    n = 120
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    donor_ids = np.repeat(np.array(["d1", "d2"]), n // 2)
    f = rng.random(n) < 0.22
    n_perm = 41
    out = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=20,
        n_perm=n_perm,
        seed=12,
        donor_stratified=True,
        return_null_T=True,
    )
    p_t = float(out["p_T"])
    p_min = 1.0 / (n_perm + 1)
    assert p_t > 0.0
    assert p_t >= (p_min - 1e-12)


def test_edge_rule_trigger_and_metadata_b():
    f = np.zeros(100, dtype=bool)
    f[:85] = True
    cfg = EdgeRuleConfigB(threshold=0.8, strategy="complement")
    f_test, info = apply_foreground_edge_rule_b(f, pi_target=0.9, cfg=cfg)
    assert bool(info["prevalence_edge_triggered"]) is True
    assert str(info["fg_rule_applied"]) == "complement"
    assert "pi_target" in list(info["edge_trigger_reasons"])
    assert "fg_fraction" in list(info["edge_trigger_reasons"])
    assert int(np.sum(f_test)) == 15


def test_strict_null_qc_independence_near_zero_b():
    _, qc = simulate_run_context_b(
        geometry="density_gradient_disk",
        n_cells=4000,
        d_value=10,
        sigma_eta=0.4,
        bins_b=72,
        seed_run=123,
    )
    assert np.isfinite(float(qc["max_abs_corr"]))
    assert float(qc["max_abs_corr"]) < 0.06
