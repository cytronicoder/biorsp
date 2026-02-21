import numpy as np

from experiments.simulations.expA_null_calibration.cell_runner import (
    EdgeRuleConfig,
    apply_foreground_edge_rule,
    make_cell_key,
    simulate_run_context,
)


def test_make_cell_key_format():
    key = make_cell_key(0.9, 2, 0.0)
    assert key == "prevalence_bin=0.9|D=2|sigma_eta=0"


def test_qc_independence_near_zero_for_strict_null_context():
    _, qc = simulate_run_context(
        n_cells=5000,
        n_bins=36,
        n_donors=5,
        sigma_eta=0.4,
        seed_run=844899,
        mu0=10.0,
        sigma_l=0.5,
    )
    assert np.isfinite(float(qc["max_abs_corr"]))
    assert float(qc["max_abs_corr"]) < 0.03


def test_prevalence_edge_rule_complement_triggered():
    f_raw = np.zeros(100, dtype=bool)
    f_raw[:90] = True
    cfg = EdgeRuleConfig(threshold=0.8, strategy="complement")

    f_test, info = apply_foreground_edge_rule(f_raw, prevalence_bin=0.9, cfg=cfg)

    assert bool(info["prevalence_edge_triggered"]) is True
    assert str(info["fg_rule_applied"]) == "complement"
    assert "prevalence_bin" in list(info["edge_trigger_reasons"])
    assert "fg_fraction" in list(info["edge_trigger_reasons"])
    assert int(np.sum(f_test)) == 10


def test_prevalence_edge_rule_no_trigger_when_below_threshold():
    rng = np.random.default_rng(9)
    f_raw = rng.random(120) < 0.5
    cfg = EdgeRuleConfig(threshold=0.8, strategy="complement")

    f_test, info = apply_foreground_edge_rule(f_raw, prevalence_bin=0.2, cfg=cfg)

    assert bool(info["prevalence_edge_triggered"]) is False
    assert str(info["fg_rule_applied"]) == "none"
    np.testing.assert_array_equal(f_test, f_raw)
