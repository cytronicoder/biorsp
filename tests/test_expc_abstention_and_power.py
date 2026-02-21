import numpy as np
import pandas as pd

from analysis.expC_acceptance_checks import _compute_strict_null_table
from biorsp.scoring import evaluate_underpowered
from experiments.simulations.expC_power_surfaces.run_expC_power_surfaces import (
    summarize,
)


def test_deff_gate_triggers_underpowered() -> None:
    donor_ids = np.array(
        [
            "d1",
            "d1",
            "d1",
            "d1",
            "d2",
            "d2",
            "d2",
            "d2",
            "d3",
            "d3",
            "d3",
            "d3",
        ]
    )
    f = np.array(
        [
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
        ],
        dtype=bool,
    )

    gate = evaluate_underpowered(
        donor_ids=donor_ids,
        f=f,
        n_perm=300,
        p_min=0.0,
        min_fg_total=4,
        min_fg_per_donor=2,
        min_bg_per_donor=1,
        d_eff_min=3,
        min_perm=200,
    )
    assert gate["underpowered"] is True
    assert gate["underpowered_reasons"]["d_eff_lt_d_eff_min"] is True
    assert "D_eff_below_min" in gate["abstain_reasons"]
    assert "fg_per_donor_below_min" in gate["abstain_reasons"]


def test_dual_power_formulas() -> None:
    rows = []
    for i in range(10):
        under = i < 2
        if under:
            p_t = np.nan
        else:
            p_t = 0.01 if i in {2, 3, 4} else 0.2
        rows.append(
            {
                "N": 5000,
                "D": 5,
                "sigma_eta": 0.4,
                "pi_target": 0.05,
                "beta": 1.0,
                "underpowered": under,
                "p_T": p_t,
                "prev_obs": 0.05,
                "D_eff": 5,
            }
        )
    metrics = pd.DataFrame(rows)
    out = summarize(metrics)
    assert out.shape[0] == 1
    row = out.iloc[0]

    assert np.isclose(row["analyzable_rate"], 0.8)
    assert np.isclose(row["conditional_power_alpha05"], 3.0 / 8.0)
    assert np.isclose(row["operational_power_alpha05"], 3.0 / 10.0)

    for col in [
        "analyzable_rate_ci_low",
        "analyzable_rate_ci_high",
        "conditional_power_alpha05_ci_low",
        "conditional_power_alpha05_ci_high",
        "operational_power_alpha05_ci_low",
        "operational_power_alpha05_ci_high",
    ]:
        assert col in out.columns
        assert 0.0 <= float(row[col]) <= 1.0

    # Backward compatibility aliases should remain populated for beta>0.
    assert np.isclose(row["power_alpha05"], row["conditional_power_alpha05"])


def test_strict_null_acceptance_flagging() -> None:
    base = pd.DataFrame(
        {
            "N": [5000] * 10,
            "D": [10] * 10,
            "sigma_eta": [0.0] * 10,
            "pi_target": [0.01] * 10,
            "beta": [0.0] * 10,
            "underpowered": [False] * 10,
            "p_T": [0.001] * 10,
        }
    )
    strict = _compute_strict_null_table(base, n_min=5, tol=0.01)
    assert strict.shape[0] == 1
    assert np.isclose(strict.iloc[0]["typeI_alpha05_conditional"], 1.0)
    assert bool(strict.iloc[0]["typeI_violation"]) is True

    sparse = base.copy()
    sparse.loc[:7, "underpowered"] = True
    sparse.loc[:7, "p_T"] = np.nan
    strict_sparse = _compute_strict_null_table(sparse, n_min=5, tol=0.01)
    assert strict_sparse.shape[0] == 1
    assert int(strict_sparse.iloc[0]["n_analyzable"]) == 2
    assert bool(strict_sparse.iloc[0]["typeI_violation"]) is False
