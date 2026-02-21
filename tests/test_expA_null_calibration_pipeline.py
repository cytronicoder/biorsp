from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from biorsp.scoring import bh_fdr
from experiments.simulations.expA_null_calibration.cell_runner import (
    EdgeRuleConfig,
    GateConfig,
    run_cell_null_replay,
)
from experiments.simulations.expA_null_calibration.run_expA_null_calibration import (
    bh_feasibility_metrics,
    summarize_by_bin,
)


def _toy_metrics_df() -> pd.DataFrame:
    pvals = np.array([0.01, 0.02, 0.05, 0.10, 0.30, 0.80], dtype=float)
    qvals = bh_fdr(pvals)
    return pd.DataFrame(
        {
            "D": [5] * pvals.size,
            "sigma_eta": [0.4] * pvals.size,
            "prevalence_bin": ["0.2"] * pvals.size,
            "prev_obs": np.full(pvals.size, 0.2, dtype=float),
            "underpowered": [False] * pvals.size,
            "D_eff": np.full(pvals.size, 5, dtype=float),
            "p_T": pvals,
            "q_T": qvals,
        }
    )


def test_expA_bh_feasibility_flags() -> None:
    feas = bh_feasibility_metrics(m_full=2000, n_perm=300)
    assert np.isclose(float(feas["min_attainable_p"]), 1.0 / 301.0, atol=1e-12)
    assert np.isclose(float(feas["bh_min_rejectable_p_q05"]), 0.05 / 2000.0, atol=1e-12)
    assert np.isclose(float(feas["bh_min_rejectable_p_q10"]), 0.10 / 2000.0, atol=1e-12)
    assert bool(feas["bh_feasible_q05"]) is False
    assert bool(feas["bh_feasible_q10"]) is False


def test_expA_p_floor_invariants() -> None:
    n_perm = 30
    metrics, _, _ = run_cell_null_replay(
        n_cells=600,
        g_total=60,
        n_bins=18,
        n_perm=n_perm,
        prevalence=0.2,
        donor_count=5,
        sigma_eta=0.4,
        prevalence_grid=[0.01, 0.2, 0.9],
        gate_cfg=GateConfig(
            p_min=0.005,
            min_fg_total=10,
            min_fg_per_donor=2,
            min_bg_per_donor=2,
            d_eff_min=2,
            min_perm=20,
        ),
        edge_cfg=EdgeRuleConfig(threshold=0.8, strategy="complement"),
        mu0=10.0,
        sigma_l=0.5,
        master_seed=123,
        n_genes_subset=30,
        include_null_t_values=False,
    )
    assert not metrics.empty
    pvals = pd.to_numeric(metrics["p_T"], errors="coerce").to_numpy(dtype=float)
    pvals = pvals[np.isfinite(pvals)]
    assert pvals.size > 0
    assert not np.any(np.isclose(pvals, 0.0, atol=1e-12, rtol=0.0))
    expected_floor = 1.0 / (n_perm + 1)
    min_p = float(np.min(pvals))
    assert min_p >= expected_floor - 1e-12
    assert np.isclose(min_p, expected_floor, atol=1e-12, rtol=0.0) or (
        min_p > expected_floor
    )


def test_expA_tail_columns_present_numeric() -> None:
    summary = summarize_by_bin(_toy_metrics_df(), n_perm=300, m_full_tests=2000)
    row = summary.iloc[0]
    required_cols = [
        "typeI_p01",
        "typeI_p01_ci_low",
        "typeI_p01_ci_high",
        "typeI_p005",
        "typeI_p005_ci_low",
        "typeI_p005_ci_high",
        "typeI_p00333",
        "typeI_p00333_ci_low",
        "typeI_p00333_ci_high",
        "tail_inflation_flag_p05",
        "tail_inflation_flag_p01",
        "tail_inflation_flag_p005",
        "m_full_tests",
        "min_attainable_p",
        "bh_min_rejectable_p_q05",
        "bh_min_rejectable_p_q10",
        "bh_feasible_q05",
        "bh_feasible_q10",
        "se_typeI_p05",
        "n_required_pm01_95ci",
        "mc_sufficient_pm01_95ci",
    ]
    for col in required_cols:
        assert col in summary.columns
    assert np.isfinite(float(row["typeI_p01"]))
    assert isinstance(bool(row["tail_inflation_flag_p05"]), bool)


def test_expA_bh_suppressed_when_infeasible() -> None:
    summary = summarize_by_bin(_toy_metrics_df(), n_perm=300, m_full_tests=2000)
    row = summary.iloc[0]
    assert np.isnan(float(row["typeI_q05"]))
    assert str(row["typeI_q05_status"]) == "BH infeasible; metric suppressed"


def test_expA_smoke_run_tiny_ci(tmp_path: Path) -> None:
    script = Path(
        "experiments/simulations/expA_null_calibration/run_expA_null_calibration.py"
    )
    outdir = tmp_path / "expa_smoke"
    cmd = [
        sys.executable,
        str(script),
        "--outdir",
        str(outdir),
        "--N",
        "400",
        "--G",
        "21",
        "--bins",
        "18",
        "--n_perm",
        "25",
        "--donor_grid",
        "2,5",
        "--sigma_eta_grid",
        "0.0",
        "--prevalence_bins",
        "0.01,0.2,0.9",
        "--progress_every",
        "9999",
        "--master_seed",
        "123",
        "--bh_validation_mode",
        "panel_bh",
        "--bh_panel_size",
        "7",
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr + "\n" + completed.stdout

    runs_dir = outdir / "runs"
    run_dirs = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    )
    assert run_dirs
    run_dir = run_dirs[-1]
    results_dir = run_dir / "results"
    assert (results_dir / "summary_by_bin.csv").exists()
    assert (results_dir / "summary.csv").exists()
    assert (results_dir / "ks_uniformity.csv").exists()
    assert (results_dir / "bh_panel_validation.csv").exists()
    assert (run_dir / "REPORT.md").exists()

    summary = pd.read_csv(results_dir / "summary_by_bin.csv")
    assert "typeI_p01" in summary.columns
    assert "bh_feasible_q05" in summary.columns
