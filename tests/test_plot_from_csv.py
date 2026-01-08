import warnings
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from case_studies.simulations import plot_from_csv

matplotlib.use("Agg")

pytestmark = pytest.mark.filterwarnings("ignore:vert:PendingDeprecationWarning")

warnings.filterwarnings(
    "ignore",
    message="vert: bool will be deprecated",
    category=PendingDeprecationWarning,
    module="seaborn.categorical",
)


def _touch_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_plot_functions_create_files(tmp_path):
    base = tmp_path / "results"
    tables = base / "tables"
    outdir = tmp_path / "figs"
    outdir.mkdir(parents=True, exist_ok=True)

    # calibration CSV
    cal = pd.DataFrame(
        {
            "expected": [0.01, 0.5, 0.99],
            "observed": [0.02, 0.47, 0.98],
            "p_value": [0.01, 0.5, 0.99],
        }
    )
    _touch_csv(tables / "calibration_summary.csv", cal)

    # power CSV
    power = pd.DataFrame(
        {
            "N": [50, 100, 200],
            "power": [0.1, 0.4, 0.8],
            "ci_low": [0.05, 0.35, 0.75],
            "ci_high": [0.15, 0.45, 0.85],
            "alpha": [0.05, 0.05, 0.05],
        }
    )
    _touch_csv(tables / "power_vs_N.csv", power)

    # robustness CSV
    robust = pd.DataFrame(
        {
            "param": ["theta_grid_size", "sector_width"],
            "value": [1, 2],
            "similarity": [0.9, 0.85],
            "type": ["A", "B"],
        }
    )
    _touch_csv(tables / "param_sweep_runs.csv", robust)

    # baselines CSV
    base_df = pd.DataFrame({"type": ["A", "B", "A"], "metric1": [0.1, 0.2, 0.15]})
    _touch_csv(tables / "baseline_comparison.csv", base_df)

    # separability CSV
    sep = pd.DataFrame({"fpr": [0, 0.1, 1.0], "tpr": [0, 0.7, 1.0]})
    _touch_csv(tables / "type_separability.csv", sep)

    # failure CSV
    fail = pd.DataFrame({"case": ["c1", "c2"], "abstain_flag": [0.1, 0.3]})
    _touch_csv(tables / "failure_modes_runs.csv", fail)

    # call main with base and outdir
    plot_from_csv.plot_calibration(tables / "calibration_summary.csv", outdir)
    plot_from_csv.plot_power_vs_N(tables / "power_vs_N.csv", outdir)
    plot_from_csv.plot_robustness(tables / "param_sweep_runs.csv", outdir)
    plot_from_csv.plot_baselines(tables / "baseline_comparison.csv", outdir)
    plot_from_csv.plot_separability(tables / "type_separability.csv", outdir)
    plot_from_csv.plot_failure_modes(tables / "failure_modes_runs.csv", outdir)

    # assert some files exist
    assert (outdir / "calibration_qq.png").exists()
    assert (outdir / "power_vs_N.png").exists()
    assert (outdir / "robustness_sensitivity.png").exists()
    assert (outdir / "baselines.png").exists()
    assert (outdir / "separability_roc.png").exists()
    assert (outdir / "failure_modes.png").exists()
