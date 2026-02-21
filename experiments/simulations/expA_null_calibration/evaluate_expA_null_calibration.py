#!/usr/bin/env python3
"""Recompute Experiment A evaluation artifacts from existing metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.simulations._shared.io import atomic_write_csv
from experiments.simulations.expA_null_calibration.run_expA_null_calibration import (
    _format_prevalence_label,
    bh_feasibility_metrics,
    make_plots,
    summarize,
    validate_pvalue_floor,
    write_report_markdown,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute ExpA summary/report artifacts from existing metrics_long.csv."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/simulations/expA_null_calibration",
        help="ExpA output directory containing config.json and results/metrics_long.csv.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    outdir = Path(args.outdir)
    config_path = outdir / "config.json"
    results_dir = outdir / "results"
    plots_dir = outdir / "plots"
    metrics_path = results_dir / "metrics_long.csv"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)

    n_perm = int(cfg.get("n_perm", 0))
    m_full_tests = int(cfg.get("G", 0))
    if n_perm <= 0 or m_full_tests <= 0:
        raise ValueError("config.json must include positive n_perm and G.")

    mt_cfg = dict(cfg.get("multiple_testing_validation", {}))
    bh_mode = str(mt_cfg.get("mode", cfg.get("bh_validation_mode", "panel_bh")))
    bh_panel_size = int(mt_cfg.get("bh_panel_size", cfg.get("bh_panel_size", 15)))
    bh_panel_strategy = str(
        mt_cfg.get(
            "bh_panel_strategy",
            cfg.get("bh_panel_strategy", "prevalence_stratified_fixed"),
        )
    )
    bh_panel_seed = int(mt_cfg.get("bh_panel_seed", cfg.get("master_seed", 123)))

    min_p = validate_pvalue_floor(metrics, n_perm=n_perm)
    full_bh = bh_feasibility_metrics(m_full=m_full_tests, n_perm=n_perm)
    if not bool(full_bh["bh_feasible_q05"]):
        print(
            "WARNING: Full-BH q=0.05 is infeasible under current n_perm/G; "
            "full-BH metric will be suppressed."
        )

    summary, ks_df, panel_df = summarize(
        metrics,
        n_perm=n_perm,
        m_full_tests=m_full_tests,
        bh_validation_mode=bh_mode,
        bh_panel_size=bh_panel_size,
        bh_panel_strategy=bh_panel_strategy,
        bh_panel_seed=bh_panel_seed,
    )

    atomic_write_csv(results_dir / "summary_by_bin.csv", summary)
    atomic_write_csv(results_dir / "summary.csv", summary)
    atomic_write_csv(results_dir / "ks_uniformity.csv", ks_df)
    atomic_write_csv(results_dir / "bh_panel_validation.csv", panel_df)

    prevalence_labels = [
        _format_prevalence_label(float(x)) for x in cfg.get("prevalence_bins", [])
    ]
    donor_grid = [int(x) for x in cfg.get("donor_grid", [])]
    sigma_eta_grid = [float(x) for x in cfg.get("sigma_eta_grid", [])]
    test_mode = bool(cfg.get("test_mode", False))
    if prevalence_labels and donor_grid and sigma_eta_grid:
        make_plots(
            metrics,
            summary,
            ks_df,
            plots_dir=plots_dir,
            prevalence_labels=prevalence_labels,
            donor_grid=donor_grid,
            sigma_eta_grid=sigma_eta_grid,
            test_mode=test_mode,
        )

    report_path = write_report_markdown(outdir, results_dir)
    print(
        f"Recomputed ExpA artifacts: min_p={min_p:.6g}, "
        f"summary={results_dir / 'summary_by_bin.csv'}, report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
