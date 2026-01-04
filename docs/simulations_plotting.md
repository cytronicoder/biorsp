# Plot simulation outputs from CSVs ✅

New helper script to regenerate figures from simulation CSV tables (Phase 3 outputs):

- Script: `scripts/plot_simulation_csv.py`
- Module: `case_studies.simulations.plot_from_csv`

Usage example:

```bash
python3 scripts/plot_simulation_csv.py \
  --input-dir results/simulations_phase3 \
  --outdir results/simulations_phase3/figures_from_csv \
  --which all
```

The script looks for common tables under `results/simulations_phase3/tables/` such as `calibration_summary.csv`, `power_vs_N.csv`, `param_sweep_runs.csv`, `baseline_comparison.csv`, `type_separability.csv`, and `failure_modes_runs.csv` and generates PNG/PDF pairs with standardized axes, labels, and file formats. If a CSV is missing, the script prints a message and continues.
