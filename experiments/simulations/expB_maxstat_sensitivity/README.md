# Experiment B: Max-Statistic Sensitivity

This simulation checks calibration of permutation p-values for
`T = max_theta |E(theta)|` across binning/smoothing choices and embedding geometries.

## Commands

Full run:
```bash
python experiments/simulations/expB_maxstat_sensitivity/run_expB_maxstat_sensitivity.py \
  --outdir experiments/simulations/expB_maxstat_sensitivity
```

Test mode (tiny CI/local sanity check):
```bash
python experiments/simulations/expB_maxstat_sensitivity/run_expB_maxstat_sensitivity.py \
  --test_mode --outdir experiments/simulations/expB_maxstat_sensitivity
```

`test_mode` writes to `<outdir>/test_mode/` and is intentionally noisy due to small `n_perm` and gene counts.

Each run also writes `REPORT.md` with calibration tables, plot embeds, and critical interpretation.
