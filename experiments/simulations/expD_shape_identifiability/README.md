# Experiment D: Shape Identifiability

Simulates unimodal, bimodal, trimodal, and patchy 4-lobe genes and evaluates rule-based BioRSP shape classification under dropout/noise and bin/smoothing sensitivity.

## Full-scale command (spec)
```bash
python experiments/simulations/expD_shape_identifiability/run_expD_shape_identifiability.py \
  --outdir experiments/simulations/expD_shape_identifiability \
  --master_seed 123 --N 20000 --D 10 --n_perm 500 \
  --bins_grid 24 36 48 --w_grid 1 3 5 --modes raw smoothed \
  --pi_grid 0.05 0.2 --beta_grid 0.25 0.5 0.75 1.0 1.25 \
  --dropout_grid 0.0 0.1 0.2 0.3 --genes_per_class 200
```

## Practical run used in this workspace
```bash
python experiments/simulations/expD_shape_identifiability/run_expD_shape_identifiability.py \
  --outdir experiments/simulations/expD_shape_identifiability \
  --master_seed 123 --N 6000 --D 10 --n_perm 80 --min_perm 80 \
  --bins_grid 24 36 48 --w_grid 1 3 --modes raw smoothed \
  --pi_grid 0.05 0.2 --beta_grid 0.0 0.5 1.0 1.25 \
  --dropout_grid 0.0 0.1 0.2 0.3 --genes_per_class 1 \
  --sigma_eta_grid 0.0 0.4 --n_boot 3
```

Outputs:
- `config.json`
- `results/metrics_long.csv`
- `results/summary.csv`
- `results/validation_debug_report.txt`
- `confusion_matrices/*.csv`
- `plots/*.png`
