# Experiment E: Gradient vs Step (BioRSP vs DE)

Simulates two matched-effect settings to compare BioRSP anisotropy scoring against Moran's I and cluster-DE proxy scores:
- `S1_step`: two-island cluster-separable step change
- `S2_gradient`: single-manifold continuous angular gradient

## Full-scale command (spec)
```bash
python3 experiments/simulations/expE_gradient_vs_step_DE/run_expE_gradient_vs_step_DE.py \
  --outdir experiments/simulations/expE_gradient_vs_step_DE \
  --master_seed 123 --N 20000 --D_grid 5 10 --sigma_eta_grid 0.0 0.4 \
  --pi_grid 0.01 0.05 0.2 0.6 --beta_grid 0.0 0.5 1.0 1.25 \
  --genes_per_condition 200 --n_perm 500 --bins 36 --k_nn 15
```

## Fast deterministic sanity run
```bash
python3 experiments/simulations/expE_gradient_vs_step_DE/run_expE_gradient_vs_step_DE.py \
  --test_mode --outdir /tmp/expE
```

Outputs:
- `config.json`
- `results/metrics_long.csv`
- `results/summary.csv`
- `results/validation_debug_report.txt`
- `plots/*.png`
