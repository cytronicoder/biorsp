# Experiment C: Power Surfaces + Abstention Boundaries

This simulation measures BioRSP power for angular anisotropy signals under donor-aware permutation testing using:
`T = max_theta |E(theta)|`.

## Full-scale command (spec)
```bash
python experiments/simulations/expC_power_surfaces/run_expC_power_surfaces.py \
  --outdir experiments/simulations/expC_power_surfaces \
  --master_seed 123 --n_perm 500 --bins 36 --genes_per_condition 200
```

## Practical local run (used in this workspace)
```bash
python experiments/simulations/expC_power_surfaces/run_expC_power_surfaces.py \
  --outdir experiments/simulations/expC_power_surfaces \
  --master_seed 123 --n_perm 120 --min_perm 120 --bins 36 \
  --genes_per_condition 2 --n_master_seeds 1 --beta_grid 0.0,0.5,1.0
```

Outputs are written to:
- `config.json`
- `results/metrics_long.csv`
- `results/summary.csv`
- `plots/*.png`
