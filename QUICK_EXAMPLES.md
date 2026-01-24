# Quick examples for smoke runs

This file lists minimal commands for checking that benchmark and kidney workflows execute and emit standard outputs.

## 1. Benchmark runners (quick mode)

```bash
python analysis/benchmarks/runners/run_archetypes.py \
  --mode quick \
  --n_workers 1 \
  --outdir test_output/benchmarks

python analysis/benchmarks/runners/run_calibration.py \
  --mode quick \
  --n_workers 1 \
  --outdir test_output/benchmarks
```

Each run creates `test_output/benchmarks/<benchmark>/<run_id>/` containing `runs.csv`, `summary.csv`, `manifest.json`, `report.md`, and `figures/`.

## 2. Kidney workflows (requires KPMP data)

```bash
python analysis/kidney_atlas/runners/run_tal_analysis.py \
  --ref_data /path/to/kpmp.h5ad \
  --outdir test_output/kidney_tal \
  --controls "SLC12A1,UMOD,EGF" \
  --max_genes 10 \
  --n_permutations 0

python analysis/kidney_atlas/runners/run_disease_stratified_analysis.py \
  --ref_data /path/to/kpmp.h5ad \
  --outdir test_output/kidney_disease \
  --max_genes 10 \
  --n_permutations 0 \
  --smoke

python analysis/kidney_atlas/runners/run_kpmp_archetypes_all_genes.py \
  --h5ad /path/to/kpmp.h5ad \
  --outdir test_output/kidney_all_genes \
  --max-cells 1000 \
  --min-coverage 0.1 \
  --n-permutations 0
```

## 3. Check expected artifacts

```bash
find test_output/ -name "runs.csv" -exec ls -la {} \;
find test_output/ -name "summary.csv" -exec ls -la {} \;
find test_output/ -name "fig_cs_scatter.png" -exec ls -la {} \;
```
