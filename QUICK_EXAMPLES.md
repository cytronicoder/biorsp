# Quick examples for figure generation and smoke tests

This document provides minimal commands to regenerate figures and run smoke tests used during development. Use these examples to quickly validate plotting and benchmark workflows.

## 1. CLI: Regenerate figures (fast, no simulation)

Use the `make_figures` CLI to regenerate panels from precomputed outputs.

```bash
# Regenerate panel 'A' only
python -m biorsp.plotting.make_figures --indir test_output/smoke_test --panels A

# Regenerate all panels
python -m biorsp.plotting.make_figures --indir test_output/smoke_test

# Regenerate with debug plots
python -m biorsp.plotting.make_figures --indir test_output/smoke_test --debug

# Export as PDF
python -m biorsp.plotting.make_figures --indir test_output/smoke_test --format pdf
```

---

## 2. Simulation benchmarks (quick smoke runs)

Run lightweight benchmarks locally. These require the simulation code but not external data.

```bash
# Archetypes (quick smoke)
python analysis/benchmarks/runners/run_archetypes.py \
  --mode quick \
  --n_workers 1 \
  --outdir test_output/archetypes_smoke

# Calibration (quick smoke)
python analysis/benchmarks/runners/run_calibration.py \
  --mode quick \
  --n_workers 1 \
  --outdir test_output/calibration_smoke
```

---

## 3. Kidney analyses (require KPMP data)

These scripts operate on KPMP `.h5ad` files. Provide a local path to your data when running.

```bash
# TAL analysis (minimal)
python analysis/kidney_atlas/runners/run_tal_analysis.py \
  --ref_data /path/to/kpmp.h5ad \
  --outdir test_output/tal_smoke \
  --controls "SLC12A1,UMOD,EGF" \
  --max_genes 10 \
  --n_permutations 0

# Disease-stratified analysis (minimal)
python analysis/kidney_atlas/runners/run_disease_stratified_analysis.py \
  --ref_data /path/to/kpmp.h5ad \
  --outdir test_output/disease_smoke \
  --max_genes 10 \
  --n_permutations 0 \
  --smoke

# All-genes pipeline (minimal)
python analysis/kidney_atlas/runners/run_kpmp_archetypes_all_genes.py \
  --h5ad /path/to/kpmp.h5ad \
  --outdir test_output/all_genes_smoke \
  --max_cells 1000 \
  --min_coverage 0.1 \
  --n_permutations 0
```

> Note: replace `/path/to/kpmp.h5ad` with a valid local path to run these commands.

---

## 4. Verify outputs

```bash
# Confirm expected figure files exist in test outputs
find test_output/ -name "A_archetype_scatter.png" -exec ls -la {} \;
find test_output/ -name "fig_story_onepager.png" -exec ls -la {} \;
```

---

## 5. Story / one-pager generation

```bash
python -c "from biorsp.plotting.story import generate_onepager_from_dir; generate_onepager_from_dir('test_output/smoke_test')"
```

---

## 6. Run plotting standardization tests

```bash
python -m pytest tests/test_plot_standardization.py -v
```
