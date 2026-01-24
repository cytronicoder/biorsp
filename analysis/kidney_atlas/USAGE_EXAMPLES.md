# Kidney atlas usage examples

This page provides minimal commands for the kidney runners. Replace `/path/to/kpmp.h5ad` with your local file.

## Example 1: Disease-stratified analysis (all cells)

```bash
python analysis/kidney_atlas/runners/run_disease_stratified_analysis.py \
  --ref_data /path/to/kpmp.h5ad \
  --outdir results/disease_stratified \
  --max_genes 200
```

## Example 2: Disease-stratified analysis for a cell type

```bash
python analysis/kidney_atlas/runners/run_disease_stratified_analysis.py \
  --ref_data /path/to/kpmp.h5ad \
  --outdir results/disease_tal \
  --celltype_key subclass.l1 \
  --celltype_filter TAL \
  --max_genes 200
```

## Example 3: TAL analysis with controls

```bash
python analysis/kidney_atlas/runners/run_tal_analysis.py \
  --ref_data /path/to/kpmp.h5ad \
  --outdir results/tal \
  --controls "SLC12A1,UMOD,EGF" \
  --max_genes 200
```

## Example 4: All-genes archetype summary

```bash
python analysis/kidney_atlas/runners/run_kpmp_archetypes_all_genes.py \
  --h5ad /path/to/kpmp.h5ad \
  --outdir results/kpmp_all_genes \
  --max-cells 100000
```

## Metadata expectations

The disease stratified runner searches common disease keys such as `disease_category`, `disease`, and `disease_state`. Cell-type filtering requires a column name supplied via `--celltype_key`.
