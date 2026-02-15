# Quickstart

This is the canonical command path for the heart case study.

## 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 2) Required Input Fields

Expected `.h5ad` fields:

- `adata.obsm["X_umap"]` (2D embedding)
- donor labels in `adata.obs` (preferred: `hubmap_id`)
- cluster labels in `adata.obs` (preferred: `azimuth_id`)
- cell type labels in `adata.obs` (preferred: `azimuth_label`)
- `total_counts` and `pct_counts_mt` in `adata.obs` (recommended for QC diagnostics)

Example input path:

- `data/processed/HT_pca_umap.h5ad`

## 3) Run The Heart Case Study

```bash
PYTHONPATH=. python3 analysis/heart_case_study/run.py \
  --h5ad data/processed/HT_pca_umap.h5ad \
  --out outputs/heart_case_study/run1 \
  --donor_key hubmap_id \
  --cluster_key azimuth_id \
  --celltype_key azimuth_label \
  --do_hierarchy true
```

## 4) Minimal CI Sanity Check (Optional)

```bash
PYTHONPATH=. python3 experiments/heart_smoketest/run_heart_smoketest.py \
  --h5ad data/processed/HT_pca_umap.h5ad \
  --out experiments/heart_smoketest/outputs/heart_ci_sanity \
  --do_hierarchy false
```

## 5) Expected Top-Level Outputs

```text
outputs/heart_case_study/run1/
  metadata.json
  run_metadata.json
  logs/runlog.md
  tables/
  plots/
  hierarchy/
```
