# Heart Smoke Test (CI Sanity Only)

This path is a lightweight CI sanity check for the canonical heart case study.

Primary scientific workflow:

- `/Users/cytronicoder/Documents/GitHub/biorsp-topaz/analysis/heart_case_study/run.py`

## CI sanity run

```bash
PYTHONPATH=. python3 experiments/heart_smoketest/run_heart_smoketest.py \
  --h5ad data/processed/HT_pca_umap.h5ad \
  --out experiments/heart_smoketest/outputs/heart_ci_sanity \
  --do_hierarchy false \
  --n_perm 20 \
  --seed 0
```

`outputs/` contents are gitignored.
