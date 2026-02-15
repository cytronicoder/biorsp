# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests And Lint

```bash
pytest -q
ruff check .
ruff format .
```

## Canonical Docs

- User entrypoint: `/Users/cytronicoder/Documents/GitHub/biorsp-topaz/docs/01_quickstart.md`
- Method and inference: `/Users/cytronicoder/Documents/GitHub/biorsp-topaz/docs/02_method.md`, `/Users/cytronicoder/Documents/GitHub/biorsp-topaz/docs/03_inference_and_nulls.md`
- Developer guide: `/Users/cytronicoder/Documents/GitHub/biorsp-topaz/docs/06_developer_guide.md`

## Case Study Command

```bash
PYTHONPATH=. python3 analysis/heart_case_study/run.py \
  --h5ad data/processed/HT_pca_umap.h5ad \
  --out outputs/heart_case_study/dev_run \
  --donor_key hubmap_id \
  --cluster_key azimuth_id \
  --celltype_key azimuth_label \
  --do_hierarchy true
```
