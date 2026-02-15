# BioRSP

BioRSP is focused on a single heart case study for representation-conditional directional enrichment on single-cell UMAP embeddings.

This repository is intentionally scoped to **method validation**, not subtype discovery.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run The Heart Case Study

```bash
PYTHONPATH=. python3 analysis/heart_case_study/run.py \
  --h5ad data/processed/HT_pca_umap.h5ad \
  --out outputs/heart_case_study/run1 \
  --donor_key hubmap_id \
  --cluster_key azimuth_id \
  --celltype_key azimuth_label \
  --do_hierarchy true
```

## Read The Docs

- Overview: `/Users/cytronicoder/Documents/GitHub/biorsp-topaz/docs/00_overview.md`
- Quickstart: `/Users/cytronicoder/Documents/GitHub/biorsp-topaz/docs/01_quickstart.md`
- Method: `/Users/cytronicoder/Documents/GitHub/biorsp-topaz/docs/02_method.md`
- Inference and nulls: `/Users/cytronicoder/Documents/GitHub/biorsp-topaz/docs/03_inference_and_nulls.md`
- Limitations and controls: `/Users/cytronicoder/Documents/GitHub/biorsp-topaz/docs/04_limitations_and_negative_controls.md`
- Outputs and reproducibility: `/Users/cytronicoder/Documents/GitHub/biorsp-topaz/docs/05_outputs_and_reproducibility.md`
- Developer guide: `/Users/cytronicoder/Documents/GitHub/biorsp-topaz/docs/06_developer_guide.md`

## License

MIT (`/Users/cytronicoder/Documents/GitHub/biorsp-topaz/LICENSE`).
