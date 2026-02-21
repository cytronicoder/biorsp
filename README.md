# BioRSP

BioRSP is focused on a single heart case study for representation-conditional directional enrichment on single-cell UMAP embeddings.

This repository is intentionally scoped to **method validation**, not subtype discovery.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Minimal import check:

```bash
python -c "import biorsp; print(getattr(biorsp, '__version__', 'ok'))"
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

## Run The Heart Smoketest

Show options:

```bash
python experiments/heart_smoketest/run_heart_smoketest.py --help
```

Example invocation:

```bash
python experiments/heart_smoketest/run_heart_smoketest.py \
  --h5ad data/processed/HT_pca_umap.h5ad \
  --out experiments/heart_smoketest/outputs/heart_ci_sanity
```

## Run Simulations

Dry-run full suite orchestration:

```bash
python experiments/simulations/run_all.py --dry_run
```

Smoke profile batch run:

```bash
python experiments/simulations/run_all.py \
  --profile smoke \
  --n_jobs 8 \
  --continue_on_error
```

Canonical config source for simulation runs is `experiments/simulations/configs/`.

## Output Paths

Simulation outputs: `experiments/simulations/_results/`

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
