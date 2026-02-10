# BioRSP

Directional enrichment analysis for single-cell embeddings with donor-aware statistics.

## Install

```bash
pip install -e .
```

Optional dev tools:

```bash
pip install -e ".[dev]"
```

## Config format

Pipeline configs are strict JSON files (`.json`).

## Smoke CLIs

```bash
biorsp-smoke-rsp --h5ad adata_embed_graph.h5ad --outdir .
biorsp-smoke-moran --h5ad adata_embed_graph.h5ad --outdir .
biorsp-smoke-perm --h5ad adata_embed_graph.h5ad --outdir . --n-perm 100
```

Legacy `scripts/smoke_*.py` wrappers are deprecated and delegate to the canonical `biorsp-smoke-*` entrypoints.

Smoke outputs are written under `outputs/` (including `outputs/logs/runlog.md`).

## Preregistered evaluation pipeline

```bash
python scripts/prereg_pipeline.py --config configs/biorsp_prereg.json
```

Outputs:
- `outputs/results/biorsp_stratum_results.csv`
- `outputs/results/null_calibration.csv`
- `outputs/results/preprocessing_report.csv` (+ `.md`)
- Figures under `outputs/figures/<stratum>/` for Ventricular cardiomyocytes, Fibroblasts, Capillary EC
- Logs under `outputs/logs/biorsp_prereg.log`

## Genome-wide pipeline

```bash
python scripts/genomewide_pipeline.py --config configs/biorsp_genomewide.json
```

Outputs:
- `outputs/results/biorsp_genomewide_results.csv`
- `outputs/results/biorsp_genomewide_top_hits.csv`
- `outputs/results/gene_filtering.csv`
- Figures under `outputs/figures/genomewide/<stratum>/`
- Logs under `outputs/logs/biorsp_genomewide.log`

Generated artifacts (`outputs/`, `results/`, `figures/`, `logs/`) are intentionally not tracked.
Archived snapshots from local runs can be kept under `artifacts/` (also ignored).

## Repository Docs

Additional docs are organized under:
- `docs/`
- `paper/memos/`
- `notebooks/`

## Input requirements

The `.h5ad` input must contain:
- `adata.obsm["X_umap"]` for smoke tests
- `adata.obsp["connectivities"]` or allow neighbor recomputation
- `adata.obs["donor"]` (or configure `donor_col` in the prereg config)
- `adata.obs["cell_type"]` (or configure `celltype_col`)

## Tests

```bash
pytest -q
```

## Lint/Format

```bash
ruff check .
ruff format .
```

## Task runner

```bash
make install
make test
make lint
make format
make smoke H5AD=/path/to/input.h5ad
```
