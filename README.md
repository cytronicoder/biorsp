# BioRSP

BioRSP computes coverage and spatial organization scores for genes in a two-dimensional embedding and can classify genes into archetypes using explicit thresholds.

## Start here

### Installation

```bash
pip install .
```

### First run (Python API)

```python
import scanpy as sc
from biorsp import score_genes, classify_genes

adata = sc.datasets.pbmc3k_processed()

scores = score_genes(adata, adata.var_names, embedding_key="X_umap")
results = classify_genes(scores)

print(results.head())
```

Expected output: a pandas DataFrame with at least `Coverage`, `Spatial_Bias_Score`, `Directionality`, and `Archetype` columns. The exact values depend on the embedding and dataset.

### Benchmarks vs. kidney case studies

- **Benchmarks** (synthetic simulations): run scripts in `analysis/benchmarks/runners/`. These produce contract outputs under `OUTDIR/<benchmark>/<run_id>/`.
  - Example: `python analysis/benchmarks/runners/run_archetypes.py --mode quick --outdir results/benchmarks`
- **Kidney case studies** (KPMP h5ad): run scripts in `analysis/kidney_atlas/runners/` with a local KPMP `.h5ad` file.
  - Example: `python analysis/kidney_atlas/runners/run_kpmp_archetypes_all_genes.py --h5ad /path/to/kpmp.h5ad --outdir results/kpmp`

### Output artifacts

Benchmark runs follow a contract with these artifacts in the run directory:

- `runs.csv`: per-replicate results table
- `summary.csv`: aggregated metrics with confidence intervals
- `manifest.json`: run metadata, parameters, and provenance
- `report.md`: human-readable summary
- `figures/`: standardized plots

Kidney runners use similar conventions and typically emit `runs.csv`, `manifest.json`, `report.md` (when applicable), and `figures/` or `examples/` directories.

## Reproducibility and determinism

- Set the random seed in configs or CLI flags (for benchmarks, `--seed`).
- For deterministic CPU execution, set thread controls before running:
  - `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.
- Benchmark run folders include a run identifier (`run_id`) and timestamp to distinguish repeated runs.

## Documentation

Start with `docs/1_start_here/quickstart.md` and `docs/1_start_here/intro.md` for additional context. CLI details are in `docs/3_guides/cli_reference.md`.

## Testing

See `docs/testing.md` for local test commands.

## Citation

Please cite the relevant BioRSP publication when using this codebase in scientific work.
