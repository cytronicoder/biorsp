# CLI reference

BioRSP provides a single CLI entry point via `biorsp run` for scoring datasets stored as files.

## Basic usage

```bash
python -m biorsp.cli run \
  --expression data/expression.csv \
  --coords data/coords.csv \
  --output results.csv
```

`--expression` should be a cell-by-gene matrix in CSV/TSV format. `--coords` must contain `x` and `y` columns for each cell.

## Common options

```bash
python -m biorsp.cli run \
  --expression data/expression.csv \
  --coords data/coords.csv \
  --output results.csv \
  --outdir results_run \
  --seed 42 \
  --inference \
  --n-perm 200
```

- `--outdir` writes a `run_metadata.json` manifest for reproducibility.
- `--inference` enables permutation testing; `--n-perm` controls the number of permutations.

## Parameter summary

### Inputs

- `--expression`: CSV/TSV file with expression values (cells × genes).
- `--coords`: CSV/TSV file with `x` and `y` columns.
- `--umis`: optional UMI counts file for stratified inference.
- `--umi-column`: column name in the UMI file.
- `--transpose`: transpose the expression matrix if it is gene × cell.

### Geometry and foreground definition

- `--B`: number of angular sectors.
- `--delta`: sector width in degrees.
- `--q`: foreground quantile threshold.
- `--min-count`, `--min-bg-count`, `--min-fg-total`: adequacy thresholds for foreground/background support.

### Permutation inference

- `--inference`: enable permutation testing.
- `--n-perm`: number of permutations.
- `--perm-mode`: `radial`, `joint`, `rt_umi`, or `none`.
- `--n-r-bins`, `--n-theta-bins`, `--n-umi-bins`, `--min-stratum-size`: stratification parameters.

### Sector weighting

- `--sector-weight-mode`: `none`, `sqrt_frac`, `effective_min`, or `logistic_support`.
- `--sector-weight-k`: parameter controlling sector weighting.

## Outputs

- `--output`: CSV table with per-feature summaries.
- `--outdir`: optional directory containing `run_metadata.json` (manifest with parameters and dataset summary).

For programmatic use, see `biorsp.api` and the quickstart.
