# Usage Guide & CLI Reference

This guide covers the command-line interface and advanced configuration options for BioRSP.

## Command-line interface

BioRSP provides a command-line interface for running analyses without writing code. The CLI supports the full analysis workflow with permutation testing and flexible configuration.

### Basic usage

```bash
# Run analysis on a dataset
python -m biorsp.cli run \
  --expression data/expression.csv \
  --coords data/coords.csv \
  --outdir results/ \
  --seed 42
```

### Common examples

**1. Marker discovery with high sensitivity (narrow sectors)**

```bash
python -m biorsp.cli run \
  --expression kidney_expr.csv \
  --coords kidney_umap.csv \
  --outdir results/markers/ \
  --B 72 \
  --delta 60 \
  --q 0.90 \
  --inference \
  --n-perm 1000 \
  --seed 42
```

This uses 72 sectors with 60° width for high angular resolution, 90th percentile foreground threshold, and runs 1000 permutations for significance testing.

**2. Co-expression analysis (broader integration)**

```bash
python -m biorsp.cli run \
  --expression gene_pairs.csv \
  --coords embedding.csv \
  --outdir results/coexpression/ \
  --B 36 \
  --delta 120 \
  --q 0.50 \
  --inference \
  --n-perm 200 \
  --seed 42
```

Uses 36 sectors with 120° width for broad spatial integration and 50th percentile foreground threshold.

**3. With UMI stratification for improved calibration**

```bash
python -m biorsp.cli run \
  --expression expr.csv \
  --coords coords.csv \
  --umis umi_counts.csv \
  --umi-column total_counts \
  --outdir results/stratified/ \
  --inference \
  --perm-mode rt_umi \
  --n-r-bins 10 \
  --n-umi-bins 10 \
  --n-perm 1000 \
  --seed 42
```

Enables stratified permutation testing using radial bins, angular bins, and UMI count bins for better null calibration.

### CLI parameters

**Input files:**

- `--expression`: CSV/TSV file with expression matrix (genes as columns, cells as rows, or use `--transpose`)
- `--coords`: CSV/TSV file with 2D coordinates (x, y columns)
- `--umis`: Optional CSV/TSV file with UMI counts per cell (for stratified inference)
- `--umi-column`: Column name for UMI counts (default: auto-detect "umi" or "umis")

**Geometry:**

- `--B`: Number of sectors (default: 360)
- `--delta`: Sector width in degrees (default: 20)

**Foreground:**

- `--q`: Foreground quantile threshold (default: 0.90)
- `--min-fg-total`: Minimum total foreground cells (default: 100)

**Adequacy thresholds:**

- `--min-count`: Min foreground cells per sector (default: 10)
- `--min-bg-count`: Min background cells per sector (default: 50)
- `--min_adequacy_fraction`: Min fraction of adequate sectors (default: 0.9)

**Inference:**

- `--inference`: Enable permutation testing (required for p-values)
- `--n-perm`: Number of permutations (default: 200)
- `--perm-mode`: Permutation strategy (`radial`, `joint`, `rt_umi`, `none`)
- `--n-r-bins`: Number of radial bins for stratification (default: 10)
- `--n-theta-bins`: Number of angular bins (default: 4)
- `--n-umi-bins`: Number of UMI bins (default: 10)
- `--min-stratum-size`: Min cells per stratum (default: 50)

**Sector weighting:**

- `--sector-weight-mode`: Weighting mode (`none`, `sqrt_frac`, `effective_min`, `logistic_support`)
- `--sector-weight-k`: Tunable weighting parameter (default: 5.0)

**Output:**

- `--outdir`: Directory for results and plots (default: ".")
- `--seed`: Random seed for reproducibility (default: 42)

### Output files

The CLI creates the following outputs in `--outdir`:

- `results.csv`: Per-feature summary statistics (anisotropy, p-value, coverage, etc.)
- `manifest.json`: Run metadata (config, version, timestamp)
- `radar_*.png`: Radar plots for each feature (if `--save-plots` enabled)

### Programmatic usage

For more control, use the Python API directly. See the [API Reference](../api/index.md) and [Quickstart](../1_start_here/quickstart.md).
