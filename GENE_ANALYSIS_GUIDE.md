# BioRSP Gene Expression Analysis - Complete Guide

## Overview

This document provides a complete guide for scanning gene expression features and creating RSP (Radar Scanning Plot) visualizations using the BioRSP package.

## What Was Added

### 1. Plotting Module (`src/plotting.py`)

A comprehensive visualization module with four main functions:

- **`plot_rsp_heatmap()`**: Plot a single RSP heatmap in polar coordinates
- **`plot_rsp_grid()`**: Create a grid of multiple RSP heatmaps
- **`plot_rsp_summary()`**: Generate a 3-panel summary (embedding, distribution, RSP)
- **`save_top_results()`**: Batch save individual plots for top results

### 2. Enhanced Example Scripts

- **`examples/kpmp.py`**: Complete analysis with gene scanning and inline plotting
- **`examples/kpmp_with_plots.py`**: Full pipeline with all plotting utilities
- **`examples/README.md`**: Comprehensive usage documentation

### 3. Bug Fixes

- **Fixed IndexError in `_bin_to_wedges`**: Added optional mask parameter for batch-wise operations
- **Fixed dimension mismatch in `_compute_Z_grid`**: Properly handle single-band case to avoid 3D arrays

## Quick Start

### Installation

```bash
# Install the package in editable mode
cd /path/to/biorsp_v3
pip install -e .
```

### Basic Gene Scanning

```python
import anndata
import numpy as np
from src import ScanParams, RadarScanner
from src.plotting import plot_rsp_heatmap, plot_rsp_grid

# Load data
adata = anndata.read_h5ad("your_data.h5ad")
coords = adata.obsm["X_umap"]

# Configure scanner
params = ScanParams(
    B=180,
    widths_deg=(15, 30, 60, 90, 120, 180),
    radial_mode="quantile",
    n_bands=2,
    standardize="rank",
    null_model="within_batch_rotation",
    R=500,
)

# Fit scanner
scanner = RadarScanner(params).fit(
    coords,
    batches=adata.obs["donor_id"].to_numpy()
)

# Scan genes from var['feature_name']
gene_names = adata.var['feature_name'].tolist()
results = []

for i, gene_name in enumerate(gene_names[:50]):  # Top 50 genes
    gene_expr = adata.X[:, i].toarray().ravel()
    res = scanner.scan_feature(gene_expr, name=gene_name)
    results.append(res)

# Create visualizations
fig = plot_rsp_grid(
    results,
    ncols=4,
    max_plots=12,
    sort_by='p_value',
    suptitle='Top Genes'
)
fig.savefig('top_genes.png', dpi=150)
```

## Complete Pipeline Example

See `examples/kpmp_with_plots.py` for a full pipeline that:

1. Loads KPMP data
2. Scans metadata features (donors, cell classes)
3. Scans top 50 variable genes
4. Creates multiple visualization outputs:
   - Grid of top genes
   - Grid of cell classes
   - Detailed summary for top gene
   - Individual high-res plots
   - CSV with all statistics

## Output Files Generated

When running `kpmp_with_plots.py`:

```
examples/output/
├── top_genes_grid.png           # 12-panel grid of top genes
├── cell_classes_grid.png        # Grid of cell class features
├── summary_<gene>.png            # 3-panel detailed view
├── gene_rsp_01_<gene>.png       # Individual plots (top 5)
├── gene_rsp_02_<gene>.png
├── gene_rsp_03_<gene>.png
├── gene_rsp_04_<gene>.png
├── gene_rsp_05_<gene>.png
└── rsp_summary_statistics.csv   # All results table
```

## Understanding RSP Plots

### Visual Elements

- **Colors**:

  - Red = Enrichment (positive Z-scores)
  - Blue = Depletion (negative Z-scores)
  - Intensity = Magnitude of Z-score

- **Green Line**: Peak direction (maximum significance)

- **Radial Bands**: Different distance scales from center

- **Angular Sectors**: Different directions in the embedding

### Interpreting Results

**Significant Gene** (p < 0.05, high Z_max):

- Shows clear directional pattern
- Gene expression enriched in specific embedding direction
- Biological interpretation: Gene marks cells in that spatial region

**Non-significant Gene** (p > 0.05):

- Uniform or noisy pattern
- No strong directional enrichment
- May still have biological relevance in other contexts

## Customization

### Scanner Parameters

```python
params = ScanParams(
    B=360,          # Higher resolution (more angular bins)
    widths_deg=(10, 20, 30, 45, 60, 90, 120, 180),  # More scales
    radial_mode="quantile",  # or "percentile", "fixed"
    n_bands=3,      # More radial bands
    standardize="rank",  # or "zscore", "none"
    residualize="ols",  # Remove covariate effects
    density_correction="2d",  # Correct for density
    var_mode="binomial",  # Variance model
    null_model="within_batch_rotation",  # Batch-aware
    R=1000,         # More permutations = better p-values
)
```

### Plotting Customization

```python
from src.plotting import plot_rsp_heatmap
import matplotlib.pyplot as plt

# Custom single plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10),
                       subplot_kw=dict(projection='polar'))
plot_rsp_heatmap(
    result,
    ax=ax,
    cmap='seismic',  # Different colormap
    show_peak=True,
    title='My Custom Title'
)
plt.savefig('custom.png', dpi=300)
```

### Grid Layout

```python
from src.plotting import plot_rsp_grid

fig = plot_rsp_grid(
    results,
    ncols=5,           # 5 columns
    figsize=(20, 12),  # Custom size
    max_plots=20,      # Show top 20
    sort_by='Z_max',   # Sort by Z-score instead of p-value
    suptitle='High Z-score Genes'
)
```

## Performance Optimization

### For Large Datasets

```python
# 1. Reduce permutations for testing
params = ScanParams(..., R=100)  # Instead of R=500

# 2. Scan genes in batches
batch_size = 100
for i in range(0, len(gene_names), batch_size):
    batch_genes = gene_names[i:i+batch_size]
    # Process batch

# 3. Use simpler null model
params = ScanParams(..., null_model='rotation')  # Faster than 'within_batch_rotation'
```

### Memory Management

```python
# For sparse matrices
if hasattr(adata.X, 'toarray'):
    gene_expr = adata.X[:, gene_idx].toarray().ravel()
else:
    gene_expr = adata.X[:, gene_idx].ravel()
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

**Solution**: Install the package

```bash
pip install -e .
```

### "IndexError: boolean index did not match"

**Solution**: Update to latest code with fixed `_bin_to_wedges` method

### "ValueError: too many values to unpack"

**Solution**: Update to latest code with fixed `_compute_Z_grid` method

### "ValueError: wedge_weights must be nonnegative"

**Solution**: Use `standardize='none'` for binary features

```python
params = ScanParams(..., standardize='none')
```

## Best Practices

### 1. Gene Selection

- Start with top variable genes (faster, more likely significant)
- Or select known marker genes for validation
- Avoid genes with very low expression (high noise)

```python
# Select top variable genes
gene_vars = np.var(adata.X.toarray(), axis=0)
top_indices = np.argsort(gene_vars)[::-1][:100]
```

### 2. Quality Control

- Check `flags` in results for QC warnings
- Look at `runtime` to identify slow features
- Examine `ER` (enrichment ratio) for effect size

```python
for res in results:
    if res.p_value < 0.05:
        print(f"{res.name}: p={res.p_value:.4f}, ER={res.ER:.2f}")
        if res.flags:
            print(f"  Warnings: {res.flags}")
```

### 3. Multiple Testing Correction

```python
from src import bh_qvalues
import numpy as np

# Collect p-values
p_values = np.array([r.p_value for r in results])

# Compute q-values
q_values = bh_qvalues(p_values)

# Add to results
for res, qval in zip(results, q_values):
    res.q_value = qval

# Filter by FDR
significant = [r for r in results if r.q_value < 0.05]
```

## Example Workflows

### Workflow 1: Marker Gene Validation

```python
# Known marker genes
markers = ['APOE', 'CD3D', 'CD79A', 'DCN']

# Scan markers
marker_results = []
for gene in markers:
    idx = gene_names.index(gene)
    expr = adata.X[:, idx].toarray().ravel()
    res = scanner.scan_feature(expr, name=gene)
    marker_results.append(res)

# Visualize
fig = plot_rsp_grid(marker_results, ncols=2)
fig.savefig('markers.png')
```

### Workflow 2: Cell Type Specific Genes

```python
# Get cell type specific genes
cell_type = 'T_cell'
ct_mask = adata.obs['cell_type'] == cell_type

# Find DE genes for this cell type (pseudo-bulk or other method)
# ... differential expression analysis ...

# Scan top DE genes
de_results = []
for gene in top_de_genes:
    idx = gene_names.index(gene)
    expr = adata.X[:, idx].toarray().ravel()
    res = scanner.scan_feature(expr, name=f"{cell_type}_{gene}")
    de_results.append(res)
```

### Workflow 3: Batch Effect Analysis

```python
# Scan each batch separately
batch_results = {}
for batch in adata.obs['batch'].unique():
    batch_mask = adata.obs['batch'] == batch
    batch_coords = coords[batch_mask]

    scanner_batch = RadarScanner(params).fit(batch_coords)
    batch_results[batch] = scanner_batch.scan_feature(
        gene_expr[batch_mask],
        name=f"batch_{batch}"
    )
```

## Citation

If you use BioRSP in your research, please cite:

```
[Citation information to be added]
```

## Support

- GitHub Issues: [repository URL]
- Documentation: `examples/README.md`
- Examples: `examples/kpmp.py`, `examples/kpmp_with_plots.py`

## Changelog

### Latest Update (October 2025)

- ✅ Added comprehensive plotting module (`src/plotting.py`)
- ✅ Fixed IndexError in `_bin_to_wedges` for batch-wise operations
- ✅ Fixed dimension mismatch in `_compute_Z_grid` for single-band case
- ✅ Added complete gene expression scanning examples
- ✅ Created detailed documentation and guides
- ✅ Added visualization utilities to `__init__.py` exports
