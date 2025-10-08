# KPMP Example Usage Guide

This directory contains examples demonstrating BioRSP analysis on the KPMP single-nucleus RNA-seq dataset.

## Files

- **`kpmp.py`**: Complete analysis script with gene expression scanning and RSP visualization
- **`kpmp_with_plots.py`**: Enhanced version with comprehensive plotting utilities
- **`data/kpmp_sn.h5ad`**: KPMP single-nucleus RNA-seq data (not included in repo)

## Quick Start

### Basic Usage

```python
python kpmp.py
```

This will:
1. Load the KPMP dataset
2. Scan donor, class, and gene expression features
3. Generate RSP plots for the most significant features

### Enhanced Usage with Plotting Utilities

```python
python kpmp_with_plots.py
```

This creates:
- Grid plots of top significant genes
- Cell class RSP heatmaps
- Detailed summary plots
- Individual high-resolution plots
- CSV file with summary statistics

## Output Files

All outputs are saved to `examples/output/`:

- `top_genes_grid.png` - Grid of top 12 most significant genes
- `cell_classes_grid.png` - Grid of cell class features
- `summary_<gene_name>.png` - Detailed 3-panel view of top gene
- `gene_rsp_*.png` - Individual high-resolution plots for top 5 genes
- `rsp_summary_statistics.csv` - Complete results table

## Customization

### Scanning Specific Genes

```python
# Scan a specific gene by name
gene_name = "APOE"
gene_idx = kpmp_adata.var['feature_name'].tolist().index(gene_name)
gene_expr = kpmp_adata.X[:, gene_idx].toarray().ravel()
result = scanner.scan_feature(gene_expr, name=f"gene={gene_name}")

# Plot the result
from src.plotting import plot_rsp_heatmap
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='polar'))
plot_rsp_heatmap(result, ax=ax)
plt.savefig(f"{gene_name}_rsp.png", dpi=200)
```

### Custom Scanner Parameters

```python
from src import ScanParams, RadarScanner

# Adjust scanning parameters
params = ScanParams(
    B=360,  # More angular bins for higher resolution
    widths_deg=(10, 20, 30, 45, 60, 90, 120, 180),  # More width scales
    radial_mode="quantile",
    n_bands=3,  # More radial bands
    standardize="rank",  # or "zscore", "none"
    null_model="within_batch_rotation",  # or "rotation", "permutation"
    R=1000,  # More permutations for more accurate p-values
)

scanner = RadarScanner(params).fit(coords, batches=batches)
```

### Creating Custom Visualizations

```python
from src.plotting import plot_rsp_grid, plot_rsp_summary

# Create grid of results
fig = plot_rsp_grid(
    results,
    ncols=4,
    max_plots=16,
    sort_by='p_value',  # or 'Z_max'
    suptitle='My Custom Analysis'
)
fig.savefig('custom_grid.png', dpi=150)

# Create comprehensive summary for a single feature
fig = plot_rsp_summary(
    result,
    coords=kpmp_adata.obsm["X_umap"],
    feature_values=gene_expression,
    figsize=(16, 5)
)
fig.savefig('detailed_summary.png', dpi=200)
```

## Understanding the Results

### FeatureResult Attributes

Each scan returns a `FeatureResult` object with:

- `name`: Feature name
- `Z_max`: Maximum Z-score across all positions
- `p_value`: Statistical significance
- `phi_star`: Peak angle in radians
- `width_idx`: Index of best width scale
- `center_idx`: Index of best angular center
- `ER`: Enrichment ratio
- `R_conc`: Concentration measure
- `Z_heat`: Full heatmap (J × B array) for visualization

### Interpreting RSP Plots

- **Color**: Red indicates enrichment, blue indicates depletion
- **Green line**: Marks the peak direction (maximum significance)
- **Radial bands**: Different distance scales from embedding center
- **Angular sectors**: Different directions in the embedding

### Statistical Significance

- **p-value < 0.05**: Statistically significant directional pattern
- **High Z_max**: Strong directional enrichment
- **Low p-value + High Z_max**: Confident significant finding

## Performance Tips

1. **Reduce R for faster testing**: Use `R=100` instead of `R=500`
2. **Limit genes scanned**: Start with top 20-50 most variable genes
3. **Use appropriate null model**: 
   - `within_batch_rotation`: Best for batch-corrected data
   - `rotation`: Faster, assumes no batch effects
   - `permutation`: Most stringent, slowest

## Data Requirements

The example expects an AnnData object with:
- `.obsm['X_umap']`: 2D embedding coordinates
- `.obs['donor_id']`: Batch/donor identifiers
- `.obs['class']`: Cell type annotations
- `.var['feature_name']`: Gene names
- `.X`: Gene expression matrix (cells × genes)

## Troubleshooting

### "No module named 'src'"

Make sure you've installed the package:
```bash
pip install -e .
```

### "IndexError" during scanning

This was fixed in the latest version. Make sure you have the updated code with the mask parameter in `_bin_to_wedges`.

### Memory issues with large datasets

Process genes in batches:
```python
batch_size = 100
for i in range(0, n_genes, batch_size):
    genes_batch = range(i, min(i + batch_size, n_genes))
    # Process batch
```

### Negative feature values error

Use `standardize='none'` if your features are already non-negative (e.g., binary indicators).

## Citation

If you use this code in your research, please cite:
[Add citation information here]

## Contact

For questions or issues, please open an issue on GitHub.
