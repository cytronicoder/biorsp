# KPMP Embedding Visualization Results

Generated on: January 11, 2026

This directory contains publication-ready 2D UMAP embedding plots of the KPMP kidney single-cell dataset (225,177 cells × 31,332 genes).

## Plot Collections

### 1. Main Embedding Plots (`embedding_plots/`)

Auto-discovered metadata features and generated comprehensive visualizations:

- **Overview** (`kpmp_embedding_overview.*`): Uncolored density impression showing overall cell distribution
- **Cell Types** (`kpmp_embedding_by_cell_type.*`): 25 cell types discovered from `cell_type` column
- **Disease Status** (`kpmp_embedding_by_disease.*`): Three disease categories (acute kidney failure, chronic kidney disease, normal)
- **Sample Origin** (`kpmp_embedding_by_donor_id.*`): 77 unique donors (legend omitted due to high count)
- **Density** (`kpmp_embedding_density.*`): Hexbin density plot showing cell concentration

**Discovered columns:**
- Cell type: `cell_type`
- Condition: `disease`
- Sample: `donor_id`

### 2. QC Metrics (`embedding_plots_qc/`)

Quality control visualizations showing technical features:

- **Gene Counts** (`kpmp_embedding_by_nFeature_RNA.*`): Number of genes detected per cell (continuous colormap)
- **Mitochondrial %** (`kpmp_embedding_by_percent.mt.*`): Percentage of mitochondrial transcripts (continuous colormap)

These plots help identify technical artifacts or low-quality cells in the embedding space.

### 3. Faceted Analysis (`embedding_plots_faceted/`)

Small multiples showing disease heterogeneity:

- **Disease Category Facets** (`kpmp_embedding_facet_disease_category_cell_type.*`): Four panels (AKI, CKD, Healthy_living_donor, Healthy_stone_donor), each colored by cell type

This enables direct comparison of cell type distributions across disease conditions.

## Technical Details

All plots generated with:
- **Embedding**: X_umap (UMAP coordinates from `.obsm`)
- **Cells plotted**: 225,177 (no subsampling needed)
- **Rasterization**: Enabled (for manageable file sizes with large datasets)
- **DPI**: 300 (publication quality)
- **Formats**: PNG (for presentations/quick viewing) and PDF (vector graphics for publication)
- **Point size**: 2.0
- **Alpha**: 0.6 (60% transparency)

## Reproducibility

Each directory contains a `metadata.json` file with:
- Exact parameters used
- Columns discovered and plotted
- Dataset statistics
- Embedding key used

To regenerate these plots:

```bash
# Main plots
python plot_kpmp_embedding.py --density

# QC plots
python plot_kpmp_embedding.py --color-by nFeature_RNA --outdir results/embedding_plots_qc
python plot_kpmp_embedding.py --color-by percent.mt --outdir results/embedding_plots_qc

# Faceted plots
python plot_kpmp_embedding.py --facet-by disease_category --color-by cell_type --outdir results/embedding_plots_faceted
```

## Usage Notes

- **Cell type plot**: Shows clear spatial organization with distinct clusters for epithelial, immune, endothelial, and interstitial cells
- **Disease plot**: Reveals mixture of conditions across the embedding, suggesting disease effects may be more subtle than cell type identity
- **Donor plot**: High donor count (77) means individual donors are hard to distinguish; useful for checking batch effects
- **QC plots**: Help validate that technical quality metrics don't drive major embedding structure
- **Faceted plot**: Enables per-condition analysis while preserving spatial relationships

## Citation

If using these visualizations, please cite:
- KPMP data source
- BioRSP analysis framework (if applicable)
- Scanpy/AnnData for data handling
