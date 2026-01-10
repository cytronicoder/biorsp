# BioRSP Case Study: Human Kidney Spatial Analysis

This case study applies the BioRSP framework to analyze spatial organization patterns in human kidney tissue using the Azimuth Human Kidney reference dataset.

## Overview

This analysis focuses on the **Thick Ascending Limb (TAL)** cell population in human kidney tissue, demonstrating BioRSP's ability to quantify radial spatial organization and identify genes with significant localization patterns. The study uses publicly available reference datasets from the Azimuth project and KPMP (Kidney Precision Medicine Project).

## Directory Structure

```
kidney/
├── README.md                           # This file
├── data/                               # Raw data matrices and metadata
│   ├── kpmp_*.{mtx,tsv,csv}           # KPMP dataset (counts, features, cells, metadata)
│   ├── lake_*.{mtx,tsv,csv}           # Lake et al. dataset
│   └── kidney_demo_*.{csv,mtx}        # Demo subset for quick testing
├── reference/                          # Processed reference files
│   ├── ref.h5ad                       # Main reference (AnnData format)
│   ├── ref.Rds                        # Seurat reference object
│   ├── kpmp.h5ad                      # KPMP-specific reference
│   ├── ref_kpmp.h5ad / ref_lake.h5ad  # Dataset-specific references
│   └── config.json                    # Reference configuration metadata
├── scripts/                            # Data processing and conversion utilities
│   ├── convert_to_h5ad.R              # Convert Seurat RDS to H5AD
│   ├── export_for_python.R            # Export RDS contents for Python
│   ├── build_h5ad_from_export.py      # Assemble H5AD from exported files
│   ├── setup.R                        # Download and setup reference data
│   └── analyze_ref.py                 # Reference dataset exploration
├── notebooks/                          # Interactive analysis notebooks
│   └── explore_kidney_reference.ipynb # Exploratory analysis of reference data
├── results/                            # Output from analysis runs
│   └── tal_case_study/                # TAL cell population analysis results
└── run_tal_analysis.py                # Main analysis pipeline for TAL cells

```

## Getting Started

### Prerequisites

- Python 3.8+ with `biorsp`, `scanpy`, `anndata`, `pandas`, `numpy`, `scipy`, `matplotlib`
- R 4.0+ with `Seurat`, `SeuratDisk` (optional, for RDS conversion)
- For full reference setup: ~10GB disk space for downloaded datasets

### Quick Start

Run the TAL cell analysis using the demo dataset:

```bash
# Using the provided H5AD reference
python run_tal_analysis.py \
  --ref_data reference/ref.h5ad \
  --outdir results/tal_case_study \
  --controls "SLC12A1,UMOD,EGF" \
  --donor_key donor_id \
  --n_workers 4
```

### Full Reference Setup

If you need to rebuild the reference from scratch:

```bash
# 1. Download raw data (requires links/dropbox_links.txt)
Rscript scripts/setup.R

# 2. Convert to H5AD format
Rscript scripts/export_for_python.R reference/ref.Rds
python scripts/build_h5ad_from_export.py

# Or use the Snakemake pipeline (if available)
snakemake --cores 4
```

## Analysis Pipeline

The TAL analysis (`run_tal_analysis.py`) performs the following steps:

1. **Data Loading**: Loads the Azimuth kidney reference (H5AD format preferred)
2. **Cell Subsetting**: Filters to TAL cell population using cell type annotations
3. **Gene Selection**: Chooses control genes (known TAL markers) + discovery genes
4. **BioRSP Analysis**: Computes radial spatial organization scores (C_g, S_g) for each gene
5. **Permutation Testing**: Evaluates statistical significance with depth-aware permutations
6. **Stability Diagnostics**: Performs donor-aware reruns to assess robustness
7. **Visualization**: Generates radar plots, ranked summaries, and diagnostic figures

### Key Parameters

- `--ref_data`: Path to reference H5AD file (default: `reference/ref.h5ad`)
- `--controls`: Comma-separated list of known TAL marker genes
- `--donor_key`: Metadata column for donor/batch information
- `--n_workers`: Number of parallel workers for permutation tests
- `--n_permutations`: Number of permutation resamples (default: 1000)
- `--outdir`: Output directory for results

## Data Sources

### KPMP Dataset

- **Source**: Kidney Precision Medicine Project
- **Cell types**: Multiple kidney cell populations including TAL
- **Files**: `kpmp_counts.mtx`, `kpmp_features.tsv`, `kpmp_cells.tsv`, `kpmp_metadata.csv`

### Lake et al. Dataset

- **Source**: Lake et al. 2019 (single-nucleus RNA-seq)
- **Files**: `lake_counts.mtx`, `lake_features.tsv`, `lake_cells.tsv`, `lake_metadata.csv`

### Azimuth Reference

- **Provenance**: https://github.com/satijalab/azimuth-references/tree/master/human_kidney
- **Commit**: b8b07dcdfcc09816a85aad07362e7bad4de03976
- **Format**: Seurat RDS → converted to H5AD for Python workflows

## Expected Outputs

### Results Directory Structure

```
results/tal_case_study/
├── summary_all_genes.csv              # Ranked gene results (C_g, S_g, p-values)
├── summary_significant_genes.csv      # Filtered significant hits
├── metadata.json                      # Run configuration and provenance
├── figures/
│   ├── radar_<gene>.png              # Per-gene radial profile plots
│   ├── qq_plot.png                   # P-value calibration
│   └── top_genes_grid.png            # Grid of top hits
└── diagnostics/
    ├── stability_<gene>.csv          # Donor-aware rerun results
    └── coverage_summary.txt          # Spatial coverage statistics
```

## Known Markers for TAL Cells

- **SLC12A1**: Na-K-2Cl cotransporter (definitive TAL marker)
- **UMOD**: Uromodulin (Tamm-Horsfall protein)
- **EGF**: Epidermal growth factor
- **CLDN10**: Claudin 10 (tight junction protein)
- **CLDN16**: Claudin 16 (paracellular cation channel)

## Notes and Caveats

- **Annotation Keys**: Some reference versions may lack standard Azimuth annotation columns (e.g., `annotation.l2`). The pipeline falls back to marker-based heuristics in such cases.
- **Large Files**: Raw data files are large (hundreds of MBs to GBs). Consider using `.gitignore` to exclude `data/` and `reference/` from version control.
- **Computational Requirements**: Full permutation testing (1000 resamples) can take 30-60 minutes depending on gene count and hardware.
- **Memory**: Processing full reference datasets requires ~16GB RAM. Use subsampling for demo/test runs.

## References

1. **BioRSP**: [Citation pending - please refer to the main repository]
2. **Azimuth**: Hao et al. (2021). "Integrated analysis of multimodal single-cell data." Cell. https://doi.org/10.1016/j.cell.2021.04.048
3. **KPMP**: Lake et al. (2021). "An atlas of healthy and injured cell states and niches in the human kidney." Nature. https://doi.org/10.1038/s41586-021-03549-5

## Citation

If you use this case study or the BioRSP framework, please cite:

```
[BioRSP citation - to be updated]
```

## Contact

For questions about this case study or the BioRSP framework, please open an issue in the main repository.
