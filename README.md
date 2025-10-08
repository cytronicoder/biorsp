> [!NOTE]
> I spent the past few months refining BioRSP in response to invaluable advice on usability, reproducibility, and design from conference attendees, collaborators, and early users during experimentation. The current roadmap can be found [here](https://github.com/cytronicoder/biorsp/issues/1). Thanks to everyone who provided feedback and helped improve the project!

### Motivation

Single-cell genomics datasets are often visualized with 2D embeddings (e.g., UMAP, t-SNE, PHATE) to reveal structure and heterogeneity. However, interpreting these visualizations is often subjective and lacks statistical rigor.

Most tools still analyze 2D embeddings by eyeballing coverage, density, and shape, then backing that impression with cluster DE or a black-box classifier. These approaches rarely quantify directional or radial structure and often miss multiscale patterns (narrow hot spots vs broad lobes) or overinterpret density artifacts.

### What is BioRSP?

BioRSP (Biological Radar Scanning Plot) is a Python package designed to convert 2D embeddings of single-cell data into polar coordinates and perform a multiscale "radar" sector scan to identify regions of feature enrichment.

BioRSP addresses the aforementioned issues of interpretation and statistical rigor by providing a systematic, multiscale method to scan 2D embeddings for regions enriched in specific features (e.g., gene expression, protein levels, metadata). By converting to polar coordinates and performing sector scans at multiple scales, BioRSP can identify both narrow and broad patterns of enrichment that might be missed by traditional methods.

## Installation

### Install in development mode (editable)

```bash
# Clone the repository
git clone https://github.com/cytronicoder/biorsp.git
cd biorsp

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Using Make commands

```bash
# Install package
make install

# Install with dev dependencies
make install-dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Lint code
make lint

# Clean build artifacts
make clean
```

## Quick Start

### Gene Expression Analysis with Foreground/Background Comparison

For gene expression analysis, use **threshold-based binarization** to compare expressing vs non-expressing cells:

```python
import anndata
from src.radar_scan import ScanParams, RadarScanner
from src.plotting import plot_rsp_heatmap

# Load your data
adata = anndata.read_h5ad("your_data.h5ad")

# Get 2D coordinates (e.g., UMAP)
coords = adata.obsm["X_umap"]

# Configure scan parameters with threshold mode for gene analysis
params = ScanParams(
    B=180,                              # Number of angles
    widths_deg=(15, 30, 60, 90, 120),  # Sector widths in degrees
    radial_mode="quantile",             # Radial binning mode
    n_bands=2,                          # Number of radial bands
    null_model="within_batch_rotation", # Null model
    R=500,                              # Number of permutations
    random_state=0,
    # NEW: Threshold parameters for foreground/background comparison
    threshold_mode="positive",          # Compare expressing vs non-expressing cells
    threshold_value=None,               # Not needed for 'positive' mode
)

# Initialize and fit scanner
scanner = RadarScanner(params).fit(
    coords,
    batches=adata.obs["donor_id"].to_numpy(),  # For batch-aware null model
)

# Scan a gene with foreground/background comparison
gene_expr = adata[:, "gene_name"].X.toarray().flatten()
result = scanner.scan_feature(gene_expr, name="gene_name")

# Visualize result
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
plot_rsp_heatmap(result, ax=ax)
plt.show()
```

### Threshold Modes

BioRSP supports four threshold modes for feature binarization:

- **`threshold_mode="none"`**: No binarization (continuous values)
- **`threshold_mode="positive"`**: Foreground = cells with value > 0 (recommended for scRNA-seq)
- **`threshold_mode="percentile"`**: Foreground = cells above percentile (requires `threshold_value`)
- **`threshold_mode="value"`**: Foreground = cells above absolute threshold (requires `threshold_value`)

See [GENE_ANALYSIS_GUIDE.md](GENE_ANALYSIS_GUIDE.md) for detailed documentation and examples.

## Testing

The package includes a test suite using pytest:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_radar_scan.py

# Run with verbose output
pytest -v
```

## Development

```bash
# Format code with black
black src/ tests/ examples/

# Lint with flake8
flake8 src/ tests/

# Or use make commands
make format
make lint
```


