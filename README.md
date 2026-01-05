# BioRSP: Bayesian Inference and Robustness for Radar Scanning Plots

BioRSP is a geometry-aware framework for statistically quantifying directional radial enrichment of signals in 2D spatial or embedding coordinates.

## Quickstart

```python
import biorsp
import numpy as np
import pandas as pd

# 1. Load your data (N cells x 2 coords, N cells x G genes)
coords = np.random.rand(1000, 2)
expr = pd.DataFrame(np.random.poisson(1, (1000, 10)), columns=[f"Gene_{i}" for i in range(10)])

# 2. Run the pipeline
summary = biorsp.run(coords, expr, outdir="results")

# 3. Inspect results
results_df = summary.to_dataframe()
print(results_df.head())
```

## Installation

```bash
pip install biorsp
```

For development:

```bash
git clone https://github.com/cytronicoder/biorsp.git
cd biorsp
pip install -e ".[dev]"
```

## Quickstart

The minimal example below demonstrates the core workflow:

```python
import biorsp
import numpy as np
import pandas as pd

# 1. Load your data (N cells × 2 coords, N cells × G genes)
coords = np.random.rand(1000, 2)
expr = pd.DataFrame(
    np.random.poisson(1, (1000, 10)),
    columns=[f"Gene_{i}" for i in range(10)]
)

# 2. Run the pipeline with default settings
summary = biorsp.run(coords, expr, outdir="results", seed=42)

# 3. Inspect results
results_df = summary.to_dataframe()
print(results_df[["feature", "anisotropy", "p_value", "feature_type"]].head())
```

See [examples/](examples/) for complete, reproducible workflows.

## Recommended Defaults

BioRSP provides flexible configuration for different spatial analysis tasks. Based on extensive validation, we recommend the following baselines:

| Task | Sector Width ($\Delta$) | Sectors ($B$) | Quantile ($q$) | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Marker Discovery** | $60^\circ$ | 72 | 0.90 | High sensitivity for localized "wedge" patterns. |
| **Co-expression** | $120^\circ$ | 36 | 0.50 | Broader integration for large-scale spatial gradients. |

**Note on Selection Bias:** Using wide sectors ($\Delta \ge 90^\circ$) can sometimes misclassify localized signals as global shifts. BioRSP implements a "zero-filling" policy for empty sectors and provides `coverage_fg` metrics to help identify these artifacts.

## Examples

We provide task-specific example scripts in the `examples/` directory:

- `examples/marker_discovery.py`: Demonstrates high-sensitivity settings for detecting localized "wedge" markers and global "rim" markers.
- `examples/gene_gene_copatterns.py`: Demonstrates broad spatial integration and pairwise radar correlation for identifying co-expressed gene modules.

Run them with:
```bash
python examples/marker_discovery.py --outdir results/marker
python examples/gene_gene_copatterns.py --outdir results/copatterns
```

## Key Concepts

- **Vantage Point**: The center of the radar scan (default: geometric median).
- **RSP Profile**: The signed radial enrichment $R_g(\theta)$ across $B$ sectors.
- **Anisotropy**: The global magnitude of directional patterning.
- **Stratified Inference**: Calibrated p-values that respect local cell density.

## Interpreting Outputs

BioRSP provides multiple metrics to characterize spatial patterns:

### Core Metrics

- **`anisotropy` ($A_g$)**: RMS magnitude of the RSP profile. Higher values indicate stronger directional patterning.
- **`p_value`**: Permutation-based significance test for anisotropy (stratified by radial bins and UMI counts).
- **`feature_type`**: Classification into "wedge" (localized), "rim" (global distal), "core" (global proximal), or "null".

### Signed Summaries

- **`r_mean`**: Mean signed shift. Positive = core bias (foreground closer to center), negative = rim bias (foreground farther from center).
- **`polarity`**: Signed energy ratio. Values near ±1 indicate globally one-signed patterns; near 0 indicates mixed or localized.
- **`localization_entropy` ($L_g$)**: Shannon entropy-based localization index. Near 0 = diffuse/global, near 1 = highly localized.

### Coverage Metrics

- **`coverage_bg`**: Fraction of sectors with adequate background support.
- **`coverage_fg`**: Fraction of sectors with foreground cells present.

Low `coverage_fg` with wide sectors ($\Delta \ge 90°$) may indicate a localized pattern being analyzed with overly broad windows. Consider narrower sectors for better localization.

### Empty Sector Policy

- **`zero`** (default): Sectors without foreground are filled with 0, preserving global bias interpretation.
- **`nan`**: Empty sectors are left as NaN, excluding them from anisotropy computation.

Choose `zero` for global patterns (rim/core) and `nan` for localized patterns (wedges) where empty regions are not meaningful background.

## Choosing Parameters

### Sector Width ($\Delta$)

- **Narrow (30–60°)**: High angular resolution, best for detecting localized wedge patterns.
- **Wide (90–120°)**: Broad integration, best for detecting global rim/core shifts.

**Trade-off**: Narrow sectors increase sensitivity to localized patterns but reduce statistical power per sector. Wide sectors smooth over local structure but can misclassify localized patterns as global.

### Number of Sectors ($B$)

- **High (72–144)**: Fine-grained angular sampling, reduces discretization artifacts.
- **Low (12–36)**: Coarse sampling, faster computation, suitable for exploratory analysis.

**Recommendation**: Use $B \approx 2\pi / (\Delta \text{ in radians})$ to ensure sectors overlap by ~50% for smooth profiles.

### Foreground Quantile ($q$)

- **High (0.85–0.95)**: Focus on the most extreme expressors, high specificity for marker genes.
- **Low (0.50–0.75)**: Broader foreground definition, better for subtle gradients.

**Recommendation**: Use $q = 0.90$ for marker discovery, $q = 0.50$ for co-expression analysis.

## Reproducibility

BioRSP supports fully reproducible analyses:

```python
import biorsp

# Set a seed for reproducibility
summary = biorsp.run(
    coords, expr,
    seed=42,  # Controls permutation test and foreground sampling
    outdir="results",
    config=biorsp.BioRSPConfig(
        B=72,
        delta_deg=60,
        foreground_quantile=0.90,
        n_perm=1000
    )
)

# Save run metadata
summary.save("results/run_summary.json")
```

### Run Metadata

BioRSP automatically saves:
- Configuration parameters (B, Δ, q, etc.)
- Software version and dependencies
- Random seeds used for permutation tests
- Execution timestamps

Access this metadata via:
```python
import json
with open("results/run_summary.json") as f:
    metadata = json.load(f)
print(metadata["config"], metadata["version"])
```

### Citing BioRSP

If you use BioRSP in your research, please cite:

```bibtex
@software{biorsp2025,
  title={BioRSP: Bayesian Inference and Robustness for Radar Scanning Plots},
  author={Your Name},
  year={2025},
  url={https://github.com/cytronicoder/biorsp}
}
```

## Contributing

Contributions are welcome! Please see [DEVELOPER_NOTES.md](DEVELOPER_NOTES.md) for guidelines.

### Development Setup

```bash
# Clone and install with dev dependencies
git clone https://github.com/cytronicoder/biorsp.git
cd biorsp
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install

# Run formatting and linting
make format
make lint

# Run tests
make test
```

## License

MIT License. See [LICENSE](LICENSE) for details.
