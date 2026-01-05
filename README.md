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
pip install -e .
```

## Key Concepts

- **Vantage Point**: The center of the radar scan (default: geometric median).
- **RSP Profile**: The signed radial enrichment $R_g(\theta)$ across $B$ sectors.
- **Anisotropy**: The global magnitude of directional patterning.
- **Stratified Inference**: Calibrated p-values that respect local cell density.
