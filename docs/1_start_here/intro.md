# Introduction to BioRSP

BioRSP (Biological Radar Spatial Profiling) is a method for quantifying and interpreting spatial gene expression patterns. Unlike traditional methods that rely on identifying pre-defined spatial domains (clustering), BioRSP scores each gene individually based on its "spatial coherence" relative to a tissue's geometry.

## Why BioRSP?

- **Reference-Free**: Doesn't require a reference atlas.
- **Geometry-Aware**: Accounts for tissue boundaries and shape.
- **Interpretable**: Metrics like "Directionality" map directly to biological concepts (e.g., gradients).

## Core Philosophy

BioRSP treats each gene as a signal on a radar. By analyzing the "radar profile" of gene expression from a central vantage point, we can distinguish between:

1. **Global Gradients** (Directional)
2. **Basal/Edge Patterns** (Radial)
3. **Patchy/Focal Expressions** (Sparse but organized)
4. **Noise** (Randomly scattered)
