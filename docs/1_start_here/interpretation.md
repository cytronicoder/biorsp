# Interpreting Scores

BioRSP outputs three primary standardized metrics. Here is how to interpret them.

## 1. Coverage

_Column: `Coverage`_

The fraction of cells in the region of interest (ROI) that express the gene above a background threshold.

- **Low (<0.05)**: Rare cell types or noise.
- **High (>0.8)**: Ubiquitous housekeeping genes.

## 2. Spatial Score

_Column: `Spatial_Score`_

A measure of how "organized" the expression is, ranging from 0 to 1 (or higher for very strong signals).

- **0.0 - 0.1**: Random/Noise.
- **0.1 - 0.2**: Weak spatial pattern.
- **> 0.2**: Strong spatial pattern.

**Note**: A gene with high Coverage but low Spatial Score is likely ubiquitous and uniform (not spatially variable).

## 3. Directionality

_Column: `Directionality`_

Indicates if the spatial pattern is one-sided (gradient).

- **Near 0**: Symmetric pattern (e.g., center of tissue, or ubiquitous).
- **High Abs Value**: Strong gradient towards one side.

## 4. Archetypes

_Column: `Archetype`_

Classifications derived from the metrics above:

- **I: Ubiquitous**: High Coverage, Low Spatial Score.
- **II: Gradient**: High Spatial Score, High Directionality.
- **III: Patchy**: Low Coverage, High Spatial Score.
- **IV: Basal/Edge**: Geometry-specific patterns.
