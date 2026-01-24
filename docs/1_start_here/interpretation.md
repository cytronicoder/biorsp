# Interpreting scores

This page summarizes the primary per-gene outputs returned by the API. All values depend on the embedding supplied by the user.

## Coverage

- Column: `Coverage`
- Interpretation: fraction of cells classified as foreground for the gene. Values lie in `[0, 1]`.
- Low coverage indicates rare expression; high coverage indicates widespread expression.

## Spatial organization score

- Column: `Spatial_Bias_Score`
- Interpretation: non-negative summary of directional radial structure. Larger values indicate stronger directional structure in the embedding.
- The score is embedding-specific; interpretation should be tied to the biological meaning of the chosen embedding.

## Directionality

- Column: `Directionality`
- Interpretation: signed radial summary derived from the radar profile. Positive and negative values indicate different radial biases (e.g., core vs. rim), depending on the embedding and foreground definition.

## Archetypes

- Column: `Archetype`
- BioRSP assigns archetypes by comparing each gene’s coverage and spatial score to thresholds (`C_cut`, `S_cut`).
- If thresholds are data-derived (e.g., from benchmarks), interpret archetypes relative to the training data and split described in the run report.
