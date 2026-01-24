# Diagnostics

BioRSP includes a small set of diagnostic utilities for assessing stability of the radar profiles and scalar summaries.

## Subsampling robustness

The module `biorsp.core.robustness.compute_robustness_score` computes subsampling diagnostics:

- Subsample a fraction of cells repeatedly.
- Recompute the radar profile and scalar summaries.
- Report the mean Pearson correlation between subsampled and full profiles.
- Report the coefficient of variation for anisotropy across subsamples.

These metrics are intended to identify unstable signals caused by sparse support or embedding noise.

## Diagnostics not currently exposed

Split-half reproducibility, cross-embedding sensitivity, and vantage sensitivity are not exposed as public APIs in the current package. If you need these diagnostics, treat them as intended extensions and validate them against your own workflow before drawing conclusions.
