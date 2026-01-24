# BioRSP: Overview

BioRSP (Biological Radar Scanning Plots) is a geometry-first framework for quantifying directional radial enrichment of a foreground population in a two-dimensional embedding. Given a user-supplied cell set and a foreground definition (for example, high expression of a gene), BioRSP scans the embedding by angle with respect to a fixed vantage point, compares foreground and background radial distributions within angular sectors, and reports a radar curve summarizing radial enrichment as a function of direction together with compact scalar summaries and diagnostics.

## Method at a glance

- Inputs:
  - Coordinates: `z_i` (2D embedding coordinates for each cell)
  - Feature values: `x_i^{(g)}` (gene expression or other scalar per cell)
  - Cell set: `S` (user-defined subset of cells to analyze)
  - Vantage: `v` (reference point; default geometric median)
- Foreground definition: binary (default) using within-`S` 90th-percentile threshold for each gene (`y_i^{(g)} = 1(x_i^{(g)} > t_g)`), with optional soft-weighted mode using logistic weighting
- Radar function: `R_g(θ)` — per-angle standardized Wasserstein-based radial contrast between foreground and background
- Summaries: coverage `c_g`, anisotropy `A_g`, peak direction `θ_g^*`, peak strength `P_g`
- Adequacy rules: per-sector foreground/background minimums and a minimum total foreground to enable inference
- Permutation inference: stratified by library size (Q = 10 deciles) with K = 200 exploratory / K = 1000 final
- Diagnostics: split-half reproducibility, subsampling stability, cross-embedding sensitivity, vantage sensitivity index (VSI)
- Outputs recorded in the run manifest (vantage, parameters, adequacy flags, seeds, and checksums)

## What BioRSP is not

BioRSP is neither a clustering nor a trajectory method. We treat published labels and user-defined subsets as strata for analysis and ask whether a known foreground exhibits reproducible directional structure beyond sampling variability and technical confounders.

## Reproducible outputs

Each analysis produces:

- Tabular outputs (per-gene summaries, diagnostics)
- Radar curves `R_g(θ)` (evaluated on a grid; optional 5° smoothing for visualization)
- Diagnostic summaries (split-half, subsampling, VSI, cross-embedding)
- Publication-ready figures
- A machine-readable JSON run manifest containing software version, parameters, seeds, and checksums to support exact reproduction

See also:

- Theory and methods: [`docs/theory.md`](docs/theory.md)
- Usage guide: [`docs/usage.md`](docs/usage.md)
- Diagnostics: [`docs/diagnostics.md`](docs/diagnostics.md)
