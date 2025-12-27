# Usage guide

This guide mirrors the Methods workflow. Where the Methods intentionally omits operational details (e.g., exact CLI arguments), a short TODO placeholder is provided.

## Inputs you need

- Coordinates `z` (2D embedding coordinates for each cell)
- Feature values `x` (gene expression or other scalar feature per cell)
- Cell set `S` (user-specified subset of cells to analyze)
- Optional: library sizes `u` and donor/sample labels `d(i)` for stratified inference and diagnostics

## Default workflow

1. Choose the cell set `S` to analyze (e.g., annotated cell type or metadata filter).
2. Compute vantage `v` (default: geometric median of `{z_i : i in S}`).
3. Define foreground `y_i^{(g)}` using the 90th-percentile threshold within `S` (binary default).
4. Compute the radar curve `R_g(θ)` on an angular grid (default `B = 360`) with sector width `Δ = 20°`.
5. Compute scalar summaries: coverage `c_g`, anisotropy `A_g`, peak direction `θ_g^*`, peak strength `P_g`.
6. Apply adequacy filters:
   - Per-sector minima: `n_fg^min = 10`, `n_bg^min = 50`
   - Gene-level minimum total foreground: `n_fg,tot^min = 100`
7. Run stratified permutation inference (UMI-decile stratification `Q = 10`, `K = 200` exploratory / `K = 1000` final).
8. Apply multiple-testing correction (Benjamini-Hochberg on genes passing adequacy).
9. Run diagnostics (split-half reproducibility, subsampling stability, cross-embedding sensitivity, VSI).

## Command-line examples

- TODO: add CLI usage examples (useful if a stable `biorsp` command-line interface is available).
- If you use the package programmatically, follow the usage pattern in the example notebooks in `examples/`.

## Notes and cautions

- BioRSP is not clustering or trajectory inference; use published labels or user-defined subsets to define `S`.
- The default constants (listed in theory) are chosen for robustness; they should be treated as sensible defaults rather than universally optimal settings.
