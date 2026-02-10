# BioRSP genome-wide addendum

## Purpose

This addendum defines a genome-wide, donor-aware extension of the preregistered BioRSP evaluation plan. It preserves donor-stratified inference and within-stratum analysis, while enabling discovery across all genes.

## Two-stage screening

- Stage 1 (screen): all genes within each inferential stratum are evaluated on a reference embedding (UMAP seed0) with a reduced number of donor-stratified permutations (default 100). Stage 1 is used only to select candidates and is not used for final claims.
- Stage 2 (confirm): candidates only are re-evaluated with a larger number of donor-stratified permutations (default 1000). Stage 2 p-values are the only basis for FDR and robust-hit decisions.

## Expression modes

- Continuous mode (default): log-normalized expression is used to compute a continuous-weight RSP, where E(phi) measures enrichment of expression mass relative to uniform cell density across angles.
- Binary validation: for candidates, foreground definitions t0 (expr>0), q90, and q95 are computed to validate stability against dropout thresholds.

## Filtering and performance

- Genes are filtered within each stratum based on detection fraction and detection count (configurable), and optional variance thresholding. Filtering criteria and counts removed are recorded in `outputs/results/gene_filtering.csv`.
- Embeddings, kNN graphs, donor indices, and stage results are cached per stratum; stage1 and stage2 CSVs are written under `outputs/results/genomewide/<stratum>/` and can be resumed.

## Inference and robustness

- Significance is donor-stratified permutation on E_max with BH-FDR within each stratum (Stage 2 only).
- Robust directional signal requires FDR <= 0.05, embedding stability (phi SD), Moran’s I above random-gene baseline, and donor jackknife robustness.

## Outputs

- `outputs/results/biorsp_genomewide_results.csv`
- `outputs/results/biorsp_genomewide_top_hits.csv`
- `outputs/results/gene_filtering.csv`
- Figures under `outputs/figures/genomewide/<stratum>/` with UMAP overlays, RSP polar profiles, null histograms, phi stability plots, Moran vs E_max scatter, and volcano-style plots.
