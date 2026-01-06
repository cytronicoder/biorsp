# Diagnostics

This document reproduces the diagnostics specified in the Methods and provides minimal guidance on interpretation.

## Split-half reproducibility within donor/sample

For each donor with at least $n_{\min}=500$ cells in `S`, perform repeated split-half resampling (20 repeats by default). Cells are randomly divided into two equal halves, and $A_g$ and $\theta_g^*$ are recomputed. Report:

- Spearman correlation of $A_g$ across halves.
- Directional agreement measured by mean cosine similarity $\cos(\theta_{g,1}^*-\theta_{g,2}^*).$

Low agreement indicates potential instability due to sparse sectors, boundary effects, or embedding noise.

## Subsampling stability

Repeatedly downsample groups to a common size (default 80% of the minimum group size) and recompute $A_g$. Report:

- Rank correlations across resamples.
- Coefficient of variation:
  $$\mathrm{CV}(A_g)=\frac{\mathrm{SD}(A_g)}{\mathrm{mean}(A_g)+\varepsilon}.$$

## Cross-embedding sensitivity

When multiple embeddings are available (UMAP, t-SNE, PCA 2D), compute BioRSP metrics separately for each. Because $\theta_g^*$ depends on global orientation, compare directions only after explicit alignment when feasible (e.g., Procrustes rotation). Report rank correlations of $A_g$ and aligned directional agreement; unaligned cosine similarity may be provided as a conservative lower bound.

## Vantage sensitivity index (VSI)

Evaluate a deterministic set of alternative vantages derived from `{z_i : i in S}`, recompute $A_g$ for each vantage, and report:
$$\mathrm{VSI}_g = \frac{\mathrm{SD}(A_g^{(v)})}{\mathrm{mean}(A_g^{(v)})+\varepsilon}.$$
High VSI indicates anisotropy is not identifiable from a single origin and should be interpreted cautiously.

## Interpretation guidance

- Treat diagnostics as indicators rather than hard pass/fail rules. Low reproducibility or high VSI suggests cautious interpretation.
- Underpowered genes or sectors should be visualized but excluded from formal inference (see adequacy rules in theory).
