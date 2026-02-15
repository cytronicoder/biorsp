# Limitations And Negative Controls

## Core Limitations

- Directionality is embedding-conditional and can change under embedding rotation/reflection.
- Case-study outputs are method-validation diagnostics, not discovery claims.
- Stability diagnostics do not prove biological truth.

## Confound Risks

- library size/depth gradients
- mitochondrial burden
- ribosomal content or proxies
- donor imbalance

## Required Negative Controls In This Case Study

- high `pct_counts_mt` directional profile
- ribosomal high-cell profile (or total-count proxy if ribosomal features unavailable)
- within-strata permutation sanity diagnostic

## Interpretation Guardrails

Treat results as limited when donor replication is absent and library-size strata fallback is used.
