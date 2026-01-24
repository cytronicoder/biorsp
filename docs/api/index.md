# API overview

The public API is defined in `biorsp.api` and re-exported at the package level.

## Primary functions

- `score_genes(adata, genes, embedding_key=...)`: compute coverage and spatial organization scores.
- `classify_genes(df, c_cut=None, s_cut=None, fdr_cut=0.05)`: assign archetype labels based on thresholds.
- `score_gene_pairs(adata, genes, embedding_key=...)`: compute pairwise scores for gene-gene patterns.

See docstrings in `biorsp/api.py` for parameter details.
