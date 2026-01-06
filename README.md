### BioRSP: Nonparametric Inference and Robustness for Radar Scanning Plots

BioRSP is a geometry-aware framework for quantifying angle-resolved radial bias of features in 2D coordinate spaces, including UMAP/t-SNE embeddings and physical x-y coordinates. It summarizes each gene with two interpretable scores—prevalence and spatial organization—and supports optional permutation-based significance testing.

BioRSP characterizes each gene using two complementary scores:

1. Coverage score (C)  
   The fraction of cells whose expression is at or above a biologically meaningful threshold.

2. Spatial organization score (S)  
   The weighted RMS radial shift of high-expression cells relative to background across angular sectors.

These two scores enable a simple 2×2 archetype view:

| Score profile  | Archetype              | Interpretation in embedding space                                                                                                    |
| -------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| High C, Low S  | `housekeeping_uniform` | Broadly expressed with weak spatial organization (often housekeeping, though QC/technical effects can modulate apparent uniformity). |
| Low C, High S  | `niche_biomarker`      | Restricted to a subset of cells and spatially concentrated within the embedding (e.g., a substate or subdomain).                     |
| High C, High S | `localized_program`    | Broadly expressed but structured across embedding regions (e.g., gradients or state programs).                                       |
| Low C, Low S   | `sparse_presence`      | Rare/low signal with weak organization (often indistinguishable from noise).                                                         |

> [!NOTE]
>
> - "Spatial" here refers to structure in the provided coordinate system. In embeddings, this reflects organization in the learned manifold, not necessarily physical tissue space.
> - $$S$$ is an effect size. If you enable permutations, BioRSP can additionally report p-values/FDR to support "non-random" claims.

#### Quickstart

```python
import anndata as ad
import biorsp

# 1) Load data
adata = ad.read_h5ad("my_data.h5ad")

# 2) Configure (recommended defaults)
config = biorsp.BioRSPConfig(
    delta_deg=60,
    B=72,
    empty_fg_policy="zero",
)

# 3) Score genes (choose an embedding key appropriate for your AnnData)
df = biorsp.score_genes(
    adata,
    genes=["GeneA", "GeneB"],
    embedding_key="X_umap",
    config=config,
)

# 4) Classify into archetypes
df = biorsp.classify_genes(df, c_cut=0.10)
print(df[["gene", "coverage_expr", "spatial_score", "archetype"]])
```

#### Recommended Defaults

| Parameter                        | Default | Guidance                                                                                                                                                                             |
| -------------------------------- | ------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Delta (Δ, degrees)               |      60 | Good general-purpose window width for cell-type subsets. Use smaller Δ only when the subset is large enough to maintain sector support; otherwise expect low coverage or abstention. |
| B (theta bins)                   |      72 | 5° spacing is a good compromise between angular resolution and stability.                                                                                                            |
| Internal foreground quantile (q) |    0.90 | Used internally for stable spatial scoring and is distinct from coverage (which uses an expression threshold).                                                                       |
| Empty foreground policy          |  `zero` | Treats background-supported sectors with zero foreground as no signal rather than missing data, reducing selection bias for localized patterns.                                      |

#### Gene-Gene Spatial Relationships

BioRSP supports gene-gene similarity analysis by comparing radar profiles $$R_g(\theta)$$ on a shared, background-supported angular grid.

```python
df_pairs = biorsp.score_gene_pairs(
    adata,
    genes=my_gene_list,
    embedding_key="X_umap",
    config=config,
)

# Common starting point:
# - high similarity profile: similar spatial organization patterns
# - high copattern score: similarity adjusted for sign agreement and shared support
```

If you want "modules," cluster the similarity matrix derived from `df_pairs` (e.g., hierarchical clustering or a graph community method).

#### Installation

For local development:

```bash
pip install .
```

For a clean install in editable mode (recommended during development):

```bash
pip install -e .
```

#### What BioRSP is designed to capture

BioRSP is most directly sensitive to radial organization (core vs rim bias) and localized vs global support across angles, summarized in $$S$$ and coverage diagnostics. It is not, by itself, a general angular density-enrichment detector unless you explicitly add an angular concentration metric.
