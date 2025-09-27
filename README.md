> [!NOTE]
> I spent the past few months refining BioRSP in response to invaluable advice on usability, reproducibility, and design from conference attendees, collaborators, and early users during experimentation. The current roadmap can be found [here](https://github.com/cytronicoder/biorsp/issues/1). Thanks to everyone who provided feedback and helped improve the project!

### Motivation

Single-cell genomics datasets are often visualized with 2D embeddings (e.g., UMAP, t-SNE, PHATE) to reveal structure and heterogeneity. However, interpreting these visualizations is often subjective and lacks statistical rigor.

Most tools still analyze 2D embeddings by eyeballing coverage, density, and shape, then backing that impression with cluster DE or a black-box classifier. These approaches rarely quantify directional or radial structure and often miss multiscale patterns (narrow hot spots vs broad lobes) or overinterpret density artifacts.

### What is BioRSP?

BioRSP is a Python package designed to convert 2D embeddings of single-cell data into polar coordinates and perform a multiscale "radar" sector scan to identify regions of feature enrichment.

BioRSP addresses the aforementioned issues of interpretation and statistical rigor by providing a systematic, multiscale method to scan 2D embeddings for regions enriched in specific features (e.g., gene expression, protein levels, metadata). By converting to polar coordinates and performing sector scans at multiple scales, BioRSP can identify both narrow and broad patterns of enrichment that might be missed by traditional methods.
