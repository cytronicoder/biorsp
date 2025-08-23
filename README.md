# BioRSP

**BioRSP (Biological Radar Scanning Plot)** is a framework for finding directional (anisotropic) patterns of gene expression across 2D single-cell embeddings. Instead of naively determining marker genes solely based on coverage (defined as the number of cells expressing a gene above a threshold in a given region), we ask if a gene (or gene program) is enriched in a particular angular sector of the embedding beyond what random structure would explain.

At the core of BioRSP is **ANCHOR** (**AN**isotropy-**C**orrected **H**ypothesis **OR**iented method). ANCHOR:

1. Corrects for uneven cell density and irregular occupied area.
2. Adapts bin sizes in sparse regions so rare or transitional states are not diluted.
3. Fits a periodic spline-based negative binomial model to capture smooth angular trends.
4. Calibrates significance with permutation so anisotropy scores have reliable p-values/FDR.

To make results interpretable and biologically actionable, BioRSP introduces **two complementary metrics**:

- **A1:** the _effective angular breadth_ of a gene’s distribution, computed as an entropy- or Simpson-based effective width on the exposure-corrected angular density. A1 values near 1 indicate diffuse, isotropic expression (typical of **housekeeping genes**), while values near 0 indicate restricted or focal expression (potential **localized markers**).

- **A2:** the _directional inequality_ of expression, combining the first circular moment (resultant length) with a normalized Gini coefficient. A2 values near 0 indicate uniform, isotropic distribution, while values near 1 indicate sharp polarization in one or few angular sectors (candidate **biomarkers or state-specific genes**).

By computing A1 and A2 for every gene after density correction, BioRSP provides a **principled 2D map** of gene types:

| A2 / A1 | Low A1                                         | High A1                                             |
| ------- | ---------------------------------------------- | --------------------------------------------------- |
| Low A2  | Weak or noisy signal; _likely no significance_ | Broad, isotropic; _housekeeping_                    |
| High A2 | Focal, polarized; _biomarker candidate_        | Widespread but directional; _axis/gradient markers_ |

Researchers can use BioRSP to classify genes into housekeeping vs. biomarker vs. no-signal categories with reproducible, density-corrected scores. We provide biologists with clear, interpretable criteria to prioritize markers that can highlight biological features that broad, density-biased marker scans often miss.
