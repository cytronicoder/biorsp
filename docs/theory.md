# Theory and Methods

Below we reproduce the Methods section verbatim (headings and equations preserved) from the BioRSP manuscript. Minimal connective text is included only to improve navigation.

## Inputs and notation

We analyze a single-cell or single-nucleus dataset with $$N$$ cells indexed by $$i \in \{1,\dots,N\}.$$ Each cell $$i$$ is associated with a two-dimensional coordinate $$z_i \in \mathbb{R}^2$$, representing a user-supplied low-dimensional embedding or projection of the dataset (e.g., UMAP, t-SNE, or a user-defined 2D layout). We do not assume that Euclidean distances in this embedding are globally faithful. Instead, the embedding is treated strictly as a coordinate system for defining angular directions and radial distances relative to a reference point.

For each gene $$g$$, let $$x_i^{(g)} \ge 0$$ denote the expression value of gene $$g$$ in cell $$i$$, after the user’s chosen normalization (e.g., log-normalized counts). More generally, $$x_i$$ may represent any scalar feature aligned to cells, including protein abundance, chromatin accessibility scores, module scores, or pathway activity scores. All definitions below apply unchanged provided the feature values are numeric and comparable within the analyzed cell set.

All analyses are performed within a user-defined subset of cells $$S \subseteq \{1,\dots,N\}$$, typically defined by published annotations or user-specified metadata filters. BioRSP does not compute clusters or trajectories; the cell set $$S$$ is defined externally, and all quantities are computed using only cells in $$S.$$

When multi-sample structure is available, donor or sample membership is denoted by d(i), and library size (total counts for the relevant modality) by $$u_i.$$ These quantities are used only for stratified permutation inference and robustness diagnostics, not for defining the radar statistic itself.

A key geometric input is a vantage point $$v \in \mathbb{R}^2$$, which serves as the origin for radar scanning. By default, $$v$$ is set to the geometric median of $$\{z_i : i \in S\}$$, which is more robust to outliers and irregular boundaries than the mean. The geometric median is computed using a standard iterative procedure with a fixed convergence tolerance, recorded in the run manifest. All reported results include the chosen vantage point, and sensitivity to the choice of $$v$$ is quantified using a vantage sensitivity diagnostic (Section 2.5). Alternative vantages are used only for sensitivity analyses and are not used to tune reported results.

## Foreground definition

BioRSP requires an operational definition of “high expression” to define a foreground subset of cells for each gene. The default approach uses a within-cell-set quantile threshold to stabilize foreground size across genes and mitigate sparsity-driven artifacts.

### Binary foreground (default)

For gene $$g$$ within cell set $$S$$, we define a binary foreground indicator
$$y_i^{(g)} = \mathbf{1}!\left(x_i^{(g)} > t_g\right)$$
$$t_g = Q_{0.90}!\left(\{x_i^{(g)} : i \in S\}\right)$$
where $$Q_{0.90}$$ denotes the 90th percentile, and $$\mathbf{1}(\cdot)$$ is the indicator function (i.e., $$y_i^{(g)}=1$$ when $$x_i^{(g)}$$ exceeds the threshold $$t_g$$, and $$y_i^{(g)}=0$$ when it does not).

This choice fixes the expected foreground coverage near 10% (up to ties), improving comparability across genes and preventing low-coverage genes from producing unstable sector estimates. Because single-cell data frequently contain ties at low expression, the threshold is applied using a strict inequality ($$x_i^{(g)} > t_g$$). We report the realized foreground fraction
$$c_g = \frac{1}{|S|}\sum_{i\in S} y_i^{(g)}$$
to make the effective foreground size explicit. If $$c_g = 0$$, the gene is treated as underpowered and excluded from downstream inference.

Users may override the quantile threshold for sensitivity analyses; reported conclusions are expected to remain stable under modest changes in this setting for adequately expressed genes.

### Optional soft foreground weighting

As an optional robustness mode, we allow continuous foreground weights
$$w_i^{(g)} = \sigma!\left(\frac{x_i^{(g)}-\mu_g}{s_g}\right)$$
where $$\sigma$$ is the logistic function, $$\mu_g$$ is the median expression of gene $$g$$ in $$S$$, and $$s_g$$ is the median absolute deviation plus a small constant to avoid division by zero. In this mode, sector-wise radial comparisons are computed using weighted empirical distributions with the same Wasserstein-based statistic. The precise estimator for weighted Wasserstein distance is defined in the software documentation and recorded in the run manifest.

This soft mode is intended as a sensitivity check rather than the default, as binary foreground definitions align more directly with common biological interpretations and simplify null model construction.

## Radar Scanning Plot estimand

### Polar coordinates and angular windows

For each cell $$i \in S$$, we define polar coordinates relative to the vantage point $$v$$:
$$r_i = \|z_i - v\|_2$$
$$\theta_i = \mathrm{atan2}(z_{i,y} - v_y,\ z_{i,x} - v_x) \in [-\pi,\pi)$$
Let $$\Theta = \{\theta^{(1)},\dots,\theta^{(B)}\}$$ be an equally spaced grid of angles on $$[-\pi,\pi)$$. Unless stated otherwise, we use $$B = 360$$ (1° spacing). For each grid angle $$\theta \in \Theta$$, we define an angular window (sector) of width $$\Delta$$ radians:
$$\mathcal{W}(\theta) = \left\{ i \in S : \mathrm{dist}_{\mathbb{S}^1}(\theta_i,\theta) \le \frac{\Delta}{2} \right\}$$
where
$$\mathrm{dist}_{\mathbb{S}^1}(\alpha,\beta) = \min_{k\in\mathbb{Z}}|\alpha - \beta + 2\pi k|$$
denotes wrapped angular distance on the unit circle. All computations use radians internally; degrees are used only for reporting. We use $$\Delta = 20^\circ$$ by default, which typically yields adequate per-sector counts in modestly sized cell sets while retaining directional resolution.

### Foreground/background radial comparison and radar radius function

For a fixed gene $$g$$, cells in $$S$$ are partitioned into a foreground set
$$F_g = \{ i \in S : y_i^{(g)} = 1 \}$$
and a background set
$$B_g = S \setminus F_g.$$
Within each angular window $$\mathcal{W}(\theta)$$, we compare the radial distributions
$$\mathcal{R}_F(\theta) = \{r_i : i \in \mathcal{W}(\theta)\cap F_g\}$$
$$\mathcal{R}_B(\theta) = \{r_i : i \in \mathcal{W}(\theta)\cap B_g\}$$
Let $$W_1(\mathcal{R}_F,\mathcal{R}_B)$$ denote the one-dimensional Wasserstein-1 (earth mover’s) distance between the two empirical samples of radii, computed directly from sorted values without binning. We define the radar radius function
$$R_g(\theta) \;=\; s(\theta)\,\frac{W_1(\mathcal{R}_F(\theta),\,\mathcal{R}_B(\theta))}{\mathrm{IQR}(\mathcal{R}_B(\theta))+\varepsilon}$$
where $$\mathrm{IQR}$$ denotes the interquartile range, $$\varepsilon = 10^{-8}$$ prevents division by zero, and
$$s(\theta) = \mathrm{sign}!\left(\mathrm{median}(\mathcal{R}_F(\theta)) - \mathrm{median}(\mathcal{R}_B(\theta))\right)$$

The sign encodes whether foreground cells are radially closer to the vantage point ($$R_g(\theta) < 0$$) or farther away ($$R_g(\theta) > 0$$) than background cells in that direction. Standardizing by $$\mathrm{IQR}(\mathcal{R}_B(\theta))$$ improves comparability across sectors and genes under heterogeneous embedding scales and local densities.

The function $$R_g(\theta)$$ is evaluated on the grid $$\Theta$$. For visualization only, we optionally apply a circular moving-average smoother with a 5° window after masking underpowered sectors; all inferential quantities are computed from the unsmoothed values.

## Scalar summaries

Each gene’s radar function is summarized using three primary scalars.

**Coverage.** The fraction of cells in $$S$$ classified as foreground:
$$c_g = \frac{1}{|S|}\sum_{i\in S} y_i^{(g)}$$

**Anisotropy magnitude (primary score).** Let $$\Theta_g \subseteq \Theta$$ denote the set of directions for which $$R_g(\theta)$$ is defined (i.e., sectors meeting adequacy criteria; see below). The anisotropy score is
$$A_g = \left(\frac{1}{B}\sum_{\theta\in \Theta} R_g(\theta)^2\right)^{\tfrac{1}{2}}$$
If $$|\Theta_g| = 0$$, $$A_g$$ is set to missing and the gene is excluded from ranking and inference.

**Peak direction and strength.** The direction of strongest relative concentration toward the vantage point is
$$\theta_g^* = \arg\min_{\theta\in \Theta} R_g(\theta)$$
$$P_g = \min_{\theta} R_g(\theta)$$

These summaries support stable ranking (via $$A_g$$) and interpretable localization (via $$\theta_g^*$$ and $$P_g$$) without requiring clustering, pseudotime inference, or neighborhood graph construction.

## Power and adequacy rules

Directional sector statistics are unreliable when computed from too few foreground or background cells. We therefore impose explicit adequacy criteria and treat under-supported sectors as missing.

For each direction $$\theta$$, define the number of foreground cells in the sector
$$n_{\mathrm{fg}}(\theta)=\left|\mathcal{W}(\theta)\cap F_g\right|$$
and the number of background cells in the sector
$$n_{\mathrm{bg}}(\theta)=\left|\mathcal{W}(\theta)\cap B_g\right|.$$
A sector is considered adequate only if $$n_{\mathrm{fg}}(\theta)\ge n_{\mathrm{fg}}^{\min}$$ and $$n_{\mathrm{bg}}(\theta)\ge n_{\mathrm{bg}}^{\min}$$, with default thresholds $$n_{\mathrm{fg}}^{\min}=10$$ and $$n_{\mathrm{bg}}^{\min}=50.$$ If a sector fails these criteria, $$R_g(\theta)$$ is set to missing and excluded from all scalar summaries. We report the fraction $$\tfrac{|\Theta_g|}{|\Theta|}$$ as a diagnostic of local support.

In addition, gene-level inference requires a minimum total number of foreground cells
$$|F_g| \ge n_{\mathrm{fg,tot}}^{\min}$$
with default $$n_{\mathrm{fg,tot}}^{\min}=100.$$ Genes failing this criterion are labeled underpowered and excluded from p-value and FDR reporting, although they may still be visualized cautiously. These thresholds are conservative by design and prioritize stability and false-positive control.

## Inference

Inference is performed on the gene-level anisotropy statistic $$A_g$$ using permutation tests. The null hypothesis is that, conditional on sequencing depth (and donor structure when applicable), foreground membership is independent of spatial position in the embedding.

### Default null: UMI-stratified permutation

Within a fixed cell set $$S$$ and gene $$g$$, embedding coordinates $$\{z_i\}$$ and expression values $$\{x_i^{(g)}\}$$ are held fixed. Cells are partitioned into $$Q=10$$ strata based on the library size $$u_i$$ (deciles within $$S$$). Foreground indicators $$y_i^{(g)}$$ are permuted independently within each stratum, preserving the foreground count per stratum and thus the depth–foreground relationship expected under technical confounding.

For each permutation $$k=1,\dots,K$$, we recompute $$R_g^{(k)}(\theta)$$ and $$A_g^{(k)}$$ using the same adequacy rules. The empirical one-sided p-value is
$$p^{\mathrm{strat}}_g = \frac{1 + \sum_{k=1}^K \mathbf{1}\left(A_g^{(k)} \ge A_g\right)}{1+K}.$$
We use $$K=200$$ permutations for exploratory analyses and $$K=1000$$ for final reporting.

### Donor-stratified null (optional)

When donor or sample labels $$d(i)$$ are available, permutations may be further stratified within the Cartesian product of donor and UMI bin to avoid donor-driven confounding. A minimum per-stratum cell count (default 50 cells) is required; strata failing this requirement are merged or fall back to UMI-only stratification according to a deterministic rule recorded in the run manifest.

### Multiple testing correction

Within each analyzed cell set $$S$$, we apply the Benjamini-Hochberg procedure across genes that meet gene-level adequacy criteria ($$|F_g| \ge n_{\mathrm{fg,tot}}^{\min}$$ and $$|\Theta_g|>0$$). We report q-values $$q_g$$ derived from $$p_g^{\mathrm{strat}}$$.

For transparency, we also report a naive permutation p-value $$p_g^{\mathrm{naive}}$$ obtained by permuting foreground labels uniformly within $$S$$ without depth stratification. Genes significant only under the naive null are flagged as likely depth-driven.

## Robustness diagnostics

Because geometric signals can be sensitive to sampling, embedding choice, and reference origin, BioRSP reports robustness diagnostics as first-class outputs.

### Split-half reproducibility within donor/sample

For each donor with at least $$n_{\min}=500$$ cells in S, we perform repeated split-half resampling (20 repeats by default). Cells are randomly divided into two equal halves, and $$A_g$$ and $$\theta_g^*$$ are recomputed. We report:

- Spearman correlation of $$A_g$$ across halves.
- Directional agreement measured by mean cosine similarity $$\cos(\theta_{g,1}^*-\theta_{g,2}^*).$$

Low agreement indicates potential instability due to sparse sectors, boundary effects, or embedding noise.

### Subsampling stability

To assess sensitivity to cell-count variation, we repeatedly downsample groups to a common size (default 80% of the minimum group size) and recompute $$A_g$$. We report rank correlations across resamples and the coefficient of variation
$$\mathrm{CV}(A_g)=\frac{\mathrm{SD}(A_g)}{\mathrm{mean}(A_g)+\varepsilon}.$$

### Cross-embedding sensitivity

When multiple embeddings are available (UMAP, t-SNE, PCA 2D), BioRSP metrics are computed separately for each. Because $$\theta_g^*$$ depends on global orientation, directional comparisons across embeddings are reported only after explicit alignment when feasible (e.g., Procrustes rotation on shared cell coordinates). Rank correlations of $$A_g$$ and aligned directional agreement are reported; unaligned cosine similarity is provided as a conservative lower bound.

### Vantage sensitivity index

To quantify sensitivity to the choice of vantage point, we evaluate a fixed, deterministic set of alternative vantages derived from $$\{z_i : i\in S\}$$, in addition to the geometric median. For each vantage $$v$$, we recompute $$A_g$$ and define the vantage sensitivity index
$$\mathrm{VSI}_g = \frac{\mathrm{SD}(A_g^{(v)})}{\mathrm{mean}(A_g^{(v)})+\varepsilon}.$$
High $$\mathrm{VSI}_g$$ indicates that anisotropy is not identifiable from a single origin and should be interpreted cautiously.

## Implementation details

### Default constants

Unless otherwise specified:

- Angular grid: $$B=360.$$
- Sector width: $$\Delta=20^\circ.$$
- Minimum per-sector support: $$n_{\mathrm{fg}}^{\min}=10$$, $$n_{\mathrm{bg}}^{\min}=50.$$
- Minimum total foreground: $$n_{\mathrm{fg,tot}}^{\min}=100.$$
- Permutations: $$K=200$$ (exploratory), $$K=1000$$ (final).
- Library-size strata: $$Q=10.$$
- Numerical stabilization: $$\varepsilon=10^{-8}.$$
- Visualization smoothing: 5° circular moving average.

All realized per-sector supports and adequacy fractions are recorded and reported.

### Runtime complexity and scaling

Let $$n=|S|$$ and $$G$$ the number of genes. Polar coordinate computation is $$O(n)$$. Sector extraction is accelerated by sorting cells by angle and using sliding windows, yielding approximately $$O(n+B)$$ per gene for fixed $$\Delta$$. Sector-wise Wasserstein computations scale linearly in sector size, giving $$O(n)$$ per gene. Permutation inference scales as $$O(GKn)$$ in the worst case, but genes failing adequacy criteria are filtered prior to permutation testing. Memory usage is $$O(n)$$ plus $$O(G)$$ summaries; permutation results are streamed.

### Determinism and reproducible outputs

All stochastic components (permutations, subsampling, split-halves) are controlled by explicit random seeds recorded in output metadata. Each analysis produces tabular outputs, radar curves, diagnostic summaries, publication-ready figures, and a machine-readable JSON run manifest containing software version, parameters, dataset identifiers, and checksums to support exact reproduction.
