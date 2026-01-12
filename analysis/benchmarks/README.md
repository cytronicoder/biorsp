# BioRSP Method Validation Through Simulation

This directory contains a comprehensive simulation framework for rigorous validation of the BioRSP method, which quantifies spatial gene organization through two complementary metrics: the Spatial Organization Score ($S_g$) and Coverage Score ($C_g$). The validation strategy employs synthetic datasets with known ground truth to systematically evaluate statistical properties, classification performance, robustness characteristics, and sensitivity to spatial relationships across diverse biological scenarios.

## Methodological Framework

### Validation Strategy

We designed a four-pronged validation approach to address fundamental statistical requirements and biological realism:

1. **Statistical Calibration**: Verification of type I error control under multiple null hypotheses, including spatially-structured confounders that violate standard independence assumptions
2. **Archetype Classification**: Assessment of discriminative power across biologically-motivated spatial patterns spanning uniform housekeeping expression to highly localized niche markers
3. **Robustness Analysis**: Characterization of metric stability under geometric transformations and expected failure modes
4. **Pairwise Relationships**: Evaluation of gene-gene co-patterning detection for identifying functional gene modules and spatial neighborhoods

### Computational Infrastructure

The simulation framework consists of:

- **`simlib/`**: A modular Python library implementing reproducible dataset generation, geometric transformations, expression models, and standardized evaluation metrics
- **`benchmarks/`**: All validation experiments including:
  - **Story Figure**: One-page methods paper figure validating core 2×2 archetype model
  - **Core Benchmarks**: Statistical calibration, archetype recovery, gene-gene co-patterning, robustness analysis
  - **Supporting Analysis**: Null calibration, cross-embedding stability, abstention evaluation
- **`outputs/`**: Structured results storage with CSV data tables, JSON metadata manifests, and automated markdown reports
- **`tests/`**: Unit test suite ensuring correctness of simulation primitives
- **`scripts/`**: Utility tools for smoke testing and result verification

## Simulation Primitives

The `simlib` package provides biologically-grounded building blocks:

**Tissue Geometries**: Six archetypal shapes representing diverse tissue architectures (disk, ellipse, crescent, peanut, annulus, disconnected blobs)

**Expression Patterns**: Nine spatial archetypes spanning the biological spectrum from uniform housekeeping (no spatial structure) through localized programs (wedge, core, rim) to sparse niche markers

**Confounding Models**: Four null models that preserve spatial structure in non-expression covariates (library depth, cell density) to test robustness against common technical artifacts

**Geometric Distortions**: Six transformations including rotation-invariant operations (rotate, jitter, subsample) and symmetry-breaking perturbations (anisotropic scaling, radial warping) to probe expected invariances and sensitivities

**Reproducibility**: All random number generation uses NumPy's `SeedSequence` entropy system, ensuring bit-exact reproducibility across platforms and Python versions

## Validation Experiments

All benchmarks and analyses are now unified in `benchmarks/`:

### Story Figure (Main Methods Paper Deliverable)

**`run_story_onepager.py`** - One-page validation figure demonstrating BioRSP's core functionality:

- **Panel A**: 2×2 archetype classification based on Coverage (C) and Spatial Score (S)
- **Panel B**: Confusion matrix for 4-class classification accuracy
- **Panel C**: Marker recovery showing precision@K for structured genes
- **Panel D**: Gene-gene module detection via co-patterning scores

```bash
python benchmarks/run_story_onepager.py --mode quick --seed 42
```

**Supporting Analyses:**
- **`run_null_calibration.py`**: Data-driven threshold derivation (S_cut, C_cut) from null simulations
- **`run_stability.py`**: Cross-embedding stability—scores robust to different UMAP embeddings
- **`run_abstention.py`**: Failure mode evaluation—correctly flags unreliable results under stress

### Core Benchmarks

**1. Statistical Calibration (`run_calibration.py`)

**Objective**: Establish that p-values are uniformly distributed under true null hypotheses and that type I error is controlled at the nominal level (α = 0.05).

**Design**: We simulate expression data under three null models:
- **IID Null**: Standard null hypothesis with no spatial structure in expression
- **Depth-Confounded**: Library size varies spatially (correlated with distance from tissue center), but expression remains spatially independent—tests robustness against technical gradients
- **Mask Stress**: Extremely low gene prevalence (1-5% of cells) to stress-test the sector masking and background estimation procedures

For each null model, we generate 100 independent replicates across varying sample sizes (N = 500, 1000, 2000 cells) and tissue geometries (disk, ellipse, crescent). Each replicate computes BioRSP metrics with 1000 permutations for p-value estimation.

**Success Criteria**: 
- QQ plots of observed p-values vs. Uniform(0,1) should show near-perfect agreement
- False positive rate at α = 0.05 should be 0.05 ± 0.01 (95% CI)
- Calibration should hold across all geometries and sample sizes

### 2. Archetype Classification (`run_archetypes.py`)

**Objective**: Demonstrate that the (C, S) coordinate system effectively discriminates biologically-distinct spatial patterns and provides interpretable classification boundaries.

**Design**: We simulate expression patterns representing four archetypal classes:
- **Housekeeping** (uniform): Expected high coverage C, low spatial score S
- **Niche Markers** (core/rim/wedge): Expected varied coverage, high S reflecting localization
- **Regional Programs** (broad domains): Expected high C, moderate S
- **Sparse/Noisy** (scattered): Expected low C, low S

For each pattern × geometry combination, we generate 75 replicates with controlled prevalence and signal strength parameters. We then examine the distribution of (C, S) coordinates and compute separation metrics (silhouette scores, pairwise distances).

**Success Criteria**:
- Clear separation of archetype clusters in (C, S) space
- Housekeeping patterns cluster at high C, low S
- Localized patterns (wedge, core, rim) achieve S > 0.15
- Classification accuracy > 90% using simple geometric boundaries

### 3. Robustness Analysis (`run_robustness.py`)

**Objective**: Characterize which geometric transformations preserve BioRSP metrics (desired invariances) and which break them (expected failure modes).

**Design**: We apply six distortions to baseline spatial patterns:

*Expected Invariances*:
- **Rotation**: Arbitrary rotation of coordinates (S should be rotation-invariant)
- **Jitter**: Small random positional noise (σ = 1-5% of tissue diameter)
- **Subsampling**: Random removal of 20-50% of cells

*Expected Sensitivities*:
- **Anisotropic Scaling**: Stretching along one axis (breaks circular symmetry assumed by radial statistics)
- **Swirl**: Radial-dependent angular warping (distorts radial patterns)

For each distortion type and intensity level, we compute the correlation between original and distorted S values across 50 replicates.

**Success Criteria**:
- Invariant transformations: Pearson correlation r > 0.95 between original and distorted S
- Sensitive transformations: Documented degradation curves showing where method breaks down
- Clear documentation of failure modes for users

### 4. Gene-Gene Co-patterning (`run_genegene.py`)

**Objective**: Validate that pairwise correlation and complementarity metrics correctly identify co-localized and mutually-exclusive spatial patterns.

**Design**: We simulate four pairwise scenarios:
- **Co-localization**: Both genes follow identical wedge pattern (angle_center = 0°) → expect high positive correlation
- **Exclusion**: Genes occupy opposite wedges (0° vs 180°) → expect high complementarity score
- **Orthogonal**: Genes occupy perpendicular wedges (0° vs 90°) → expect near-zero correlation
- **Core-Rim**: One gene in center, one at periphery → expect negative correlation or complementarity

For each scenario, we generate 100 replicate pairs and compute Pearson correlation on radar profiles along with BioRSP's complementarity index.

**Success Criteria**:
- Co-localized pairs: correlation > 0.8
- Exclusion pairs: complementarity > 0.7 or correlation < -0.3
- Orthogonal pairs: |correlation| < 0.2
- Precision-recall curve for "true positive" pair detection shows AUC > 0.9

## Experimental Parameters

The benchmark suite operates in three tiers to balance computational cost with statistical rigor:

| Tier | Replicates | Permutations | Scope | Runtime | Use Case |
|:-----|:-----------|:-------------|:------|:--------|:---------|
| **Quick** | 5-10 | 100 | `none` | 5-15 min | Code verification, continuous integration |
| **Validation** | 30-50 | 500 | `topk` | 30 min - 2 hr | Preliminary analysis, method development |
| **Publication** | 75-100 | 1000 | `all` | 4-12 hr | Peer-review manuscripts, final validation |

The `topk` permutation scope restricts expensive permutation tests to the most significant results, providing an order-of-magnitude speedup for large benchmarks.

## Running the Validation Suite

Execute individual benchmarks or the complete suite:

```bash
# Full validation pipeline in quick mode (verify all benchmarks execute correctly)
python3 run_benchmarks.py --mode quick --n_workers 4

# Publication-grade calibration experiment (with checkpointing for fault tolerance)
python3 benchmarks/run_calibration.py --mode publication --n_workers -1 --checkpoint_every 25 --resume

# Archetype analysis (no permutations needed for classification task)
python3 benchmarks/run_archetypes.py --mode publication --n_workers 8 --permutation_scope none

# Gene-gene experiment (topk permutations for top 500 pairs only)
python3 benchmarks/run_genegene.py --mode publication --n_workers 8 --permutation_scope topk --topk_perm 500
```

**Computational Requirements**: Publication-tier validation requires 8-12 hours on modern multi-core hardware (8+ cores recommended). Results are deterministically reproducible given identical random seeds.

## Result Interpretation

Upon completion, each benchmark generates:

1. **`runs.csv`**: Per-replicate results with all computed metrics
2. **`summary.csv`**: Aggregated statistics (means, confidence intervals, effect sizes)
3. **`report.md`**: Automated markdown summary with pass/fail criteria
4. **`manifest.json`**: Complete provenance including git commit, software versions, and parameter settings
5. **Diagnostic plots**: QQ plots (calibration), scatter plots (archetypes), robustness curves, correlation distributions

Visualize results across all benchmarks:

```bash
python3 plot_benchmarks.py
```

This generates publication-quality figure panels in `figures/` suitable for methods papers and supplementary materials.

## Design Rationale

**Reproducibility**: SeedSequence-based random number generation ensures bit-exact reproducibility, critical for peer review and method comparison.

**Computational Efficiency**: Parallel execution, geometry caching, and selective permutation testing reduce runtime by 5-10× compared to naive implementations.

**Statistical Rigor**: Publication tier uses 1000 permutations per condition, providing p-value precision of ±0.001 for robust multiple testing correction.

**Biological Realism**: Tissue geometries and expression patterns are grounded in spatial transcriptomics observations (e.g., kidney nephron structures, brain cortical layers, tumor microenvironments).

For detailed parameter documentation and performance optimization guidelines, see `benchmarks/README.md`.
