# BioRSP Methods Paper Simulations

Standardized simulation benchmarks for the BioRSP Methods publication.

## Quick Start (Optimized for Speed)

All benchmarks now support **parallel execution**, **checkpointing**, and **selective permutation testing** for 5-10x speedup:

```bash
# Calibration (parallelized with 8 workers, checkpoint every 25 runs)
python methods_paper/run_calibration.py --mode publication --n_workers 8 --checkpoint_every 25 --resume

# Archetypes (skip permutations for faster exploratory analysis)
python methods_paper/run_archetypes.py --mode publication --n_workers 8 --permutation_scope none

# Gene-Gene (topk permutations: only test top 500 pairs)
python methods_paper/run_genegene.py --mode publication --n_workers 8 --permutation_scope topk --topk_perm 500

# Robustness (no permutations needed - just metric stability)
python methods_paper/run_robustness.py --mode publication --n_workers 8 --permutation_scope none
```

**Performance**: Full publication pipeline reduced from 48+ hours → ~3-4 hours total on modern hardware (8+ cores).

See [PERFORMANCE_NOTES.md](../PERFORMANCE_NOTES.md) for detailed optimization documentation.

## Overview

The benchmarks evaluate BioRSP v3 (Spatial Score `S`, Coverage `C`) on synthetic datasets with known ground truth. All scripts use the modular `simlib` package for reproducible dataset generation, scoring, metrics, plotting, and reporting.

## Simulation Framework

The `simlib` package (in `case_studies/simulations/simlib/`) provides:

- **Shapes**: disk, ellipse, crescent, peanut, annulus, disconnected_blobs
- **Signal Patterns**: uniform, sparse, core, rim, wedge, wedge_core, wedge_rim, two_wedges, halfplane_gradient
- **Null Models**: iid, depth_confounded, density_confounded, mask_stress
- **Distortions**: rotate, aniso_scale, swirl, radial_warp, jitter, subsample
- **Deterministic RNG**: SeedSequence-based for full reproducibility
- **Standardized I/O**: CSV outputs, manifest JSON with git commit tracking, markdown reports
- **Optimizations**: Parallelization, geometry caching, checkpointing, selective permutations

## Benchmarks

### 1. Calibration (Type I Error Control)

Tests p-value uniformity under various null hypotheses, including spatial confounders.

```bash
# Quick smoke test (10 reps, 2 sample sizes, 2 null types)
python case_studies/simulations/methods_paper/run_calibration.py --mode quick

# Publication mode (100 reps, full grid) - OPTIMIZED
python case_studies/simulations/methods_paper/run_calibration.py \
  --mode publication \
  --n_workers 8 \
  --checkpoint_every 50 \
  --resume
```

**Outputs**: `outputs/calibration/runs.csv`, `summary.csv`, `report.md`, QQ plots, FPR grids  
**Expectation**: P-values ~ Uniform(0,1), FPR ≈ 0.05 at α=0.05

### 2. Archetypes (Pattern Classification)

Evaluates ability to distinguish spatial archetypes via (C, S) coordinates.

```bash
# Quick test
python case_studies/simulations/methods_paper/run_archetypes.py --mode quick

# Publication mode - OPTIMIZED (no permutations for classification)
python case_studies/simulations/methods_paper/run_archetypes.py \
  --mode publication \
  --n_workers 8 \
  --permutation_scope none
```

**Outputs**: `outputs/archetypes/runs.csv`, `summary.csv`, `report.md`, archetype scatter plots  
**Expectation**: High S for structured patterns (rim, wedge), low S for uniform/sparse

### 3. Gene-Gene Co-patterning

Tests pairwise correlation and complementarity metrics.

```bash
# Quick test
python case_studies/simulations/methods_paper/run_genegene.py --mode quick

# Publication mode - OPTIMIZED (topk permutations for best pairs only)
python case_studies/simulations/methods_paper/run_genegene.py \
  --mode publication \
  --n_workers 8 \
  --permutation_scope topk \
  --topk_perm 500
```

**Outputs**: `outputs/genegene/runs.csv`, `summary.csv`, `report.md`, correlation distributions  
**Expectation**: High correlation for co-localized patterns, low/negative for orthogonal

### 4. Robustness (Stability Analysis)

Evaluates stability of (C, S) under coordinate perturbations.

```bash
# Quick test
python case_studies/simulations/methods_paper/run_robustness.py --mode quick

# Publication mode - OPTIMIZED (no permutations needed)
python case_studies/simulations/methods_paper/run_robustness.py \
  --mode publication \
  --n_workers 8 \
  --permutation_scope none
```

**Outputs**: `outputs/robustness/runs.csv`, `summary.csv`, `report.md`, robustness curves  
**Expectation**: S invariant to rotation, robust to mild jitter/subsampling

## CLI Options

All scripts support consistent CLI arguments:

### Standard Options
- `--outdir`: Output directory (default: `outputs/<benchmark>`)
- `--seed`: Random seed (default varies per benchmark)
- `--n_reps`: Number of replicates per condition
- `--N`: Cell counts (e.g., `--N 500 2000 5000`)
- `--shape`: Shapes (e.g., `--shape disk annulus peanut`)
- `--n_permutations`: BioRSP permutations (default: 250)
- `--mode`: `quick` (smoke test) or `publication` (full grid)

### Performance Options (NEW)
- `--n_workers`: Number of parallel workers (-1 = use all cores, default: -1)
- `--checkpoint_every`: Save progress every N runs (default: 25)
- `--resume`: Resume from checkpoint if interrupted
- `--permutation_scope`: Permutation strategy
  - `none`: Skip permutations (faster, no p-values)
  - `all`: Compute p-values for all replicates (slower, complete)
  - `topk`: Only permute top K pairs by effect size (gene-gene only)
- `--topk_perm`: Number of top pairs for topk mode (default: 500)

### Recommended Publication Settings

```bash
# Fastest: No permutations (archetypes, robustness)
--n_workers 8 --permutation_scope none

# Balanced: Selective permutations (gene-gene)
--n_workers 8 --permutation_scope topk --topk_perm 500

# Complete: All permutations with checkpointing (calibration)
--n_workers 8 --permutation_scope all --checkpoint_every 50 --resume
```

## Output Structure

```
case_studies/simulations/
  outputs/
    calibration/
      runs.csv          # Per-replicate results
      summary.csv       # Aggregated statistics
      report.md         # Biologist-friendly report
      manifest.json     # Metadata + git commit
    archetypes/
      ...
    genegene/
      ...
    robustness/
      ...
  figs/
    calibration_qq_iid.png
    calibration_fpr_grid.png
    archetypes_scatter_disk.png
    ...
```

## Dependencies

- `biorsp` (local package)
- `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`
- Optional: `sklearn` (for KDE/kNN density estimation)

## Reproducibility

All scripts use deterministic RNG via `np.random.SeedSequence`. Git commit is recorded in `manifest.json`. To reproduce:

```bash
git checkout <commit_hash>
python case_studies/simulations/methods_paper/run_calibration.py --mode publication
```
