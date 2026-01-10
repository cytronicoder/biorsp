# BioRSP Benchmarks & Methods Paper Validation

Comprehensive simulation benchmarks for BioRSP validation, including:
- **Core benchmarks**: Calibration, archetype recovery, gene-gene co-patterning, robustness
- **Methods paper**: One-page story figure with supporting analysis

## Quick Start

### Story Figure (Main Deliverable)
```bash
# Generate the one-page validation figure (quick mode, ~15s)
python benchmarks/run_story_onepager.py --mode quick --seed 42

# For publication quality (~15 min)
python benchmarks/run_story_onepager.py --mode publication --seed 42
```

Output: `outputs/story/figures/fig_story_onepager.png` + 4 supporting panels

### Core Benchmarks
```bash
# Run all core benchmarks in publication mode (optimized)
python benchmarks/run_calibration.py --mode publication --n_workers 8 --checkpoint_every 50
python benchmarks/run_archetypes.py --mode publication --n_workers 8 --permutation_scope none
python benchmarks/run_genegene.py --mode publication --n_workers 8 --permutation_scope topk
python benchmarks/run_robustness.py --mode publication --n_workers 8 --permutation_scope none
```

### Smoke Test
```bash
python benchmarks/smoke_benchmarks.py  # Runs all benchmarks in quick mode (~2 min)
```

## Overview

All benchmarks evaluate BioRSP using synthetic datasets with **known ground truth**. The modular `simlib` package provides reproducible dataset generation, scoring, metrics, and plotting.

## Available Benchmarks

### Story Figure Scripts

#### 1. **run_story_onepager.py** - Main Methods Paper Figure
The one-page story figure that validates BioRSP's intended outcome: 2×2 archetypes via Coverage (C) and Spatial Score (S), plus gene-gene co-pattern recovery.

**What it tests:**
- Panel A: Archetype classification into 4 quadrants (housekeeping, regional program, sparse noise, niche marker)
- Panel B: Confusion matrix for 2×2 classification
- Panel C: Marker recovery (precision@K for structured genes)
- Panel D: Gene-gene module recovery (co-patterning accuracy)

**Quick mode:**
```bash
python benchmarks/run_story_onepager.py --mode quick --seed 42 --outdir outputs/story
```

**Success metrics:**
- Classification Accuracy ≥ 60% (quick), ≥ 80% (publication)
- Module AUPRC ≥ 0.25 (quick), ≥ 0.65 (publication)

#### 2. **run_null_calibration.py** - Threshold Derivation
Derives data-driven S_cut and C_cut from null simulations (95th percentile).

**Quick mode:**
```bash
python benchmarks/run_null_calibration.py --mode quick --seed 42
```

**Output:** `thresholds.json` with derived values

#### 3. **run_stability.py** - Cross-Embedding Stability
Tests whether BioRSP scores are stable across different UMAP embeddings.

**Quick mode:**
```bash
python benchmarks/run_stability.py --mode quick --seed 42 --n_embeddings 3
```

**Success metric:** Score correlation ≥ 0.80 across embeddings

#### 4. **run_abstention.py** - Failure Mode Evaluation
Tests that BioRSP appropriately flags unreliable results under stress conditions:
- Low coverage (1-5%)
- Small sample size (N=50-200)
- Disconnected geometry

**Quick mode:**
```bash
python benchmarks/run_abstention.py --mode quick --seed 42
```

**Expected behavior:** Abstention rate > 50% for extreme low-coverage genes

### Core Benchmark Scripts

#### 5. **run_calibration.py** - Type I Error Control
Tests p-value uniformity under null hypotheses, including spatial confounders.

**Options:**
```bash
python benchmarks/run_calibration.py \
  --mode publication \
  --n_workers 8 \
  --checkpoint_every 50 \
  --resume
```

**Expectation:** FPR ≈ 0.05 at α=0.05, p-values uniform under null

#### 6. **run_archetypes.py** - Pattern Classification
Evaluates ability to distinguish 4 spatial archetypes via (C, S) coordinates.

**Options:**
```bash
python benchmarks/run_archetypes.py \
  --mode publication \
  --n_workers 8 \
  --permutation_scope none  # Faster: skip permutations for classification
```

**Expectation:** High S for structured patterns, low S for uniform/sparse

#### 7. **run_genegene.py** - Gene-Gene Co-patterning
Tests pairwise correlation metrics and co-localization detection.

**Options:**
```bash
python benchmarks/run_genegene.py \
  --mode publication \
  --n_workers 8 \
  --permutation_scope topk \
  --topk_perm 500  # Only permute top 500 pairs
```

**Expectation:** High correlation for co-localized patterns, low for orthogonal

#### 8. **run_robustness.py** - Stability Analysis
Evaluates robustness of (C, S) under coordinate perturbations: rotation, anisotropic scaling, jitter, subsampling.

**Options:**
```bash
python benchmarks/run_robustness.py \
  --mode publication \
  --n_workers 8 \
  --permutation_scope none  # Faster: no permutations needed
```

**Expectation:** S invariant to rotation, robust to mild jitter/subsampling

## Modes & Performance

| Mode | Runtime | Use Case | Parameters |
|------|---------|----------|------------|
| `quick` | ~1 min (story) / 2-5 min (core) | Development, CI, smoke tests | Minimal reps, few permutations |
| `validation` | ~5 min (story) / 10-20 min (core) | Preliminary results | Standard reps, selective permutations |
| `publication` | ~15 min (story) / 30+ min (core) | Final manuscript figures | Full grid, optimized permutations |

### Performance Optimization

All benchmarks support parallel execution and selective permutation testing:

```bash
# Speed up with these flags:
--n_workers 8               # Use 8 CPUs (default: auto-detect)
--checkpoint_every 50       # Save progress every N runs
--resume                    # Resume from checkpoint if interrupted
--permutation_scope none    # Skip permutations (fastest)
--permutation_scope topk    # Only permute top K pairs (gene-gene only)
```

**Expected speedups:**
- Calibration: 5-8x faster with 8 workers + selective permutations
- Archetypes: 5-8x faster with parallelization
- Gene-gene: 8-12x faster with topk permutations + parallelization

## Output Structure

```
outputs/
├── story/
│   ├── figures/
│   │   ├── fig_story_A_archetypes.png
│   │   ├── fig_story_B_confusion.png
│   │   ├── fig_story_C_marker_recovery.png
│   │   ├── fig_story_D_genegene.png
│   │   └── fig_story_onepager.png
│   ├── runs.csv            # Per-gene results
│   ├── summary.csv         # Summary metrics
│   ├── manifest.json       # Metadata + git commit
│   └── report.md           # Human-readable report
├── calibration/
│   ├── runs.csv
│   ├── summary.csv
│   ├── report.md
│   └── manifest.json
├── archetypes/
│   └── ...
├── genegene/
│   └── ...
└── robustness/
    └── ...
```

## Archetypes Explained

The 2×2 factorial design represents spatial gene expression patterns:

| Archetype | Coverage | Spatial Pattern | Interpretation |
|-----------|----------|-----------------|----------------|
| **Housekeeping** | High (60-90%) | Random (iid) | Ubiquitous genes, no spatial preference |
| **Regional Program** | High (60-90%) | Structured (wedge/core/rim) | Broadly expressed but spatially organized |
| **Sparse/Noisy** | Low (5-20%) | Random (iid) | Rare genes, scattered expression |
| **Niche Marker** | Low (5-20%) | Structured (wedge/core/rim) | Localized to specific tissue region |

**Success criterion:** Clear separation in (C, S) space with high classification accuracy.

## CLI Reference

### Common Options (All Scripts)
```bash
--outdir PATH             # Output directory (default: outputs/<benchmark>)
--seed INT                # Random seed
--mode {quick|validation|publication}  # Benchmark mode
```

### Parallelization
```bash
--n_workers INT           # Number of workers (-1 = auto, default: -1)
--checkpoint_every INT    # Checkpoint frequency (default: 25)
--resume                  # Resume from checkpoint
```

### Permutation Testing
```bash
--permutation_scope {none|topk|all}
--topk_perm INT           # Top K pairs for topk mode (default: 500)
--n_permutations INT      # Number of permutations (mode-dependent)
```

### Story Figure Specific
```bash
python benchmarks/run_story_onepager.py \
  --mode publication \
  --seed 42 \
  --n_null_genes 100      # Genes for null calibration
  --n_genes_per_archetype 50  # Genes per archetype
```

## Reproducibility

All scripts use deterministic RNG and record git commit in manifests:

```bash
git checkout <commit_hash>
python benchmarks/run_story_onepager.py --mode publication --seed 42
# Produces identical results (modulo floating-point rounding)
```

## Interpreting Results

### Story Figure Panels

**Panel A (Archetype Scatter):**
- Points should separate into 4 distinct quadrants
- Colors show ground truth archetypes
- Dashed lines are derived thresholds (S_cut, C_cut)
- Good result: Colors match quadrants, diagonal separation

**Panel B (Confusion Matrix):**
- Diagonal should be bright (high recall/precision)
- Numbers are count (percentage)
- Good result: Diagonal > 70% for each class

**Panel C (Marker Recovery):**
- Precision should decrease as K increases
- Should be well above 50% (random) for top-ranked genes
- Good result: Top-20 precision > 60%

**Panel D (Module Recovery):**
- AUPRC = Area under precision-recall curve
- AUPRC > 0.5 indicates useful co-patterning scores
- Good result: AUPRC > 0.65 (publication) or > 0.25 (quick)

## Troubleshooting

### Script fails with "module not found"
```bash
cd case_studies/simulations
export PYTHONPATH="$PWD:$PYTHONPATH"
python benchmarks/run_story_onepager.py --mode quick
```

### Memory errors
```bash
# Reduce dataset size or use smaller modes
python benchmarks/run_story_onepager.py --mode quick --n_genes_per_archetype 5
```

### Slow execution
```bash
# Enable parallelization and skip permutations
python benchmarks/run_archetypes.py \
  --mode publication \
  --n_workers 8 \
  --permutation_scope none
```

## Dependencies

- `biorsp` (local package)
- `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`
- Optional: `scikit-learn`, `umap-learn`

## File Organization

**Story Figure & Methods Paper Validation:**
- `run_story_onepager.py` - Main one-page figure
- `run_null_calibration.py` - Threshold derivation
- `run_stability.py` - Cross-embedding stability
- `run_abstention.py` - Failure mode evaluation
- `smoke_benchmarks.py` - Smoke test for all scripts

**Core Benchmarks:**
- `run_calibration.py` - Type I error control
- `run_archetypes.py` - Pattern classification
- `run_genegene.py` - Gene-gene co-patterning
- `run_robustness.py` - Robustness analysis

All scripts use the shared `simlib` package for simulation utilities.
