# BioRSP Methods Paper Simulations

This folder contains the standardized simulation benchmarks for the BioRSP Methods publication (Phase 3/4).

## Overview

The benchmarks evaluate the BioRSP v3 algorithm (Spatial Score `S`, Coverage `C`) on synthetic data with known ground truth.

## Benchmarks

### 1. Unified Simulation Engine (`simlib.py`)

A shared library `simlib.py` provides:

- **Shapes**: disk, ellipse, crescent, peanut, annulus, disconnected.
- **Archetypes**: uniform, rim, core, wedge, wedge_rim, two_wedges.
- **Null Models**: i.i.d., depth-confounded, density-confounded.
- **Distortions**: rotation, stretch, jitter, subsample.

### 2. Runs

#### Calibration (Type I Error)

Checks p-value uniformity under null hypotheses, including spatial confounders (library depth, density).

```bash
python run_calibration.py --output results_calibration.csv --reps 200
```

**Expectation**: P-values should be Uniform(0,1).

#### Robustness (Sensitivity Analysis)

Tests stability of `S` and `C` metrics under geometric distortions.

```bash
python run_robustness.py --output results_robustness.csv --reps 50
```

**Expectation**: `S` score should specific to topology (invariant to rotation, robust to mild jitter/subsampling).

#### Archetype Recovery (Power)

Evaluates ability to differentiate distinct spatial patterns.

```bash
python run_archetypes.py --output results_archetypes.csv --reps 50
```

**Expectation**: High `S` for structured archetypes, Low `S` for Uniform/Sparse.

#### Gene-Gene Co-patterning

Evaluates pairwise metrics (Synergy/Correlation).

```bash
python run_genegene.py --output results_genegene.csv --reps 50
```

**Expectation**: High Correlation for co-localized, High Complementarity for exclusion.

## Reproducibility

To reproduce the full suite of figures:

1. Run all scripts above.
2. Use the plotting notebooks in `case_studies/simulations/` (if available) or standard plotting libraries to visualize `results_*.csv`.

```bash
cd case_studies/simulations/methods_paper
python run_calibration.py --output results_calibration.csv --reps 200
python run_robustness.py --output results_robustness.csv --reps 50
python run_archetypes.py --output results_archetypes.csv --reps 50
python run_genegene.py --output results_genegene.csv --reps 50
```

## Dependencies

- `biorsp` (local package)
- `numpy`, `pandas`, `scipy`, `tqdm`
