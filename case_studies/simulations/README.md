# BioRSP Simulation Benchmarks

This directory contains the standardized simulation suite for evaluating BioRSP (Spatial Organization Score $S_g$ and Coverage Score $C_g$).

## Directory Structure

- **`benchmarks/`**: Core experiment scripts for calibration, archetypes, robustness, and gene-gene analysis.
- **`simlib/`**: Modular library for reproducible dataset generation, scoring, and metrics.
- **`figures/`**: Generated publication-quality figures from benchmark results.
- **`outputs/`**: CSV results, manifest metadata, and markdown summaries from simulation runs.
- **`scripts/`**: Utility scripts (e.g., smoke tests, maintenance).
- **`tests/`**: Unit tests for the simulation library.

## Getting Started

The benchmark suite supports three execution tiers designed for different research stages:

1. **Quick Mode** (`--mode quick`): For code verification and pipeline testing (5-15 min).
2. **Validation Tier**: For preliminary results and paper planning (30 min - 2 hours).
3. **Publication Tier** (`--mode publication`): For peer-review ready Results (4-12 hours).

### Running Benchmarks

You can run individual benchmarks or the entire suite:

```bash
# Run all benchmarks in quick mode to verify everything is working
python3 run_benchmarks.py --mode quick --n_workers 4

# Run a specific benchmark in publication mode
python3 benchmarks/run_calibration.py --mode publication --n_workers -1
```

### Visualizing Results

Once benchmarks have finished and generated CSV files in `outputs/`, use the plotting script:

```bash
python3 plot_benchmarks.py
```

## Three-Tier Framework

| Tier | Replicates | Permutations | Scope | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Quick** | 5-10 | 100 | `none` | Debugging, CI/CD |
| **Validation** | 30-50 | 500 | `topk` | Preliminary results |
| **Publication** | 75-100 | 1000 | `all` | Final peer-review ready |

For more details on benchmark parameters, see the documentation in `benchmarks/README.md`.

## Key Design Principles

- **Reproducibility**: All simulations use `SeedSequence`-based deterministic random number generation.
- **Efficiency**: Parallel execution, geometry caching, and checkpointing are supported across all benchmarks.
- **Rigor**: The publication tier uses 1000 permutations per condition to ensure robust p-value computation.
