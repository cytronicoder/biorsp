# Benchmark suite (simulation)

This directory contains simulation benchmarks for BioRSP. All benchmark runners write contract-compliant outputs and are intended to be reproducible with fixed seeds.

## Output contract

Each benchmark run writes a run directory under `OUTDIR/<benchmark>/<run_id>/` containing:

- `runs.csv`: per-replicate results with benchmark-specific columns.
- `summary.csv`: aggregated metrics with confidence intervals.
- `manifest.json`: run metadata, configuration, and provenance.
- `report.md`: concise benchmark summary.
- `figures/`: standardized plots.

The contract schema is enforced by `analysis/benchmarks/simlib/io_contract.py`.

## Reproducibility

- Use `--seed` to fix random seeds in runners.
- For deterministic CPU execution, set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.
- Run identifiers (`run_id`) default to a UTC timestamp for uniqueness.

## Benchmarks

### Archetypes (`run_archetypes.py`)

**Purpose:** Evaluate archetype classification based on coverage and spatial scores.

**Inputs:** Synthetic expression patterns spanning high/low coverage and structured/iid spatial organization.

**Key metrics:** Accuracy, macro F1, per-archetype recall, and abstention rate in `summary.csv`.

**Held-out evaluation:** Cases are split by `case_id`; thresholds are derived on the training split and applied to the test split.

**Good behavior (intended):** Clear separation in `(Coverage, Spatial_Score)` and stable recall for each archetype on the held-out split.

**Failure modes:** High abstention, poor separation when coverage is extremely low, or misclassification when synthetic patterns overlap.

### Calibration (`run_calibration.py`)

**Purpose:** Validate null calibration of p-values and spatial scores under different null types.

**Inputs:** Synthetic nulls (e.g., iid, depth-confounded, mask stress) generated from simulated coordinates.

**Key metrics:** False positive rate (FPR) at specified alpha levels, KS statistic vs. Uniform(0,1), abstention rate.

**Held-out evaluation:** Thresholds on `Spatial_Score` are derived on the training split (by null type and shape) and evaluated on the test split.

**Good behavior (intended):** FPR near nominal alpha and QQ plots consistent with uniform p-values when the null is correct.

**Failure modes:** Excess false positives, high abstention, or strong deviations from uniform QQ plots.

### Null calibration (`run_null_calibration.py`)

**Purpose:** Derive coverage and spatial-score thresholds from null simulations for downstream story plots or archetype labeling.

**Inputs:** Synthetic nulls with configurable cell counts and permutation settings.

**Key metrics:** Quantile-derived thresholds and null distributions recorded in `runs.csv` and summary artifacts.

**Good behavior (intended):** Thresholds are stable across seeds and null types.

**Failure modes:** Thresholds dominated by sparse or abstained runs.

### Robustness (`run_robustness.py`)

**Purpose:** Quantify sensitivity of Coverage and Spatial_Score to geometric distortions.

**Inputs:** Synthetic patterns with paired baseline and distorted datasets (rotation, jitter, subsampling, anisotropic scaling, swirl).

**Key metrics:** Within-pair deltas and correlation summaries, reported in `runs.csv` and aggregated in `summary.csv`.

**Good behavior (intended):** Invariant distortions show small deltas; sensitive distortions show systematic shifts.

**Failure modes:** Large deltas under rotations or mild jitter.

### Stability (`run_stability.py`)

**Purpose:** Assess run-to-run stability under resampling or embedding perturbations.

**Inputs:** Synthetic datasets with repeated resampling and scoring.

**Key metrics:** Correlations and variability in Coverage and Spatial_Score across replicates.

**Good behavior (intended):** High correlations across replicates when sampling variation is moderate.

**Failure modes:** Instability when signals are sparse or when embeddings are highly perturbed.

### Gene–gene (`run_genegene.py`)

**Purpose:** Evaluate pairwise spatial co-patterning metrics.

**Inputs:** Synthetic gene pairs representing co-localized, opposing, and orthogonal patterns.

**Key metrics:** Pairwise similarity scores and separation between known scenarios.

**Good behavior (intended):** Co-localized pairs have higher similarity than orthogonal or opposing pairs.

**Failure modes:** Overlap between scenarios or low separation in score distributions.

### Abstention stress test (`run_abstention.py`)

**Purpose:** Quantify abstention behavior under challenging signal regimes.

**Inputs:** Synthetic conditions with low coverage or sparse foreground support.

**Key metrics:** Abstention rates and abstention reasons in `runs.csv` and `summary.csv`.

**Good behavior (intended):** High abstention when adequacy constraints are violated.

**Failure modes:** Low abstention under obvious inadequacy or large numbers of NaN scores.

## Running benchmarks

```bash
python analysis/benchmarks/runners/run_benchmarks.py --mode quick --n-workers 4 --outdir results/benchmarks
```

Individual runners can also be executed directly, for example:

```bash
python analysis/benchmarks/runners/run_archetypes.py --mode quick --outdir results/benchmarks
```

## Standardized plots

Benchmark runners use `biorsp.plotting.standard.make_standard_plot_set` to produce shared plots. See `docs/simulations_plotting.md` for details.
