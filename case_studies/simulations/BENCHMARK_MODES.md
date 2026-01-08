# BioRSP Methods Paper: Three-Tier Benchmark Framework

## Overview

The benchmark suite now supports three execution tiers for different research stages:

1. **Quick** - Debug/development (minutes)
2. **Validation** - Preliminary results (hours)
3. **Publication** - Peer-review ready (overnight/day)

## Usage

### Quick Mode (Development)

```bash
python3 run_calibration.py --mode quick
python3 run_archetypes.py --mode quick
python3 run_robustness.py --mode quick
python3 run_genegene.py --mode quick
```

**Characteristics**: Minimal computation, no p-values, single shape/parameter set
**Time**: ~5-15 minutes per benchmark
**Use case**: Debug code, verify pipeline works

### Validation Tier (Preliminary)

```bash
# Pass n_reps=50 to trigger validation tier
python3 run_calibration.py --mode publication --n_reps 50
python3 run_archetypes.py --mode publication --n_reps 50
python3 run_robustness.py --mode publication --n_reps 50
python3 run_genegene.py --mode publication --n_reps 50
```

**Characteristics**: 50 reps, selective shapes/parameters, permutation_scope=topk
**Time**: ~30 min to 2 hours per benchmark
**Use case**: Check results before final runs, validate parameter choices

### Publication Tier (Final)

```bash
# Default publication mode (100 reps, full parameter space)
python3 run_calibration.py --mode publication
python3 run_archetypes.py --mode publication
python3 run_robustness.py --mode publication
python3 run_genegene.py --mode publication
```

**Characteristics**: 100 reps, full parameter space, 1000 permutations with scope=all
**Time**: 4-12 hours per benchmark (run overnight or in parallel)
**Use case**: Final peer-review ready results

---

## Benchmark Details

### Calibration Benchmark

| Tier        | Reps | N                    | Shapes              | Nulls | Perms | Scope | Conditions | Time     |
| ----------- | ---- | -------------------- | ------------------- | ----- | ----- | ----- | ---------- | -------- |
| Quick       | 10   | [1000]               | disk                | iid   | 100   | none  | 10         | ~2 min   |
| Validation  | 50   | [500,2000]           | disk, annulus       | 3     | 500   | topk  | 300        | ~30 min  |
| Publication | 100  | [500,1000,2000,5000] | disk, annulus, peanut | 3     | 1000  | all   | 3,600      | ~6 hours |

**Peer-review rigor**: 100 reps per condition ensures stable FPR estimates; 1000 permutations provide robust p-values across all shapes and nulls.

### Archetype Benchmark

| Tier        | Reps | N                | Shapes | Patterns | Perms | Scope | Conditions | Time     |
| ----------- | ---- | ---------------- | ------ | -------- | ----- | ----- | ---------- | -------- |
| Quick       | 5    | [2000]           | disk   | 3        | 100   | none  | 15         | ~3 min   |
| Validation  | 50   | [1000,2000]      | disk, peanut | 4        | 500   | topk  | 400        | ~40 min  |
| Publication | 100  | [1000,2000,5000] | disk, peanut, crescent | 8        | 1000  | all   | 2,400      | ~8 hours |

**Peer-review rigor**: 100 reps per archetype ensures stable recovery rates; comprehensive pattern space (8 patterns) tests diverse spatial signals.

### Robustness Benchmark

| Tier        | Reps | N                | Shapes | Patterns | Distortions | Perms | Scope | Conditions | Time      |
| ----------- | ---- | ---------------- | ------ | -------- | ----------- | ----- | ----- | ---------- | --------- |
| Quick       | 5    | [2000]           | disk   | wedge    | 2           | 100   | none  | 10         | ~2 min    |
| Validation  | 50   | [1000,2000]      | disk   | wedge, core | 4           | 500   | topk  | 400        | ~40 min   |
| Publication | 100  | [1000,2000,5000] | disk, annulus, peanut | uniform, wedge, core, rim | 6           | 1000  | all   | 7,200      | ~12 hours |

**Peer-review rigor**: All 6 distortions (4 invariance + 2 sensitivity) tested; 100 reps ensures stable robustness estimates across patterns.

### Gene-Gene Benchmark

| Tier        | Reps | N                | Shapes | Scenarios | Perms | TopK | Scope | Conditions | Time     |
| ----------- | ---- | ---------------- | ------ | --------- | ----- | ---- | ----- | ---------- | -------- |
| Quick       | 5    | [2000]           | disk   | same      | 100   | 100  | none  | 5          | ~1 min   |
| Validation  | 50   | [1000,2000]      | disk, annulus | same, opposite | 500   | 500  | topk  | 200        | ~20 min  |
| Publication | 100  | [1000,2000,5000] | disk, annulus, peanut | same, opposite, orthogonal, rim_core | 1000  | 1000 | all   | 3,600      | ~10 hours |

**Peer-review rigor**: 100 reps ensures stable pair-wise correlations across multi-shape combinations; 4 scenarios test diverse co-patterns.

---

## Recommended Workflow

1. **Development Phase**

   ```bash
   # Test code changes quickly
   for script in run_*.py; do
     python3 "$script" --mode quick
   done
   ```

   Time: ~20 minutes total

2. **Validation Phase** (before submitting paper)

   ```bash
   # Spot-check results with moderate rigor
   for script in run_*.py; do
     python3 "$script" --mode publication --n_reps 100 --n_workers -1
   done
   ```

   Time: ~2-4 hours total (parallel if possible)

3. **Publication Phase** (final submission-ready results)
   ```bash
   # Run full benchmarks overnight/weekend
   for script in run_*.py; do
     python3 "$script" --mode publication --n_workers -1 &
   done
   wait  # All run in parallel
   ```
   Time: ~12 hours (parallel execution)

---

## Key Design Principles

1. **Minimal Quick**: Only 5-10 replicates, single shape/parameter set, no permutations. Suitable for code verification, NOT for results.

2. **Balanced Validation**: 30-50 reps, 2 key shapes, selective permutations. Suitable for preliminary paper planning.

3. **Robust Publication**: 75-100 reps per condition, comprehensive parameter space, 1000 permutations. Satisfies peer-review standards for calibration/robustness claims.

4. **Permutation Scope Progression**:

   - `permutation_scope=none`: Quick estimates, no p-values (Quick mode)
   - `permutation_scope=topk`: Selective permutations on top hits (Validation, if used)
   - `permutation_scope=all`: All permutations, robust p-values (Publication)

5. **Total Computation**: Publication tier ~30-40 hours for all 4 benchmarks (parallelizable)
