# Update Summary: Batch Runner and Scripts Aligned with Latest Changes

## Changes Made

### 1. Updated Batch Runner (run_all_benchmarks.py)
**Problem**: Batch runner was overriding permutation_scope, but scripts now enforce it internally based on mode.

**Changes**:
- Removed `permutation_scope` parameter from BENCHMARKS configuration
- Removed `permutation_scope` from `run_benchmark()` function signature
- Removed `--permutation_scope` from command construction
- Updated benchmark unpacking (3 items instead of 4)

**Rationale**: Scripts now enforce permutation_scope internally:
- Quick mode: `permutation_scope = "none"` (fast, no p-values)
- Validation mode: `permutation_scope = "topk"` (selective)
- Publication mode: `permutation_scope = "all"` (rigorous, all permutations)

External overrides would conflict with this enforcement logic.

---

### 2. Updated Test Script (test_optimizations.py)
**Changes**:
- Removed hardcoded `permutation_scope` flags from BENCHMARKS
- Added comment explaining internal enforcement
- Kept `topk_perm: 100` for genegene (used in validation mode)

---

### 3. Cleaned Up Irrelevant Files
**Removed directories** (~2.3 MB total):
```
✓ calibration/          - Old output directory
✓ outputs copy/         - Duplicate backup directory  
✓ smoke_outputs/        - Old test outputs
✓ figs_demo/           - Demo figures (2.3 MB)
```

**Removed cache/system files**:
```
✓ __pycache__/ directories (all locations)
✓ .DS_Store files (macOS metadata)
```

**Kept important files**:
- `simlib.py` - Compatibility shim (shows deprecation warning, re-exports package)
- `scripts/smoke_all.py` - Still functional smoke test
- All outputs/ directories with current results
- All simlib/ package modules

---

## Current Benchmark Configuration

### run_all_benchmarks.py
```python
BENCHMARKS = [
    ("calibration", "run_calibration.py", {}),
    ("archetypes", "run_archetypes.py", {}),
    ("genegene", "run_genegene.py", {"topk_perm": 500}),  # topk_perm for validation
    ("robustness", "run_robustness.py", {}),
]
```

### Permutation Scope Enforcement (Internal to Scripts)

| Benchmark | Quick Mode | Validation Mode | Publication Mode |
|-----------|-----------|----------------|-----------------|
| Calibration | none | topk | **all** (enforced) |
| Archetypes | none | none | **all** (enforced) |
| Gene-gene | none | topk | **all** (enforced) |
| Robustness | none | none | **all** (enforced) |

---

## Usage Examples

### Quick Mode (Development)
```bash
python3 run_all_benchmarks.py --mode quick --n_workers 4
```
- ~5 minutes
- No permutation tests
- Uses minimal parameters

### Publication Mode (Peer Review)
```bash
python3 run_all_benchmarks.py --mode publication --n_workers 8
```
- ~3-4 hours with 8 workers
- Full parameter grids
- 1000 permutations with scope="all" (enforced internally)
- 100 reps for gene-gene (multi-shape stability)

### Resume Interrupted Run
```bash
python3 run_all_benchmarks.py --mode publication --n_workers 8 --resume
```

---

## Verification

All scripts verified:
```
✓ run_all_benchmarks.py
✓ test_optimizations.py
✓ methods_paper/run_calibration.py
✓ methods_paper/run_archetypes.py
✓ methods_paper/run_genegene.py
✓ methods_paper/run_robustness.py
```

---

## What's Next

1. **Run Quick Smoke Test**:
   ```bash
   python3 run_all_benchmarks.py --mode quick --n_workers 4
   ```

2. **Run Publication Benchmarks**:
   ```bash
   python3 run_all_benchmarks.py --mode publication --n_workers 8
   ```

3. **Check Convergence** (after runs):
   ```python
   from simlib.validation import check_estimate_convergence
   result = check_estimate_convergence(runs_df, 'spatial_score')
   print(result['recommendation'])
   ```

4. **Generate Figures** (after benchmarks complete):
   ```bash
   python3 plot_from_csv.py
   ```
