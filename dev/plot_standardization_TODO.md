# Plot Standardization TODO

## Current Status - Phase 1: Audit Complete

This document tracks the standardization of plotting across simulation benchmarks and kidney case studies.

## Audit Summary

### Simulation Scripts (analysis/benchmarks/runners/)

#### run_archetypes.py

**Current Outputs:**

- `runs.csv`: Contains Coverage, Spatial_Bias_Score, true_archetype, predicted archetype (via classification)
- `summary.csv`: Aggregated metrics
- `manifest.json`: Run parameters, git hash, timestamp
- `report.md`: Pass/fail assessment
- Figures: archetype scatter, confusion matrix

**Plotting Functions Used:**

- `plotting.plot_archetype_scatter()` - C vs S scatter with truth labels
- `plotting.plot_confusion_matrix_styled()` - confusion matrix heatmap

**Coverage Definition:** `row["Coverage"]` from public API (percent of cells above quantile threshold)

**Spatial Bias Score Definition:** `row["Spatial_Bias_Score"]` from public API (weighted RMS of radar profile)

**Archetype Classification:** Implicit via cutoffs (c_cut=0.30, s_cut from calibration file)

**Issues Identified:**

- ✅ Uses standard column names (Coverage, Spatial_Bias_Score)
- ✅ Has archetype color mapping (ARCHETYPE_COLORS)
- ❌ Example panels not generated for each archetype
- ❌ Debug plots not standardized (no sector counts, no mask visualization)
- ❌ Cutoff lines in scatter may not match classification logic exactly

---

#### run_calibration.py

**Current Outputs:**

- QQ plots for different null distributions
- FPR grid plots
- calibration_thresholds.csv

**Coverage Definition:** Uses quantile-based foreground (default q=0.9 internally)

**Spatial Bias Score Definition:** Weighted RMS from public API

**Issues Identified:**

- ❌ No archetype scatter plot (calibration doesn't need it, but format should match)
- ❌ P-value histogram and FPR threshold plots not standardized
- ❌ No debug plots showing sector validity or masking logic

---

#### run_genegene.py

**Current Outputs:**

- Pairwise similarity scores
- Correlation/distance matrices
- Some example overlays

**Issues Identified:**

- ❌ No standardized example panel format
- ❌ Similarity metrics computed without explicit documentation of mask alignment
- ❌ No debug plots showing valid sector overlap

---

#### run_robustness.py

**Current Outputs:**

- Correlation before/after perturbation
- Some scatter plots

**Issues Identified:**

- ❌ No standardized panel showing distortion effect
- ❌ No clear "expected invariant" vs "failure mode" labeling

---

### Kidney Analysis Scripts (analysis/kidney_atlas/runners/)

#### run_tal_analysis.py

**Status:** Need to audit (TODO)

#### run_disease_stratified_analysis.py

**Status:** Need to audit (TODO)

#### run_kpmp_archetypes_all_genes.py

**Status:** Need to audit (TODO)

---

## Key Inconsistencies Found

### 1. Coverage Definition

- **Status:** ✅ CONSISTENT across simulation scripts
- All use `Coverage` from public API
- Definition: Fraction of cells above internal threshold (typically q=0.9 or biological threshold)

### 2. Spatial Bias Score Definition

- **Status:** ✅ CONSISTENT across simulation scripts
- All use `Spatial_Bias_Score` from public API
- Definition: Weighted RMS magnitude of radar profile over bg-supported sectors

### 3. Archetype Color Mapping

- **Status:** ⚠️ PARTIALLY CONSISTENT
- Simulations have ARCHETYPE_COLORS dict in plotting.py
- Colors: Ubiquitous=#4CAF50 (green), Gradient=#2196F3 (blue), Basal=#9E9E9E (gray), Patchy=#FF5722 (red)
- **TODO:** Verify kidney scripts use same colors

### 4. Cutoff Line vs Color Classification Mismatch

- **Status:** ❌ POTENTIAL ISSUE
- Cutoff lines drawn at (c_cut, s_cut) in scatter plot
- But actual classification may use different logic
- **TODO:** Verify classification function matches cutoff drawing

### 5. Example Panels Per Archetype

- **Status:** ❌ MISSING
- Simulations don't generate standardized example panels
- Should show: embedding + FG overlay + radar profile + annotated C/S values
- **TODO:** Implement shared example panel generator

### 6. Debug Plots

- **Status:** ❌ MISSING/INCONSISTENT
- No standardized debug plot set
- Need: sector counts, validity masks, foreground masks, cutoff consistency check

### 7. Figure Naming

- **Status:** ❌ INCONSISTENT
- Simulations use various names (archetype_scatter.png, confusion_matrix.png, etc.)
- **TODO:** Standardize to A_archetype_scatter.png, B_confusion.png, C_examples.png, D_pairwise.png

---

## Standardization Plan

### Phase 1: Create Shared Plotting Infrastructure ✅ (Starting)

- [ ] Create `biorsp/plotting/spec.py` with PlotSpec dataclass
- [ ] Define canonical column names, cutoffs, archetype mapping
- [ ] Create `biorsp/plotting/panels.py` with standard panel generators
- [ ] Implement archetype example panel generator
- [ ] Implement debug plot generators

### Phase 2: Standardize Simulation Scripts

- [ ] Update run_archetypes.py to use shared plotting
- [ ] Update run_calibration.py to use shared plotting
- [ ] Update run_genegene.py to use shared plotting
- [ ] Update run_robustness.py to use shared plotting
- [ ] Ensure all write standardized outputs (runs.csv, manifest.json, report.md)

### Phase 3: Standardize Kidney Scripts

- [ ] Audit kidney analysis scripts
- [ ] Update to use shared plotting API
- [ ] Ensure same figure set as simulations (with appropriate substitutions)

### Phase 4: Aggregation and CLI

- [ ] Create unified plotting aggregation script
- [ ] Add CLI for re-plotting from standardized outputs
- [ ] Create one-page story figure generator

### Phase 5: Testing

- [ ] Add tests for PlotSpec classification logic
- [ ] Add smoke tests for plot generation
- [ ] Verify all modules produce same figure set

---

## Checklist of Required Changes

### biorsp/plotting/spec.py

- [ ] PlotSpec dataclass with canonical column names
- [ ] Default cutoffs (c_cut=0.30, s_cut from calibration)
- [ ] Archetype classification function
- [ ] ARCHETYPE_COLORS mapping
- [ ] ARCHETYPE_DESCRIPTIONS mapping
- [ ] Legend ordering

### biorsp/plotting/panels.py

- [ ] plot_archetype_scatter() - standardized
- [ ] plot_confusion_or_composition() - handles both sim and kidney
- [ ] plot_example_panel() - embedding + FG + radar + annotations
- [ ] plot_examples_per_archetype() - grid of examples (Panel C)
- [ ] plot_pairwise_or_module() - for genegene / kidney modules (Panel D)

### biorsp/plotting/debug.py

- [ ] plot_debug_pointcloud()
- [ ] plot_debug_foreground_mask()
- [ ] plot_debug_sector_counts()
- [ ] plot_debug_radar_components()
- [ ] plot_debug_cutoff_consistency()

### biorsp/plotting/story.py

- [ ] generate_onepager() - combines A/B/C/D into one figure
- [ ] generate_caption() - auto-generated from manifest

### biorsp/plotting/cli.py

- [ ] replot_from_outputs() - regenerate figures from runs.csv + manifest.json

---

## Column Name Standardization

### Public API Columns (required in runs.csv)

- `gene` - gene name or replicate ID
- `Coverage` - fraction of cells above threshold
- `Spatial_Bias_Score` - weighted RMS magnitude
- `Directionality` - weighted mean sign (optional)
- `Archetype` - predicted archetype label
- `p_value` - permutation p-value (if computed)
- `abstain_flag` - whether scoring abstained

### Simulation-Only Columns

- `true_archetype` - ground truth label
- `case_id` - unique identifier for parameter combination
- `shape`, `N`, `coverage_regime`, `organization_regime` - simulation parameters

### Kidney-Only Columns

- `condition` - disease/control status
- `cluster` - cell type cluster
- `donor` - patient/donor ID

### Internal Columns (optional, for debugging)

- `n_expr_cells` - number of expressing cells
- `coverage_geom` - geometric coverage (fraction of bg-supported sectors with FG)
- `m_valid_sectors` - number of valid sectors

---

## Next Steps

1. **IMMEDIATE:** Create biorsp/plotting/spec.py with PlotSpec
2. **IMMEDIATE:** Create biorsp/plotting/panels.py with panel generators
3. **NEXT:** Update run_archetypes.py to use new plotting
4. **NEXT:** Audit kidney scripts and update them
5. **FINAL:** Add tests and smoke tests
