# Plot Standardization Implementation Summary

## Overview

This document summarizes the implementation of Phases 3-6 for plot standardization across BioRSP simulation benchmarks and kidney case studies.

## Completed Work

### Phase 3: Update Simulation Scripts ✓

**run_story_onepager.py:**
- Imported `PlotSpec` from `biorsp.plotting.spec`
- Replaced hardcoded `DEFAULT_S_CUT` and `DEFAULT_C_CUT` with `_DEFAULT_SPEC.s_cut` and `_DEFAULT_SPEC.c_cut`
- Updated archetype scatter to use `story_spec.get_legend_order()` and `story_spec.get_color()`
- Added `plot_spec=story_spec.to_dict()` to manifest for reproducible replotting

**biorsp/simulations/plotting.py:**
- Replaced local `ARCHETYPE_COLORS` and `ARCHETYPE_DESCRIPTIONS` with imports from `biorsp.plotting.spec`
- This ensures all simulation plots use the canonical color scheme

**biorsp/simulations/io.py:**
- Updated `write_manifest()` to accept `plot_spec` parameter for storing PlotSpec in manifests

### Phase 4: Fix Kidney Color Mismatch ✓

**run_kpmp_archetypes_all_genes.py:**
- Replaced local `ARCHETYPE_COLORS` with import from `biorsp.plotting.spec`
- Updated `plot_cs_scatter()` to use `ARCHETYPE_ORDER` for consistent legend ordering
- Updated `plot_top_tables()` to use `ARCHETYPE_ORDER` for consistent 2x2 grid layout

**run_disease_stratified_analysis.py:**
- Replaced local `ARCHETYPE_COLORS` with import from `biorsp.plotting.spec`

**run_tal_analysis.py:**
- Updated local archetype color mapping to use `ARCHETYPE_COLORS` from `biorsp.plotting.spec`
- Added legacy archetype name mapping to canonical names

### Phase 5: Create Replot CLI ✓

**biorsp/plotting/replot.py (new file):**
- CLI tool to regenerate panels from manifest without recomputation
- `python -m biorsp.plotting.replot --manifest <path> [--outdir <path>] [--format {png,pdf,svg}]`
- Loads `plot_spec` from manifest to ensure identical cutoffs/colors
- Auto-detects simulation vs kidney mode
- Supports grouping column for kidney Panel B

### Phase 6: Panel C/D Implementation ✓

**biorsp/plotting/panels.py:**
Added new panel generators:

1. **`plot_examples_panel()`** - Panel C for spatial expression examples
   - Shows representative genes for each archetype
   - Displays spatial patterns with expression coloring
   - Configurable number of examples per archetype

2. **`plot_pairwise_panel()`** - Panel D for gene-gene co-patterning
   - Shows distribution of pairwise similarity scores
   - Highlights true module pairs when ground truth available
   - Includes Mann-Whitney p-value for separation

3. **`plot_marker_recovery_panel()`** - Alternative Panel C
   - Precision@K bar chart for structured gene recovery
   - Shows recovery performance at different k values

4. **`generate_full_panel_suite()`** - Orchestrator function
   - Generates complete A, B, C, D panel set
   - Handles both simulation and kidney modes

### biorsp/plotting/spec.py Updates

- Added `"abstention_stress": "#000000"` to `ARCHETYPE_COLORS`
- Added lowercase archetype descriptions to `ARCHETYPE_DESCRIPTIONS`

## File Changes Summary

| File | Changes |
|------|---------|
| `biorsp/plotting/spec.py` | Added abstention_stress color, lowercase descriptions |
| `biorsp/plotting/panels.py` | Added plot_examples_panel, plot_pairwise_panel, plot_marker_recovery_panel, generate_full_panel_suite |
| `biorsp/plotting/replot.py` | **NEW** - CLI for replotting from manifest |
| `biorsp/simulations/plotting.py` | Import colors from spec.py |
| `biorsp/simulations/io.py` | Added plot_spec parameter to write_manifest |
| `analysis/benchmarks/runners/run_story_onepager.py` | Use PlotSpec for cutoffs and colors |
| `analysis/kidney_atlas/runners/run_kpmp_archetypes_all_genes.py` | Import colors from spec.py |
| `analysis/kidney_atlas/runners/run_disease_stratified_analysis.py` | Import colors from spec.py |
| `analysis/kidney_atlas/runners/run_tal_analysis.py` | Use colors from spec.py |
| `tests/test_plot_standardization.py` | Added tests for new panel functions |

## Color Scheme (Canonical)

All plots now use the Material Design palette from `biorsp.plotting.spec`:

| Archetype | Hex Color | Description |
|-----------|-----------|-------------|
| Ubiquitous | `#4CAF50` | Green - widespread, no spatial bias |
| Gradient | `#2196F3` | Blue - broad spatial domain |
| Patchy | `#FF5722` | Red-Orange - localized marker |
| Basal | `#9E9E9E` | Gray - scattered/rare |
| Abstention | `#000000` | Black - insufficient data |

## Testing

All tests pass (10/10 in test_plot_standardization.py):
- `test_plotspec_classification_logic`
- `test_plotspec_abstention`
- `test_plotspec_dataframe_classification`
- `test_plotspec_colors_consistent`
- `test_plotspec_dataframe_validation`
- `test_plotspec_to_from_dict`
- `test_plotspec_cutoff_consistency`
- `test_panel_pairwise_with_truth` (new)
- `test_panel_marker_recovery` (new)
- `test_panel_examples_with_data` (new)

## Usage Examples

### Replot from Manifest

```bash
# Basic replot
python -m biorsp.plotting.replot --manifest outputs/archetypes/manifest.json

# Replot with PDF output
python -m biorsp.plotting.replot --manifest outputs/story/manifest.json --format pdf

# Replot to custom directory
python -m biorsp.plotting.replot --manifest outputs/kpmp/manifest.json --outdir figures/updated
```

### Generate Full Panel Suite

```python
from biorsp.plotting.panels import generate_full_panel_suite
from biorsp.plotting.spec import PlotSpec

spec = PlotSpec(c_cut=0.30, s_cut=0.15)
generate_full_panel_suite(
    df=results_df,
    spec=spec,
    outdir=Path("outputs/figures"),
    coords=cell_coords,  # Optional for Panel C
    expression=expr_matrix,  # Optional for Panel C
    gene_names=var_names,  # Optional for Panel C
    pairs_df=pairwise_scores,  # Optional for Panel D
    mode="simulation",
)
```

## Migration Notes

Scripts that previously defined local `ARCHETYPE_COLORS` should now:

1. Import from spec:
   ```python
   from biorsp.plotting.spec import ARCHETYPE_COLORS, ARCHETYPE_ORDER, PlotSpec
   ```

2. Use canonical order for legends:
   ```python
   for archetype in ARCHETYPE_ORDER:
       color = ARCHETYPE_COLORS[archetype]
       ...
   ```

3. Store plot_spec in manifest:
   ```python
   io.write_manifest(..., plot_spec=spec.to_dict())
   ```
