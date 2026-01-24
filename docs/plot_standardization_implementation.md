# Plot standardization implementation

This page describes how standardized plots are generated from benchmark and case-study outputs.

## Canonical entry point

Use `biorsp.plotting.standard.make_standard_plot_set` to create the shared plot set. The function expects a DataFrame with `Coverage` and `Spatial_Score` columns (or `Spatial_Bias_Score`, which is normalized internally) and writes figures into the requested output directory.

```python
from pathlib import Path
from biorsp.plotting.standard import make_standard_plot_set

figures = make_standard_plot_set(
    scores_df=scores_df,
    outdir=Path("figures"),
    thresholds={"C_cut": 0.3, "S_cut": 0.15},
    truth_col="Archetype_true",
    pred_col="Archetype_pred",
)
```

## Threshold handling

- `C_cut` and `S_cut` define the quadrant boundaries used for archetype classification.
- Benchmarks that derive thresholds on a training split record them in the run directory and apply them to the test split.
- The plot function uses the provided thresholds for both classification and plotting.

## Debug panels

If `debug=True` and an embedding is provided, the function emits `fig_debug_embedding.png` to help validate foreground masks and spatial context.

## Data validation

The plotting utilities normalize archetype labels and check for finite scores. Missing optional inputs (truth labels or embeddings) lead to composition plots or placeholder panels rather than failures.
