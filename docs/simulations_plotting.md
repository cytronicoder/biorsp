# Standardized plotting for simulations and case studies

BioRSP provides a shared plot set for both simulation benchmarks and kidney case studies via `biorsp.plotting.standard.make_standard_plot_set`.

## Standard plot set

The standard plot set consists of the following files (PNG by default):

| Identifier | Filename | Description |
| --- | --- | --- |
| Scatter | `fig_cs_scatter.png` | Coverage vs. spatial score scatter with thresholds. |
| Marginals | `fig_cs_marginals.png` | Histograms for coverage and spatial scores. |
| Confusion/Composition | `fig_confusion_or_composition.png` | Confusion matrix if truth labels exist; otherwise archetype composition. |
| Examples | `fig_archetype_examples.png` | Example genes per archetype (top spatial scores). |
| Top tables | `fig_top_tables.png` | Bar charts for top coverage and top spatial score genes. |
| Debug (optional) | `fig_debug_embedding.png` | Embedding overlay with a foreground mask when `debug=True`. |

The function also writes `examples/example_metadata.csv` when per-archetype examples are available.

## Programmatic usage

```python
from pathlib import Path
import pandas as pd
from biorsp.plotting.standard import make_standard_plot_set

scores_df = pd.read_csv("runs.csv")
figures = make_standard_plot_set(
    scores_df=scores_df,
    outdir=Path("figures"),
    thresholds={"C_cut": 0.3, "S_cut": 0.15},
    truth_col="Archetype_true",
    pred_col="Archetype_pred",
)
```

The function returns a dictionary mapping figure identifiers to saved file paths.

## Benchmark definitions (summary)

See `analysis/benchmarks/README.md` for full details. Each benchmark uses synthetic inputs and writes contract-compliant outputs.

- **Archetypes**: classification accuracy and abstention on held-out splits.
- **Calibration**: null p-value calibration with held-out thresholds.
- **Null calibration**: derive `C_cut` and `S_cut` from null simulations.
- **Robustness**: paired distortions and metric deltas.
- **Stability**: resampling stability of Coverage and Spatial_Score.
- **Gene–gene**: pairwise co-patterning behavior in synthetic scenarios.
- **Abstention**: abstention rates under sparse or underpowered signals.
