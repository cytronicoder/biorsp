import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from biorsp.plotting.standard import (
    get_archetype_palette,
    plot_calibration_qq,
    plot_confusion_matrix,
    plot_cs_scatter,
    plot_fpr_grid,
    plot_marker_recovery,
)


def test_palette_contains_canonical_labels():
    palette = get_archetype_palette()
    assert {"Ubiquitous", "Gradient", "Patchy", "Basal"}.issubset(palette.keys())


def test_plot_cs_scatter_smoke():
    df = pd.DataFrame(
        {
            "Coverage": [0.2, 0.8, 0.1, 0.9],
            "Spatial_Score": [0.6, 0.7, 0.05, 0.1],
            "Archetype_pred": ["Patchy", "Gradient", "Basal", "Ubiquitous"],
        }
    )
    fig = plot_cs_scatter(df, C_cut=0.3, S_cut=0.2)
    assert fig is not None


def test_plot_confusion_matrix_smoke():
    cm = np.array([[5, 1], [2, 7]])
    fig = plot_confusion_matrix(cm, labels=["A", "B"], normalize=None)
    assert fig is not None


def test_plot_calibration_qq_smoke():
    pvals = np.linspace(0.01, 0.99, 50)
    fig = plot_calibration_qq(pvals, alpha=0.05, perm_floor=0.01)
    assert fig is not None


def test_plot_fpr_grid_smoke():
    df = pd.DataFrame(
        {
            "metric": ["alpha_0.05", "alpha_0.10"],
            "mean": [0.04, 0.08],
            "ci_low": [0.02, 0.05],
            "ci_high": [0.06, 0.12],
        }
    )
    fig = plot_fpr_grid(df, alpha=0.05)
    assert fig is not None


def test_plot_marker_recovery_smoke():
    df = pd.DataFrame({"label_true": [1, 0, 1, 0, 0]})
    fig = plot_marker_recovery(df, k_list=[1, 3])
    assert fig is not None
