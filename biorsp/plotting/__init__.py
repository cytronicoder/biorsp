"""Unified plotting API for BioRSP analysis pipelines."""

from biorsp.plotting.classification import (
    CLASS_COLORS,
    CLASS_ORDER,
    class_counts_dict,
    plot_classification_suite,
    plot_score_qc_suite,
)
from biorsp.plotting.pairs import plot_pair_metrics
from biorsp.plotting.qc import (
    plot_categorical_umap,
    plot_umap_vantage_diagnostic,
    save_numeric_umap,
    write_cluster_celltype_counts,
)
from biorsp.plotting.rsp import (
    compute_and_plot_pair,
    plot_rsp,
    plot_rsp_to_file,
    plot_umap_rsp_pair,
)
from biorsp.plotting.style import (
    DEFAULT_STYLE,
    LABEL_ALPHA,
    LABEL_D,
    LABEL_DEFF,
    LABEL_NEGLOG10_KS,
    LABEL_PI,
    LABEL_PT,
    LABEL_SIGMA,
    apply_style,
    choose_text_color,
    finalize_fig,
    fmt_params,
    render_na_panel,
    safe_suptitle,
    should_plot,
)
from biorsp.plotting.styles import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    apply_plot_style,
    plot_style_dict,
)
from biorsp.plotting.utils import make_gene_stem, sanitize_feature_label, save_figure

__all__ = [
    "PlotStyle",
    "DEFAULT_PLOT_STYLE",
    "apply_plot_style",
    "plot_style_dict",
    "save_figure",
    "sanitize_feature_label",
    "make_gene_stem",
    "save_numeric_umap",
    "plot_umap_vantage_diagnostic",
    "plot_categorical_umap",
    "write_cluster_celltype_counts",
    "plot_rsp",
    "plot_rsp_to_file",
    "plot_umap_rsp_pair",
    "compute_and_plot_pair",
    "CLASS_ORDER",
    "CLASS_COLORS",
    "plot_classification_suite",
    "plot_score_qc_suite",
    "class_counts_dict",
    "plot_pair_metrics",
    "DEFAULT_STYLE",
    "apply_style",
    "safe_suptitle",
    "finalize_fig",
    "fmt_params",
    "should_plot",
    "render_na_panel",
    "choose_text_color",
    "LABEL_D",
    "LABEL_DEFF",
    "LABEL_SIGMA",
    "LABEL_PI",
    "LABEL_PT",
    "LABEL_NEGLOG10_KS",
    "LABEL_ALPHA",
]
