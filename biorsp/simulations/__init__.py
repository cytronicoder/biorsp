"""
BioRSP Simulation Library (simlib)

A modular framework for generating synthetic spatial transcriptomics datasets
and benchmarking spatial methods.

Public API
----------
RNG:
    make_rng, condition_key, seed_all

Shapes:
    generate_coords

Distortions:
    apply_distortion

Geometry:
    compute_polar, radial_density_proxy

Density:
    kde_density, knn_density

Expression:
    simulate_library_size, generate_signal_field, generate_expression_from_field,
    generate_confounded_null

Datasets:
    make_gene_panel, make_module_panel, package_as_anndata

Scoring:
    score_dataset, score_pairs

Metrics:
    fpr_with_ci, ks_uniform, qq_quantiles, power_with_ci, confusion_matrix,
    macro_f1, auprc, topk_precision, median_abs_delta, flip_rate, kendall_tau

Plotting:
    plot_qq, plot_fpr_grid, plot_power_curve, plot_confusion_matrix, plot_pr_curve,
    plot_robustness_delta, plot_spatial_embedding

I/O:
    ensure_output_dir, write_runs_csv, write_summary_csv, write_manifest,
    load_runs_csv, load_summary_csv, load_manifest, save_figure

Docs:
    write_report, interpret_calibration, interpret_power, interpret_archetypes,
    interpret_genegene, interpret_robustness

Sweeps:
    expand_grid, run_replicates, aggregate_replicates, stratify_by, replicate_seed
"""

__version__ = "3.0.0"

# Import canonical plot specification
from biorsp.plotting.spec import ARCHETYPE_COLORS, ARCHETYPE_DESCRIPTIONS

from .cache import GeometryCache, clear_cache, get_cache
from .datasets import (
    make_factorial_panel,
    make_gene_panel,
    make_module_panel,
    make_module_panel_structured,
    package_as_anndata,
)
from .density import kde_density, knn_density
from .distortions import apply_distortion
from .docs import (
    interpret_archetypes,
    interpret_calibration,
    interpret_genegene,
    interpret_power,
    interpret_robustness,
    write_report,
)
from .expression import (
    generate_confounded_null,
    generate_expression_from_field,
    generate_expression_targeted,
    generate_factorial_gene,
    generate_signal_field,
    simulate_library_size,
)
from .geometry import compute_polar, radial_density_proxy
from .io import (  # noqa: F401
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    ensure_output_dir,
    load_manifest,
    load_runs_csv,
    load_summary_csv,
    save_figure,
    serialize_biorsp_config,
    validate_output_schema,
    write_manifest,
    write_runs_csv,
    write_summary_csv,
)
from .metrics import (
    auprc,
    classify_by_quadrant,
    compute_classification_metrics,
    confusion_matrix,
    derive_thresholds_from_null,
    embedding_stability_metrics,
    flip_rate,
    fpr_with_ci,
    kendall_tau,
    ks_uniform,
    macro_f1,
    median_abs_delta,
    module_recovery_metrics,
    power_with_ci,
    precision_at_k_curve,
    qq_quantiles,
    topk_precision,
)
from .plotting import (
    plot_abstention_summary,
    plot_archetype_scatter,
    plot_confusion_matrix,
    plot_confusion_matrix_styled,
    plot_fpr_grid,
    plot_marker_recovery,
    plot_module_recovery,
    plot_power_curve,
    plot_pr_curve,
    plot_qq,
    plot_robustness_delta,
    plot_spatial_embedding,
    plot_stability_summary,
)
from .rng import condition_key, make_rng, seed_all
from .scoring import score_dataset, score_pairs
from .shapes import generate_coords
from .sweeps import (
    aggregate_replicates,
    expand_grid,
    replicate_seed,
    run_replicates,
    stratify_by,
)

__all__ = [
    "make_rng",
    "condition_key",
    "seed_all",
    "generate_coords",
    "apply_distortion",
    "compute_polar",
    "radial_density_proxy",
    "kde_density",
    "knn_density",
    "simulate_library_size",
    "generate_signal_field",
    "generate_expression_from_field",
    "generate_expression_targeted",
    "generate_factorial_gene",
    "generate_confounded_null",
    "make_gene_panel",
    "make_module_panel",
    "make_factorial_panel",
    "make_module_panel_structured",
    "package_as_anndata",
    "score_dataset",
    "score_pairs",
    "fpr_with_ci",
    "ks_uniform",
    "qq_quantiles",
    "power_with_ci",
    "confusion_matrix",
    "macro_f1",
    "auprc",
    "topk_precision",
    "median_abs_delta",
    "flip_rate",
    "kendall_tau",
    "derive_thresholds_from_null",
    "classify_by_quadrant",
    "compute_classification_metrics",
    "precision_at_k_curve",
    "module_recovery_metrics",
    "embedding_stability_metrics",
    "plot_qq",
    "plot_fpr_grid",
    "plot_power_curve",
    "plot_confusion_matrix",
    "plot_confusion_matrix_styled",
    "plot_pr_curve",
    "plot_robustness_delta",
    "plot_spatial_embedding",
    "plot_archetype_scatter",
    "plot_marker_recovery",
    "plot_module_recovery",
    "plot_abstention_summary",
    "plot_stability_summary",
    "ARCHETYPE_COLORS",
    "ARCHETYPE_DESCRIPTIONS",
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION",
    "serialize_biorsp_config",
    "validate_output_schema",
    "ensure_output_dir",
    "write_runs_csv",
    "write_summary_csv",
    "write_manifest",
    "load_runs_csv",
    "load_summary_csv",
    "load_manifest",
    "save_figure",
    "write_report",
    "interpret_calibration",
    "interpret_power",
    "interpret_archetypes",
    "interpret_genegene",
    "interpret_robustness",
    "expand_grid",
    "run_replicates",
    "aggregate_replicates",
    "stratify_by",
    "replicate_seed",
    "GeometryCache",
    "clear_cache",
    "get_cache",
]
