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

# Datasets
from .datasets import make_gene_panel, make_module_panel, package_as_anndata

# Density
from .density import kde_density, knn_density

# Distortions
from .distortions import apply_distortion

# Docs
from .docs import (
    interpret_archetypes,
    interpret_calibration,
    interpret_genegene,
    interpret_power,
    interpret_robustness,
    write_report,
)

# Expression
from .expression import (
    generate_confounded_null,
    generate_expression_from_field,
    generate_signal_field,
    simulate_library_size,
)

# Geometry
from .geometry import compute_polar, radial_density_proxy

# I/O
from .io import (
    ensure_output_dir,
    load_manifest,
    load_runs_csv,
    load_summary_csv,
    save_figure,
    write_manifest,
    write_runs_csv,
    write_summary_csv,
)

# Metrics
from .metrics import (
    auprc,
    confusion_matrix,
    flip_rate,
    fpr_with_ci,
    kendall_tau,
    ks_uniform,
    macro_f1,
    median_abs_delta,
    power_with_ci,
    qq_quantiles,
    topk_precision,
)

# Plotting
from .plotting import (
    plot_confusion_matrix,
    plot_fpr_grid,
    plot_power_curve,
    plot_pr_curve,
    plot_qq,
    plot_robustness_delta,
    plot_spatial_embedding,
)

# RNG
from .rng import condition_key, make_rng, seed_all

# Scoring
from .scoring import score_dataset, score_pairs

# Shapes
from .shapes import generate_coords

# Sweeps
from .sweeps import (
    aggregate_replicates,
    expand_grid,
    replicate_seed,
    run_replicates,
    stratify_by,
)

__all__ = [
    # RNG
    "make_rng",
    "condition_key",
    "seed_all",
    # Shapes
    "generate_coords",
    # Distortions
    "apply_distortion",
    # Geometry
    "compute_polar",
    "radial_density_proxy",
    # Density
    "kde_density",
    "knn_density",
    # Expression
    "simulate_library_size",
    "generate_signal_field",
    "generate_expression_from_field",
    "generate_confounded_null",
    # Datasets
    "make_gene_panel",
    "make_module_panel",
    "package_as_anndata",
    # Scoring
    "score_dataset",
    "score_pairs",
    # Metrics
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
    # Plotting
    "plot_qq",
    "plot_fpr_grid",
    "plot_power_curve",
    "plot_confusion_matrix",
    "plot_pr_curve",
    "plot_robustness_delta",
    "plot_spatial_embedding",
    # I/O
    "ensure_output_dir",
    "write_runs_csv",
    "write_summary_csv",
    "write_manifest",
    "load_runs_csv",
    "load_summary_csv",
    "load_manifest",
    "save_figure",
    # Docs
    "write_report",
    "interpret_calibration",
    "interpret_power",
    "interpret_archetypes",
    "interpret_genegene",
    "interpret_robustness",
    # Sweeps
    "expand_grid",
    "run_replicates",
    "aggregate_replicates",
    "stratify_by",
    "replicate_seed",
]
