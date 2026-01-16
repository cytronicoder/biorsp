"""Plotting modules for BioRSP."""

from biorsp.plotting.embedding import compute_embedding, plot_embedding
from biorsp.plotting.panels import (
    generate_full_panel_suite,
    generate_standard_panels,
    plot_archetype_scatter,
    plot_composition_bar,
    plot_confusion_matrix,
    plot_examples_panel,
    plot_marker_recovery_panel,
    plot_pairwise_panel,
    save_panel_with_caption,
)
from biorsp.plotting.radar import (
    plot_localization_scatter,
    plot_phenotype_map,
    plot_radar,
    plot_radar_absolute,
    plot_radar_split,
    plot_summary,
)
from biorsp.plotting.spec import (
    ARCHETYPE_COLORS,
    ARCHETYPE_DESCRIPTIONS,
    ARCHETYPE_ORDER,
    PlotSpec,
)

__all__ = [
    "compute_embedding",
    "plot_embedding",
    "plot_localization_scatter",
    "plot_phenotype_map",
    "plot_radar",
    "plot_radar_absolute",
    "plot_radar_split",
    "plot_summary",
    "PlotSpec",
    "ARCHETYPE_COLORS",
    "ARCHETYPE_DESCRIPTIONS",
    "ARCHETYPE_ORDER",
    "plot_archetype_scatter",
    "plot_confusion_matrix",
    "plot_composition_bar",
    "plot_examples_panel",
    "plot_pairwise_panel",
    "plot_marker_recovery_panel",
    "generate_standard_panels",
    "generate_full_panel_suite",
    "save_panel_with_caption",
]
