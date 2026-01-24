"""Plotting modules for BioRSP."""

from biorsp.plotting.embedding import compute_embedding, plot_embedding
from biorsp.plotting.make_figures import make_figures
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
    load_spec_from_manifest,
)
from biorsp.plotting.story import (
    generate_onepager,
    generate_onepager_from_dir,
)

__all__ = [
    # Embedding
    "compute_embedding",
    "plot_embedding",
    # Radar
    "plot_localization_scatter",
    "plot_phenotype_map",
    "plot_radar",
    "plot_radar_absolute",
    "plot_radar_split",
    "plot_summary",
    # Spec
    "PlotSpec",
    "ARCHETYPE_COLORS",
    "ARCHETYPE_DESCRIPTIONS",
    "ARCHETYPE_ORDER",
    "load_spec_from_manifest",
    # Panels
    "plot_archetype_scatter",
    "plot_confusion_matrix",
    "plot_composition_bar",
    "plot_examples_panel",
    "plot_pairwise_panel",
    "plot_marker_recovery_panel",
    "generate_standard_panels",
    "generate_full_panel_suite",
    "save_panel_with_caption",
    # Story
    "generate_onepager",
    "generate_onepager_from_dir",
    # CLI
    "make_figures",
]
