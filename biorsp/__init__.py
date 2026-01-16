"""BioRSP public surface.

Only public API entry points are exported here. Internal helpers should be
imported from their respective submodules.
"""

from biorsp._version import __version__
from biorsp.api import BioRSPConfig, classify_genes, score_gene_pairs, score_genes

__all__ = ["score_genes", "classify_genes", "score_gene_pairs", "BioRSPConfig", "__version__"]
