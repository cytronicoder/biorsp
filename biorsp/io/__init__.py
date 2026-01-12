"""IO modules for BioRSP.

IO features:
- Smart index detection (no blind index_col=0)
- Robust type conversion
- align_inputs() to prevent silent mismatches
- Enhanced manifest with git dirty status and file fingerprints
"""

from biorsp.io.loaders import (
    align_inputs,
    load_expression_matrix,
    load_spatial_coords,
    load_umi_counts,
    save_results,
)
from biorsp.io.manifest import (
    BioRSPManifest,
    compute_file_fingerprint,
    create_manifest,
    get_git_hash,
    load_manifest,
    save_manifest,
)

__all__ = [
    "load_expression_matrix",
    "load_spatial_coords",
    "load_umi_counts",
    "save_results",
    "align_inputs",
    "BioRSPManifest",
    "create_manifest",
    "save_manifest",
    "load_manifest",
    "get_git_hash",
    "compute_file_fingerprint",
]
