"""
Reproducibility manifest module for BioRSP.

Implements run manifest generation and validation:
- Software versions
- Parameters
- Seeds
- Checksums
"""

import json
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import scipy

from ._version import __version__


@dataclass
class BioRSPManifest:
    """
    Reproducibility manifest for a BioRSP run.
    """

    software_versions: Dict[str, str]
    parameters: Dict[str, Any]
    random_seed: int
    metadata: Dict[str, Any]


def create_manifest(
    parameters: Dict[str, Any],
    seed: int,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> BioRSPManifest:
    """
    Create a reproducibility manifest.

    Args:
        parameters: Dictionary of run parameters (B, delta, thresholds, etc.).
        seed: Random seed used.
        extra_metadata: Additional metadata (dataset ID, checksums).

    Returns:
        BioRSPManifest object.
    """
    software = {
        "biorsp": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }

    return BioRSPManifest(
        software_versions=software,
        parameters=parameters,
        random_seed=seed,
        metadata=extra_metadata or {},
    )


def save_manifest(manifest: BioRSPManifest, filepath: str) -> None:
    """Save manifest to JSON file using UTF-8 encoding."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)


def load_manifest(filepath: str) -> BioRSPManifest:
    """Load manifest from JSON file (expects UTF-8 encoding)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BioRSPManifest(**data)


__all__ = ["BioRSPManifest", "create_manifest", "save_manifest", "load_manifest"]
