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
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import scipy

from biorsp._version import __version__


def get_git_hash() -> str:
    """Get the current git commit hash if available."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


@dataclass
class BioRSPManifest:
    """
    Reproducibility manifest for a BioRSP run.
    """

    software_versions: Dict[str, str]
    parameters: Dict[str, Any]
    random_seed: int
    dataset_summary: Dict[str, Any]
    timings: Dict[str, float]
    metadata: Dict[str, Any]


def create_manifest(
    parameters: Dict[str, Any],
    seed: int,
    dataset_summary: Optional[Dict[str, Any]] = None,
    timings: Optional[Dict[str, float]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> BioRSPManifest:
    """
    Create a reproducibility manifest.

    Args:
        parameters: Dictionary of run parameters (B, delta, thresholds, etc.).
        seed: Random seed used.
        dataset_summary: Summary of the dataset (#cells, #genes, etc.).
        timings: Execution timings.
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
        "git_hash": get_git_hash(),
    }

    return BioRSPManifest(
        software_versions=software,
        parameters=parameters,
        random_seed=seed,
        dataset_summary=dataset_summary or {},
        timings=timings or {},
        metadata=extra_metadata or {},
    )


def save_manifest(manifest: BioRSPManifest, filepath: str) -> None:
    """Save manifest to JSON file using UTF-8 encoding."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)


def load_manifest(filepath: str) -> BioRSPManifest:
    """Load manifest from JSON file (expects UTF-8 encoding)."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return BioRSPManifest(**data)


__all__ = ["BioRSPManifest", "create_manifest", "save_manifest", "load_manifest"]
