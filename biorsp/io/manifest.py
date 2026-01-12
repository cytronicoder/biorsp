"""Reproducibility manifest module for BioRSP.

Implements run manifest generation and validation:
- Software versions (biorsp, python, dependencies)
- Parameters (config serialization)
- Seeds
- File fingerprints (checksums)
- Git provenance (commit hash, dirty status)

Schema version: 2.0
"""

import contextlib
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import scipy

from biorsp._version import __version__

logger = logging.getLogger(__name__)


def get_git_hash() -> Tuple[str, bool]:
    """Get the current git commit hash and dirty status.

    Returns:
        Tuple of (commit_hash, is_dirty)
        Returns ("unknown", False) if git not available or not a repo.
    """
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )

        try:
            subprocess.check_output(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL)
            is_dirty = False
        except subprocess.CalledProcessError:
            is_dirty = True

        return commit, is_dirty
    except Exception:
        return "unknown", False


def compute_file_fingerprint(
    path: Path, mode: Literal["fast", "strict"] = "fast"
) -> Dict[str, Any]:
    """Compute file fingerprint for reproducibility.

    Args:
        path: Path to file
        mode: "fast" (size + mtime) or "strict" (sha256 hash)

    Returns:
        Dict with path, size, mtime, and optionally sha256
    """
    if not path.exists():
        return {"path": str(path), "exists": False}

    stat = path.stat()
    fingerprint = {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
    }

    if mode == "strict":
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        fingerprint["sha256"] = sha256.hexdigest()

    return fingerprint


@dataclass
class BioRSPManifest:
    """Reproducibility manifest for a BioRSP run.

    Schema version 2.0 changes:
    - Added schema_version field
    - Added git_dirty boolean
    - Added input_fingerprints for file provenance
    - Enhanced software_versions with more dependencies
    - Improved config serialization robustness
    """

    schema_version: str = "2.0"
    software_versions: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    random_seed: Optional[int] = None
    dataset_summary: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    input_fingerprints: Dict[str, Any] = field(default_factory=dict)


def _serialize_config(config: Any) -> Dict[str, Any]:
    """Robustly serialize config object to dict.

    Tries multiple strategies:
    1. to_dict() method (custom)
    2. model_dump() method (pydantic)
    3. asdict() (dataclasses)
    4. vars() (regular classes)
    """
    if hasattr(config, "to_dict"):
        return config.to_dict()
    elif hasattr(config, "model_dump"):
        return config.model_dump()
    elif hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    elif hasattr(config, "__dict__"):
        return vars(config)
    else:
        try:
            return dict(config)
        except Exception:
            logger.warning(f"Could not serialize config of type {type(config)}. Storing as string.")
            return {"_repr": str(config)}


def create_manifest(
    parameters: Any,
    seed: Optional[int] = None,
    dataset_summary: Optional[Dict[str, Any]] = None,
    timings: Optional[Dict[str, float]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    input_files: Optional[Dict[str, Path]] = None,
    fingerprint_mode: Literal["fast", "strict"] = "fast",
) -> BioRSPManifest:
    """Create a reproducibility manifest.

    Args:
        parameters: Config object or dict of run parameters (B, delta, thresholds, etc.).
        seed: Random seed used.
        dataset_summary: Summary of the dataset (#cells, #genes, etc.).
        timings: Execution timings.
        extra_metadata: Additional metadata (dataset ID, alignment report, etc.).
        input_files: Dict mapping input names to Paths for fingerprinting.
        fingerprint_mode: "fast" (size+mtime) or "strict" (sha256).

    Returns:
        BioRSPManifest object ready for JSON serialization.
    """
    git_hash, git_dirty = get_git_hash()
    software = {
        "biorsp": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "git_hash": git_hash,
        "git_dirty": git_dirty,
    }

    with contextlib.suppress(ImportError, importlib.metadata.PackageNotFoundError):
        software["anndata"] = importlib.metadata.version("anndata")

    with contextlib.suppress(ImportError, importlib.metadata.PackageNotFoundError):
        software["scanpy"] = importlib.metadata.version("scanpy")

    with contextlib.suppress(ImportError, importlib.metadata.PackageNotFoundError):
        software["matplotlib"] = importlib.metadata.version("matplotlib")

    params_dict = parameters if isinstance(parameters, dict) else _serialize_config(parameters)

    def convert_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        return obj

    params_dict = convert_paths(params_dict)

    fingerprints = {}
    if input_files:
        for name, path in input_files.items():
            fingerprints[name] = compute_file_fingerprint(Path(path), mode=fingerprint_mode)

    return BioRSPManifest(
        schema_version="2.0",
        software_versions=software,
        parameters=params_dict,
        random_seed=seed,
        dataset_summary=dataset_summary or {},
        timings=timings or {},
        metadata=extra_metadata or {},
        input_fingerprints=fingerprints,
    )


def save_manifest(manifest: BioRSPManifest, filepath: str) -> None:
    """Save manifest to JSON file with stable formatting.

    Creates parent directories if needed.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2, sort_keys=True)


def load_manifest(filepath: str) -> BioRSPManifest:
    """Load manifest from JSON file (expects UTF-8 encoding)."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return BioRSPManifest(**data)


__all__ = [
    "BioRSPManifest",
    "create_manifest",
    "save_manifest",
    "load_manifest",
    "get_git_hash",
    "compute_file_fingerprint",
]
