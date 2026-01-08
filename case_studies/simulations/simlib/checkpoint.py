"""
Checkpointing utilities for long-running benchmarks.

Enables resume functionality and periodic saving.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd


def make_checkpoint_key(config: Dict[str, Any], replicate: int) -> str:
    """
    Generate unique key for a (config, replicate) tuple.

    Parameters
    ----------
    config : dict
        Configuration dict (condition)
    replicate : int
        Replicate number

    Returns
    -------
    str
        Unique checkpoint key
    """
    # Sort keys for consistency
    config_items = sorted(config.items())
    return f"{config_items}__rep_{replicate}"


def load_completed_runs(runs_csv_path: Path) -> Set[str]:
    """
    Load completed runs from existing CSV.

    Parameters
    ----------
    runs_csv_path : Path
        Path to runs.csv

    Returns
    -------
    Set[str]
        Set of checkpoint keys for completed runs
    """
    if not runs_csv_path.exists():
        return set()

    try:
        df = pd.read_csv(runs_csv_path)
        if len(df) == 0:
            return set()

        # Extract config columns (exclude replicate, seed, and result columns)
        exclude_cols = {
            "replicate",
            "seed",
            "time",
            "p_value",
            "spatial_score",
            "coverage_expr",
            "abstain_flag",
            "similarity_profile",
            "copattern_score",
            "shared_mask_fraction",
        }
        config_cols = [c for c in df.columns if c not in exclude_cols]

        completed = set()
        for _, row in df.iterrows():
            config = {col: row[col] for col in config_cols}
            rep = int(row["replicate"])
            key = make_checkpoint_key(config, rep)
            completed.add(key)

        return completed
    except Exception as e:
        print(f"Warning: Failed to load checkpoint from {runs_csv_path}: {e}")
        return set()


def append_to_runs_csv(
    results: List[Dict[str, Any]],
    runs_csv_path: Path,
    overwrite: bool = False,
):
    """
    Append results to runs.csv (or overwrite if first write).

    Parameters
    ----------
    results : List[dict]
        List of result dictionaries
    runs_csv_path : Path
        Path to runs.csv
    overwrite : bool
        If True, overwrite existing file
    """
    if len(results) == 0:
        return

    df_new = pd.DataFrame(results)

    if overwrite or not runs_csv_path.exists():
        df_new.to_csv(runs_csv_path, index=False)
    else:
        # Append mode
        df_new.to_csv(runs_csv_path, mode="a", header=False, index=False)


def write_manifest(
    output_dir: Path,
    benchmark_name: str,
    params: Dict[str, Any],
    status: str = "running",
    completed: int = 0,
    total: int = 0,
    **metadata,
):
    """
    Write manifest.json with run metadata.

    Parameters
    ----------
    output_dir : Path
        Output directory
    benchmark_name : str
        Name of benchmark
    params : dict
        Run parameters
    status : str
        "running", "completed", "failed"
    completed : int
        Number of completed replicates
    total : int
        Total number of replicates
    **metadata
        Additional metadata (runtime, git hash, etc.)
    """
    manifest = {
        "benchmark": benchmark_name,
        "status": status,
        "progress": {
            "completed": completed,
            "total": total,
            "percent": 100.0 * completed / total if total > 0 else 0.0,
        },
        "params": params,
        "metadata": metadata,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
