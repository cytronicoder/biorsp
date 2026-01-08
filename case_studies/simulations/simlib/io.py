"""
I/O utilities for simulation benchmarks.

Provides CSV writers, manifest JSON generation, and output directory management.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def ensure_output_dir(benchmark_name: str, base_dir: str = "outputs") -> Path:
    """
    Create output directory for benchmark.

    Parameters
    ----------
    benchmark_name : str
        Name of benchmark (e.g., 'calibration', 'power')
    base_dir : str, optional
        Base output directory

    Returns
    -------
    output_dir : Path
        Path to output directory
    """
    output_dir = Path(base_dir) / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_runs_csv(runs_df: pd.DataFrame, output_dir: Path, filename: str = "runs.csv") -> None:
    """
    Write per-replicate runs CSV.

    Parameters
    ----------
    runs_df : pd.DataFrame
        Dataframe with one row per replicate
    output_dir : Path
        Output directory
    filename : str, optional
        Filename
    """
    filepath = output_dir / filename
    runs_df.to_csv(filepath, index=False)
    print(f"Wrote {len(runs_df)} runs to {filepath}")


def write_summary_csv(
    summary_df: pd.DataFrame, output_dir: Path, filename: str = "summary.csv"
) -> None:
    """
    Write aggregated summary CSV.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Dataframe with aggregated metrics
    output_dir : Path
        Output directory
    filename : str, optional
        Filename
    """
    filepath = output_dir / filename
    summary_df.to_csv(filepath, index=False)
    print(f"Wrote summary to {filepath}")


def write_manifest(
    output_dir: Path,
    benchmark_name: str,
    params: Dict,
    n_replicates: int,
    runtime_seconds: float,
    filename: str = "manifest.json",
) -> None:
    """
    Write manifest JSON with metadata.

    Parameters
    ----------
    output_dir : Path
        Output directory
    benchmark_name : str
        Name of benchmark
    params : Dict
        Parameters used
    n_replicates : int
        Number of replicates
    runtime_seconds : float
        Total runtime
    filename : str, optional
        Filename
    """
    # Get git commit hash
    git_commit = get_git_commit()

    manifest = {
        "benchmark": benchmark_name,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "n_replicates": n_replicates,
        "runtime_seconds": runtime_seconds,
        "parameters": params,
    }

    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {filepath}")


def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash.

    Returns
    -------
    commit_hash : str or None
        Git commit hash (short)
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_runs_csv(output_dir: Path, filename: str = "runs.csv") -> pd.DataFrame:
    """
    Load per-replicate runs CSV.

    Parameters
    ----------
    output_dir : Path
        Output directory
    filename : str, optional
        Filename

    Returns
    -------
    runs_df : pd.DataFrame
        Runs dataframe
    """
    filepath = output_dir / filename
    return pd.read_csv(filepath)


def load_summary_csv(output_dir: Path, filename: str = "summary.csv") -> pd.DataFrame:
    """
    Load aggregated summary CSV.

    Parameters
    ----------
    output_dir : Path
        Output directory
    filename : str, optional
        Filename

    Returns
    -------
    summary_df : pd.DataFrame
        Summary dataframe
    """
    filepath = output_dir / filename
    return pd.read_csv(filepath)


def load_manifest(output_dir: Path, filename: str = "manifest.json") -> Dict:
    """
    Load manifest JSON.

    Parameters
    ----------
    output_dir : Path
        Output directory
    filename : str, optional
        Filename

    Returns
    -------
    manifest : Dict
        Manifest dictionary
    """
    filepath = output_dir / filename
    with open(filepath) as f:
        return json.load(f)


def save_figure(fig, output_dir: Path, filename: str, dpi: int = 300) -> None:
    """
    Save matplotlib figure.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    output_dir : Path
        Output directory
    filename : str
        Filename (include extension)
    dpi : int, optional
        DPI for raster formats
    """
    figs_dir = output_dir.parent.parent / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    filepath = figs_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {filepath}")
