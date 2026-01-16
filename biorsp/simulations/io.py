"""
I/O utilities for simulation benchmarks.

Provides CSV writers, manifest JSON generation, and output directory management.

Schema Versioning
-----------------
All outputs include a schema_version to enable backward-compatible parsing.
Current schema: v2.0

Output Contracts
----------------
Each benchmark MUST produce:
- runs.csv: One row per replicate (long format)
- summary.csv: Aggregated by condition
- manifest.json: Metadata including seed, config, runtime
- report.md: Human-readable interpretation

See SCHEMA.md for complete column documentation.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

SCHEMA_VERSION = "2.0"

REQUIRED_COLUMNS = {
    "calibration": {
        "runs": [
            "shape",
            "N",
            "null_type",
            "permutation_scheme",
            "p_value",
            "Spatial_Bias_Score",
            "Coverage",
            "abstain_flag",
        ],
        "summary": [
            "shape",
            "N",
            "null_type",
            "permutation_scheme",
            "fpr_05",
            "fpr_05_ci_low",
            "fpr_05_ci_high",
            "ks_stat",
            "abstain_rate",
            "n_reps",
        ],
    },
    "archetypes": {
        "runs": [
            "shape",
            "N",
            "coverage_regime",
            "organization_regime",
            "true_archetype",
            "Spatial_Bias_Score",
            "Coverage",
            "abstain_flag",
        ],
        "summary": [
            "shape",
            "N",
            "coverage_regime",
            "organization_regime",
            "true_archetype",
            "Spatial_Bias_Score_mean",
            "Coverage_mean",
            "classification_accuracy",
            "abstain_rate",
            "n_reps",
        ],
    },
    "genegene": {
        "runs": [
            "shape",
            "N",
            "scenario",
            "similarity_profile",
            "copattern_score",
            "shared_mask_fraction",
        ],
        "summary": [
            "shape",
            "N",
            "scenario",
            "similarity_profile_mean",
            "copattern_score_mean",
            "n_reps",
        ],
    },
    "robustness": {
        "runs": [
            "shape",
            "N",
            "pattern",
            "distortion_kind",
            "distortion_strength",
            "baseline_spatial_score",
            "distorted_spatial_score",
            "delta_s",
        ],
        "summary": [
            "shape",
            "N",
            "pattern",
            "distortion_kind",
            "distortion_strength",
            "abs_delta_median",
            "correlation",
            "is_stable",
            "n_pairs",
        ],
    },
}


def validate_output_schema(df: pd.DataFrame, benchmark: str, output_type: str) -> None:
    """
    Validate DataFrame against expected schema.

    Parameters
    ----------
    df : pd.DataFrame
        Output DataFrame
    benchmark : str
        Benchmark name (calibration, archetypes, genegene, robustness)
    output_type : str
        Output type ('runs' or 'summary')

    Raises
    ------
    ValueError
        If required columns are missing
    """
    if benchmark not in REQUIRED_COLUMNS:
        return

    required = REQUIRED_COLUMNS[benchmark].get(output_type, [])
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Schema validation failed for {benchmark}/{output_type}. "
            f"Missing columns: {missing}. "
            f"Available: {list(df.columns)}. "
            f"Schema version: {SCHEMA_VERSION}"
        )


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


def write_runs_csv(
    runs_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "runs.csv",
    benchmark: Optional[str] = None,
) -> None:
    """
    Write per-replicate runs CSV with schema validation.

    Parameters
    ----------
    runs_df : pd.DataFrame
        Dataframe with one row per replicate
    output_dir : Path
        Output directory
    filename : str, optional
        Filename
    benchmark : str, optional
        Benchmark name for schema validation
    """

    runs_df = runs_df.copy()
    runs_df["schema_version"] = SCHEMA_VERSION

    if benchmark:
        validate_output_schema(runs_df, benchmark, "runs")

    filepath = output_dir / filename
    runs_df.to_csv(filepath, index=False)
    print(f"Wrote {len(runs_df)} runs to {filepath}")


def write_summary_csv(
    summary_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "summary.csv",
    benchmark: Optional[str] = None,
) -> None:
    """
    Write aggregated summary CSV with schema validation.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Dataframe with aggregated metrics
    output_dir : Path
        Output directory
    filename : str, optional
        Filename
    benchmark : str, optional
        Benchmark name for schema validation
    """

    summary_df = summary_df.copy()
    summary_df["schema_version"] = SCHEMA_VERSION

    if benchmark:
        validate_output_schema(summary_df, benchmark, "summary")

    filepath = output_dir / filename
    summary_df.to_csv(filepath, index=False)
    print(f"Wrote summary to {filepath}")


def write_manifest(
    output_dir: Path,
    benchmark_name: str,
    params: Dict,
    n_replicates: int,
    runtime_seconds: float,
    biorsp_config: Optional[Any] = None,
    filename: str = "manifest.json",
    plot_spec: Optional[Dict] = None,
    **extra_metadata,
) -> None:
    """
    Write manifest JSON with complete metadata.

    Parameters
    ----------
    output_dir : Path
        Output directory
    benchmark_name : str
        Name of benchmark
    params : Dict
        CLI parameters used
    n_replicates : int
        Number of replicates
    runtime_seconds : float
        Total runtime
    biorsp_config : BioRSPConfig, optional
        BioRSP configuration object (will be serialized)
    filename : str, optional
        Filename
    plot_spec : Dict, optional
        Plot specification dictionary for reproducible re-plotting
    **extra_metadata
        Additional key-value pairs to include in manifest
    """

    git_commit = get_git_commit()

    config_dict = None
    if biorsp_config is not None:
        config_dict = serialize_biorsp_config(biorsp_config)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark_name,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "n_replicates": n_replicates,
        "runtime_seconds": runtime_seconds,
        "parameters": params,
        "biorsp_config": config_dict,
    }

    if plot_spec is not None:
        manifest["plot_spec"] = plot_spec

    manifest.update(extra_metadata)

    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {filepath}")


def serialize_biorsp_config(config: Any) -> Dict[str, Any]:
    """
    Serialize BioRSPConfig to dictionary.

    Parameters
    ----------
    config : BioRSPConfig
        Configuration object

    Returns
    -------
    config_dict : Dict
        Serializable dictionary
    """

    return {
        "B": getattr(config, "B", None),
        "delta_deg": getattr(config, "delta_deg", None),
        "n_permutations": getattr(config, "n_permutations", None),
        "min_coverage_fg": getattr(config, "min_coverage_fg", None),
        "min_coverage_geom": getattr(config, "min_coverage_geom", None),
        "min_sector_coverage": getattr(config, "min_sector_coverage", None),
        "empty_fg_policy": getattr(config, "empty_fg_policy", None),
        "stratification_keys": getattr(config, "stratification_keys", None),
        "adequacy_thresholds": getattr(config, "adequacy_thresholds", None),
    }


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
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {filepath}")
