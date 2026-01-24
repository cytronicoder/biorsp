"""Benchmark contract helpers for standardized outputs.

All benchmarks must emit the same core artifacts and adhere to the runs/summary
schema defined here. This module centralizes directory creation, manifest
writing, CSV validation, and final checks to prevent silent failures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from biorsp.utils.labels import assert_archetype_labels

CONTRACT_VERSION = "1.0.0"

UNIVERSAL_RUN_COLUMNS = [
    "run_id",
    "benchmark",
    "mode",
    "seed",
    "replicate_id",
    "status",
    "abstain_flag",
    "abstain_reason",
    "shape",
    "n_cells",
    "timestamp",
]

ARCHETYPE_RUN_COLUMNS = [
    "center_x",
    "center_y",
    "pattern_family",
    "pattern_variant",
    "target_prevalence",
    "prevalence_empirical",
    "n_fg",
    "Coverage",
    "Spatial_Score",
    "Directionality",
    "Archetype_true",
    "Archetype_pred",
    "C_cut",
    "S_cut",
    "thresholds_source",
]

CALIBRATION_RUN_COLUMNS = [
    "null_type",
    "test_stat",
    "p_value",
    "n_permutations",
    "perm_floor",
    "alpha",
    "is_fp",
]

GENEGENE_RUN_COLUMNS = [
    "gene_a",
    "gene_b",
    "pair_scenario",
    "similarity",
    "score",
    "label_true",
]

ROBUSTNESS_RUN_COLUMNS = [
    "transform",
    "transform_level",
    "metric_name",
    "metric_base",
    "metric_transformed",
    "delta_metric",
]

BENCHMARK_REQUIRED = {
    "archetypes": UNIVERSAL_RUN_COLUMNS + ARCHETYPE_RUN_COLUMNS,
    "calibration": UNIVERSAL_RUN_COLUMNS + CALIBRATION_RUN_COLUMNS,
    "genegene": UNIVERSAL_RUN_COLUMNS + GENEGENE_RUN_COLUMNS,
    "robustness": UNIVERSAL_RUN_COLUMNS + ROBUSTNESS_RUN_COLUMNS,
}

SUMMARY_REQUIRED = ["metric", "group_keys", "mean", "std", "n", "ci_low", "ci_high", "method"]


@dataclass
class BenchmarkContractConfig:
    """Configuration for a benchmark run adhering to the contract."""

    outdir: Path | str
    benchmark: str
    run_id: str
    seed: int
    mode: str
    git_commit: str | None = None
    package_versions: dict[str, str] | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["outdir"] = str(self.outdir)
        return data


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_run_dir(config: BenchmarkContractConfig) -> dict[str, Path]:
    """Initialize the run directory structure and return key paths."""

    root = Path(config.outdir) / config.benchmark / config.run_id
    _ensure_dir(root)
    _ensure_dir(root / "figures")
    _ensure_dir(root / "debug")

    return {
        "root": root,
        "runs_csv": root / "runs.csv",
        "summary_csv": root / "summary.csv",
        "manifest_json": root / "manifest.json",
        "report_md": root / "report.md",
        "figures": root / "figures",
        "debug": root / "debug",
    }


def _check_required_columns(df: pd.DataFrame, required: Iterable[str], benchmark: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{benchmark}: missing required columns {missing}. Present columns: {list(df.columns)}"
        )


def assert_contract_runs(df: pd.DataFrame, benchmark: str) -> None:
    if benchmark not in BENCHMARK_REQUIRED:
        raise ValueError(f"Unknown benchmark '{benchmark}' for contract validation")

    _check_required_columns(df, BENCHMARK_REQUIRED[benchmark], benchmark)

    key_cols = ["run_id", "benchmark", "mode", "seed", "replicate_id", "status", "timestamp"]
    for col in key_cols:
        if df[col].isna().any():
            raise AssertionError(f"Contract violation: column '{col}' contains NaN")

    if benchmark == "archetypes":
        required_non_null = [
            "shape",
            "n_cells",
            "pattern_family",
            "pattern_variant",
            "target_prevalence",
            "prevalence_empirical",
            "n_fg",
            "Coverage",
            "Spatial_Score",
            "Archetype_true",
            "Archetype_pred",
            "C_cut",
            "S_cut",
            "thresholds_source",
        ]
        non_abstain_mask = ~df.get("abstain_flag", False).astype(bool)
        for col in required_non_null:
            col_series = df[col]
            if col_series[non_abstain_mask].isna().any():
                raise AssertionError(
                    f"Contract violation: '{col}' cannot be NaN for archetypes non-abstain rows"
                )

    if "Coverage" in df.columns:
        c_vals = df["Coverage"].astype(float)
        if ((c_vals < 0) | (c_vals > 1)).any():
            raise AssertionError("Coverage must lie in [0, 1]")

    if "Spatial_Score" in df.columns:
        s_vals = df["Spatial_Score"].astype(float)
        if (s_vals < 0).any() or np.isinf(s_vals).any():
            raise AssertionError("Spatial_Score must be finite and non-negative")

    if "Archetype_true" in df.columns:
        assert_archetype_labels(df, "Archetype_true", allow_abstain=False)
    if "Archetype_pred" in df.columns:
        assert_archetype_labels(df, "Archetype_pred", allow_abstain=True)


def validate_runs_df(df: pd.DataFrame, benchmark: str) -> None:
    """Validate runs dataframe against the contract schema."""

    assert_contract_runs(df, benchmark)


def validate_summary_df(df: pd.DataFrame) -> None:
    """Validate summary dataframe to ensure CI columns exist."""

    _check_required_columns(df, SUMMARY_REQUIRED, benchmark="summary")

    if df.empty:
        raise ValueError("summary.csv is empty")

    if df["n"].isna().any():
        raise ValueError("summary.csv must include counts (column 'n')")


def write_runs_csv(paths: dict[str, Path], df_runs: pd.DataFrame, benchmark: str) -> None:
    validate_runs_df(df_runs, benchmark)
    df_runs.to_csv(paths["runs_csv"], index=False)


def write_summary_csv(paths: dict[str, Path], df_summary: pd.DataFrame) -> None:
    validate_summary_df(df_summary)
    df_summary.to_csv(paths["summary_csv"], index=False)


def write_report_md(paths: dict[str, Path], markdown_text: str) -> None:
    report_path = paths["report_md"]
    report_path.write_text(markdown_text.rstrip() + "\n")


def write_manifest(
    paths: dict[str, Path],
    config: BenchmarkContractConfig,
    metadata: dict[str, object],
) -> None:
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
    }
    manifest.update(metadata)
    with open(paths["manifest_json"], "w") as f:
        json.dump(manifest, f, indent=2)


def save_figure(paths: dict[str, Path], fig, name: str, subdir: str = "figures") -> Path:
    target_dir = paths.get(subdir, paths["root"] / subdir)
    _ensure_dir(target_dir)
    outfile = target_dir / name
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    return outfile


def save_debug(paths: dict[str, Path], fig, name: str, subdir: str = "debug") -> Path:
    return save_figure(paths, fig, name, subdir=subdir)


def finalize_and_validate(paths: dict[str, Path]) -> None:
    """Check that all required contract artifacts exist and are non-empty."""

    required_files = ["runs_csv", "summary_csv", "manifest_json", "report_md"]
    missing = [key for key in required_files if not paths.get(key, Path("missing")).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required outputs: {missing}")

    for key in required_files:
        fpath = paths[key]
        if fpath.stat().st_size == 0:
            raise ValueError(f"Output {fpath} is empty")

    runs_df = pd.read_csv(paths["runs_csv"])
    summary_df = pd.read_csv(paths["summary_csv"])

    if runs_df.empty:
        raise ValueError("runs.csv contains zero rows; check pipeline for silent failures")
    if summary_df.empty:
        raise ValueError("summary.csv contains zero rows; check aggregation logic")

    # Ensure manifest and report are readable
    paths["manifest_json"].read_text()
    paths["report_md"].read_text()
