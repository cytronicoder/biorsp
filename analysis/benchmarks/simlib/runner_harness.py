"""Shared runner harness utilities for benchmark normalization and contracts.

This module provides deterministic helpers used across simulation benchmarks to
normalize score tables, enforce label semantics, perform held-out splits, and
finalize outputs under the benchmark contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from analysis.benchmarks.simlib import io_contract
from analysis.benchmarks.simlib.metrics_ci import binomial_wilson_ci
from biorsp.utils.labels import (
    assert_archetype_labels,
    normalize_archetype_series,
)

CANONICAL_SCORE_COLUMNS = ["Coverage", "Spatial_Score"]
# Metadata columns that may be promised by runners. Only enforced if present.
CANONICAL_META_COLUMNS = [
    "gene",
    "case_id",
    "seed",
    "shape",
    "pattern",
    "prevalence",
    "signal_strength",
    "density",
    "n_cells",
    "delta_deg",
    "B",
    "timestamp",
]


def normalize_scores_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize score columns to the canonical benchmark schema.

    Args:
        df: Input runs DataFrame expected to contain `Coverage` and a spatial score
            column (`Spatial_Score` or legacy `Spatial_Bias_Score`).

    Returns:
        A copy of the input DataFrame with `Spatial_Score` present and validated.

    Raises:
        AssertionError: If required columns are missing, `Coverage` is outside
            `[0, 1]`, `Spatial_Score` contains non-finite values, or a metadata
            column is entirely NaN.

    Notes:
        If both `Spatial_Score` and `Spatial_Bias_Score` exist, the values must
        agree on non-NaN entries and the legacy column is dropped.
    """

    df_norm = df.copy()

    has_spatial = "Spatial_Score" in df_norm.columns
    has_bias = "Spatial_Bias_Score" in df_norm.columns
    if not has_spatial and has_bias:
        df_norm = df_norm.rename(columns={"Spatial_Bias_Score": "Spatial_Score"})
        has_spatial = True
    elif has_spatial and has_bias:
        lhs = df_norm["Spatial_Score"].to_numpy()
        rhs = df_norm["Spatial_Bias_Score"].to_numpy()
        mask = ~(np.isnan(lhs) | np.isnan(rhs))
        if mask.any() and not np.allclose(lhs[mask], rhs[mask]):
            raise AssertionError(
                "Spatial_Score and Spatial_Bias_Score disagree; ensure runner uses a single canonical column"
            )
        df_norm = df_norm.drop(columns=["Spatial_Bias_Score"])
    elif not has_spatial and not has_bias:
        raise AssertionError(
            "Missing spatial score column; expected Spatial_Score or Spatial_Bias_Score"
        )
    else:
        # Spatial_Score exists and Spatial_Bias_Score absent: nothing to do
        pass

    if "Coverage" not in df_norm.columns:
        raise AssertionError("Missing Coverage column; runners must emit Coverage")

    cov = df_norm["Coverage"].astype(float)
    cov_mask = cov.notna()
    if ((cov[cov_mask] < 0) | (cov[cov_mask] > 1)).any():
        raise AssertionError("Coverage values must lie in [0, 1]")

    spatial = df_norm["Spatial_Score"].astype(float)
    spatial_mask = spatial.notna()
    if not np.isfinite(spatial[spatial_mask]).all():
        raise AssertionError("Spatial_Score must be finite where defined; NaN denotes abstention")

    for meta_col in CANONICAL_META_COLUMNS:
        if meta_col in df_norm.columns:
            col = df_norm[meta_col]
            if col.isna().all():
                raise AssertionError(
                    f"Metadata column '{meta_col}' is entirely NaN; remove or populate it"
                )

    return df_norm


def normalize_labels(
    df: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    *,
    allow_abstain_pred: bool = True,
    allow_abstain_truth: bool = False,
) -> pd.DataFrame:
    """Normalize truth and prediction labels to canonical archetype names.

    Args:
        df: DataFrame containing label columns.
        truth_col: Column name for ground-truth archetypes.
        pred_col: Column name for predicted archetypes.
        allow_abstain_pred: Whether to permit abstention labels in predictions.
        allow_abstain_truth: Whether to permit abstention labels in the truth column.

    Returns:
        A copy of the input DataFrame with normalized label columns.

    Raises:
        KeyError: If either of the specified columns is missing.
        AssertionError: If labels contain unknown values after normalization.
    """

    df_norm = df.copy()

    if truth_col not in df_norm.columns:
        raise KeyError(f"Missing truth column '{truth_col}' for normalization")
    if pred_col not in df_norm.columns:
        raise KeyError(f"Missing prediction column '{pred_col}' for normalization")

    df_norm[truth_col] = normalize_archetype_series(
        df_norm[truth_col], allow_abstain=allow_abstain_truth
    )
    df_norm[pred_col] = normalize_archetype_series(
        df_norm[pred_col], allow_abstain=allow_abstain_pred
    )

    assert_archetype_labels(df_norm, truth_col, allow_abstain=allow_abstain_truth)
    assert_archetype_labels(df_norm, pred_col, allow_abstain=allow_abstain_pred)

    return df_norm


def compute_binomial_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute Wilson score confidence intervals for a binomial proportion.

    Args:
        k: Number of successes.
        n: Number of trials.
        alpha: Two-sided significance level.

    Returns:
        Lower and upper bounds of the confidence interval.
    """

    return binomial_wilson_ci(k, n, alpha=alpha)


@dataclass(frozen=True)
class SplitResult:
    train_idx: pd.Index
    test_idx: pd.Index


def split_train_test(
    df: pd.DataFrame, group_cols: list[str], test_frac: float, seed: int
) -> SplitResult:
    """Split groups into deterministic train/test indices.

    Groups defined by `group_cols` are shuffled deterministically using `seed`
    and assigned entirely to train/test according to `test_frac`, preventing
    leakage between splits.

    Args:
        df: DataFrame containing the grouping columns.
        group_cols: Column names that define groups (each group assigned to one split).
        test_frac: Fraction of groups assigned to the test split (must be in (0, 1)).
        seed: Random seed for deterministic shuffling.

    Returns:
        SplitResult with `train_idx` and `test_idx` indices into `df`.

    Raises:
        ValueError: If `test_frac` is not in (0, 1).
        KeyError: If any grouping column is missing.
        AssertionError: If the resulting train split is empty.
    """

    if not 0 < test_frac < 1:
        raise ValueError("test_frac must lie in (0, 1)")

    if not all(col in df.columns for col in group_cols):
        missing = [c for c in group_cols if c not in df.columns]
        raise KeyError(f"Missing group columns for split: {missing}")

    groups = df[group_cols].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n_test = max(1, int(np.floor(len(groups) * test_frac)))
    test_groups = groups.iloc[order[:n_test]]
    train_groups = groups.iloc[order[n_test:]]
    if train_groups.empty:
        raise AssertionError("Train split is empty; reduce test_frac or increase data size")

    def _mask(groups_df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(False, index=df.index)
        for _, row in groups_df.iterrows():
            cond = np.ones(len(df), dtype=bool)
            for col in group_cols:
                cond &= df[col] == row[col]
            mask |= cond
        return mask

    test_mask = _mask(test_groups)
    train_mask = _mask(train_groups)

    return SplitResult(train_idx=df.index[train_mask], test_idx=df.index[test_mask])


def safe_metric_mask(series: pd.Series) -> pd.Series:
    """Return a boolean mask indicating finite (non-abstained) metric values.

    Args:
        series: Metric series (e.g., `p_value` or `Spatial_Score`).

    Returns:
        Boolean mask where values are not NaN and finite.
    """

    return series.notna() & np.isfinite(series.to_numpy())


def finalize_contract(
    outdir: Path,
    runs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    manifest: Mapping[str, object],
    report_md: str,
    figures: Mapping[str, Path],
) -> None:
    """Write contract artifacts and validate them.

    Args:
        outdir: Output directory where `runs.csv`, `summary.csv`,
            `manifest.json`, and `report.md` are written.
        runs_df: Per-run results table containing contract-required columns.
        summary_df: Aggregated metrics table with confidence intervals.
        manifest: Extra metadata for `manifest.json` (merged with contract headers).
            Must include a `benchmark` field.
        report_md: Markdown report contents.
        figures: Mapping of figure identifiers to on-disk paths. Paths are stored
            in the manifest after existence checks.

    Raises:
        KeyError: If `benchmark` is missing from the manifest.
        FileNotFoundError: If any figure path is missing on disk.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "root": outdir,
        "runs_csv": outdir / "runs.csv",
        "summary_csv": outdir / "summary.csv",
        "manifest_json": outdir / "manifest.json",
        "report_md": outdir / "report.md",
    }

    benchmark = manifest.get("benchmark") if isinstance(manifest, Mapping) else None
    if not benchmark:
        raise KeyError("manifest must include 'benchmark' to validate contract")

    io_contract.assert_contract_runs(runs_df, benchmark=benchmark)
    io_contract.write_runs_csv(paths, runs_df, benchmark=benchmark)
    io_contract.write_summary_csv(paths, summary_df)
    io_contract.write_report_md(paths, report_md)

    figures_rel: dict[str, str] = {}
    for key, fpath in figures.items():
        fpath = Path(fpath)
        if not fpath.exists():
            raise FileNotFoundError(f"Figure missing on disk: {fpath}")
        figures_rel[key] = str(fpath.relative_to(outdir)) if fpath.is_absolute() else str(fpath)

    manifest_payload = {
        "contract_version": io_contract.CONTRACT_VERSION,
        "figures": figures_rel,
        **manifest,
    }
    with open(paths["manifest_json"], "w") as f:
        import json

        json.dump(manifest_payload, f, indent=2)

    io_contract.finalize_and_validate(paths)
