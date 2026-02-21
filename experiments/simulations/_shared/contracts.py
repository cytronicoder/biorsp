"""Shared contracts and schema validation helpers for simulation runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd


@dataclass(frozen=True)
class RunConfig:
    exp_name: str
    run_id: str
    run_dir: Path
    master_seed: int
    test_mode: bool
    args: dict[str, Any]


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    params: dict[str, Any]
    seed: int


@dataclass
class CellResult:
    cell_id: str
    success: bool
    runtime_sec: float
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ExperimentSummary:
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)
    plots: dict[str, str] = field(default_factory=dict)
    validations: list[str] = field(default_factory=list)


class CellStatusRow(TypedDict):
    cell_id: str
    success: bool
    runtime_sec: float
    error: str


def _missing_cols(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    have = set(str(c) for c in df.columns)
    return [str(c) for c in required_cols if str(c) not in have]


def validate_metrics_long(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    """Return missing required columns for long-form metrics schema."""
    miss = _missing_cols(df, required_cols)
    if miss:
        raise ValueError(f"metrics_long schema missing required columns: {miss}")
    return []


def validate_summary(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    """Return missing required columns for summary schema."""
    miss = _missing_cols(df, required_cols)
    if miss:
        raise ValueError(f"summary schema missing required columns: {miss}")
    return []
