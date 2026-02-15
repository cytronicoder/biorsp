"""Pipeline I/O, logging, and utility helpers."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ensure_dir(path: str | Path) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def setup_logger(log_path: Path, logger_name: str) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def detect_obs_col(adata, provided: str | None, candidates: Iterable[str]) -> str:
    if provided is not None:
        if provided in adata.obs.columns:
            return str(provided)
        raise KeyError(f"adata.obs['{provided}'] not found.")
    for c in candidates:
        if c in adata.obs.columns:
            return str(c)
    raise KeyError(f"Required column not found. Tried: {', '.join(candidates)}")


def normalize_label(label: str) -> str:
    return str(label).strip().lower()


def circular_sd(angles_rad: np.ndarray) -> float:
    ang = np.asarray(angles_rad, dtype=float)
    if ang.size == 0:
        return float("nan")
    sin_mean = float(np.mean(np.sin(ang)))
    cos_mean = float(np.mean(np.cos(ang)))
    r = math.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    if r <= 0:
        return float("nan")
    return math.sqrt(max(0.0, -2.0 * math.log(r)))


def write_scope_outputs(
    outdir: Path,
    *,
    metadata: dict[str, Any],
    stats: dict[str, Any],
) -> tuple[Path, Path]:
    metadata_path = outdir / "metadata.json"
    stats_path = outdir / "stats.json"
    write_json(metadata_path, metadata)
    write_json(stats_path, stats)
    return metadata_path, stats_path
