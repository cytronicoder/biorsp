"""Shared helpers for BioRSP pipeline modules."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from biorsp.scoring import bh_fdr as _canonical_bh_fdr
from biorsp.utils import ensure_dir


def setup_logger(log_path: Path, logger_name: str) -> logging.Logger:
    ensure_dir(log_path.parent.as_posix())
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
    if provided:
        if provided in adata.obs.columns:
            return provided
        raise KeyError(f"adata.obs['{provided}'] not found.")
    for c in candidates:
        if c in adata.obs.columns:
            return c
    raise KeyError(f"Required column not found. Tried: {', '.join(candidates)}")


def normalize_label(label: str) -> str:
    return str(label).strip().lower()


def circular_sd(angles_rad: np.ndarray) -> float:
    ang = np.asarray(angles_rad, dtype=float)
    if ang.size == 0:
        return float("nan")
    sin_mean = float(np.mean(np.sin(ang)))
    cos_mean = float(np.mean(np.cos(ang)))
    R = math.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    if R <= 0:
        return float("nan")
    return math.sqrt(max(0.0, -2.0 * math.log(R)))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    return _canonical_bh_fdr(np.asarray(pvals, dtype=float))


def plot_null_hist(
    null_emax: np.ndarray, observed: float, out_png: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(null_emax, bins=25, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(observed, color="red", linestyle="--", linewidth=2, label="Observed")
    ax.set_title(title)
    ax.set_xlabel("E_max")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png.as_posix(), dpi=150)
    plt.close(fig)


def plot_qq(pvals: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    n = pvals.size
    if n == 0:
        return
    obs = np.sort(pvals)
    exp = np.arange(1, n + 1) / (n + 1)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(exp, obs, s=10, color="black")
    ax.plot([0, 1], [0, 1], color="red", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Expected p")
    ax.set_ylabel("Observed p")
    fig.tight_layout()
    fig.savefig(out_png.as_posix(), dpi=150)
    plt.close(fig)
