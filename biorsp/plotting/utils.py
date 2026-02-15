"""Shared plotting utilities used by figure factories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt

from biorsp.plotting.styles import DEFAULT_PLOT_STYLE, PlotStyle


def sanitize_feature_label(label: str, max_len: int = 40) -> str:
    """Create deterministic filesystem-safe stems for feature labels."""
    clean = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(label))
    clean = clean.strip("_") or "feature"
    return clean[:max_len]


def make_gene_stem(label: str, idx: int, used: dict[str, int], max_len: int = 64) -> str:
    """Build a unique deterministic filename stem for a gene."""
    base = sanitize_feature_label(label, max_len=max_len)
    if base not in used:
        used[base] = idx
        return base
    if used[base] == idx:
        return base
    suffix = f"__idx{idx}"
    allowed = max_len - len(suffix)
    if allowed <= 0:
        stem = suffix[-max_len:]
    else:
        stem = f"{base[:allowed]}{suffix}"
    used[stem] = idx
    return stem


def save_figure(
    fig: matplotlib.figure.Figure,
    out_path: Path,
    *,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    bbox_tight: bool = False,
    close: bool = True,
) -> None:
    """Save figure deterministically and optionally close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {
        "dpi": style.dpi,
        "facecolor": "white",
        "pad_inches": 0.02,
    }
    if bbox_tight:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(out_path, **save_kwargs)
    if close:
        plt.close(fig)
