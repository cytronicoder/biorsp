"""Shared matplotlib style and formatting helpers for simulation plotting."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import ensure_dir

DPI = 200
FIGSIZE_SMALL = (6.0, 5.0)
FIGSIZE_WIDE = (8.0, 4.5)
FIGSIZE_GRID_3COL = (10.0, 6.0)

FONT_SIZES = {
    "suptitle": 16,
    "title": 13,
    "label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 10,
}

LINE_WIDTH = 1.8
MARKER_SIZE = 4.5
ALPHA_LINE = 0.9
ALPHA_FILL = 0.18
ALPHA_SCATTER_BG = 0.10
ALPHA_SCATTER_FG = 0.90

COLORBAR_SHRINK = 0.96
COLORBAR_PAD = 0.02


def apply_style() -> None:
    """Apply deterministic plot styling via matplotlib rcParams."""
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.size": FONT_SIZES["tick"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["label"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "legend.fontsize": FONT_SIZES["legend"],
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "axes.grid": False,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.default": "regular",
        }
    )


def safe_suptitle(fig: plt.Figure, text: str, *, y: float = 0.98) -> None:
    """Place a suptitle and reserve top space to avoid panel/title collisions."""
    fig.suptitle(str(text), fontsize=FONT_SIZES["suptitle"], y=float(y))
    target_top = float(y) - 0.07
    if target_top > 0.55:
        fig.subplots_adjust(top=min(fig.subplotpars.top, target_top))


def finalize_fig(fig: plt.Figure, outpath: str | Path) -> Path:
    """Finalize, save, and close a figure safely."""
    out = Path(outpath)
    ensure_dir(out.parent)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout",
            )
            fig.tight_layout()
    except (RuntimeError, ValueError):
        pass
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def fmt_param_sigma(sigma_eta: float) -> str:
    return rf"$\sigma_{{\eta}}={float(sigma_eta):g}$"


def fmt_param_pi(pi: float) -> str:
    return rf"$\pi_{{\mathrm{{target}}}}={float(pi):.2f}$"


def fmt_param_D(D: int) -> str:
    return rf"$D={int(D)}$"


def fmt_param_Deff(val: float) -> str:
    return rf"$D_{{\mathrm{{eff}}}}={float(val):.1f}$"


def fmt_title_expA(
    prefix: str,
    D: int | None = None,
    sigma: float | None = None,
    pi: float | None = None,
) -> str:
    params: list[str] = []
    if D is not None:
        params.append(fmt_param_D(int(D)))
    if sigma is not None:
        params.append(fmt_param_sigma(float(sigma)))
    if pi is not None:
        params.append(fmt_param_pi(float(pi)))
    if not params:
        return str(prefix)
    return f"{prefix}\n" + ", ".join(params)


def annotate_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    im,
    fmt: str = "{:.3g}",
) -> None:
    """Annotate heatmap cells with contrast-aware text color."""
    arr = np.asarray(data, dtype=float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                label = fmt.format(val)
                r, g, b, _ = im.cmap(im.norm(val))
                luminance = 0.2126 * float(r) + 0.7152 * float(g) + 0.0722 * float(b)
                text_color = "black" if luminance > 0.55 else "white"
            else:
                label = "NA"
                text_color = "black"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=FONT_SIZES["annotation"],
                color=text_color,
            )
