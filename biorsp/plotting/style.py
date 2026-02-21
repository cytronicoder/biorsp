"""Centralized publication-style plotting defaults and mathtext label helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotStyle:
    dpi: int = 200
    fig_small: tuple[float, float] = (6.0, 5.0)
    fig_wide: tuple[float, float] = (8.0, 4.5)
    fig_grid3: tuple[float, float] = (10.0, 6.0)
    fs_suptitle: int = 16
    fs_title: int = 13
    fs_label: int = 12
    fs_tick: int = 10
    fs_legend: int = 10
    fs_annot: int = 10
    line_width: float = 1.8
    marker_size: float = 4.8
    alpha_line: float = 0.90
    alpha_fill: float = 0.18
    alpha_bg: float = 0.10
    alpha_fg: float = 0.90
    cbar_shrink: float = 0.96
    cbar_pad: float = 0.02
    cmap_heat: str = "viridis"
    cmap_ks: str = "magma"
    cmap_counts: str = "cividis"
    n_min: int = 10


DEFAULT_STYLE = PlotStyle()

LABEL_D = r"$D$"
LABEL_DEFF = r"$D_{\mathrm{eff}}$"
LABEL_SIGMA = r"$\sigma_\eta$"
LABEL_PI = r"$\pi_{\mathrm{target}}$"
LABEL_PT = r"$p_T$"
LABEL_NEGLOG10_KS = r"$-\log_{10}(p_{KS})$"
LABEL_ALPHA = r"$\alpha$"


def apply_style(style: PlotStyle = DEFAULT_STYLE) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": style.dpi,
            "savefig.dpi": style.dpi,
            "font.size": style.fs_tick,
            "axes.titlesize": style.fs_title,
            "axes.labelsize": style.fs_label,
            "xtick.labelsize": style.fs_tick,
            "ytick.labelsize": style.fs_tick,
            "legend.fontsize": style.fs_legend,
            "lines.linewidth": style.line_width,
            "lines.markersize": style.marker_size,
            "axes.grid": False,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.default": "regular",
        }
    )


def fmt_params(
    D: int | None = None, sigma_eta: float | None = None, pi_target: float | None = None
) -> str:
    parts: list[str] = []
    if D is not None:
        parts.append(rf"$D={int(D)}$")
    if sigma_eta is not None:
        parts.append(rf"$\sigma_\eta={float(sigma_eta):g}$")
    if pi_target is not None:
        parts.append(rf"$\pi_{{\mathrm{{target}}}}={float(pi_target):.2f}$")
    return r",\ ".join(parts)


def safe_suptitle(
    fig: plt.Figure, text: str, style: PlotStyle = DEFAULT_STYLE, y: float = 0.98
) -> None:
    fig.suptitle(str(text), fontsize=style.fs_suptitle, y=float(y))


def finalize_fig(
    fig: plt.Figure, outpath: str | Path, style: PlotStyle = DEFAULT_STYLE
) -> Path:
    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def should_plot(n: int, n_min: int = DEFAULT_STYLE.n_min) -> bool:
    return int(n) >= int(n_min)


def render_na_panel(
    ax: plt.Axes, n: int, reason: str = "n<N_MIN", style: PlotStyle = DEFAULT_STYLE
) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5,
        0.55,
        "NA",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=style.fs_title,
    )
    ax.text(
        0.5,
        0.38,
        f"(n={int(n)}; {reason})",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=style.fs_tick,
        color="#555555",
    )


def choose_text_color(rgba: Sequence[float]) -> str:
    r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.55 else "white"
