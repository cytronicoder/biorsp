import os

import matplotlib.pyplot as plt

# Standard colors
COLOR_BG = "#e0e0e0"  # Light gray for background
COLOR_FG = "#d62728"  # Red for foreground
COLOR_RSP = "#1f77b4"  # Blue for RSP profile


def setup_style():
    """Set up global matplotlib style for publication-ready figures."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "lines.linewidth": 2,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, output_dir: str, filename: str):
    """Save figure as both PNG and PDF."""
    os.makedirs(output_dir, exist_ok=True)

    # Ensure filename doesn't have extension
    base_name = os.path.splitext(filename)[0]

    png_path = os.path.join(output_dir, f"{base_name}.png")
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure to {png_path} and {pdf_path}")


def clean_axis(ax: plt.Axes):
    """Remove ticks and set axis equal."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
