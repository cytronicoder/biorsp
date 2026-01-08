"""Embedding module for BioRSP.

Provides wrappers for common dimensionality reduction methods:
- PCA (via sklearn)
- UMAP (via umap-learn)
- t-SNE (via sklearn)
- PHATE (via phate)
"""

from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np


def _as_1d(a) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def compute_embedding(
    x: np.ndarray,
    method: Literal["pca", "umap", "tsne", "phate", "custom"] = "pca",
    n_components: int = 2,
    random_state: int = 42,
    z_custom: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Compute a low-dimensional embedding of high-dimensional data.

    Notes:
    - Supported methods: 'pca', 'umap', 'tsne', 'phate', 'custom'.
    - For 'custom', provide `z_custom` (shape must match rows of `x`).
    - t-SNE's `random_state` support depends on the scikit-learn version.

    Args:
        x: (N, D) input data matrix.
        method: Embedding method.
        n_components: Number of dimensions (default 2).
        random_state: Random seed.
        z_custom: Precomputed embedding (required if method='custom').
        **kwargs: Additional arguments passed to the embedding class.

    Returns:
        z: (N, n_components) embedding array.

    """
    if method == "custom":
        if z_custom is None:
            raise ValueError("z_custom must be provided when method='custom'")
        if z_custom.shape[0] != x.shape[0]:
            raise ValueError(f"z_custom shape {z_custom.shape} does not match x {x.shape}")
        return z_custom

    elif method == "pca":
        from sklearn.decomposition import PCA

        model = PCA(n_components=n_components, random_state=random_state, **kwargs)
        return model.fit_transform(x)

    elif method == "umap":
        import umap

        model = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
        return model.fit_transform(x)

    elif method == "tsne":
        from sklearn.manifold import TSNE

        model = TSNE(n_components=n_components, random_state=random_state, **kwargs)
        return model.fit_transform(x)

    elif method == "phate":
        import phate

        model = phate.PHATE(n_components=n_components, random_state=random_state, **kwargs)
        return model.fit_transform(x)

    else:
        raise ValueError(f"Unknown embedding method: {method}")


def plot_embedding(
    Z: np.ndarray,
    c: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    s: int = 4,
    fg_color: str = "red",
    bg_color: str = "grey",
    show_vantage: bool = False,
    vantage: Optional[np.ndarray] = None,
    show_legend: bool = True,
    legend_loc: str = "best",
    vantage_seed: Optional[int] = None,
    **kwargs,
) -> plt.Axes:
    """Scatter plot of embedding.

    Special behavior:
      - If `c` is a binary array (0/1 or booleans), the function will plot background
        points in `bg_color` and foreground points in `fg_color` to enforce a consistent
        visual convention across the codebase.
      - If `show_vantage` is True, a large, prominent "X" marker is plotted at the
        vantage point (geometric median by default) with a small "v" label.

    Args:
        Z: (N, 2) embedding coordinates.
        c: (N,) color values (e.g. expression) or binary foreground mask.
        ax: Axes to plot on (creates new if None).
        title: Optional plot title.
        cmap: Colormap for continuous values.
        s: Point size.
        fg_color: Color for binary foreground.
        bg_color: Color for binary background.
        show_vantage: Whether to marker the spatial vantage point.
        vantage: Option to provide precomputed vantage point.
        show_legend: Whether to show legend for binary mask.
        legend_loc: Legend position.
        vantage_seed: Random seed for vantage computation sampling.
        **kwargs: Passed to scatter plot.
        ax: Matplotlib axes. If None, created.
        title: Plot title.
        cmap: Colormap (used when not plotting binary mask).
        s: Base marker size for points (default 1 per request).
        fg_color: Color string for foreground points (default: 'red').
        bg_color: Color string for background points (default: 'grey').
        show_vantage: Whether to mark the vantage point on the plot.
        vantage: Optional (2,) vantage coordinates; if None and show_vantage True, computed.
        **kwargs: Additional kwargs forwarded to `scatter`.

    Returns:
        ax: The axes object.

    """
    Z = np.asarray(Z)
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(f"Z must have shape (N, 2); got {Z.shape}")

    if c is not None:
        c = _as_1d(c)
        if c.size != Z.shape[0]:
            raise ValueError(f"c must have length N={Z.shape[0]}; got {c.size}")

    if ax is None:
        _, ax = plt.subplots()

    plotted_sc = None
    if c is not None:
        uniq = np.unique(c)
        try:
            uniq_set = set(uniq.tolist())
        except Exception:
            uniq_set = set(uniq)
        if uniq_set <= {0, 1}:
            mask_fg = c.astype(bool)
            mask_bg = ~mask_fg
            if np.any(mask_bg):
                ax.scatter(
                    Z[mask_bg, 0],
                    Z[mask_bg, 1],
                    c=bg_color,
                    s=s,
                    alpha=0.8,
                    edgecolors="none",
                    label="Background",
                    **kwargs,
                )
            if np.any(mask_fg):
                ax.scatter(
                    Z[mask_fg, 0],
                    Z[mask_fg, 1],
                    c=fg_color,
                    s=s,
                    alpha=0.95,
                    edgecolors="none",
                    label="Foreground",
                    **kwargs,
                )
        else:
            plotted_sc = ax.scatter(
                Z[:, 0], Z[:, 1], c=c, s=s, cmap=cmap, edgecolors="none", **kwargs
            )
    else:
        plotted_sc = ax.scatter(Z[:, 0], Z[:, 1], c=None, s=s, edgecolors="none", **kwargs)

    if plotted_sc is not None and c is not None and not (set(np.unique(c).tolist()) <= {0, 1}):
        plt.colorbar(plotted_sc, ax=ax)

    if title:
        ax.set_title(title)

    if show_vantage:
        if vantage is None:
            from biorsp.preprocess.geometry import compute_vantage

            v = compute_vantage(Z, seed=vantage_seed)
        else:
            v = vantage
        ax.scatter(
            v[0], v[1], marker="X", s=300, color="black", zorder=20, linewidths=1.5, label="Vantage"
        )
        ax.text(v[0], v[1], "v", fontsize=10, fontweight="bold", va="bottom", ha="left", zorder=21)

    if show_legend and (c is not None or show_vantage):
        ax.legend(loc=legend_loc)

    return ax
