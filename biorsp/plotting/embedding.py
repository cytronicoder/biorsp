"""Embedding module for BioRSP.

Provides wrappers for common dimensionality reduction methods:
- PCA (via sklearn)
- UMAP (via umap-learn)
- t-SNE (via sklearn)
- PHATE (via phate)

Also provides scalable plotting utilities with automatic downsampling and rasterization.
"""

from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np


def _as_1d(a) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def _compute_point_size(n_points: int, base_size: float = 10.0) -> float:
    """Compute adaptive point size based on number of points.

    Heuristic: Reduce point size for dense plots to avoid overplotting.
    """
    if n_points < 1000:
        return base_size
    elif n_points < 10000:
        return base_size * 0.7
    elif n_points < 50000:
        return base_size * 0.5
    else:
        return base_size * 0.3


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
    s: Optional[int] = None,
    fg_color: str = "red",
    bg_color: str = "grey",
    show_vantage: bool = False,
    vantage: Optional[np.ndarray] = None,
    vantage_method: str = "geometric_median",
    show_legend: bool = True,
    legend_loc: str = "best",
    subsample_seed: Optional[int] = None,
    max_points: Optional[int] = None,
    rasterized: bool = False,
    config: Optional[object] = None,
    **kwargs,
) -> plt.Axes:
    """Scatter plot of embedding with scalability and API correctness.

    Special behavior:
      - If `c` is a binary array (0/1 or booleans), plots background in `bg_color`
        and foreground in `fg_color` for consistent visual convention.
      - If `show_vantage` is True, a large "X" marker is plotted at the vantage point.
      - Supports deterministic subsampling for large datasets.
      - Auto-adjusts point size based on data size.

    Args:
        Z: (N, 2) embedding coordinates.
        c: (N,) color values (expression) or binary foreground mask. Optional.
        ax: Axes to plot on (creates new if None).
        title: Optional plot title.
        cmap: Colormap for continuous values (default: "viridis").
        s: Point size. If None, computed automatically from data size.
        fg_color: Color for binary foreground (default: "red").
        bg_color: Color for binary background (default: "grey").
        show_vantage: Whether to mark the spatial vantage point (default: False).
        vantage: Precomputed (2,) vantage coords. If None and show_vantage=True, computed.
        vantage_method: Method for compute_vantage (default: "geometric_median").
        show_legend: Whether to show legend for binary mask and vantage (default: True).
        legend_loc: Legend position (default: "best").
        subsample_seed: Random seed for deterministic subsampling. If None, no subsampling.
        max_points: Maximum points to plot. If Z exceeds this, randomly subsample.
        rasterized: Whether to rasterize scatter points (useful for large plots in vector formats).
        config: Optional BioRSPConfig for vantage computation parameters.
        **kwargs: Additional kwargs forwarded to `scatter` (avoid 'c', 'label', 'cmap').

    Returns:
        ax: The axes object.

    Example:
        >>> plot_embedding(Z, c=expr, max_points=5000, subsample_seed=42, rasterized=True)
    """
    Z = np.asarray(Z)
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(f"Z must have shape (N, 2); got {Z.shape}")

    N = Z.shape[0]
    idx_plot = np.arange(N)
    if max_points is not None and max_points < N:
        rng = np.random.default_rng(subsample_seed)
        idx_plot = rng.choice(N, size=max_points, replace=False)
        idx_plot = np.sort(idx_plot)
        Z = Z[idx_plot]
        if c is not None:
            c = _as_1d(c)[idx_plot]

    if c is not None:
        c = _as_1d(c)
        if c.size != Z.shape[0]:
            raise ValueError(f"c must have length {Z.shape[0]} after subsampling; got {c.size}")

    if s is None:
        s = _compute_point_size(Z.shape[0])

    if ax is None:
        _, ax = plt.subplots()

    kwargs_clean = {k: v for k, v in kwargs.items() if k not in ["c", "label", "cmap"]}

    plotted_sc = None
    if c is not None:
        uniq = np.unique(c[np.isfinite(c)])
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
                    rasterized=rasterized,
                    **kwargs_clean,
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
                    rasterized=rasterized,
                    **kwargs_clean,
                )
        else:
            plotted_sc = ax.scatter(
                Z[:, 0],
                Z[:, 1],
                c=c,
                s=s,
                cmap=cmap,
                edgecolors="none",
                rasterized=rasterized,
                **kwargs_clean,
            )
    else:
        plotted_sc = ax.scatter(
            Z[:, 0],
            Z[:, 1],
            c=None,
            s=s,
            edgecolors="none",
            rasterized=rasterized,
            **kwargs_clean,
        )

    if plotted_sc is not None and c is not None and not (set(np.unique(c).tolist()) <= {0, 1}):
        plt.colorbar(plotted_sc, ax=ax)

    if title:
        ax.set_title(title)

    if show_vantage:
        if vantage is None:
            from biorsp.core.geometry import compute_vantage

            vantage_kwargs = {"method": vantage_method}
            if config is not None:
                vantage_kwargs.update(
                    {
                        "knn_k": config.center_knn_k,
                        "density_percentile": config.center_density_percentile,
                        "tol": config.geom_median_tol,
                        "max_iter": config.geom_median_max_iter,
                        "seed": config.seed,
                    }
                )
            elif subsample_seed is not None:
                vantage_kwargs["seed"] = subsample_seed

            v = compute_vantage(Z, **vantage_kwargs)
        else:
            v = vantage

        ax.scatter(
            v[0], v[1], marker="X", s=300, color="black", zorder=20, linewidths=1.5, label="Vantage"
        )
        ax.text(v[0], v[1], "v", fontsize=10, fontweight="bold", va="bottom", ha="left", zorder=21)

    if show_legend and ((c is not None and set(np.unique(c).tolist()) <= {0, 1}) or show_vantage):
        ax.legend(loc=legend_loc, fontsize=8, framealpha=0.9)

    ax.set_aspect("equal")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    return ax
