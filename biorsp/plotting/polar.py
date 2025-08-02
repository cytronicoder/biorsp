import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from anndata import AnnData
import numpy as np

__all__ = ["plot_polar_transform"]


def plot_polar_transform(
    adata: AnnData,
    polar_coords_key: str = "X_polar",
    color_by: str = None,
    cmap: str = None,
    ax=None,
    **scatter_kwargs,
):
    """
    Plot the polar coordinates of each cell in the embedding on a polar axis, optionally colored by an obs column.

    Args:
        adata: Annotated data matrix with polar coords in `adata.obsm[polar_coords_key]`.
        polar_coords_key: Key in `adata.obsm` where (r, theta) coords are stored.
        color_by: Name of an `adata.obs` column to use for coloring. If None, no coloring applied.
        cmap: Matplotlib colormap name. If None, defaults to 'tab10' for categorical or 'viridis' for continuous.
        ax: A matplotlib Axes (polar projection). If None, a new one is created.
        **scatter_kwargs: Other keyword args passed to `ax.scatter` (e.g., s, marker).

    Returns:
        A matplotlib Axes instance with the polar scatter plot.
    """
    polar_coords = adata.obsm.get(polar_coords_key)
    if polar_coords is None:
        raise KeyError(
            f"'{polar_coords_key}' not found in adata.obsm; available: {list(adata.obsm.keys())}"
        )
    if polar_coords.ndim != 2 or polar_coords.shape[1] != 2:
        raise ValueError(
            f"Expected a (n_obs, 2) array in obsm['{polar_coords_key}'], got shape {polar_coords.shape}"
        )

    r, theta = polar_coords[:, 0], polar_coords[:, 1]
    plot_kwargs = scatter_kwargs.copy()
    handles = None

    if color_by:
        if color_by not in adata.obs:
            raise KeyError(
                f"'{color_by}' not found in adata.obs; available columns: {list(adata.obs.columns)}"
            )
        values = adata.obs[color_by]
        is_categorical = values.dtype.name == "category" or values.dtype == object
        if is_categorical:
            labels = values.astype("category")
            codes = labels.cat.codes.values
            n = len(labels.cat.categories)

            # Create a discrete n-color colormap
            cmap_name = cmap or "tab10"
            cmap_obj = plt.get_cmap(cmap_name, n)
            norm = BoundaryNorm(boundaries=np.arange(n + 1) - 0.5, ncolors=n)

            plot_kwargs["c"] = codes
            plot_kwargs["cmap"] = cmap_obj
            plot_kwargs["norm"] = norm

            # Legend handles that match the scatter colors exactly
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=cmap_obj(i),
                    label=lab,
                )
                for i, lab in enumerate(labels.cat.categories)
            ]
        else:
            arr = values.values.astype(float)
            cmap_name = cmap or "viridis"
            plot_kwargs["c"] = arr
            plot_kwargs["cmap"] = cmap_name

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    ax.scatter(theta, r, **plot_kwargs)
    ax.set_title("Polar coordinate projection")
    ax.set_rlabel_position(0)
    ax.set_xlabel("Angle (radians)")
    ax.set_ylabel("Radius")
    if handles is not None:
        ax.legend(handles=handles, title=color_by, bbox_to_anchor=(1.1, 1.05))
    return ax
