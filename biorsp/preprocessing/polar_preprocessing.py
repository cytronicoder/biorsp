"""
Polar preprocessing for AnnData embeddings.
"""

from typing import Optional, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData

__all__ = ["polar_transform", "cartesian_to_polar"]


def polar_transform(
    adata: AnnData,
    use_rep: str = "X_umap",
    key_added: str = "X_polar",
    vantage_point: Optional[Tuple[float, float]] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Compute polar coordinates for a 2D embedding.

    Args:
        adata: Input AnnData object.
        use_rep: Key in `adata.obsm` to fetch the 2D embedding (e.g., "X_pca", "X_umap").
        key_added: Key under which to store the (r, θ) output in `adata.obsm`.
        vantage_point: Tuple (x, y) to use as the origin; if None, defaults to the embedding centroid.
        copy: If True, operate on a copy and return it; otherwise modify `adata` in place.

    Returns:
        If `copy=True`, returns a new AnnData with polar coords added; otherwise returns None.

    Examples:
        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc3k()
        >>> sc.pp.neighbors(adata)
        >>> sc.tl.umap(adata)
        >>> polar_adata = polar_transform(adata, use_rep="X_umap", copy=True)
        >>> polar_coords = polar_adata.obsm["X_polar"]
    """
    adata = adata.copy() if copy else adata

    X = adata.obsm.get(use_rep)
    if X is None:
        raise KeyError(
            f"'{use_rep}' not found in adata.obsm; available: {list(adata.obsm.keys())}"
        )
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(
            f"Expected a (n_obs, 2) array in obsm['{use_rep}'], got shape {X.shape}"
        )

    if vantage_point is None:
        vantage_point = tuple(np.mean(X, axis=0))
        sc.logging.debug(f"Computed centroid vantage_point = {vantage_point!r}")

    r, theta = cartesian_to_polar(X, vantage_point=vantage_point)
    adata.obsm[key_added] = np.column_stack((r, theta))
    sc.logging.info(f"Stored polar coords in obsm['{key_added}']")

    if copy:
        return adata


def cartesian_to_polar(
    X: np.ndarray,
    vantage_point: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to polar coordinates.

    Args:
        X: Array of shape (n_obs, 2) with (x, y) pairs.
        vantage_point: Tuple (x0, y0) serving as the origin for transformation.

    Returns:
        r: 1D array of radial distances from `vantage_point`.
        theta: 1D array of angles in radians relative to the x-axis.

    Examples:
        >>> import numpy as np
        >>> coords = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> r, theta = cartesian_to_polar(coords)
        >>> np.allclose(r, [1.0, 1.0])
        True
        >>> np.allclose(theta, [0.0, np.pi/2])
        True
    """
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("`X` must be a (n_obs, 2) array.")

    origin = np.asarray(vantage_point, dtype=float)
    diff = X - origin
    dx, dy = diff[:, 0], diff[:, 1]

    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)

    sc.logging.debug(
        f"Converted {X.shape[0]} points relative to vantage_point {vantage_point!r}"
    )
    return r, theta
