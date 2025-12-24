"""
Embedding module for BioRSP.

Provides wrappers for common dimensionality reduction methods:
- PCA (via sklearn)
- UMAP (via umap-learn)
- t-SNE (via sklearn)
- PHATE (via phate)
"""

from typing import Literal, Optional

import numpy as np


def compute_embedding(
    x: np.ndarray,
    method: Literal["pca", "umap", "tsne", "phate", "custom"] = "pca",
    n_components: int = 2,
    random_state: int = 42,
    z_custom: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    Compute a low-dimensional embedding of high-dimensional data.

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
            raise ValueError(
                f"z_custom shape {z_custom.shape} does not match x {x.shape}"
            )
        return z_custom

    elif method == "pca":
        # sklearn is a runtime dependency; import locally to avoid import-time errors in
        # environments where scikit-learn is not installed (e.g., lightweight linter runs).
        # pylint: disable=import-outside-toplevel
        from sklearn.decomposition import PCA

        model = PCA(n_components=n_components, random_state=random_state, **kwargs)
        return model.fit_transform(x)

    elif method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "umap-learn is required for 'umap' embed. Install with: pip install umap-learn"
            ) from exc

        model = umap.UMAP(
            n_components=n_components, random_state=random_state, **kwargs
        )
        return model.fit_transform(x)

    elif method == "tsne":
        # TSNE may be provided by scikit-learn
        # pylint: disable=import-outside-toplevel
        from sklearn.manifold import TSNE

        model = TSNE(n_components=n_components, random_state=random_state, **kwargs)
        return model.fit_transform(x)

    elif method == "phate":
        try:
            import phate
        except ImportError as exc:
            raise ImportError(
                "phate is required for 'phate' embed. Install with: pip install phate"
            ) from exc

        model = phate.PHATE(
            n_components=n_components, random_state=random_state, **kwargs
        )
        return model.fit_transform(x)

    else:
        raise ValueError(f"Unknown embedding method: {method}")


__all__ = ["compute_embedding"]
