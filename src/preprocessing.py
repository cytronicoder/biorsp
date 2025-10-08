from typing import Tuple, Literal, Optional, Sequence, Union, TYPE_CHECKING
import warnings
import numpy as np

from .utils import geometric_median as _geometric_median_fallback


if TYPE_CHECKING:
    from scipy import sparse


def validate_inputs(
    coords: np.ndarray,
    feature: Optional[np.ndarray] = None,
    *,
    covariates: Optional[np.ndarray] = None,
    bands: Optional[np.ndarray] = None,
    B: Optional[int] = None,
) -> None:
    """
    Sanity checks for shapes, finiteness, and parameter consistency.

    Args:
        coords (np.ndarray): 2D embedding coordinates, should be shape (N, 2).
        feature (Optional[np.ndarray], optional): Feature values, should be length N.
        covariates (Optional[np.ndarray], optional): Covariate matrix, should be shape (N, P).
        bands (Optional[np.ndarray], optional): Radial band edges, should be shape (M, 2)
            with 0 <= r_in < r_out, strictly increasing, no overlaps.
        B (Optional[int], optional): Number of angular bins, should be >= 4.

    Raises:
        ValueError: If any fatal inconsistency is detected with a clear message.

    Warnings:
        UserWarning: If N < 100 or if many radii are identical (possible duplicates).

    Notes:
        - Checks coords shape (N, 2), finite.
        - Checks feature length N if provided; finite.
        - Checks covariates shape (N, P) if provided.
        - Checks bands (M, 2) with 0 <= r_in < r_out, strictly increasing; no overlaps.
        - Checks B >= 4 (at least 90° resolution).
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"`coords` must be an (N, 2) array, got shape {coords.shape}.")

    n_points = coords.shape[0]

    if not np.all(np.isfinite(coords)):
        raise ValueError("`coords` contains non-finite values (NaN or Inf).")

    # Warn if too few points
    if n_points < 100:
        warnings.warn(
            f"Only {n_points} points provided. Results may be unreliable with N < 100.",
            UserWarning,
            stacklevel=2,
        )

    if n_points > 1:
        centroid = np.mean(coords, axis=0)
        radii = np.linalg.norm(coords - centroid, axis=1)
        unique_radii = np.unique(radii)

        if len(unique_radii) < n_points * 0.5:
            warnings.warn(
                f"Many points have identical radii ({len(unique_radii)} unique out of "
                f"{n_points}). This may indicate duplicate coordinates.",
                UserWarning,
                stacklevel=2,
            )

    if feature is not None:
        feature = np.asarray(feature)
        if feature.ndim != 1:
            raise ValueError(
                f"`feature` must be a 1D array, got shape {feature.shape}."
            )
        if len(feature) != n_points:
            raise ValueError(
                f"`feature` length ({len(feature)}) must match coords ({n_points})."
            )
        if not np.all(np.isfinite(feature)):
            raise ValueError("`feature` contains non-finite values (NaN or Inf).")

    if covariates is not None:
        covariates = np.asarray(covariates)
        if covariates.ndim == 1:
            if len(covariates) != n_points:
                raise ValueError(
                    f"`covariates` length ({len(covariates)}) must match coords ({n_points})."
                )
        elif covariates.ndim == 2:
            if covariates.shape[0] != n_points:
                raise ValueError(
                    f"`covariates` first dimension ({covariates.shape[0]}) must match "
                    f"coords ({n_points})."
                )
        else:
            raise ValueError(
                f"`covariates` must be 1D or 2D array, got shape {covariates.shape}."
            )
        if not np.all(np.isfinite(covariates)):
            raise ValueError("`covariates` contains non-finite values (NaN or Inf).")

    if bands is not None:
        bands = np.asarray(bands)
        if bands.ndim != 2 or bands.shape[1] != 2:
            raise ValueError(
                f"`bands` must be an (M, 2) array, got shape {bands.shape}."
            )

        if not np.all(np.isfinite(bands)):
            raise ValueError("`bands` contains non-finite values (NaN or Inf).")

        for i, (r_in, r_out) in enumerate(bands):
            if r_in < 0:
                raise ValueError(
                    f"Band {i}: inner radius ({r_in}) must be non-negative."
                )
            if r_in >= r_out:
                raise ValueError(
                    f"Band {i}: inner radius ({r_in}) must be < outer radius ({r_out})."
                )

        if len(bands) > 1:
            for i in range(len(bands) - 1):
                r_in_curr, r_out_curr = bands[i]
                r_in_next, r_out_next = bands[i + 1]

                if r_in_next < r_out_curr:
                    raise ValueError(
                        f"Bands {i} and {i+1} overlap: [{r_in_curr}, {r_out_curr}] "
                        f"and [{r_in_next}, {r_out_next}]."
                    )

                if r_in_next < r_in_curr:
                    raise ValueError(
                        f"Bands are not ordered: band {i} starts at {r_in_curr}, "
                        f"but band {i+1} starts at {r_in_next}."
                    )

    if B is not None:
        if not isinstance(B, (int, np.integer)):
            raise ValueError(f"`B` must be an integer, got {type(B).__name__}.")
        if B < 4:
            raise ValueError(f"`B` must be >= 4 (at least 90° resolution), got {B}.")


def to_polar(
    coords: np.ndarray, center: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 2D Cartesian coords to polar angles and radii around `center`.

    Args:
        coords (np.ndarray): An array of shape (N, 2) of float coordinates.
        center (Tuple[float, float]): A tuple (x0, y0) for the center point.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - theta (np.ndarray): Angle in radians in [0, 2π).
            - r (np.ndarray): Euclidean distance from `center`.

    Notes:
        - Points exactly at the center get r=0; angle is defined as 0.0.
    """
    coords = np.asarray(coords)
    center = np.asarray(center)

    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    diff = coords - center
    r = np.linalg.norm(diff, axis=1)

    theta = np.arctan2(diff[:, 1], diff[:, 0])
    theta = np.mod(theta, 2 * np.pi)
    return (theta, r)


def bin_angles(theta: np.ndarray, n_bins: int, *, offset: float = 0.0) -> np.ndarray:
    """
    Discretize angles into B cyclic wedges (0..B-1).

    Args:
        theta (np.ndarray): An array of angles in radians, assumed to be in [0, 2π).
        n_bins (int): The number of wedges (resolution).
        offset (float, optional): Rotational offset in radians applied before
                                  binning (e.g., to align 0°). Defaults to 0.0.

    Returns:
        np.ndarray: An array of integer bin indices in [0, B-1].
    """
    theta = np.asarray(theta)
    adjusted_theta = (theta + offset) % (2 * np.pi)
    wedge_idx = np.floor(adjusted_theta * n_bins / (2 * np.pi)).astype(int)
    return wedge_idx


def geometric_median(
    coords: np.ndarray,
    weights: Optional[np.ndarray] = None,
    backend: Literal["auto", "geom_median", "skfda", "internal"] = "auto",
    **kwargs,
) -> np.ndarray:
    """
    Unified entry point for geometric median calculation.

    This function attempts to use one of three backends in a specific order,
    falling back to the next if a preferred backend is not installed or fails.

    Args:
        coords (np.ndarray): The coordinates for which to find the median.
        weights (Optional[np.ndarray], optional): Weights for each coordinate.
            Defaults to equal weights.
        backend (Literal["auto", "geom_median", "skfda", "internal"], optional):
            The preferred backend. Defaults to "auto".
            - "auto": Try `geom_median`, then `skfda`, then `internal`.
            - "geom_median": Prefer `geom_median`, but fall back.
            - "skfda": Prefer `skfda`, but fall back.
            - "internal": Use the internal fallback directly.
        **kwargs: Additional keyword arguments passed to the backend.

    Returns:
        np.ndarray: The geometric median of the coordinates.
    """
    if weights is None:
        weights = np.ones(len(coords))

    def _try_geom_median():
        try:
            # pylint: disable=import-outside-toplevel
            from geom_median.numpy import geometric_median as gm_np

            return gm_np(coords, weights=weights, **kwargs)
        except (ImportError, TypeError):
            return None

    def _try_skfda():
        try:
            # pylint: disable=import-outside-toplevel
            from skfda.exploratory.stats import geometric_median as gm_skfda

            # pylint: disable=import-outside-toplevel
            from skfda.representation import FDataGrid

            # sk-fda expects data to be in a FDataGrid object
            # for multivariate data, it should be shaped as
            # (n_samples, n_features, 1)
            coords_reshaped = coords[:, :, np.newaxis]

            coords_fd = FDataGrid(
                data_matrix=coords_reshaped,
                grid_points=[0],
            )

            median_fd = gm_skfda(coords_fd)
            return median_fd.data_matrix.squeeze()
        except (ImportError, TypeError, ValueError):
            return None

    def _internal_fallback():
        return _geometric_median_fallback(
            coords,
            weights=weights,
            max_iter=kwargs.get("max_iter", 256),
            tol=kwargs.get("tol", 1e-7),
        )

    if backend in ("auto", "geom_median"):
        result = _try_geom_median()
        if result is not None:
            return result

    if backend in ("auto", "geom_median", "skfda"):
        result = _try_skfda()
        if result is not None:
            return result

    return _internal_fallback()


def choose_center(
    coords: np.ndarray,
    mode: Literal["geom_median", "cluster", "user"] = "geom_median",
    *,
    cluster_ids: Optional[np.ndarray] = None,
    cluster_label: Optional[Union[int, str]] = None,
    user_center: Optional[Tuple[float, float]] = None,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 256,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Return the origin (x0, y0) used for polar projection.

    Args:
        coords (np.ndarray): 2D embedding coordinates, shape (N, 2).
        mode (Literal["geom_median", "cluster", "user"], optional):
            - "geom_median": robust geometric median via Weiszfeld (default).
            - "cluster": centroid of points with specified cluster_label.
            - "user": use `user_center` exactly.
        cluster_ids (Optional[np.ndarray], optional): Cluster assignments per cell,
            shape (N,). Required for "cluster" mode.
        cluster_label (Optional[Union[int, str]], optional): Specific cluster to center on.
            Required for "cluster" mode.
        user_center (Optional[Tuple[float, float]], optional): Required for "user" mode.
        weights (Optional[np.ndarray], optional): Nonnegative weights for median/centroid.
        max_iter (int, optional): Max iterations for geometric median. Defaults to 256.
        tol (float, optional): Convergence tolerance for geometric median. Defaults to 1e-7.

    Returns:
        np.ndarray: The chosen origin (x, y).

    Raises:
        ValueError: If inputs are inconsistent (e.g., missing user_center for "user" mode,
            or no points found for the specified cluster_label).
    """
    if mode == "user":
        if user_center is None:
            raise ValueError("`user_center` must be provided for mode='user'.")
        return np.asarray(user_center, dtype=float)

    if weights is None:
        weights = np.ones(len(coords))
    else:
        weights = np.asarray(weights)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    if mode == "cluster":
        if cluster_ids is None:
            raise ValueError("`cluster_ids` must be provided for mode='cluster'.")
        if cluster_label is None:
            raise ValueError("`cluster_label` must be provided for mode='cluster'.")

        cluster_ids = np.asarray(cluster_ids)
        if cluster_ids.shape[0] != coords.shape[0]:
            raise ValueError(
                f"`cluster_ids` length ({cluster_ids.shape[0]}) must match "
                f"`coords` ({coords.shape[0]})."
            )

        mask = cluster_ids == cluster_label
        if not np.any(mask):
            raise ValueError(f"No points found for cluster_label={cluster_label!r}.")

        w = None if weights is None else weights[mask]
        if w is not None and np.any(w < 0):
            raise ValueError("Weights must be non-negative.")

        return np.average(coords[mask], axis=0, weights=w)

    if mode == "geom_median":
        return geometric_median(coords, weights=weights, max_iter=max_iter, tol=tol)

    raise ValueError(
        f"Unknown mode: '{mode}'. Must be 'geom_median', 'cluster', or 'user'."
    )


def standardize_feature(
    f: np.ndarray,
    method: Literal["z", "rank", "mad", "none"] = "z",
    *,
    clip: Optional[Tuple[float, float]] = None,
    ddof: int = 1,
) -> np.ndarray:
    """
    Map raw feature values to a stable scale for weighting.

    Args:
        f (np.ndarray): Continuous feature values (e.g., gene scores).
        method (Literal["z", "rank", "mad", "none"], optional):
            - "z": (f - mean) / std (with ddof).
            - "rank": Rank-normalize to (0, 1]; robust to outliers.
            - "mad": (f - median) / (1.4826 * MAD); robust to outliers.
            - "none": Return f unchanged.
        clip (Optional[Tuple[float, float]], optional): Clip standardized values.
        ddof (int, optional): Degrees of freedom for std. Defaults to 1.

    Returns:
        np.ndarray: Standardized feature values.

    Notes:
        - Use "rank" or "mad" for skewed or zero-inflated distributions.
    """
    f = np.asarray(f)
    f_std = f.copy()

    if method == "none":
        pass
    elif method == "z":
        mean = np.mean(f)
        std = np.std(f, ddof=ddof)
        if std > 0:
            f_std = (f - mean) / std
    elif method == "rank":
        from scipy.stats import rankdata

        f_std = rankdata(f, method="average") / len(f)
    elif method == "mad":
        from scipy.stats import median_abs_deviation

        median = np.median(f)
        mad = median_abs_deviation(f, scale="normal")
        if mad > 0:
            f_std = (f - median) / mad
    else:
        raise ValueError(f"Unknown standardization method: '{method}'")

    if clip is not None:
        f_std = np.clip(f_std, clip[0], clip[1])

    return f_std


def binarize_feature(
    f: np.ndarray,
    mode: Literal["none", "positive", "percentile", "value"] = "positive",
    *,
    threshold_value: Optional[float] = None,
) -> np.ndarray:
    """
    Convert continuous feature values to binary foreground/background labels.

    This function is essential for gene expression analysis where you want to
    compare the spatial distribution of cells expressing a gene (foreground)
    vs cells not expressing it (background).

    Args:
        f (np.ndarray): Continuous feature values (e.g., gene expression).
        mode (Literal["none", "positive", "percentile", "value"], optional):
            - "none": Return f unchanged (no binarization).
            - "positive": Foreground = f > 0; useful for sparse gene expression.
            - "percentile": Foreground = f > percentile(f, threshold_value).
            - "value": Foreground = f > threshold_value.
            Defaults to "positive".
        threshold_value (Optional[float], optional): 
            - For "percentile": percentile value (0-100).
            - For "value": explicit threshold value.
            - Ignored for "none" and "positive".

    Returns:
        np.ndarray: Binary feature (1.0 = foreground, 0.0 = background).

    Raises:
        ValueError: If mode is invalid or threshold_value is missing when required.

    Examples:
        >>> expr = np.array([0, 0, 0.5, 1.2, 2.3, 0, 0.1])
        >>> binarize_feature(expr, mode="positive")
        array([0., 0., 1., 1., 1., 0., 1.])
        >>> binarize_feature(expr, mode="percentile", threshold_value=50)
        array([0., 0., 0., 1., 1., 0., 0.])
        >>> binarize_feature(expr, mode="value", threshold_value=1.0)
        array([0., 0., 0., 1., 1., 0., 0.])

    Notes:
        - Binarization enables proper foreground vs background comparison.
        - For sparse RNA-seq data, "positive" mode is often most appropriate.
        - For continuous features, "percentile" or "value" modes may be better.
    """
    f = np.asarray(f)
    
    if mode == "none":
        return f
    elif mode == "positive":
        return (f > 0).astype(float)
    elif mode == "percentile":
        if threshold_value is None:
            raise ValueError("threshold_value must be provided for mode='percentile'")
        if not 0 <= threshold_value <= 100:
            raise ValueError(f"Percentile must be in [0, 100], got {threshold_value}")
        threshold = np.percentile(f, threshold_value)
        return (f > threshold).astype(float)
    elif mode == "value":
        if threshold_value is None:
            raise ValueError("threshold_value must be provided for mode='value'")
        return (f > threshold_value).astype(float)
    else:
        raise ValueError(
            f"Unknown binarization mode: '{mode}'. "
            "Must be 'none', 'positive', 'percentile', or 'value'."
        )


def residualize_feature(
    f: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    *,
    add_intercept: bool = True,
    method: Literal["ols", "ridge"] = "ols",
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Remove linear effects of covariates (e.g., library size, mito%).

    Args:
        f (np.ndarray): Response variable, shape (N,).
        covariates (Optional[np.ndarray], optional): Design matrix, shape (N, P).
            Categorical batches should be one-hot encoded.
        add_intercept (bool, optional): If True, prepend a column of ones. Defaults to True.
        method (Literal["ols", "ridge"], optional): Solver for regression. Defaults to "ols".
        alpha (float, optional): Ridge penalty (ignored for OLS). Defaults to 1.0.

    Returns:
        np.ndarray: Residuals of the regression, shape (N,).

    Notes:
        - If covariates is None, returns f unchanged.
        - For nonlinear trends, handle upstream by adding transformed features
          to the covariates matrix.
    """
    f = np.asarray(f)
    if covariates is None:
        return f

    # pylint: disable=invalid-name
    X = np.asarray(covariates)
    if X.ndim == 1:
        X = X[:, np.newaxis]

    if add_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])

    if method == "ols":
        coeffs, _, _, _ = np.linalg.lstsq(X, f, rcond=None)
    elif method == "ridge":
        # (X.T @ X + alpha * I) @ beta = X.T @ y
        XtX = X.T @ X
        I = np.eye(XtX.shape[0])
        A = XtX + alpha * I
        b = X.T @ f
        coeffs = np.linalg.solve(A, b)
    else:
        raise ValueError(f"Unknown method: '{method}'")

    f_pred = X @ coeffs
    residuals = f - f_pred

    return residuals


def assign_radial_bands(
    r: np.ndarray,
    bands: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    mode: Literal["all", "quantile", "width"] = "all",
    n_bands: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partition radii into annuli (radial bands).

    Args:
        r (np.ndarray): Radii from center, shape (N,).
        bands (Optional[Sequence[Tuple[float, float]]], optional): Explicit annuli edges.
            Must be non-overlapping and increasing.
        mode (Literal["all", "quantile", "width"], optional):
            - "all": Single band spanning [min(r), max(r)).
            - "quantile": n_bands with equal counts.
            - "width": n_bands with equal radial width.
        n_bands (int, optional): Number of bands for automatic modes. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - band_idx (np.ndarray): Band assignment per point, shape (N,).
            - edges (np.ndarray): The actual [(r_in, r_out), ...] edges used, shape (M, 2).

    Raises:
        ValueError: On invalid edges, negative radii, or empty bands.
    """
    r = np.asarray(r)
    if np.any(r < 0):
        raise ValueError("Radii must be non-negative.")
    if r.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float).reshape(0, 2)

    min_r, max_r = r.min(), r.max()

    if bands is not None:
        edges = np.asarray(bands)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("`bands` must be a sequence of (r_in, r_out) tuples.")
    else:
        if min_r == max_r:
            edges = np.array([[min_r, max_r]])
        elif mode == "all":
            edges = np.array([[min_r, max_r]])
        elif mode == "quantile":
            quantiles = np.linspace(0, 100, n_bands + 1)
            edge_points = np.percentile(r, quantiles)
            edges = np.column_stack([edge_points[:-1], edge_points[1:]])
        elif mode == "width":
            edge_points = np.linspace(min_r, max_r, n_bands + 1)
            edges = np.column_stack([edge_points[:-1], edge_points[1:]])
        else:
            raise ValueError(f"Unknown mode: '{mode}'")

    if edges.size == 0:
        raise ValueError("Could not determine any radial bands.")

    band_idx = np.full(r.shape, -1, dtype=int)
    for i, (r_in, r_out) in enumerate(edges):
        if i == len(edges) - 1:
            mask = (r >= r_in) & (r <= r_out)
        else:
            mask = (r >= r_in) & (r < r_out)
        band_idx[mask] = i

    if np.any(band_idx == -1):
        unassigned_r = r[band_idx == -1]
        if bands is not None:
            raise ValueError(
                f"{unassigned_r.size} radii fall outside explicit bands. "
                f"First few examples: {unassigned_r[:5]}. "
                f"Band range: [{edges[0, 0]}, {edges[-1, 1]}]."
            )
        else:
            raise ValueError(
                f"Some radii were not assigned to a band. Unassigned values: {unassigned_r}"
            )

    return band_idx, edges


def density_weights(
    coords_2d: np.ndarray,
    *,
    k: int = 30,
    eps: float = 1e-12,
    return_density: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate local 2D crowding and return inverse-density weights.

    Args:
        coords_2d (np.ndarray): 2D embedding coordinates, shape (N, 2).
        k (int, optional): k-th neighbor distance used as local scale. Defaults to 30.
        eps (float, optional): Numerical floor to avoid division by zero. Defaults to 1e-12.
        return_density (bool, optional): If True, also return the raw density proxy.
            Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: If return_density is False,
            returns only the weights array. If True, returns a tuple containing:
            - weights (np.ndarray): Inverse-density weights, shape (N,).
              Higher values in sparse regions; normalized to mean 1.
            - d_k (np.ndarray): Distance to k-th neighbor (density proxy), shape (N,).

    Notes:
        - Uses cKDTree to compute d_k efficiently (O(N log N)).
        - Density proxy: density ≈ 1 / (d_k^2).
        - Inverse-density weight: w ≈ d_k^2 (larger in sparse regions).
        - Weights are normalized to mean 1 for stability in downstream calculations.
    """
    coords_2d = np.asarray(coords_2d)
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError("coords_2d must be an (N, 2) array.")

    n_points = coords_2d.shape[0]
    k_actual = min(k, n_points - 1)

    if k_actual < 1:
        d_k = np.ones(n_points)
    else:
        from scipy.spatial import cKDTree

        tree = cKDTree(coords_2d)
        distances, _ = tree.query(coords_2d, k=k_actual + 1)
        d_k = distances[:, k_actual]

    inv_density_w = d_k**2
    inv_density_w = inv_density_w / (inv_density_w.mean() + eps)

    if return_density:
        density = 1.0 / (d_k**2 + eps)
        return inv_density_w, d_k
    return inv_density_w


def density_ratio(
    highd: Optional[np.ndarray] = None,
    coords_2d: Optional[np.ndarray] = None,
    *,
    knn_graph_hd: Optional["sparse.csr_matrix"] = None,
    k_hd: int = 30,
    k_2d: int = 30,
    eps: float = 1e-12,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute per-cell density correction ratio: rho_HD / rho_2D.

    Args:
        highd (Optional[np.ndarray], optional): High-dimensional features, shape (N, D)
            (e.g., PCA scores). Used if knn_graph_hd is None.
        coords_2d (Optional[np.ndarray], optional): 2D embedding coordinates, shape (N, 2).
            Required to estimate rho_2D.
        knn_graph_hd (Optional[sparse.csr_matrix], optional): Precomputed kNN graph
            in high-D (binary or weighted). If provided, infer a density proxy from
            degrees/edge weights.
        k_hd (int, optional): k used for high-D density estimation. Defaults to 30.
        k_2d (int, optional): k used for 2D density estimation. Defaults to 30.
        eps (float, optional): Numerical floor to avoid division by zero. Defaults to 1e-12.
        normalize (bool, optional): If True, scale ratios to have median 1.0. Defaults to True.

    Returns:
        np.ndarray: Multiplicative factor to reweight per-cell contributions before scanning,
            shape (N,). ratio > 1 upweights cells underrepresented in 2D relative to high-D.

    Raises:
        ValueError: If neither (highd or knn_graph_hd) nor coords_2d is supplied.

    Notes:
        - rho_HD: 1 / (d_k(highd)^2) using kNN radius OR degree from knn_graph_hd.
        - rho_2D: 1 / (d_k(2D)^2) using kNN radius in 2D.
        - ratio > 1 upweights cells underrepresented in 2D relative to high-D.
    """
    if highd is None and knn_graph_hd is None:
        raise ValueError(
            "Must provide either `highd` or `knn_graph_hd` for high-D density estimation."
        )
    if coords_2d is None:
        raise ValueError("`coords_2d` is required for 2D density estimation.")

    coords_2d = np.asarray(coords_2d)
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError("`coords_2d` must be an (N, 2) array.")

    n_points = coords_2d.shape[0]

    if knn_graph_hd is not None:
        from scipy import sparse

        if not sparse.issparse(knn_graph_hd):
            raise ValueError("`knn_graph_hd` must be a scipy sparse matrix.")

        if not isinstance(knn_graph_hd, sparse.csr_matrix):
            knn_graph_hd = knn_graph_hd.tocsr()

        degrees = np.asarray(knn_graph_hd.sum(axis=1)).flatten()
        # use degree as a proxy: higher degree = higher density
        rho_hd = (degrees + eps) / (k_hd + eps)
    else:
        highd = np.asarray(highd)
        if highd.shape[0] != n_points:
            raise ValueError(
                f"`highd` has {highd.shape[0]} points but `coords_2d` has {n_points}."
            )

        k_hd_actual = min(k_hd, n_points - 1)
        if k_hd_actual < 1:
            d_k_hd = np.ones(n_points)
        else:
            from scipy.spatial import cKDTree

            tree_hd = cKDTree(highd)
            distances_hd, _ = tree_hd.query(highd, k=k_hd_actual + 1)
            d_k_hd = distances_hd[:, k_hd_actual]

        rho_hd = 1.0 / (d_k_hd**2 + eps)

    k_2d_actual = min(k_2d, n_points - 1)
    if k_2d_actual < 1:
        d_k_2d = np.ones(n_points)
    else:
        from scipy.spatial import cKDTree

        tree_2d = cKDTree(coords_2d)
        distances_2d, _ = tree_2d.query(coords_2d, k=k_2d_actual + 1)
        d_k_2d = distances_2d[:, k_2d_actual]

    rho_2d = 1.0 / (d_k_2d**2 + eps)
    ratio = rho_hd / (rho_2d + eps)

    if normalize:
        median_ratio = np.median(ratio)
        if median_ratio > eps:
            ratio = ratio / median_ratio

    return ratio
