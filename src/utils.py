from typing import Literal, Optional, Tuple, Union
import hashlib
import json
import numpy as np
import warnings


def config_hash(cfg: dict, *, algo: str = "sha1") -> str:
    """
    Stable short hash for parameter dicts (for caching/reproducibility).

    Args:
        cfg (dict): JSON-serializable configuration dictionary.
        algo (str, optional): Hash algorithm ("sha1", "sha256", etc.).
            Defaults to "sha1".

    Returns:
        str: Hex digest string.

    Notes:
        - Uses json.dumps(sort_keys=True) to get deterministic serialization.
        - Useful for generating unique identifiers for parameter combinations.
        - The hash is stable across Python sessions for the same configuration.
    """
    json_str = json.dumps(cfg, sort_keys=True)
    hash_obj = hashlib.new(algo)
    hash_obj.update(json_str.encode("utf-8"))

    return hash_obj.hexdigest()


def ensure_float64(x: np.ndarray) -> np.ndarray:
    """
    Return x as C-contiguous float64 (copy only if needed).

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Array as float64 in C-contiguous layout.

    Notes:
        - Only copies if dtype is not float64 or layout is not C-contiguous.
        - Ensures consistent memory layout for performance-critical operations.
    """
    if x.dtype == np.float64 and x.flags["C_CONTIGUOUS"]:
        return x
    return np.asarray(x, dtype=np.float64, order="C")


def ensure_c_contiguous(x: np.ndarray) -> np.ndarray:
    """
    Return x as C-contiguous (copy only if needed).

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Array in C-contiguous layout.

    Notes:
        - Only copies if the array is not already C-contiguous.
        - Preserves the original dtype.
    """
    if x.flags["C_CONTIGUOUS"]:
        return x
    return np.ascontiguousarray(x)


def safe_divide(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Elementwise a / max(b, eps) with dtype promotion to float64.

    Args:
        a (np.ndarray): Numerator array.
        b (np.ndarray): Denominator array.
        eps (float, optional): Floor for denominator to prevent division by zero.
            Defaults to 1e-12.

    Returns:
        np.ndarray: Result of safe division as float64.

    Notes:
        - Prevents division by zero by ensuring denominator >= eps.
        - Always promotes to float64 for numerical stability.
        - Useful for computing ratios where denominator may be zero or very small.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a / np.maximum(b, eps)


def discretize_width_to_bins(
    *,
    B: int,
    width_deg: Optional[float] = None,
    width_bins: Optional[int] = None,
    min_bins: int = 1,
    round_to: Literal["nearest", "floor", "ceil"] = "nearest",
) -> int:
    """
    Convert a sector width (deg or bins) to an integer bin count.

    Args:
        B (int): Number of angular bins around the circle.
        width_deg (float, optional): Width in degrees (0 < width <= 360).
        width_bins (int, optional): Width directly in bins (1..B).
        min_bins (int, optional): Lower bound after rounding. Defaults to 1.
        round_to (Literal["nearest", "floor", "ceil"], optional): Rounding policy
            when using `width_deg`. Defaults to "nearest".

    Returns:
        int: Integer bin width in [min_bins, B].

    Raises:
        ValueError: If neither width is provided or inputs are out of range.

    Notes:
        - Centralizes bin-width rounding so scanning and kernel creation agree.
    """
    if width_deg is None and width_bins is None:
        raise ValueError("Must provide either `width_deg` or `width_bins`.")

    if width_deg is not None and width_bins is not None:
        raise ValueError("Cannot provide both `width_deg` and `width_bins`.")

    if B < 1:
        raise ValueError(f"`B` must be >= 1, got {B}.")

    if min_bins < 1:
        raise ValueError(f"`min_bins` must be >= 1, got {min_bins}.")

    if width_bins is not None:
        if not isinstance(width_bins, (int, np.integer)):
            raise ValueError(
                f"`width_bins` must be an integer, got {type(width_bins).__name__}."
            )
        if width_bins < 1 or width_bins > B:
            raise ValueError(f"`width_bins` must be in [1, {B}], got {width_bins}.")
        return int(width_bins)

    if width_deg <= 0 or width_deg > 360:
        raise ValueError(f"`width_deg` must be in (0, 360], got {width_deg}.")

    bins_float = (width_deg / 360.0) * B

    if round_to == "nearest":
        m = int(np.round(bins_float))
    elif round_to == "floor":
        m = int(np.floor(bins_float))
    elif round_to == "ceil":
        m = int(np.ceil(bins_float))
    else:
        raise ValueError(
            f"Unknown round_to: '{round_to}'. Must be 'nearest', 'floor', or 'ceil'."
        )

    m = max(min_bins, min(m, B))

    return m


def argmax2d(M: np.ndarray) -> Tuple[int, int]:
    """
    Return (i, j) index of the first maximum in a 2D array.

    Args:
        M (np.ndarray): 2D array to find the maximum in.

    Returns:
        Tuple[int, int]: Row and column indices (i, j) of the maximum value.

    Raises:
        ValueError: If the input array is all NaN.

    Notes:
        - Uses np.nanargmax; raises ValueError if all-NaN.
        - Tie-breaker: earliest in row-major order.
    """
    M = np.asarray(M)
    flat_idx = np.nanargmax(M)
    indices = np.unravel_index(flat_idx, M.shape)
    return int(indices[0]), int(indices[1])


def effective_sample_size(
    w: np.ndarray, *, axis: Optional[int] = None, eps: float = 1e-12
) -> np.ndarray:
    """
    Compute ESS = (sum w)^2 / sum(w^2) along `axis`.

    Args:
        w (np.ndarray): Nonnegative weights.
        axis (Optional[int], optional): If None, returns scalar ESS; else reduced
            along axis. Defaults to None.
        eps (float, optional): Floor in denominator to avoid division by zero.
            Defaults to 1e-12.

    Returns:
        np.ndarray: Effective sample size, either scalar or array depending on axis.

    Notes:
        - Useful to gate low-information sectors/genes before testing.
        - ESS ranges from ~1 (single dominant weight) to N (uniform weights).
        - Higher ESS indicates more balanced weight distribution.
    """
    w = np.asarray(w)
    sum_w = np.sum(w, axis=axis)
    sum_w2 = np.sum(w**2, axis=axis)
    ess = (sum_w**2) / (sum_w2 + eps)
    return ess


def check_random_state(
    seed: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
) -> np.random.Generator:
    """
    Normalize various seed types to a NumPy Generator.

    Args:
        seed (Optional[Union[int, np.random.RandomState, np.random.Generator]], optional):
            Random state specification. Can be:
            - None: Returns default PCG64 Generator
            - int: Returns new Generator(PCG64(seed))
            - RandomState: Converts to Generator
            - Generator: Returns as-is
            Defaults to None.

    Returns:
        np.random.Generator: A NumPy random number generator.

    Notes:
        - None  -> default PCG64 Generator
        - int   -> new Generator(PCG64(seed))
        - RandomState/Generator -> convert/return a Generator
        - Ensures consistent modern RNG interface across the codebase.
    """
    if seed is None:
        return np.random.default_rng()
    elif isinstance(seed, int):
        return np.random.default_rng(seed)
    elif isinstance(seed, np.random.Generator):
        return seed
    elif isinstance(seed, np.random.RandomState):
        seed_sequence = np.random.SeedSequence(seed.get_state()[1][0])
        return np.random.Generator(np.random.PCG64(seed_sequence))
    else:
        raise ValueError(
            f"seed must be None, int, RandomState, or Generator, got {type(seed)}"
        )


def bootstrap_indices(
    n: int,
    R: int,
    *,
    stratify: Optional[np.ndarray] = None,
    replace: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """
    Draw bootstrap index samples.

    Args:
        n (int): Number of items to sample from (0..n-1).
        R (int): Number of bootstrap replicates (rows).
        stratify (Optional[np.ndarray], optional): If provided, resample within
            each stratum independently and concatenate. Shape (n,).
        replace (bool, optional): Bootstrap (with replacement) when True;
            subsample otherwise. Defaults to True.
        random_state (Optional[Union[int, np.random.Generator]], optional):
            Reproducible RNG seed or Generator. Defaults to None.

    Returns:
        np.ndarray: Index array of shape (R, n). Row r contains indices for
            bootstrap replicate r.

    Raises:
        ValueError: If any stratum has no members.
        ValueError: If replace=False and any stratum size < required sample.

    Notes:
        - Checks that each stratum has at least one member.
        - If replace=False and any stratum size < required sample, raises ValueError.
        - When stratify is provided, samples are drawn within each stratum and
          concatenated to maintain the original distribution.
    """
    rng = check_random_state(random_state)

    if stratify is None:
        idx = rng.choice(n, size=(R, n), replace=replace)
        return idx

    stratify = np.asarray(stratify)
    if stratify.shape != (n,):
        raise ValueError(f"`stratify` must have shape ({n},), got {stratify.shape}.")

    unique_strata = np.unique(stratify)
    if len(unique_strata) == 0:
        raise ValueError("No strata found in `stratify` array.")

    strata_indices = {}
    for stratum in unique_strata:
        stratum_mask = stratify == stratum
        stratum_idx = np.where(stratum_mask)[0]

        if len(stratum_idx) == 0:
            raise ValueError(f"Stratum {stratum} has no members.")

        strata_indices[stratum] = stratum_idx

    bootstrap_samples = np.zeros((R, n), dtype=int)

    for r in range(R):
        sampled_indices = []

        for stratum in unique_strata:
            stratum_idx = strata_indices[stratum]
            n_stratum = len(stratum_idx)

            sampled = rng.choice(stratum_idx, size=n_stratum, replace=replace)
            sampled_indices.append(sampled)

        bootstrap_samples[r] = np.concatenate(sampled_indices)

    return bootstrap_samples


def geometric_median(
    coords: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 256,
    tol: float = 1e-7,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Robust center of a point cloud via weighted Weiszfeld algorithm.

    Args:
        coords (np.ndarray): Points (d=2 for embeddings; works for general d),
            shape (N, d).
        weights (Optional[np.ndarray], optional): Nonnegative weights, shape (N,).
            Defaults to uniform weights.
        max_iter (int, optional): Max iterations for convergence. Defaults to 256.
        tol (float, optional): Convergence threshold. Defaults to 1e-7.
        eps (float, optional): Small radius floor to avoid division by zero.
            Defaults to 1e-12.

    Returns:
        np.ndarray: Geometric median coordinates, shape (d,).

    Notes:
        - More robust to outliers than the mean.
        - If any point coincides with current estimate (distance < eps),
          returns that point (weighted rule).
        - If algorithm fails to converge, returns the last iterate.
    """
    coords = np.asarray(coords)
    if weights is None:
        weights = np.ones(len(coords))
    else:
        weights = np.asarray(weights)
        if np.any(weights < 0):
            raise ValueError("Weights must be nonnegative.")

    center = np.average(coords, axis=0, weights=weights)

    for _ in range(max_iter):
        diff = coords - center
        dist = np.maximum(np.linalg.norm(diff, axis=1), eps)

        mask = dist < eps
        if np.any(mask):
            idx = np.argmax(weights * mask)
            return coords[idx]

        w = weights / dist
        new_center = np.average(coords, axis=0, weights=w)
        delta = np.linalg.norm(new_center - center)
        if delta < tol:
            return new_center

        center = new_center

    warnings.warn(
        f"Weiszfeld algorithm did not converge after {max_iter} iterations. "
        f"Last update was {delta:.2e}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return center


def circ_fft_convolve(
    arr: np.ndarray,
    kernel: np.ndarray,
    *,
    axis: int = -1,
    backend: Literal["numpy", "scipy"] = "numpy",
    real_output: bool = True,
) -> np.ndarray:
    """
    Circularly convolve `arr` with a length-B kernel along `axis`.

    Args:
        arr (np.ndarray): Input sequence(s), shape (..., B). Can be (B,), (A, B),
            or higher rank; B must match kernel.
        kernel (np.ndarray): Circular kernel centered at index 0, shape (B,).
            Must be already normalized as desired.
        axis (int, optional): Axis of length B to convolve over. Defaults to -1.
        backend (Literal["numpy", "scipy"], optional): FFT implementation to use.
            Defaults to "numpy". "scipy" may be faster if available.
        real_output (bool, optional): If True, return real-valued output via irfft
            when inputs are real. Defaults to True.

    Returns:
        np.ndarray: Convolved array with same shape as `arr`.

    Raises:
        ValueError: If kernel is not 1D.
        ValueError: If axis is invalid for arr's dimensionality.
        ValueError: If arr.shape[axis] doesn't match kernel length.
        ValueError: If backend is not "numpy" or "scipy".
        ImportError: If scipy backend requested but not available.

    Notes:
        - Handles NaNs by propagating them (caller should validate).
        - Preserves dtype when possible (promotes to float64 for safety).
        - Uses real FFT for efficiency when inputs are real.
        - Provides both numpy.fft and scipy.fft backends.
    """
    arr = np.asarray(arr)
    kernel = np.asarray(kernel)

    if kernel.ndim != 1:
        raise ValueError(f"`kernel` must be 1D, got shape {kernel.shape}.")

    B = kernel.shape[0]
    axis = axis if axis >= 0 else arr.ndim + axis
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(f"Invalid axis {axis} for array with {arr.ndim} dimensions.")

    if arr.shape[axis] != B:
        raise ValueError(
            f"`arr.shape[axis]` ({arr.shape[axis]}) must match "
            f"`kernel` length ({B})."
        )

    if backend == "scipy":
        try:
            from scipy import fft as fft_module
        except ImportError:
            fft_module = np.fft
    elif backend == "numpy":
        fft_module = np.fft
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Must be 'numpy' or 'scipy'.")

    arr_moved = np.moveaxis(arr, axis, -1)
    use_rfft = real_output and np.isrealobj(arr) and np.isrealobj(kernel)

    if use_rfft:
        arr_fft = fft_module.rfft(arr_moved, n=B, axis=-1)
        kernel_fft = fft_module.rfft(kernel, n=B)
        kernel_fft_bc = kernel_fft.reshape(
            (1,) * (arr_fft.ndim - 1) + (kernel_fft.shape[0],)
        )

        conv_fft = arr_fft * kernel_fft_bc
        result = fft_module.irfft(conv_fft, n=B, axis=-1)

    else:
        arr_fft = fft_module.fft(arr_moved, n=B, axis=-1)
        kernel_fft = fft_module.fft(kernel, n=B)
        kernel_fft_bc = kernel_fft.reshape(
            (1,) * (arr_fft.ndim - 1) + (kernel_fft.shape[0],)
        )

        conv_fft = arr_fft * kernel_fft_bc
        result = fft_module.ifft(conv_fft, n=B, axis=-1)

        if real_output and np.isrealobj(arr) and np.isrealobj(kernel):
            result = result.real

    result = np.moveaxis(result, -1, axis)

    return result


def circ_prefix_convolve(
    arr: np.ndarray, window_len: int, *, axis: int = -1
) -> np.ndarray:
    """
    Circular boxcar (flat) convolution via prefix sums; faster than FFT for small B.

    Args:
        arr (np.ndarray): Input sequence(s), shape (..., B).
        window_len (int): Window size in bins (1..B).
        axis (int, optional): Axis to convolve over. Defaults to -1.

    Returns:
        np.ndarray: Windowed sums array with same shape as input. For each center c,
            contains sum over a window of length `window_len` centered at c.

    Raises:
        ValueError: If axis is invalid for arr's dimensionality.
        ValueError: If window_len < 1 or window_len > B.

    Notes:
        - Uses wrap-around by concatenation for circular boundary conditions.
        - Requires symmetric centering convention to match scanning code.
        - O(B) complexity vs O(B log B) for FFT-based convolution.
        - Particularly efficient for small window sizes.
    """
    arr = np.asarray(arr)
    axis = axis if axis >= 0 else arr.ndim + axis
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(f"Invalid axis {axis} for array with {arr.ndim} dimensions.")

    B = arr.shape[axis]

    if window_len < 1:
        raise ValueError(f"`window_len` must be >= 1, got {window_len}.")

    if window_len > B:
        raise ValueError(
            f"`window_len` ({window_len}) cannot exceed array length ({B})."
        )

    arr_moved = np.moveaxis(arr, axis, -1)
    half_width = window_len // 2

    arr_extended = np.concatenate([arr_moved, arr_moved], axis=-1)
    prefix = np.cumsum(arr_extended, axis=-1)
    result = np.zeros_like(arr_moved)

    for c in range(B):
        start = c - half_width
        end = c + (window_len - half_width - 1)

        if start < 0:
            start += B

        if start == 0:
            result[..., c] = prefix[..., end]
        else:
            result[..., c] = prefix[..., end] - prefix[..., start - 1]

    result = np.moveaxis(result, -1, axis)

    return result


def circ_shift(x: np.ndarray, shift: int, *, axis: int = -1) -> np.ndarray:
    """
    Circularly roll an array along `axis` by integer `shift` bins.

    Args:
        x (np.ndarray): Input array to be shifted.
        shift (int): Number of positions to shift. Positive shifts right,
            negative shifts left.
        axis (int, optional): Axis along which to shift. Defaults to -1.

    Returns:
        np.ndarray: Shifted array with same shape as input.

    Raises:
        TypeError: If shift is not an integer.
        ValueError: If axis is invalid for x's dimensionality.

    Notes:
        - Wrapper over np.roll with explicit type/axis validation.
        - Performs circular shift: elements shifted off one end are wrapped
          around to the other end.
    """
    x = np.asarray(x)

    if not isinstance(shift, (int, np.integer)):
        raise TypeError(f"`shift` must be an integer, got type {type(shift)}.")

    axis = axis if axis >= 0 else x.ndim + axis
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"Invalid axis {axis} for array with {x.ndim} dimensions.")

    return np.roll(x, shift, axis=axis)


def circ_shift_batch(
    X: np.ndarray, shifts: np.ndarray, *, axis: int = -1
) -> np.ndarray:
    """
    Apply per-row (or per-slice) circular shifts: X[i] -> roll by shifts[i].

    Args:
        X (np.ndarray): Input array, shape (A, B) or (..., A, B).
        shifts (np.ndarray): Per-row shift amounts, shape (A,). Positive values
            shift right, negative values shift left.
        axis (int, optional): Target axis for shifting. The batch dimension is
            assumed to be the axis before this. Defaults to -1.

    Returns:
        np.ndarray: Array of same shape as X, with each slice along the batch
            dimension shifted by its corresponding amount in shifts.

    Raises:
        ValueError: If axis is 0 (need batch dimension before shift axis).
        ValueError: If axis is invalid for X's dimensionality.
        ValueError: If shifts.shape != (A,) where A is X's batch dimension.

    Notes:
        - Used by rotation (spin) null to avoid recomputing convolutions.
        - Efficiently handles multiple independent circular shifts.
        - Batch dimension must precede the shift axis.
    """
    X = np.asarray(X)
    shifts = np.asarray(shifts, dtype=int)

    axis = axis if axis >= 0 else X.ndim + axis
    if axis < 0 or axis >= X.ndim:
        raise ValueError(f"Invalid axis {axis} for array with {X.ndim} dimensions.")

    if axis == 0:
        raise ValueError(
            "Cannot apply batch shifts when axis=0; "
            "need at least one dimension before the shift axis."
        )

    batch_axis = axis - 1
    A = X.shape[batch_axis]

    if shifts.shape != (A,):
        raise ValueError(
            f"`shifts` must have shape ({A},) to match dimension {batch_axis}, "
            f"got shape {shifts.shape}."
        )

    X_moved = np.moveaxis(X, [batch_axis, axis], [0, 1])
    Y = np.zeros_like(X_moved)

    for i in range(A):
        Y[i] = np.roll(X_moved[i], shifts[i], axis=0)

    Y = np.moveaxis(Y, [0, 1], [batch_axis, axis])

    return Y
