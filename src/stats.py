from typing import Literal, Optional, Sequence, Tuple
import numpy as np

from . import utils


def make_kernel(
    B: int,
    *,
    width_bins: Optional[int] = None,
    width_deg: Optional[float] = None,
    kind: Literal["boxcar", "vonmises"] = "boxcar",
    kappa: Optional[float] = None,
    normalize: Literal["area", "l1", "l2"] = "area",
) -> np.ndarray:
    """
    Construct a length-B circular kernel for angular sector scanning.

    Args:
        B (int): Number of angular bins around the circle.
        width_bins (Optional[int], optional): Sector width in bins. Required for
            kind="boxcar" unless width_deg is given.
        width_deg (Optional[float], optional): Sector width in degrees; converted to
            bins as round(B * width_deg / 360).
        kind (Literal["boxcar", "vonmises"], optional): Kernel type. Defaults to "boxcar".
            - "boxcar": flat window over the sector; zeros elsewhere (fast, interpretable).
            - "vonmises": tapered circular window; symmetric, reduces edge effects.
        kappa (Optional[float], optional): Concentration parameter for von Mises.
            If not provided, it is estimated from width.
        normalize (Literal["area", "l1", "l2"], optional): Kernel scaling. Defaults to "area".
            - "area": sum(kernel) == effective width (preferred for Obs/Exp comparability).
            - "l1": sum(|kernel|) == 1.
            - "l2": ||kernel||_2 == 1.

    Returns:
        np.ndarray: Circular kernel aligned to index 0 as the "center", shape (B,).

    Raises:
        ValueError: If parameters are invalid or inconsistent.

    Notes:
        - For von Mises without kappa, a heuristic maps FWHM ≈ width to kappa via:
          kappa ≈ ln(2) / (1 - cos(width/2)), with width in radians.
          Exact match is not critical; it's a taper to stabilize edges.
        - Kernel is centered at bin 0; to scan across centers, use circular convolution.
    """
    if B < 4:
        raise ValueError(f"`B` must be >= 4, got {B}.")

    if width_bins is None and width_deg is None:
        raise ValueError("Must provide either `width_bins` or `width_deg`.")

    if width_bins is not None and width_deg is not None:
        raise ValueError("Provide only one of `width_bins` or `width_deg`, not both.")

    if width_deg is not None:
        if not (0 < width_deg <= 360):
            raise ValueError(f"`width_deg` must be in (0, 360], got {width_deg}.")
        width_bins = round(B * width_deg / 360.0)

    if width_bins < 1:
        raise ValueError(f"`width_bins` must be >= 1, got {width_bins}.")
    if width_bins > B:
        raise ValueError(f"`width_bins` ({width_bins}) cannot exceed B ({B}).")

    if kind == "boxcar":
        kernel = _make_boxcar_kernel(B, width_bins)
    elif kind == "vonmises":
        kernel = _make_vonmises_kernel(B, width_bins, kappa)
    else:
        raise ValueError(
            f"Unknown kernel kind: '{kind}'. Must be 'boxcar' or 'vonmises'."
        )

    kernel = _normalize_kernel(kernel, normalize)

    return kernel


def _make_boxcar_kernel(B: int, width_bins: int) -> np.ndarray:
    """
    Create a boxcar (flat, rectangular) kernel.

    Args:
        B (int): Number of bins.
        width_bins (int): Width of the boxcar in bins.

    Returns:
        np.ndarray: Boxcar kernel centered at index 0, shape (B,).
    """
    kernel = np.zeros(B)
    half_width = width_bins // 2

    for i in range(width_bins):
        offset = i - half_width
        idx = offset % B
        kernel[idx] = 1.0

    return kernel


def _make_vonmises_kernel(
    B: int, width_bins: int, kappa: Optional[float] = None
) -> np.ndarray:
    """
    Create a von Mises (circular normal) kernel.

    Args:
        B (int): Number of bins.
        width_bins (int): Approximate width in bins (used to estimate kappa if not provided).
        kappa (Optional[float], optional): Concentration parameter. If None, estimated
            from width_bins.

    Returns:
        np.ndarray: Von Mises kernel centered at index 0, shape (B,).
    """
    angles = 2 * np.pi * np.arange(B) / B

    if kappa is None:
        width_rad = 2 * np.pi * width_bins / B

        # heuristic: map FWHM to kappa
        # for von Mises, FWHM relates to kappa through the half-width at half-maximum
        # approx. kappa ≈ ln(2) / (1 - cos(width/2))
        half_width_rad = width_rad / 2

        denominator = 1 - np.cos(half_width_rad)
        if denominator < 1e-10:
            kappa = 100.0
        else:
            kappa = np.log(2) / denominator

    # PDF(θ; μ=0, κ) = exp(κ * cos(θ)) / (2π * I₀(κ))
    kernel = np.exp(kappa * np.cos(angles))

    return kernel


def _normalize_kernel(
    kernel: np.ndarray, normalize: Literal["area", "l1", "l2"]
) -> np.ndarray:
    """
    Normalize the kernel according to the specified method.

    Args:
        kernel (np.ndarray): Unnormalized kernel.
        normalize (Literal["area", "l1", "l2"]): Normalization method.

    Returns:
        np.ndarray: Normalized kernel.

    Raises:
        ValueError: If normalization method is unknown or kernel sum is zero.
    """
    if normalize == "area":
        total = np.sum(kernel)
        if total == 0:
            raise ValueError("Kernel sum is zero; cannot normalize with 'area' method.")
        return kernel / total * np.sum(kernel > 0.01 * kernel.max())

    elif normalize == "l1":
        total = np.sum(np.abs(kernel))
        if total == 0:
            raise ValueError("Kernel L1 norm is zero; cannot normalize.")
        return kernel / total

    elif normalize == "l2":
        norm = np.linalg.norm(kernel)
        if norm == 0:
            raise ValueError("Kernel L2 norm is zero; cannot normalize.")
        return kernel / norm

    else:
        raise ValueError(
            f"Unknown normalization method: '{normalize}'. "
            "Must be 'area', 'l1', or 'l2'."
        )


def expected_from_uniform(total_weight: float, window_len_bins: int, B: int) -> float:
    """
    Expected sum inside a sector under angular uniformity.

    Args:
        total_weight (float): Sum of weights over all bins for the band/feature.
        window_len_bins (int): Effective sector length in bins (sum of kernel if
            "area" normalization).
        B (int): Total number of bins.

    Returns:
        float: Expected value = total_weight * (window_len_bins / B).

    Notes:
        - Used to compute enrichment ratios Obs/Exp and standardized Z-scores.
        - Assumes uniform distribution across all angular bins.
        - For a sector covering window_len_bins out of B total bins, the expected
          proportion of total weight is window_len_bins / B.
    """
    return total_weight * (window_len_bins / B)


def variance_binomial_like(
    total_weight: float,
    window_len_bins: int,
    B: int,
    *,
    overdispersion: float = 0.0,
) -> float:
    """
    Approximate variance of sector sum under a binomial/Poisson-like model.

    Args:
        total_weight (float): Total count/weight in the circle (e.g., number of positives).
        window_len_bins (int): Sector width in bins.
        B (int): Number of bins around the circle.
        overdispersion (float, optional): Nonnegative extra-Poisson term (phi);
            variance *= (1 + phi). Defaults to 0.0.

    Returns:
        float: Variance of the sector sum.

    Raises:
        ValueError: If overdispersion is negative or B is zero.

    Notes:
        - Good for binary/indicator features or count-like weights.
        - For continuous weights with strong heteroscedasticity, prefer plug-in variance.
        - Under a binomial model with n trials and success probability p:
          var = n * p * (1 - p)
        - Here: n = total_weight, p = window_len_bins / B
        - Overdispersion factor (1 + phi) accounts for extra-binomial variation.
    """
    if overdispersion < 0:
        raise ValueError(
            f"`overdispersion` must be non-negative, got {overdispersion}."
        )

    if B == 0:
        raise ValueError("`B` cannot be zero.")

    p = window_len_bins / B
    var_binomial = total_weight * p * (1 - p)
    variance = var_binomial * (1 + overdispersion)

    return variance


def variance_plugin(
    wedge_sums: np.ndarray, kernel: np.ndarray, *, ddof: int = 1
) -> float:
    """
    Plug-in variance estimator using wedge-level second moments.

    Args:
        wedge_sums (np.ndarray): Per-bin totals (e.g., sum of weights per bin for
            a given band), shape (B,).
        kernel (np.ndarray): Circular kernel used for the sector (aligned to center
            at idx 0), shape (B,).
        ddof (int, optional): Degrees of freedom for variance estimation. Defaults to 1.

    Returns:
        float: Approximate Var(sum(kernel * wedge_sums shifted over c)) at a given center.

    Raises:
        ValueError: If wedge_sums and kernel have different lengths or if there are
            insufficient bins for variance estimation.

    Notes:
        - Approximates variance via sum_{b} kernel[b]^2 * Var(wedge_sums[b]).
        - Var(wedge_sums[b]) is estimated from wedge_sums' dispersion across bins
          after removing the mean (stationarity assumption).
        - Used when weights are continuous or overdispersed.
        - Assumes approximate independence or weak correlation between bins.
    """
    wedge_sums = np.asarray(wedge_sums)
    kernel = np.asarray(kernel)

    if wedge_sums.ndim != 1:
        raise ValueError(f"`wedge_sums` must be 1D, got shape {wedge_sums.shape}.")

    if kernel.ndim != 1:
        raise ValueError(f"`kernel` must be 1D, got shape {kernel.shape}.")

    if len(wedge_sums) != len(kernel):
        raise ValueError(
            f"`wedge_sums` length ({len(wedge_sums)}) must match "
            f"`kernel` length ({len(kernel)})."
        )

    B = len(wedge_sums)

    if B <= ddof:
        raise ValueError(
            f"Insufficient bins for variance estimation: B={B}, ddof={ddof}. "
            f"Need B > ddof."
        )

    # under stationarity assumption, all bins have the same variance
    var_per_bin = np.var(wedge_sums, ddof=ddof)

    # for a weighted sum S = sum_b (kernel[b] * wedge_sums[b]),
    # if wedge_sums[b] are independent with variance var_per_bin,
    # then Var(S) = sum_b (kernel[b]^2 * var_per_bin)
    #             = var_per_bin * sum_b (kernel[b]^2)
    variance = var_per_bin * np.sum(kernel**2)

    return variance


def z_scores(
    obs: np.ndarray, exp: np.ndarray, var: np.ndarray, *, eps: float = 1e-12
) -> np.ndarray:
    """
    Standardize observed-minus-expected by its standard deviation.

    Args:
        obs (np.ndarray): Observed sum in sector (broadcastable array).
        exp (np.ndarray): Expected sum under uniformity (broadcastable array).
        var (np.ndarray): Variance estimate (broadcastable array).
        eps (float, optional): Numerical floor to avoid division by zero. Defaults to 1e-12.

    Returns:
        np.ndarray: Z-scores = (obs - exp) / sqrt(var + eps).

    Notes:
        - Clips/guards var to remain nonnegative before sqrt.
        - Arrays must be broadcastable to the same shape.
        - Z-scores indicate how many standard deviations the observed value
          is from the expected value under the null hypothesis.
        - Positive Z-scores indicate enrichment, negative indicate depletion.
    """
    obs = np.asarray(obs)
    exp = np.asarray(exp)
    var = np.asarray(var)

    var_clipped = np.maximum(var, 0.0)
    std = np.sqrt(var_clipped + eps)
    z = (obs - exp) / std

    return z


def aggregate_bands(
    Z_per_band: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    method: Literal["fixed", "random"] = "fixed",
    tau2: Optional[float] = None,
) -> np.ndarray:
    """
    Combine Z across radial bands into a single Z per center.

    Args:
        Z_per_band (np.ndarray): Z-scores for each band, shape (A, B) where A is the
            number of bands and B is the number of centers (or (A, ...) broadcastable).
        weights (Optional[np.ndarray], optional): Inverse-variance or ESS-based weights,
            shape (A,). If None, equal weights are used.
        method (Literal["fixed", "random"], optional): Aggregation method. Defaults to "fixed".
            - "fixed": weighted average of Z (normalized by sqrt of sum weights).
            - "random": DerSimonian-Laird-style with tau2 (between-band variance).
        tau2 (Optional[float], optional): Between-band variance for random-effects.
            If None, it is estimated from Z when using random effects.

    Returns:
        np.ndarray: Aggregated Z-scores over centers, shape (B,) or matching the
            trailing dimensions of Z_per_band.

    Raises:
        ValueError: If method is unknown or if inputs have incompatible shapes.

    Notes:
        - Keep simple; goal is stability across rings, not meta-analysis perfection.
        - Fixed effects: assumes all bands estimate the same underlying signal.
        - Random effects: allows heterogeneity between bands.
        - For random effects without tau2, uses simple variance-based estimate.
    """
    Z_per_band = np.asarray(Z_per_band)

    if Z_per_band.ndim < 1:
        raise ValueError(
            f"`Z_per_band` must have at least 1 dimension, got shape {Z_per_band.shape}."
        )

    A = Z_per_band.shape[0]
    if weights is None:
        weights = np.ones(A)
    else:
        weights = np.asarray(weights)
        if weights.shape != (A,):
            raise ValueError(
                f"`weights` must have shape ({A},) to match number of bands, "
                f"got shape {weights.shape}."
            )

    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")

    weights_reshaped = weights.reshape(A, *([1] * (Z_per_band.ndim - 1)))

    if method == "fixed":
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            raise ValueError("Sum of weights is zero; cannot compute fixed effects.")

        weighted_sum = np.sum(weights_reshaped * Z_per_band, axis=0)
        Z_agg = weighted_sum / np.sqrt(sum_weights)

    elif method == "random":
        if tau2 is None:
            mean_Z = np.mean(Z_per_band, axis=0)
            var_Z = (
                np.var(Z_per_band, axis=0, ddof=1) if A > 1 else np.zeros_like(mean_Z)
            )
            mean_weight = np.mean(weights)
            if mean_weight > 0:
                tau2_est = np.maximum(0, var_Z - 1.0 / mean_weight)
                tau2 = float(np.median(tau2_est))
            else:
                tau2 = 0.0

        adjusted_weights = 1.0 / (1.0 / (weights + 1e-12) + tau2)
        adjusted_weights_reshaped = adjusted_weights.reshape(
            A, *([1] * (Z_per_band.ndim - 1))
        )

        sum_adjusted_weights = np.sum(adjusted_weights)
        if sum_adjusted_weights == 0:
            raise ValueError(
                "Sum of adjusted weights is zero; cannot compute random effects."
            )

        weighted_sum = np.sum(adjusted_weights_reshaped * Z_per_band, axis=0)
        Z_agg = weighted_sum / np.sqrt(sum_adjusted_weights)

    else:
        raise ValueError(f"Unknown method: '{method}'. Must be 'fixed' or 'random'.")

    return Z_agg


def enrichment_ratio(
    obs: np.ndarray, exp: np.ndarray, *, eps: float = 1e-12
) -> np.ndarray:
    """
    Compute Obs/Exp with a small guard.

    Args:
        obs (np.ndarray): Observed sum in sector (broadcastable array).
        exp (np.ndarray): Expected sum under uniformity (broadcastable array).
        eps (float, optional): Floor for denominator to avoid division by zero.
            Defaults to 1e-12.

    Returns:
        np.ndarray: Enrichment ratio = obs / max(exp, eps).

    Notes:
        - Useful as an effect size alongside Z-scores.
        - ER > 1 indicates enrichment (observed > expected).
        - ER < 1 indicates depletion (observed < expected).
        - ER ≈ 1 indicates no enrichment (observed ≈ expected).
        - Guards against division by zero when expected value is very small.
    """
    obs = np.asarray(obs)
    exp = np.asarray(exp)

    exp_guarded = np.maximum(exp, eps)
    er = obs / exp_guarded

    return er


def mean_resultant_length(
    wedge_weights: np.ndarray, *, B: int, offset_rad: float = 0.0
) -> float:
    """
    Circular concentration metric R in [0,1] from wedge weights.

    Args:
        wedge_weights (np.ndarray): Weights per angular bin (nonnegative), shape (B,).
        B (int): Number of bins; defines the bin centers at 2π*(0..B-1)/B.
        offset_rad (float, optional): Rotational offset applied to bin centers before
            computing R. Defaults to 0.0.

    Returns:
        float: Mean resultant length R = |Σ_b w_b × e^(i×θ_b)| / Σ_b w_b, in range [0, 1].

    Raises:
        ValueError: If wedge_weights has wrong shape or contains negative values.

    Notes:
        - R → 1 for highly concentrated angles (strong directional signal).
        - R → 0 for uniform weights (no preferred direction).
        - Measures how concentrated the angular distribution is.
        - Commonly used in circular statistics (Rayleigh test, etc.).
        - Formula uses complex numbers: sum of unit vectors weighted by bin weights.
    """
    wedge_weights = np.asarray(wedge_weights)

    if wedge_weights.ndim != 1:
        raise ValueError(
            f"`wedge_weights` must be 1D, got shape {wedge_weights.shape}."
        )

    if len(wedge_weights) != B:
        raise ValueError(
            f"`wedge_weights` length ({len(wedge_weights)}) must match B ({B})."
        )

    if np.any(wedge_weights < 0):
        raise ValueError("`wedge_weights` must be nonnegative.")

    total_weight = np.sum(wedge_weights)
    if total_weight == 0:
        return 0.0

    bin_centers = 2 * np.pi * np.arange(B) / B
    theta = bin_centers + offset_rad

    complex_sum = np.sum(wedge_weights * np.exp(1j * theta))
    R = np.abs(complex_sum) / total_weight

    return float(R)


def circular_fourier(
    wedge_weights: np.ndarray,
    orders: Sequence[int] = (1, 2, 3),
    *,
    B: int,
    offset_rad: float = 0.0,
    normalize: Literal["none", "l1", "l2"] = "none",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Low-order circular Fourier coefficients to detect multi-lobed patterns.

    Args:
        wedge_weights (np.ndarray): Nonnegative weights per bin, shape (B,).
        orders (Sequence[int], optional): Harmonic orders (k=1 dipole, k=2 quadrupole, ...).
            Defaults to (1, 2, 3).
        B (int): Number of bins; sets bin-center angles.
        offset_rad (float, optional): Rotate angles before computing coefficients.
            Defaults to 0.0.
        normalize (Literal["none", "l1", "l2"], optional): Optional scaling of input weights.
            Defaults to "none".
            - "none": Use weights as-is.
            - "l1": Normalize weights to sum to 1.
            - "l2": Normalize weights by L2 norm.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - coeffs (np.ndarray): Complex circular Fourier coefficients a_k,
              shape (len(orders),), where a_k = Σ_b w_b × e^(-i×k×θ_b).
            - magnitudes (np.ndarray): |a_k| normalized by sum(w) for comparability,
              shape (len(orders),).

    Raises:
        ValueError: If wedge_weights has wrong shape, contains negative values,
            or normalization method is invalid.

    Notes:
        - Useful diagnostic for "two-lobed" or "three-lobed" structure across angle.
        - Order k=1: dipole pattern (one main direction)
        - Order k=2: quadrupole pattern (two-lobed, opposite directions)
        - Order k=3: hexapole pattern (three-lobed)
        - Higher |a_k| indicates stronger k-fold symmetry.
        - Phase of a_k indicates orientation of the pattern.
    """
    wedge_weights = np.asarray(wedge_weights)

    if wedge_weights.ndim != 1:
        raise ValueError(
            f"`wedge_weights` must be 1D, got shape {wedge_weights.shape}."
        )

    if len(wedge_weights) != B:
        raise ValueError(
            f"`wedge_weights` length ({len(wedge_weights)}) must match B ({B})."
        )

    if np.any(wedge_weights < 0):
        raise ValueError("`wedge_weights` must be nonnegative.")

    weights = wedge_weights.copy()
    if normalize == "l1":
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
    elif normalize == "l2":
        norm = np.linalg.norm(weights)
        if norm > 0:
            weights = weights / norm
    elif normalize != "none":
        raise ValueError(
            f"Unknown normalization method: '{normalize}'. "
            "Must be 'none', 'l1', or 'l2'."
        )

    total_weight = np.sum(weights)
    if total_weight == 0:
        n_orders = len(orders)
        return np.zeros(n_orders, dtype=complex), np.zeros(n_orders, dtype=float)

    bin_centers = 2 * np.pi * np.arange(B) / B
    theta = bin_centers + offset_rad

    orders_array = np.asarray(orders)
    n_orders = len(orders_array)

    coeffs = np.zeros(n_orders, dtype=complex)
    magnitudes = np.zeros(n_orders, dtype=float)

    for i, k in enumerate(orders_array):
        a_k = np.sum(weights * np.exp(-1j * k * theta))

        coeffs[i] = a_k
        magnitudes[i] = np.abs(a_k) / total_weight

    return coeffs, magnitudes


def sector_sums_convolved(
    wedge_sums: np.ndarray,
    kernel: np.ndarray,
    *,
    engine: Literal["fft", "prefix"] = "fft",
) -> np.ndarray:
    """
    Convolve wedge sums with a circular kernel to get sector totals at all centers.

    Args:
        wedge_sums (np.ndarray): Per-bin sums (optionally across A radial bands),
            shape (B,) or (A, B).
        kernel (np.ndarray): Circular kernel centered at index 0, shape (B,).
        engine (Literal["fft", "prefix"], optional): Convolution method. Defaults to "fft".
            - "fft": use utils.circ_fft_convolve (O(B log B)).
            - "prefix": use utils.circ_prefix_convolve (boxcar-only, O(B)).

    Returns:
        np.ndarray: Sector totals for each center (and band), shape matches input.

    Raises:
        ValueError: If kernel is not 1D or if engine is unknown.
        ValueError: For prefix method, if window length cannot be determined.

    Notes:
        - This is a thin wrapper that delegates to `utils` for speed.
        - Kernel must already be normalized as desired.
        - "fft" method works for any kernel shape.
        - "prefix" method requires boxcar (flat) kernel for O(B) performance.
    """
    wedge_sums = np.asarray(wedge_sums)
    kernel = np.asarray(kernel)

    if kernel.ndim != 1:
        raise ValueError(f"`kernel` must be 1D, got shape {kernel.shape}.")

    if engine == "fft":
        result = utils.circ_fft_convolve(wedge_sums, kernel, axis=-1)

    elif engine == "prefix":
        window_len = int(np.sum(kernel > 0.01 * kernel.max()))
        if window_len < 1:
            raise ValueError(
                "Cannot determine window length from kernel for prefix method."
            )

        result = utils.circ_prefix_convolve(wedge_sums, window_len, axis=-1)

    else:
        raise ValueError(f"Unknown engine: '{engine}'. Must be 'fft' or 'prefix'.")

    return result


def compute_Z_grid(
    wedge_sums: np.ndarray,
    totals_per_band: np.ndarray,
    kernel: np.ndarray,
    *,
    B: int,
    var_mode: Literal["binomial", "plugin"] = "binomial",
    overdispersion: float = 0.0,
    engine: Literal["fft", "prefix"] = "fft",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce Obs/Exp/Var and Z for a given kernel across all centers.

    Parameters
    ----------
    wedge_sums : (A, B) or (B,)
        Per-band per-bin sums (A bands) or single band.
    totals_per_band : (A,) or (1,)
        Totals per band (sum over bins).
    kernel : (B,)
        Normalized circular kernel.
    B : int
        Number of angular bins.
    var_mode : {"binomial","plugin"}
        Variance model.
    overdispersion : float
        Extra-Poisson factor for binomial-like variance.
    engine : {"fft","prefix"}
        Convolution engine (delegates to utils).

    Returns
    -------
    Obs : (A, B) or (B,)
    Exp : (A, B) or (B,)
    Var : (A, B) or (B,)
    Z   : (A, B) or (B,)

    Notes
    -----
    - Shapes are preserved: if input is (A, B), outputs are (A, B).
    - Caller can aggregate across bands with `aggregate_bands`.
    """
    wedge_sums = np.asarray(wedge_sums)
    totals_per_band = np.asarray(totals_per_band)
    kernel = np.asarray(kernel)

    if kernel.ndim != 1:
        raise ValueError(f"`kernel` must be 1D, got shape {kernel.shape}.")

    if len(kernel) != B:
        raise ValueError(f"`kernel` length ({len(kernel)}) must match B ({B}).")

    if wedge_sums.ndim == 1:
        if len(wedge_sums) != B:
            raise ValueError(
                f"`wedge_sums` length ({len(wedge_sums)}) must match B ({B})."
            )
        wedge_sums = wedge_sums.reshape(1, B)
        totals_per_band = totals_per_band.reshape(-1)
        if len(totals_per_band) != 1:
            raise ValueError(
                f"For 1D wedge_sums, `totals_per_band` must have length 1, "
                f"got {len(totals_per_band)}."
            )
        squeeze_output = True
    elif wedge_sums.ndim == 2:
        A, B_check = wedge_sums.shape
        if B_check != B:
            raise ValueError(
                f"`wedge_sums` second dimension ({B_check}) must match B ({B})."
            )
        if len(totals_per_band) != A:
            raise ValueError(
                f"`totals_per_band` length ({len(totals_per_band)}) must match "
                f"number of bands ({A})."
            )
        squeeze_output = False
    else:
        raise ValueError(
            f"`wedge_sums` must be 1D or 2D, got shape {wedge_sums.shape}."
        )

    A = wedge_sums.shape[0]
    Obs = sector_sums_convolved(wedge_sums, kernel, engine=engine)
    window_len_bins = int(np.sum(kernel > 0.01 * kernel.max()))

    Exp = np.zeros((A, B), dtype=float)
    for a in range(A):
        exp_val = expected_from_uniform(totals_per_band[a], window_len_bins, B)
        Exp[a, :] = exp_val

    Var = np.zeros((A, B), dtype=float)
    if var_mode == "binomial":
        for a in range(A):
            var_val = variance_binomial_like(
                totals_per_band[a],
                window_len_bins,
                B,
                overdispersion=overdispersion,
            )
            Var[a, :] = var_val

    elif var_mode == "plugin":
        for a in range(A):
            var_val = variance_plugin(wedge_sums[a], kernel)
            Var[a, :] = var_val

    else:
        raise ValueError(
            f"Unknown var_mode: '{var_mode}'. Must be 'binomial' or 'plugin'."
        )

    Z = z_scores(Obs, Exp, Var)
    if squeeze_output:
        Obs = Obs.squeeze(axis=0)
        Exp = Exp.squeeze(axis=0)
        Var = Var.squeeze(axis=0)
        Z = Z.squeeze(axis=0)

    return Obs, Exp, Var, Z


def compute_fg_bg_difference(
    wedge_idx,
    feature,
    weights=None,
    B=256,
):
    """
    Compute difference in proportions between foreground and background distributions.

    For binary features (foreground=1, background=all cells), this computes:
        diff[wedge] = (n_fg_wedge / n_fg_total) - (n_all_wedge / n_all_total)

    This tests whether the foreground (expressing cells) distribution differs from 
    the background (all cells) distribution. The background includes the foreground
    as a subset, making this a proper one-sample test against the reference distribution.

    Parameters
    ----------
    wedge_idx : array-like, shape (N,) or (M, N)
        Wedge bin assignment for each cell (integer in [0, B-1]).
    feature : array-like, shape (N,) or (M, N)
        Binary feature values (1 = foreground/expressing, 0 = non-expressing).
        Note: Background = ALL cells (both expressing and non-expressing).
    weights : array-like, shape (N,) or (M, N), optional
        Weights for each cell. If None, uniform weights are used.
    B : int, default=256
        Number of angular bins.

    Returns
    -------
    diff : ndarray, shape (B,) or (M, B)
        Difference in proportions for each wedge position.
        Positive values indicate foreground is enriched, negative indicates depletion.
    fg_prop : ndarray, shape (B,) or (M, B)
        Foreground proportion in each wedge.
    bg_prop : ndarray, shape (B,) or (M, B)
        Background (all cells) proportion in each wedge.
    """
    wedge_idx = np.atleast_1d(wedge_idx)
    feature = np.atleast_1d(feature)

    if weights is None:
        weights = np.ones_like(feature, dtype=float)
    else:
        weights = np.atleast_1d(weights)

    # Foreground: expressing cells only (feature > 0)
    fg_mask = feature > 0
    fg_weights = weights * fg_mask
    total_fg = np.sum(fg_weights)

    # Background: ALL cells (not just non-expressing!)
    # This is the key fix - background includes foreground as a subset
    bg_weights = weights  # All cells, not weights * (feature == 0)
    total_bg = np.sum(bg_weights)

    fg_sums = np.bincount(wedge_idx, weights=fg_weights, minlength=B)
    bg_sums = np.bincount(wedge_idx, weights=bg_weights, minlength=B)

    fg_prop = np.zeros(B, dtype=float)
    bg_prop = np.zeros(B, dtype=float)

    if total_fg > 0:
        fg_prop = fg_sums / total_fg

    if total_bg > 0:
        bg_prop = bg_sums / total_bg

    diff = fg_prop - bg_prop

    return diff, fg_prop, bg_prop
