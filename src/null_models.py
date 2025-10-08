from typing import Literal, Optional, Tuple, Union
import numpy as np
from .utils import check_random_state, circ_shift
from . import stats


def _safe_weights(totals_per_band: np.ndarray) -> np.ndarray:
    """
    Filter weights to handle degenerate bands (total_weight <= 0).

    Args:
        totals_per_band (np.ndarray): Per-band totals, shape (A,).

    Returns:
        np.ndarray: Safe weights with zero weight for degenerate bands.

    Notes:
        - Bands with total_weight <= 0 are assigned zero weight to avoid
          division by zero in variance calculations.
        - This ensures numerical stability in band aggregation.
    """
    weights = totals_per_band.copy()
    weights[weights <= 0] = 0.0
    return weights


def rotation_shifts(
    B: int, R: int, *, random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
    """
    Sample R independent cyclic shifts (spin angles in bin units) ~ Uniform{0..B-1}.

    Args:
        B (int): Number of angular bins around the circle.
        R (int): Number of rotation replicates.
        random_state (Optional[Union[int, np.random.Generator]], optional):
            Reproducible RNG seed or Generator. Defaults to None.

    Returns:
        np.ndarray: Array of shifts, shape (R,). Each entry is a shift in [0, B-1].

    Notes:
        - Use these shifts to roll pre-binned wedge sums; cheaper than rotating raw θ.
        - Shifts are uniformly sampled from {0, 1, ..., B-1}.
    """
    rng = check_random_state(random_state)
    shifts = rng.integers(0, B, size=R, dtype=int)
    return shifts


def max_stat_under_rotations(
    wedge_sums: np.ndarray,
    totals_per_band: np.ndarray,
    kernels: np.ndarray,
    *,
    B: int,
    R: int = 500,
    var_mode: Literal["binomial", "plugin"] = "binomial",
    overdispersion: float = 0.0,
    engine: Literal["fft", "prefix"] = "fft",
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """
    Rotation (spin) null: empirical distribution of the max Z over all centers × widths.

    Args:
        wedge_sums (np.ndarray): Per-band (A) wedge totals for the feature (e.g.,
            sums of weights in each bin), shape (A, B).
        totals_per_band (np.ndarray): Per-band totals (sum over B), shape (A,).
        kernels (np.ndarray): Bank of J circular kernels (widths) pre-normalized
            as in stats.make_kernel, shape (J, B).
        B (int): Number of angular bins.
        R (int, optional): Number of rotation replicates. Defaults to 500.
        var_mode (Literal["binomial", "plugin"], optional): Variance model for Z.
            Defaults to "binomial".
            - "binomial": Use binomial-like variance model.
            - "plugin": Use plug-in variance estimator.
        overdispersion (float, optional): Extra-Poisson factor for "binomial" variance.
            Defaults to 0.0.
        engine (Literal["fft", "prefix"], optional): Convolution engine. Defaults to "fft".
            - "fft": FFT-based convolution.
            - "prefix": Prefix sum-based convolution (boxcar kernels only).
        random_state (Optional[Union[int, np.random.Generator]], optional):
            Reproducible RNG seed or Generator. Defaults to None.

    Returns:
        np.ndarray: Max Z across centers × widths for each rotation replicate,
            shape (R,).

    Notes:
        - For each replicate, roll wedge_sums by a single random shift (same for all bands),
          compute Z grids for all kernels, and record the global maximum.
        - Preserves radial distribution, per-band totals, and wedge autocorrelation,
          while breaking absolute angle → calibrates "we tried many angles & widths".
        - Delegates variance and Z computation to stats module functions.
    """
    wedge_sums = np.asarray(wedge_sums)
    totals_per_band = np.asarray(totals_per_band)
    kernels = np.asarray(kernels)

    A, B_check = wedge_sums.shape
    if B_check != B:
        raise ValueError(
            f"wedge_sums has {B_check} bins (axis 1), but B={B} was specified."
        )

    J, B_kern = kernels.shape
    if B_kern != B:
        raise ValueError(
            f"kernels have {B_kern} bins (axis 1), but B={B} was specified."
        )

    if totals_per_band.shape != (A,):
        raise ValueError(
            f"totals_per_band must have shape ({A},), got {totals_per_band.shape}."
        )

    shifts = rotation_shifts(B, R, random_state=random_state)
    Zmax_null = np.zeros(R)

    for r in range(R):
        wedge_sums_rotated = circ_shift(wedge_sums, shifts[r], axis=-1)
        max_z = -np.inf

        for j in range(J):
            kernel = kernels[j]
            _, _, _, Z_band = stats.compute_Z_grid(
                wedge_sums_rotated,
                totals_per_band,
                kernel,
                B=B,
                var_mode=var_mode,
                overdispersion=overdispersion,
                engine=engine,
            )

            Z = stats.aggregate_bands(
                Z_band, weights=_safe_weights(totals_per_band), method="fixed"
            )

            max_z = max(max_z, np.nanmax(Z))

        Zmax_null[r] = max_z

    return Zmax_null


def max_stat_within_batch_rotations(
    wedge_sums_per_batch: np.ndarray,
    totals_per_band_per_batch: np.ndarray,
    kernels: np.ndarray,
    *,
    B: int,
    R: int = 500,
    var_mode: Literal["binomial", "plugin"] = "binomial",
    overdispersion: float = 0.0,
    engine: Literal["fft", "prefix"] = "fft",
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """
    Blocked (batch-wise) rotation null: each batch is spun independently.

    Args:
        wedge_sums_per_batch (np.ndarray): G batches, A radial bands, B wedges —
            per-batch per-band wedge sums, shape (G, A, B).
        totals_per_band_per_batch (np.ndarray): Totals per batch & band, shape (G, A).
        kernels (np.ndarray): Bank of J circular kernels, shape (J, B).
        B (int): Number of angular bins.
        R (int, optional): Number of rotation replicates. Defaults to 500.
        var_mode (Literal["binomial", "plugin"], optional): Variance model for Z.
            Defaults to "binomial".
        overdispersion (float, optional): Extra-Poisson factor for "binomial" variance.
            Defaults to 0.0.
        engine (Literal["fft", "prefix"], optional): Convolution engine. Defaults to "fft".
        random_state (Optional[Union[int, np.random.Generator]], optional):
            Reproducible RNG seed or Generator. Defaults to None.

    Returns:
        np.ndarray: Max Z across centers × widths for each rotation replicate,
            shape (R,).

    Notes:
        - For each replicate, draw an independent shift for each batch g, roll the
          (A, B) slab for that batch, then sum over g to get overall wedge_sums before
          computing Z across kernels.
        - Preserves batch composition per wedge and guards against batch-driven angles.
        - Each batch is rotated independently to break batch-specific angular patterns
          while maintaining the overall batch structure.
    """
    wedge_sums_per_batch = np.asarray(wedge_sums_per_batch)
    totals_per_band_per_batch = np.asarray(totals_per_band_per_batch)
    kernels = np.asarray(kernels)

    if wedge_sums_per_batch.ndim != 3:
        raise ValueError(
            f"wedge_sums_per_batch must be 3D (G, A, B), got shape {wedge_sums_per_batch.shape}."
        )

    G, A, B_check = wedge_sums_per_batch.shape
    if B_check != B:
        raise ValueError(
            f"wedge_sums_per_batch has {B_check} bins (axis 2), but B={B} was specified."
        )

    if totals_per_band_per_batch.shape != (G, A):
        raise ValueError(
            f"totals_per_band_per_batch must have shape ({G}, {A}), "
            f"got {totals_per_band_per_batch.shape}."
        )

    J, B_kern = kernels.shape
    if B_kern != B:
        raise ValueError(
            f"kernels have {B_kern} bins (axis 1), but B={B} was specified."
        )

    rng = check_random_state(random_state)
    Zmax_null = np.zeros(R)

    for r in range(R):
        batch_shifts = rng.integers(0, B, size=G, dtype=int)
        wedge_sums_rotated_batches = np.zeros((G, A, B))

        for g in range(G):
            wedge_sums_rotated_batches[g] = circ_shift(
                wedge_sums_per_batch[g], batch_shifts[g], axis=-1
            )

        wedge_sums_rotated = np.sum(wedge_sums_rotated_batches, axis=0)
        totals_per_band = np.sum(totals_per_band_per_batch, axis=0)

        max_z = -np.inf

        for j in range(J):
            kernel = kernels[j]
            _, _, _, Z_band = stats.compute_Z_grid(
                wedge_sums_rotated,
                totals_per_band,
                kernel,
                B=B,
                var_mode=var_mode,
                overdispersion=overdispersion,
                engine=engine,
            )

            Z = stats.aggregate_bands(
                Z_band, weights=_safe_weights(totals_per_band), method="fixed"
            )

            max_z = max(max_z, np.nanmax(Z))

        Zmax_null[r] = max_z

    return Zmax_null


def label_permutation_within_bands(
    weights: np.ndarray,
    wedge_idx: np.ndarray,
    band_idx: np.ndarray,
    *,
    kernels: np.ndarray,
    B: int,
    A: int,
    R: int = 500,
    var_mode: Literal["binomial", "plugin"] = "binomial",
    overdispersion: float = 0.0,
    engine: Literal["fft", "prefix"] = "fft",
    batches: Optional[np.ndarray] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """
    Permutation null: shuffle feature weights within radial bands (and optionally batches).

    Args:
        weights (np.ndarray): Per-cell feature weights (binary or continuous), shape (N,).
        wedge_idx (np.ndarray): Angular bin assignment for each cell, shape (N,)
            with values in [0..B-1].
        band_idx (np.ndarray): Radial band assignment for each cell, shape (N,)
            with values in [0..A-1].
        kernels (np.ndarray): Bank of J circular kernels, shape (J, B).
        B (int): Number of angular bins.
        A (int): Number of radial bands.
        R (int, optional): Number of permutation replicates. Defaults to 500.
        var_mode (Literal["binomial", "plugin"], optional): Variance model for Z.
            Defaults to "binomial".
        overdispersion (float, optional): Extra-Poisson factor for "binomial" variance.
            Defaults to 0.0.
        engine (Literal["fft", "prefix"], optional): Convolution engine. Defaults to "fft".
        batches (Optional[np.ndarray], optional): Batch/donor labels, shape (N,).
            If provided, permutations occur within (band, batch) strata. Defaults to None.
        random_state (Optional[Union[int, np.random.Generator]], optional):
            Reproducible RNG seed or Generator. Defaults to None.

    Returns:
        np.ndarray: Max Z across centers × widths for each permutation replicate,
            shape (R,).

    Notes:
        - Within each stratum (band or band×batch), permute weights across cells,
          rebuild wedge sums, and compute the max Z across centers×widths.
        - Breaks any real angular association while preserving band totals and
          weight distribution; stricter than global rotations if batch imbalance exists.
        - Re-binning each replicate is O(N), so this is slower than rotations.
          Use when blocked rotations are insufficient (e.g., label-specific confounding).
    """
    weights = np.asarray(weights)
    wedge_idx = np.asarray(wedge_idx)
    band_idx = np.asarray(band_idx)
    kernels = np.asarray(kernels)

    N = len(weights)

    if wedge_idx.shape != (N,):
        raise ValueError(f"wedge_idx must have shape ({N},), got {wedge_idx.shape}.")
    if band_idx.shape != (N,):
        raise ValueError(f"band_idx must have shape ({N},), got {band_idx.shape}.")

    if np.any(wedge_idx < 0) or np.any(wedge_idx >= B):
        raise ValueError(f"wedge_idx must be in [0, {B-1}].")
    if np.any(band_idx < 0) or np.any(band_idx >= A):
        raise ValueError(f"band_idx must be in [0, {A-1}].")

    J, B_kern = kernels.shape
    if B_kern != B:
        raise ValueError(
            f"kernels have {B_kern} bins (axis 1), but B={B} was specified."
        )

    if batches is not None:
        batches = np.asarray(batches)
        if batches.shape != (N,):
            raise ValueError(f"batches must have shape ({N},), got {batches.shape}.")

    rng = check_random_state(random_state)
    Zmax_null = np.zeros(R)

    if batches is None:
        strata_labels = band_idx
    else:
        unique_batches = np.unique(batches)
        batch_to_idx = {b: i for i, b in enumerate(unique_batches)}
        batch_indices = np.array([batch_to_idx[b] for b in batches])
        n_batches = len(unique_batches)
        strata_labels = band_idx * n_batches + batch_indices

    unique_strata = np.unique(strata_labels)

    for r in range(R):
        weights_permuted = np.zeros(N)

        for stratum in unique_strata:
            stratum_mask = strata_labels == stratum
            stratum_indices = np.where(stratum_mask)[0]
            permuted_indices = rng.permutation(stratum_indices)
            weights_permuted[stratum_indices] = weights[permuted_indices]

        wedge_sums = np.zeros((A, B))
        for i in range(N):
            a = band_idx[i]
            b = wedge_idx[i]
            wedge_sums[a, b] += weights_permuted[i]

        totals_per_band = np.sum(wedge_sums, axis=1)
        max_z = -np.inf

        for j in range(J):
            kernel = kernels[j]
            _, _, _, Z_band = stats.compute_Z_grid(
                wedge_sums,
                totals_per_band,
                kernel,
                B=B,
                var_mode=var_mode,
                overdispersion=overdispersion,
                engine=engine,
            )

            Z = stats.aggregate_bands(
                Z_band, weights=_safe_weights(totals_per_band), method="fixed"
            )

            max_z = max(max_z, np.nanmax(Z))

        Zmax_null[r] = max_z

    return Zmax_null


def empirical_pvalue(
    stat_obs: float,
    stat_null: np.ndarray,
    *,
    kind: Literal["right", "left", "two-sided"] = "right",
) -> float:
    """
    Compute an empirical p-value with a +1 continuity correction.

    Args:
        stat_obs (float): Observed statistic (e.g., Z_max).
        stat_null (np.ndarray): Null replicate statistics, shape (R,).
        kind (Literal["right", "left", "two-sided"], optional): Tail for the test.
            Defaults to "right".
            - "right": p = (1 + #{null >= obs}) / (1 + R)
            - "left": p = (1 + #{null <= obs}) / (1 + R)
            - "two-sided": Uses max of counts for both tails relative to |obs|

    Returns:
        float: Empirical p-value in (0, 1].

    Raises:
        ValueError: If stat_null is empty or kind is not recognized.

    Notes:
        - Uses +1 continuity correction: (1 + count) / (1 + R).
        - For "two-sided", computes max(#{null >= |obs|}, #{null <= -|obs|}).
        - This ensures p-values are never exactly 0, which is important for
          multiple testing correction.
        - Ties are counted conservatively (null >= obs includes equality).
    """
    stat_null = np.asarray(stat_null)

    if stat_null.size == 0:
        raise ValueError("stat_null is empty; cannot compute empirical p-value.")

    R = len(stat_null)

    if kind == "right":
        count = np.sum(stat_null >= stat_obs)
        p = (1 + count) / (1 + R)

    elif kind == "left":
        count = np.sum(stat_null <= stat_obs)
        p = (1 + count) / (1 + R)

    elif kind == "two-sided":
        abs_obs = np.abs(stat_obs)
        count_right = np.sum(stat_null >= abs_obs)
        count_left = np.sum(stat_null <= -abs_obs)
        count = max(count_right, count_left)
        p = (1 + count) / (1 + R)

    else:
        raise ValueError(
            f"Unknown kind: '{kind}'. Must be 'right', 'left', or 'two-sided'."
        )

    return p


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg step-up procedure for FDR control.

    Args:
        pvals (np.ndarray): P-values, shape (G,), with values in [0, 1].

    Returns:
        np.ndarray: Q-values (adjusted p-values), shape (G,). Monotone
            non-decreasing after sorting.

    Raises:
        ValueError: If pvals contains values outside [0, 1] (excluding NaN).

    Notes:
        - NaNs are treated as 1.0 with a warning to handle missing data gracefully.
        - The BH procedure sorts p-values, computes adjusted values as
          p[i] * m / (i+1), then enforces monotonicity by reverse cummin.
        - Q-values represent the expected false discovery rate if we reject
          all hypotheses with q-value ≤ threshold.
        - Future extensions could include BY (Benjamini-Yekutieli) or IHW
          (Independent Hypothesis Weighting), but are not required here.
    """
    pvals = np.asarray(pvals, dtype=np.float64)

    if pvals.ndim != 1:
        raise ValueError(f"pvals must be 1D, got shape {pvals.shape}.")

    m = len(pvals)
    if m == 0:
        return np.array([])

    nan_mask = np.isnan(pvals)
    if np.any(nan_mask):
        import warnings

        warnings.warn(
            f"Found {np.sum(nan_mask)} NaN values in pvals; treating as 1.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        pvals = pvals.copy()
        pvals[nan_mask] = 1.0

    if np.any((pvals < 0) | (pvals > 1)):
        raise ValueError("All p-values must be in [0, 1].")

    sort_idx = np.argsort(pvals)
    pvals_sorted = pvals[sort_idx]
    ranks = np.arange(1, m + 1)
    qvals_sorted = pvals_sorted * m / ranks

    qvals_sorted = np.minimum.accumulate(qvals_sorted[::-1])[::-1]
    qvals_sorted = np.clip(qvals_sorted, 0.0, 1.0)

    qvals = np.empty(m, dtype=np.float64)
    qvals[sort_idx] = qvals_sorted

    return qvals


def max_from_Zheat(Z_heat: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    """
    Convenience: get the global max and its (width_idx, center_idx).

    Args:
        Z_heat (np.ndarray): Z scores per kernel j (width) and center c, shape (J, B).

    Returns:
        Tuple[float, Tuple[int, int]]: A tuple containing:
            - zmax (float): Maximum Z-score value.
            - argmax (Tuple[int, int]): Indices (j*, c*) where j* is the kernel/width
              index and c* is the center/bin index.

    Raises:
        ValueError: If Z_heat is not 2D or is empty.

    Notes:
        - Useful for extracting the peak signal from a heatmap of Z-scores across
          different sector widths and center positions.
        - Uses np.nanargmax to handle potential NaN values in the heatmap.
        - Returns the first occurrence if there are multiple maxima (tie-breaking).
    """
    Z_heat = np.asarray(Z_heat)

    if Z_heat.ndim != 2:
        raise ValueError(f"Z_heat must be 2D (J, B), got shape {Z_heat.shape}.")

    if Z_heat.size == 0:
        raise ValueError("Z_heat is empty; cannot find maximum.")

    flat_idx = np.nanargmax(Z_heat)
    j_max, c_max = np.unravel_index(flat_idx, Z_heat.shape)
    zmax = Z_heat[j_max, c_max]

    return zmax, (int(j_max), int(c_max))


def rotation_null_pvalue(
    wedge_sums: np.ndarray,
    totals_per_band: np.ndarray,
    kernels: np.ndarray,
    *,
    B: int,
    R: int = 500,
    var_mode: Literal["binomial", "plugin"] = "binomial",
    overdispersion: float = 0.0,
    engine: Literal["fft", "prefix"] = "fft",
    zmax_obs: Optional[float] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, np.ndarray]:
    """
    End-to-end helper: compute rotation-null p-value for the observed max Z.

    Args:
        wedge_sums (np.ndarray): Per-band wedge totals for the feature, shape (A, B).
        totals_per_band (np.ndarray): Per-band totals (sum over B), shape (A,).
        kernels (np.ndarray): Bank of J circular kernels, shape (J, B).
        B (int): Number of angular bins.
        R (int, optional): Number of rotation replicates. Defaults to 500.
        var_mode (Literal["binomial", "plugin"], optional): Variance model for Z.
            Defaults to "binomial".
        overdispersion (float, optional): Extra-Poisson factor for "binomial" variance.
            Defaults to 0.0.
        engine (Literal["fft", "prefix"], optional): Convolution engine. Defaults to "fft".
        zmax_obs (Optional[float], optional): Observed max Z. If None, will compute
            Z_heat from the unrotated input to obtain zmax_obs. Defaults to None.
        random_state (Optional[Union[int, np.random.Generator]], optional):
            Reproducible RNG seed or Generator. Defaults to None.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing:
            - p (float): Empirical (right-tail) p-value for zmax_obs.
            - Zmax_null (np.ndarray): Null distribution of max Z, shape (R,),
              returned for diagnostics.

    Notes:
        - This is the workhorse function you'll call from `radar_scan.py` for one feature.
        - If zmax_obs is not provided, computes the Z-score heatmap once on the
          unrotated data to find the observed maximum.
        - Then generates R rotation replicates and computes max Z for each.
        - Returns both the p-value and the null distribution for diagnostic purposes.
    """
    wedge_sums = np.asarray(wedge_sums)
    totals_per_band = np.asarray(totals_per_band)
    kernels = np.asarray(kernels)

    if zmax_obs is None:
        A, B_check = wedge_sums.shape
        if B_check != B:
            raise ValueError(
                f"wedge_sums has {B_check} bins (axis 1), but B={B} was specified."
            )

        J, B_kern = kernels.shape
        if B_kern != B:
            raise ValueError(
                f"kernels have {B_kern} bins (axis 1), but B={B} was specified."
            )

        if totals_per_band.shape != (A,):
            raise ValueError(
                f"totals_per_band must have shape ({A},), got {totals_per_band.shape}."
            )

        zmax_obs = -np.inf

        for j in range(J):
            kernel = kernels[j]
            _, _, _, Z_band = stats.compute_Z_grid(
                wedge_sums,
                totals_per_band,
                kernel,
                B=B,
                var_mode=var_mode,
                overdispersion=overdispersion,
                engine=engine,
            )

            Z = stats.aggregate_bands(
                Z_band, weights=_safe_weights(totals_per_band), method="fixed"
            )

            zmax_obs = max(zmax_obs, np.nanmax(Z))

    Zmax_null = max_stat_under_rotations(
        wedge_sums=wedge_sums,
        totals_per_band=totals_per_band,
        kernels=kernels,
        B=B,
        R=R,
        var_mode=var_mode,
        overdispersion=overdispersion,
        engine=engine,
        random_state=random_state,
    )

    p = empirical_pvalue(zmax_obs, Zmax_null, kind="right")

    return p, Zmax_null


def within_batch_rotation_pvalue(
    wedge_sums_per_batch: np.ndarray,
    totals_per_band_per_batch: np.ndarray,
    kernels: np.ndarray,
    *,
    B: int,
    R: int = 500,
    var_mode: Literal["binomial", "plugin"] = "binomial",
    overdispersion: float = 0.0,
    engine: Literal["fft", "prefix"] = "fft",
    zmax_obs: Optional[float] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, np.ndarray]:
    """
    End-to-end helper: blocked rotation p-value for the observed max Z.

    Args:
        wedge_sums_per_batch (np.ndarray): G batches, A radial bands, B wedges —
            per-batch per-band wedge sums, shape (G, A, B).
        totals_per_band_per_batch (np.ndarray): Totals per batch & band, shape (G, A).
        kernels (np.ndarray): Bank of J circular kernels, shape (J, B).
        B (int): Number of angular bins.
        R (int, optional): Number of rotation replicates. Defaults to 500.
        var_mode (Literal["binomial", "plugin"], optional): Variance model for Z.
            Defaults to "binomial".
        overdispersion (float, optional): Extra-Poisson factor for "binomial" variance.
            Defaults to 0.0.
        engine (Literal["fft", "prefix"], optional): Convolution engine. Defaults to "fft".
        zmax_obs (Optional[float], optional): Observed max Z. If None, will compute
            Z_heat from the unrotated input to obtain zmax_obs. Defaults to None.
        random_state (Optional[Union[int, np.random.Generator]], optional):
            Reproducible RNG seed or Generator. Defaults to None.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing:
            - p (float): Empirical (right-tail) p-value for zmax_obs.
            - Zmax_null (np.ndarray): Null distribution of max Z, shape (R,),
              returned for diagnostics.

    Notes:
        - Same semantics as rotation_null_pvalue, but for per-batch inputs.
        - Requires the caller to prepare per-batch wedge sums and band totals.
        - Use when batch/donor imbalance could drive angle artifacts.
        - Each batch is rotated independently in the null, preserving batch
          composition while breaking batch-specific angular patterns.
    """
    wedge_sums_per_batch = np.asarray(wedge_sums_per_batch)
    totals_per_band_per_batch = np.asarray(totals_per_band_per_batch)
    kernels = np.asarray(kernels)

    if zmax_obs is None:
        wedge_sums = np.sum(wedge_sums_per_batch, axis=0)
        totals_per_band = np.sum(totals_per_band_per_batch, axis=0)

        A, B_check = wedge_sums.shape
        if B_check != B:
            raise ValueError(
                f"wedge_sums has {B_check} bins (axis 1), but B={B} was specified."
            )

        J, B_kern = kernels.shape
        if B_kern != B:
            raise ValueError(
                f"kernels have {B_kern} bins (axis 1), but B={B} was specified."
            )

        if totals_per_band.shape != (A,):
            raise ValueError(
                f"totals_per_band must have shape ({A},), got {totals_per_band.shape}."
            )

        zmax_obs = -np.inf

        for j in range(J):
            kernel = kernels[j]
            _, _, _, Z_band = stats.compute_Z_grid(
                wedge_sums,
                totals_per_band,
                kernel,
                B=B,
                var_mode=var_mode,
                overdispersion=overdispersion,
                engine=engine,
            )

            Z = stats.aggregate_bands(
                Z_band, weights=_safe_weights(totals_per_band), method="fixed"
            )

            zmax_obs = max(zmax_obs, np.nanmax(Z))

    Zmax_null = max_stat_within_batch_rotations(
        wedge_sums_per_batch=wedge_sums_per_batch,
        totals_per_band_per_batch=totals_per_band_per_batch,
        kernels=kernels,
        B=B,
        R=R,
        var_mode=var_mode,
        overdispersion=overdispersion,
        engine=engine,
        random_state=random_state,
    )

    p = empirical_pvalue(zmax_obs, Zmax_null, kind="right")

    return p, Zmax_null


def permutation_null_pvalue(
    weights: np.ndarray,
    wedge_idx: np.ndarray,
    band_idx: np.ndarray,
    kernels: np.ndarray,
    *,
    B: int,
    A: int,
    R: int = 500,
    var_mode: Literal["binomial", "plugin"] = "binomial",
    overdispersion: float = 0.0,
    engine: Literal["fft", "prefix"] = "fft",
    batches: Optional[np.ndarray] = None,
    zmax_obs: Optional[float] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, np.ndarray]:
    """
    End-to-end helper: permutation-based p-value within bands (and optionally batches).

    Args:
        weights (np.ndarray): Cell-level weights, shape (N,).
        wedge_idx (np.ndarray): Wedge assignment for each cell, shape (N,), in [0, B-1].
        band_idx (np.ndarray): Radial band assignment for each cell, shape (N,), in [0, A-1].
        kernels (np.ndarray): Bank of J circular kernels, shape (J, B).
        B (int): Number of angular bins.
        A (int): Number of radial bands.
        R (int, optional): Number of permutation replicates. Defaults to 500.
        var_mode (Literal["binomial", "plugin"], optional): Variance model for Z.
            Defaults to "binomial".
        overdispersion (float, optional): Extra-Poisson factor for "binomial" variance.
            Defaults to 0.0.
        engine (Literal["fft", "prefix"], optional): Convolution engine. Defaults to "fft".
        batches (Optional[np.ndarray], optional): Batch assignment for each cell, shape (N,).
            If provided, permute within (band, batch) strata. Defaults to None.
        zmax_obs (Optional[float], optional): Observed max Z. If None, will compute
            Z_heat from the unpermuted input to obtain zmax_obs. Defaults to None.
        random_state (Optional[Union[int, np.random.Generator]], optional):
            Reproducible RNG seed or Generator. Defaults to None.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing:
            - p (float): Empirical (right-tail) p-value for zmax_obs.
            - Zmax_null (np.ndarray): Null distribution of max Z, shape (R,),
              returned for diagnostics.

    Notes:
        - Stronger exchangeability assumptions than rotations but can handle certain
          confounders rotations cannot. Typically slower (O(R·N)).
        - If batches is None, permutes within radial bands only.
        - If batches is provided, permutes within (band, batch) strata.
        - Use when rotation tests are inappropriate (e.g., non-uniform angular coverage).
    """
    weights = np.asarray(weights)
    wedge_idx = np.asarray(wedge_idx)
    band_idx = np.asarray(band_idx)
    kernels = np.asarray(kernels)

    N = len(weights)
    if len(wedge_idx) != N or len(band_idx) != N:
        raise ValueError(
            f"weights, wedge_idx, and band_idx must have same length. "
            f"Got {N}, {len(wedge_idx)}, {len(band_idx)}."
        )

    if batches is not None:
        batches = np.asarray(batches)
        if len(batches) != N:
            raise ValueError(
                f"batches must have same length as weights. Got {len(batches)} vs {N}."
            )

    if zmax_obs is None:
        wedge_sums = np.zeros((A, B))
        for i in range(N):
            a = band_idx[i]
            w = wedge_idx[i]
            wedge_sums[a, w] += weights[i]

        totals_per_band = np.sum(wedge_sums, axis=1)

        J, B_kern = kernels.shape
        if B_kern != B:
            raise ValueError(
                f"kernels have {B_kern} bins (axis 1), but B={B} was specified."
            )

        zmax_obs = -np.inf

        for j in range(J):
            kernel = kernels[j]
            _, _, _, Z_band = stats.compute_Z_grid(
                wedge_sums,
                totals_per_band,
                kernel,
                B=B,
                var_mode=var_mode,
                overdispersion=overdispersion,
                engine=engine,
            )

            Z = stats.aggregate_bands(
                Z_band, weights=_safe_weights(totals_per_band), method="fixed"
            )

            zmax_obs = max(zmax_obs, np.nanmax(Z))

    Zmax_null = label_permutation_within_bands(
        weights=weights,
        wedge_idx=wedge_idx,
        band_idx=band_idx,
        kernels=kernels,
        B=B,
        A=A,
        R=R,
        var_mode=var_mode,
        overdispersion=overdispersion,
        engine=engine,
        batches=batches,
        random_state=random_state,
    )

    p = empirical_pvalue(zmax_obs, Zmax_null, kind="right")

    return p, Zmax_null


def foreground_background_pvalue(
    binary_labels: np.ndarray,
    wedge_idx: np.ndarray,
    band_idx: np.ndarray,
    kernels: np.ndarray,
    *,
    B: int,
    A: int,
    R: int = 500,
    var_mode: Literal["binomial", "plugin"] = "binomial",
    overdispersion: float = 0.0,
    engine: Literal["fft", "prefix"] = "fft",
    batches: Optional[np.ndarray] = None,
    zmax_obs: Optional[float] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, np.ndarray]:
    """
    End-to-end helper: foreground vs background p-value for binary features.

    This function tests whether foreground cells (binary_labels=1) have a different
    angular distribution than background cells (binary_labels=0) by permuting the
    binary labels among cells.

    Args:
        binary_labels (np.ndarray): Binary feature values (0 or 1), shape (N,).
            1 = foreground (e.g., expressing cells), 0 = background.
        wedge_idx (np.ndarray): Wedge assignment for each cell, shape (N,), in [0, B-1].
        band_idx (np.ndarray): Radial band assignment for each cell, shape (N,), in [0, A-1].
        kernels (np.ndarray): Bank of J circular kernels, shape (J, B).
        B (int): Number of angular bins.
        A (int): Number of radial bands.
        R (int, optional): Number of permutation replicates. Defaults to 500.
        var_mode (Literal["binomial", "plugin"], optional): Variance model for Z.
            Defaults to "binomial".
        overdispersion (float, optional): Extra-Poisson factor for "binomial" variance.
            Defaults to 0.0.
        engine (Literal["fft", "prefix"], optional): Convolution engine. Defaults to "fft".
        batches (Optional[np.ndarray], optional): Batch assignment for each cell, shape (N,).
            If provided, permute within (band, batch) strata. Defaults to None.
        zmax_obs (Optional[float], optional): Observed max Z. If None, will compute
            Z_heat from the unpermuted input to obtain zmax_obs. Defaults to None.
        random_state (Optional[Union[int, np.random.Generator]], optional):
            Reproducible RNG seed or Generator. Defaults to None.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing:
            - p (float): Empirical (right-tail) p-value for zmax_obs.
            - Zmax_null (np.ndarray): Null distribution of max Z, shape (R,),
              returned for diagnostics.

    Notes:
        - This is the recommended null model for binary/binarized features.
        - Tests: "Are foreground cells more concentrated in a sector than expected
          if the foreground/background labels were randomly assigned?"
        - Permutes binary labels within strata to preserve:
          * Total number of foreground and background cells
          * Radial band membership
          * Batch structure (if provided)
        - Use this null model when threshold_mode != 'none' (i.e., for binarized features).
    """
    binary_labels = np.asarray(binary_labels)
    wedge_idx = np.asarray(wedge_idx)
    band_idx = np.asarray(band_idx)
    kernels = np.asarray(kernels)

    N = len(binary_labels)
    if len(wedge_idx) != N or len(band_idx) != N:
        raise ValueError(
            f"binary_labels, wedge_idx, and band_idx must have same length. "
            f"Got {N}, {len(wedge_idx)}, {len(band_idx)}."
        )

    if batches is not None:
        batches = np.asarray(batches)
        if len(batches) != N:
            raise ValueError(
                f"batches must have same length as binary_labels. Got {len(batches)} vs {N}."
            )

    if zmax_obs is None:
        wedge_sums = np.zeros((A, B))
        for i in range(N):
            a = band_idx[i]
            w = wedge_idx[i]
            wedge_sums[a, w] += binary_labels[i]

        totals_per_band = np.sum(wedge_sums, axis=1)

        J, B_kern = kernels.shape
        if B_kern != B:
            raise ValueError(
                f"kernels have {B_kern} bins (axis 1), but B={B} was specified."
            )

        zmax_obs = -np.inf

        for j in range(J):
            kernel = kernels[j]
            _, _, _, Z_band = stats.compute_Z_grid(
                wedge_sums,
                totals_per_band,
                kernel,
                B=B,
                var_mode=var_mode,
                overdispersion=overdispersion,
                engine=engine,
            )

            Z = stats.aggregate_bands(
                Z_band, weights=_safe_weights(totals_per_band), method="fixed"
            )

            zmax_obs = max(zmax_obs, np.nanmax(Z))

    Zmax_null = foreground_background_permutation(
        binary_labels=binary_labels,
        wedge_idx=wedge_idx,
        band_idx=band_idx,
        kernels=kernels,
        B=B,
        A=A,
        R=R,
        var_mode=var_mode,
        overdispersion=overdispersion,
        engine=engine,
        batches=batches,
        random_state=random_state,
    )

    p = empirical_pvalue(zmax_obs, Zmax_null, kind="right")

    return p, Zmax_null


def rotation_null_pvalue_fg_bg(
    wedge_idx,
    feature,
    weights=None,
    kernel=None,
    *,
    B=256,
    R=500,
    random_state=None,
):
    """
    Permutation null for foreground vs background difference test (no batch structure).

    **Null Hypothesis**: Gene expression is spatially random - any cell is equally 
    likely to express the gene, regardless of its angular position.

    **Null Model**: Randomly permute expression labels across all cells, preserving:
    - Total number of expressing cells (global expression rate)
    - Spatial positions of all cells
    
    This breaks spatial correlation between expression and angular position.

    **Use Case**: Testing if expressing cells are angularly enriched/depleted relative 
    to the overall cell distribution when there are no batch effects to account for.

    Parameters
    ----------
    wedge_idx : ndarray, shape (N,)
        Wedge bin assignment for each cell (angular position).
    feature : ndarray, shape (N,)
        Binary feature values (1 = expressing, 0 = non-expressing).
    weights : ndarray, shape (N,), optional
        Weights for each cell (typically None for binary features).
    kernel : ndarray, shape (B,), optional
        Circular kernel for smoothing the difference profile.
    B : int, default=256
        Number of angular bins.
    R : int, default=500
        Number of permutation replicates.
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.

    Returns
    -------
    p : float
        Empirical p-value (proportion of null replicates >= observed).
    diff_max_obs : float
        Observed maximum absolute difference.
    diff_max_null : ndarray, shape (R,)
        Null distribution of maximum absolute differences.

    Notes
    -----
    - This is the scientifically correct null for binary gene expression features
      when there are no batch effects.
    - For data with batch structure, use within_batch_rotation_pvalue_fg_bg instead
      to preserve batch-specific expression rates.
    - Each permutation creates a new spatial arrangement where expression is completely
      random with respect to angular position.
    
    Examples
    --------
    >>> # Test if gene expression is angularly enriched
    >>> p, obs, null = rotation_null_pvalue_fg_bg(
    ...     wedge_idx=angles,
    ...     feature=gene_binary,  # 0/1 for non-expressing/expressing
    ...     B=180,
    ...     R=500
    ... )
    """
    rng = check_random_state(random_state)

    # Compute observed difference
    diff_obs, _, _ = stats.compute_fg_bg_difference(
        wedge_idx=wedge_idx, feature=feature, weights=weights, B=B
    )

    if kernel is not None:
        diff_obs = stats.sector_sums_convolved(diff_obs[None, :], kernel)[0]

    diff_max_obs = np.max(np.abs(diff_obs))

    # Generate null distribution by permuting expression labels
    diff_max_null = np.zeros(R)

    for r in range(R):
        # Randomly permute expression labels across all cells
        perm_indices = rng.permutation(len(feature))
        feature_perm = feature[perm_indices]

        # Compute difference with permuted labels
        diff_null, _, _ = stats.compute_fg_bg_difference(
            wedge_idx=wedge_idx,  # Same positions
            feature=feature_perm,  # Permuted expression
            weights=weights,
            B=B
        )

        if kernel is not None:
            diff_null = stats.sector_sums_convolved(diff_null[None, :], kernel)[0]

        diff_max_null[r] = np.max(np.abs(diff_null))

    p = empirical_pvalue(diff_max_obs, diff_max_null, kind="right")

    return p, diff_max_obs, diff_max_null


def within_batch_rotation_pvalue_fg_bg(
    wedge_idx,
    feature,
    batches,
    weights=None,
    kernel=None,
    *,
    B=256,
    R=500,
    random_state=None,
):
    """
    Permutation null for foreground vs background difference test with batch stratification.

    **Null Hypothesis**: Gene expression is spatially random within each batch - any cell 
    within a batch is equally likely to express the gene, regardless of its angular position.

    **Null Model**: Randomly permute expression labels within each batch independently, 
    preserving:
    - Number of expressing cells per batch (batch-specific expression rate)
    - Spatial positions of all cells
    - Batch structure
    
    This breaks spatial correlation between expression and angular position while respecting
    batch-specific biological variation in expression rates.

    **Use Case**: Testing if expressing cells are angularly enriched/depleted relative to 
    the overall cell distribution, accounting for batch effects.

    Parameters
    ----------
    wedge_idx : ndarray, shape (N,)
        Wedge bin assignment for each cell (angular position).
    feature : ndarray, shape (N,)
        Binary feature values (1 = expressing, 0 = non-expressing).
    batches : ndarray, shape (N,)
        Batch labels for each cell.
    weights : ndarray, shape (N,), optional
        Weights for each cell (typically None for binary features).
    kernel : ndarray, shape (B,), optional
        Circular kernel for smoothing the difference profile.
    B : int, default=256
        Number of angular bins.
    R : int, default=500
        Number of permutation replicates.
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.

    Returns
    -------
    p : float
        Empirical p-value (proportion of null replicates >= observed).
    diff_max_obs : float
        Observed maximum absolute difference.
    diff_max_null : ndarray, shape (R,)
        Null distribution of maximum absolute differences.

    Notes
    -----
    - This is the scientifically correct null for binary gene expression features.
    - Each permutation creates a new spatial arrangement where expression is random
      conditional on batch membership.
    - Accounts for batch-specific expression rates (e.g., batch A might have 80% 
      expressing while batch B has 60%).
    - Previous rotation-based null was fundamentally flawed: rotation preserves 
      max|diff|, giving zero variance in null distribution.
    
    Examples
    --------
    >>> # Test if SLC12A1 expression is angularly enriched in TAL cells
    >>> p, obs, null = within_batch_rotation_pvalue_fg_bg(
    ...     wedge_idx=angles,
    ...     feature=slc12a1_binary,  # 0/1 for non-expressing/expressing
    ...     batches=donor_ids,
    ...     B=180,
    ...     R=500
    ... )
    """
    rng = check_random_state(random_state)

    # Compute observed difference
    diff_obs, _, _ = stats.compute_fg_bg_difference(
        wedge_idx=wedge_idx, feature=feature, weights=weights, B=B
    )

    if kernel is not None:
        diff_obs = stats.sector_sums_convolved(diff_obs[None, :], kernel)[0]

    diff_max_obs = np.max(np.abs(diff_obs))

    # Generate null distribution
    diff_max_null = np.zeros(R)
    unique_batches = np.unique(batches)

    for r in range(R):
        # Permute feature labels within each batch independently
        feature_perm = feature.copy()
        
        for batch_label in unique_batches:
            batch_mask = batches == batch_label
            batch_indices = np.where(batch_mask)[0]
            
            # Randomly permute expression labels within this batch
            perm_indices = rng.permutation(batch_indices)
            feature_perm[batch_indices] = feature[perm_indices]

        # Compute difference with permuted labels
        diff_null, _, _ = stats.compute_fg_bg_difference(
            wedge_idx=wedge_idx,  # Same positions
            feature=feature_perm,  # Permuted expression
            weights=weights,
            B=B
        )

        if kernel is not None:
            diff_null = stats.sector_sums_convolved(diff_null[None, :], kernel)[0]

        diff_max_null[r] = np.max(np.abs(diff_null))

    p = empirical_pvalue(diff_max_obs, diff_max_null, kind="right")

    return p, diff_max_obs, diff_max_null


def permutation_pvalue_fg_bg(
    wedge_idx,
    feature,
    band_idx=None,
    batches=None,
    weights=None,
    kernel=None,
    *,
    B=256,
    R=500,
    random_state=None,
):
    """
    Permutation null for foreground vs background with radial band and batch stratification.

    **Null Hypothesis**: Gene expression is spatially random within each stratum 
    (radial band × batch combination).

    **Null Model**: Randomly permute expression labels within each stratum independently,
    preserving:
    - Number of expressing cells per stratum
    - Spatial positions of all cells
    - Radial band structure
    - Batch structure
    
    This is the most conservative null, accounting for both radial and batch heterogeneity
    in expression rates.

    **Use Case**: When expression rates vary across both radial bands and batches, and
    you want to test for angular enrichment while controlling for these sources of variation.

    Parameters
    ----------
    wedge_idx : ndarray, shape (N,)
        Wedge bin assignment for each cell (angular position).
    feature : ndarray, shape (N,)
        Binary feature values (1 = expressing, 0 = non-expressing).
    band_idx : ndarray, shape (N,), optional
        Radial band assignment. If None, assumes single band (no radial stratification).
    batches : ndarray, shape (N,), optional
        Batch labels. If None, assumes single batch (no batch stratification).
    weights : ndarray, shape (N,), optional
        Weights for each cell (typically None for binary features).
    kernel : ndarray, shape (B,), optional
        Circular kernel for smoothing the difference profile.
    B : int, default=256
        Number of angular bins.
    R : int, default=500
        Number of permutation replicates.
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.

    Returns
    -------
    p : float
        Empirical p-value (proportion of null replicates >= observed).
    diff_max_obs : float
        Observed maximum absolute difference.
    diff_max_null : ndarray, shape (R,)
        Null distribution of maximum absolute differences.

    Notes
    -----
    - Most conservative option: preserves expression rate in each (band, batch) stratum.
    - Use when you suspect expression varies with both distance from vantage point
      and batch membership.
    - If only batch effects matter, use within_batch_rotation_pvalue_fg_bg instead
      (more powerful).
    - If no structure, use rotation_null_pvalue_fg_bg (most powerful).
    
    Examples
    --------
    >>> # Test with both radial and batch structure
    >>> p, obs, null = permutation_pvalue_fg_bg(
    ...     wedge_idx=angles,
    ...     feature=gene_binary,
    ...     band_idx=radial_bands,
    ...     batches=donor_ids,
    ...     B=180,
    ...     R=500
    ... )
    """
    rng = check_random_state(random_state)

    N = len(wedge_idx)

    if band_idx is None:
        band_idx = np.zeros(N, dtype=int)

    if batches is None:
        batches = np.zeros(N, dtype=int)

    # Compute observed difference
    diff_obs, _, _ = stats.compute_fg_bg_difference(
        wedge_idx=wedge_idx, feature=feature, weights=weights, B=B
    )

    if kernel is not None:
        diff_obs = stats.sector_sums_convolved(diff_obs[None, :], kernel)[0]

    diff_max_obs = np.max(np.abs(diff_obs))

    diff_max_null = np.zeros(R)

    strata = np.char.add(band_idx.astype(str), np.char.add("_", batches.astype(str)))
    unique_strata = np.unique(strata)

    for r in range(R):
        feature_perm = feature.copy()

        for stratum in unique_strata:
            mask = strata == stratum
            indices = np.where(mask)[0]
            perm_indices = rng.permutation(indices)
            feature_perm[indices] = feature[perm_indices]

        diff_null, _, _ = stats.compute_fg_bg_difference(
            wedge_idx=wedge_idx, feature=feature_perm, weights=weights, B=B
        )

        if kernel is not None:
            diff_null = stats.sector_sums_convolved(diff_null[None, :], kernel)[0]

        diff_max_null[r] = np.max(np.abs(diff_null))

    p = empirical_pvalue(diff_max_obs, diff_max_null, kind="right")

    return p, diff_max_obs, diff_max_null
