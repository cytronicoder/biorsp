"""
Inference module for BioRSP.

Implements permutation testing for statistical significance:
- Null hypothesis: Gene expression is independent of spatial location (angle).
- Test statistic: RMS Anisotropy (A_g).
- Stratified permutation of labels (optional) to control for UMI count confounders.
"""

import concurrent.futures
from typing import Optional, Tuple

import numpy as np

from .constants import N_BG_MIN_DEFAULT, N_FG_MIN_DEFAULT, UMI_BINS_DEFAULT
from .radar import compute_rsp_radar


def _rms_with_mask(rsp: np.ndarray, valid_mask: np.ndarray) -> float:
    """Compute RMS anisotropy using a fixed sector mask.

    Notes:
        - If any sector in `valid_mask` contains NaN in `rsp`, return NaN to indicate
          the statistic is invalid for that permutation. This prevents shrinking the
          null distribution by treating missing sectors as zeros.
    """
    masked_rsp = rsp[valid_mask]

    # If any NaNs occur in the masked regions, mark the statistic invalid
    if np.isnan(masked_rsp).any():
        return np.nan

    if masked_rsp.size == 0:
        return np.nan

    return float(np.sqrt(np.mean(masked_rsp**2)))


def _compute_permutation_stat(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    strata_indices: list,
    B: int,
    delta_deg: float,
    min_fg_sector: int,
    min_bg_sector: int,
    seed: int,
    valid_mask: np.ndarray,
) -> float:
    """Compute the test statistic for a single permutation."""
    rng = np.random.default_rng(seed)
    y_perm = y.copy()
    for idx in strata_indices:
        y_subset = y_perm[idx]
        rng.shuffle(y_subset)
        y_perm[idx] = y_subset

    radar_perm = compute_rsp_radar(r, theta, y_perm, B, delta_deg, min_fg_sector, min_bg_sector)
    return _rms_with_mask(radar_perm.rsp, valid_mask)


def compute_p_value(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    B: int = 360,
    delta_deg: float = 20.0,
    n_perm: int = 1000,
    umi_counts: Optional[np.ndarray] = None,
    umi_bins: int = UMI_BINS_DEFAULT,
    seed: int = 42,
    min_fg_sector: int = N_FG_MIN_DEFAULT,
    min_bg_sector: int = N_BG_MIN_DEFAULT,
) -> Tuple[float, np.ndarray, float]:
    """
    Compute p-value for the observed anisotropy using permutation test.
    Foreground labels are permuted within UMI strata to keep geometry fixed.

    Args:
        r: (N,) array of radial distances.
        theta: (N,) array of angles for all cells.
        y: (N,) boolean foreground indicator.
        B: Number of sectors.
        delta_deg: Sector width.
        n_perm: Number of permutations (number of *valid* nulls desired).
        umi_counts: (N,) array of UMI counts for stratification (optional).
        umi_bins: Number of bins for UMI stratification.
        seed: Random seed; different seeds are consumed per trial.
        min_fg_sector: Minimum foreground counts per sector.
        min_bg_sector: Minimum background counts per sector.

    Returns:
        p_value: Estimated p-value (NaN if insufficient valid permutations collected).
        null_stats: (n_perm,) array of valid null statistics (padded with NaN if insufficient).
        observed_stat: Observed statistic recomputed from data.
    """
    # Prepare output buffer
    null_stats = np.full(n_perm, np.nan)

    n_cells = len(theta)

    # Stratification logic
    if umi_counts is not None:
        # Create deterministic strata using ranks
        if n_cells < umi_bins:
            strata = np.zeros(n_cells, dtype=int)
        else:
            ranks = np.argsort(np.argsort(umi_counts))  # dense ranks 0..N-1
            strata = (ranks * umi_bins) // n_cells
    else:
        strata = np.zeros(n_cells, dtype=int)

    unique_strata = np.unique(strata)

    # Pre-calculate indices for each stratum to speed up shuffling
    strata_indices = [np.where(strata == s)[0] for s in unique_strata]

    # Observed radar and valid mask
    radar_obs = compute_rsp_radar(r, theta, y, B, delta_deg, min_fg_sector, min_bg_sector)
    valid_mask = ~np.isnan(radar_obs.rsp)
    if not np.any(valid_mask):
        return np.nan, null_stats, np.nan

    observed_stat = _rms_with_mask(radar_obs.rsp, valid_mask)

    # Parallelized sampling loop: keep drawing trials until we collect n_perm valid stats
    import logging

    logger = logging.getLogger(__name__)

    # Reduce worker count to avoid oversubscription on systems with heavy BLAS/OpenMP usage
    max_workers = min(4, n_perm)

    # Cap the number of attempts to a reasonable multiplier to avoid very long hangs
    max_attempts = max(3 * n_perm, n_perm + 100)

    collected = 0
    attempts = 0
    seed_counter = seed

    logger.debug(
        f"compute_p_value start: n_perm={n_perm}, max_workers={max_workers}, max_attempts={max_attempts}"
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while collected < n_perm and attempts < max_attempts:
            # Submit a batch of jobs (bounded by remaining needed and workers)
            batch_size = min(max_workers, n_perm - collected)
            start_seed = seed_counter
            # Map each future to its absolute trial id (seed number) for correct bookkeeping
            futures = {
                executor.submit(
                    _compute_permutation_stat,
                    r,
                    theta,
                    y,
                    strata_indices,
                    B,
                    delta_deg,
                    min_fg_sector,
                    min_bg_sector,
                    seed_counter + i,
                    valid_mask,
                ): (seed_counter + i)
                for i in range(batch_size)
            }
            seed_counter += batch_size
            logger.debug(
                f"Submitted permutation batch seeds {start_seed}..{seed_counter-1} (batch_size={batch_size})"
            )

            # Collect results as they finish
            for future in concurrent.futures.as_completed(futures):
                trial_id = futures[future]
                attempts += 1
                try:
                    stat = future.result()
                except Exception as e:
                    logger.debug(f"Permutation trial {trial_id} failed: {e}")
                    stat = np.nan

                # Preserve the mapping between trial seed and statistic in logs (debug-friendly)
                # Accept only finite statistics as valid samples
                if np.isfinite(stat):
                    null_stats[collected] = stat
                    collected += 1

                logger.debug(f"Permutation progress: attempts={attempts}, collected={collected}")

                # Stop early if we have enough
                if collected >= n_perm:
                    break

            # If we are making no progress (many attempts with few valid samples), break early
            if attempts > 0 and collected == 0 and attempts >= 5 * n_perm:
                logger.debug(
                    "Aborting permutation loop early: no valid nulls collected after many attempts"
                )
                break

    # If we didn't collect enough valid nulls, return NaN p-value to indicate failure
    if collected < n_perm:
        return np.nan, null_stats, observed_stat

    # Compute p-value using only the collected valid nulls
    valid_nulls = null_stats[:n_perm]
    p_value = (np.sum(valid_nulls >= observed_stat) + 1) / (len(valid_nulls) + 1)

    return p_value, null_stats, observed_stat


__all__ = ["compute_p_value"]
