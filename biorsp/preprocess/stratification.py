"""Stratification logic for geometry-aware permutations in BioRSP.

Statistical Justification:
-------------------------
To ensure calibrated p-values, the null model must respect the exchangeability
of labels conditional on the spatial geometry. In spatial transcriptomics,
cell density and radial structure (e.g., distance from center) often induce
spurious anisotropy if not accounted for.

1. Exchangeability: By permuting foreground labels only within strata defined
   by radius (r) and angle (theta), we preserve the local spatial density
   and radial distribution of cells, ensuring that the null distribution
   reflects geometry-induced artifacts rather than true spatial patterning.
2. Fixed Sector Mask: Freezing the adequacy mask (Θ_g) across permutations
   prevents bias introduced by varying the set of valid sectors. If a
   permutation results in an empty sector that was valid in the observed data,
   we assign it a safe value (0.0) to maintain a consistent RMS calculation.
3. Finite Correction: We use the (1 + count(A_perm >= A_obs)) / (K + 1)
   formula to ensure that p-values are strictly positive and valid for
   finite permutation tests.
"""

import logging
from typing import List, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


def get_strata_indices(
    r: np.ndarray,
    theta: np.ndarray,
    umi_counts: Optional[np.ndarray] = None,
    donor_labels: Optional[np.ndarray] = None,
    n_r_bins: int = 10,
    n_theta_bins: int = 4,
    n_umi_bins: int = 10,
    min_stratum_size: int = 50,
    mode: Literal["radial", "joint", "rt_umi", "none"] = "radial",
) -> List[np.ndarray]:
    """Compute strata indices for permutations.

    Parameters
    ----------
    r : np.ndarray
        (N,) array of radial distances.
    theta : np.ndarray
        (N,) array of angles in radians.
    umi_counts : np.ndarray, optional
        (N,) array of UMI counts.
    donor_labels : np.ndarray, optional
        (N,) array of donor labels for mandatory stratification.
    n_r_bins : int
        Number of radial bins.
    n_theta_bins : int
        Number of angular bins.
    n_umi_bins : int
        Number of UMI bins.
    min_stratum_size : int
        Minimum number of cells per stratum.
    mode : str
        Stratification mode: 'radial', 'joint', 'rt_umi', or 'none'.

    Returns
    -------
    List[np.ndarray]
        List of arrays, each containing indices for one stratum.

    """
    n_cells = len(r)
    if mode == "none" or n_cells == 0:
        return [np.arange(n_cells)]

    # If donor labels are provided, partition by donor first
    if donor_labels is not None:
        unique_donors = np.unique(donor_labels)
        all_strata = []
        for donor in unique_donors:
            donor_mask = donor_labels == donor
            donor_idx = np.where(donor_mask)[0]
            if donor_idx.size == 0:
                continue

            # Recursive call or sub-logic for this donor
            donor_strata = get_strata_indices(
                r=r[donor_idx],
                theta=theta[donor_idx],
                umi_counts=umi_counts[donor_idx] if umi_counts is not None else None,
                donor_labels=None,  # Already partitioned
                n_r_bins=n_r_bins,
                n_theta_bins=n_theta_bins,
                n_umi_bins=n_umi_bins,
                min_stratum_size=min_stratum_size,
                mode=mode,
            )
            # Map back to global indices
            all_strata.extend(donor_idx[s] for s in donor_strata)
        return all_strata

    r_bins = (np.argsort(np.argsort(r)) * n_r_bins) // n_cells

    if mode in ["joint", "rt_umi"]:
        theta_norm = theta % (2 * np.pi)
        theta_bins = (np.argsort(np.argsort(theta_norm)) * n_theta_bins) // n_cells
    else:
        theta_bins = np.zeros(n_cells, dtype=int)

    if mode == "rt_umi" and umi_counts is not None:
        umi_bins = (np.argsort(np.argsort(umi_counts)) * n_umi_bins) // n_cells
    else:
        umi_bins = np.zeros(n_cells, dtype=int)

    strata_ids = r_bins + n_r_bins * (theta_bins + n_theta_bins * umi_bins)

    unique_ids, counts = np.unique(strata_ids, return_counts=True)
    strata_map = {sid: np.where(strata_ids == sid)[0] for sid in unique_ids}

    sorted_ids = np.sort(unique_ids)
    final_strata = []
    current_stratum = []

    for sid in sorted_ids:
        idx = strata_map[sid]
        if len(current_stratum) == 0:
            current_stratum = idx
        else:
            current_stratum = np.concatenate([current_stratum, idx])

        if len(current_stratum) >= min_stratum_size:
            final_strata.append(current_stratum)
            current_stratum = []

    if len(current_stratum) > 0:
        if len(final_strata) > 0:
            final_strata[-1] = np.concatenate([final_strata[-1], current_stratum])
        else:
            if len(current_stratum) > 0:
                logger.warning(
                    f"Only one stratum found with size {len(current_stratum)}, "
                    f"which is less than min_stratum_size {min_stratum_size}."
                )
                final_strata.append(current_stratum)

    return final_strata
