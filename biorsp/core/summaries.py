"""
Scalar summaries module for BioRSP.

Implements scalar statistics derived from the radar function:
- Peak distal (rim) and peak proximal directions.
- RMS anisotropy.
- Extremal peak based on absolute magnitude.
- Localization index (Fix 5.1).
- Signed summaries (Fix 5.2):
    - R_mean indicates net radial bias (core vs rim).
    - Polarity indicates whether the pattern is globally one-signed or mixed.
    - These resolve ambiguity in A_g where rim and core patterns can have similar magnitude.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from biorsp.core.typing import RadarResult
from biorsp.utils.stats import compute_localization, compute_signed_summaries


@dataclass
class ScalarSummaries:
    """
    Scalar summary statistics for a gene's RSP.

    Attributes:
        peak_distal: Minimum RSP value (rim-enriched).
        peak_distal_angle: Angle corresponding to peak_distal.
        peak_proximal: Maximum RSP value (core-enriched).
        peak_proximal_angle: Angle corresponding to peak_proximal.
        peak_extremal: RSP value with maximum absolute magnitude.
        peak_extremal_angle: Angle corresponding to peak_extremal.
        anisotropy: RMS of RSP values (A_g).
        max_rsp: Maximum RSP value (for diagnostics).
        min_rsp: Minimum RSP value (same as peak_distal).
        integrated_rsp: Sum of RSP values (net directionality).
        localization_entropy: Shannon entropy-based localization index (L_g).
        localization_gini: Gini-based localization index.
        m_valid_sectors: Number of valid sectors used for computation.
        sum_abs_rsp: Sum of absolute RSP values.
        localization_status: Status of localization computation (e.g., 'ok', 'no_signal').
        r_mean: Mean signed shift (net radial bias).
        r_median: Median signed shift.
        polarity: Signed energy ratio (one-signed vs mixed).
        a_signed: Signed anisotropy (sign(r_mean) * anisotropy).
        frac_pos: Fraction of sectors with positive RSP.
        frac_neg: Fraction of sectors with negative RSP.
        signed_status: Status of signed summary computation.
    """

    peak_distal: float
    peak_distal_angle: float
    peak_proximal: float
    peak_proximal_angle: float
    peak_extremal: float
    peak_extremal_angle: float
    anisotropy: float
    max_rsp: float
    min_rsp: float
    integrated_rsp: float
    localization_entropy: float
    localization_gini: float
    m_valid_sectors: int
    sum_abs_rsp: float
    localization_status: str
    r_mean: float
    r_median: float
    polarity: float
    a_signed: float
    frac_pos: float
    frac_neg: float
    signed_status: str

    @property
    def rms_anisotropy(self) -> float:
        """
        Backward compatible alias for anisotropy.

        Note: Anisotropy (A_g) measures the magnitude of spatial patterning but
        can conflate global shifts (e.g., rim/core) with localized patterns (e.g., wedges).
        Use localization_entropy (L_g) to distinguish these phenotypes, and
        r_mean or polarity to distinguish core vs rim bias.
        """
        return self.anisotropy


def compute_scalar_summaries(
    radar: RadarResult, valid_mask: Optional[np.ndarray] = None
) -> ScalarSummaries:
    """
    Compute scalar summaries from radar result.
    Ignores NaN values (underpowered sectors) unless a mask is provided.

    Args:
        radar: RadarResult object containing rsp values and centers.
        valid_mask: Optional boolean mask to define sectors to include.

    Returns:
        ScalarSummaries object.
    """
    rsp = radar.rsp
    centers = radar.centers

    # Filter NaNs
    if valid_mask is None:
        valid_mask = ~np.isnan(rsp)

    if not np.any(valid_mask):
        # Handle case with no valid sectors
        # Return NaNs instead of zeros to avoid bias
        return ScalarSummaries(
            peak_distal=np.nan,
            peak_distal_angle=np.nan,
            peak_proximal=np.nan,
            peak_proximal_angle=np.nan,
            peak_extremal=np.nan,
            peak_extremal_angle=np.nan,
            anisotropy=np.nan,
            max_rsp=np.nan,
            min_rsp=np.nan,
            integrated_rsp=np.nan,
            localization_entropy=np.nan,
            localization_gini=np.nan,
            m_valid_sectors=0,
            sum_abs_rsp=0.0,
            localization_status="no_valid_sectors",
            r_mean=np.nan,
            r_median=np.nan,
            polarity=np.nan,
            a_signed=np.nan,
            frac_pos=0.0,
            frac_neg=0.0,
            signed_status="no_valid_sectors",
        )

    valid_rsp = rsp[valid_mask]
    valid_centers = centers[valid_mask]

    # Min/Max
    # P_g = min(R_g)
    min_idx = np.argmin(valid_rsp)
    max_idx = np.argmax(valid_rsp)

    min_rsp = valid_rsp[min_idx]
    max_rsp = valid_rsp[max_idx]

    peak_distal = min_rsp
    peak_distal_angle = valid_centers[min_idx]
    peak_proximal = max_rsp
    peak_proximal_angle = valid_centers[max_idx]

    extremal_idx = np.argmax(np.abs(valid_rsp))
    peak_extremal = valid_rsp[extremal_idx]
    peak_extremal_angle = valid_centers[extremal_idx]

    # A_g = RMS
    anisotropy = np.sqrt(np.mean(valid_rsp**2))

    # Integrated (Sum)
    integrated_rsp = np.sum(valid_rsp)

    # Localization
    # Note: This is computed on the same profile version used for anisotropy (A_g).
    # If support-weighting is enabled in the config, it is applied to the profile
    # before this function is called, so localization reflects the final reported profile.
    l_entropy, info = compute_localization(rsp, valid_mask=valid_mask, method="entropy")
    l_gini = info["gini"]

    # Signed summaries
    # Also computed on the final reported profile.
    signed = compute_signed_summaries(rsp, valid_mask=valid_mask)

    return ScalarSummaries(
        peak_distal=float(peak_distal),
        peak_distal_angle=float(peak_distal_angle),
        peak_proximal=float(peak_proximal),
        peak_proximal_angle=float(peak_proximal_angle),
        peak_extremal=float(peak_extremal),
        peak_extremal_angle=float(peak_extremal_angle),
        anisotropy=float(anisotropy),
        max_rsp=float(max_rsp),
        min_rsp=float(min_rsp),
        integrated_rsp=float(integrated_rsp),
        localization_entropy=float(l_entropy),
        localization_gini=float(l_gini),
        m_valid_sectors=int(info["M"]),
        sum_abs_rsp=float(info["sum_abs"]),
        localization_status=str(info["status"]),
        r_mean=float(signed["R_mean"]),
        r_median=float(signed["R_median"]),
        polarity=float(signed["polarity"]),
        a_signed=float(signed["A_signed"]),
        frac_pos=float(signed["frac_pos"]),
        frac_neg=float(signed["frac_neg"]),
        signed_status=str(signed["status"]),
    )


__all__ = ["ScalarSummaries", "compute_scalar_summaries"]
