"""
Scalar summaries module for BioRSP.

Implements scalar statistics derived from the radar function:
- Peak distal (rim) and peak proximal directions.
- RMS anisotropy.
- Extremal peak based on absolute magnitude.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .typing import RadarResult


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

    @property
    def rms_anisotropy(self) -> float:
        """Backward compatible alias for anisotropy."""
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
    )


__all__ = ["ScalarSummaries", "compute_scalar_summaries"]
