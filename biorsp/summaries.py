"""
Scalar summaries module for BioRSP.

Implements scalar statistics derived from the radar function:
- Peak Strength (P_g): Minimum RSP value (most negative).
- RMS Anisotropy (A_g): Root Mean Square of RSP values.
- Peak Angle (theta_g*): Angle corresponding to P_g.
"""

from dataclasses import dataclass

import numpy as np

from .radar import RadarResult


@dataclass
class ScalarSummaries:
    """
    Scalar summary statistics for a gene's RSP.

    Attributes:
        peak_strength: Minimum RSP value (P_g).
        rms_anisotropy: RMS of RSP values (A_g).
        peak_angle: Angle corresponding to peak_strength (theta_g*).
        max_rsp: Maximum RSP value (for diagnostics).
        min_rsp: Minimum RSP value (same as peak_strength).
        integrated_rsp: Sum of RSP values (net directionality).
    """

    peak_strength: float
    rms_anisotropy: float
    peak_angle: float
    max_rsp: float
    min_rsp: float
    integrated_rsp: float


def compute_scalar_summaries(radar: RadarResult) -> ScalarSummaries:
    """
    Compute scalar summaries from radar result.
    Ignores NaN values (underpowered sectors).

    Args:
        radar: RadarResult object containing rsp values and centers.

    Returns:
        ScalarSummaries object.
    """
    rsp = radar.rsp
    centers = radar.centers

    # Filter NaNs
    valid_mask = ~np.isnan(rsp)

    if not np.any(valid_mask):
        # Handle case with no valid sectors
        # Return NaNs instead of zeros to avoid bias
        return ScalarSummaries(
            peak_strength=np.nan,
            rms_anisotropy=np.nan,
            peak_angle=np.nan,
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

    peak_strength = min_rsp
    peak_angle = valid_centers[min_idx]

    # A_g = RMS
    rms_anisotropy = np.sqrt(np.mean(valid_rsp**2))

    # Integrated (Sum)
    integrated_rsp = np.sum(valid_rsp)

    return ScalarSummaries(
        peak_strength=float(peak_strength),
        rms_anisotropy=float(rms_anisotropy),
        peak_angle=float(peak_angle),
        max_rsp=float(max_rsp),
        min_rsp=float(min_rsp),
        integrated_rsp=float(integrated_rsp),
    )


__all__ = ["ScalarSummaries", "compute_scalar_summaries"]
