"""
Scalar summaries module for BioRSP.

Implements scalar statistics derived from the radar function:
- Max/Min RSP
- Mean Absolute RSP
- Peak/Trough angles
"""

from dataclasses import dataclass

import numpy as np

from .radar import RadarResult


@dataclass
class ScalarSummaries:
    """
    Scalar summary statistics for a gene's RSP.

    Attributes:
        max_rsp: Maximum RSP value (peak strength).
        min_rsp: Minimum RSP value (trough strength).
        mean_abs_rsp: Mean absolute RSP value (overall anisotropy).
        peak_angle: Angle corresponding to max RSP.
        trough_angle: Angle corresponding to min RSP.
        integrated_rsp: Sum of RSP values (net directionality).
    """

    max_rsp: float
    min_rsp: float
    mean_abs_rsp: float
    peak_angle: float
    trough_angle: float
    integrated_rsp: float


def compute_scalar_summaries(radar: RadarResult) -> ScalarSummaries:
    """
    Compute scalar summaries from radar result.

    Args:
        radar: RadarResult object containing rsp values and centers.

    Returns:
        ScalarSummaries object.
    """
    rsp = radar.rsp
    centers = radar.centers

    # Max/Min
    max_idx = np.argmax(rsp)
    min_idx = np.argmin(rsp)

    max_rsp = rsp[max_idx]
    min_rsp = rsp[min_idx]

    peak_angle = centers[max_idx]
    trough_angle = centers[min_idx]

    # Mean Absolute
    mean_abs_rsp = np.mean(np.abs(rsp))

    # Integrated (Sum)
    integrated_rsp = np.sum(rsp)

    return ScalarSummaries(
        max_rsp=float(max_rsp),
        min_rsp=float(min_rsp),
        mean_abs_rsp=float(mean_abs_rsp),
        peak_angle=float(peak_angle),
        trough_angle=float(trough_angle),
        integrated_rsp=float(integrated_rsp),
    )


__all__ = ["ScalarSummaries", "compute_scalar_summaries"]
