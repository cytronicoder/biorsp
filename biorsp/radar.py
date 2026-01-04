"""
Backward compatibility shim for biorsp.radar.
"""

from .core import compute_rsp_radar
from .typing import RadarResult

__all__ = ["compute_rsp_radar", "RadarResult"]
