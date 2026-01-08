"""Backward compatibility shim for biorsp.radar."""

from biorsp.core.engine import compute_rsp_radar
from biorsp.core.typing import RadarResult

__all__ = ["compute_rsp_radar", "RadarResult"]
