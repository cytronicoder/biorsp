"""
Project constants for BioRSP.

Single source of truth for Methods defaults and numeric stabilizers.
"""

import numpy as np

# Grid and Sector defaults
B_DEFAULT = 360
DELTA_DEG_DEFAULT = 180.0
DELTA_RAD_DEFAULT = np.deg2rad(DELTA_DEG_DEFAULT)
SMOOTH_DEG_DEFAULT = 5.0  # Visualization only

# Adequacy defaults
N_FG_MIN_DEFAULT = 10
N_BG_MIN_DEFAULT = 50
N_FG_TOT_MIN_DEFAULT = 100
ADEQUACY_FRACTION_DEFAULT = 0.9
N_FG_DISTINCT_MIN_DEFAULT = 3
N_FG_RANK_MIN_DEFAULT = 3

# Inference defaults
UMI_BINS_DEFAULT = 10  # Q=10
K_EXPLORATORY_DEFAULT = 200
K_FINAL_DEFAULT = 1000
IQR_FLOOR_DEFAULT = 0.1

# Numeric stabilizers
EPS = 1e-8

# Reason codes for adequacy
REASON_SECTOR_FG_TOO_SMALL = "sector_fg_too_small"
REASON_SECTOR_BG_TOO_SMALL = "sector_bg_too_small"
REASON_SECTOR_MIXED_TOO_SMALL = "sector_mixed_too_small"
REASON_GENE_UNDERPOWERED = "gene_underpowered"
REASON_GENE_UNIDENTIFIABLE = "gene_unidentifiable"
REASON_OK = "ok"

__all__ = [
    "B_DEFAULT",
    "DELTA_DEG_DEFAULT",
    "DELTA_RAD_DEFAULT",
    "SMOOTH_DEG_DEFAULT",
    "N_FG_MIN_DEFAULT",
    "N_BG_MIN_DEFAULT",
    "N_FG_TOT_MIN_DEFAULT",
    "ADEQUACY_FRACTION_DEFAULT",
    "N_FG_DISTINCT_MIN_DEFAULT",
    "N_FG_RANK_MIN_DEFAULT",
    "UMI_BINS_DEFAULT",
    "K_EXPLORATORY_DEFAULT",
    "K_FINAL_DEFAULT",
    "EPS",
    "REASON_SECTOR_FG_TOO_SMALL",
    "REASON_SECTOR_BG_TOO_SMALL",
    "REASON_SECTOR_MIXED_TOO_SMALL",
    "REASON_GENE_UNDERPOWERED",
    "REASON_GENE_UNIDENTIFIABLE",
    "REASON_OK",
]
