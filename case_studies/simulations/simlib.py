"""
Compatibility shim for legacy simlib.py

DEPRECATED: This module is deprecated. Please import from the package instead:
    from simulations.simlib import generate_coords, score_dataset, etc.

The monolithic simlib.py has been refactored into a modular package structure:
    simulations/simlib/
        rng.py          - Deterministic random number generation
        shapes.py       - Coordinate generators
        distortions.py  - Transformations
        geometry.py     - Polar coordinates
        density.py      - Density estimation
        expression.py   - Expression simulation
        datasets.py     - Multi-gene panels
        scoring.py      - BioRSP API wrappers
        metrics.py      - Evaluation metrics
        plotting.py     - Visualization
        io.py           - CSV and JSON I/O
        docs.py         - Markdown reports
        sweeps.py       - Parameter sweeps

This file re-exports the new API for backward compatibility.
"""

import warnings

warnings.warn(
    "Importing from simlib.py is deprecated. Use 'from simulations.simlib import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Legacy constants
DEFAULT_DELTA = 60.0
DEFAULT_B = 72
SECONDARY_DELTA = 180.0


def get_base_config_v3():
    """Legacy helper - use BioRSPConfig directly."""
    from biorsp import BioRSPConfig

    return BioRSPConfig(
        B=DEFAULT_B,
        delta_deg=DEFAULT_DELTA,
        n_permutations=250,
        qc_mode="principled",
    )
