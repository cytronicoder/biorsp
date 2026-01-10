"""
Simulation benchmarks for BioRSP validation and methods comparison.

This package contains four main benchmark scripts:
- run_calibration.py: Statistical calibration tests
- run_archetypes.py: Validation against ground-truth patterns
- run_genegene.py: Gene-gene co-patterns discovery
- run_robustness.py: Sensitivity analysis across parameter variations
"""

from . import run_archetypes, run_calibration, run_genegene, run_robustness

__all__ = ["run_archetypes", "run_calibration", "run_genegene", "run_robustness"]
