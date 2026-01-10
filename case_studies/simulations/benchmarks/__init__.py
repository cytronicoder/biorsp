"""
Simulation benchmarks for BioRSP validation and methods comparison.

This package contains comprehensive benchmark scripts for validating BioRSP:

Core Benchmarks (original):
- run_calibration.py: Type I error calibration and threshold derivation
- run_archetypes.py: Archetype recovery with known ground truth
- run_genegene.py: Gene-gene co-patterning discovery
- run_robustness.py: Robustness under distortions and stress conditions

Methods Paper Benchmarks (story figure and supporting analysis):
- run_story_onepager.py: Main one-page validation figure with 4 panels
- run_null_calibration.py: Null-derived threshold calibration
- run_stability.py: Cross-embedding stability analysis
- run_abstention.py: Failure mode evaluation under extreme conditions
- smoke_benchmarks.py: Smoke test for all benchmark scripts

See README.md for full documentation.
"""

__version__ = "1.0.0"
