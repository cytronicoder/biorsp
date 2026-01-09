#!/usr/bin/env python3
"""Plot simulation results from CSV files with robust schema validation.

This script provides a command-line interface for plotting various types of
simulation results. It validates CSV schemas before plotting and provides
actionable error messages if data is malformed or empty after filtering.

Usage:
    # Plot all available figures
    python scripts/plot_simulation_csv.py --input-dir results/sim --outdir figures --which all

    # Plot specific types
    python scripts/plot_simulation_csv.py --input-dir results/sim --outdir figures --which calibration,power

    # Legacy mode (single CSV)
    python scripts/plot_simulation_csv.py results/sim/calibration.csv figures/out --plot_type calibration

Requires:
    Package installation: pip install -e .

Note: This is a thin wrapper around case_studies.simulations.plot_benchmarks.
      The validation and plotting logic is delegated to that module.
"""

import sys

try:
    from case_studies.simulations import plot_benchmarks
except ImportError as e:
    print("ERROR: Cannot import case_studies.simulations module.")
    print("This module should exist in the case_studies/ directory.")
    print(f"Details: {e}")
    print("\nPlease ensure the package is installed: pip install -e .")
    sys.exit(1)

if __name__ == "__main__":
    plot_benchmarks.main()
