"""Plot simulation results from CSV files with robust schema validation.

This script provides a command-line interface for plotting various types of
simulation results. It validates CSV schemas before plotting and provides
actionable error messages if data is malformed or empty after filtering.

Usage:
    python scripts/plot_simulation_csv.py --input-dir results/sim --outdir scripts/output --which all

    python scripts/plot_simulation_csv.py --input-dir results/sim --outdir scripts/output --which calibration,power

    python scripts/plot_simulation_csv.py results/sim/calibration.csv scripts/output/out --plot-type calibration

Requires:
    Package installation: pip install -e .

Note: This is a thin wrapper around analysis.benchmarks.plot_benchmarks.
      The validation and plotting logic is delegated to that module.
"""

import sys

try:
    from analysis.benchmarks import plot_benchmarks
except ImportError as e:
    print("ERROR: Cannot import analysis.benchmarks module.")
    print("This module should exist in the analysis/ directory.")
    print(f"Details: {e}")
    print("\nPlease ensure the package is installed: pip install -e .")
    sys.exit(1)

if __name__ == "__main__":
    plot_benchmarks.main()
