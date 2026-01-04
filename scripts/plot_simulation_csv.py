"""Small wrapper CLI that calls `case_studies.simulations.plot_from_csv` to produce figures
from existing CSV tables.

Example:
    python3 scripts/plot_simulation_csv.py --input-dir results/simulations_phase3 --outdir results/simulations_phase3/figures_from_csv --which all
"""

from case_studies.simulations import plot_from_csv

if __name__ == "__main__":
    plot_from_csv.main()
