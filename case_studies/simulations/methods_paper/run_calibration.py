"""
Calibration Benchmark for BioRSP Methods Paper.

This script evaluates type I error control under various null hypotheses:
1. I.I.D. Null (Standard)
2. Depth Confounded (Library size correlates with spatial position)
3. Density Confounded (Expression probability correlates with cell density)

We expect uniform p-value distribution (P(p < alpha) ~= alpha) for well-calibrated methods.
"""

import argparse
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from case_studies.simulations import simlib


def run_calibration_benchmark(output_file: str, n_reps: int = 200):
    shapes = ["disk", "annulus", "peanut", "disconnected"]
    confounders = ["none", "depth_radial", "depth_angular", "density"]
    n_cells_list = [500, 2000, 5000]

    results = []

    config = simlib.get_base_config_v3()

    total_iters = len(shapes) * len(confounders) * len(n_cells_list) * n_reps
    pbar = tqdm(total=total_iters, desc="Calibration Benchmark")

    seed_counter = 42

    for shape in shapes:
        for n_cells in n_cells_list:
            # Generate geometry once per shape/size combo to save time?
            # Ideally regenerate to be robust.

            for rep in range(n_reps):
                # New geometry every time to marginalize over geometry
                coords = simlib.generate_coords(shape, n_cells, seed=seed_counter, params={})

                for confound in confounders:
                    seed_counter += 1

                    # 1. Null Model: Uniform probability map
                    # 'uniform' archetype means no spatial pattern in probability
                    prob_map = simlib.generate_true_probability(
                        coords, "uniform", params={"p_base": 0.1}
                    )

                    # 2. Simulate Expression (Negative Binomial)
                    # For density confounder, 'generate_expression' handles the correlation
                    counts, libs, _ = simlib.generate_expression(
                        coords,
                        prob_map,
                        mode="nb",
                        seed=seed_counter,
                        params={"confounder": confound, "mean_lib": 1500},
                    )

                    # 3. Score
                    t0 = time.time()
                    res = simlib.score_with_biorsp(
                        coords,
                        counts,
                        libs,
                        gene_name=f"null_{confound}_{rep}",
                        config=config,
                        seed=seed_counter,
                    )
                    elapsed = time.time() - t0

                    res.update(
                        {
                            "shape": shape,
                            "n_points": n_cells,
                            "confounder": confound,
                            "rep": rep,
                            "time": elapsed,
                        }
                    )
                    results.append(res)
                    pbar.update(1)

            # Intermediate save
            if rep % 10 == 0:
                pd.DataFrame(results).to_csv(output_file, index=False)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="calibration_results.csv")
    parser.add_argument("--reps", type=int, default=100)
    args = parser.parse_args()

    run_calibration_benchmark(args.output, args.reps)
