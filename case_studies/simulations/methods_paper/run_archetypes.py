"""
Archetype Recovery Benchmark for BioRSP Methods Paper.

Evaluates the method's ability to detect and distinguish diverse spatial archetypes.
1. Rim (Niche)
2. Core (Niche)
3. Wedge (Program)
4. Mixed (Wedge + Rim)
5. Uniform (Negative Control)
6. Sparse (Negative Control)

Metrics: Spatial Score (S), Coverage (C), Polarity.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from case_studies.simulations import simlib


def run_archetype_benchmark(output_file: str, n_reps: int = 50):
    shapes = ["disk", "peanut", "crescent"]
    archetypes = ["uniform", "sparse", "rim", "core", "wedge", "wedge_rim", "two_wedges"]

    n_cells = 3000
    results = []
    config = simlib.get_base_config_v3()

    total_iters = len(shapes) * len(archetypes) * n_reps
    pbar = tqdm(total=total_iters, desc="Archetype Benchmark")

    seed_counter = 5000

    for shape in shapes:
        for arch in archetypes:
            for rep in range(n_reps):
                seed_counter += 1

                coords = simlib.generate_coords(shape, n_cells, seed=seed_counter)

                # Different params for specific archetypes if needed
                params = {}
                if arch == "wedge":
                    params = {"width_rad": np.pi / 6}

                prob_map = simlib.generate_true_probability(coords, arch, params)
                counts, libs, _ = simlib.generate_expression(
                    coords, prob_map, mode="nb", seed=seed_counter
                )

                res = simlib.score_with_biorsp(
                    coords,
                    counts,
                    libs,
                    gene_name=f"{arch}_{rep}",
                    config=config,
                    seed=seed_counter,
                )

                res.update({"shape": shape, "archetype": arch, "rep": rep})
                results.append(res)
                pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="archetype_results.csv")
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()

    run_archetype_benchmark(args.output, args.reps)
