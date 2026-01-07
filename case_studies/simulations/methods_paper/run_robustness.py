"""
Robustness Benchmark for BioRSP Methods Paper.

Evaluates the stability of BioRSP metrics (Spatial Score S, Coverage C) under:
1. Rotation (Should be invariant for S, C)
2. Anisotropic Scaling (Stretching) - Should degrade S slightly or change direction
3. Jitter (Positional noise) - Should be robust up to a point
4. Subsampling (Downsampling cells) - Should be robust until sparse
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from case_studies.simulations import simlib


def run_robustness_benchmark(output_file: str, n_reps: int = 50):
    # Base configuration
    base_shape = "disk"
    base_n = 2000
    archetype = "wedge"  # Strong directional signal

    distortions = {
        "rotation": [0, 15, 45, 90, 180],
        "stretch": [1.0, 1.2, 1.5, 2.0, 3.0],
        "jitter": [0.0, 0.02, 0.05, 0.1, 0.2],
        "subsample": [1.0, 0.8, 0.5, 0.2, 0.1],
        "swirl": [0, 1, 2, 5],
    }

    results = []
    config = simlib.get_base_config_v3()

    # Pre-compute ground truth for undistorted
    # Actually we compare distorted vs undistorted for EACH rep

    total_iters = sum(len(v) for v in distortions.values()) * n_reps
    pbar = tqdm(total=total_iters, desc="Robustness Benchmark")

    seed_counter = 1000

    for rep in range(n_reps):
        seed_counter += 1

        # 1. Base Geometry and Signal
        # We define PROBABILITY on the UNDISTORTED coordinates
        # Then we distort coordinates but keep expression fixed
        # (simulating tissue deformation after expression)
        # OR we generate coords, distort them, then simulate expression?
        # "Robustness to distortion" usually means: Real biology is X, measurement is distorted X'.
        # So we generate True Coords -> True Expression. Then apply Distortion to Coords.

        coords_true = simlib.generate_coords(base_shape, base_n, seed=seed_counter)
        prob_map = simlib.generate_true_probability(coords_true, archetype, params={})
        counts, libs, _ = simlib.generate_expression(
            coords_true, prob_map, mode="nb", seed=seed_counter
        )

        # Get baseline score (Distortion 0 / Identity)
        # We treat "None" or param=0 as baseline.
        # But wait, subsample=1.0 is baseline.

        # Let's run the distortions
        for dist_name, values in distortions.items():
            for val in values:

                # Apply distortion
                coords_dist = simlib.apply_distortion(
                    coords_true, dist_name, float(val), seed=seed_counter
                )

                # Subsampling requires filtering counts too
                current_counts = counts
                current_libs = libs

                if dist_name == "subsample":
                    if val < 1.0:
                        n_keep = int(len(coords_true) * val)
                        rng = np.random.default_rng(seed_counter)
                        idx = rng.choice(len(coords_true), n_keep, replace=False)
                        coords_dist = coords_true[idx]  # No spatial distortion, just subset
                        current_counts = counts[idx]
                        current_libs = libs[idx]
                    else:
                        coords_dist = coords_true

                # Score
                res = simlib.score_with_biorsp(
                    coords_dist,
                    current_counts,
                    current_libs,
                    gene_name=f"{dist_name}_{val}_{rep}",
                    config=config,
                    seed=seed_counter,
                )

                res.update(
                    {
                        "distortion": dist_name,
                        "value": val,
                        "rep": rep,
                        "original_shape": base_shape,
                        "archetype": archetype,
                    }
                )
                results.append(res)
                pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="robustness_results.csv")
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()

    run_robustness_benchmark(args.output, args.reps)
