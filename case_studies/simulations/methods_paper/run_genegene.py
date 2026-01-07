"""
Gene-Gene Pairwise Benchmark for BioRSP Methods Paper.

Evaluates the ability to detect co-patterns and exclusion patterns.
1. Co-localization (Same pattern) -> High Correlation
2. Exclusion (Opposite pattern) -> High Complementarity (Negative Correlation)
3. Orthogonal (e.g. Wedge 0 deg vs Wedge 90 deg) -> Near 0 Correlation
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from biorsp import compute_rsp_radar
from biorsp.core.pairwise import compute_pairwise_relationships
from case_studies.simulations import simlib


def get_radar(c, lib_size, coords, config):
    """Helper to compute BioRSP Radar for a single gene."""
    x_norm = c / (lib_size + 1e-6) * 10000
    from biorsp.preprocess.foreground import define_foreground

    y, _ = define_foreground(x_norm, mode=config.foreground_mode, q=config.foreground_quantile)
    from biorsp.preprocess.geometry import compute_vantage, polar_coordinates

    v = compute_vantage(coords)
    r, theta = polar_coordinates(coords, v)
    from biorsp import assess_adequacy

    adequacy = assess_adequacy(r, theta, y, config=config)
    if not adequacy.is_adequate:
        return None
    return compute_rsp_radar(
        r,
        theta,
        y,
        config=config,
        sector_indices=adequacy.sector_indices,
        frozen_mask=adequacy.sector_mask,
    )


def run_genegene_benchmark(output_file: str, n_reps: int = 50):
    shape = "disk"
    n_cells = 3000

    scenarios = [
        ("same", "wedge", {"angle_center": 0}, "wedge", {"angle_center": 0}),
        ("opposite", "wedge", {"angle_center": 0}, "wedge", {"angle_center": np.pi}),
        ("orthogonal", "wedge", {"angle_center": 0}, "wedge", {"angle_center": np.pi / 2}),
        ("rim_core", "rim", {}, "core", {}),  # Complementary
    ]

    results = []
    config = simlib.get_base_config_v3()

    pbar = tqdm(total=len(scenarios) * n_reps, desc="Gene-Gene Benchmark")
    seed_counter = 8000

    for sc_name, arch1, p1, arch2, p2 in scenarios:
        for rep in range(n_reps):
            seed_counter += 1

            coords = simlib.generate_coords(shape, n_cells, seed=seed_counter)

            # Gene 1
            pmap1 = simlib.generate_true_probability(coords, arch1, p1)
            c1, l1, _ = simlib.generate_expression(coords, pmap1, "nb", seed=seed_counter)

            # Gene 2
            pmap2 = simlib.generate_true_probability(coords, arch2, p2)
            c2, l2, _ = simlib.generate_expression(coords, pmap2, "nb", seed=seed_counter + 1)

            # Score individually to get Radars
            # We can't use score_with_biorsp directly because we need the RADAR object,
            # and score_with_biorsp returns a dict.
            # We'll use simlib but access internals or modify simlib?
            # Creating a helper here is easier.

            r1 = get_radar(c1, l1, coords, config)
            r2 = get_radar(c2, l2, coords, config)

            if r1 is None or r2 is None:
                continue

            radar_map = {"g1": r1, "g2": r2}
            syn, comp = compute_pairwise_relationships(radar_map)

            # Extract pair result for g1-g2 (should be single entry)
            # syn is list of PairwiseResult
            if not syn:
                continue

            res = {
                "scenario": sc_name,
                "rep": rep,
                "correlation": syn[0].correlation,
                "complementarity": syn[0].complementarity,
                "peak_dist": syn[0].peak_distance,
            }
            results.append(res)
            pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="genegene_results.csv")
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()

    run_genegene_benchmark(args.output, args.reps)
