"""
Gene-Gene Pairwise Benchmark for BioRSP Methods Paper.

Evaluates the ability to detect co-patterns and exclusion patterns:
1. Co-localization (Same pattern) → High Correlation
2. Exclusion (Opposite pattern) → High Complementarity
3. Orthogonal (Different wedges) → Near 0 Correlation

Outputs: runs.csv, summary.csv, report.md, manifest.json, PR curves
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from biorsp import BioRSPConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_genegene_condition(config_dict: dict, seed: int, config: BioRSPConfig) -> dict:
    """Run one gene-gene replicate."""
    from simlib import (
        datasets,
        expression,
        rng,
        scoring,
        shapes,
    )

    shape = config_dict["shape"]
    N = config_dict["N"]
    scenario = config_dict["scenario"]

    pattern_pairs = {
        "same": ("wedge", {"angle_center": 0}, "wedge", {"angle_center": 0}),
        "opposite": ("wedge", {"angle_center": 0}, "wedge", {"angle_center": np.pi}),
        "orthogonal": ("wedge", {"angle_center": 0}, "wedge", {"angle_center": np.pi / 2}),
        "rim_core": ("rim", {}, "core", {}),
    }

    if scenario not in pattern_pairs:
        raise ValueError(f"Unknown scenario: {scenario}")

    pattern1, params1, pattern2, params2 = pattern_pairs[scenario]

    condition_key = rng.condition_key(shape, N, scenario)
    gen = rng.make_rng(seed, "genegene", condition_key)

    coords, shape_meta = shapes.generate_coords(shape, N, gen)

    libsize = expression.simulate_library_size(
        N, gen, model="lognormal", params={"mean": 2000, "std": 0.5}
    )

    field1 = expression.generate_signal_field(coords, pattern1, params1)
    counts1 = expression.generate_expression_from_field(
        field1, libsize, gen, expr_model="nb", params={"phi": 10.0, "abundance": 1e-3}
    )

    field2 = expression.generate_signal_field(coords, pattern2, params2)
    counts2 = expression.generate_expression_from_field(
        field2, libsize, gen, expr_model="nb", params={"phi": 10.0, "abundance": 1e-3}
    )

    X = np.column_stack([counts1, counts2])
    adata = datasets.package_as_anndata(
        coords, X, var_names=["gene1", "gene2"], obs_meta=None, embedding_key="X_sim"
    )

    t0 = time.time()
    pairs_df = scoring.score_pairs(adata, genes=["gene1", "gene2"], config=config)
    elapsed = time.time() - t0
    if len(pairs_df) == 0:
        return {
            "shape": shape,
            "N": N,
            "scenario": scenario,
            "similarity_profile": np.nan,
            "copattern_score": np.nan,
            "shared_mask_fraction": np.nan,
            "time": elapsed,
        }

    row = pairs_df.iloc[0]
    return {
        "shape": shape,
        "N": N,
        "scenario": scenario,
        "similarity_profile": row["similarity_profile"],
        "copattern_score": row["copattern_score"],
        "shared_mask_fraction": row["shared_mask_fraction"],
        "time": elapsed,
    }


def main():
    from simlib import (
        checkpoint,
        docs,
        io,
        sweeps,
        validation,
    )

    parser = argparse.ArgumentParser(description="Gene-gene co-patterning benchmark")
    parser.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "genegene"))
    parser.add_argument("--seed", type=int, default=8000)
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--N", type=int, nargs="+", default=[2000, 5000])
    parser.add_argument("--shape", type=str, nargs="+", default=["disk"])
    parser.add_argument(
        "--scenario", type=str, nargs="+", default=["same", "opposite", "orthogonal", "rim_core"]
    )
    parser.add_argument("--n_permutations", type=int, default=250)
    parser.add_argument("--mode", type=str, choices=["quick", "publication"], default="quick")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--n_workers", type=int, default=-1, help="Number of parallel workers")
    parser.add_argument(
        "--checkpoint_every", type=int, default=25, help="Save checkpoint every N runs"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--permutation_scope",
        type=str,
        choices=["none", "topk", "all"],
        default="topk",
        help="Permutation strategy: 'none', 'topk' (only top K pairs), 'all'",
    )
    parser.add_argument(
        "--topk_perm", type=int, default=500, help="Number of top pairs for topk permutation"
    )
    args = parser.parse_args()

    if args.n_workers != -1:
        args.n_jobs = args.n_workers

    if args.mode == "quick":

        args.n_reps = 5
        args.N = [2000]
        args.shape = ["disk"]
        args.scenario = ["same"]
        args.n_permutations = 100
        args.permutation_scope = "none"
        args.topk_perm = 100
    elif args.mode == "publication":

        if args.n_reps == 50:

            args.n_reps = 50
            args.N = [1000, 2000]
            args.shape = ["disk", "annulus"]
            args.scenario = ["same", "opposite"]
            args.n_permutations = 500
            args.permutation_scope = "topk"
            args.topk_perm = 500
        else:

            args.n_reps = max(args.n_reps, 100)

            args.N = [1000, 2000, 5000]

            args.shape = ["disk", "annulus", "peanut"]

            args.scenario = ["same", "opposite", "orthogonal", "rim_core"]

            args.n_permutations = 1000
            args.topk_perm = 1000

            args.permutation_scope = "all"

    n_perms = 0
    if args.permutation_scope == "all" or args.permutation_scope == "topk":
        n_perms = args.n_permutations

    output_dir = io.ensure_output_dir("genegene", base_dir=args.outdir.rsplit("/", 1)[0])

    runs_csv_path = output_dir / "runs.csv"
    skip_completed = set()
    if args.resume and runs_csv_path.exists():
        skip_completed = checkpoint.load_completed_runs(runs_csv_path)
        print(f"Resuming: {len(skip_completed)} runs already completed")

    config = BioRSPConfig(
        B=72,
        delta_deg=60.0,
        n_permutations=n_perms,
        qc_mode="principled",
    )

    configs = sweeps.expand_grid(shape=args.shape, N=args.N, scenario=args.scenario)

    print(f"Running gene-gene benchmark: {len(configs)} conditions × {args.n_reps} reps")

    def save_checkpoint(results: list):
        """Save incremental checkpoint."""
        if not results:
            return
        checkpoint_df = pd.DataFrame(results)
        checkpoint.append_to_runs_csv(checkpoint_df, runs_csv_path)
        print(f"✓ Checkpoint saved ({len(results)} results)")

    start_time = time.time()

    runs_df = sweeps.run_replicates(
        run_genegene_condition,
        configs,
        args.n_reps,
        seed_start=args.seed,
        progress=True,
        n_jobs=args.n_jobs,
        fn_args=(config,),
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=save_checkpoint,
        skip_completed=skip_completed,
    )

    runtime = time.time() - start_time

    io.write_runs_csv(runs_df, output_dir, benchmark="genegene")

    summary_rows = []
    for (shape, N, scenario), group in runs_df.groupby(["shape", "N", "scenario"]):
        sim_mean = group["similarity_profile"].mean()
        sim_std = group["similarity_profile"].std()
        cop_mean = group["copattern_score"].mean()
        cop_std = group["copattern_score"].std()
        mask_mean = group["shared_mask_fraction"].mean()

        summary_rows.append(
            {
                "shape": shape,
                "N": N,
                "scenario": scenario,
                "similarity_profile_mean": sim_mean,
                "similarity_profile_std": sim_std,
                "copattern_score_mean": cop_mean,
                "copattern_score_std": cop_std,
                "shared_mask_fraction_mean": mask_mean,
                "shared_mask_fraction_std": group["shared_mask_fraction"].std(),
                "n_tests": len(group),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    io.write_summary_csv(summary_df, output_dir, benchmark="genegene")

    print("Generating plots...")
    figs_dir = ROOT / "outputs" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    try:
        validation.validate_dataframe_for_plot(
            runs_df,
            required_columns=["scenario", "similarity_profile"],
            min_rows=1,
            name="gene-gene similarity distribution plot",
        )
    except validation.ValidationError as e:
        print(f"⚠ Skipping plots: {e}")
    else:

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        for scenario in args.scenario:
            subset = runs_df[runs_df["scenario"] == scenario]
            vals = subset["similarity_profile"].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=20, alpha=0.5, label=scenario)
        ax.set_xlabel("Similarity Profile")
        ax.set_ylabel("Count")
        ax.set_title("Gene-Gene Similarity Profile Distribution")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        io.save_figure(fig, output_dir, "genegene_similarity_dist.png")

    interpretation = docs.interpret_genegene(summary_df)
    docs.write_report(
        output_dir,
        "genegene",
        summary_df,
        params=vars(args),
        interpretation=interpretation,
    )

    io.write_manifest(
        output_dir,
        benchmark_name="genegene",
        params=vars(args),
        n_replicates=args.n_reps * len(configs),
        runtime_seconds=runtime,
        biorsp_config=config,
    )

    print("\n✅ Gene-gene benchmark complete!")
    print(f"   Outputs: {output_dir}")
    print(f"   Figures: {figs_dir}")
    print(f"   Runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
