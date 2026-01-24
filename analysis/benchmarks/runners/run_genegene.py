"""
Gene-Gene Pairwise Benchmark for BioRSP Methods Paper.

Evaluates the ability to detect co-patterns and exclusion patterns:
1. Co-localization (Same pattern) → High Correlation
2. Exclusion (Opposite pattern) → High Complementarity
3. Orthogonal (Different wedges) → Near 0 Correlation

Outputs: runs.csv, summary.csv, report.md, manifest.json, debug plots
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biorsp import BioRSPConfig

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local helper imports are performed inside `main()` to avoid module-level import-after-code issues (E402).


def run_genegene_condition(config_dict: dict, seed: int, config: "BioRSPConfig") -> dict:
    """Run one gene-gene replicate."""
    from biorsp.simulations import (
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

    # Normalize and handle abstention
    if len(pairs_df) == 0:
        return {
            "benchmark": "genegene",
            "shape": shape,
            "N": N,
            "scenario": scenario,
            "similarity_profile": np.nan,
            "copattern_score": np.nan,
            "shared_mask_fraction": np.nan,
            "abstain_flag": True,
            "abstain_reason": "no_results",
            "time": elapsed,
        }

    row = pairs_df.iloc[0]
    # Check if finite results
    has_finite = pd.notna(row.get("similarity_profile", np.nan))

    return {
        "benchmark": "genegene",
        "shape": shape,
        "N": N,
        "scenario": scenario,
        "similarity_profile": row["similarity_profile"],
        "copattern_score": row["copattern_score"],
        "shared_mask_fraction": row["shared_mask_fraction"],
        "abstain_flag": not has_finite,
        "abstain_reason": "" if has_finite else "undefined_similarity",
        "time": elapsed,
    }


def main():
    from analysis.benchmarks.simlib.io_contract import BenchmarkContractConfig, init_run_dir
    from analysis.benchmarks.simlib.runner_harness import finalize_contract
    from biorsp.simulations import (
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
    parser.add_argument(
        "--mode", type=str, choices=["quick", "validation", "publication"], default="quick"
    )
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
    elif args.mode == "validation":
        args.n_reps = 20
        args.N = [1500, 2500]
        args.shape = ["disk", "annulus"]
        args.scenario = ["same", "opposite", "rim_core"]
        args.n_permutations = 300
        args.permutation_scope = "topk"
        args.topk_perm = 300
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

    output_dir = Path(args.outdir)
    contract_cfg = BenchmarkContractConfig(require_runs_csv=True, require_summary_csv=True)
    session_id = init_run_dir(
        output_dir, clear_existing=not args.resume, contract_config=contract_cfg
    )

    runs_csv_path = output_dir / "runs.csv"
    skip_completed = set()
    if args.resume and runs_csv_path.exists():
        skip_completed = checkpoint.load_completed_runs(runs_csv_path)
        print(f"Resuming: {len(skip_completed)} runs already completed")

    config = BioRSPConfig(
        B=72,
        delta_deg=60.0,
        n_permutations=n_perms,
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
                "n_reps": len(group),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    io.write_summary_csv(summary_df, output_dir, benchmark="genegene")

    print("Generating plots...")

    # Debug figure: joint scatter for example gene pairs
    plot_genegene_debug(output_dir, runs_df, args)

    # Main similarity distribution plot
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

    # Finalize contract
    finalize_contract(
        output_dir=output_dir,
        benchmark_name="genegene",
        contract_config=contract_cfg,
        session_id=session_id,
        mode=args.mode,
        n_conditions=len(configs) * args.n_reps,
        n_completed=len(runs_df),
        runtime_seconds=runtime,
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
    print(f"   Output directory: {output_dir}")
    print(f"   Runtime: {runtime:.1f}s")


def plot_genegene_debug(output_dir, runs_df, args):
    """Generate debug figure: joint scatter and similarity distributions."""
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    # Similarity distribution finite-only
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for scenario in args.scenario:
        subset = runs_df[runs_df["scenario"] == scenario]
        vals = subset["similarity_profile"].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=20, alpha=0.5, label=f"{scenario} (n={len(vals)})")

    ax.set_xlabel("Similarity Profile (finite only)")
    ax.set_ylabel("Count")
    ax.set_title("Gene-Gene Similarity Distribution (Abstention-Aware)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(debug_dir / "fig_similarity_finite_only.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Abstention summary
    abstain_counts = runs_df.groupby("scenario")["abstain_flag"].sum()
    total_counts = runs_df.groupby("scenario").size()
    abstain_rates = (abstain_counts / total_counts).fillna(0)

    if len(abstain_rates) > 0 and abstain_rates.sum() > 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        abstain_rates.plot(kind="bar", ax=ax, color="orange", edgecolor="black")
        ax.set_ylabel("Abstention Rate")
        ax.set_xlabel("Scenario")
        ax.set_title("Abstention Rate by Scenario")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(debug_dir / "fig_abstention_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
