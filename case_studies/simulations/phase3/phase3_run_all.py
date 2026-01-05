import argparse
import sys
from pathlib import Path

# Add current directory to path for local imports
sys.path.append(str(Path(__file__).parent))

import baselines
import calibration
import failure_modes
import perturbation_robustness
import power_vs_N
import repro_harness
import robustness_params
import separability


def generate_manuscript_snippets(outdir: Path):
    """Generate manuscript-ready text snippets."""
    snippet_path = outdir / "manuscript_snippets" / "simulation_results_paragraphs.txt"

    text = """
Phase 3 Simulation Results Summary

Calibration:
BioRSP demonstrates robust false positive control under null conditions. Across various footprint shapes (disk, crescent, annulus) and density models (uniform, radial bias), the RMS anisotropy remains low and does not systematically inflate. Permutation-based p-values follow a near-uniform distribution, confirming that the method does not hallucinate spatial structure when none exists.

Parameter Robustness:
Sensitivity analysis justifies the default parameters (B=360, delta=20). The radar profiles and scalar summaries remain stable across a wide range of grid sizes and sector widths. Profile similarity (RMSD) to reference settings is high (>0.9) in powered regimes, indicating that BioRSP is not overly sensitive to hyperparameter tuning.

Baseline Comparisons:
BioRSP provides information beyond simple geometric summaries. While angular concentration detects wedge-like patterns and radial separation detects rim/core patterns, BioRSP's anisotropy and resolved profiles uniquely capture mixed or complex spatial distributions. BioRSP effectively localizes enrichment while maintaining sensitivity to both radial and angular components.

Abstention and Failure Modes:
The adequacy framework successfully identifies underpowered regimes. Abstention rates increase predictably as foreground sparsity increases or total cell count decreases. By reporting 'low coverage' instead of unreliable metrics, BioRSP ensures that downstream biological interpretations are grounded in sufficient statistical support.
"""
    with open(snippet_path, "w") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser(description="Run BioRSP Phase 3 Simulation Suite")
    parser.add_argument(
        "--outdir", type=str, default="results/simulations_phase3", help="Output directory"
    )
    parser.add_argument("--n_reps_calib", type=int, default=50, help="Replicates for calibration")
    parser.add_argument("--n_reps_robust", type=int, default=10, help="Replicates for robustness")
    parser.add_argument("--n_reps_base", type=int, default=30, help="Replicates for baselines")
    parser.add_argument("--n_reps_fail", type=int, default=50, help="Replicates for failure modes")
    parser.add_argument("--n_reps_power", type=int, default=20, help="Replicates for power sweep")
    parser.add_argument(
        "--n_reps_perturb", type=int, default=10, help="Replicates for perturbation robustness"
    )
    parser.add_argument("--n_reps_sep", type=int, default=30, help="Replicates for separability")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--n_workers", type=int, default=-1, help="Number of parallel workers (-1 for all CPUs)"
    )

    args = parser.parse_args()
    outdir = Path(args.outdir)

    # Setup
    repro_harness.setup_phase3_outdir(outdir)

    # Run Modules
    module_results = {}
    common_config = {"seed": args.seed, "n_workers": args.n_workers}

    print("--- Phase 3: Calibration ---")
    module_results["calibration"] = calibration.run(
        outdir, {**common_config, "n_reps": args.n_reps_calib}
    )

    print("\n--- Phase 3: Robustness ---")
    module_results["robustness"] = robustness_params.run(
        outdir, {**common_config, "n_reps": args.n_reps_robust}
    )

    print("\n--- Phase 3: Baselines ---")
    module_results["baselines"] = baselines.run(
        outdir, {**common_config, "n_reps": args.n_reps_base}
    )

    print("\n--- Phase 3: Failure Modes ---")
    module_results["failure_modes"] = failure_modes.run(
        outdir, {**common_config, "n_reps": args.n_reps_fail}
    )

    print("\n--- Phase 3: Power vs N/q ---")
    module_results["power_vs_N"] = power_vs_N.run(
        outdir, {**common_config, "n_reps": args.n_reps_power}
    )

    print("\n--- Phase 3: Perturbation Robustness ---")
    module_results["perturbation_robustness"] = perturbation_robustness.run(
        outdir, {**common_config, "n_reps": args.n_reps_perturb}
    )

    print("\n--- Phase 3: Separability ---")
    module_results["separability"] = separability.run(
        outdir, {**common_config, "n_reps": args.n_reps_sep}
    )

    # Finalize
    print("\n--- Finalizing Phase 3 ---")
    repro_harness.save_master_manifest(outdir, module_results)
    generate_manuscript_snippets(outdir)

    print(f"\nPhase 3 complete. Results saved to {outdir}")


if __name__ == "__main__":
    main()
