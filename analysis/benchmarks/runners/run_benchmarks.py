"""
Batch runner for all BioRSP simulation benchmarks.

Runs all benchmarks sequentially with optimized settings, tracking progress
and aggregating runtime statistics. Validates contract outputs strictly.

Usage:
    python run_benchmarks.py --mode publication --n_workers 8

    python run_benchmarks.py --mode quick --n_workers 4

    python run_benchmarks.py --mode publication --n_workers 8 --checkpoint_every 50 --resume
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent
BENCHMARKS_DIR = ROOT

BENCHMARKS = [
    ("null_calibration", "run_null_calibration.py", {}),
    ("calibration", "run_calibration.py", {}),
    ("archetypes", "run_archetypes.py", {}),
    ("abstention", "run_abstention.py", {}),
    ("genegene", "run_genegene.py", {"topk_perm": 500}),
    ("robustness", "run_robustness.py", {}),
    ("stability", "run_stability.py", {}),
]


def validate_benchmark_outputs(name: str, output_dir: Path) -> Tuple[bool, str]:
    """Validate that benchmark outputs are complete and non-empty.

    Args:
        name: Benchmark name.
        output_dir: Output directory for the benchmark.

    Returns:
        Tuple of (is_valid, message).
    """
    required_files = {
        "report.md": "report",
        "manifest.json": "manifest",
    }

    # Most benchmarks need runs.csv and summary.csv
    if name not in ["stability"]:
        required_files["runs.csv"] = "runs"
        required_files["summary.csv"] = "summary"

    for filename, description in required_files.items():
        filepath = output_dir / filename
        if not filepath.exists():
            return False, f"Missing {description} file: {filename}"

        # Check file is non-empty
        if filepath.stat().st_size == 0:
            return False, f"Empty {description} file: {filename}"

        # For CSV files, check they have content beyond header
        if filename.endswith(".csv"):
            try:
                import pandas as pd

                df = pd.read_csv(filepath)
                if len(df) == 0:
                    return False, f"Empty data in {filename}"
            except Exception as e:
                return False, f"Invalid CSV {filename}: {e}"

    # Check for at least one figure
    fig_files = list(output_dir.glob("fig_*.png"))
    if len(fig_files) == 0:
        return False, "No figures generated (missing fig_*.png)"

    return True, "All outputs validated"


def run_benchmark(
    name: str,
    script: str,
    mode: str,
    n_workers: int,
    checkpoint_every: int,
    resume: bool,
    extra_flags: dict,
) -> Tuple[bool, float, str]:
    """Run a single benchmark and validate outputs.

    Args:
        name: Benchmark name.
        script: Runner script name.
        mode: Benchmark mode.
        n_workers: Number of workers.
        checkpoint_every: Checkpoint cadence.
        resume: Whether to resume from checkpoints.
        extra_flags: Additional CLI flags.

    Returns:
        Tuple of (success, runtime_seconds, status_message).
    """
    output_dir = ROOT.parent / "outputs" / name

    cmd = [
        sys.executable,
        str(BENCHMARKS_DIR / script),
        "--mode",
        mode,
        "--n_workers",
        str(n_workers),
        "--checkpoint_every",
        str(checkpoint_every),
        "--outdir",
        str(output_dir),
    ]

    if resume:
        cmd.append("--resume")

    for key, val in extra_flags.items():
        cmd.extend([f"--{key}", str(val)])

    print(f"\n{'=' * 70}")
    print(f"Running {name.upper()} benchmark...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        elapsed = time.time() - start

        if result.returncode == 0:
            # Validate outputs
            valid, msg = validate_benchmark_outputs(name, output_dir)
            if valid:
                print(f"\n✅ {name}: COMPLETED ({elapsed:.1f}s = {elapsed / 60:.1f}m)")
                return True, elapsed, "success"
            else:
                print(f"\n❌ {name}: OUTPUT VALIDATION FAILED - {msg}")
                return False, elapsed, f"validation_failed: {msg}"
        else:
            print(f"\n❌ {name}: FAILED (exit code {result.returncode})")
            return False, elapsed, f"exit_code_{result.returncode}"

    except KeyboardInterrupt:
        elapsed = time.time() - start
        print(f"\n⚠️  {name}: INTERRUPTED by user ({elapsed:.1f}s)")
        raise
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n💥 {name}: ERROR - {e}")
        return False, elapsed, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Run all BioRSP simulation benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py --mode publication --n_workers 8

  python run_benchmarks.py --mode quick --n_workers 4

  python run_benchmarks.py --mode publication --n_workers 8 --resume
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "publication"],
        default="quick",
        help="Benchmark mode: 'quick' (smoke test) or 'publication' (full grid)",
    )
    parser.add_argument(
        "--n_workers", type=int, default=-1, help="Number of parallel workers (-1 = use all cores)"
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=25, help="Save checkpoint every N runs"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint if interrupted"
    )
    parser.add_argument(
        "--skip",
        type=str,
        nargs="+",
        default=[],
        help="Skip specific benchmarks (e.g., --skip calibration robustness)",
    )
    args = parser.parse_args()

    benchmarks_to_run = [
        (name, script, extra) for name, script, extra in BENCHMARKS if name not in args.skip
    ]

    if not benchmarks_to_run:
        print("❌ No benchmarks to run (all were skipped)")
        sys.exit(1)

    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║  BioRSP Simulation Benchmark Suite                                ║
║  Mode: {args.mode:16}  Workers: {args.n_workers:4}                    ║
║  Benchmarks: {len(benchmarks_to_run)}/4 enabled                                     ║
╚════════════════════════════════════════════════════════════════════╝
""")

    results = []
    total_time = 0

    try:
        for name, script, extra_flags in benchmarks_to_run:
            success, elapsed, message = run_benchmark(
                name=name,
                script=script,
                mode=args.mode,
                n_workers=args.n_workers,
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
                extra_flags=extra_flags,
            )
            results.append((name, success, elapsed, message))
            total_time += elapsed

            if not success:
                print(f"\n⚠️  Warning: {name} failed, continuing with remaining benchmarks...")

    except KeyboardInterrupt:
        print("\n\n⚠️  Batch run interrupted by user")
        print("You can resume by running with --resume flag")

    print("\n" + "=" * 70)
    print("BATCH RUN SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success, _, _ in results if success)
    failed = len(results) - passed

    for name, success, elapsed, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {name:15} ({elapsed:6.1f}s = {elapsed / 60:5.1f}m)  {message}")

    print(f"\nTotal: {passed}/{len(results)} passed, {failed} failed")
    print(f"Total runtime: {total_time:.1f}s = {total_time / 60:.1f}m = {total_time / 3600:.2f}h")

    if failed > 0:
        print("\n❌ Some benchmarks failed. Check output above for details.")
        print("You can rerun with --resume to continue from checkpoints.")
        sys.exit(1)
    else:
        print("\n✅ All benchmarks completed successfully!")
        print("\nResults location:")
        print(f"  Outputs: {ROOT / 'outputs'}")
        print(f"  Figures: {ROOT / 'outputs' / 'figures'}")
        sys.exit(0)


if __name__ == "__main__":
    main()
