#!/usr/bin/env python3
"""
Batch runner for all BioRSP simulation benchmarks.

Runs all 4 benchmarks sequentially with optimized settings, tracking progress
and aggregating runtime statistics.

Usage:
    # Full publication pipeline (~3-4 hours with 8 workers)
    python run_benchmarks.py --mode publication --n_workers 8

    # Quick smoke test (~5 minutes)
    python run_benchmarks.py --mode quick --n_workers 4

    # Custom configuration
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
    ("calibration", "run_calibration.py", {}),
    ("archetypes", "run_archetypes.py", {}),
    ("genegene", "run_genegene.py", {"topk_perm": 500}),
    ("robustness", "run_robustness.py", {}),
]


def run_benchmark(
    name: str,
    script: str,
    mode: str,
    n_workers: int,
    checkpoint_every: int,
    resume: bool,
    extra_flags: dict,
) -> Tuple[bool, float, str]:
    """Run a single benchmark."""
    cmd = [
        sys.executable,
        str(BENCHMARKS_DIR / script),
        "--mode",
        mode,
        "--n_workers",
        str(n_workers),
        "--checkpoint_every",
        str(checkpoint_every),
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
            print(f"\n✅ {name}: COMPLETED ({elapsed:.1f}s = {elapsed / 60:.1f}m)")
            return True, elapsed, "success"
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
  # Full publication pipeline
  python run_benchmarks.py --mode publication --n_workers 8

  # Quick test
  python run_benchmarks.py --mode quick --n_workers 4

  # Resume interrupted run
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

    print(
        f"""
╔════════════════════════════════════════════════════════════════════╗
║  BioRSP Simulation Benchmark Suite                                ║
║  Mode: {args.mode:16}  Workers: {args.n_workers:4}                    ║
║  Benchmarks: {len(benchmarks_to_run)}/4 enabled                                     ║
╚════════════════════════════════════════════════════════════════════╝
"""
    )

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
