#!/usr/bin/env python3
"""
Unified Smoke Test Runner for BioRSP.

This script executes a series of quick checks to ensure the repository is healthy.
It runs:
1. Basic scoring example (synthetic data).
2. Advanced configuration example.
3. A quick benchmark run (archetypes).

Usage:
    python dev/smoke_tests/run_smoke.py

"""

import subprocess
import sys
from pathlib import Path

# Repository root
ROOT = Path(__file__).resolve().parents[2]


def run_command(cmd, cwd=ROOT, desc=None):
    """Run a shell command and check for success."""
    if desc:
        print(f"\n[SMOKE] Running: {desc}...")
        print(f"        Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=cwd, check=True)
        print(f"[SMOKE] SUCCESS: {desc}")
    except subprocess.CalledProcessError:
        print(f"[SMOKE] FAILED: {desc}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("BioRSP Unified Smoke Tests")
    print("=" * 60)

    # 1. Basic Example
    run_command(["python", "examples/1_basic_scoring.py"], desc="Basic Scoring Example")

    # 2. Advanced Example (if exists)
    adv_ex = ROOT / "examples/2_advanced_config.py"
    if adv_ex.exists():
        run_command(
            ["python", "examples/2_advanced_config.py"], desc="Advanced Configuration Example"
        )

    # 3. Quick Benchmark
    # We use runs/analysis/benchmarks/runners/run_archetypes.py
    # Note: --mode quick --n_reps 1
    bench_cmd = [
        "python",
        "analysis/benchmarks/runners/run_archetypes.py",
        "--mode",
        "quick",
        "--n_reps",
        "1",
        "--outdir",
        "dev/smoke_tests/output_bench_smoke",
    ]
    run_command(bench_cmd, desc="Quick Archetype Benchmark")

    print("\n" + "=" * 60)
    print("All Smoke Tests Passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
