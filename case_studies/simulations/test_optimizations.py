#!/usr/bin/env python3
"""
Smoke test suite for optimized BioRSP simulation benchmarks.

Runs all 4 benchmarks in quick mode with parallelization to verify:
- Module imports
- CLI flags work
- Checkpointing functions
- Validation guards prevent failures
- Outputs are generated correctly

Usage:
    python test_optimizations.py

Expected runtime: ~5 minutes with 4 workers
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
METHODS_PAPER = ROOT / "methods_paper"


BENCHMARKS = [
    ("calibration", "run_calibration.py", {}),
    ("archetypes", "run_archetypes.py", {}),
    ("genegene", "run_genegene.py", {"topk_perm": 100}),
    ("robustness", "run_robustness.py", {}),
]


def run_benchmark(name: str, script: str, extra_flags: dict) -> tuple[bool, float, str]:
    """Run a single benchmark in quick mode."""
    cmd = [
        sys.executable,
        str(METHODS_PAPER / script),
        "--mode",
        "quick",
        "--n_workers",
        "4",
    ]

    for key, val in extra_flags.items():
        cmd.extend([f"--{key}", str(val)])

    print(f"\n{'='*60}")
    print(f"Testing {name} benchmark...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"✅ {name}: PASSED ({elapsed:.1f}s)")
            return True, elapsed, result.stdout
        else:
            print(f"❌ {name}: FAILED (exit code {result.returncode})")
            print(f"STDERR:\n{result.stderr}")
            return False, elapsed, result.stderr

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"⏱️  {name}: TIMEOUT after {elapsed:.1f}s")
        return False, elapsed, "Timeout"
    except Exception as e:
        elapsed = time.time() - start
        print(f"💥 {name}: ERROR - {e}")
        return False, elapsed, str(e)


def check_outputs(name: str) -> bool:
    """Check that expected output files were created."""
    output_dir = ROOT / "outputs" / name
    expected_files = [
        "runs.csv",
        "summary.csv",
        "report.md",
        "manifest.json",
    ]

    missing = []
    for fname in expected_files:
        fpath = output_dir / fname
        if not fpath.exists():
            missing.append(fname)

    if missing:
        print(f"  ⚠️  Missing outputs: {', '.join(missing)}")
        return False
    else:
        print("  ✅ All expected outputs present")
        return True


def test_imports():
    """Test that all new modules import correctly."""
    print("\n" + "=" * 60)
    print("Testing module imports...")
    print("=" * 60)

    from simlib import cache, checkpoint, validation

    print("✅ All modules import successfully")

    c = cache.GeometryCache()
    stats = c.get_stats()
    print(f"  Cache initialized: {stats}")

    key = checkpoint.make_checkpoint_key(
        {"shape": "disk", "N": 1000, "pattern": "wedge"}, replicate=42
    )
    print(f"  Checkpoint key: {key[:16]}...")

    import pandas as pd

    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    validation.validate_dataframe_for_plot(df, required_columns=["x", "y"], min_rows=1, name="test")
    print("  Validation passed")


def main():
    """Run full smoke test suite."""
    print(
        """
╔════════════════════════════════════════════════════════════╗
║  BioRSP Simulation Optimization Smoke Test Suite          ║
║  Testing all 4 benchmarks in quick mode with parallelism  ║
╚════════════════════════════════════════════════════════════╝
"""
    )

    try:
        test_imports()
    except Exception as e:
        print(f"\n❌ Import test failed: {e}. Aborting.")
        sys.exit(1)

    results = []
    total_time = 0

    for name, script, flags in BENCHMARKS:
        success, elapsed, output = run_benchmark(name, script, flags)
        results.append((name, success, elapsed))
        total_time += elapsed

        if success:
            check_outputs(name)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    for name, success, elapsed in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {name:15} ({elapsed:5.1f}s)")

    print(f"\nTotal: {passed}/{len(results)} passed, {failed} failed")
    print(f"Total runtime: {total_time:.1f}s")

    if failed > 0:
        print("\n❌ Some tests failed. Check output above for details.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        print("\nYou can now run publication mode with:")
        print("  python methods_paper/run_calibration.py --mode publication --n_workers 8")
        sys.exit(0)


if __name__ == "__main__":
    main()
