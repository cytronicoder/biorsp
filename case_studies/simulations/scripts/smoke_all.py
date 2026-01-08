#!/usr/bin/env python3
"""
Smoke test runner for BioRSP simulations.
Runs all four benchmarks in quick mode to verify end-to-end functionality.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():

    script_dir = Path(__file__).resolve().parent
    sim_root = script_dir.parent
    methods_dir = sim_root / "methods_paper"
    output_dir = sim_root / "smoke_outputs"

    if output_dir.exists():
        print(f"Cleaning {output_dir}...")
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    scripts = [
        ("run_calibration.py", ["--null_type", "iid"]),
        ("run_archetypes.py", ["--shape", "disk", "--pattern", "niche_core"]),
        ("run_robustness.py", ["--parameter", "dropout"]),
    ]

    print(f"Starting smoke tests in {output_dir}...")

    failed = []

    for script_name, extra_args in scripts:
        print(f"\n=== Running {script_name} ===")
        t0 = time.time()

        script_path = methods_dir / script_name
        phase_name = script_name.replace("run_", "").replace(".py", "")
        phase_out = output_dir / phase_name

        cmd = [
            sys.executable,
            str(script_path),
            "--mode",
            "quick",
            "--n_reps",
            "2",
            "--outdir",
            str(phase_out),
            "--seed",
            "123",
            "--n_workers",
            "1",
        ] + extra_args

        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=str(sim_root),
                env={**sys.modules["os"].environ, "PYTHONPATH": str(sim_root.parent.parent)},
            )
            print(f"✓ {script_name} passed in {time.time() - t0:.1f}s")
        except subprocess.CalledProcessError:
            print(f"❌ {script_name} FAILED")
            failed.append(script_name)

    if failed:
        print(f"\nSummary: {len(failed)} scripts failed.")
        sys.exit(1)
    else:
        print("\nAll scripts passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
