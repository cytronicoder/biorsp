#!/usr/bin/env python3
"""
Smoke Test for Story Figure Generation.

Runs quick mode and verifies that all outputs are created correctly.

Usage:
    python smoke_story.py

Exit codes:
    0: All checks pass
    1: Some checks failed
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_smoke_test():
    """Run smoke test for story figure."""
    import shutil
    import tempfile

    print("=" * 60)
    print("BioRSP Story Figure Smoke Test")
    print("=" * 60)

    temp_dir = Path(tempfile.mkdtemp(prefix="biorsp_smoke_"))
    print(f"\nTemp output: {temp_dir}")

    all_passed = True

    try:
        print("\n[1/4] Testing story onepager...")
        from benchmarks.run_story_onepager import run_story_benchmark

        class Args:
            mode = "quick"
            outdir = str(temp_dir / "story")
            seed = 42

        result = run_story_benchmark(Args())

        story_dir = temp_dir / "story"
        figures_dir = story_dir / "figures"

        checks = [
            ("runs.csv exists", (story_dir / "runs.csv").exists()),
            ("summary.csv exists", (story_dir / "summary.csv").exists()),
            ("manifest.json exists", (story_dir / "manifest.json").exists()),
            ("report.md exists", (story_dir / "report.md").exists()),
            (
                "fig_story_A_archetypes.png exists",
                (figures_dir / "fig_story_A_archetypes.png").exists(),
            ),
            (
                "fig_story_B_confusion.png exists",
                (figures_dir / "fig_story_B_confusion.png").exists(),
            ),
            (
                "fig_story_C_marker_recovery.png exists",
                (figures_dir / "fig_story_C_marker_recovery.png").exists(),
            ),
            (
                "fig_story_D_genegene.png exists",
                (figures_dir / "fig_story_D_genegene.png").exists(),
            ),
            ("fig_story_onepager.png exists", (figures_dir / "fig_story_onepager.png").exists()),
            ("accuracy > 0.5", result["accuracy"] > 0.5),
            ("macro_f1 > 0.4", result["macro_f1"] > 0.4),
        ]

        for name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"   {status} {name}")
            if not passed:
                all_passed = False

        # Check figure sizes
        for fig_path in figures_dir.glob("*.png"):
            size = fig_path.stat().st_size
            if size < 1000:
                print(f"   ✗ {fig_path.name} is too small ({size} bytes)")
                all_passed = False
            else:
                print(f"   ✓ {fig_path.name} size OK ({size} bytes)")

        print("\n[2/4] Testing null calibration...")
        from benchmarks.run_null_calibration import run_calibration

        Args.outdir = str(temp_dir / "calibration")
        thresholds = run_calibration(Args())

        checks = [
            ("thresholds.json exists", (temp_dir / "calibration" / "thresholds.json").exists()),
            ("s_cut > 0", thresholds["s_cut"] > 0),
            ("s_cut < 1", thresholds["s_cut"] < 1),
        ]

        for name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"   {status} {name}")
            if not passed:
                all_passed = False

        print("\n[3/4] Testing stability benchmark...")
        from benchmarks.run_stability import run_stability

        Args.outdir = str(temp_dir / "stability")
        stability = run_stability(Args())

        checks = [
            (
                "fig_stability_embeddings.png exists",
                (temp_dir / "stability" / "fig_stability_embeddings.png").exists(),
            ),
            ("score_correlation > 0.5", stability["score_correlation"] > 0.5),
        ]

        for name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"   {status} {name}")
            if not passed:
                all_passed = False

        print("\n[4/4] Testing abstention evaluation...")
        from benchmarks.run_abstention import run_abstention

        Args.outdir = str(temp_dir / "abstention")
        abstention_df = run_abstention(Args())

        checks = [
            (
                "fig_abstention.png exists",
                (temp_dir / "abstention" / "fig_abstention.png").exists(),
            ),
            ("results have entries", len(abstention_df) > 0),
        ]

        for name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"   {status} {name}")
            if not passed:
                all_passed = False

    finally:
        print(f"\nCleaning up {temp_dir}...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL SMOKE TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        print("✗ SOME SMOKE TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_test())
