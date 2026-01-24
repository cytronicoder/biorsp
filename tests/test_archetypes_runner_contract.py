"""Integration and contract tests for the archetype benchmark runner."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from analysis.benchmarks.simlib.io_contract import assert_contract_runs
from biorsp.utils.labels import CANONICAL_ARCHETYPES, normalize_archetype_label


def _run_quick(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "analysis.benchmarks.runners.run_archetypes",
        "--mode",
        "quick",
        "--seed",
        "42",
        "--outdir",
        str(outdir),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    run_root = outdir / "archetypes"
    run_dirs = sorted(run_root.iterdir())
    if not run_dirs:
        raise RuntimeError("No run directory produced by archetype runner")
    return run_dirs[-1]


def test_archetypes_runner_contract(tmp_path: Path):
    run_dir = _run_quick(tmp_path / "outputs")

    runs_df = pd.read_csv(run_dir / "runs.csv")
    assert_contract_runs(runs_df, benchmark="archetypes")

    for label in CANONICAL_ARCHETYPES:
        assert (runs_df["Archetype_true"] == label).any(), f"missing {label} ground truth"

    assert runs_df["pattern_family"].nunique() >= 2
    assert runs_df["pattern_variant"].nunique() >= 3

    expected_files = [
        "runs.csv",
        "summary.csv",
        "manifest.json",
        "report.md",
        "thresholds_used.json",
        "fig_cs_scatter.png",  # Standard plot set
        "fig_cs_marginals.png",
        "fig_confusion_or_composition.png",
        "diagnostics/misclassification_patterns.csv",
    ]
    for rel in expected_files:
        path = run_dir / rel
        assert path.exists(), f"missing output file: {rel}"
        assert path.stat().st_size > 0, f"output file empty: {rel}"

    abstain_rate = runs_df["abstain_flag"].mean()
    assert abstain_rate < 1.0


def test_normalize_archetype_label_alias():
    legacy = ["housekeeping", "gradient", "niche_marker", "sparse_noise", "abstention"]
    expected = ["Ubiquitous", "Gradient", "Patchy", "Basal", "Abstain"]
    assert [normalize_archetype_label(x) for x in legacy] == expected
