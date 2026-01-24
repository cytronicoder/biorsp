from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def benchmarks_root(project_root: Path) -> Path:
    return project_root / "analysis" / "benchmarks"


@pytest.fixture(scope="module")
def runners_dir(benchmarks_root: Path) -> Path:
    return benchmarks_root / "runners"


def run_script(
    script_name: str,
    out_name: str,
    run_cli,
    runners_dir: Path,
    benchmarks_root: Path,
    tmp_outdir: Path,
):
    script_path = runners_dir / script_name
    out_path = tmp_outdir / out_name

    cmd = [
        str(script_path),
        "--mode",
        "quick",
        "--n-reps",
        "2",
        "--outdir",
        str(out_path),
        "--seed",
        "42",
        "--n-workers",
        "1",
    ]

    result = run_cli(cmd, cwd=benchmarks_root, timeout=180)

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")

    assert result.returncode == 0, f"Script {script_name} failed"

    # benchmark runners sometimes nest outputs under <outdir>/<benchmark>/<run_id>
    root = out_path
    contract_root = out_path / out_name
    if contract_root.exists():
        run_dirs = sorted(contract_root.iterdir())
        if run_dirs:
            root = run_dirs[-1]

    assert (root / "runs.csv").exists()
    assert (root / "summary.csv").exists()
    assert (root / "manifest.json").exists()
    assert (root / "report.md").exists()

    runs = pd.read_csv(root / "runs.csv")
    assert len(runs) >= 2, f"Expected at least 2 runs, got {len(runs)}"


@pytest.mark.slow
def test_run_calibration(run_cli, runners_dir: Path, benchmarks_root: Path, tmp_outdir: Path):
    run_script(
        "run_calibration.py",
        "calibration",
        run_cli,
        runners_dir,
        benchmarks_root,
        tmp_outdir,
    )


@pytest.mark.slow
def test_run_archetypes(run_cli, runners_dir: Path, benchmarks_root: Path, tmp_outdir: Path):
    run_script(
        "run_archetypes.py",
        "archetypes",
        run_cli,
        runners_dir,
        benchmarks_root,
        tmp_outdir,
    )


@pytest.mark.slow
def test_run_genegene(run_cli, runners_dir: Path, benchmarks_root: Path, tmp_outdir: Path):
    run_script(
        "run_genegene.py",
        "genegene",
        run_cli,
        runners_dir,
        benchmarks_root,
        tmp_outdir,
    )


@pytest.mark.slow
def test_run_robustness(run_cli, runners_dir: Path, benchmarks_root: Path, tmp_outdir: Path):
    run_script(
        "run_robustness.py",
        "robustness",
        run_cli,
        runners_dir,
        benchmarks_root,
        tmp_outdir,
    )
