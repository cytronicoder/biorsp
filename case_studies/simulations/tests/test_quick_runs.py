import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
SIM_ROOT = TEST_DIR.parent
BENCHMARKS_DIR = SIM_ROOT / "benchmarks"
OUTPUT_DIR = SIM_ROOT / "outputs" / "test"


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    """Clean up test outputs before and after."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield

    pass


def run_script(script_name: str, out_name: str, args: list = None):
    """Run a benchmarks script via subprocess."""
    if args is None:
        args = []
    script_path = BENCHMARKS_DIR / script_name
    out_path = OUTPUT_DIR / out_name

    cmd = [
        sys.executable,
        str(script_path),
        "--mode",
        "quick",
        "--n_reps",
        "2",
        "--outdir",
        str(out_path),
        "--seed",
        "42",
        "--n_workers",
        "1",
    ] + args

    result = subprocess.run(
        cmd,
        cwd=str(SIM_ROOT),
        env={**sys.modules["os"].environ, "PYTHONPATH": str(SIM_ROOT.parent.parent)},
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")

    assert result.returncode == 0, f"Script {script_name} failed"

    assert (out_path / "runs.csv").exists()
    assert (out_path / "summary.csv").exists()
    assert (out_path / "manifest.json").exists()
    assert (out_path / "report.md").exists()

    import pandas as pd

    runs = pd.read_csv(out_path / "runs.csv")
    assert len(runs) >= 2, f"Expected at least 2 runs, got {len(runs)}"


def test_run_calibration():
    run_script("run_calibration.py", "calibration")


def test_run_archetypes():
    run_script("run_archetypes.py", "archetypes", ["--shape", "disk"])


def test_run_genegene():

    run_script("run_genegene.py", "genegene")


def test_run_robustness():
    run_script("run_robustness.py", "robustness", ["--distortion_kind", "jitter"])
