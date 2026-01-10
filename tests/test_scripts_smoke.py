"""Smoke tests for scripts to prevent empty plots and ensure basic functionality.

These tests run scripts in minimal mode to verify:
1. Script can be executed without errors
2. Output files are created
3. Files have non-zero size
4. Help text is available

This prevents regressions like empty plots or broken imports.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
EXAMPLES_DIR = ROOT_DIR / "examples"


def get_env():
    """Get environment with workspace root in PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.mark.parametrize(
    "script_path",
    [
        EXAMPLES_DIR / "quickstart.py",
        EXAMPLES_DIR / "simulation.py",
        SCRIPTS_DIR / "plot_simulation_csv.py",
        SCRIPTS_DIR / "make_end_to_end_workflow.py",
        SCRIPTS_DIR / "make_polar_embedding_figure.py",
        SCRIPTS_DIR / "make_schematic_diagram.py",
    ],
)
def test_script_help(script_path):
    """Check if scripts can show their help message without crashing."""
    if not script_path.exists():
        pytest.skip(f"Script {script_path} not found")

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env=get_env(),
        check=False,
    )
    assert result.returncode == 0, f"Help failed for {script_path.name}"
    assert (
        "help" in result.stdout.lower() or "usage" in result.stdout.lower()
    ), f"No help/usage text for {script_path.name}"


def test_biorsp_cli_help():
    """Check if the main CLI works."""
    result = subprocess.run(
        [sys.executable, "-m", "biorsp.cli", "--help"],
        capture_output=True,
        text=True,
        env=get_env(),
        check=False,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_make_schematic_diagram_runs():
    """Test that make_schematic_diagram.py executes and creates output."""
    script = SCRIPTS_DIR / "make_schematic_diagram.py"
    if not script.exists():
        pytest.skip(f"Script not found: {script}")

    outdir = Path("outputs") / "tests" / "schematic"
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, str(script), "--outdir", str(outdir)],
            capture_output=True,
            text=True,
            timeout=120,
            env=get_env(),
            check=False,
        )

        assert (
            result.returncode == 0
        ), f"Script failed\nStdout: {result.stdout}\nStderr: {result.stderr}"

        # Check output exists
        expected_files = list(outdir.glob("fig_schematic_diagram*"))
        assert len(expected_files) > 0, f"No output files created in {outdir}"

        for fpath in expected_files:
            assert fpath.stat().st_size > 2000, f"Output file too small: {fpath}"
    finally:
        # Clean up
        if outdir.exists():
            for fpath in outdir.glob("*"):
                fpath.unlink()
            outdir.rmdir()


def test_make_end_to_end_workflow_demo():
    """Test make_end_to_end_workflow.py with demo data."""
    script = SCRIPTS_DIR / "make_end_to_end_workflow.py"
    if not script.exists():
        pytest.skip(f"Script not found: {script}")

    outdir = Path("outputs") / "tests"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "test_end_to_end_demo.png"

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--feature",
                "TestGene",
                "--out",
                str(outpath),
                "--seed",
                "123",
                "--B",
                "18",
                "--delta-deg",
                "60",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env=get_env(),
            check=False,
        )

        assert (
            result.returncode == 0
        ), f"Script failed\nStdout: {result.stdout}\nStderr: {result.stderr}"

        assert outpath.exists(), f"Output not created: {outpath}"
        assert outpath.stat().st_size > 5000, "Output file suspiciously small"
    finally:
        outpath.unlink(missing_ok=True)


def test_make_polar_embedding_figure_demo():
    """Test make_polar_embedding_figure.py with demo data."""
    script = SCRIPTS_DIR / "make_polar_embedding_figure.py"
    if not script.exists():
        pytest.skip(f"Script not found: {script}")

    outdir = Path("outputs") / "tests"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "test_polar_demo.png"

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--feature",
                "TestGene",
                "--out",
                str(outpath),
                "--seed",
                "123",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=get_env(),
            check=False,
        )

        assert (
            result.returncode == 0
        ), f"Script failed\nStdout: {result.stdout}\nStderr: {result.stderr}"

        assert outpath.exists(), "Output not created"
        assert outpath.stat().st_size > 5000, "Output file suspiciously small"
    finally:
        outpath.unlink(missing_ok=True)
        (outdir / outpath.with_suffix(".pdf").name).unlink(missing_ok=True)


def test_debug_end_to_end_runs():
    """Test that debug_end_to_end.py runs without crashing."""
    script = SCRIPTS_DIR / "debug_end_to_end.py"
    if not script.exists():
        pytest.skip(f"Script not found: {script}")

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=180,
            env=get_env(),
            check=False,
        )

        assert (
            result.returncode == 0
        ), f"Script failed\nStdout: {result.stdout}\nStderr: {result.stderr}"
    finally:
        # Clean up any generated debug figures
        debug_dir = Path("outputs") / "debug"
        if debug_dir.exists():
            for pattern in ["debug_*.png", "debug_*.pdf"]:
                for fpath in debug_dir.glob(pattern):
                    fpath.unlink(missing_ok=True)


def test_debug_selection_bias_runs():
    """Test that debug_selection_bias.py runs without crashing."""
    script = SCRIPTS_DIR / "debug_selection_bias.py"
    if not script.exists():
        pytest.skip(f"Script not found: {script}")

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=180,
        env=get_env(),
        check=False,
    )

    assert (
        result.returncode == 0
    ), f"Script failed\nStdout: {result.stdout}\nStderr: {result.stderr}"
