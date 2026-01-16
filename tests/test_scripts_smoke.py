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

# Find repo root by looking for pyproject.toml
_test_file = Path(__file__).resolve()
_search_path = _test_file.parent
while _search_path != _search_path.parent:
    if (_search_path / "pyproject.toml").exists():
        ROOT_DIR = _search_path
        break
    _search_path = _search_path.parent
else:
    # Fallback to tests/..
    ROOT_DIR = _test_file.parent.parent

EXAMPLES_DIR = ROOT_DIR / "examples"

# Import the manifest
sys.path.insert(0, str(ROOT_DIR / "dev"))
from scripts_manifest import get_all_scripts  # noqa: E402


def get_env():
    """Get environment with workspace root in PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    return env


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


@pytest.mark.parametrize(
    "script_path",
    [
        EXAMPLES_DIR / "quickstart.py",
        EXAMPLES_DIR / "simulation.py",
    ],
)
def test_example_help(script_path):
    """Check if example scripts can show their help message without crashing."""
    assert script_path.exists(), f"Example script {script_path} not found"

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


@pytest.mark.parametrize("script_info", get_all_scripts(), ids=lambda x: x["id"])
def test_script_smoke(script_info, tmp_path):
    """Run each script in smoke mode and verify it completes successfully."""
    script_id = script_info["id"]
    script_relpath = script_info["relpath"]
    args_smoke = script_info.get("args_smoke", [])
    timeout = script_info.get("timeout_seconds", 120)
    expected_outputs = script_info.get("expected_outputs", [])

    script_path = ROOT_DIR / script_relpath
    assert script_path.exists(), (
        f"Script {script_id} not found at {script_path}. "
        f"Update dev/scripts_manifest.py if the script was moved or removed."
    )

    # Build command with smoke args
    cmd = [sys.executable, str(script_path)] + args_smoke

    # If args include --smoke, add --outdir to tmp_path
    if "--smoke" in args_smoke:
        cmd.extend(["--outdir", str(tmp_path)])

    # Run script
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=get_env(),
        cwd=str(tmp_path),
        check=False,
    )

    assert result.returncode == 0, (
        f"Script {script_id} failed with return code {result.returncode}\n"
        f"Command: {' '.join(cmd)}\n"
        f"Stdout: {result.stdout}\n"
        f"Stderr: {result.stderr}"
    )

    # Check expected outputs if specified
    for pattern in expected_outputs:
        matches = list(tmp_path.glob(pattern))
        assert len(matches) > 0, (
            f"Script {script_id} did not create expected output matching '{pattern}' in {tmp_path}\n"
            f"Files created: {list(tmp_path.iterdir())}"
        )
        # Verify files are non-empty
        for fpath in matches:
            assert (
                fpath.stat().st_size > 1000
            ), f"Output file {fpath.name} is suspiciously small ({fpath.stat().st_size} bytes)"
