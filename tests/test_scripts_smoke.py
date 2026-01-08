import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize(
    "script_path",
    [
        EXAMPLES_DIR / "quickstart.py",
        EXAMPLES_DIR / "simulation.py",
        SCRIPTS_DIR / "plot_simulation_csv.py",
    ],
)
def test_script_help(script_path):
    """Check if scripts can at least show their help message without crashing."""
    if not script_path.exists():
        pytest.skip(f"Script {script_path} not found")

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "help" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_biorsp_cli_help():
    """Check if the main CLI works."""
    result = subprocess.run(
        [sys.executable, "-m", "biorsp.cli", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
