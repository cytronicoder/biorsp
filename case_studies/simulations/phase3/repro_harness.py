import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def capture_environment() -> Dict[str, Any]:
    """Capture environment details for reproducibility."""
    env = {
        "python_version": sys.version,
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dependencies": {},
    }

    # Try to get versions of key dependencies
    deps = ["numpy", "scipy", "matplotlib", "pandas", "sklearn"]
    for dep in deps:
        try:
            mod = __import__(dep)
            env["dependencies"][dep] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env["dependencies"][dep] = "not installed"

    return env


def capture_git_info() -> Dict[str, Any]:
    """Capture git commit hash and dirty flag."""
    git_info = {"commit": "unknown", "dirty": False}
    try:
        git_info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        )
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
        git_info["dirty"] = len(status) > 0
    except Exception:
        pass
    return git_info


def get_seeds(n: int, base_seed: int = 42) -> list[int]:
    """Generate a deterministic list of seeds."""
    return [base_seed + i for i in range(n)]


def save_master_manifest(outdir: Path, module_results: Dict[str, Any]):
    """Save the master run manifest."""
    manifest_dir = outdir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Convert non-serializable objects (like DataFrames) to strings or dicts
    serializable_results = {}
    for k, v in module_results.items():
        if isinstance(v, pd.DataFrame):
            serializable_results[k] = v.to_dict(orient="records")
        else:
            serializable_results[k] = v

    master_manifest = {
        "environment": capture_environment(),
        "git": capture_git_info(),
        "modules": serializable_results,
    }

    with open(manifest_dir / "master_run.json", "w") as f:
        json.dump(master_manifest, f, indent=4, default=str)


def setup_phase3_outdir(outdir: Path):
    """Create the Phase 3 output directory structure."""
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "manifests").mkdir(parents=True, exist_ok=True)
    (outdir / "manuscript_snippets").mkdir(parents=True, exist_ok=True)
