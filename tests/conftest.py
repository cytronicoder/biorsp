"""Shared pytest configuration and fixtures for the BioRSP test suite.

This module centralizes deterministic RNG, filesystem helpers, CLI runners,
and lightweight contract assertions for benchmark artifacts. Markers are
applied based on folder layout to keep selection consistent in CI.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent


def _discover_project_root() -> Path:
    """Find the repository root by looking for pyproject.toml upward."""

    cursor = Path(__file__).resolve().parent
    while cursor != cursor.parent:
        if (cursor / "pyproject.toml").exists():
            return cursor
        cursor = cursor.parent
    return ROOT_DIR


PROJECT_ROOT = _discover_project_root()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--seed",
        action="store",
        default=os.environ.get("PYTEST_SEED", "1337"),
        help="Base seed used by the rng fixture and global PRNGs.",
    )
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Force fast/quick mode for tests that support it.",
    )


def pytest_configure(config: pytest.Config) -> None:
    markers = {
        "unit": "Fast, pure unit tests (<1s per file)",
        "integration": "Moderate integration tests using filesystem/subprocess (<30s)",
        "scientific": "Bounded statistical/scientific correctness checks",
        "plotting": "Plot generation or matplotlib-dependent tests",
        "slow": "Long-running or heavy tests gated out of default runs",
    }
    for name, desc in markers.items():
        config.addinivalue_line("markers", f"{name}: {desc}")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    marker_map = [
        ("tests/plotting/", [pytest.mark.plotting]),
        ("tests/integration/", [pytest.mark.integration]),
        ("tests/scientific/", [pytest.mark.scientific, pytest.mark.slow]),
        ("tests/unit/", [pytest.mark.unit]),
        ("tests/slow/", [pytest.mark.slow]),
    ]

    slow_files = {
        "test_run_calibration_quick_smoke.py",
    }

    for item in items:
        fspath = str(item.fspath)
        applied = False
        for prefix, marks in marker_map:
            if prefix in fspath:
                for mark in marks:
                    item.add_marker(mark)
                applied = True
        filename = Path(fspath).name
        if "plot" in filename:
            item.add_marker(pytest.mark.plotting)
        if filename in slow_files:
            item.add_marker(pytest.mark.slow)
        if not applied:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Repository root for resolving data/scripts."""

    return PROJECT_ROOT


@pytest.fixture(scope="session")
def base_seed(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--seed"))


@pytest.fixture(scope="session", autouse=True)
def seed_prngs(base_seed: int) -> None:
    random.seed(base_seed)
    np.random.seed(base_seed)
    os.environ.setdefault("PYTHONHASHSEED", str(base_seed))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


@pytest.fixture(scope="session", autouse=True)
def enable_noninteractive_backend() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        # Matplotlib is optional in some environments.
        pass


@pytest.fixture(scope="session")
def fast_mode(request: pytest.FixtureRequest) -> bool:
    fast = bool(
        request.config.getoption("--fast") or os.environ.get("FAST_TESTS") or os.environ.get("CI")
    )
    if fast:
        os.environ.setdefault("BIORSP_TEST_MODE", "quick")
    return fast


@pytest.fixture
def rng(base_seed: int) -> np.random.Generator:
    return np.random.default_rng(base_seed)


@pytest.fixture
def tmp_outdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("out")


@pytest.fixture(scope="session")
def benchmark_cache(pytestconfig: pytest.Config):
    from tests._helpers.benchmark_cache import BenchmarkCache

    return BenchmarkCache(pytestconfig)


@pytest.fixture(scope="session")
def fast_params() -> dict:
    """Minimal sweep parameters for quick CI-safe runs."""

    return {
        "shapes": ["disk"],
        "sample_sizes": [128, 256],
        "replicates": 1,
        "n_permutations": 32,
    }


@pytest.fixture
def small_synthetic_dataset(rng: np.random.Generator) -> dict[str, object]:
    coords = rng.normal(loc=0.0, scale=1.0, size=(32, 2))
    expr = rng.poisson(lam=2.0, size=(32, 3)).astype(float)

    dataset: dict[str, object] = {"coords": coords, "expr": expr}
    try:
        import anndata as ad

        adata = ad.AnnData(X=expr, obsm={"X_spatial": coords})
        adata.var_names = [f"gene_{i}" for i in range(expr.shape[1])]
        dataset["adata"] = adata
    except Exception:
        dataset["adata"] = None

    return dataset


@pytest.fixture
def run_cli(
    project_root: Path,
) -> Callable[[Iterable[str], Path | None, dict | None], subprocess.CompletedProcess]:
    def _run_cli(
        args: Iterable[str],
        cwd: Path | None = None,
        env: dict | None = None,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess:
        run_env = os.environ.copy()
        run_env.setdefault(
            "PYTHONPATH", f"{project_root}{os.pathsep}{run_env.get('PYTHONPATH', '')}"
        )
        run_env.setdefault("MPLBACKEND", "Agg")
        if env:
            run_env.update(env)

        return subprocess.run(
            [sys.executable, *args],
            cwd=str(cwd or project_root),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    return _run_cli


class ContractAssertions:
    """Helper for validating benchmark artifacts consistently."""

    runs_required = {
        "run_id",
        "benchmark",
        "mode",
        "seed",
        "replicate_id",
        "status",
        "abstain_flag",
        "abstain_reason",
        "shape",
        "n_cells",
    }

    summary_required = {
        "benchmark",
        "mode",
        "n_runs",
        "runtime_seconds",
    }

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _load_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df.sort_index(axis=1).reset_index(drop=True)

    def assert_runs(self, path: Path) -> pd.DataFrame:
        df = self._load_csv(path)
        missing = self.runs_required - set(df.columns)
        assert not missing, f"runs.csv missing columns: {sorted(missing)}"
        return df

    def assert_summary(self, path: Path) -> pd.DataFrame:
        df = self._load_csv(path)
        missing = self.summary_required - set(df.columns)
        assert not missing, f"summary.csv missing columns: {sorted(missing)}"
        return df

    def assert_manifest(self, path: Path) -> dict:
        with open(path) as f:
            manifest = json.load(f)
        assert "benchmark_name" in manifest and "biorsp_config" in manifest
        return manifest


@pytest.fixture
def contract_assertions(tmp_path: Path) -> ContractAssertions:
    return ContractAssertions(tmp_path)
