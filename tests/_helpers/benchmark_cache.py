"""Caching helper for benchmark subprocess invocations.

Avoids rerunning heavy runners when parameters are identical within a test session.
Uses pytest's cache directory to store outputs keyed by command/env/cwd.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Mapping

import pytest


def _hash_key(cmd: Iterable[str], cwd: Path, env: Mapping[str, str]) -> str:
    payload = json.dumps(
        {
            "cmd": list(cmd),
            "cwd": str(cwd),
            "env": {k: env.get(k, "") for k in sorted(env)},
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class BenchmarkCache:
    def __init__(self, pytestconfig: pytest.Config) -> None:
        self.cache_dir = Path(pytestconfig.cache.makedir("bench_runs"))

    def run(
        self,
        cmd: Iterable[str],
        cwd: Path,
        env: Mapping[str, str],
        timeout: int = 180,
        expected_files: list[str] | None = None,
    ) -> Path:
        key = _hash_key(cmd, cwd, env)
        run_dir = self.cache_dir / key
        run_dir.mkdir(parents=True, exist_ok=True)

        sentinel = run_dir / ".completed"
        if sentinel.exists():
            if expected_files and not all((run_dir / f).exists() for f in expected_files):
                sentinel.unlink()
            else:
                return run_dir

        # Clean any partial contents and rerun
        for child in run_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

        result = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=dict(env),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Benchmark command failed (code {result.returncode}): {' '.join(cmd)}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        sentinel.touch()
        return run_dir
