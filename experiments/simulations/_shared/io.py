"""Shared filesystem and metadata helpers for simulation experiments."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as ``Path``."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def safe_mkdir(path: str | Path) -> Path:
    """Alias for ``ensure_dir`` for API readability."""
    return ensure_dir(path)


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> Path:
    """Write JSON payload to disk."""
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def write_yaml(path: str | Path, payload: dict[str, Any] | list[Any]) -> Path:
    """Write YAML payload to disk (falls back to JSON if PyYAML is unavailable)."""
    out = Path(path)
    ensure_dir(out.parent)
    try:
        import yaml  # type: ignore

        out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    except Exception:
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def write_csv(path: str | Path, df: pd.DataFrame) -> Path:
    """Write CSV with atomic replacement."""
    return atomic_write_csv(path, df)


def write_config(outdir: str | Path, config_dict: dict[str, Any]) -> Path:
    """Write ``config.json`` into ``outdir`` and return the path."""
    out = ensure_dir(outdir)
    cfg_path = out / "config.json"
    cfg_path.write_text(
        json.dumps(config_dict, indent=2, sort_keys=True), encoding="utf-8"
    )
    return cfg_path


def atomic_write_csv(path: str | Path, df: pd.DataFrame) -> Path:
    """Safely write a CSV by replacing a temporary file."""
    out = Path(path)
    ensure_dir(out.parent)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    return out


def timestamped_run_id(prefix: str = "run") -> str:
    """Return a compact UTC timestamp-based run id."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def git_commit_hash(cwd: str | Path | None = None, short: bool = True) -> str:
    """Return git commit hash or ``unknown`` without raising."""
    cmd = ["git", "rev-parse", "HEAD"]
    if short:
        cmd.insert(2, "--short")
    try:
        return subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=str(cwd) if cwd is not None else None,
        ).strip()
    except Exception:
        return "unknown"


def git_is_dirty(cwd: str | Path | None = None) -> bool:
    """Return whether the git working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=str(cwd) if cwd is not None else None,
        )
        return bool(out.strip())
    except Exception:
        return False


def _git_root(cwd: str | Path | None = None) -> Path | None:
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=str(cwd) if cwd is not None else None,
        ).strip()
        return Path(root)
    except Exception:
        return None


def _git_diff(cwd: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "diff", "--binary"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=str(cwd) if cwd is not None else None,
        )
    except Exception:
        return ""


def _sanitize_tag(tag: str) -> str:
    keep = []
    for ch in str(tag):
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("-")
    out = "".join(keep).strip("-_")
    return out or "tag"


def init_run_dir(
    outdir: str | Path,
    exp_name: str,
    run_tag: str | None,
    test_mode: bool,
    overwrite: bool,
    master_seed: int | None = None,
) -> Path:
    """Create standardized run directory and canonical subfolders.

    Layout:
    - config/
    - logs/
    - results/
    - plots/
    - diagnostics/
    - artifact_snapshot/ (optional; created lazily)
    """
    root = Path(outdir)
    if root.name.startswith("run_") and root.parent.name == "runs":
        run_dir = root
    else:
        runs_root = ensure_dir(root / "runs")
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        seed_part = f"__seed{int(master_seed)}" if master_seed is not None else ""
        tag_part = f"__{_sanitize_tag(run_tag)}" if run_tag else ""
        mode_part = "__test" if bool(test_mode) else ""
        run_dir = runs_root / f"run_{ts}{seed_part}{tag_part}{mode_part}"

    if run_dir.exists() and bool(overwrite):
        shutil.rmtree(run_dir)
    ensure_dir(run_dir)
    for sub in ["config", "logs", "results", "plots", "diagnostics", "cache"]:
        ensure_dir(run_dir / sub)
    return run_dir


def _sanitize_component(value: str) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-_") or "x"


def _first_scalar(config: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


def make_run_dir(
    root: str | Path,
    exp_name: str,
    tag: str | None = None,
    *,
    seed: int | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Create unique run dir: <root>/<exp>/runs/run_<timestamp>__<meta...>."""
    root_p = ensure_dir(root)
    exp_root = ensure_dir(root_p / str(exp_name))
    runs_root = ensure_dir(exp_root / "runs")

    cfg = dict(config or {})
    seed_val = (
        seed
        if seed is not None
        else _first_scalar(cfg, ["master_seed", "seed", "global_seed"])
    )
    n_val = _first_scalar(cfg, ["N", "N_total", "N_cells"])
    d_val = _first_scalar(cfg, ["D", "donors", "donor_count"])
    nperm_val = _first_scalar(cfg, ["n_perm", "n_perm_pool", "n_perm_donor"])
    if d_val is None and isinstance(cfg.get("D_grid"), list) and cfg["D_grid"]:
        d_val = cfg["D_grid"][0]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    shortgit = git_commit_hash(cwd=root_p, short=True)
    parts = [f"run_{ts}", _sanitize_component(shortgit)]
    if seed_val is not None:
        parts.append(f"seed{int(seed_val)}")
    if n_val is not None:
        parts.append(f"N{int(n_val)}")
    if d_val is not None:
        parts.append(f"D{int(d_val)}")
    if nperm_val is not None:
        parts.append(f"nperm{int(nperm_val)}")
    if tag:
        parts.append(_sanitize_component(tag))

    stem = "__".join(parts)
    run_dir = runs_root / stem
    suffix = 1
    while run_dir.exists():
        run_dir = runs_root / f"{stem}__u{suffix:02d}"
        suffix += 1

    ensure_dir(run_dir)
    for sub in [
        "config",
        "logs",
        "results",
        "plots",
        "diagnostics",
        "figures",
        "cache",
    ]:
        ensure_dir(run_dir / sub)
    return run_dir


def write_config_snapshot(
    run_dir: str | Path,
    config_path: str | Path,
    *,
    resolved_config: dict[str, Any] | None = None,
) -> Path:
    """Persist source config and resolved config in run directory."""
    rd = Path(run_dir)
    ensure_dir(rd / "config")
    src = Path(config_path)
    out = rd / "config_snapshot.json"
    payload: dict[str, Any] = {
        "source_config_path": str(src),
    }
    if src.exists():
        payload["source_config"] = json.loads(src.read_text(encoding="utf-8"))
    if resolved_config is not None:
        payload["resolved_config"] = resolved_config
        (rd / "config" / "resolved_config.json").write_text(
            json.dumps(resolved_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def write_env_snapshot(run_dir: str | Path) -> Path:
    """Persist runtime environment metadata (python/os/cpu/packages)."""
    rd = Path(run_dir)
    out = rd / "env.json"

    pip_freeze = ""
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pip_freeze = ""

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "environment_variables_subset": {
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "pip_freeze": [line for line in pip_freeze.splitlines() if line.strip()],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def write_git_snapshot(run_dir: str | Path, *, cwd: str | Path | None = None) -> Path:
    """Persist git commit/dirty metadata and diff patch when dirty."""
    rd = Path(run_dir)
    root = _git_root(cwd)
    commit_full = git_commit_hash(cwd=root or cwd, short=False)
    commit_short = git_commit_hash(cwd=root or cwd, short=True)
    dirty = git_is_dirty(cwd=root or cwd)

    payload = {
        "git_root": str(root) if root else None,
        "commit": commit_full,
        "commit_short": commit_short,
        "dirty": bool(dirty),
    }
    out = rd / "git.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if dirty:
        diff = _git_diff(cwd=root or cwd)
        if diff.strip():
            (rd / "diff.patch").write_text(diff, encoding="utf-8")
    return out


def save_fig(fig: Any, path: str | Path, dpi: int = 220) -> Path:
    """Save a matplotlib figure with consistent defaults."""
    out = Path(path)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    return out


def copy_artifact_snapshot(paths: list[str | Path], dest: str | Path) -> Path:
    """Copy a minimal artifact snapshot (scripts/config/shared helpers)."""
    dst = ensure_dir(dest)
    for p in paths:
        src = Path(p)
        if not src.exists():
            continue
        target = dst / src.name
        if src.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)
    return dst


def snapshot_artifacts(
    run_dir: str | Path, files: list[str | Path] | None = None
) -> Path:
    """Copy selected files into ``artifact_snapshot/`` under run_dir."""
    rd = Path(run_dir)
    dst = ensure_dir(rd / "artifact_snapshot")
    for f in files or []:
        src = Path(f)
        if not src.exists():
            continue
        target = dst / src.name
        if src.is_file():
            shutil.copy2(src, target)
    return dst


def publish_latest(
    run_dir: str | Path, exp_dir: str | Path, compat_copy: bool = False
) -> None:
    """Publish latest-run pointer (and optional compatibility copies)."""
    rd = Path(run_dir)
    ed = Path(exp_dir)
    ensure_dir(ed)

    if bool(compat_copy):
        for sub in ["results", "plots"]:
            src = rd / sub
            if not src.exists():
                continue
            dst = ed / sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

        # Keep a root-level config for legacy tooling while preserving canonical config/.
        for candidate in [rd / "config" / "config.json", rd / "config.json"]:
            if candidate.exists():
                shutil.copy2(candidate, ed / "config.json")
                break

        report = rd / "REPORT.md"
        if report.exists():
            shutil.copy2(report, ed / "REPORT.md")

    latest_ptr = ed / "LATEST"
    try:
        rel = os.path.relpath(str(rd), str(ed))
    except Exception:
        rel = str(rd)
    latest_ptr.write_text(rel + "\n", encoding="utf-8")


def init_debug_dir(exp_dir: str | Path, debug_name: str) -> Path:
    """Create standardized debug directory under ``debug_runs``."""
    base = ensure_dir(Path(exp_dir) / "debug_runs" / _sanitize_tag(debug_name))
    run_id = timestamped_run_id(prefix="run")
    out = ensure_dir(base / run_id)
    for sub in ["config", "logs", "results", "plots", "diagnostics"]:
        ensure_dir(out / sub)
    return out


def _cache_token(key: str) -> str:
    payload = str(key).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(key))[
        :60
    ]
    return f"{safe}__{digest}"


def _cache_paths(cache_dir: str | Path, key: str) -> tuple[Path, Path]:
    root = ensure_dir(cache_dir)
    token = _cache_token(str(key))
    return root / f"{token}.npz", root / f"{token}.pkl"


def cache_set(cache_dir: str | Path, key: str, obj: Any) -> Path:
    """Persist a cache value under `cache_dir` keyed by `key`."""
    npz_path, pkl_path = _cache_paths(cache_dir, str(key))
    if isinstance(obj, np.ndarray):
        np.savez_compressed(npz_path, arr=obj)
        return npz_path
    if (
        isinstance(obj, dict)
        and obj
        and all(isinstance(v, np.ndarray) for v in obj.values())
    ):
        np.savez_compressed(npz_path, **obj)
        return npz_path
    with pkl_path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path


def cache_get(cache_dir: str | Path, key: str) -> Any | None:
    """Load cached object by `key`; returns `None` when no cache is present."""
    npz_path, pkl_path = _cache_paths(cache_dir, str(key))
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            keys = list(data.keys())
            if keys == ["arr"]:
                return data["arr"]
            return {k: data[k] for k in keys}
    if pkl_path.exists():
        with pkl_path.open("rb") as fh:
            return pickle.load(fh)
    return None
