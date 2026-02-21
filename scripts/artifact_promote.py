#!/usr/bin/env python3
"""Promote selected run artifacts into curated docs folders with manifest metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp"}
TABLE_EXTENSIONS = {".csv", ".tsv", ".json", ".md", ".txt"}
DEFAULT_MAX_MB = 5


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_component(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip("/"))


def get_git_commit(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True
        ).strip()
    except Exception:
        return "unknown"


def get_run_metadata(run_dir: Path) -> dict[str, Any]:
    args_json = run_dir / "config" / "args.json"
    config_snapshot = run_dir / "config_snapshot.json"
    config_fallback = run_dir / "config" / "config.json"

    args_payload: dict[str, Any] = {}
    if args_json.exists():
        try:
            args_payload = json.loads(args_json.read_text(encoding="utf-8"))
        except Exception:
            args_payload = {}

    config_hash = None
    config_source = None
    if config_snapshot.exists():
        config_hash = sha256_file(config_snapshot)
        config_source = str(config_snapshot)
    elif config_fallback.exists():
        config_hash = sha256_file(config_fallback)
        config_source = str(config_fallback)

    return {
        "n_perm": args_payload.get("n_perm")
        or args_payload.get("n_perm_pool")
        or args_payload.get("n_perm_donor"),
        "seed": args_payload.get("seed"),
        "master_seed": args_payload.get("master_seed"),
        "seeds": args_payload.get("seeds"),
        "config_snapshot_hash": config_hash,
        "config_snapshot_source": config_source,
    }


def parse_select(select: str) -> list[str]:
    out = []
    for token in select.split(","):
        candidate = token.strip()
        if candidate:
            out.append(candidate)
    return out


def choose_destination(relpath: str) -> str:
    ext = Path(relpath).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "figures"
    return "tables"


def copy_selected(
    run_dir: Path,
    selected: list[str],
    dest_prefix: str,
    max_mb: int,
    force: bool,
) -> list[dict[str, Any]]:
    root = repo_root()
    docs_figures = root / "docs" / "figures"
    docs_tables = root / "docs" / "tables"
    docs_figures.mkdir(parents=True, exist_ok=True)
    docs_tables.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, Any]] = []
    max_bytes = int(max_mb * 1024 * 1024)

    for rel in selected:
        src = (run_dir / rel).resolve()
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Selected artifact not found: {rel}")

        size_bytes = src.stat().st_size
        if size_bytes > max_bytes and not force:
            raise RuntimeError(
                f"Refusing to promote {rel} ({size_bytes} bytes > {max_bytes} bytes). "
                "Use --force to override."
            )

        bucket = choose_destination(rel)
        stem = sanitize_component(Path(rel).with_suffix("").as_posix())
        ext = src.suffix.lower()
        out_name = f"{sanitize_component(dest_prefix)}__{stem}{ext}"
        dst = (docs_figures if bucket == "figures" else docs_tables) / out_name
        shutil.copy2(src, dst)

        copied.append(
            {
                "source_relpath": rel,
                "source_path": str(src),
                "dest_bucket": bucket,
                "dest_path": str(dst),
                "size_bytes": size_bytes,
                "sha256": sha256_file(dst),
            }
        )

    return copied


def write_manifest(
    dest_prefix: str, run_dir: Path, copied: list[dict[str, Any]]
) -> Path:
    root = repo_root()
    manifest_dir = root / "docs" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{sanitize_component(dest_prefix)}.json"

    run_meta = get_run_metadata(run_dir)
    payload = {
        "dest_prefix": dest_prefix,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "git_commit": get_git_commit(root),
        "config_snapshot_hash": run_meta.get("config_snapshot_hash"),
        "config_snapshot_source": run_meta.get("config_snapshot_source"),
        "n_perm": run_meta.get("n_perm"),
        "seed": run_meta.get("seed"),
        "master_seed": run_meta.get("master_seed"),
        "seeds": run_meta.get("seeds"),
        "selected_artifacts": copied,
        "release_bundle": {
            "url": None,
            "checksum_sha256": None,
            "notes": "Optional: add GitHub Release URL and checksum for full run bundle.",
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote selected run artifacts into docs/ with manifest metadata."
    )
    parser.add_argument(
        "--run_dir", required=True, help="Path to a completed run directory."
    )
    parser.add_argument(
        "--select",
        required=True,
        help="Comma-separated list of relative file paths to copy from run_dir.",
    )
    parser.add_argument(
        "--dest_prefix",
        required=True,
        help="Prefix used for destination file names and manifest name.",
    )
    parser.add_argument(
        "--max_mb",
        type=int,
        default=DEFAULT_MAX_MB,
        help="Maximum file size allowed without --force.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Allow files larger than --max_mb."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    selected = parse_select(args.select)
    if not selected:
        raise ValueError("--select must contain at least one file path")

    copied = copy_selected(
        run_dir=run_dir,
        selected=selected,
        dest_prefix=args.dest_prefix,
        max_mb=int(args.max_mb),
        force=bool(args.force),
    )
    manifest = write_manifest(args.dest_prefix, run_dir, copied)

    print(f"Promoted {len(copied)} artifacts.")
    for item in copied:
        print(f"- {item['source_relpath']} -> {item['dest_path']}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
