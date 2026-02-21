#!/usr/bin/env python3
"""Audit tracked artifact files for size and policy violations."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

RUN_STAMP_RE = re.compile(r"(^|/)run_[0-9]{8}([_T][0-9]{6}Z?)?(/|$)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_allowlist(root: Path, allowlist_path: str | None) -> set[str]:
    if allowlist_path is None:
        candidate = root / ".artifact_audit_allowlist"
    else:
        candidate = (root / allowlist_path).resolve()
    if not candidate.exists():
        return set()

    allowed: set[str] = set()
    for raw in candidate.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        allowed.add(line.replace("\\", "/"))
    return allowed


def git_tracked_files(root: Path) -> list[Path]:
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(root))
    rel = [p for p in out.decode("utf-8", errors="replace").split("\x00") if p]
    return [root / p for p in rel]


def is_results_tracked(rel: str) -> bool:
    normalized = rel.replace("\\", "/")
    return "/_results/" in f"/{normalized}" or normalized.startswith("_results/")


def is_run_stamped(rel: str) -> bool:
    normalized = rel.replace("\\", "/")
    return bool(RUN_STAMP_RE.search(normalized))


def format_bytes(num: int) -> str:
    mb = num / (1024 * 1024)
    return f"{mb:.2f} MB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit tracked files for size and policy violations."
    )
    parser.add_argument("--warn_mb", type=int, default=5, help="Warn threshold in MB.")
    parser.add_argument("--max_mb", type=int, default=10, help="Fail threshold in MB.")
    parser.add_argument(
        "--fail", action="store_true", help="Return non-zero when violations are found."
    )
    parser.add_argument(
        "--allowlist",
        default=None,
        help="Optional repo-relative allowlist file (default: .artifact_audit_allowlist if present).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    tracked = git_tracked_files(root)
    allowlist = load_allowlist(root, args.allowlist)

    warn_bytes = int(args.warn_mb * 1024 * 1024)
    max_bytes = int(args.max_mb * 1024 * 1024)

    large_warn: list[tuple[str, int]] = []
    large_fail: list[tuple[str, int]] = []
    tracked_results: list[str] = []
    tracked_run_stamped: list[str] = []

    for path in tracked:
        rel = path.relative_to(root).as_posix()
        if rel in allowlist:
            continue
        if not path.exists() or not path.is_file():
            continue

        size = path.stat().st_size
        if size > warn_bytes:
            large_warn.append((rel, size))
        if size > max_bytes:
            large_fail.append((rel, size))
        if is_results_tracked(rel):
            tracked_results.append(rel)
        if is_run_stamped(rel):
            tracked_run_stamped.append(rel)

    if large_warn:
        print(f"[WARN] Tracked files above {args.warn_mb} MB:")
        for rel, size in sorted(large_warn, key=lambda x: x[1], reverse=True):
            print(f"  - {rel} ({format_bytes(size)})")

    violations = False

    if large_fail:
        violations = True
        print(f"[FAIL] Tracked files above {args.max_mb} MB:")
        for rel, size in sorted(large_fail, key=lambda x: x[1], reverse=True):
            print(f"  - {rel} ({format_bytes(size)})")

    if tracked_results:
        violations = True
        print(
            "[FAIL] Tracked files found under experiments/**/_results or top-level _results:"
        )
        for rel in sorted(tracked_results):
            print(f"  - {rel}")

    if tracked_run_stamped:
        violations = True
        print("[FAIL] Tracked files found in run-stamped directories (run_YYYY...):")
        for rel in sorted(tracked_run_stamped):
            print(f"  - {rel}")

    if violations:
        print("\nRemediation:")
        print("1) Move bulk run outputs to experiments/**/_results/ (gitignored)")
        print("2) Promote only curated artifacts via scripts/artifact_promote.py")
        print(
            "3) Keep full run bundle in a GitHub Release and reference it in docs/manifests/*.json"
        )
        if args.fail:
            return 1

    print("Artifact audit complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
