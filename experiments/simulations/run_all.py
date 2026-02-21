#!/usr/bin/env python3
"""Run preregistered simulations A-J with standardized orchestration."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = SIM_ROOT / "_results"

ORDERED_EXPERIMENTS = [
    ("A", "expA_null_calibration"),
    ("B", "expB_maxstat_sensitivity"),
    ("C", "expC_power_surfaces"),
    ("D", "expD_shape_identifiability"),
    ("E", "expE_gradient_vs_step_DE"),
    ("F", "expF_confound_resistance"),
    ("G", "expG_donor_replication"),
    ("H", "expH_fdr_pipeline_scale"),
    ("I", "expI_embedding_robustness"),
    ("J", "expJ_baselines_ablation"),
]


def _parse_set(val: str | None) -> set[str]:
    if not val:
        return set()
    return {token.strip().upper() for token in val.split(",") if token.strip()}


def _write_index(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "experiment",
                "exp_name",
                "config",
                "status",
                "exit_code",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all simulation experiments A-J.")
    parser.add_argument("--out_root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument(
        "--only", type=str, default=None, help='Comma list like "A,B,D".'
    )
    parser.add_argument("--skip", type=str, default=None, help='Comma list like "H".')
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--strict_prereg", action="store_true")
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    only = _parse_set(args.only)
    skip = _parse_set(args.skip)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root).resolve()
    batch_root = out_root / batch_id
    batch_root.mkdir(parents=True, exist_ok=True)
    index_path = batch_root / "INDEX.csv"

    rows: list[dict[str, str]] = []
    launcher = REPO_ROOT / "experiments" / "simulations" / "run_experiment.py"

    for letter, exp_name in ORDERED_EXPERIMENTS:
        if only and letter not in only:
            continue
        if letter in skip:
            continue

        cfg = f"experiments/simulations/configs/{exp_name}.{args.profile}.json"

        cmd = [
            sys.executable,
            str(launcher),
            "--exp",
            letter,
            "--config",
            cfg,
            "--out_root",
            str(out_root),
            "--batch_id",
            batch_id,
        ]
        if args.tag:
            cmd.extend(["--tag", str(args.tag)])
        if args.n_jobs is not None:
            cmd.extend(["--n_jobs", str(int(args.n_jobs))])
        if bool(args.dry_run):
            cmd.append("--dry_run")
        if bool(args.strict_prereg):
            cmd.append("--strict_prereg")

        print("COMMAND=" + shlex.join(cmd), flush=True)
        proc_handle = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc_handle.communicate()

        run_dir = ""
        for line in stdout.splitlines():
            if line.startswith("RUN_DIR="):
                run_dir = line.split("=", 1)[1].strip()
                break

        status = "ok" if proc_handle.returncode == 0 else "failed"
        rows.append(
            {
                "experiment": letter,
                "exp_name": exp_name,
                "config": cfg,
                "status": status,
                "exit_code": str(proc_handle.returncode),
                "run_dir": run_dir,
            }
        )
        _write_index(index_path, rows)

        sys.stdout.write(stdout)
        sys.stderr.write(stderr)

        if proc_handle.returncode != 0 and not bool(args.continue_on_error):
            print(
                f"Stopped on failed experiment {letter} (exit={proc_handle.returncode}).",
                flush=True,
            )
            return int(proc_handle.returncode)

    print(f"Batch complete. INDEX={index_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
