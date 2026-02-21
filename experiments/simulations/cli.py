#!/usr/bin/env python3
"""Canonical CLI for simulation orchestration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.simulations.run_experiment import main as run_one_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulation suite CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser(
        "simulations", help="Run one preregistered simulation experiment"
    )
    sim.add_argument("exp", help="Experiment key A-J")
    sim.add_argument("--config", required=True, type=str)
    sim.add_argument(
        "--out",
        "--out_root",
        dest="out_root",
        default="experiments/simulations/_results",
    )
    sim.add_argument("--tag", default=None)
    sim.add_argument("--seed", type=int, default=None)
    sim.add_argument("--n_jobs", type=int, default=None)
    sim.add_argument("--batch_id", default=None)
    sim.add_argument("--dry_run", action="store_true")
    sim.add_argument("--strict_prereg", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "simulations":
        forwarded = [
            "--exp",
            str(args.exp).upper(),
            "--config",
            str(args.config),
            "--out_root",
            str(args.out_root),
        ]
        if args.tag:
            forwarded.extend(["--tag", str(args.tag)])
        if args.seed is not None:
            forwarded.extend(["--seed", str(int(args.seed))])
        if args.n_jobs is not None:
            forwarded.extend(["--n_jobs", str(int(args.n_jobs))])
        if args.batch_id:
            forwarded.extend(["--batch_id", str(args.batch_id)])
        if bool(args.dry_run):
            forwarded.append("--dry_run")
        if bool(args.strict_prereg):
            forwarded.append("--strict_prereg")
        return int(run_one_main(forwarded))
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
