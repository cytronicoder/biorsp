#!/usr/bin/env python3
"""CLI entrypoint for the preregistered BioRSP evaluation pipeline."""

from __future__ import annotations

import argparse

from biorsp.evaluation import run_prereg_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preregistered BioRSP evaluation pipeline."
    )
    parser.add_argument(
        "--config", required=True, help="Path to JSON config for prereg pipeline."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_prereg_pipeline(str(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
