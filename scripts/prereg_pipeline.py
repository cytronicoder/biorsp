#!/usr/bin/env python3
"""Run preregistered BioRSP evaluation pipeline."""

from __future__ import annotations

import argparse

from biorsp.evaluation import run_prereg_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="BioRSP preregistered evaluation pipeline")
    parser.add_argument(
        "--config",
        default="configs/biorsp_prereg.json",
        help="Path to JSON config",
    )
    args = parser.parse_args()
    run_prereg_pipeline(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
