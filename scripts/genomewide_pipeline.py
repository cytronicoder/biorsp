#!/usr/bin/env python3
"""Run genome-wide BioRSP screening pipeline."""

from __future__ import annotations

import argparse

from biorsp.genomewide import run_genomewide_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="BioRSP genome-wide screening pipeline")
    parser.add_argument(
        "--config",
        default="configs/biorsp_genomewide.json",
        help="Path to JSON config",
    )
    args = parser.parse_args()
    run_genomewide_pipeline(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
