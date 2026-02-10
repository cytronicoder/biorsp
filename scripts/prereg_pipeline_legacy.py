#!/usr/bin/env python3
"""Run the BioRSP evaluation pipeline within strata."""

from __future__ import annotations

import argparse
import warnings

from biorsp.evaluation import run_prereg_pipeline


def main() -> int:
    """Main CLI entry point for BioRSP evaluation.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="BioRSP preregistered evaluation pipeline"
    )
    parser.add_argument(
        "--config",
        default="configs/biorsp_prereg.json",
        help="Path to pipeline config",
    )

    args = parser.parse_args()
    warnings.warn(
        "scripts/prereg_pipeline_legacy.py is deprecated; use scripts/prereg_pipeline.py.",
        DeprecationWarning,
        stacklevel=2,
    )
    run_prereg_pipeline(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
