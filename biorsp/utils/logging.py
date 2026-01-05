"""Logging utilities for BioRSP."""

import logging
import sys


def setup_logging(level: int = logging.INFO):
    """Set up basic logging to stdout."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(f"biorsp.{name}")
