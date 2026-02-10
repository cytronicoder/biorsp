"""Configuration loading utilities for BioRSP pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a pipeline config from a JSON file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix.lower() != ".json":
        raise ValueError(
            f"Unsupported config format for '{config_path}'. Use a .json config file."
        )

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in config '{config_path}' at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid config root in '{config_path}': expected JSON object, got {type(data).__name__}."
        )
    return data
