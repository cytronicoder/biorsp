"""Deterministic seeding helpers that avoid Python's salted hash."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _token_to_str(token: Any) -> str:
    try:
        return json.dumps(token, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return repr(token)


def stable_seed(master_seed: int, *tokens: Any) -> int:
    """Derive a stable uint32 seed from a master seed and arbitrary tokens."""
    parts = [str(int(master_seed))] + [_token_to_str(tok) for tok in tokens]
    payload = "|".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    offset = int.from_bytes(digest[:8], "big")
    return int((int(master_seed) + offset) % (2**32))


def stable_int_hash(obj: Any) -> int:
    """Stable non-negative integer hash for JSON-serializable payloads."""
    payload = _token_to_str(obj).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def seed_for_cell(
    master_seed: int, exp_name: str, cell_spec_dict: dict[str, Any]
) -> int:
    """Stable seed derivation for experiment cells."""
    return stable_seed(int(master_seed), str(exp_name), cell_spec_dict)


def rng_from_seed(seed: int) -> np.random.Generator:
    """Construct a NumPy Generator from seed."""
    return np.random.default_rng(int(seed))
