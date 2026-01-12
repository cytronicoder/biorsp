"""
Deterministic RNG utilities for reproducible simulations.

Ensures that the same seed + condition parameters always produce identical results.
"""

import hashlib
from typing import Any

import numpy as np
from numpy.random import Generator, SeedSequence


def make_rng(seed: int, *tags: str) -> Generator:
    """
    Create a deterministic RNG from a base seed and condition tags.

    Uses SeedSequence to derive independent streams. Same seed + same tags = same stream.

    Parameters
    ----------
    seed : int
        Base seed for the simulation run
    *tags : str
        Tags to identify the condition (e.g., "calibration", "rep_0", "disk")

    Returns
    -------
    Generator
        Numpy random generator with deterministic stream
    """

    tag_str = "_".join(str(t) for t in tags)
    tag_hash = int(hashlib.sha256(tag_str.encode()).hexdigest()[:16], 16)

    ss = SeedSequence(seed, spawn_key=(tag_hash,))
    return np.random.default_rng(ss)


def condition_key(*parts: Any) -> str:
    """
    Generate stable condition identifier from parameters.

    Parameters
    ----------
    *parts : Any
        Condition parameters (shape, N, distortion, etc.)

    Returns
    -------
    str
        Stable condition identifier for reproducibility
    """
    return "_".join(str(p) for p in parts)


def seed_all(seed: int) -> None:
    """
    Seed all random number generators for full reproducibility.

    Parameters
    ----------
    seed : int
        Master seed
    """
    np.random.seed(seed)
