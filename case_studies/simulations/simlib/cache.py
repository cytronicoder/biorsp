"""
Geometry caching for simulation benchmarks.

Caches expensive computations like coordinate generation, polar transforms,
and sector indices to avoid recomputation across replicates/genes.
"""

import hashlib
from typing import Any, Dict, List, Optional

import numpy as np


class GeometryCache:
    """
    LRU-like cache for geometry objects.

    Stores:
    - coords: (N, 2) spatial coordinates
    - center: (2,) vantage point
    - r_norm: (N,) normalized radii
    - theta: (N,) angles
    - sector_indices: List[np.ndarray] per-sector cell indices
    """

    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _make_key(self, **kwargs) -> str:
        """Generate cache key from kwargs."""

        items = sorted(kwargs.items())
        key_str = str(items)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieve cached geometry."""
        key = self._make_key(**kwargs)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, value: Dict[str, Any], **kwargs):
        """Store geometry in cache."""
        key = self._make_key(**kwargs)

        if len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = value

    def clear(self):
        """Clear cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
        }

    def get_stats(self) -> Dict[str, int]:
        """Alias for stats() for compatibility."""
        return self.stats()


_GEOMETRY_CACHE = GeometryCache(max_size=200)


def get_cache() -> GeometryCache:
    """Get global geometry cache."""
    return _GEOMETRY_CACHE


def clear_cache():
    """Clear global cache."""
    _GEOMETRY_CACHE.clear()


def cache_geometry(
    coords: np.ndarray,
    center: np.ndarray,
    r_norm: np.ndarray,
    theta: np.ndarray,
    sector_indices: List[np.ndarray],
    **key_kwargs,
):
    """Cache geometry objects."""
    value = {
        "coords": coords,
        "center": center,
        "r_norm": r_norm,
        "theta": theta,
        "sector_indices": sector_indices,
    }
    _GEOMETRY_CACHE.put(value, **key_kwargs)


def get_cached_geometry(**key_kwargs) -> Optional[Dict[str, Any]]:
    """Retrieve cached geometry."""
    return _GEOMETRY_CACHE.get(**key_kwargs)
