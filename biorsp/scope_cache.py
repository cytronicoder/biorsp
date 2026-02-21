"""Scope-level caching for staged BioRSP discovery pipelines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _stable_hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_hash_array(arr: np.ndarray) -> str:
    view = np.ascontiguousarray(arr).view(np.uint8)
    return _stable_hash_bytes(view.tobytes())


def _sanitize_scope_id(scope_id: str) -> str:
    keep = []
    for ch in str(scope_id):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _build_bin_id(angles: np.ndarray, bins: int) -> np.ndarray:
    if int(bins) <= 0:
        raise ValueError("bins must be a positive integer.")
    bins_i = int(bins)
    bin_width = (2.0 * np.pi) / float(bins_i)
    bin_id = np.floor(
        (np.asarray(angles, dtype=float) % (2.0 * np.pi)) / bin_width
    ).astype(np.int32)
    return np.clip(bin_id, 0, bins_i - 1)


def _build_cells_in_bin(bin_id: np.ndarray, bins: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for b in range(int(bins)):
        out.append(np.flatnonzero(bin_id == b).astype(np.int32))
    return out


def _build_perm_indices(donor_ids: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    donor_arr = np.asarray(donor_ids)
    n_cells = int(donor_arr.size)
    n_perm_i = int(n_perm)
    if n_perm_i <= 0:
        raise ValueError("n_perm must be positive.")

    rng = np.random.default_rng(int(seed))
    perm_indices = np.tile(np.arange(n_cells, dtype=np.int32), (n_perm_i, 1))

    unique_donors = np.unique(donor_arr)
    for donor in unique_donors:
        idx = np.flatnonzero(donor_arr == donor).astype(np.int32)
        if idx.size <= 1:
            continue
        for k in range(n_perm_i):
            perm_indices[k, idx] = idx[rng.permutation(idx.size)]
    return perm_indices


@dataclass(frozen=True)
class ScopeCacheMetadata:
    scope_id: str
    bins: int
    n_perm: int
    seed: int
    n_cells: int
    donor_hash: str
    angles_hash: str
    cache_version: int = 1

    def to_json(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "bins": int(self.bins),
            "n_perm": int(self.n_perm),
            "seed": int(self.seed),
            "n_cells": int(self.n_cells),
            "donor_hash": str(self.donor_hash),
            "angles_hash": str(self.angles_hash),
            "cache_version": int(self.cache_version),
        }

    @staticmethod
    def from_json(payload: dict[str, Any]) -> "ScopeCacheMetadata":
        return ScopeCacheMetadata(
            scope_id=str(payload["scope_id"]),
            bins=int(payload["bins"]),
            n_perm=int(payload["n_perm"]),
            seed=int(payload["seed"]),
            n_cells=int(payload["n_cells"]),
            donor_hash=str(payload["donor_hash"]),
            angles_hash=str(payload["angles_hash"]),
            cache_version=int(payload.get("cache_version", 1)),
        )


@dataclass
class ScopeCache:
    scope_id: str
    angles: np.ndarray
    bin_id: np.ndarray
    cells_in_bin: list[np.ndarray]
    bin_counts_total: np.ndarray
    perm_indices: np.ndarray
    metadata: ScopeCacheMetadata
    cache_path: Path
    loaded_from_disk: bool


def _cache_paths(
    cache_dir: Path, scope_id: str, bins: int, n_perm: int, seed: int
) -> tuple[Path, Path]:
    safe_scope = _sanitize_scope_id(scope_id)
    stem = f"{safe_scope}.bins{int(bins)}.perm{int(n_perm)}.seed{int(seed)}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.meta.json"


def _build_metadata(
    *,
    scope_id: str,
    bins: int,
    n_perm: int,
    seed: int,
    angles: np.ndarray,
    donor_ids: np.ndarray,
) -> ScopeCacheMetadata:
    donor_str = np.asarray(donor_ids).astype(str)
    donor_hash = _stable_hash_bytes("|".join(donor_str.tolist()).encode("utf-8"))
    angles_hash = _stable_hash_array(np.asarray(angles, dtype=np.float64))
    return ScopeCacheMetadata(
        scope_id=str(scope_id),
        bins=int(bins),
        n_perm=int(n_perm),
        seed=int(seed),
        n_cells=int(np.asarray(angles).size),
        donor_hash=donor_hash,
        angles_hash=angles_hash,
        cache_version=1,
    )


def _meta_matches(left: ScopeCacheMetadata, right: ScopeCacheMetadata) -> bool:
    return left.to_json() == right.to_json()


def build_or_load_scope_cache(
    *,
    scope_id: str,
    angles: np.ndarray,
    donor_ids: np.ndarray,
    bins: int,
    n_perm: int,
    seed: int,
    cache_dir: Path | str,
) -> ScopeCache:
    """Build/load scope-level cache with deterministic invalidation."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = _cache_paths(cache_root, scope_id, bins, n_perm, seed)

    angles_arr = np.asarray(angles, dtype=float).ravel()
    donor_arr = np.asarray(donor_ids)
    if donor_arr.size != angles_arr.size:
        raise ValueError("donor_ids and angles must have the same length.")

    expected_meta = _build_metadata(
        scope_id=scope_id,
        bins=bins,
        n_perm=n_perm,
        seed=seed,
        angles=angles_arr,
        donor_ids=donor_arr,
    )

    if npz_path.exists() and meta_path.exists():
        try:
            on_disk = ScopeCacheMetadata.from_json(
                json.loads(meta_path.read_text(encoding="utf-8"))
            )
            if _meta_matches(on_disk, expected_meta):
                loaded = np.load(npz_path, allow_pickle=False)
                bin_id = loaded["bin_id"].astype(np.int32, copy=False)
                bin_counts_total = loaded["bin_counts_total"].astype(
                    np.int32, copy=False
                )
                perm_indices = loaded["perm_indices"].astype(np.int32, copy=False)
                cells_in_bin = _build_cells_in_bin(bin_id, int(bins))
                return ScopeCache(
                    scope_id=str(scope_id),
                    angles=angles_arr,
                    bin_id=bin_id,
                    cells_in_bin=cells_in_bin,
                    bin_counts_total=bin_counts_total,
                    perm_indices=perm_indices,
                    metadata=expected_meta,
                    cache_path=npz_path,
                    loaded_from_disk=True,
                )
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            pass

    bin_id = _build_bin_id(angles_arr, int(bins))
    bin_counts_total = np.bincount(bin_id, minlength=int(bins)).astype(np.int32)
    cells_in_bin = _build_cells_in_bin(bin_id, int(bins))
    perm_indices = _build_perm_indices(donor_arr, int(n_perm), int(seed))

    np.savez_compressed(
        npz_path,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
        perm_indices=perm_indices,
    )
    meta_path.write_text(
        json.dumps(expected_meta.to_json(), indent=2), encoding="utf-8"
    )

    return ScopeCache(
        scope_id=str(scope_id),
        angles=angles_arr,
        bin_id=bin_id,
        cells_in_bin=cells_in_bin,
        bin_counts_total=bin_counts_total,
        perm_indices=perm_indices,
        metadata=expected_meta,
        cache_path=npz_path,
        loaded_from_disk=False,
    )
