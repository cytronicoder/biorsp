"""Deterministic parallel helpers for simulation workloads."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import is_dataclass
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _item_seed(item: Any) -> int | None:
    if isinstance(item, dict):
        seed = item.get("seed")
        return int(seed) if seed is not None else None
    if hasattr(item, "seed"):
        seed = getattr(item, "seed")
        return int(seed) if seed is not None else None
    if is_dataclass(item) and hasattr(item, "seed"):
        seed = getattr(item, "seed")
        return int(seed) if seed is not None else None
    return None


def _validate_items_have_seed(items: list[T]) -> None:
    missing = [idx for idx, item in enumerate(items) if _item_seed(item) is None]
    if missing:
        head = ",".join(str(i) for i in missing[:5])
        raise ValueError(
            "parallel_map requires every item to carry a deterministic `seed` "
            f"(missing at indices: {head}{'...' if len(missing) > 5 else ''})."
        )


def _call_indexed(func: Callable[[T], R], indexed: tuple[int, T]) -> tuple[int, R]:
    idx, item = indexed
    return idx, func(item)


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    n_jobs: int = 1,
    backend: str = "loky",
    chunk_size: int = 25,
    progress: bool = True,
) -> list[R]:
    """Apply `func` to items with deterministic, order-stable aggregation.

    Notes:
    - Every item must include a deterministic `seed`.
    - Output order is always aligned to input order, independent of scheduling.
    """
    seq = list(items)
    if not seq:
        return []
    _validate_items_have_seed(seq)

    jobs = max(1, int(n_jobs))
    chunks = max(1, int(chunk_size))
    backend_name = str(backend)
    indexed = list(enumerate(seq))

    if jobs == 1 or len(seq) == 1:
        if progress:
            print(f"[parallel_map] serial execution: n_items={len(seq)}")
        return [func(item) for item in seq]

    if progress:
        print(
            f"[parallel_map] n_items={len(seq)} n_jobs={jobs} "
            f"backend={backend_name} chunk_size={chunks}"
        )

    if backend_name in {"loky", "multiprocessing", "threading"}:
        try:
            from joblib import Parallel, delayed

            rows = Parallel(
                n_jobs=jobs,
                backend=backend_name,
                batch_size=chunks,
            )(delayed(_call_indexed)(func, pair) for pair in indexed)
            rows.sort(key=lambda x: x[0])
            return [row for _, row in rows]
        except Exception:
            # Fall back to stdlib implementations.
            # If joblib is unavailable for `loky`/`multiprocessing`,
            # prefer threading to avoid pickle issues for local callables.
            if backend_name in {"loky", "multiprocessing"}:
                backend_name = "threading"

    if backend_name == "threading":
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            rows = list(
                ex.map(
                    lambda pair: _call_indexed(func, pair), indexed, chunksize=chunks
                )
            )
        rows.sort(key=lambda x: x[0])
        return [row for _, row in rows]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs) as pool:
        rows = pool.map(
            _parallel_worker_adapter,
            [(func, pair) for pair in indexed],
            chunksize=chunks,
        )
    rows.sort(key=lambda x: x[0])
    return [row for _, row in rows]


def _parallel_worker_adapter(
    payload: tuple[Callable[[T], R], tuple[int, T]],
) -> tuple[int, R]:
    func, indexed = payload
    return _call_indexed(func, indexed)
