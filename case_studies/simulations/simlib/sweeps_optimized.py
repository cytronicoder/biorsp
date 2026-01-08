import itertools
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

import pandas as pd
from tqdm import tqdm


def expand_grid(**kwargs: Iterable[Any]) -> List[Dict[str, Any]]:
    """
    Create Cartesian product of parameter values (like R's expand.grid).

    Parameters
    ----------
    **kwargs : dict of iterables
        Parameter names and values

    Returns
    -------
    configs : list of dict
        All parameter combinations
    """
    keys = kwargs.keys()
    values = kwargs.values()
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return configs


def _worker_init():
    """Initialize worker with BLAS thread limits."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _run_single_task(args):
    """Wrapper for parallel execution."""
    fn, config, seed, rep_idx, fn_args = args
    try:
        result = fn(config, seed, *fn_args)
        return {**config, "replicate": rep_idx, "seed": seed, **result}
    except Exception as e:
        import traceback

        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "config": config,
            "replicate": rep_idx,
            "seed": seed,
        }


def run_replicates(
    fn: Callable,
    configs: List[Dict[str, Any]],
    n_reps: int,
    seed_start: int = 0,
    progress: bool = True,
    n_jobs: int = 1,
    fn_args: tuple = (),
    checkpoint_every: int = 0,
    checkpoint_callback: Optional[Callable] = None,
    skip_completed: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Run function over parameter grid with replicates.

    Parameters
    ----------
    fn : Callable
        Function to run: fn(config, seed, *fn_args) -> dict of results
    configs : list of dict
        Parameter configurations
    n_reps : int
        Number of replicates per config
    seed_start : int, optional
        Starting seed
    progress : bool, optional
        Show progress bar
    n_jobs : int, optional
        Number of parallel jobs. If -1, use all available cores.
    fn_args : tuple, optional
        Additional arguments to pass to fn.
    checkpoint_every : int, optional
        Save checkpoint every N completed tasks (0 = no checkpointing)
    checkpoint_callback : Callable, optional
        Function to call with accumulated results for checkpointing
    skip_completed : Set[str], optional
        Set of checkpoint keys to skip (for resume)

    Returns
    -------
    results_df : pd.DataFrame
        Results with config parameters + replicate columns
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    # Build task list
    total_runs = len(configs) * n_reps
    tasks = []
    skipped_count = 0

    for idx in range(total_runs):
        config_idx = idx // n_reps
        rep_idx = idx % n_reps
        config = configs[config_idx]
        seed = seed_start + idx

        # Skip if already completed
        if skip_completed is not None:
            from .checkpoint import make_checkpoint_key

            key = make_checkpoint_key(config, rep_idx)
            if key in skip_completed:
                skipped_count += 1
                continue

        tasks.append((fn, config, seed, rep_idx, fn_args))

    if len(tasks) == 0:
        print("All tasks already completed (resume mode).")
        return pd.DataFrame()

    if skipped_count > 0:
        print(
            f"Resuming: {len(tasks)}/{total_runs} tasks to run ({skipped_count} already completed)"
        )

    results = []
    completed = 0

    if n_jobs > 1:
        # Parallel execution with BLAS control
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_worker_init) as executor:
            futures = {executor.submit(_run_single_task, task): task for task in tasks}

            iterator = as_completed(futures)
            if progress:
                iterator = tqdm(iterator, total=len(tasks), desc=f"Replicates (n_jobs={n_jobs})")

            for future in iterator:
                try:
                    row = future.result(timeout=3600)  # 1 hour timeout per task
                    if "error" in row:
                        print(f"\nError in task: {row['error']}")
                        print(row.get("traceback", ""))
                    else:
                        results.append(row)
                        completed += 1

                        # Checkpoint
                        if (
                            checkpoint_every > 0
                            and completed % checkpoint_every == 0
                            and checkpoint_callback is not None
                        ):
                            checkpoint_callback(results)
                except Exception as e:  # noqa: PERF203
                    print(f"\nFailed to retrieve result: {e}")
    else:
        # Serial execution
        iterator = tasks
        if progress:
            iterator = tqdm(tasks, desc="Replicates (serial)")

        for task in iterator:
            row = _run_single_task(task)
            if "error" in row:
                print(f"\nError in task: {row['error']}")
            else:
                results.append(row)
                completed += 1

                # Checkpoint
                if (
                    checkpoint_every > 0
                    and completed % checkpoint_every == 0
                    and checkpoint_callback is not None
                ):
                    checkpoint_callback(results)

    return pd.DataFrame(results)


def run_replicates_parallel_simple(
    fn: Callable,
    configs: List[Dict[str, Any]],
    n_reps: int,
    seed_start: int = 0,
    n_workers: int = 1,
) -> pd.DataFrame:
    """
    Simplified parallel version for backward compatibility.

    Use run_replicates() for full features (checkpointing, resume).
    """
    return run_replicates(
        fn=fn,
        configs=configs,
        n_reps=n_reps,
        seed_start=seed_start,
        progress=True,
        n_jobs=n_workers,
        fn_args=(),
    )
