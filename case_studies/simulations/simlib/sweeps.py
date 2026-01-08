"""
Parameter sweep utilities for simulation benchmarks.

Provides grid expansion and replicate execution helpers.
"""

import itertools
from typing import Any, Callable, Dict, Iterable, List

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

    Examples
    --------
    >>> expand_grid(shape=['disk', 'ellipse'], n=[100, 500])
    [
        {'shape': 'disk', 'n': 100},
        {'shape': 'disk', 'n': 500},
        {'shape': 'ellipse', 'n': 100},
        {'shape': 'ellipse', 'n': 500}
    ]
    """
    keys = kwargs.keys()
    values = kwargs.values()
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return configs


def run_replicates(
    fn: Callable,
    configs: List[Dict[str, Any]],
    n_reps: int,
    seed_start: int = 0,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Run function over parameter grid with replicates.

    Parameters
    ----------
    fn : Callable
        Function to run: fn(config, seed) -> dict of results
    configs : list of dict
        Parameter configurations
    n_reps : int
        Number of replicates per config
    seed_start : int, optional
        Starting seed
    progress : bool, optional
        Show progress bar

    Returns
    -------
    results_df : pd.DataFrame
        Results with config parameters + replicate columns

    Examples
    --------
    >>> def test_fn(config, seed):
    ...     return {'fpr': 0.05, 'power': 0.8}
    >>> configs = expand_grid(shape=['disk'], n=[100])
    >>> df = run_replicates(test_fn, configs, n_reps=3)
    """
    results = []

    total_runs = len(configs) * n_reps
    iterator = range(total_runs)
    if progress:
        iterator = tqdm(iterator, desc="Running replicates")

    for idx in iterator:
        config_idx = idx // n_reps
        rep_idx = idx % n_reps
        config = configs[config_idx]
        seed = seed_start + idx

        # Run function
        result = fn(config, seed)

        # Combine config + result
        row = {**config, "replicate": rep_idx, "seed": seed, **result}
        results.append(row)

    return pd.DataFrame(results)


def aggregate_replicates(
    runs_df: pd.DataFrame, group_by: List[str], metrics: List[str]
) -> pd.DataFrame:
    """
    Aggregate replicate results (mean, std, CI).

    Parameters
    ----------
    runs_df : pd.DataFrame
        Per-replicate results
    group_by : list of str
        Columns to group by
    metrics : list of str
        Metric columns to aggregate

    Returns
    -------
    summary_df : pd.DataFrame
        Aggregated statistics
    """
    agg_dict = {}
    for metric in metrics:
        agg_dict[f"{metric}_mean"] = (metric, "mean")
        agg_dict[f"{metric}_std"] = (metric, "std")
        agg_dict[f"{metric}_min"] = (metric, "min")
        agg_dict[f"{metric}_max"] = (metric, "max")

    summary_df = runs_df.groupby(group_by).agg(**agg_dict).reset_index()
    return summary_df


def stratify_by(
    runs_df: pd.DataFrame, strata_col: str, group_by: List[str], metrics: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Stratify results by column and aggregate within strata.

    Parameters
    ----------
    runs_df : pd.DataFrame
        Per-replicate results
    strata_col : str
        Column to stratify by
    group_by : list of str
        Columns to group by within strata
    metrics : list of str
        Metrics to aggregate

    Returns
    -------
    summaries : dict of pd.DataFrame
        Aggregated results per stratum
    """
    summaries = {}
    for stratum, stratum_df in runs_df.groupby(strata_col):
        summaries[stratum] = aggregate_replicates(stratum_df, group_by, metrics)
    return summaries


def replicate_seed(base_seed: int, config_idx: int, rep_idx: int) -> int:
    """
    Compute deterministic seed for a replicate.

    Parameters
    ----------
    base_seed : int
        Base seed for experiment
    config_idx : int
        Configuration index
    rep_idx : int
        Replicate index

    Returns
    -------
    seed : int
        Replicate seed
    """
    return base_seed + config_idx * 1000 + rep_idx
