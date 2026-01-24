"""Contract-compliant archetype benchmark runner.

This script standardizes outputs across benchmarks by enforcing the benchmark
contract (runs.csv, summary.csv, manifest.json, report.md, figures, debug).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.benchmarks.simlib.io_contract import BenchmarkContractConfig, init_run_dir
from analysis.benchmarks.simlib.runner_harness import (
    compute_binomial_ci,
    finalize_contract,
    normalize_labels,
    normalize_scores_df,
    split_train_test,
)
from biorsp import BioRSPConfig
from biorsp.plotting.standard import make_standard_plot_set
from biorsp.simulations import datasets, expression, rng, scoring, shapes, sweeps
from biorsp.utils.labels import (
    ABSTAIN_LABEL,
    CANONICAL_ARCHETYPES,
    classify_from_thresholds,
    normalize_archetype_label,
    normalize_archetype_series,
)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARCHETYPE_SPECS: dict[tuple[str, str], dict[str, Any]] = {
    ("high", "iid"): {"pattern_variant": "iid", "archetype_name": "Ubiquitous"},
    ("high", "structured"): {"pattern_variant": "radial_gradient", "archetype_name": "Gradient"},
    ("low", "iid"): {"pattern_variant": "iid", "archetype_name": "Basal"},
    ("low", "structured"): {"pattern_variant": "wedge_core", "archetype_name": "Patchy"},
}


def get_pattern_for_archetype(coverage_regime: str, organization_regime: str) -> str:
    key = (coverage_regime, organization_regime)
    if key not in ARCHETYPE_SPECS:
        raise ValueError(f"Unknown archetype combination: {key}")
    if organization_regime == "iid":
        return "iid"
    return ARCHETYPE_SPECS[key]["pattern_variant"]


EPS = 1e-6
MIN_EXPR_CELLS = 50


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit_clipped(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def calibrate_intercept(
    prob: np.ndarray, target: float, tol: float = 1e-3, max_iter: int = 60
) -> tuple[float, float]:
    """Find intercept shift so mean(sigmoid(logit(prob) + b)) ≈ target.

    Parameters
    ----------
    prob : np.ndarray
        Array of probabilities in (0, 1).
    target : float
        Desired mean prevalence between 0 and 1.
    tol : float, optional
        Convergence tolerance (default 1e-3).
    max_iter : int, optional
        Maximum number of iterations (default 60).

    Returns
    -------
    tuple[float, float]
        Intercept shift and achieved mean prevalence.
    """

    if not (0 < target < 1):
        raise ValueError(f"Target prevalence must lie in (0,1); got {target}")

    logits = _logit_clipped(np.asarray(prob, dtype=float))

    def _mean_with(b: float) -> float:
        return float(_sigmoid(logits + b).mean())

    lo, hi = -14.0, 14.0
    mid = 0.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        m = _mean_with(mid)
        if abs(m - target) <= tol:
            return mid, m
        if m < target:
            lo = mid
        else:
            hi = mid

    return mid, _mean_with(mid)


def make_truth_spec(archetype: str, gen: np.random.Generator) -> dict[str, Any]:
    """Generate synthetic pattern specification for an archetype.

    Parameters
    ----------
    archetype : str
        Archetype name (may be canonical or informal).
    gen : np.random.Generator
        RNG to draw random parameters.

    Returns
    -------
    dict
        Spec containing keys 'pattern_family', 'pattern_variant',
        'target_prevalence', and 'generator_params'.
    """

    label = normalize_archetype_label(archetype)
    if label == "Ubiquitous":
        base = float(gen.uniform(0.78, 0.9))
        return {
            "pattern_family": "ubiquitous",
            "pattern_variant": "uniform",
            "target_prevalence": base,
            "generator_params": {"base": base, "abundance": 1.1e-3},
        }

    if label == "Basal":
        base = float(gen.uniform(0.04, 0.12))
        return {
            "pattern_family": "basal",
            "pattern_variant": "uniform",
            "target_prevalence": base,
            "generator_params": {"base": base, "abundance": 8.0e-4},
        }

    if label == "Gradient":
        variants = [
            ("radial_gradient", {"direction": "outward", "strength": float(gen.uniform(0.5, 0.9))}),
            ("core", {"steepness": float(gen.uniform(3.5, 5.5))}),
            ("rim", {"steepness": float(gen.uniform(3.5, 5.5))}),
            ("halfplane_gradient", {"phi": float(gen.uniform(-np.pi, np.pi))}),
        ]
        idx = int(gen.integers(0, len(variants)))
        variant, params = variants[idx]
        return {
            "pattern_family": "gradient",
            "pattern_variant": variant,
            "target_prevalence": float(gen.uniform(0.58, 0.72)),
            "generator_params": {**params, "abundance": 1.2e-3},
        }

    if label == "Patchy":
        variants = [
            (
                "wedge_core",
                {
                    "angle_center": float(gen.uniform(-np.pi, np.pi)),
                    "width_rad": float(gen.uniform(np.pi / 5, np.pi / 3)),
                    "steepness": float(gen.uniform(3.5, 5.0)),
                },
            ),
            (
                "wedge_rim",
                {
                    "angle_center": float(gen.uniform(-np.pi, np.pi)),
                    "width_rad": float(gen.uniform(np.pi / 5, np.pi / 3)),
                    "steepness": float(gen.uniform(3.5, 5.0)),
                },
            ),
        ]
        idx = int(gen.integers(0, len(variants)))
        variant, params = variants[idx]
        return {
            "pattern_family": "patchy",
            "pattern_variant": variant,
            "target_prevalence": float(gen.uniform(0.18, 0.32)),
            "generator_params": {**params, "abundance": 1.0e-3},
        }

    raise ValueError(f"Unsupported archetype: {archetype}")


def _simulate_expression(
    coords: np.ndarray,
    libsize: np.ndarray,
    spec: dict[str, Any],
    gen: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    params = dict(spec.get("generator_params", {}))
    field_params = {k: v for k, v in params.items() if k != "abundance"}
    field = expression.generate_signal_field(coords, spec["pattern_variant"], params=field_params)

    b, achieved = calibrate_intercept(field, spec["target_prevalence"])
    prob = _sigmoid(_logit_clipped(field) + b)
    fg_mask = gen.binomial(1, prob).astype(bool)

    counts = expression.generate_expression_from_field(
        prob,
        libsize,
        gen,
        expr_model="nb",
        params={"abundance": params.get("abundance", 1e-3), "phi": 8.0},
    )
    prevalence_empirical = float(fg_mask.mean())
    return counts, fg_mask, prevalence_empirical, b


def _assert_non_empty(df: pd.DataFrame, context: str) -> None:
    if df.empty:
        raise ValueError(f"{context} is empty; cannot proceed")


def write_contract_failure(paths: dict[str, Path], err: Exception, df: pd.DataFrame) -> None:
    debug_dir = paths.get("debug", paths["root"] / "debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    outfile = debug_dir / "contract_failure.txt"
    lines = [
        "Contract validation failed:",
        str(err),
        "",
        "Columns:",
        ", ".join(map(str, df.columns)),
        "",
        "Head:",
        df.head().to_string(),
    ]
    outfile.write_text("\n".join(lines))


def reconstruct_truth(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    condition_key = rng.condition_key(
        row["shape"], int(row["n_cells"]), row["coverage_regime"], row["organization_regime"]
    )
    gen = rng.make_rng(int(row["seed"]), "Archetype", condition_key)
    coords, _ = shapes.generate_coords(row["shape"], int(row["n_cells"]), gen)
    libsize = expression.simulate_library_size(
        int(row["n_cells"]), gen, model="lognormal", params={"mean": 2000, "std": 0.5}
    )
    params = json.loads(row.get("generator_params_json", "{}"))
    spec = {
        "pattern_family": row["pattern_family"],
        "pattern_variant": row["pattern_variant"],
        "target_prevalence": float(row["target_prevalence"]),
        "generator_params": params,
    }
    _, fg_mask, _, _ = _simulate_expression(coords, libsize, spec, gen)
    return coords, fg_mask


def _run_single_condition(config_dict: dict, seed: int, rsp_config: BioRSPConfig) -> dict:
    """Generate one replicate and score it."""

    def _true_archetype() -> str:
        key = (coverage_regime, organization_regime)
        if key not in ARCHETYPE_SPECS:
            raise ValueError(f"Unknown archetype combination: {key}")
        return normalize_archetype_label(ARCHETYPE_SPECS[key]["archetype_name"])

    shape = config_dict["shape"]
    n_cells = int(config_dict["N"])
    coverage_regime = config_dict["coverage_regime"]
    organization_regime = config_dict["organization_regime"]

    condition_key = rng.condition_key(shape, n_cells, coverage_regime, organization_regime)
    gen = rng.make_rng(seed, "Archetype", condition_key)

    coords, _ = shapes.generate_coords(shape, n_cells, gen)
    center_x, center_y = float(np.median(coords[:, 0])), float(np.median(coords[:, 1]))
    libsize = expression.simulate_library_size(
        n_cells, gen, model="lognormal", params={"mean": 2000, "std": 0.5}
    )

    truth_spec = make_truth_spec(_true_archetype(), gen)
    counts, fg_mask, prevalence_empirical, calib_b = _simulate_expression(
        coords, libsize, truth_spec, gen
    )
    n_fg = int((counts > 0).sum())

    abstain_flag = False
    abstain_reason = "ok"

    if abs(prevalence_empirical - truth_spec["target_prevalence"]) > 0.05:
        abstain_flag = True
        abstain_reason = "prevalence_calibration_failed"

    adata = datasets.package_as_anndata(
        coords, counts[:, None], var_names=["factorial_gene"], obs_meta=None, embedding_key="X_sim"
    )

    t0 = time.time()
    results_df = scoring.score_dataset(adata, genes=["factorial_gene"], config=rsp_config)
    elapsed = time.time() - t0

    coverage = np.nan
    spatial_score = np.nan
    directionality = np.nan

    if results_df.empty:
        abstain_flag = True
        abstain_reason = "no_scores"
    else:
        row = results_df.iloc[0]
        coverage = float(row.get("Coverage", np.nan))
        spatial_score = float(row.get("Spatial_Bias_Score", row.get("Spatial_Score", np.nan)))
        directionality = float(row.get("Directionality", np.nan))
        abstain_flag = abstain_flag or bool(row.get("abstain_flag", False))
        if abstain_reason == "ok":
            abstain_reason = row.get("abstain_reason", "ok")

    if n_fg < MIN_EXPR_CELLS and abstain_reason == "ok":
        abstain_flag = True
        abstain_reason = "insufficient_support"

    if (np.isnan(coverage) or np.isnan(spatial_score)) and abstain_reason == "ok":
        abstain_flag = True
        abstain_reason = "nan_metrics"

    return {
        "status": "abstain" if abstain_flag else "ok",
        "abstain_flag": abstain_flag,
        "abstain_reason": abstain_reason,
        "Coverage": coverage,
        "Spatial_Score": spatial_score,
        "Directionality": directionality,
        "pattern_family": truth_spec["pattern_family"],
        "pattern_variant": truth_spec["pattern_variant"],
        "target_prevalence": truth_spec["target_prevalence"],
        "prevalence_empirical": prevalence_empirical,
        "prevalence_calibration_b": calib_b,
        "generator_params_json": json.dumps(truth_spec["generator_params"]),
        "Archetype_true": _true_archetype(),
        "Archetype_pred": ABSTAIN_LABEL if abstain_flag else _true_archetype(),
        "time_seconds": elapsed,
        "n_cells": n_cells,
        "shape": shape,
        "center_x": center_x,
        "center_y": center_y,
        "n_fg": n_fg,
    }


def derive_thresholds(runs_df: pd.DataFrame) -> dict:
    """Derive default thresholds (S_cut, C_cut) from runs.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame of runs containing 'Spatial_Score' and 'organization_regime'.

    Returns
    -------
    dict
        Thresholds with keys 'S_cut', 'C_cut', and 'source'.
    """
    base = runs_df.loc[~runs_df["abstain_flag"]].copy()
    null_mask = base["organization_regime"] == "iid"
    s_cut = 0.18
    source = "fixed_default"
    if null_mask.sum() >= 8 and base.loc[null_mask, "Spatial_Score"].notna().sum() >= 5:
        s_cut = float(np.quantile(base.loc[null_mask, "Spatial_Score"].dropna(), 0.95))
        source = "iid_0.95_quantile"
    c_cut = 0.30
    return {"S_cut": s_cut, "C_cut": c_cut, "source": source}


def _bootstrap_macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 500
) -> tuple[float, float, float]:
    labels = list(CANONICAL_ARCHETYPES)
    rng_local = np.random.default_rng(0)

    def _macro_f1(sample_idx: np.ndarray) -> float:
        sample_true = y_true[sample_idx]
        sample_pred = y_pred[sample_idx]
        f1s = []
        for label in labels:
            tp = np.sum((sample_true == label) & (sample_pred == label))
            fp = np.sum((sample_true != label) & (sample_pred == label))
            fn = np.sum((sample_true == label) & (sample_pred != label))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1s.append(f1)
        return float(np.mean(f1s))

    base = _macro_f1(np.arange(len(y_true)))
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng_local.choice(len(y_true), size=len(y_true), replace=True)
        stats[i] = _macro_f1(idx)
    low, high = np.quantile(stats, [0.025, 0.975])
    return base, float(low), float(high)


def build_summary(runs_df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Build summary metrics with confidence intervals for a split.

    Parameters
    ----------
    runs_df : pd.DataFrame
        Runs dataframe including 'Archetype_true', 'Archetype_pred' and 'split'.
    split : str, optional
        Which split to summarize ('train' or 'test'), by default 'test'.

    Returns
    -------
    pd.DataFrame
        Summary dataframe with metrics such as accuracy, macro_f1, and recall.
    """
    runs_df = runs_df.copy()
    subset = runs_df[runs_df.get("split", split) == split]
    if subset.empty:
        raise ValueError(f"No rows available for split '{split}' to build summary")

    scored_df = subset[subset["Archetype_pred"] != ABSTAIN_LABEL]
    n_total = len(subset)
    n_tested = len(scored_df)
    n_abstained = n_total - n_tested
    abstain_rate = n_abstained / n_total if n_total > 0 else np.nan

    summary_rows = [
        {
            "metric": "n_tested",
            "group_keys": json.dumps({"scope": split}),
            "mean": n_tested,
            "std": float(np.nan),
            "n": 1,
            "ci_low": n_tested,
            "ci_high": n_tested,
            "method": "count",
        },
        {
            "metric": "n_abstained",
            "group_keys": json.dumps({"scope": split}),
            "mean": n_abstained,
            "std": float(np.nan),
            "n": 1,
            "ci_low": n_abstained,
            "ci_high": n_abstained,
            "method": "count",
        },
        {
            "metric": "abstention_rate",
            "group_keys": json.dumps({"scope": split}),
            "mean": abstain_rate,
            "std": float(np.nan),
            "n": n_total,
            "ci_low": compute_binomial_ci(int(n_abstained), n_total)[0] if n_total > 0 else np.nan,
            "ci_high": compute_binomial_ci(int(n_abstained), n_total)[1] if n_total > 0 else np.nan,
            "method": "wilson",
        },
    ]

    if n_tested == 0:
        return pd.DataFrame(summary_rows)

    correct = scored_df["Archetype_pred"] == scored_df["Archetype_true"]
    n = len(scored_df)
    k = int(correct.sum())
    acc_low, acc_high = compute_binomial_ci(k, n)
    macro_f1, f1_low, f1_high = _bootstrap_macro_f1(
        scored_df["Archetype_true"].to_numpy(), scored_df["Archetype_pred"].to_numpy()
    )

    summary_rows.extend(
        [
            {
                "metric": "accuracy",
                "group_keys": json.dumps({"scope": split}),
                "mean": k / n if n > 0 else np.nan,
                "std": float(correct.std()) if n > 1 else float(np.nan),
                "n": n,
                "ci_low": acc_low,
                "ci_high": acc_high,
                "method": "wilson",
            },
            {
                "metric": "macro_f1",
                "group_keys": json.dumps({"scope": split}),
                "mean": macro_f1,
                "std": float(np.nan),
                "n": n,
                "ci_low": f1_low,
                "ci_high": f1_high,
                "method": "bootstrap",
            },
        ]
    )

    for label in CANONICAL_ARCHETYPES:
        mask = scored_df["Archetype_true"] == label
        n_label = int(mask.sum())
        if n_label == 0:
            continue
        k_label = int((scored_df.loc[mask, "Archetype_pred"] == label).sum())
        low, high = compute_binomial_ci(k_label, n_label)
        summary_rows.append(
            {
                "metric": "recall",
                "group_keys": json.dumps({"archetype": label, "scope": split}),
                "mean": k_label / n_label,
                "std": float(np.nan),
                "n": n_label,
                "ci_low": low,
                "ci_high": high,
                "method": "wilson",
            }
        )

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description="Archetype benchmark (contract-compliant)")
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "analysis" / "benchmarks" / "outputs"),
        help="Base output directory",
    )
    parser.add_argument(
        "--run_id", type=str, default=None, help="Run identifier (default: timestamp)"
    )
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--n_reps", type=int, default=20)
    parser.add_argument("--N", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--shape", type=str, nargs="+", default=["disk", "peanut"])
    parser.add_argument(
        "--mode", type=str, choices=["quick", "validation", "publication"], default="quick"
    )
    parser.add_argument("--n_workers", type=int, default=-1)
    parser.add_argument(
        "--debug_panels", type=str, choices=["none", "minimal", "all"], default="minimal"
    )
    args = parser.parse_args()

    if args.mode == "quick":
        args.n_reps = 8
        args.N = [500, 2000]
        args.shape = ["disk", "peanut"]
    elif args.mode == "validation":
        args.n_reps = max(args.n_reps, 20)
    elif args.mode == "publication":
        args.n_reps = max(args.n_reps, 50)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    config = BenchmarkContractConfig(
        outdir=args.outdir,
        benchmark="archetypes",
        run_id=run_id,
        seed=args.seed,
        mode=args.mode,
    )
    paths = init_run_dir(config)
    diagnostics_dir = paths["root"] / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    rsp_config = BioRSPConfig(B=72, delta_deg=60.0, n_permutations=0)

    grid = sweeps.expand_grid(
        shape=args.shape,
        N=args.N,
        coverage_regime=["high", "low"],
        organization_regime=["structured", "iid"],
    )

    results = sweeps.run_replicates(
        _run_single_condition,
        grid,
        args.n_reps,
        seed_start=args.seed,
        progress=True,
        n_jobs=args.n_workers,
        fn_args=(rsp_config,),
    )

    if results.empty:
        raise RuntimeError(
            "No results produced; check label mapping / filtering / schema mismatch."
        )

    if "error" in results.columns and results["error"].notna().any():
        first_error = results[results["error"].notna()].iloc[0]
        raise RuntimeError(
            f"Error during replicate execution: {first_error['error']}\n{first_error.get('traceback', '')}"
        )

    results.rename(columns={"replicate": "replicate_id"}, inplace=True)
    results["run_id"] = run_id
    results["benchmark"] = "archetypes"
    results["mode"] = args.mode
    results["timestamp"] = datetime.now(timezone.utc).isoformat()
    results["seed"] = results["seed"].astype(int)
    results["status"] = results.get("status", "ok").fillna("ok")
    if "abstain_reason" not in results.columns:
        results["abstain_reason"] = "ok"

    # Canonical columns and split definition
    results = normalize_scores_df(results)
    results["case_id"] = results.apply(
        lambda r: f"{r['shape']}-{r['n_cells']}-{r['coverage_regime']}-{r['organization_regime']}-{r['seed']}",
        axis=1,
    )

    split = split_train_test(results, group_cols=["case_id"], test_frac=0.25, seed=args.seed)
    results["split"] = "train"
    results.loc[split.test_idx, "split"] = "test"

    results["Archetype_true"] = normalize_archetype_series(
        results["Archetype_true"], allow_abstain=False
    )

    train_df = results.loc[split.train_idx]
    thresholds = derive_thresholds(train_df)
    results["C_cut"] = thresholds["C_cut"]
    results["S_cut"] = thresholds["S_cut"]
    results["thresholds_source"] = thresholds["source"]

    preds: list[str] = []
    abstain_flags = results["abstain_flag"].to_numpy().astype(bool)
    abstain_reasons = results["abstain_reason"].tolist()
    for idx, (c, s, flag) in enumerate(
        zip(results["Coverage"], results["Spatial_Score"], abstain_flags)
    ):
        if flag or np.isnan(c) or np.isnan(s):
            preds.append(ABSTAIN_LABEL)
            abstain_flags[idx] = True
            if abstain_reasons[idx] == "ok":
                abstain_reasons[idx] = (
                    "nan_metrics" if np.isnan(c) or np.isnan(s) else abstain_reasons[idx]
                )
            continue
        preds.append(classify_from_thresholds(c, s, thresholds["C_cut"], thresholds["S_cut"]))

    results["abstain_flag"] = abstain_flags.astype(bool)
    results["abstain_reason"] = abstain_reasons
    results["Archetype_pred"] = preds
    results = normalize_labels(
        results, truth_col="Archetype_true", pred_col="Archetype_pred", allow_abstain_pred=True
    )

    mask_reason_fix = (results["abstain_flag"]) & (results["abstain_reason"] == "ok")
    results.loc[mask_reason_fix, "abstain_reason"] = "abstain"
    results["status"] = np.where(results["abstain_flag"], "abstain", results["status"])

    # Write thresholds artifact (train-derived)
    thresholds_path = paths["root"] / "thresholds_used.json"
    with open(thresholds_path, "w") as f:
        json.dump(
            {"thresholds": thresholds, "source": thresholds["source"], "n_train": len(train_df)},
            f,
            indent=2,
        )

    test_df = results[results["split"] == "test"]
    misclass_df = test_df[test_df["Archetype_true"] != test_df["Archetype_pred"]]
    if not misclass_df.empty:
        misclass_df.sort_values("Spatial_Score", ascending=False).to_csv(
            diagnostics_dir / "misclassified.csv", index=False
        )

    misclass_patterns = (
        test_df.groupby(
            ["Archetype_true", "Archetype_pred", "shape", "coverage_regime", "organization_regime"]
        )
        .size()
        .reset_index(name="count")
    )
    misclass_patterns.to_csv(diagnostics_dir / "misclassification_patterns.csv", index=False)

    # Summary with CI on test split
    summary_df = build_summary(results, split="test")

    figures = make_standard_plot_set(
        scores_df=results,
        outdir=paths["root"],
        thresholds={"C_cut": thresholds["C_cut"], "S_cut": thresholds["S_cut"]},
        truth_col="Archetype_true",
        pred_col="Archetype_pred",
        gene_col="gene" if "gene" in results.columns else "gene",
        title=f"Archetypes ({args.mode})",
        debug=args.debug_panels != "none",
    )

    # Report
    acc_row = summary_df[summary_df["metric"] == "accuracy"].head(1)
    macro_row = summary_df[summary_df["metric"] == "macro_f1"].head(1)
    abst_row = summary_df[summary_df["metric"] == "abstention_rate"].head(1)
    report_lines = [
        f"# Archetype Benchmark ({args.mode})",
        "",
        f"Run ID: {run_id}",
        f"Replicates: {len(results)}",
        f"Train/Test split: {len(train_df)}/{len(test_df)} cases",
    ]
    if not acc_row.empty:
        report_lines.append(f"Accuracy (test, non-abstain): {acc_row['mean'].iloc[0]:.3f}")
    if not macro_row.empty:
        report_lines.append(f"Macro F1 (test): {macro_row['mean'].iloc[0]:.3f}")
    if not abst_row.empty:
        report_lines.append(f"Abstention rate (test): {abst_row['mean'].iloc[0]:.3f}")

    manifest = {
        "benchmark": "archetypes",
        "config": config.to_dict(),
        "thresholds": thresholds,
        "n_rows": len(results),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }

    finalize_contract(
        paths["root"],
        runs_df=results,
        summary_df=summary_df,
        manifest=manifest,
        report_md="\n".join(report_lines),
        figures=figures,
    )

    print(f"✅ Archetype benchmark complete → {paths['root']}")


if __name__ == "__main__":
    main()
