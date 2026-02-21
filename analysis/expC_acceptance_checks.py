#!/usr/bin/env python3
"""Acceptance checks for Experiment C abstention and power reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEY_COLS = ["N", "D", "sigma_eta", "pi_target", "beta"]


def _resolve_power_cols(summary: pd.DataFrame) -> tuple[str, str, str]:
    cond_col = (
        "conditional_power_alpha05"
        if "conditional_power_alpha05" in summary.columns
        else "power_alpha05"
    )
    oper_col = (
        "operational_power_alpha05"
        if "operational_power_alpha05" in summary.columns
        else cond_col
    )
    analyzable_col = (
        "analyzable_rate"
        if "analyzable_rate" in summary.columns
        else "__derived_analyzable_rate"
    )
    if analyzable_col == "__derived_analyzable_rate":
        num = pd.to_numeric(summary.get("n_non_underpowered"), errors="coerce")
        den = pd.to_numeric(summary.get("n_genes"), errors="coerce")
        summary[analyzable_col] = np.where(den > 0, num / den, np.nan)
    return cond_col, oper_col, analyzable_col


def _compute_strict_null_table(
    metrics: pd.DataFrame, n_min: int, tol: float
) -> pd.DataFrame:
    m = metrics.copy()
    m["underpowered"] = m["underpowered"].astype(bool)
    beta0 = m.loc[np.isclose(pd.to_numeric(m["beta"], errors="coerce"), 0.0)].copy()

    rows: list[dict[str, Any]] = []
    for keys, grp in beta0.groupby(KEY_COLS, sort=True):
        n_total = int(grp.shape[0])
        analyzable = grp.loc[~grp["underpowered"]]
        n_an = int(analyzable.shape[0])
        p = pd.to_numeric(analyzable["p_T"], errors="coerce")
        k = int(
            np.sum(
                np.isfinite(p.to_numpy(dtype=float)) & (p.to_numpy(dtype=float) <= 0.05)
            )
        )
        cond = float(k / n_an) if n_an > 0 else np.nan
        oper = float(k / n_total) if n_total > 0 else np.nan
        rows.append(
            {
                "N": int(keys[0]),
                "D": int(keys[1]),
                "sigma_eta": float(keys[2]),
                "pi_target": float(keys[3]),
                "beta": float(keys[4]),
                "n_total": n_total,
                "n_analyzable": n_an,
                "analyzable_rate": float(n_an / n_total) if n_total > 0 else np.nan,
                "typeI_alpha05_conditional": cond,
                "typeI_alpha05_operational": oper,
                "typeI_violation": bool(
                    n_an >= int(n_min)
                    and np.isfinite(cond)
                    and abs(cond - 0.05) > float(tol)
                ),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["typeI_alpha05_conditional", "analyzable_rate", "N"],
        ascending=[False, False, True],
    )
    return out


def _monotonic_violations(
    summary: pd.DataFrame,
    *,
    metric_col: str,
    analyzable_col: str,
    eps: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add_viols(
        sub: pd.DataFrame,
        order_col: str,
        fixed_cols: list[str],
        direction: str,
        metric: str,
    ) -> None:
        sub = sub.sort_values(order_col)
        x = pd.to_numeric(sub[order_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[metric_col], errors="coerce").to_numpy(dtype=float)
        a = pd.to_numeric(sub[analyzable_col], errors="coerce").to_numpy(dtype=float)
        for i in range(1, len(sub)):
            if not (np.isfinite(y[i - 1]) and np.isfinite(y[i])):
                continue
            delta = float(y[i] - y[i - 1])
            if delta < -float(eps):
                analyz_delta = (
                    float(a[i] - a[i - 1])
                    if np.isfinite(a[i - 1]) and np.isfinite(a[i])
                    else float("nan")
                )
                row = {c: sub.iloc[i][c] for c in fixed_cols}
                row.update(
                    {
                        "metric": metric,
                        "direction": direction,
                        "from_value": float(x[i - 1]),
                        "to_value": float(x[i]),
                        "delta": delta,
                        "analyzable_rate_delta": analyz_delta,
                        "explainable_by_analyzable_rate": bool(
                            np.isfinite(analyz_delta) and analyz_delta < 0.0
                        ),
                    }
                )
                out.append(row)

    base = summary.copy()
    for keys, grp in base.groupby(["pi_target", "N", "sigma_eta", "D"], sort=True):
        add_viols(
            grp,
            "beta",
            ["pi_target", "N", "sigma_eta", "D"],
            "beta_non_decreasing",
            metric_col,
        )
    for keys, grp in base.groupby(["pi_target", "N", "sigma_eta", "beta"], sort=True):
        add_viols(
            grp,
            "D",
            ["pi_target", "N", "sigma_eta", "beta"],
            "D_non_decreasing",
            metric_col,
        )
    for keys, grp in base.groupby(["pi_target", "D", "sigma_eta", "beta"], sort=True):
        add_viols(
            grp,
            "N",
            ["pi_target", "D", "sigma_eta", "beta"],
            "N_non_decreasing",
            metric_col,
        )
    return out


def _run_checks(
    summary_path: Path,
    metrics_path: Path,
    outdir: Path,
    *,
    n_min: int,
    tol: float,
    eps: float,
    before_summary: Path | None,
    before_metrics: Path | None,
) -> dict[str, Any]:
    summary = pd.read_csv(summary_path)
    metrics = pd.read_csv(metrics_path)

    cond_col, oper_col, analyzable_col = _resolve_power_cols(summary)

    viol_rows: list[dict[str, Any]] = []
    viol_rows.extend(
        _monotonic_violations(
            summary, metric_col=cond_col, analyzable_col=analyzable_col, eps=eps
        )
    )
    cond_n = len(viol_rows)
    oper_viol = _monotonic_violations(
        summary, metric_col=oper_col, analyzable_col=analyzable_col, eps=eps
    )
    for row in oper_viol:
        row["metric"] = "operational_power_alpha05"
    viol_rows.extend(oper_viol)

    strict = _compute_strict_null_table(metrics, n_min=n_min, tol=tol)

    outdir.mkdir(parents=True, exist_ok=True)
    strict_path = outdir / "strict_null_typei_by_cell.csv"
    viol_path = outdir / "monotonicity_violations.csv"
    strict.to_csv(strict_path, index=False)
    pd.DataFrame(viol_rows).to_csv(viol_path, index=False)

    before_after_path: Path | None = None
    if before_metrics is not None:
        b_metrics = pd.read_csv(before_metrics)
        b_strict = _compute_strict_null_table(b_metrics, n_min=n_min, tol=tol)
        merged = b_strict.merge(
            strict,
            on=KEY_COLS,
            how="outer",
            suffixes=("_before", "_after"),
        )
        before_after_path = outdir / "strict_null_before_after.csv"
        merged.to_csv(before_after_path, index=False)

    worst_after = strict.head(1)
    summary_json = {
        "summary_path": str(summary_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "n_min": int(n_min),
        "typeI_tolerance": float(tol),
        "monotonicity_epsilon": float(eps),
        "conditional_power_col": cond_col,
        "operational_power_col": oper_col,
        "analyzable_rate_col": analyzable_col,
        "n_monotonicity_violations": int(len(viol_rows)),
        "n_monotonicity_conditional": int(cond_n),
        "n_monotonicity_operational": int(len(oper_viol)),
        "n_typei_violations": int(strict["typeI_violation"].sum()),
        "worst_strict_null_after": worst_after.to_dict(orient="records"),
        "strict_null_path": str(strict_path.resolve()),
        "monotonicity_path": str(viol_path.resolve()),
        "before_after_path": (
            str(before_after_path.resolve()) if before_after_path is not None else None
        ),
    }

    (outdir / "acceptance_summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )
    return summary_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment C acceptance checks")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--before-summary", type=Path, default=None)
    parser.add_argument("--before-metrics", type=Path, default=None)
    parser.add_argument("--n_min", type=int, default=20)
    parser.add_argument("--typei_tol", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.05)
    args = parser.parse_args()

    outdir = args.outdir
    if outdir is None:
        outdir = args.summary.resolve().parent.parent / "diagnostics"

    res = _run_checks(
        summary_path=args.summary,
        metrics_path=args.metrics,
        outdir=outdir,
        n_min=int(args.n_min),
        tol=float(args.typei_tol),
        eps=float(args.epsilon),
        before_summary=args.before_summary,
        before_metrics=args.before_metrics,
    )

    print("acceptance_summary:")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
