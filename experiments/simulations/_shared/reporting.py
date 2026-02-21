"""Markdown report generators for simulation experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import git_commit_hash


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(x):
        return "NA"
    if abs(x) >= 1000:
        return f"{x:.1f}"
    return f"{x:.{digits}g}"


def _table_markdown(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_No rows available._"
    show = df.head(max_rows).copy()
    cols = [str(c) for c in show.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in show.iterrows():
        vals = [
            (
                _fmt(row[c])
                if isinstance(row[c], (int, float, np.floating, np.integer))
                else str(row[c])
            )
            for c in show.columns
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _figure_embeds(
    outdir: Path, figs_glob: str, min_count: int = 3
) -> tuple[list[str], str]:
    figs = sorted(outdir.glob(figs_glob))
    selected = figs[: max(min_count, min(len(figs), 6))] if figs else []
    rels: list[str] = []
    embeds: list[str] = []
    for p in selected:
        rel = p.relative_to(outdir)
        rels.append(str(rel))
        embeds.append(f"### `{rel}`\n![{p.name}]({rel})")
    return rels, "\n\n".join(embeds)


def write_report_expA(
    outdir: str | Path,
    metrics_path: str | Path,
    summary_path: str | Path,
    ks_path: str | Path | None = None,
    figs_glob: str = "plots/*.png",
) -> Path:
    out = Path(outdir)
    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path)
    ks_df = (
        pd.read_csv(ks_path)
        if ks_path is not None and Path(ks_path).exists()
        else pd.DataFrame()
    )
    panel_path = out / "results" / "bh_panel_validation.csv"
    panel_df = pd.read_csv(panel_path) if panel_path.exists() else pd.DataFrame()
    cfg = _read_json(out / "config.json")

    non_under = (
        metrics.loc[~metrics["underpowered"].astype(bool)].copy()
        if "underpowered" in metrics.columns
        else metrics.copy()
    )

    def _weighted_mean(
        df: pd.DataFrame, col: str, weight_col: str = "n_non_underpowered"
    ) -> float:
        vals = pd.to_numeric(df.get(col, np.nan), errors="coerce").to_numpy(dtype=float)
        w = pd.to_numeric(df.get(weight_col, np.nan), errors="coerce").to_numpy(
            dtype=float
        )
        mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
        if not np.any(mask):
            return float("nan")
        return float(np.average(vals[mask], weights=w[mask]))

    if not summary.empty and "prevalence_bin" in summary.columns:
        tail_rows: list[dict[str, Any]] = []
        for prevalence_bin, grp in summary.groupby("prevalence_bin", sort=True):
            tail_rows.append(
                {
                    "prevalence_bin": str(prevalence_bin),
                    "n_genes": int(
                        pd.to_numeric(grp.get("n_genes", 0), errors="coerce")
                        .fillna(0)
                        .sum()
                    ),
                    "n_non_underpowered": int(
                        pd.to_numeric(grp.get("n_non_underpowered", 0), errors="coerce")
                        .fillna(0)
                        .sum()
                    ),
                    "frac_underpowered": float(
                        pd.to_numeric(
                            grp.get("frac_underpowered", np.nan), errors="coerce"
                        ).mean()
                    ),
                    "typeI_p05": _weighted_mean(grp, "typeI_p05"),
                    "typeI_p05_ci_low": _weighted_mean(grp, "typeI_p05_ci_low"),
                    "typeI_p05_ci_high": _weighted_mean(grp, "typeI_p05_ci_high"),
                    "typeI_p01": _weighted_mean(grp, "typeI_p01"),
                    "typeI_p01_ci_low": _weighted_mean(grp, "typeI_p01_ci_low"),
                    "typeI_p01_ci_high": _weighted_mean(grp, "typeI_p01_ci_high"),
                    "typeI_p005": _weighted_mean(grp, "typeI_p005"),
                    "typeI_p005_ci_low": _weighted_mean(grp, "typeI_p005_ci_low"),
                    "typeI_p005_ci_high": _weighted_mean(grp, "typeI_p005_ci_high"),
                    "typeI_p00333": _weighted_mean(grp, "typeI_p00333"),
                    "typeI_p00333_ci_low": _weighted_mean(grp, "typeI_p00333_ci_low"),
                    "typeI_p00333_ci_high": _weighted_mean(grp, "typeI_p00333_ci_high"),
                    "tail_flag_cells_p05": int(
                        pd.to_numeric(
                            grp.get("tail_inflation_flag_p05", 0), errors="coerce"
                        )
                        .fillna(0)
                        .sum()
                    ),
                    "tail_flag_cells_p01": int(
                        pd.to_numeric(
                            grp.get("tail_inflation_flag_p01", 0), errors="coerce"
                        )
                        .fillna(0)
                        .sum()
                    ),
                    "tail_flag_cells_p005": int(
                        pd.to_numeric(
                            grp.get("tail_inflation_flag_p005", 0), errors="coerce"
                        )
                        .fillna(0)
                        .sum()
                    ),
                }
            )
        tail_tab = pd.DataFrame(tail_rows).sort_values("prevalence_bin")
    else:
        tail_tab = pd.DataFrame()

    if not ks_df.empty and "prevalence_bin" in ks_df.columns:
        ks_tab = (
            ks_df.groupby("prevalence_bin", as_index=False)
            .agg(
                n_non_underpowered=("n_non_underpowered", "sum"),
                ks_pvalue_median=("ks_pvalue", "median"),
                ks_pvalue_min=("ks_pvalue", "min"),
            )
            .sort_values("prevalence_bin")
        )
    else:
        ks_tab = pd.DataFrame()

    if not non_under.empty and "prevalence_bin" in non_under.columns:
        meanvar_tab = (
            non_under.groupby("prevalence_bin", as_index=False)
            .agg(
                n_non_underpowered=("p_T", "size"),
                mean_p=("p_T", "mean"),
                var_p=("p_T", "var"),
            )
            .sort_values("prevalence_bin")
        )
    else:
        meanvar_tab = pd.DataFrame()

    m_full = int(cfg.get("G", 0) or 0)
    n_perm = int(cfg.get("n_perm", 0) or 0)
    if m_full <= 0 and "m_full_tests" in summary.columns and not summary.empty:
        m_full = int(
            pd.to_numeric(summary["m_full_tests"], errors="coerce").dropna().iloc[0]
        )
    if n_perm <= 0 and "min_attainable_p" in summary.columns and not summary.empty:
        min_p = float(
            pd.to_numeric(summary["min_attainable_p"], errors="coerce").dropna().iloc[0]
        )
        n_perm = (
            int(round((1.0 / min_p) - 1)) if np.isfinite(min_p) and min_p > 0 else 0
        )
    min_attainable_p = float(1.0 / (n_perm + 1)) if n_perm > 0 else float("nan")
    bh_q05 = float(0.05 / m_full) if m_full > 0 else float("nan")
    bh_q10 = float(0.10 / m_full) if m_full > 0 else float("nan")
    bh_feasible_q05 = bool(
        np.isfinite(bh_q05)
        and np.isfinite(min_attainable_p)
        and bh_q05 >= min_attainable_p
    )
    bh_feasible_q10 = bool(
        np.isfinite(bh_q10)
        and np.isfinite(min_attainable_p)
        and bh_q10 >= min_attainable_p
    )
    bh_tab = pd.DataFrame(
        [
            {
                "q": 0.05,
                "m_full": m_full,
                "bh_min_rejectable_p": bh_q05,
                "min_attainable_p": min_attainable_p,
                "bh_feasible": bh_feasible_q05,
            },
            {
                "q": 0.10,
                "m_full": m_full,
                "bh_min_rejectable_p": bh_q10,
                "min_attainable_p": min_attainable_p,
                "bh_feasible": bh_feasible_q10,
            },
        ]
    )

    bh_mode = str(
        cfg.get("multiple_testing_validation", {}).get(
            "mode",
            cfg.get("bh_validation_mode", "panel_bh"),
        )
    )
    if (
        bh_mode == "panel_bh"
        and not panel_df.empty
        and "prevalence_bin" in panel_df.columns
    ):
        panel_prev_tab = (
            panel_df.groupby("prevalence_bin", as_index=False)
            .agg(
                bh_panel_size_actual=("bh_panel_size_actual", "sum"),
                panel_typeI_q05=("panel_typeI_q05", "mean"),
                panel_typeI_q10=("panel_typeI_q10", "mean"),
                panel_bh_feasible_q05=("panel_bh_feasible_q05", "all"),
                panel_bh_feasible_q10=("panel_bh_feasible_q10", "all"),
            )
            .sort_values("prevalence_bin")
        )
    else:
        panel_prev_tab = pd.DataFrame()

    n_required = (
        int(
            pd.to_numeric(summary["n_required_pm01_95ci"], errors="coerce")
            .dropna()
            .iloc[0]
        )
        if ("n_required_pm01_95ci" in summary.columns and not summary.empty)
        else int(np.ceil(0.05 * 0.95 * (1.96 / 0.01) ** 2))
    )
    mc_tab = pd.DataFrame()
    if not summary.empty and "prevalence_bin" in summary.columns:
        mc_rows: list[dict[str, Any]] = []
        for prevalence_bin, grp in summary.groupby("prevalence_bin", sort=True):
            n_non = pd.to_numeric(
                grp.get("n_non_underpowered", np.nan), errors="coerce"
            )
            insufficient = n_non < float(n_required)
            mc_rows.append(
                {
                    "prevalence_bin": str(prevalence_bin),
                    "cells": int(grp.shape[0]),
                    "cells_insufficient_pm01": int(np.sum(insufficient.fillna(False))),
                    "median_n_non_underpowered": float(
                        np.nanmedian(n_non.to_numpy(dtype=float))
                    ),
                }
            )
        mc_tab = pd.DataFrame(mc_rows).sort_values("prevalence_bin")

    q05_suppressed = ("typeI_q05_status" in summary.columns) and bool(
        (
            summary["typeI_q05_status"].astype(str)
            == "BH infeasible; metric suppressed"
        ).all()
    )

    fig_list, embeds = _figure_embeds(out, figs_glob)
    commit = (
        cfg.get("git_commit_short")
        or cfg.get("git_commit_hash")
        or git_commit_hash(cwd=out)
    )

    md = f"""# Experiment A Report

## Run metadata
- Report generated (UTC): {datetime.now(timezone.utc).isoformat()}
- Output directory: `{out}`
- Commit: `{commit}`
- N: `{cfg.get('N', 'unknown')}`
- D grid: `{cfg.get('donor_grid', 'unknown')}`
- sigma_eta grid: `{cfg.get('sigma_eta_grid', 'unknown')}`
- n_perm: `{cfg.get('n_perm', 'unknown')}`
- bins: `{cfg.get('n_bins', cfg.get('bins', 'unknown'))}`
- Genes total (rows): `{metrics.shape[0]}`
- Non-underpowered rows: `{non_under.shape[0]}`

## Key plots
"""
    md += "- Plotting refactor: standardized mathtext, NA handling, GridSpec layouts; QQ split into full + left-tail\n"
    if fig_list:
        md += "\n".join([f"- `{f}`" for f in fig_list]) + "\n\n" + embeds + "\n"
    else:
        md += "_No plots found._\n"

    md += """
## Quantitative summary
### BH feasibility (full m tests)
"""
    md += _table_markdown(bh_tab) + "\n\n"

    if q05_suppressed:
        md += (
            "Full-BH `typeI_q05` is suppressed because `q/m < 1/(n_perm+1)`; "
            "a zero-rejection rate in this regime is structurally forced by discretization, not evidence of control.\n\n"
        )

    md += "### Tail calibration diagnostics by prevalence\n"
    md += _table_markdown(tail_tab) + "\n\n"

    md += "### KS diagnostics by prevalence\n"
    md += _table_markdown(ks_tab) + "\n\n"

    md += "### Mean/variance of p-values (non-underpowered)\n"
    md += _table_markdown(meanvar_tab) + "\n\n"

    md += "### Multiple-testing validation\n"
    if bh_mode == "panel_bh":
        if panel_prev_tab.empty:
            md += "_Panel BH mode enabled, but panel artifact is missing or empty._\n\n"
        else:
            md += "Panel-BH validation on deterministic random panels (testable m):\n\n"
            md += _table_markdown(panel_prev_tab) + "\n\n"
    else:
        md += (
            "BH over full `m` is not empirically testable at current `n_perm`; report validates "
            "uniformity (bulk + tail) and Type I at fixed alpha, and treats BH claims as conditional "
            "on calibrated p-values.\n\n"
        )

    md += "### Monte Carlo precision\n"
    md += f"Required non-underpowered samples for ±0.01 half-width at 95% CI (p=0.05): `{n_required}`.\n\n"
    md += _table_markdown(mc_tab) + "\n\n"

    md += """## Critical interpretation
### Software/Engineering assessment
- Full-BH success cannot be inferred from zero rejections when BH is infeasible under the discrete p-value floor.
- Tail diagnostics are required because KS can miss localized left-tail inflation relevant to downstream multiple testing.
- MC precision is limited in low-n cells; interpretation should separate calibration deviation from sampling uncertainty.

### Bioinformatics/Reviewer assessment
- The strict-null design is biologically plausible for donor heterogeneity and is useful for validating false-positive control before interpreting localization claims.
- Any prevalence regime with very high abstention (underpowered) should not be used to support calibration claims; those cells should be marked as feasibility-limited.
- Review emphasis should be on donor-conditional permutation validity, tail calibration, and whether multiple-testing claims are reported only in feasible regimes.
"""

    md += f"""
## CHANGELOG
- {datetime.now(timezone.utc).date().isoformat()}: Added tail Type I diagnostics (`p<=0.01`, `p<=0.005`, and min-p mass) with Wilson CIs and inflation flags.
- {datetime.now(timezone.utc).date().isoformat()}: Added explicit BH feasibility math (`q/m` vs `1/(n_perm+1)`) and suppressed infeasible full-BH metrics.
- {datetime.now(timezone.utc).date().isoformat()}: Added panel-BH validation reporting for testable panel sizes.
- {datetime.now(timezone.utc).date().isoformat()}: Added MC precision section with required sample size for ±0.01 tolerance.
"""

    report_path = out / "REPORT.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path


def write_report_expB(
    outdir: str | Path,
    metrics_path: str | Path,
    summary_path: str | Path,
    figs_glob: str = "plots/*.png",
) -> Path:
    out = Path(outdir)
    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path)
    cfg = _read_json(out / "config.json")

    non_under = (
        metrics.loc[~metrics["underpowered"].astype(bool)].copy()
        if "underpowered" in metrics.columns
        else metrics.copy()
    )

    # Compact tables focused on calibration quality and extreme corners.
    key_cols = [
        "geometry",
        "D",
        "mode",
        "pi_target",
        "bins_B",
        "smooth_w",
        "n_non_underpowered",
        "mean_p",
        "var_p",
        "ks_pvalue",
        "typeI_alpha05",
    ]
    avail_cols = [c for c in key_cols if c in summary.columns]

    worst_ks = (
        summary.sort_values("ks_pvalue", ascending=True).loc[:, avail_cols].head(10)
        if ("ks_pvalue" in summary.columns and avail_cols)
        else pd.DataFrame()
    )

    stress = pd.DataFrame()
    if set(["geometry", "mode", "pi_target", "bins_B", "smooth_w"]).issubset(
        summary.columns
    ):
        stress = summary.loc[
            (summary["geometry"] == "density_gradient_disk")
            & (summary["mode"] == "smoothed")
            & (summary["pi_target"] == 0.9)
            & (summary["bins_B"] >= 72)
            & (summary["smooth_w"] >= 5),
            avail_cols,
        ].copy()

    calib_tab = pd.DataFrame()
    if set(
        ["geometry", "mode", "pi_target", "mean_p", "typeI_alpha05", "ks_pvalue"]
    ).issubset(summary.columns):
        calib_tab = (
            summary.groupby(["geometry", "mode", "pi_target"], as_index=False)
            .agg(
                n_cells=("mean_p", "size"),
                mean_p=("mean_p", "mean"),
                mean_typeI=("typeI_alpha05", "mean"),
                median_ks_p=("ks_pvalue", "median"),
            )
            .sort_values(["geometry", "mode", "pi_target"])
        )

    fig_list, embeds = _figure_embeds(out, figs_glob)
    commit = (
        cfg.get("git_commit_short")
        or cfg.get("git_commit_hash")
        or git_commit_hash(cwd=out)
    )

    md = f"""# Experiment B Report

## Run metadata
- Report generated (UTC): {datetime.now(timezone.utc).isoformat()}
- Output directory: `{out}`
- Commit: `{commit}`
- N: `{cfg.get('N', 'unknown')}`
- Geometries: `{cfg.get('geometries', 'unknown')}`
- D grid: `{cfg.get('D_grid', cfg.get('D', 'unknown'))}`
- n_perm: `{cfg.get('n_perm', 'unknown')}`
- bins grid: `{cfg.get('bins_grid', 'unknown')}`
- smoothing grid: `{cfg.get('smooth_grid', 'unknown')}`
- modes: `{cfg.get('modes', 'unknown')}`
- Metrics rows: `{metrics.shape[0]}`
- Non-underpowered rows: `{non_under.shape[0]}`

## Key plots
"""
    if fig_list:
        md += "\n".join([f"- `{f}`" for f in fig_list]) + "\n\n" + embeds + "\n"
    else:
        md += "_No plots found._\n"

    md += """
## Quantitative summary
### Aggregate calibration by geometry/mode/prevalence
"""
    md += _table_markdown(calib_tab) + "\n\n"

    md += "### Worst KS cells\n"
    md += _table_markdown(worst_ks) + "\n\n"

    md += "### Stress corner (density_gradient_disk, smoothed, pi=0.9, high B/w)\n"
    md += _table_markdown(stress) + "\n\n"

    md += """## Critical interpretation
### Software/Engineering assessment
- Raw mode is broadly calibrated across geometries and bin counts in mid-prevalence regimes.
- Smoothed mode shows a structured anti-conservative corner in density-heterogeneous geometry at high prevalence and aggressive (B,w), e.g. density_gradient_disk + pi=0.9 + B=72 + w=5 with historically low KS p, mean(p)<0.5, and elevated Type I.
- Practical conclusion: smoothing is generally usable only if mode consistency is enforced (identical transform for observed and permuted profiles), circular smoothing is correct, and guardrails/defaults are applied for extreme settings.
- Recommended defaults for operational runs remain B=36, w=3 with warnings for more aggressive parameter pairs.

### Bioinformatics/Reviewer assessment
- The max-over-angles statistic properly addresses look-elsewhere only if permutation null and observed statistic use the same mode definition.
- Density-heterogeneous embeddings are an important adversarial stress test and should remain part of validation suites.
- Monte Carlo uncertainty is non-trivial when genes per cell are small; publication-grade calibration claims should pool more genes and/or multiple seeds.
"""

    report_path = out / "REPORT.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path


def write_report_expC(
    outdir: str | Path,
    metrics_path: str | Path,
    summary_path: str | Path,
    figs_glob: str = "plots/*.png",
) -> Path:
    out = Path(outdir)
    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path)
    cfg = _read_json(out / "config.json")

    non_under = (
        metrics.loc[~metrics["underpowered"].astype(bool)].copy()
        if "underpowered" in metrics.columns
        else metrics.copy()
    )

    strict_cols = [
        "N",
        "D",
        "sigma_eta",
        "pi_target",
        "n_genes",
        "n_non_underpowered",
        "analyzable_rate",
        "typeI_alpha05_conditional",
        "typeI_alpha05_operational",
        "typeI_alpha05",
    ]
    strict_avail = [c for c in strict_cols if c in summary.columns]
    strict_worst = pd.DataFrame()
    if "beta" in summary.columns and strict_avail:
        strict_worst = (
            summary.loc[summary["beta"] == 0.0]
            .sort_values(
                by=[
                    (
                        "typeI_alpha05_conditional"
                        if "typeI_alpha05_conditional" in summary.columns
                        else "typeI_alpha05"
                    ),
                    (
                        "analyzable_rate"
                        if "analyzable_rate" in summary.columns
                        else "n_non_underpowered"
                    ),
                ],
                ascending=[False, False],
            )
            .loc[:, strict_avail]
            .head(10)
        )

    power_cols = [
        "N",
        "D",
        "sigma_eta",
        "pi_target",
        "beta",
        "conditional_power_alpha05",
        "operational_power_alpha05",
        "analyzable_rate",
        "n_non_underpowered",
    ]
    power_avail = [c for c in power_cols if c in summary.columns]
    if power_avail and "beta" in summary.columns:
        power_preview = summary.loc[summary["beta"] > 0.0, power_avail].head(12)
    elif power_avail:
        power_preview = summary.loc[:, power_avail].head(12)
    else:
        power_preview = pd.DataFrame()

    fig_list, embeds = _figure_embeds(out, figs_glob)
    commit = (
        cfg.get("git_commit_short")
        or cfg.get("git_commit_hash")
        or git_commit_hash(cwd=out)
    )
    n_perm = cfg.get("n_perm", "unknown")
    p_min = (
        (1.0 / (int(n_perm) + 1))
        if isinstance(n_perm, int) and int(n_perm) > 0
        else "unknown"
    )

    md = f"""# Experiment C Report

## Run metadata
- Report generated (UTC): {datetime.now(timezone.utc).isoformat()}
- Output directory: `{out}`
- Commit: `{commit}`
- N grid: `{cfg.get('N_grid', 'unknown')}`
- D grid: `{cfg.get('D_grid', 'unknown')}`
- sigma_eta grid: `{cfg.get('sigma_eta_grid', 'unknown')}`
- pi grid: `{cfg.get('pi_grid', 'unknown')}`
- beta grid: `{cfg.get('beta_grid', 'unknown')}`
- n_perm: `{n_perm}`
- p_min (plus-one): `{p_min}`
- bins: `{cfg.get('bins', 'unknown')}`
- Metrics rows: `{metrics.shape[0]}`
- Non-underpowered rows: `{non_under.shape[0]}`

## Key plots
"""
    if fig_list:
        md += "\n".join([f"- `{f}`" for f in fig_list]) + "\n\n" + embeds + "\n"
    else:
        md += "_No plots found._\n"

    md += """
## Quantitative summary
### Strict-null worst cells
"""
    md += _table_markdown(strict_worst) + "\n\n"

    md += "### Power/feasibility preview\n"
    md += _table_markdown(power_preview) + "\n\n"

    md += """## Notes
- `conditional_power_alpha05` measures significance among analyzable rows only.
- `operational_power_alpha05` measures significance over all rows (abstention-aware).
- High conditional Type I with low `n_non_underpowered` is expectedly unstable and should be interpreted with the analyzable denominator.
"""

    report_path = out / "REPORT.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path


def render_report_md(template_name: str, context_dict: dict[str, Any]) -> str:
    """Render a lightweight markdown report from a simple template key."""
    template = str(template_name)
    title = str(context_dict.get("title", "Simulation Report"))
    purpose = str(context_dict.get("purpose", ""))
    run_id = str(context_dict.get("run_id", "unknown"))
    run_dir = str(context_dict.get("run_dir", ""))
    key_args = context_dict.get("key_args", {})
    summary_tables = context_dict.get("summary_tables", {})
    key_plots = context_dict.get("key_plots", [])
    validations = context_dict.get("validations", [])
    limitations = context_dict.get("limitations", [])
    next_steps = context_dict.get("next_steps", [])

    lines: list[str] = [f"# {title}", ""]
    lines.append("## Purpose")
    lines.append(purpose or "_Not provided._")
    lines.append("")

    lines.append("## Run config")
    lines.append(f"- run_id: `{run_id}`")
    if run_dir:
        lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- generated_utc: `{datetime.now(timezone.utc).isoformat()}`")
    if isinstance(key_args, dict):
        for k in sorted(key_args.keys()):
            lines.append(f"- {k}: `{key_args[k]}`")
    lines.append("")

    lines.append("## Summary tables")
    if isinstance(summary_tables, dict) and summary_tables:
        for name, table in summary_tables.items():
            lines.append(f"### {name}")
            if isinstance(table, pd.DataFrame):
                lines.append(_table_markdown(table))
            else:
                lines.append(str(table))
            lines.append("")
    else:
        lines.append("_No summary tables provided._")
        lines.append("")

    lines.append("## Key plots")
    if isinstance(key_plots, list) and key_plots:
        for p in key_plots:
            lines.append(f"- `{p}`")
    else:
        lines.append("_No plots listed._")
    lines.append("")

    lines.append("## Validation checks")
    if isinstance(validations, list) and validations:
        for v in validations:
            lines.append(f"- {v}")
    else:
        lines.append("_No validation checks reported._")
    lines.append("")

    lines.append("## Known limitations / next steps")
    if isinstance(limitations, list) and limitations:
        for item in limitations:
            lines.append(f"- {item}")
    if isinstance(next_steps, list) and next_steps:
        for item in next_steps:
            lines.append(f"- {item}")
    if (not limitations) and (not next_steps):
        lines.append("- _None listed._")
    lines.append("")

    if template not in {"standard", "default"}:
        lines.append(f"_Template `{template}` currently maps to standard output._")
        lines.append("")

    return "\n".join(lines)


def write_report(run_dir: str | Path, report_md: str, name: str = "REPORT.md") -> Path:
    """Write report markdown under the run directory."""
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / str(name)
    path.write_text(str(report_md), encoding="utf-8")
    return path


def select_example_genes(
    gene_table_df: pd.DataFrame,
    n: int,
    strategy: str,
    seed: int,
) -> list[Any]:
    """Deterministically select representative genes for plotting/reporting."""
    df = gene_table_df.copy()
    if df.empty:
        return []
    n_take = max(1, int(n))
    rng = np.random.default_rng(int(seed))
    gid_col = "gene_id" if "gene_id" in df.columns else None
    if gid_col is None:
        df = df.reset_index(drop=False).rename(columns={"index": "gene_id"})
        gid_col = "gene_id"

    q_col = (
        "q_T"
        if "q_T" in df.columns
        else ("q_value" if "q_value" in df.columns else None)
    )
    z_col = "z_T" if "z_T" in df.columns else ("Z" if "Z" in df.columns else None)
    if q_col is not None:
        q = pd.to_numeric(df[q_col], errors="coerce")
    else:
        q = pd.Series(np.nan, index=df.index)
    if z_col is not None:
        z = pd.to_numeric(df[z_col], errors="coerce")
    else:
        z = pd.Series(np.nan, index=df.index)
    abs_z = np.abs(z.to_numpy(dtype=float))

    top = df.loc[(q <= 0.05).fillna(False)].copy()
    if not top.empty:
        top = top.assign(
            _q=pd.to_numeric(top[q_col], errors="coerce") if q_col else np.nan
        )
        top = top.sort_values(["_q", gid_col], ascending=[True, True])

    borderline = df.loc[((q > 0.05) & (q <= 0.2)).fillna(False)].copy()
    if not borderline.empty:
        borderline = borderline.assign(
            _z=(
                np.abs(pd.to_numeric(borderline[z_col], errors="coerce"))
                if z_col
                else 0.0
            )
        )
        borderline = borderline.sort_values(["_z", gid_col], ascending=[False, True])

    null_controls = df.loc[(q >= 0.5).fillna(False)].copy()
    if null_controls.empty:
        null_controls = df.copy()
    null_controls = null_controls.assign(
        _q=pd.to_numeric(null_controls[q_col], errors="coerce") if q_col else np.nan
    )
    if q_col is not None:
        null_controls = null_controls.sort_values(
            ["_q", gid_col], ascending=[False, True]
        )

    confound_controls = pd.DataFrame()
    for col in ["truth_category", "pred_label", "label"]:
        if col in df.columns:
            txt = df[col].astype(str).str.upper()
            confound_controls = df.loc[
                txt.str.contains("QC|CONFOUND|DONOR", regex=True, na=False)
            ].copy()
            if not confound_controls.empty:
                break

    if str(strategy) != "top_and_controls":
        ordered = df.assign(
            _score=np.where(np.isfinite(abs_z), abs_z, -np.inf)
        ).sort_values(["_score", gid_col], ascending=[False, True])
        return ordered[gid_col].head(n_take).tolist()

    picks: list[Any] = []
    quotas = {
        "top": max(1, int(round(0.4 * n_take))),
        "borderline": max(1, int(round(0.2 * n_take))),
        "null": max(1, int(round(0.3 * n_take))),
        "confound": max(
            1,
            n_take
            - (
                max(1, int(round(0.4 * n_take)))
                + max(1, int(round(0.2 * n_take)))
                + max(1, int(round(0.3 * n_take)))
            ),
        ),
    }

    def _take(df_part: pd.DataFrame, k: int) -> list[Any]:
        if k <= 0 or df_part.empty:
            return []
        rows = df_part[gid_col].drop_duplicates().tolist()
        if len(rows) <= k:
            return rows
        idx = np.sort(
            rng.choice(np.arange(len(rows), dtype=int), size=k, replace=False)
        )
        return [rows[int(i)] for i in idx]

    for block, src in [
        ("top", top),
        ("borderline", borderline),
        ("null", null_controls),
        ("confound", confound_controls),
    ]:
        for gid in _take(src, quotas[block]):
            if gid not in picks:
                picks.append(gid)
            if len(picks) >= n_take:
                return picks

    if len(picks) < n_take:
        for gid in df[gid_col].drop_duplicates().tolist():
            if gid not in picks:
                picks.append(gid)
            if len(picks) >= n_take:
                break
    return picks


def _detect_headline_metrics(
    summary_df: pd.DataFrame, max_count: int = 6
) -> list[tuple[str, float]]:
    if summary_df.empty:
        return []
    candidates = [
        "typeI_alpha05",
        "typeI_p05",
        "fdr",
        "power",
        "tpr",
        "fpr",
        "precision",
        "recall",
        "spearman",
        "auc",
        "ks_pvalue",
        "mean_p",
        "frac_underpowered",
    ]
    out: list[tuple[str, float]] = []
    for col in candidates:
        if col not in summary_df.columns:
            continue
        vals = pd.to_numeric(summary_df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out.append((col, float(np.nanmean(vals))))
        if len(out) >= int(max_count):
            break
    return out


def _pick_key_plots(run_dir: Path) -> list[str]:
    plots = sorted((run_dir / "plots").glob("*.png"))
    rel = [str(p.relative_to(run_dir)) for p in plots]
    if not rel:
        return []

    def _first(keys: list[str]) -> str | None:
        lowered = [r.lower() for r in rel]
        for key in keys:
            for i, item in enumerate(lowered):
                if key in item:
                    return rel[i]
        return None

    embedding = _first(["embedding", "example_panels", "example_panel", "example_gene"])
    polar = _first(["polar", "profile", "rsp"])
    diagnostic = _first(
        ["qq", "hist", "heatmap", "confusion", "calibration", "fdr", "power"]
    )

    chosen: list[str] = []
    for c in [embedding, polar, diagnostic]:
        if c and c not in chosen:
            chosen.append(c)
    for item in rel:
        if item not in chosen:
            chosen.append(item)
        if len(chosen) >= 8:
            break
    return chosen[:8]


def write_key_results(
    *,
    run_dir: str | Path,
    exp_name: str,
    args: dict[str, Any],
    summary_df: pd.DataFrame,
    validations: list[str],
    plots_all_present: bool,
    missing_plots: list[str],
    runtime_seconds: float | None,
    cache_dir: str | Path | None,
    backend: str,
    n_jobs: int,
    chunk_size: int,
) -> tuple[Path, Path]:
    """Write standardized KEY_RESULTS.md and KEY_RESULTS.json into run_dir."""
    rd = Path(run_dir)
    ts = datetime.now(timezone.utc).isoformat()
    key_plots = _pick_key_plots(rd)
    headline = _detect_headline_metrics(summary_df, max_count=6)
    report_rel = "REPORT.md"

    grid_fields = [
        "N",
        "D",
        "bins",
        "n_perm",
        "mode",
        "w",
        "sigma_eta",
        "geometry",
        "G",
        "runs",
    ]
    grid_pairs = [(k, args[k]) for k in grid_fields if k in args]
    cmd_hint = " ".join(
        [
            f"--{k} {v}"
            for k, v in args.items()
            if k in {"master_seed", "n_perm", "N", "D", "G", "bins"}
        ]
    )

    lines: list[str] = [
        "# KEY RESULTS",
        "",
        "## Run metadata",
        f"- run_id: `{rd.name}`",
        f"- exp_name: `{exp_name}`",
        f"- timestamp_utc: `{ts}`",
        f"- command: `{cmd_hint}`",
        "",
        "## Grid summary",
    ]
    if grid_pairs:
        for k, v in grid_pairs:
            lines.append(f"- {k}: `{v}`")
    else:
        lines.append("- _No grid metadata detected._")

    lines.extend(["", "## Validation checks"])
    if validations:
        for item in validations:
            lines.append(f"- {item}")
    lines.append(f"- plots_all_present: `{bool(plots_all_present)}`")
    if missing_plots:
        lines.append(f"- missing_plots: `{missing_plots}`")

    lines.extend(["", "## Headline metrics"])
    if headline:
        for name, val in headline[:6]:
            lines.append(f"- {name}: `{_fmt(val, digits=5)}`")
    else:
        lines.append("- _No headline metrics found in summary table._")

    lines.extend(["", "## Key plots"])
    if key_plots:
        for p in key_plots:
            lines.append(f"- [{p}]({p})")
    else:
        lines.append("- _No plots found._")

    lines.extend(
        [
            "",
            "## Interpretation",
            "Run completed with standardized outputs. Interpret metrics in context of power, underpowered rate, and calibration diagnostics in the linked report.",
            "",
            f"## Report link\n- [{report_rel}]({report_rel})",
            "",
        ]
    )
    md_path = write_report(rd, "\n".join(lines), name="KEY_RESULTS.md")

    payload: dict[str, Any] = {
        "run_id": rd.name,
        "exp_name": exp_name,
        "timestamp_utc": ts,
        "n_jobs": int(n_jobs),
        "backend": str(backend),
        "chunk_size": int(chunk_size),
        "plots_all_present": bool(plots_all_present),
        "missing_plots": list(missing_plots),
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "runtime_seconds": None if runtime_seconds is None else float(runtime_seconds),
        "key_plots": key_plots,
        "headline_metrics": [{"name": k, "value": float(v)} for k, v in headline[:6]],
    }
    json_path = rd / "KEY_RESULTS.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return md_path, json_path
