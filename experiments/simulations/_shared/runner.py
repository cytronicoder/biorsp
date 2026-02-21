"""Shared execution helpers for composable simulation-cell runners."""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .contracts import (
    CellResult,
    CellSpec,
    ExperimentSummary,
    validate_metrics_long,
    validate_summary,
)
from .io import (
    atomic_write_csv,
    copy_artifact_snapshot,
    ensure_dir,
    init_run_dir,
    publish_latest,
    write_json,
)
from .io import (
    cache_get as io_cache_get,
)
from .io import (
    cache_set as io_cache_set,
)
from .plot_registry import (
    assert_plots_present,
    default_patterns_for_exp,
    validate_minimum_required_plot_types,
)
from .reporting import render_report_md, write_key_results, write_report


@dataclass
class RunContext:
    exp_name: str
    exp_dir: Path
    run_dir: Path
    config_dir: Path
    logs_dir: Path
    results_dir: Path
    plots_dir: Path
    diagnostics_dir: Path
    cache_dir: Path
    args: Any
    cache: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def cache_get(self, key: str) -> Any | None:
        if key in self.cache:
            return self.cache[key]
        obj = io_cache_get(self.cache_dir, key)
        if obj is not None:
            self.cache[key] = obj
        return obj

    def cache_set(self, key: str, obj: Any, *, persist: bool = True) -> Any:
        self.cache[key] = obj
        if persist:
            io_cache_set(self.cache_dir, key, obj)
        return obj


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")


def prepare_legacy_run(args: Any, exp_name: str, script_path: str | Path) -> RunContext:
    """Initialize standardized run folders for legacy experiment scripts."""
    script = Path(script_path).resolve()
    outdir = Path(getattr(args, "outdir", script.parent))
    if outdir.name.startswith("run_") and outdir.parent.name == "runs":
        exp_dir = outdir.parent.parent
    else:
        exp_dir = outdir

    run_dir = init_run_dir(
        outdir=outdir,
        exp_name=str(exp_name),
        run_tag=getattr(args, "run_tag", None),
        test_mode=bool(getattr(args, "test_mode", False)),
        overwrite=bool(getattr(args, "overwrite", False)),
        master_seed=int(getattr(args, "master_seed", 0)),
    )

    ctx = RunContext(
        exp_name=str(exp_name),
        exp_dir=exp_dir,
        run_dir=run_dir,
        config_dir=run_dir / "config",
        logs_dir=run_dir / "logs",
        results_dir=run_dir / "results",
        plots_dir=run_dir / "plots",
        diagnostics_dir=run_dir / "diagnostics",
        cache_dir=ensure_dir(
            Path(getattr(args, "cache_dir", "") or (run_dir / "cache"))
        ),
        args=args,
    )

    # Direct legacy scripts to write into standardized run_dir.
    setattr(args, "outdir", str(run_dir))
    setattr(args, "cache_dir", str(ctx.cache_dir))
    os.environ["BIORSP_SIM_RUN_DIR"] = str(run_dir)

    write_json(ctx.config_dir / "args.json", vars(args))
    _append_log(
        ctx.logs_dir / "run.log", f"start exp={ctx.exp_name} run_dir={ctx.run_dir}"
    )

    if bool(getattr(args, "artifact_snapshot", False)):
        shared_dir = script.parents[1] / "_shared"
        copy_artifact_snapshot([script, shared_dir], ctx.run_dir / "artifact_snapshot")

    return ctx


def finalize_legacy_run(
    ctx: RunContext,
    *,
    validations: list[str] | None = None,
    summary_tables: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Finalize standardized run and publish compatibility outputs."""
    os.environ.pop("BIORSP_SIM_RUN_DIR", None)

    missing_plots: list[str] = []
    plots_all_present = True
    if str(getattr(ctx.args, "plots", "all")) == "all" and (
        not bool(getattr(ctx.args, "no_plots", False))
    ):
        expected = default_patterns_for_exp(ctx.exp_name)
        try:
            assert_plots_present(ctx.run_dir, expected)
            if not bool(getattr(ctx.args, "test_mode", False)):
                validate_minimum_required_plot_types(ctx.run_dir)
        except FileNotFoundError as exc:
            plots_all_present = False
            missing_plots = [
                line.strip("- ").strip()
                for line in str(exc).splitlines()
                if line.startswith("-")
            ]
            _append_log(ctx.logs_dir / "run.log", f"plot_check_failed {exc}")
            raise

    # Canonical config snapshot inside config/.
    root_cfg = ctx.run_dir / "config.json"
    if root_cfg.exists():
        write_json(
            ctx.config_dir / "config.json",
            json.loads(root_cfg.read_text(encoding="utf-8")),
        )

    report_path = ctx.run_dir / "REPORT.md"
    if (not bool(getattr(ctx.args, "no_reports", False))) and (
        not report_path.exists()
    ):
        key_plots = []
        if ctx.plots_dir.exists():
            key_plots = [
                str(p.relative_to(ctx.run_dir))
                for p in sorted(ctx.plots_dir.glob("*.png"))[:12]
            ]
        md = render_report_md(
            "standard",
            {
                "title": f"{ctx.exp_name} Report",
                "purpose": "Standardized simulation run report.",
                "run_id": ctx.run_dir.name,
                "run_dir": str(ctx.run_dir),
                "key_args": {
                    k: v
                    for k, v in vars(ctx.args).items()
                    if k
                    in {
                        "master_seed",
                        "test_mode",
                        "n_perm",
                        "N",
                        "G",
                        "bins",
                        "mode",
                        "w",
                    }
                },
                "summary_tables": summary_tables or {},
                "key_plots": key_plots,
                "validations": validations or [],
            },
        )
        write_report(ctx.run_dir, md)

    # Lightweight schema checks (diagnostic only).
    schema_lines: list[str] = []
    metrics_required = [
        "run_id",
        "seed",
        "pi_target",
        "beta",
    ]
    summary_required = [
        "n_genes",
    ]
    candidate_metrics = [
        ctx.results_dir / "metrics_long.csv",
        ctx.results_dir / "gene_table.csv",
        ctx.results_dir / "gene_variant_table.csv",
        ctx.results_dir / "gene_scores.csv",
        ctx.results_dir / "donor_metrics_long.csv",
    ]
    for p in candidate_metrics:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, nrows=10)
            need = [c for c in metrics_required if c in df.columns]
            if need:
                validate_metrics_long(df, need)
                schema_lines.append(f"{p.name}: metrics schema check OK ({need})")
            else:
                schema_lines.append(
                    f"{p.name}: metrics schema check skipped (no overlapping required cols)"
                )
        except Exception as exc:  # noqa: BLE001
            schema_lines.append(f"{p.name}: metrics schema check FAILED ({exc})")

    candidate_summary = [
        ctx.results_dir / "summary.csv",
        ctx.results_dir / "metrics_summary.csv",
        ctx.results_dir / "summary_runlevel.csv",
    ]
    for p in candidate_summary:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, nrows=10)
            need = [c for c in summary_required if c in df.columns]
            if need:
                validate_summary(df, need)
                schema_lines.append(f"{p.name}: summary schema check OK ({need})")
            else:
                schema_lines.append(
                    f"{p.name}: summary schema check skipped (no overlapping required cols)"
                )
        except Exception as exc:  # noqa: BLE001
            schema_lines.append(f"{p.name}: summary schema check FAILED ({exc})")

    if schema_lines:
        (ctx.diagnostics_dir / "schema_validation.txt").write_text(
            "\n".join(schema_lines) + "\n", encoding="utf-8"
        )

    summary_path_candidates = [
        ctx.results_dir / "summary.csv",
        ctx.results_dir / "metrics_summary.csv",
        ctx.results_dir / "summary_runlevel.csv",
    ]
    summary_df = pd.DataFrame()
    for spath in summary_path_candidates:
        if spath.exists():
            try:
                summary_df = pd.read_csv(spath)
                break
            except Exception:
                continue

    write_key_results(
        run_dir=ctx.run_dir,
        exp_name=ctx.exp_name,
        args=vars(ctx.args),
        summary_df=summary_df,
        validations=validations or [],
        plots_all_present=bool(plots_all_present),
        missing_plots=missing_plots,
        runtime_seconds=(time.time() - ctx.started_at),
        cache_dir=ctx.cache_dir,
        backend=str(getattr(ctx.args, "backend", "loky")),
        n_jobs=int(getattr(ctx.args, "n_jobs", 1)),
        chunk_size=int(getattr(ctx.args, "chunk_size", 25)),
    )

    publish_latest(ctx.run_dir, ctx.exp_dir)

    elapsed = time.time() - ctx.started_at
    _append_log(ctx.logs_dir / "run.log", f"end status=ok elapsed_sec={elapsed:.3f}")


def run_cells(
    cells: list[CellSpec],
    run_cell_fn: Callable[[CellSpec, RunContext], CellResult],
    run_ctx: RunContext,
) -> list[CellResult]:
    """Execute cells serially with fault-tolerant status capture."""
    out: list[CellResult] = []
    status_rows: list[dict[str, Any]] = []

    for i, cell in enumerate(cells, start=1):
        t0 = time.time()
        _append_log(
            run_ctx.logs_dir / "run.log",
            f"cell_start idx={i}/{len(cells)} cell_id={cell.cell_id}",
        )
        try:
            res = run_cell_fn(cell, run_ctx)
            if not isinstance(res, CellResult):
                raise TypeError("run_cell_fn must return CellResult.")
            out.append(res)
            status_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "success": bool(res.success),
                    "runtime_sec": float(res.runtime_sec),
                    "error": str(res.error or ""),
                }
            )
        except Exception as exc:  # noqa: BLE001
            runtime = time.time() - t0
            tb = traceback.format_exc()
            diag = run_ctx.diagnostics_dir / f"cell_{cell.cell_id}_error.txt"
            diag.write_text(tb, encoding="utf-8")
            fail = CellResult(
                cell_id=cell.cell_id,
                success=False,
                runtime_sec=float(runtime),
                metrics={},
                artifacts={"traceback": str(diag.relative_to(run_ctx.run_dir))},
                error=str(exc),
            )
            out.append(fail)
            status_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "success": False,
                    "runtime_sec": float(runtime),
                    "error": str(exc),
                }
            )
        _append_log(
            run_ctx.logs_dir / "run.log",
            f"cell_end idx={i}/{len(cells)} cell_id={cell.cell_id}",
        )

    status = pd.DataFrame(status_rows)
    atomic_write_csv(run_ctx.results_dir / "cell_status.csv", status)
    return out


def aggregate_results(
    cell_results: list[CellResult],
    aggregator_fn: Callable[[list[CellResult], RunContext], ExperimentSummary],
    run_ctx: RunContext,
) -> ExperimentSummary:
    """Aggregate successful cell outputs into experiment summary."""
    return aggregator_fn(cell_results, run_ctx)


def finalize_run(
    summary: ExperimentSummary,
    run_ctx: RunContext,
    reporting_fn: Callable[[ExperimentSummary, RunContext], str] | None = None,
    plot_fn: Callable[[ExperimentSummary, RunContext], None] | None = None,
) -> None:
    """Finalize composable cell-runner pipeline."""
    if plot_fn is not None and not bool(getattr(run_ctx.args, "no_plots", False)):
        plot_fn(summary, run_ctx)

    if reporting_fn is not None and not bool(
        getattr(run_ctx.args, "no_reports", False)
    ):
        md = reporting_fn(summary, run_ctx)
        write_report(run_ctx.run_dir, md)

    publish_latest(run_ctx.run_dir, run_ctx.exp_dir)
