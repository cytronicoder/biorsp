#!/usr/bin/env python3
"""Standardized launcher for simulation experiments A-J."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.simulations._shared.io import (
    make_run_dir,
    write_config_snapshot,
    write_env_snapshot,
    write_git_snapshot,
)

SIM_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = SIM_ROOT / "_results"
DEFAULT_PREVALENCE_GRID = [0.002, 0.005, 0.01, 0.05, 0.2, 0.6, 0.9]

EXPERIMENTS: dict[str, dict[str, str]] = {
    "A": {
        "name": "expA_null_calibration",
        "script": "experiments/simulations/expA_null_calibration/run_expA_null_calibration.py",
        "default_config": "experiments/simulations/configs/expA_null_calibration.full.json",
    },
    "B": {
        "name": "expB_maxstat_sensitivity",
        "script": "experiments/simulations/expB_maxstat_sensitivity/run_expB_maxstat_sensitivity.py",
        "default_config": "experiments/simulations/configs/expB_maxstat_sensitivity.full.json",
    },
    "C": {
        "name": "expC_power_surfaces",
        "script": "experiments/simulations/expC_power_surfaces/run_expC_power_surfaces.py",
        "default_config": "experiments/simulations/configs/expC_power_surfaces.full.json",
    },
    "D": {
        "name": "expD_shape_identifiability",
        "script": "experiments/simulations/expD_shape_identifiability/run_expD_shape_identifiability.py",
        "default_config": "experiments/simulations/configs/expD_shape_identifiability.full.json",
    },
    "E": {
        "name": "expE_gradient_vs_step_DE",
        "script": "experiments/simulations/expE_gradient_vs_step_DE/run_expE_gradient_vs_step_DE.py",
        "default_config": "experiments/simulations/configs/expE_gradient_vs_step_DE.full.json",
    },
    "F": {
        "name": "expF_confound_resistance",
        "script": "experiments/simulations/expF_confound_resistance/run_expF_confound_resistance.py",
        "default_config": "experiments/simulations/configs/expF_confound_resistance.full.json",
    },
    "G": {
        "name": "expG_donor_replication",
        "script": "experiments/simulations/expG_donor_replication/run_expG_donor_replication.py",
        "default_config": "experiments/simulations/configs/expG_donor_replication.full.json",
    },
    "H": {
        "name": "expH_fdr_pipeline_scale",
        "script": "experiments/simulations/expH_fdr_pipeline_scale/run_expH_fdr_pipeline_scale.py",
        "default_config": "experiments/simulations/configs/expH_fdr_pipeline_scale.full.json",
    },
    "I": {
        "name": "expI_embedding_robustness",
        "script": "experiments/simulations/expI_embedding_robustness/run_expI_embedding_robustness.py",
        "default_config": "experiments/simulations/configs/expI_embedding_robustness.full.json",
    },
    "J": {
        "name": "expJ_baselines_ablation",
        "script": "experiments/simulations/expJ_baselines_ablation/run_expJ_baselines_ablation.py",
        "default_config": "experiments/simulations/configs/expJ_baselines_ablation.full.json",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_config(path: Path) -> dict[str, Any]:
    cfg = _read_json(path)
    parent = cfg.get("extends")
    if not parent:
        return cfg
    parent_path = (path.parent / str(parent)).resolve()
    parent_cfg = _resolve_config(parent_path)
    merged = _deep_merge(parent_cfg, {k: v for k, v in cfg.items() if k != "extends"})
    merged["resolved_from"] = [
        *parent_cfg.get("resolved_from", [str(parent_path)]),
        str(path),
    ]
    return merged


def _to_list(val: Any) -> list[float]:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        out = []
        for x in val:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(val, str):
        out = []
        for token in val.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                out.append(float(token))
            except (TypeError, ValueError):
                continue
        return out
    return []


def _first(args: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key in args and args[key] is not None:
            return args[key]
    return None


def _assertions_for_prereg(
    config: dict[str, Any], args: dict[str, Any]
) -> list[dict[str, Any]]:
    prereg = dict(config.get("preregistered", {}))
    checks: list[dict[str, Any]] = []

    n_perm = _first(args, ["n_perm", "n_perm_pool", "n_perm_donor", "n_perm_default"])
    checks.append(
        {
            "name": "n_perm_declared",
            "passed": bool(n_perm is not None and int(n_perm) > 0),
            "detail": f"n_perm_source={n_perm}",
        }
    )

    plus_one = bool(prereg.get("plus_one_correction", False))
    checks.append(
        {
            "name": "plus_one_enabled",
            "passed": plus_one,
            "detail": "preregistered.plus_one_correction must be true",
        }
    )

    req_prev = _to_list(prereg.get("prevalence_grid", DEFAULT_PREVALENCE_GRID))
    cfg_prev = _to_list(
        _first(args, ["prevalence_bins", "pi_grid", "pi_bins", "prevalence_grid"])
    )
    has_prev = all(any(abs(a - b) < 1e-12 for a in cfg_prev) for b in req_prev)
    checks.append(
        {
            "name": "prevalence_grid_preregistered",
            "passed": bool(cfg_prev) and has_prev,
            "detail": f"required={req_prev}, configured={cfg_prev}",
        }
    )

    donor_strat = prereg.get("donor_stratification", {})
    donor_enabled = bool(donor_strat.get("enabled", True))
    checks.append(
        {
            "name": "donor_stratification_enabled",
            "passed": donor_enabled,
            "detail": "set preregistered.donor_stratification.enabled=true unless explicitly disabled",
        }
    )

    bins_b = prereg.get("bins_B")
    smooth_w = prereg.get("smoothing_w")
    checks.append(
        {
            "name": "bins_and_smoothing_fixed",
            "passed": bins_b is not None and smooth_w is not None,
            "detail": f"bins_B={bins_b}, smoothing_w={smooth_w}",
        }
    )

    seed = _first(args, ["master_seed", "seed", "global_seed"])
    checks.append(
        {
            "name": "seed_policy_set",
            "passed": seed is not None,
            "detail": f"seed={seed}",
        }
    )

    gating_keys = ["p_min", "min_fg_total", "min_fg_per_donor", "d_eff_min"]
    missing = [
        key
        for key in gating_keys
        if _first(args, [key]) is None and prereg.get(key) is None
    ]
    checks.append(
        {
            "name": "underpowered_gating_active",
            "passed": not missing,
            "detail": (
                "missing=" + ",".join(missing)
                if missing
                else "all gating parameters set"
            ),
        }
    )

    perm_strategy = str(prereg.get("permutation_strategy", "donor_aware"))
    checks.append(
        {
            "name": "donor_aware_default",
            "passed": perm_strategy == "donor_aware",
            "detail": f"permutation_strategy={perm_strategy}",
        }
    )
    return checks


def _write_prereg_checks(run_dir: Path, checks: list[dict[str, Any]]) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": bool(all(bool(c.get("passed", False)) for c in checks)),
        "checks": checks,
    }
    (run_dir / "PREREG_CHECKS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _value_to_cli(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return str(value)


def _arg_uses_nargs(script: Path, key: str) -> bool:
    try:
        text = script.read_text(encoding="utf-8")
    except Exception:
        return False
    pattern = re.compile(
        rf"add_argument\(\s*[\"']--{re.escape(key)}[\"'][^\)]*nargs\s*=",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return bool(pattern.search(text))


def _build_command(script: Path, args: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, str(script)]
    for key, value in args.items():
        if value is None:
            continue
        opt = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(opt)
            continue
        if isinstance(value, (list, tuple)):
            if _arg_uses_nargs(script, key):
                cmd.append(opt)
                cmd.extend([str(x) for x in value])
            else:
                cmd.extend([opt, _value_to_cli(value)])
            continue
        if _arg_uses_nargs(script, key) and isinstance(value, str) and "," in value:
            tokens = [token.strip() for token in value.split(",") if token.strip()]
            if tokens:
                cmd.append(opt)
                cmd.extend(tokens)
                continue
        cmd.extend([opt, _value_to_cli(value)])
    return cmd


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _first_effective(payload: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key not in payload:
            continue
        val = payload.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            if not val:
                continue
            return val[0]
        if isinstance(val, str) and "," in val:
            toks = [tok.strip() for tok in val.split(",") if tok.strip()]
            if not toks:
                continue
            return toks[0]
        return val
    return None


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_dir_tokens(run_dir: Path) -> dict[str, int | None]:
    name = run_dir.name
    token_seed = re.search(r"(?:^|__)seed(\d+)(?:__|$)", name)
    token_n = re.search(r"(?:^|__)N(\d+)(?:__|$)", name)
    token_d = re.search(r"(?:^|__)D(\d+)(?:__|$)", name)
    token_nperm = re.search(r"(?:^|__)nperm(\d+)(?:__|$)", name)
    return {
        "seed": int(token_seed.group(1)) if token_seed else None,
        "N": int(token_n.group(1)) if token_n else None,
        "D": int(token_d.group(1)) if token_d else None,
        "n_perm": int(token_nperm.group(1)) if token_nperm else None,
    }


def _collect_reported_params(runner_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": _as_int_or_none(
            _first_effective(runner_args, ["master_seed", "seed", "global_seed"])
        ),
        "N": _as_int_or_none(
            _first_effective(runner_args, ["N", "N_total", "N_cells", "N_grid"])
        ),
        "D": _as_int_or_none(
            _first_effective(runner_args, ["D", "D_grid", "donors", "donor_count"])
        ),
        "n_perm": _as_int_or_none(
            _first_effective(
                runner_args, ["n_perm", "n_perm_pool", "n_perm_donor", "n_perm_default"]
            )
        ),
        "bins": _as_int_or_none(
            _first_effective(runner_args, ["bins", "bins_B", "bins_grid"])
        ),
        "w": _as_int_or_none(
            _first_effective(runner_args, ["w", "smooth_w", "w_grid", "smooth_grid"])
        ),
        "sigma_eta": _as_float_or_none(
            _first_effective(runner_args, ["sigma_eta", "sigma_eta_grid"])
        ),
        "pi": _as_float_or_none(
            _first_effective(
                runner_args,
                [
                    "pi_target",
                    "pi_grid",
                    "pi_bins",
                    "prevalence_bins",
                    "prevalence_grid",
                ],
            )
        ),
        "G": _as_int_or_none(
            _first_effective(
                runner_args, ["G", "G_total", "genes_per_condition", "genes_per_class"]
            )
        ),
    }


def _collect_effective_params(
    run_dir: Path, fallback: dict[str, Any]
) -> dict[str, Any]:
    effective_cfg = _load_json_if_exists(run_dir / "config.json")
    source = effective_cfg if effective_cfg else fallback
    return {
        "seed": _as_int_or_none(
            _first_effective(source, ["master_seed", "seed", "global_seed"])
        ),
        "N": _as_int_or_none(
            _first_effective(source, ["N", "N_total", "N_cells", "N_grid"])
        ),
        "D": _as_int_or_none(
            _first_effective(source, ["D", "D_grid", "donors", "donor_count"])
        ),
        "n_perm": _as_int_or_none(
            _first_effective(
                source, ["n_perm", "n_perm_pool", "n_perm_donor", "n_perm_default"]
            )
        ),
        "bins": _as_int_or_none(
            _first_effective(source, ["bins", "bins_B", "bins_grid"])
        ),
        "w": _as_int_or_none(
            _first_effective(source, ["w", "smooth_w", "w_grid", "smooth_grid"])
        ),
        "sigma_eta": _as_float_or_none(
            _first_effective(source, ["sigma_eta", "sigma_eta_grid"])
        ),
        "pi": _as_float_or_none(
            _first_effective(
                source,
                [
                    "pi_target",
                    "pi_grid",
                    "pi_bins",
                    "prevalence_bins",
                    "prevalence_grid",
                ],
            )
        ),
        "G": _as_int_or_none(
            _first_effective(
                source, ["G", "G_total", "genes_per_condition", "genes_per_class"]
            )
        ),
    }


def assert_effective_equals_reported(
    run_dir: Path, runner_args: dict[str, Any]
) -> dict[str, Any]:
    reported = _collect_reported_params(runner_args)
    effective = _collect_effective_params(run_dir, reported)
    tokens = _run_dir_tokens(run_dir)

    checks: list[dict[str, Any]] = []
    for field in ["seed", "N", "D", "n_perm"]:
        tok = tokens.get(field)
        eff = effective.get(field)
        passed = True
        detail = "no token"
        if tok is not None:
            passed = bool(eff is not None and int(tok) == int(eff))
            detail = f"token={tok}, effective={eff}"
        checks.append({"field": field, "passed": passed, "detail": detail})

    if not all(bool(c["passed"]) for c in checks):
        failed = [c for c in checks if not bool(c["passed"])]
        raise RuntimeError(
            "Run metadata mismatch between run_dir token(s) and effective config: "
            + "; ".join(f"{c['field']} ({c['detail']})" for c in failed)
        )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "metadata_consistent": True,
        "run_dir_tokens": tokens,
        "reported_params": reported,
        "effective_params": effective,
        "checks": checks,
    }
    (run_dir / "RUN_METADATA.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def _ensure_csv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"status": "missing_from_legacy_runner", "note": f"autocreated: {path.name}"}]
    ).to_csv(path, index=False)


def _ensure_artifacts(run_dir: Path) -> list[str]:
    missing_before: list[str] = []
    required_csv = [
        run_dir / "results" / "metrics_long.csv",
        run_dir / "results" / "summary_by_bin.csv",
        run_dir / "results" / "summary.csv",
        run_dir / "results" / "ks_uniformity.csv",
        run_dir / "results" / "qc_correlation_checks.csv",
    ]
    optional_csv = [run_dir / "results" / "prevalence_deviation.csv"]
    required_files = [
        run_dir / "config_snapshot.json",
        run_dir / "env.json",
        run_dir / "git.json",
        run_dir / "REPORT.md",
    ]

    for req in required_files:
        if not req.exists():
            missing_before.append(str(req.relative_to(run_dir)))
    for req in required_csv:
        if not req.exists():
            missing_before.append(str(req.relative_to(run_dir)))
            _ensure_csv(req)
    for opt in optional_csv:
        if not opt.exists():
            _ensure_csv(opt)

    report = run_dir / "REPORT.md"
    if not report.exists():
        report.write_text(
            "# Simulation Report\n\nAutogenerated placeholder report.\n",
            encoding="utf-8",
        )

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for candidate in (run_dir / "plots").glob("*"):
        if candidate.suffix.lower() not in {".png", ".pdf"}:
            continue
        target = figures_dir / candidate.name
        if not target.exists():
            target.write_bytes(candidate.read_bytes())

    return missing_before


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one preregistered simulation experiment."
    )
    parser.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS.keys()))
    parser.add_argument(
        "--config", type=str, default=None, help="Experiment config JSON."
    )
    parser.add_argument(
        "--out_root", "--out", dest="out_root", type=str, default=str(DEFAULT_OUT_ROOT)
    )
    parser.add_argument(
        "--batch_id", type=str, default=None, help="Existing batch id under out_root."
    )
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--strict_prereg", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exp = EXPERIMENTS[args.exp]

    cfg_path = (
        (REPO_ROOT / args.config).resolve()
        if args.config
        else (REPO_ROOT / exp["default_config"]).resolve()
    )
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    resolved = _resolve_config(cfg_path)
    runner = dict(resolved.get("runner", {}))
    runner_args = dict(runner.get("args", {}))
    runner_script = Path(runner.get("script", exp["script"]))
    if not runner_script.is_absolute():
        runner_script = (REPO_ROOT / runner_script).resolve()
    if not runner_script.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_script}")

    if args.seed is not None:
        runner_args["master_seed"] = int(args.seed)
    if args.n_jobs is not None:
        runner_args["n_jobs"] = int(args.n_jobs)
    if bool(args.dry_run):
        runner_args["dry_run"] = True

    batch_id = args.batch_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_root = Path(args.out_root).resolve() / batch_id
    batch_root.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(
        root=batch_root,
        exp_name=exp["name"],
        tag=args.tag,
        seed=runner_args.get("master_seed"),
        config=runner_args,
    )

    write_config_snapshot(run_dir, cfg_path, resolved_config=resolved)
    write_env_snapshot(run_dir)
    write_git_snapshot(run_dir, cwd=REPO_ROOT)

    checks = _assertions_for_prereg(resolved, runner_args)
    _write_prereg_checks(run_dir, checks)
    if args.strict_prereg and (not all(bool(c["passed"]) for c in checks)):
        print(f"RUN_DIR={run_dir}")
        print("Prereg checks failed in strict mode.")
        return 2

    runner_args["outdir"] = str(run_dir)
    cmd = _build_command(runner_script, runner_args)
    print(f"RUN_DIR={run_dir}")
    print("COMMAND=" + shlex.join(cmd))

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, check=False)
    if proc.returncode != 0:
        return int(proc.returncode)

    metadata = assert_effective_equals_reported(run_dir, runner_args)

    missing = _ensure_artifacts(run_dir)
    report_path = run_dir / "REPORT.md"
    effective = metadata["effective_params"]
    validation_path = run_dir / "validation_debug_report.txt"
    validation_path_results = run_dir / "results" / "validation_debug_report.txt"
    if validation_path_results.exists() and not validation_path.exists():
        validation_path.write_text(
            validation_path_results.read_text(encoding="utf-8"), encoding="utf-8"
        )
    report_append = [
        "",
        "## Run Metadata",
        f"- Experiment: {exp['name']}",
        f"- Run dir: {run_dir}",
        f"- Config: {cfg_path}",
        f"- Seed: {effective.get('seed', runner_args.get('master_seed', 'unknown'))}",
    ]
    report_append.append("- Prereg checks: [PREREG_CHECKS.json](PREREG_CHECKS.json)")
    report_append.append("- Metadata checks: [RUN_METADATA.json](RUN_METADATA.json)")
    if validation_path.exists():
        report_append.append(
            "- Validation warnings/debug: [validation_debug_report.txt](validation_debug_report.txt)"
        )
    report_append.extend(
        [
            "",
            "## Effective Parameters",
            f"- N: `{effective.get('N')}`",
            f"- G: `{effective.get('G')}`",
            f"- n_perm: `{effective.get('n_perm')}`",
            f"- D: `{effective.get('D')}`",
            f"- bins: `{effective.get('bins')}`",
            f"- w: `{effective.get('w')}`",
            f"- sigma_eta: `{effective.get('sigma_eta')}`",
            f"- pi: `{effective.get('pi')}`",
        ]
    )
    if not report_path.read_text(encoding="utf-8").endswith("\n"):
        report_append.insert(0, "")
    with report_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(report_append) + "\n")

    if missing:
        diagnostics = run_dir / "diagnostics" / "artifact_enforcement.json"
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.write_text(
            json.dumps({"autofilled": missing}, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
