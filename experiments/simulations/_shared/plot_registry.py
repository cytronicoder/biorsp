"""Plot completeness registry and assertions for simulation experiments."""

from __future__ import annotations

from pathlib import Path

PLOT_REGISTRY: dict[str, list[str]] = {
    "expA_null_calibration": ["plots/*.png"],
    "expB_maxstat_sensitivity": ["plots/*.png"],
    "expC_power_surfaces": ["plots/*.png"],
    "expD_shape_identifiability": ["plots/*.png"],
    "expE_gradient_vs_step_DE": ["plots/*.png"],
    "expF_confound_resistance": ["plots/*.png"],
    "expG_donor_replication": ["plots/*.png"],
    "expH_fdr_pipeline_scale": ["plots/*.png"],
    "expI_embedding_robustness": ["plots/*.png"],
    "expJ_baselines_ablation": ["plots/*.png"],
}


EMBEDDING_GLOBS = [
    "plots/*embedding*.png",
    "plots/*example*panel*.png",
    "plots/*example*gene*.png",
]


POLAR_RSP_GLOBS = [
    "plots/*polar*rsp*.png",
    "plots/*rsp*polar*.png",
    "plots/*example*profile*.png",
    "plots/*profile*.png",
]


def default_patterns_for_exp(exp_name: str) -> list[str]:
    return PLOT_REGISTRY.get(str(exp_name), ["plots/*.png"])


def _matches(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(str(pattern)))


def assert_plots_present(run_dir: str | Path, patterns: list[str]) -> None:
    root = Path(run_dir)
    missing: list[str] = []
    for pat in patterns:
        if not _matches(root, pat):
            missing.append(str(pat))
    if missing:
        msg = "Missing required plots:\n" + "\n".join(f"- {m}" for m in missing)
        raise FileNotFoundError(msg)


def validate_minimum_required_plot_types(run_dir: str | Path) -> None:
    root = Path(run_dir)
    emb = []
    for pat in EMBEDDING_GLOBS:
        emb.extend(_matches(root, pat))
    pol = []
    for pat in POLAR_RSP_GLOBS:
        pol.extend(_matches(root, pat))

    missing: list[str] = []
    if len(set(emb)) < 1:
        missing.append(
            ">=1 embedding plot (pattern contains `embedding` or `example_panels`)"
        )
    if len(set(pol)) < 1:
        missing.append(
            ">=1 polar/RSP profile plot (pattern contains `polar/rsp/profile`)"
        )
    if missing:
        msg = "Missing required plot categories:\n" + "\n".join(
            f"- {m}" for m in missing
        )
        raise FileNotFoundError(msg)
