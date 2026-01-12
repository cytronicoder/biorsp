"""
Documentation utilities for simulation benchmarks.

Generates markdown reports readable by biologists.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd


def write_report(
    output_dir: Path,
    benchmark_name: str,
    summary_df: pd.DataFrame,
    params: Dict,
    interpretation: str,
    filename: str = "report.md",
) -> None:
    """
    Write markdown report with biologist-friendly interpretation.

    Parameters
    ----------
    output_dir : Path
        Output directory
    benchmark_name : str
        Name of benchmark
    summary_df : pd.DataFrame
        Summary statistics
    params : Dict
        Parameters used
    interpretation : str
        Prose interpretation of results
    filename : str, optional
        Filename
    """
    lines = []

    lines.append(f"# {benchmark_name.title()} Benchmark Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n**Directory:** `{output_dir}`\n")

    lines.append("## Parameters\n")
    for key, value in params.items():
        lines.append(f"- **{key}:** {value}")

    lines.append("\n## Summary Statistics\n")
    lines.append(summary_df.to_markdown(index=False))

    lines.append("\n## Interpretation\n")
    lines.append(interpretation)

    lines.append("\n---\n")
    lines.append("*This report was generated automatically by the BioRSP simulation framework.*")

    filepath = output_dir / filename
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote report to {filepath}")


def interpret_calibration(summary_df: pd.DataFrame, alpha: float = 0.05) -> str:
    """
    Generate interpretation for calibration results.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary with FPR, KS stats
    alpha : float, optional
        Significance threshold

    Returns
    -------
    interpretation : str
        Prose summary
    """
    lines = []

    lines.append("### What does this mean?")
    lines.append(
        f"\nA well-calibrated method should reject {alpha * 100:.1f}% of null hypotheses at α={alpha}. "
        "We test this across different spatial shapes, distortions, and null models.\n"
    )

    # Schema v2.0 uses fpr_05, older versions used fpr_0p05 or fpr_mean
    fpr_col = None
    for candidate in ["fpr_05", "fpr_0p05", "fpr_mean"]:
        if candidate in summary_df.columns:
            fpr_col = candidate
            break

    if fpr_col is None:
        lines.append("⚠️ **No FPR column found:** Cannot compute calibration metrics.")
        return "\n".join(lines)

    fpr_values = summary_df[fpr_col].dropna()

    if len(fpr_values) == 0:
        lines.append("⚠️ **No FPR data available:** All tests abstained.")
        return "\n".join(lines)

    fpr_mean = fpr_values.mean()
    fpr_max = fpr_values.max()

    if fpr_mean < alpha * 1.5:
        lines.append(
            f"✅ **Good calibration:** Average FPR = {fpr_mean:.3f} (close to nominal {alpha:.3f})"
        )
    else:
        lines.append(
            f"⚠️ **Elevated FPR:** Average FPR = {fpr_mean:.3f} (above nominal {alpha:.3f})"
        )

    if fpr_max > alpha * 2:
        lines.append(
            f"\n⚠️ Some conditions show FPR up to {fpr_max:.3f}, indicating potential miscalibration."
        )

    # Check for α=0.01 results (schema v2.0 uses fpr_01)
    fpr_01_col = None
    for candidate in ["fpr_01", "fpr_0p01"]:
        if candidate in summary_df.columns:
            fpr_01_col = candidate
            break
    if fpr_01_col is not None:
        fpr_01_values = summary_df[fpr_01_col].dropna()
        if len(fpr_01_values) > 0:
            lines.append(f"\n📊 **At α=0.01:** Mean FPR = {fpr_01_values.mean():.3f}")

    ks_col = None
    for col_name in ["ks_pval", "ks_pval_mean"]:
        if col_name in summary_df.columns:
            ks_col = col_name
            break

    if ks_col is not None:
        ks_values = summary_df[ks_col].dropna()
        if len(ks_values) > 0:
            ks_pval_mean = ks_values.mean()
            if ks_pval_mean > 0.1:
                lines.append(
                    f"\n✅ **KS test:** P-values appear uniformly distributed (mean KS p = {ks_pval_mean:.3f})"
                )
            else:
                lines.append(
                    f"\n⚠️ **KS test:** Deviation from uniform (mean KS p = {ks_pval_mean:.3f})"
                )

    lines.append("\n### Recommended actions\n")
    if fpr_mean < alpha * 1.5:
        lines.append("- Method is well-calibrated for null hypothesis testing.")
    else:
        lines.append(
            "- Consider adjusting significance threshold or investigating specific conditions with high FPR."
        )

    return "\n".join(lines)


def interpret_power(summary_df: pd.DataFrame, target_power: float = 0.8) -> str:
    """
    Generate interpretation for power results.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary with power estimates
    target_power : float, optional
        Target power

    Returns
    -------
    interpretation : str
        Prose summary
    """
    lines = []

    lines.append("### What does this mean?")
    lines.append(
        f"\nPower is the probability of detecting a true spatial pattern. "
        f"We aim for ≥{target_power * 100:.0f}% power. "
        "Results show how power varies with effect size, sample size, and spatial pattern.\n"
    )

    power_mean = summary_df["power_mean"].mean()
    power_min = summary_df["power_mean"].min()

    if power_mean >= target_power:
        lines.append(
            f"✅ **Good power:** Average power = {power_mean:.2f} (above target {target_power:.2f})"
        )
    else:
        lines.append(
            f"⚠️ **Low power:** Average power = {power_mean:.2f} (below target {target_power:.2f})"
        )

    if power_min < 0.5:
        lines.append(
            f"\n⚠️ Some conditions have power as low as {power_min:.2f}, indicating difficulty detecting weak patterns."
        )

    lines.append("\n### Recommended actions\n")
    if power_mean >= target_power:
        lines.append("- Method has adequate power for typical effect sizes.")
    else:
        lines.append("- Consider increasing sample size or effect size for reliable detection.")
        lines.append("- Focus analysis on genes with strong spatial patterns.")

    return "\n".join(lines)


def interpret_archetypes(summary_df: pd.DataFrame) -> str:
    """
    Generate interpretation for archetype classification.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary with confusion matrix metrics

    Returns
    -------
    interpretation : str
        Prose summary
    """
    lines = []

    lines.append("### What does this mean?")
    lines.append(
        "\nArchetypes represent distinct gene expression patterns: "
        "**Housekeeping** (ubiquitous), **Niche** (spatially restricted), "
        "**Regional** (broad domains), **Scattered** (sparse). "
        "We test whether BioRSP's Coverage (C) and Spatial Score (S) can distinguish these patterns.\n"
    )

    macro_f1 = summary_df["macro_f1"].mean() if "macro_f1" in summary_df.columns else None
    if macro_f1 is not None:
        if macro_f1 > 0.7:
            lines.append(f"✅ **Good classification:** Macro F1 = {macro_f1:.2f}")
        elif macro_f1 > 0.5:
            lines.append(f"⚠️ **Moderate classification:** Macro F1 = {macro_f1:.2f}")
        else:
            lines.append(f"❌ **Poor classification:** Macro F1 = {macro_f1:.2f}")

    lines.append("\n### Recommended actions\n")
    if macro_f1 and macro_f1 > 0.7:
        lines.append("- BioRSP metrics effectively distinguish archetype categories.")
        lines.append("- Use Coverage and Spatial Score for gene stratification.")
    else:
        lines.append("- Archetype boundaries may overlap; consider multi-dimensional analysis.")
        lines.append("- Review confusion matrix to identify specific misclassifications.")

    return "\n".join(lines)


def interpret_genegene(summary_df: pd.DataFrame) -> str:
    """
    Generate interpretation for gene-gene retrieval.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary with AUPRC metrics

    Returns
    -------
    interpretation : str
        Prose summary
    """
    lines = []

    lines.append("### What does this mean?")
    lines.append(
        "\nGene-gene co-patterning identifies pairs of genes with similar spatial distributions. "
        "AUPRC (Area Under Precision-Recall Curve) measures retrieval quality: 1.0 = perfect, 0.5 = random.\n"
    )

    auprc_mean = summary_df["auprc_mean"].mean() if "auprc_mean" in summary_df.columns else None
    if auprc_mean is not None:
        if auprc_mean > 0.8:
            lines.append(f"✅ **Excellent retrieval:** AUPRC = {auprc_mean:.2f}")
        elif auprc_mean > 0.6:
            lines.append(f"✅ **Good retrieval:** AUPRC = {auprc_mean:.2f}")
        else:
            lines.append(f"⚠️ **Moderate retrieval:** AUPRC = {auprc_mean:.2f}")

    lines.append("\n### Recommended actions\n")
    if auprc_mean and auprc_mean > 0.7:
        lines.append("- BioRSP effectively identifies co-patterned gene pairs.")
        lines.append("- Use spatial correlation for gene module discovery.")
    else:
        lines.append(
            "- Consider combining spatial correlation with other features (expression level, cell type)."
        )

    return "\n".join(lines)


def interpret_robustness(summary_df: pd.DataFrame) -> str:
    """
    Generate interpretation for robustness results.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary with stability metrics

    Returns
    -------
    interpretation : str
        Prose summary
    """
    lines = []

    lines.append("### What does this mean?")
    lines.append(
        "\nRobustness measures stability of BioRSP scores under coordinate perturbations "
        "(rotation, scaling, jitter). Low delta = stable, high delta = sensitive.\n"
    )

    delta_mean = (
        summary_df["median_abs_delta_mean"].mean()
        if "median_abs_delta_mean" in summary_df.columns
        else None
    )
    if delta_mean is not None:
        if delta_mean < 0.1:
            lines.append(f"✅ **Highly robust:** Median |Δ| = {delta_mean:.3f}")
        elif delta_mean < 0.2:
            lines.append(f"✅ **Robust:** Median |Δ| = {delta_mean:.3f}")
        else:
            lines.append(f"⚠️ **Moderate sensitivity:** Median |Δ| = {delta_mean:.3f}")

    lines.append("\n### Recommended actions\n")
    if delta_mean and delta_mean < 0.2:
        lines.append("- BioRSP scores are stable under typical preprocessing variations.")
    else:
        lines.append("- Be cautious with heavily distorted or subsampled datasets.")
        lines.append("- Consider preprocessing standardization for cross-study comparisons.")

    return "\n".join(lines)
