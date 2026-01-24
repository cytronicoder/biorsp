"""
One-Page Story Figure for BioRSP Methods Paper.

Generates a single-page figure panel (4 subpanels) that validates BioRSP's core claims:
- Panel A: Archetype scatter (C vs S) with ground truth coloring and quadrant boundaries
- Panel B: Confusion matrix for 2×2 classification
- Panel C: Marker recovery (precision@K for structured genes)
- Panel D: Gene-gene module recovery

Usage:
    python run_story_onepager.py --mode quick --outdir outputs/story --seed 42

    python run_story_onepager.py --mode publication --seed 42

Performance Tips:
    - Use --mode quick for testing (~15s)
    - Use --mode validation for preliminary results (~5min)
    - Use --mode publication for final manuscript figures (~15min)
    - Main bottleneck: permutation tests (controlled by mode)
    - For large-scale benchmarks, see run_calibration.py which supports --n_workers

Outputs:
    - fig_story_A_archetypes.png
    - fig_story_B_confusion.png
    - fig_story_C_marker_recovery.png
    - fig_story_D_genegene.png
    - fig_story_onepager.png (combined)
    - fig_story_onepager_caption.txt
    - runs.csv, summary.csv, manifest.json, report.md
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biorsp import BioRSPConfig  # noqa: E402
from biorsp.plotting.spec import PlotSpec  # noqa: E402

MODE_CONFIGS = {
    "quick": {
        "n_cells": 1500,
        "n_per_archetype": 10,
        "n_modules": 3,
        "genes_per_module": 8,
        "n_null_genes": 10,
        "n_permutations": 0,
        "shape": "disk",
    },
    "validation": {
        "n_cells": 2500,
        "n_per_archetype": 25,
        "n_modules": 4,
        "genes_per_module": 12,
        "n_null_genes": 20,
        "n_permutations": 100,
        "shape": "disk",
    },
    "publication": {
        "n_cells": 3000,
        "n_per_archetype": 40,
        "n_modules": 5,
        "genes_per_module": 15,
        "n_null_genes": 30,
        "n_permutations": 500,
        "shape": "disk",
    },
}

_DEFAULT_SPEC = PlotSpec()


def run_story_benchmark(args):
    """Run the complete story benchmark.

    Args:
        args: Parsed CLI arguments.
    """
    from biorsp.simulations import (
        datasets,
        expression,
        io,
        metrics,
        plotting,
        rng,
        scoring,
        shapes,
    )

    mode_cfg = MODE_CONFIGS[args.mode]

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("BioRSP Story Figure Generation")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    start_time = time.time()

    gen = rng.make_rng(args.seed, "story")
    config = BioRSPConfig(
        B=72,
        delta_deg=60.0,
        n_permutations=mode_cfg["n_permutations"],
    )

    print("\n[1/6] Generating spatial coordinates...")
    coords, shape_meta = shapes.generate_coords(mode_cfg["shape"], mode_cfg["n_cells"], gen)
    libsize = expression.simulate_library_size(
        mode_cfg["n_cells"], gen, model="lognormal", params={"mean": 2000, "sigma": 0.5}
    )
    print("[2/6] Generating factorial gene panel...")
    X_factorial, var_names_factorial, truth_df = datasets.make_factorial_panel(
        coords=coords,
        libsize=libsize,
        rng=gen,
        n_per_archetype=mode_cfg["n_per_archetype"],
        include_abstention_stress=True,
    )

    adata_factorial = datasets.package_as_anndata(
        coords,
        X_factorial,
        var_names_factorial,
        obs_meta={"libsize": libsize},
        embedding_key="X_sim",
    )

    print(f"   Created {len(var_names_factorial)} genes across archetypes")
    print(f"   Archetype counts: {truth_df['Archetype'].value_counts().to_dict()}")

    print("[3/6] Scoring genes with BioRSP...")
    main_genes = [g for g in var_names_factorial if "abstention" not in g]
    scores_df = scoring.score_dataset(
        adata_factorial, genes=main_genes, config=config, embedding_key="X_sim"
    )
    results_df = truth_df[truth_df["gene"].isin(main_genes)].merge(scores_df, on="gene", how="left")

    print(f"   Scored {len(results_df)} genes")
    print(f"   Abstention rate: {results_df['abstain_flag'].mean():.1%}")

    print("[4/6] Classifying genes into archetypes (with principled null calibration)...")

    null_mask = results_df["Archetype"].isin(["Basal", "Ubiquitous"])
    null_s = results_df.loc[
        null_mask & (results_df["organization_regime"] == "iid"), "Spatial_Bias_Score"
    ].values

    fpr_target = 0.05 if args.mode == "publication" else 0.10

    if len(null_s) >= 10:
        thresholds = metrics.derive_thresholds_principled(
            null_s_values=null_s,
            fpr_target=fpr_target,
            coverage_split=0.30,
        )
        s_cut = thresholds["s_cut"]
        c_cut = thresholds["c_cut"]
        margin = thresholds["margin"]

        null_stats = metrics.compute_null_statistics(null_s)

        print(f"   Null distribution: mean={null_stats['mean']:.3f}, std={null_stats['std']:.3f}")
        print(f"   Derived thresholds (FPR={fpr_target:.0%}): S_cut={s_cut:.3f}, C_cut={c_cut:.2f}")
        print(f"   Borderline margin: ±{margin:.3f}")
        print(f"   Empirical FPR at S_cut: {thresholds['empirical_fpr']:.1%}")
    else:
        s_cut = _DEFAULT_SPEC.s_cut
        c_cut = _DEFAULT_SPEC.c_cut
        margin = 0.02  # Default margin
        thresholds = {
            "s_cut": s_cut,
            "c_cut": c_cut,
            "margin": margin,
            "n_samples": 0,
            "warning": "Using PlotSpec defaults",
        }
        null_stats = None
        print(f"   ⚠ Insufficient null samples ({len(null_s)}), using PlotSpec defaults")
        print(f"   Using default thresholds: S_cut={s_cut:.3f}, C_cut={c_cut:.2f}")

    valid_mask = ~results_df["abstain_flag"]
    results_df["pred_archetype"] = None
    results_df["confidence"] = None

    if valid_mask.sum() > 0:
        labels, confidences = metrics.classify_with_confidence(
            coverage=results_df.loc[valid_mask, "Coverage"].values,
            spatial_score=results_df.loc[valid_mask, "Spatial_Bias_Score"].values,
            s_cut=s_cut,
            c_cut=c_cut,
            margin=margin,
        )
        results_df.loc[valid_mask, "pred_archetype"] = labels
        results_df.loc[valid_mask, "confidence"] = confidences

        conf_counts = results_df.loc[valid_mask, "confidence"].value_counts()
        print(f"   Confidence breakdown: {conf_counts.to_dict()}")

    print("[5/6] Computing classification metrics...")

    eval_mask = valid_mask & results_df["pred_archetype"].notna()

    y_true = results_df.loc[eval_mask, "Archetype"].values
    y_pred = results_df.loc[eval_mask, "pred_archetype"].values

    class_metrics = metrics.compute_classification_metrics(
        y_true, y_pred, labels=["Ubiquitous", "Gradient", "Basal", "Patchy"]
    )

    print(f"   Overall accuracy: {class_metrics['accuracy']:.1%}")
    print(f"   Macro F1: {class_metrics['macro_f1']:.3f}")

    is_structured = (results_df["organization_regime"] == "structured").values
    s_scores = results_df["Spatial_Bias_Score"].fillna(0).values

    prec_curve = metrics.precision_at_k_curve(
        y_true=is_structured.astype(int),
        y_score=s_scores,
        k_values=[10, 20, 50, min(100, len(results_df))],
    )

    print(
        f"   Precision@20 for structured genes: {prec_curve[prec_curve['k'] == 20]['precision_at_k'].values[0]:.1%}"
    )

    print("[6/6] Running gene-gene module analysis...")

    X_modules, var_names_modules, truth_edges_df, truth_genes_df = (
        datasets.make_module_panel_structured(
            coords=coords,
            libsize=libsize,
            rng=gen,
            n_modules=mode_cfg["n_modules"],
            genes_per_module=mode_cfg["genes_per_module"],
            n_null_genes=mode_cfg["n_null_genes"],
        )
    )

    adata_modules = datasets.package_as_anndata(
        coords, X_modules, var_names_modules, obs_meta={"libsize": libsize}, embedding_key="X_sim"
    )

    pairs_df = scoring.score_pairs(
        adata_modules, genes=var_names_modules, config=config, embedding_key="X_sim"
    )

    pairs_merged = truth_edges_df.merge(pairs_df, on=["gene_a", "gene_b"], how="left")

    valid_pairs = pairs_merged["similarity_profile"].notna()
    if valid_pairs.sum() > 0:
        module_metrics = metrics.module_recovery_metrics_extended(
            predicted_scores=pairs_merged.loc[valid_pairs, "similarity_profile"].values,
            true_edges=pairs_merged.loc[valid_pairs, "is_true_edge"].astype(int).values,
        )

        prevalence = module_metrics["prevalence_baseline"]
        fold_enrichment = module_metrics["fold_enrichment_auprc"]
        print(
            f"   Module AUPRC: {module_metrics['auprc']:.3f} (baseline={prevalence:.1%}, {fold_enrichment:.1f}× enrichment)"
        )
        print(f"   Module AUROC: {module_metrics['auroc']:.3f}")
        print(f"   Module Precision@10: {module_metrics['precision_at_10']:.1%}")
    else:
        module_metrics = {
            "auprc": np.nan,
            "auroc": np.nan,
            "precision_at_10": np.nan,
            "precision_at_50": np.nan,
            "prevalence_baseline": np.nan,
            "fold_enrichment_auprc": np.nan,
        }
        print("   ⚠ No valid pairs for module recovery evaluation")

    print("\n" + "=" * 60)
    print("Generating figures...")

    fig_a = plotting.plot_archetype_scatter(
        coverage=results_df.loc[eval_mask, "Coverage"].values,
        spatial_score=results_df.loc[eval_mask, "Spatial_Bias_Score"].values,
        true_archetypes=results_df.loc[eval_mask, "Archetype"].values,
        c_cut=c_cut,
        s_cut=s_cut,
        title="A. Archetype Classification (C vs S)",
    )
    io.save_figure(fig_a, figures_dir, "fig_story_A_archetypes.png")
    plt.close(fig_a)

    fig_b = plotting.plot_confusion_matrix_styled(
        cm_df=class_metrics["confusion_matrix"],
        title="B. Classification Confusion Matrix",
        accuracy=class_metrics["accuracy"],
    )
    io.save_figure(fig_b, figures_dir, "fig_story_B_confusion.png")
    plt.close(fig_b)

    fig_c = plotting.plot_marker_recovery(
        precision_df=prec_curve,
        title="C. Structured Gene Recovery",
    )
    io.save_figure(fig_c, figures_dir, "fig_story_C_marker_recovery.png")
    plt.close(fig_c)

    fig_d = plotting.plot_module_recovery(
        module_metrics=module_metrics,
        title="D. Gene-Gene Module Recovery",
    )
    io.save_figure(fig_d, figures_dir, "fig_story_D_genegene.png")
    plt.close(fig_d)

    print("Composing one-page figure...")
    fig_combined = plt.figure(figsize=(14, 12))
    gs = fig_combined.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    ax_a = fig_combined.add_subplot(gs[0, 0])
    ax_b = fig_combined.add_subplot(gs[0, 1])
    ax_c = fig_combined.add_subplot(gs[1, 0])
    ax_d = fig_combined.add_subplot(gs[1, 1])

    story_spec = PlotSpec(c_cut=c_cut, s_cut=s_cut)

    for arch in story_spec.get_legend_order():
        mask = (results_df.loc[eval_mask, "Archetype"] == arch).values
        if mask.sum() > 0:
            ax_a.scatter(
                results_df.loc[eval_mask, "Coverage"].values[mask],
                results_df.loc[eval_mask, "Spatial_Bias_Score"].values[mask],
                c=story_spec.get_color(arch),
                label=arch.replace("_", " ").title(),
                s=40,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
            )
    ax_a.axvline(c_cut, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax_a.axhline(s_cut, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax_a.set_xlabel("Coverage (C)")
    ax_a.set_ylabel("Spatial Bias Score (S)")
    ax_a.set_title("A. Archetype Classification", fontweight="bold")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_a.grid(True, alpha=0.3)

    cm = class_metrics["confusion_matrix"]
    cm_norm = cm.div(cm.sum(axis=1), axis=0)
    ax_b.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            val = cm.iloc[i, j]
            pct = cm_norm.iloc[i, j]
            color = "white" if pct > 0.5 else "black"
            ax_b.text(
                j, i, f"{val}\n({pct:.0%})", ha="center", va="center", color=color, fontsize=8
            )
    ax_b.set_xticks(range(len(cm.columns)))
    ax_b.set_yticks(range(len(cm.index)))
    short_labels = {
        "Ubiquitous": "Ubiq.",
        "Gradient": "Regional",
        "Basal": "Sparse",
        "Patchy": "Niche",
    }
    ax_b.set_xticklabels(
        [short_labels.get(c, c) for c in cm.columns], rotation=45, ha="right", fontsize=8
    )
    ax_b.set_yticklabels([short_labels.get(r, r) for r in cm.index], fontsize=8)
    ax_b.set_xlabel("Predicted")
    ax_b.set_ylabel("True")
    ax_b.set_title(f"B. Confusion Matrix (Acc: {class_metrics['accuracy']:.0%})", fontweight="bold")

    ax_c.bar(
        range(len(prec_curve)),
        prec_curve["precision_at_k"].values,
        color="#FF5722",
        alpha=0.8,
        edgecolor="black",
    )
    ax_c.set_xticks(range(len(prec_curve)))
    ax_c.set_xticklabels([f"Top {k}" for k in prec_curve["k"].values], fontsize=8)
    ax_c.set_xlabel("Ranked by S")
    ax_c.set_ylabel("Precision")
    ax_c.set_title("C. Structured Gene Recovery", fontweight="bold")
    ax_c.set_ylim(0, 1.1)
    ax_c.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax_c.grid(axis="y", alpha=0.3)

    metrics_names = ["AUPRC", "AUROC", "P@10", "P@50"]
    metrics_vals = [
        module_metrics["auprc"],
        module_metrics["auroc"],
        module_metrics["precision_at_10"],
        module_metrics["precision_at_50"],
    ]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    ax_d.bar(metrics_names, metrics_vals, color=colors, alpha=0.8, edgecolor="black")
    ax_d.set_ylabel("Score")
    ax_d.set_title("D. Gene-Gene Module Recovery", fontweight="bold")
    ax_d.set_ylim(0, 1.1)
    ax_d.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    for i, (_name, val) in enumerate(zip(metrics_names, metrics_vals)):
        if not np.isnan(val):
            ax_d.text(i, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig_combined.suptitle(
        "BioRSP Validation: One-Page Story", fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    io.save_figure(fig_combined, figures_dir, "fig_story_onepager.png", dpi=300)
    plt.close(fig_combined)

    print("\nGenerating diagnostic figures...")
    diag_dir = figures_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    try:
        if len(null_s) >= 10:
            fig_null = plotting.plot_null_distribution(
                null_s_values=null_s,
                s_cut=s_cut,
                margin=margin,
                fpr_target=fpr_target,
                title="Null S Distribution (FPR-Controlled Threshold)",
            )
            io.save_figure(fig_null, diag_dir, "fig_null_distribution.png", dpi=150)
            plt.close(fig_null)
            print("   ✓ Null distribution diagnostic saved")
        else:
            print("   ⊘ Insufficient null samples for distribution plot")
    except Exception as e:
        print(f"   ✗ Null distribution diagnostic failed: {e}")

    try:
        fig_examples = plotting.plot_archetype_examples(
            coords=coords,
            X=X_factorial,
            truth_df=truth_df,
            var_names=var_names_factorial,
            n_examples_per_archetype=3,
        )
        io.save_figure(fig_examples, diag_dir, "fig_archetype_examples.png", dpi=150)
        plt.close(fig_examples)
        print("   ✓ Archetype examples saved")
    except Exception as e:
        print(f"   ✗ Archetype examples failed: {e}")

    try:
        fig_thresh = plotting.plot_threshold_diagnostics(
            coverage=results_df.loc[eval_mask, "Coverage"].values,
            spatial_score=results_df.loc[eval_mask, "Spatial_Bias_Score"].values,
            true_archetypes=results_df.loc[eval_mask, "Archetype"].values,
            c_cut=c_cut,
            s_cut=s_cut,
        )
        io.save_figure(fig_thresh, diag_dir, "fig_threshold_diagnostics.png", dpi=150)
        plt.close(fig_thresh)
        print("   ✓ Threshold diagnostics saved")
    except Exception as e:
        print(f"   ✗ Threshold diagnostics failed: {e}")

    try:
        coverage_fg = results_df.loc[eval_mask, "Coverage"].values
        coverage_bg = 1.0 - coverage_fg  # Approximation
        fig_support = plotting.plot_support_diagnostics(
            coverage_fg=coverage_fg,
            coverage_bg=coverage_bg,
            spatial_score=results_df.loc[eval_mask, "Spatial_Bias_Score"].values,
            true_archetypes=results_df.loc[eval_mask, "Archetype"].values,
        )
        io.save_figure(fig_support, diag_dir, "fig_support_diagnostics.png", dpi=150)
        plt.close(fig_support)
        print("   ✓ Support diagnostics saved")
    except Exception as e:
        print(f"   ✗ Support diagnostics failed: {e}")

    try:
        fig_misclass, misclass_df = plotting.plot_misclassified_scatter(
            coverage=results_df.loc[eval_mask, "Coverage"].values,
            spatial_score=results_df.loc[eval_mask, "Spatial_Bias_Score"].values,
            true_archetypes=results_df.loc[eval_mask, "Archetype"].values,
            pred_archetypes=results_df.loc[eval_mask, "pred_archetype"].values,
            var_names=results_df.loc[eval_mask, "gene"].tolist(),
            c_cut=c_cut,
            s_cut=s_cut,
        )
        io.save_figure(fig_misclass, diag_dir, "fig_misclassified_scatter.png", dpi=150)
        plt.close(fig_misclass)

        if len(misclass_df) > 0:
            misclass_df.to_csv(diag_dir / "misclassified.csv", index=False)
            print(f"   ✓ Misclassification audit saved ({len(misclass_df)} genes)")

            error_summary = misclass_df.groupby("error_type").size()
            print("      Error type breakdown:")
            for error_type, count in error_summary.items():
                print(f"        {error_type}: {count}")
        else:
            print("   ✓ No misclassifications! (Perfect accuracy)")
    except Exception as e:
        print(f"   ✗ Misclassification audit failed: {e}")

    try:
        pattern_results = results_df.loc[
            eval_mask, ["pattern_variant", "Spatial_Bias_Score", "Archetype"]
        ].copy()
        pattern_results = pattern_results[pattern_results["pattern_variant"] != "none"]

        if len(pattern_results) > 0:
            fig_detect = plotting.plot_pattern_detectability(
                pattern_results=pattern_results,
                title="Pattern Detectability by S Score",
            )
            io.save_figure(fig_detect, diag_dir, "fig_pattern_detectability.png", dpi=150)
            plt.close(fig_detect)
            print("   ✓ Pattern detectability saved")
        else:
            print("   ⊘ No structured patterns to analyze")
    except Exception as e:
        print(f"   ✗ Pattern detectability failed: {e}")

    print("\nWriting outputs...")

    results_df.to_csv(output_dir / "runs.csv", index=False)

    summary_data = {
        "metric": [
            "accuracy",
            "macro_f1",
            "s_cut",
            "c_cut",
            "s_cut_margin",
            "fpr_target",
            "empirical_fpr",
            "n_genes",
            "abstention_rate",
            "n_high_confidence",
            "n_borderline",
            "module_auprc",
            "module_prevalence_baseline",
            "module_fold_enrichment",
            "precision_at_20",
        ],
        "value": [
            class_metrics["accuracy"],
            class_metrics["macro_f1"],
            s_cut,
            c_cut,
            margin,
            fpr_target,
            thresholds.get("empirical_fpr", np.nan),
            len(results_df),
            results_df["abstain_flag"].mean(),
            (results_df["confidence"] == "high").sum(),
            (results_df["confidence"] == "borderline").sum(),
            module_metrics["auprc"],
            module_metrics.get("prevalence_baseline", np.nan),
            module_metrics.get("fold_enrichment_auprc", np.nan),
            (
                prec_curve[prec_curve["k"] == 20]["precision_at_k"].values[0]
                if 20 in prec_curve["k"].values
                else np.nan
            ),
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    elapsed = time.time() - start_time
    io.write_manifest(
        output_dir=output_dir,
        benchmark_name="story_onepager",
        params={
            "mode": args.mode,
            "seed": args.seed,
            **mode_cfg,
        },
        n_replicates=1,
        runtime_seconds=elapsed,
        biorsp_config=config,
        plot_spec=story_spec.to_dict(),
    )

    write_story_report(
        output_dir=output_dir,
        class_metrics=class_metrics,
        prec_curve=prec_curve,
        module_metrics=module_metrics,
        thresholds=thresholds,
        mode=args.mode,
        n_genes=len(results_df),
        elapsed=elapsed,
    )

    write_caption(figures_dir)

    print(f"\n{'=' * 60}")
    print("✓ Story figure generation complete!")
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"  Accuracy: {class_metrics['accuracy']:.1%}")
    print(f"  Figures saved to: {figures_dir}")
    print(f"{'=' * 60}")

    return {
        "accuracy": class_metrics["accuracy"],
        "macro_f1": class_metrics["macro_f1"],
        "module_auprc": module_metrics["auprc"],
        "n_genes": len(results_df),
        "figures_created": list(figures_dir.glob("*.png")),
    }


def write_story_report(
    output_dir: Path,
    class_metrics: dict,
    prec_curve: pd.DataFrame,
    module_metrics: dict,
    thresholds: dict,
    mode: str,
    n_genes: int,
    elapsed: float,
):
    """Write a human-readable report.

    Args:
        output_dir: Output directory.
        class_metrics: Classification metrics dictionary.
        prec_curve: Precision@K DataFrame.
        module_metrics: Gene–gene module metrics dictionary.
        thresholds: Thresholds dictionary.
        mode: Benchmark mode name.
        n_genes: Number of genes scored.
        elapsed: Runtime in seconds.
    """

    acc_threshold = 0.60 if mode == "quick" else 0.80
    acc_pass = class_metrics["accuracy"] >= acc_threshold

    auprc_threshold = 0.25 if mode == "quick" else 0.65
    auprc_pass = module_metrics.get("auprc", 0) >= auprc_threshold

    report = f"""# BioRSP Story Figure Report

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
Mode: {mode}
Runtime: {elapsed:.1f}s


This report validates BioRSP's ability to classify genes into spatial archetypes
and recover gene-gene co-patterning relationships using synthetic ground-truth data.


BioRSP classifies genes into a 2×2 grid based on:
- **Coverage (C)**: Fraction of cells expressing the gene
- **Spatial Bias Score (S)**: Degree of spatial organization

| Archetype | Description | Expected Region |
|-----------|-------------|-----------------|
| Ubiquitous | Ubiquitous expression | High C, Low S |
| Gradient | Broad spatial domain | High C, High S |
| Sparse/Noisy | Rare/scattered | Low C, Low S |
| Patchy | Localized expression | Low C, High S |

**Results:**
- Overall Accuracy: **{class_metrics["accuracy"]:.1%}** {"✓ PASS" if acc_pass else "✗ FAIL"} (threshold: {acc_threshold:.0%})
- Macro F1: **{class_metrics["macro_f1"]:.3f}**
- Derived thresholds: S_cut = {thresholds["s_cut"]:.3f}, C_cut = {thresholds["c_cut"]:.3f}

**Per-class performance:**
"""

    for label, stats in class_metrics["per_class"].items():
        report += f"- {label}: Precision={stats['precision']:.2f}, Recall={stats['recall']:.2f}, F1={stats['f1']:.2f}\n"

    prec_20 = prec_curve[prec_curve["k"] == 20]["precision_at_k"].values
    prec_20[0] if len(prec_20) > 0 else np.nan

    report += """

Top genes ranked by Spatial Bias Score (S) should be enriched for truly structured genes.

| Top K | Precision | True Structured |
|-------|-----------|-----------------|
"""

    for _, row in prec_curve.iterrows():
        report += (
            f"| {int(row['k'])} | {row['precision_at_k']:.0%} | {int(row['n_true_in_top_k'])} |\n"
        )

    report += f"""

Genes sharing the same spatial pattern (module) should have high co-patterning scores.

- AUPRC: **{module_metrics["auprc"]:.3f}** {"✓ PASS" if auprc_pass else "✗ FAIL"} (threshold: {auprc_threshold:.2f})
- AUROC: **{module_metrics["auroc"]:.3f}**
- Precision@10: **{module_metrics["precision_at_10"]:.1%}**
- Precision@50: **{module_metrics["precision_at_50"]:.1%}**


**What to look at:**

1. **Panel A (Scatter)**: Genes should separate into 4 quadrants. Colors show ground truth;
   black dashed lines are derived thresholds.

2. **Panel B (Confusion Matrix)**: Diagonal should be bright (high recall). Off-diagonal
   errors often occur between adjacent quadrants (e.g., regional↔Ubiquitous).

3. **Panel C (Marker Recovery)**: Precision should be well above 50% (random) for top-ranked
   genes, indicating that S effectively identifies structured genes.

4. **Panel D (Module Recovery)**: AUPRC > 0.5 indicates that co-patterning scores can
   distinguish genes in the same spatial module from unrelated genes.


| Check | Result |
|-------|--------|
| Classification Accuracy ≥ {acc_threshold:.0%} | {"✓ PASS" if acc_pass else "✗ FAIL"} |
| Module AUPRC ≥ {auprc_threshold:.2f} | {"✓ PASS" if auprc_pass else "✗ FAIL"} |
| Figures generated | ✓ |

**Overall: {"✓ ALL CHECKS PASS" if (acc_pass and auprc_pass) else "✗ SOME CHECKS FAILED"}**
"""

    with open(output_dir / "report.md", "w") as f:
        f.write(report)

    print(f"Wrote report to {output_dir / 'report.md'}")


def write_caption(figures_dir: Path):
    """Write a caption for the one-page figure.

    Args:
        figures_dir: Figures output directory.
    """

    caption = """Figure: BioRSP Method Validation Through Simulation

(A) Coverage (C) vs Spatial Organization Score (S) for simulated genes with known ground truth.
    Points are colored by true archetype: Ubiquitous (green), regional program (blue),
    sparse/noisy (gray), and niche marker (orange). Dashed lines indicate classification
    thresholds derived from null simulations.

(B) Confusion matrix showing classification accuracy. Each cell shows count and recall
    percentage. High diagonal values indicate correct classification.

(C) Precision of structured gene recovery when ranking genes by S. Values above 50%
    (random baseline) indicate that S effectively identifies spatially organized genes.

(D) Gene-gene module recovery metrics. AUPRC and precision values above 0.5 indicate
    that BioRSP can identify genes sharing spatial patterns (co-patterning).
"""

    with open(figures_dir / "fig_story_onepager_caption.txt", "w") as f:
        f.write(caption)


def main():
    parser = argparse.ArgumentParser(
        description="Generate BioRSP one-page story figure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "validation", "publication"],
        default="quick",
        help="Benchmark mode: quick (~15s), validation (~5min), publication (~15min)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "story"),
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    run_story_benchmark(args)


if __name__ == "__main__":
    main()
