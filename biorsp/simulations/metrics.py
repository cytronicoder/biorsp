"""
Evaluation metrics for simulation benchmarks.

Provides calibration, power, archetype classification, and gene-gene retrieval metrics.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


def fpr_with_ci(
    p_values: np.ndarray, alpha: float = 0.05, method: str = "wilson"
) -> Tuple[float, float, float]:
    """
    Compute false positive rate with confidence interval.

    Parameters
    ----------
    p_values : np.ndarray
        P-values from null simulations
    alpha : float, optional
        Significance threshold
    method : str, optional
        CI method: 'wilson', 'clopper_pearson'

    Returns
    -------
    fpr : float
        False positive rate
    ci_low : float
        Lower confidence bound
    ci_high : float
        Upper confidence bound
    """
    p_clean = p_values[~np.isnan(p_values)]
    n = len(p_clean)
    if n == 0:
        return np.nan, np.nan, np.nan

    n_reject = np.sum(p_clean <= alpha)
    fpr = n_reject / n

    if method == "wilson":
        z = 1.96
        denom = 1 + z**2 / n
        center = (fpr + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(fpr * (1 - fpr) / n + z**2 / (4 * n**2)) / denom
        ci_low = max(0, center - margin)
        ci_high = min(1, center + margin)
    else:
        ci_low = 0 if n_reject == 0 else stats.beta.ppf(0.025, n_reject, n - n_reject + 1)
        ci_high = 1 if n_reject == n else stats.beta.ppf(0.975, n_reject + 1, n - n_reject)

    return fpr, ci_low, ci_high


def ks_uniform(p_values: np.ndarray) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov test for uniformity.

    Parameters
    ----------
    p_values : np.ndarray
        P-values

    Returns
    -------
    ks_stat : float
        KS statistic
    ks_pval : float
        KS p-value
    """
    p_clean = p_values[~np.isnan(p_values)]
    if len(p_clean) < 3:
        return np.nan, np.nan
    return stats.kstest(p_clean, "uniform")


def qq_quantiles(
    p_values: np.ndarray, quantiles: np.ndarray = None, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute QQ plot quantiles with enhanced resolution in significant region.

    Uses non-uniform quantile spacing to ensure sufficient points in the
    significant region (p ≤ alpha) for detailed calibration assessment.

    Parameters
    ----------
    p_values : np.ndarray
        P-values
    quantiles : np.ndarray, optional
        Expected quantiles (default: adaptive spacing with 60% points in [0, alpha])
    alpha : float, optional
        Significance threshold for adaptive spacing (default: 0.05)

    Returns
    -------
    expected : np.ndarray
        Expected quantiles under uniform
    observed : np.ndarray
        Observed quantiles
    """
    p_clean = p_values[~np.isnan(p_values)]
    if len(p_clean) < 3:
        return np.array([]), np.array([])

    if quantiles is None:
        n_total = min(len(p_clean), 100)
        n_sig = int(n_total * 0.6)
        n_nonsig = n_total - n_sig
        quantiles_sig = np.linspace(0, alpha, n_sig, endpoint=False)
        quantiles_nonsig = np.linspace(alpha, 1.0, n_nonsig + 1)[1:]
        quantiles = np.concatenate([quantiles_sig, quantiles_nonsig])

    observed = np.quantile(p_clean, quantiles)
    return quantiles, observed


def power_with_ci(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float, float]:
    """
    Compute power (rejection rate) with CI and abstention rate.

    Parameters
    ----------
    p_values : np.ndarray
        P-values from alternative simulations
    alpha : float, optional
        Significance threshold

    Returns
    -------
    power : float
        Power (fraction of rejections among valid tests)
    ci_low : float
        Lower confidence bound
    ci_high : float
        Upper confidence bound
    abstain_rate : float
        Fraction of abstained tests
    """
    n_total = len(p_values)
    p_valid = p_values[~np.isnan(p_values)]
    n_valid = len(p_valid)

    abstain_rate = 1 - n_valid / n_total if n_total > 0 else np.nan

    if n_valid == 0:
        return np.nan, np.nan, np.nan, abstain_rate

    n_reject = np.sum(p_valid <= alpha)
    power = n_reject / n_valid

    z = 1.96
    denom = 1 + z**2 / n_valid
    center = (power + z**2 / (2 * n_valid)) / denom
    margin = z * np.sqrt(power * (1 - power) / n_valid + z**2 / (4 * n_valid**2)) / denom
    ci_low = max(0, center - margin)
    ci_high = min(1, center + margin)

    return power, ci_low, ci_high, abstain_rate


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list = None) -> pd.DataFrame:
    """
    Compute confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : list, optional
        Label order

    Returns
    -------
    cm_df : pd.DataFrame
        Confusion matrix
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return cm_df


def macro_f1(cm_df: pd.DataFrame) -> float:
    """
    Compute macro-averaged F1 score from confusion matrix.

    Parameters
    ----------
    cm_df : pd.DataFrame
        Confusion matrix

    Returns
    -------
    macro_f1 : float
        Macro F1 score
    """
    f1_scores = []
    for label in cm_df.index:
        tp = cm_df.loc[label, label]
        fp = cm_df[label].sum() - tp
        fn = cm_df.loc[label].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    return np.mean(f1_scores)


def quadrant_accuracy(
    coverage: np.ndarray, spatial_score: np.ndarray, c_cut: float = 0.3, s_cut: float = 0.5
) -> float:
    """
    Compute quadrant-based accuracy for archetype classification.

    Parameters
    ----------
    coverage : np.ndarray
        Coverage values
    spatial_score : np.ndarray
        Spatial scores
    c_cut : float, optional
        Coverage threshold
    s_cut : float, optional
        Spatial score threshold

    Returns
    -------
    accuracy : float
        Fraction in correct quadrant
    """

    return np.nan


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute area under precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels
    y_score : np.ndarray
        Prediction scores

    Returns
    -------
    auprc : float
        AUPRC score
    """
    try:
        from sklearn.metrics import average_precision_score

        return average_precision_score(y_true, y_score)
    except ImportError:
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]
        n_pos = np.sum(y_true)
        if n_pos == 0:
            return np.nan
        tp = np.cumsum(y_true_sorted)
        fp = np.cumsum(1 - y_true_sorted)
        precision = tp / (tp + fp)
        recall = tp / n_pos
        return np.trapz(precision, recall)


def topk_precision(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Compute precision @ k.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels
    y_score : np.ndarray
        Prediction scores
    k : int
        Top-k cutoff

    Returns
    -------
    precision : float
        Precision in top k
    """
    order = np.argsort(-y_score)[:k]
    return np.mean(y_true[order])


def median_abs_delta(x_base: np.ndarray, x_pert: np.ndarray) -> float:
    """
    Compute median absolute difference (robustness metric).

    Parameters
    ----------
    x_base : np.ndarray
        Baseline values
    x_pert : np.ndarray
        Perturbed values

    Returns
    -------
    mad : float
        Median absolute delta
    """
    delta = np.abs(x_base - x_pert)
    return np.nanmedian(delta)


def flip_rate(y_base: np.ndarray, y_pert: np.ndarray) -> float:
    """
    Compute label flip rate.

    Parameters
    ----------
    y_base : np.ndarray
        Baseline labels
    y_pert : np.ndarray
        Perturbed labels

    Returns
    -------
    flip_rate : float
        Fraction of flipped labels
    """
    return np.mean(y_base != y_pert)


def kendall_tau(x_base: np.ndarray, x_pert: np.ndarray) -> float:
    """
    Compute Kendall's tau for rank stability.

    Parameters
    ----------
    x_base : np.ndarray
        Baseline scores
    x_pert : np.ndarray
        Perturbed scores

    Returns
    -------
    tau : float
        Kendall's tau
    """
    return stats.kendalltau(x_base, x_pert).correlation


def derive_thresholds_from_null(
    null_s_values: np.ndarray,
    null_c_values: np.ndarray = None,
    s_quantile: float = 0.95,
    c_quantile: float = 0.30,
) -> dict:
    """
    Derive classification thresholds from null distribution.

    Parameters
    ----------
    null_s_values : np.ndarray
        Spatial scores from null simulations
    null_c_values : np.ndarray, optional
        Coverage values from null simulations
    s_quantile : float, optional
        Quantile for S threshold (default: 0.95 = 5% FPR)
    c_quantile : float, optional
        Quantile for C threshold (default: 0.30 = lower tercile)

    Returns
    -------
    thresholds : dict
        {'s_cut': float, 'c_cut': float, 'n_samples': int, 'quantiles_used': dict}
    """
    s_clean = null_s_values[~np.isnan(null_s_values)]

    if len(s_clean) < 10:
        return {
            "s_cut": 0.15,
            "c_cut": 0.30,
            "n_samples": len(s_clean),
            "quantiles_used": {"s": s_quantile, "c": c_quantile},
            "warning": "Insufficient null samples, using defaults",
        }

    s_cut = np.quantile(s_clean, s_quantile)

    if null_c_values is not None:
        c_clean = null_c_values[~np.isnan(null_c_values)]
        c_cut = np.quantile(c_clean, c_quantile) if len(c_clean) > 0 else 0.30
    else:
        c_cut = 0.30  # Default based on typical iid coverage

    return {
        "s_cut": float(s_cut),
        "c_cut": float(c_cut),
        "n_samples": len(s_clean),
        "quantiles_used": {"s": s_quantile, "c": c_quantile},
    }


def classify_by_quadrant(
    coverage: np.ndarray,
    spatial_score: np.ndarray,
    c_cut: float = 0.30,
    s_cut: float = 0.15,
) -> np.ndarray:
    """
    Classify genes into 4 archetypes based on (C, S) quadrants.

    Parameters
    ----------
    coverage : np.ndarray
        Coverage values
    spatial_score : np.ndarray
        Spatial organization scores
    c_cut : float
        Coverage threshold
    s_cut : float
        Spatial score threshold

    Returns
    -------
    labels : np.ndarray
        String labels: 'housekeeping', 'regional_program', 'sparse_noise', 'niche_marker'
    """
    labels = np.empty(len(coverage), dtype=object)

    high_c = coverage >= c_cut
    high_s = spatial_score >= s_cut

    labels[high_c & ~high_s] = "housekeeping"
    labels[high_c & high_s] = "regional_program"
    labels[~high_c & ~high_s] = "sparse_noise"
    labels[~high_c & high_s] = "niche_marker"

    return labels


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list = None
) -> dict:
    """
    Compute classification metrics including confusion matrix and per-class stats.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : list, optional
        Label order

    Returns
    -------
    metrics : dict
        'accuracy', 'confusion_matrix', 'per_class' (precision, recall, f1)
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels)
    total = len(y_true)
    correct = sum(cm.iloc[i, i] for i in range(len(labels)))
    accuracy = correct / total if total > 0 else 0.0

    per_class = {}
    for label in labels:
        tp = cm.loc[label, label] if label in cm.index and label in cm.columns else 0
        fp = cm[label].sum() - tp if label in cm.columns else 0
        fn = cm.loc[label].sum() - tp if label in cm.index else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(tp + fn),
        }

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "per_class": per_class,
        "macro_f1": np.mean([pc["f1"] for pc in per_class.values()]),
    }


def precision_at_k_curve(
    y_true: np.ndarray, y_score: np.ndarray, k_values: list = None
) -> pd.DataFrame:
    """
    Compute precision at multiple K values.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (1 = positive)
    y_score : np.ndarray
        Scores (higher = more positive)
    k_values : list, optional
        K values to evaluate

    Returns
    -------
    curve_df : pd.DataFrame
        Columns: k, precision_at_k, n_true_in_top_k
    """
    if k_values is None:
        n = len(y_true)
        k_values = [5, 10, 20, 50, min(100, n)]

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    rows = []
    for k in k_values:
        if k > len(y_sorted):
            k = len(y_sorted)
        top_k = y_sorted[:k]
        n_true = np.sum(top_k)
        prec = n_true / k if k > 0 else 0
        rows.append({"k": k, "precision_at_k": prec, "n_true_in_top_k": int(n_true)})

    return pd.DataFrame(rows)


def module_recovery_metrics(
    predicted_scores: np.ndarray,
    true_edges: np.ndarray,
    score_threshold: float = None,
) -> dict:
    """
    Compute module recovery metrics for gene-gene analysis.

    Parameters
    ----------
    predicted_scores : np.ndarray
        Pairwise similarity scores
    true_edges : np.ndarray
        Binary: 1 if true module edge, 0 otherwise
    score_threshold : float, optional
        Threshold for calling predicted edges

    Returns
    -------
    metrics : dict
        'auprc', 'auroc', 'precision_at_10', 'precision_at_50'
    """
    auprc_val = auprc(true_edges, predicted_scores)

    try:
        from sklearn.metrics import roc_auc_score

        auroc_val = roc_auc_score(true_edges, predicted_scores)
    except (ImportError, ValueError):
        auroc_val = np.nan

    prec_10 = topk_precision(true_edges, predicted_scores, k=10)
    prec_50 = topk_precision(true_edges, predicted_scores, k=50)

    return {
        "auprc": auprc_val,
        "auroc": auroc_val,
        "precision_at_10": prec_10,
        "precision_at_50": prec_50,
    }


def embedding_stability_metrics(
    scores_list: list,
    labels_list: list = None,
) -> dict:
    """
    Compute stability metrics across multiple embeddings.

    Parameters
    ----------
    scores_list : list of np.ndarray
        Spatial scores from different embeddings
    labels_list : list of np.ndarray, optional
        Predicted labels from different embeddings

    Returns
    -------
    metrics : dict
        'score_correlation': mean pairwise correlation of S
        'label_agreement': mean pairwise agreement of labels
        'score_std': mean std of S across embeddings per gene
    """
    n_embeddings = len(scores_list)

    correlations = []
    for i in range(n_embeddings):
        for j in range(i + 1, n_embeddings):
            s1 = scores_list[i]
            s2 = scores_list[j]
            mask = ~(np.isnan(s1) | np.isnan(s2))
            if mask.sum() > 3:
                corr = np.corrcoef(s1[mask], s2[mask])[0, 1]
                correlations.append(corr)

    stacked = np.vstack(scores_list)
    score_std = np.nanstd(stacked, axis=0)

    result = {
        "score_correlation": np.mean(correlations) if correlations else np.nan,
        "score_correlation_min": np.min(correlations) if correlations else np.nan,
        "score_std_mean": np.nanmean(score_std),
        "n_embeddings": n_embeddings,
    }

    if labels_list is not None and len(labels_list) == n_embeddings:
        agreements = []
        for i in range(n_embeddings):
            for j in range(i + 1, n_embeddings):
                agree = np.mean(labels_list[i] == labels_list[j])
                agreements.append(agree)
        result["label_agreement"] = np.mean(agreements) if agreements else np.nan

    return result


def compute_null_statistics(null_s_values: np.ndarray) -> dict:
    """
    Compute comprehensive statistics from null S distribution.

    Parameters
    ----------
    null_s_values : np.ndarray
        S scores from null (iid) simulations

    Returns
    -------
    stats : dict
        Comprehensive statistics for threshold derivation
    """
    s_clean = null_s_values[~np.isnan(null_s_values)]

    if len(s_clean) < 5:
        return {
            "n_samples": len(s_clean),
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "iqr": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "warning": "Insufficient samples",
        }

    return {
        "n_samples": len(s_clean),
        "mean": float(np.mean(s_clean)),
        "std": float(np.std(s_clean)),
        "median": float(np.median(s_clean)),
        "q25": float(np.percentile(s_clean, 25)),
        "q75": float(np.percentile(s_clean, 75)),
        "iqr": float(np.percentile(s_clean, 75) - np.percentile(s_clean, 25)),
        "q90": float(np.percentile(s_clean, 90)),
        "q95": float(np.percentile(s_clean, 95)),
        "q99": float(np.percentile(s_clean, 99)),
        "min": float(np.min(s_clean)),
        "max": float(np.max(s_clean)),
    }


def derive_thresholds_principled(
    null_s_values: np.ndarray,
    fpr_target: float = 0.05,
    coverage_split: float = 0.30,
) -> dict:
    """
    Derive classification thresholds from null distribution with principled FPR control.

    This is the CORRECT way to derive thresholds:
    - S_cut controls FPR at target level (e.g., 5% false positive rate for structured calls)
    - C_cut splits low/high coverage (biologically motivated or data-derived)
    - Margin provides uncertainty quantification

    Parameters
    ----------
    null_s_values : np.ndarray
        S scores from null ensemble (IID + confounded nulls)
    fpr_target : float
        Target false positive rate (default: 0.05 = 5%)
    coverage_split : float
        Coverage threshold for low/high split (default: 0.30)

    Returns
    -------
    thresholds : dict
        Comprehensive threshold info for audit trail
    """
    null_stats = compute_null_statistics(null_s_values)

    if null_stats.get("warning"):
        return {
            "s_cut": 0.15,
            "c_cut": coverage_split,
            "margin": 0.05,
            "fpr_target": fpr_target,
            "empirical_fpr": np.nan,
            "null_stats": null_stats,
            "method": "default_fallback",
            "warning": "Insufficient null samples, using conservative defaults",
        }

    quantile = 1.0 - fpr_target
    s_clean = null_s_values[~np.isnan(null_s_values)]
    s_cut = np.percentile(s_clean, quantile * 100)

    margin = null_stats["iqr"] / 2.0

    empirical_fpr = np.mean(s_clean >= s_cut) if len(s_clean) > 0 else np.nan

    return {
        "s_cut": float(s_cut),
        "c_cut": float(coverage_split),
        "margin": float(margin),
        "fpr_target": fpr_target,
        "empirical_fpr": float(empirical_fpr),
        "quantile_used": quantile,
        "null_stats": null_stats,
        "method": f"null_q{int(quantile * 100)}",
    }


def classify_with_confidence(
    coverage: np.ndarray,
    spatial_score: np.ndarray,
    s_cut: float,
    c_cut: float,
    margin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify genes into 4 archetypes with confidence levels.

    Confidence levels:
    - "high": S is clearly above/below threshold (by more than margin)
    - "borderline": S is within margin of threshold
    - "low": Not applicable (all non-borderline are "high")

    Parameters
    ----------
    coverage : np.ndarray
        Coverage values (C)
    spatial_score : np.ndarray
        Spatial organization scores (S)
    s_cut : float
        S threshold from null calibration
    c_cut : float
        Coverage threshold
    margin : float
        Uncertainty margin from null IQR

    Returns
    -------
    labels : np.ndarray
        Archetype labels
    confidence : np.ndarray
        Confidence levels: 'high', 'borderline'
    """
    n = len(coverage)
    labels = np.empty(n, dtype=object)
    confidence = np.empty(n, dtype=object)

    high_c = coverage >= c_cut
    high_s = spatial_score >= s_cut

    labels[high_c & ~high_s] = "housekeeping"
    labels[high_c & high_s] = "regional_program"
    labels[~high_c & ~high_s] = "sparse_noise"
    labels[~high_c & high_s] = "niche_marker"

    dist_from_s_cut = np.abs(spatial_score - s_cut)
    confidence[:] = "high"
    borderline_mask = dist_from_s_cut < margin
    confidence[borderline_mask] = "borderline"

    return labels, confidence


def compute_fpr_at_threshold(null_s_values: np.ndarray, s_cut: float) -> float:
    """
    Compute empirical false positive rate at given threshold.

    Parameters
    ----------
    null_s_values : np.ndarray
        S scores from null simulations
    s_cut : float
        Threshold

    Returns
    -------
    fpr : float
        Fraction of nulls exceeding threshold
    """
    s_clean = null_s_values[~np.isnan(null_s_values)]
    if len(s_clean) == 0:
        return np.nan
    return float(np.mean(s_clean >= s_cut))


def module_recovery_metrics_extended(
    predicted_scores: np.ndarray,
    true_edges: np.ndarray,
    score_threshold: float = None,
) -> dict:
    """
    Compute module recovery metrics WITH prevalence baseline and fold enrichment.

    Parameters
    ----------
    predicted_scores : np.ndarray
        Pairwise similarity scores
    true_edges : np.ndarray
        Binary: 1 if true module edge, 0 otherwise
    score_threshold : float, optional
        Threshold for calling predicted edges

    Returns
    -------
    metrics : dict
        Extended metrics including prevalence baseline and fold enrichment
    """
    n_total = len(true_edges)
    n_positive = np.sum(true_edges)
    prevalence = n_positive / n_total if n_total > 0 else 0.0

    auprc_val = auprc(true_edges, predicted_scores)

    try:
        from sklearn.metrics import roc_auc_score

        auroc_val = roc_auc_score(true_edges, predicted_scores)
    except (ImportError, ValueError):
        auroc_val = np.nan

    prec_10 = topk_precision(true_edges, predicted_scores, k=10)
    prec_50 = topk_precision(true_edges, predicted_scores, k=50)

    fold_auprc = auprc_val / prevalence if prevalence > 0 else np.nan
    fold_prec_10 = prec_10 / prevalence if prevalence > 0 else np.nan
    fold_prec_50 = prec_50 / prevalence if prevalence > 0 else np.nan

    return {
        "auprc": auprc_val,
        "auroc": auroc_val,
        "precision_at_10": prec_10,
        "precision_at_50": prec_50,
        "prevalence_baseline": prevalence,
        "n_true_edges": int(n_positive),
        "n_total_pairs": n_total,
        "fold_enrichment_auprc": fold_auprc,
        "fold_enrichment_prec10": fold_prec_10,
        "fold_enrichment_prec50": fold_prec_50,
    }


def build_calibration_table(
    null_results_df: pd.DataFrame,
    coverage_bins: list = None,
    quantiles: list = None,
) -> pd.DataFrame:
    """
    Build calibration threshold table from null simulation results.

    Creates a lookup table of S thresholds conditional on (geometry, N, coverage_bin).
    This enables principled classification where "High S" means S > S_p95 for that
    specific coverage regime, avoiding sparse-noise being misclassified as niche.

    Parameters
    ----------
    null_results_df : pd.DataFrame
        Results from null simulations with columns:
        - shape (geometry)
        - N (sample size)
        - spatial_score (S)
        - coverage (C)
    coverage_bins : list, optional
        Coverage bin edges (default: [0, 0.05, 0.10, 0.20, 0.50, 1.0])
    quantiles : list, optional
        Quantiles to compute (default: [0.90, 0.95, 0.99])

    Returns
    -------
    calibration_df : pd.DataFrame
        Columns: geometry, N, C_bin, S_p90, S_p95, S_p99, n_reps, C_bin_label
    """
    if coverage_bins is None:
        coverage_bins = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]
    if quantiles is None:
        quantiles = [0.90, 0.95, 0.99]

    df = null_results_df.copy()
    df["C_bin"] = pd.cut(
        df["Coverage"],
        bins=coverage_bins,
        labels=[
            f"{coverage_bins[i]:.2f}-{coverage_bins[i + 1]:.2f}"
            for i in range(len(coverage_bins) - 1)
        ],
        include_lowest=True,
    )

    rows = []
    for (shape, N, c_bin), group in df.groupby(["shape", "N", "C_bin"], observed=True):
        s_values = group["Spatial_Bias_Score"].dropna().values
        if len(s_values) < 5:
            continue

        row = {
            "geometry": shape,
            "N": N,
            "C_bin_label": c_bin,
            "n_reps": len(s_values),
            "S_mean": float(np.mean(s_values)),
            "S_std": float(np.std(s_values)),
        }

        for q in quantiles:
            row[f"S_p{int(q * 100)}"] = float(np.percentile(s_values, q * 100))

        rows.append(row)

    return pd.DataFrame(rows)


def lookup_calibrated_threshold(
    calibration_table: pd.DataFrame,
    geometry: str,
    N: int,
    coverage: float,
    quantile: str = "S_p95",
) -> float:
    """
    Look up calibrated S threshold for given conditions.

    Parameters
    ----------
    calibration_table : pd.DataFrame
        Table from build_calibration_table()
    geometry : str
        Cell geometry (shape)
    N : int
        Sample size
    coverage : float
        Observed coverage value
    quantile : str
        Which quantile threshold to use (default: "S_p95")

    Returns
    -------
    s_cut : float
        Calibrated S threshold, or default 0.15 if not found
    """
    subset = calibration_table[
        (calibration_table["geometry"] == geometry) & (calibration_table["N"] == N)
    ]

    if len(subset) == 0:
        subset = calibration_table[calibration_table["N"] == N]

    if len(subset) == 0:
        return 0.15

    for _, row in subset.iterrows():
        bin_label = row["C_bin_label"]
        lo, hi = map(float, bin_label.split("-"))
        if lo <= coverage < hi or (coverage >= hi and hi == 1.0):
            return row[quantile]

    return float(subset[quantile].median())


def compute_min_expr_cells(N: int, base: int = 30, fraction: float = 0.01) -> int:
    """
    Compute minimum expressing cells threshold.

    For declaring niche_marker or localized_program, require n_expr_cells >= threshold
    to avoid false positives from RMS tail behavior on sparse genes.

    Parameters
    ----------
    N : int
        Total number of cells
    base : int
        Minimum absolute threshold (default: 30)
    fraction : float
        Minimum as fraction of N (default: 0.01 = 1%)

    Returns
    -------
    min_expr : int
        Minimum expressing cells required
    """
    return max(base, int(np.ceil(fraction * N)))


def apply_expr_gating(
    labels: np.ndarray,
    n_expr_cells: np.ndarray,
    N: int,
    min_base: int = 30,
    min_fraction: float = 0.01,
) -> np.ndarray:
    """
    Apply minimum expressing cells gating to archetype labels.

    Genes labeled as "niche_marker" but with too few expressing cells
    are reclassified as "sparse_noise".

    Parameters
    ----------
    labels : np.ndarray
        Archetype labels from classification
    n_expr_cells : np.ndarray
        Number of expressing cells per gene
    N : int
        Total number of cells
    min_base : int
        Minimum absolute threshold
    min_fraction : float
        Minimum as fraction of N

    Returns
    -------
    gated_labels : np.ndarray
        Labels after gating
    """
    min_thresh = compute_min_expr_cells(N, min_base, min_fraction)
    gated = labels.copy()
    mask = (labels == "niche_marker") & (n_expr_cells < min_thresh)
    gated[mask] = "sparse_noise"
    return gated


def compute_module_ari_nmi(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict:
    """
    Compute Adjusted Rand Index and Normalized Mutual Information for module recovery.

    Parameters
    ----------
    true_labels : np.ndarray
        Ground truth module assignments
    predicted_labels : np.ndarray
        Predicted cluster assignments

    Returns
    -------
    metrics : dict
        'ari', 'nmi', 'n_genes', 'n_true_modules', 'n_pred_clusters'
    """
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    except ImportError:
        ari, nmi = np.nan, np.nan

    return {
        "ari": float(ari),
        "nmi": float(nmi),
        "n_genes": len(true_labels),
        "n_true_modules": len(np.unique(true_labels)),
        "n_pred_clusters": len(np.unique(predicted_labels)),
    }


def hierarchical_cluster_from_similarity(
    similarity_matrix: np.ndarray,
    n_clusters: int,
    method: str = "average",
) -> np.ndarray:
    """
    Cluster genes using hierarchical clustering on similarity matrix.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Square similarity matrix (higher = more similar)
    n_clusters : int
        Number of clusters to form
    method : str
        Linkage method: 'average', 'complete', 'single', 'ward'

    Returns
    -------
    labels : np.ndarray
        Cluster assignments
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.clip(distance_matrix, 0, 1)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    condensed = squareform(distance_matrix)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, n_clusters, criterion="maxclust")

    return labels - 1  # Zero-indexed


def within_module_pair_classification(
    similarity_matrix: np.ndarray,
    true_module_labels: np.ndarray,
) -> dict:
    """
    Evaluate module recovery as a pairwise classification task.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Pairwise similarity scores
    true_module_labels : np.ndarray
        Ground truth module assignments

    Returns
    -------
    metrics : dict
        AUPRC, AUROC, precision at K for within-module pair detection
    """
    n = len(true_module_labels)
    scores = []
    labels = []
    for i in range(n):
        for j in range(i + 1, n):
            scores.append(similarity_matrix[i, j])
            labels.append(1 if true_module_labels[i] == true_module_labels[j] else 0)
    scores = np.array(scores)
    labels = np.array(labels)

    return module_recovery_metrics_extended(scores, labels)


def compute_paired_deltas(
    baseline_scores: np.ndarray,
    distorted_scores: np.ndarray,
) -> dict:
    """
    Compute paired delta statistics for robustness evaluation.

    Parameters
    ----------
    baseline_scores : np.ndarray
        S scores from baseline (undistorted) data
    distorted_scores : np.ndarray
        S scores from distorted data (same replicates)

    Returns
    -------
    metrics : dict
        Paired delta statistics
    """
    assert len(baseline_scores) == len(distorted_scores)

    deltas = distorted_scores - baseline_scores
    abs_deltas = np.abs(deltas)

    valid_mask = ~(np.isnan(baseline_scores) | np.isnan(distorted_scores))
    if valid_mask.sum() >= 3:
        corr = np.corrcoef(baseline_scores[valid_mask], distorted_scores[valid_mask])[0, 1]
    else:
        corr = np.nan

    return {
        "delta_mean": float(np.nanmean(deltas)),
        "delta_median": float(np.nanmedian(deltas)),
        "delta_std": float(np.nanstd(deltas)),
        "abs_delta_median": float(np.nanmedian(abs_deltas)),
        "abs_delta_iqr": float(np.nanpercentile(abs_deltas, 75) - np.nanpercentile(abs_deltas, 25)),
        "correlation": float(corr),
        "n_pairs": int(valid_mask.sum()),
    }
