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
    p_values: np.ndarray, quantiles: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute QQ plot quantiles (expected vs observed).

    Parameters
    ----------
    p_values : np.ndarray
        P-values
    quantiles : np.ndarray, optional
        Expected quantiles (default: linspace)

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
        quantiles = np.linspace(0, 1, min(len(p_clean), 100))

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
