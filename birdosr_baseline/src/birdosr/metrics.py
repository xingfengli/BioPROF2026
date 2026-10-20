from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


UNKNOWN_ID = -1


@dataclass(frozen=True)
class ThresholdResult:
    tau: float
    acc_k: float
    acc_u: float
    acc: float


@dataclass(frozen=True)
class ClasswiseThresholdResult:
    thresholds: np.ndarray
    acc_k: float
    acc_u: float
    acc: float


def _as_array(values: Any) -> np.ndarray:
    return np.asarray(values)


def binary_auroc(y_is_known: np.ndarray, scores: np.ndarray) -> float:
    y = _as_array(y_is_known).astype(bool)
    s = _as_array(scores).astype(float)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)

    # Average ranks for ties.
    unique_scores, inverse = np.unique(s, return_inverse=True)
    for group_id in range(len(unique_scores)):
        idx = np.where(inverse == group_id)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()

    n_pos = y.sum()
    n_neg = (~y).sum()
    rank_sum_pos = ranks[y].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def fpr_at_tpr(y_is_known: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    y = _as_array(y_is_known).astype(bool)
    s = _as_array(scores).astype(float)
    known_scores = s[y]
    unknown_scores = s[~y]
    if len(known_scores) == 0 or len(unknown_scores) == 0:
        return float("nan")
    tau = float(np.quantile(known_scores, 1.0 - target_tpr, method="lower"))
    return float((unknown_scores >= tau).mean())


def oscr(y_true: np.ndarray, y_pred_known: np.ndarray, scores: np.ndarray) -> float:
    y_true = _as_array(y_true).astype(int)
    y_pred_known = _as_array(y_pred_known).astype(int)
    scores = _as_array(scores).astype(float)

    known = y_true != UNKNOWN_ID
    unknown = ~known
    if known.sum() == 0 or unknown.sum() == 0:
        return float("nan")

    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    fprs = []
    ccrs = []
    correct_known = (y_pred_known == y_true) & known
    for tau in thresholds:
        accepted = scores >= tau
        fprs.append(float((accepted & unknown).sum() / unknown.sum()))
        ccrs.append(float((accepted & correct_known).sum() / known.sum()))

    order = np.argsort(fprs)
    return float(np.trapz(np.asarray(ccrs)[order], np.asarray(fprs)[order]))


def apply_threshold(y_pred_known: np.ndarray, scores: np.ndarray, tau: float) -> np.ndarray:
    y_pred_known = _as_array(y_pred_known).astype(int)
    scores = _as_array(scores).astype(float)
    return np.where(scores >= tau, y_pred_known, UNKNOWN_ID)


def apply_classwise_threshold(y_pred_known: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y_pred_known = _as_array(y_pred_known).astype(int)
    scores = _as_array(scores).astype(float)
    thresholds = _as_array(thresholds).astype(float)
    sample_thresholds = thresholds[np.clip(y_pred_known, 0, len(thresholds) - 1)]
    return np.where(scores >= sample_thresholds, y_pred_known, UNKNOWN_ID)


def acc_components(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true = _as_array(y_true).astype(int)
    y_pred = _as_array(y_pred).astype(int)
    known = y_true != UNKNOWN_ID
    unknown = ~known

    acc_k = float((y_pred[known] == y_true[known]).mean()) if known.any() else float("nan")
    acc_u = float((y_pred[unknown] == UNKNOWN_ID).mean()) if unknown.any() else float("nan")
    if np.isnan(acc_k):
        acc = acc_u
    elif np.isnan(acc_u):
        acc = acc_k
    else:
        acc = 0.5 * acc_k + 0.5 * acc_u
    return acc_k, acc_u, float(acc)


def calibrate_tau(y_true: np.ndarray, y_pred_known: np.ndarray, scores: np.ndarray) -> ThresholdResult:
    scores = _as_array(scores).astype(float)
    candidates = np.r_[np.inf, np.sort(np.unique(scores)), -np.inf]
    best = ThresholdResult(tau=float("-inf"), acc_k=-1.0, acc_u=-1.0, acc=-1.0)
    for tau in candidates:
        y_pred = apply_threshold(y_pred_known, scores, float(tau))
        acc_k, acc_u, acc = acc_components(y_true, y_pred)
        if acc > best.acc:
            best = ThresholdResult(float(tau), acc_k, acc_u, acc)
    return best


def _balanced_threshold_for_masks(scores: np.ndarray, positive: np.ndarray, negative: np.ndarray, fallback: float) -> float:
    if not positive.any() and not negative.any():
        return float(fallback)
    if not positive.any():
        return float("inf")
    if not negative.any():
        return float("-inf")

    candidates = np.r_[np.inf, np.sort(np.unique(scores[positive | negative])), -np.inf]
    best_tau = float(fallback)
    best_score = -1.0
    for tau in candidates:
        tpr = float((scores[positive] >= tau).mean())
        tnr = float((scores[negative] < tau).mean())
        objective = 0.5 * tpr + 0.5 * tnr
        if objective > best_score:
            best_tau = float(tau)
            best_score = objective
    return best_tau


def calibrate_classwise_tau(
    y_true: np.ndarray,
    y_pred_known: np.ndarray,
    scores: np.ndarray,
    num_classes: int,
) -> ClasswiseThresholdResult:
    y_true = _as_array(y_true).astype(int)
    y_pred_known = _as_array(y_pred_known).astype(int)
    scores = _as_array(scores).astype(float)
    global_tau = calibrate_tau(y_true, y_pred_known, scores).tau

    thresholds = np.full(num_classes, float(global_tau), dtype=float)
    for class_id in range(num_classes):
        predicted_class = y_pred_known == class_id
        positive = predicted_class & (y_true == class_id)
        negative = predicted_class & (y_true == UNKNOWN_ID)
        thresholds[class_id] = _balanced_threshold_for_masks(scores, positive, negative, float(global_tau))

    y_pred = apply_classwise_threshold(y_pred_known, scores, thresholds)
    acc_k, acc_u, acc = acc_components(y_true, y_pred)
    return ClasswiseThresholdResult(thresholds=thresholds, acc_k=acc_k, acc_u=acc_u, acc=acc)


def _subset_tau_candidates(scores: np.ndarray, max_candidates: int) -> np.ndarray:
    scores = _as_array(scores).astype(float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.asarray([np.inf, -np.inf], dtype=float)
    unique = np.sort(np.unique(scores))
    if len(unique) > max(2, int(max_candidates) - 2):
        quantiles = np.linspace(0.0, 1.0, max(2, int(max_candidates) - 2))
        unique = np.unique(np.quantile(unique, quantiles))
    return np.r_[np.inf, unique, -np.inf]


def _subset_objective(
    y_true: np.ndarray,
    y_pred_known: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    true_labels: np.ndarray,
    subset_size: int,
    objective: str,
) -> tuple[float, dict[str, float]]:
    y_pred = apply_classwise_threshold(y_pred_known, scores, thresholds)
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "true_label": true_labels,
        }
    )
    subset_report = unknown_subset_acc_report(df, subset_size=subset_size)
    acc_k, acc_u, acc = acc_components(y_true, y_pred)
    metrics = {
        "ACC_k": acc_k,
        "ACC_u": acc_u,
        "ACC": acc,
        **subset_report,
    }
    objective_key = str(objective)
    if objective_key not in metrics or np.isnan(float(metrics[objective_key])):
        objective_key = "ACC_subset_mean"
    value = float(metrics.get(objective_key, float("nan")))
    if np.isnan(value):
        value = float(acc)
    return value, metrics


def calibrate_subset_classwise_tau(
    y_true: np.ndarray,
    y_pred_known: np.ndarray,
    scores: np.ndarray,
    true_labels: np.ndarray,
    num_classes: int,
    subset_size: int = 3,
    max_candidates: int = 200,
    passes: int = 2,
    objective: str = "ACC_subset_mean",
) -> ClasswiseThresholdResult:
    """Calibrate class-wise thresholds for the unknown-subset ACC objective.

    The protocol evaluates every combination of `subset_size` unknown species.
    This function keeps the class-wise-threshold decision form, but searches
    thresholds on the validation set to maximize the same subset-averaged ACC
    used for reporting.
    """
    y_true = _as_array(y_true).astype(int)
    y_pred_known = _as_array(y_pred_known).astype(int)
    scores = _as_array(scores).astype(float)
    true_labels = _as_array(true_labels).astype(str)

    thresholds = calibrate_classwise_tau(y_true, y_pred_known, scores, num_classes).thresholds.copy()
    best_objective, best_metrics = _subset_objective(
        y_true,
        y_pred_known,
        scores,
        thresholds,
        true_labels,
        int(subset_size),
        objective,
    )

    for _ in range(max(1, int(passes))):
        improved = False
        for class_id in range(num_classes):
            predicted_class = y_pred_known == class_id
            candidates = _subset_tau_candidates(scores[predicted_class], int(max_candidates))
            class_best_tau = thresholds[class_id]
            class_best_objective = best_objective
            class_best_metrics = best_metrics
            for tau in candidates:
                trial = thresholds.copy()
                trial[class_id] = float(tau)
                value, metrics = _subset_objective(
                    y_true,
                    y_pred_known,
                    scores,
                    trial,
                    true_labels,
                    int(subset_size),
                    objective,
                )
                better = value > class_best_objective + 1e-12
                tied_with_better_unknown = (
                    abs(value - class_best_objective) <= 1e-12
                    and metrics.get("ACC_u_subset_mean", -1.0) > class_best_metrics.get("ACC_u_subset_mean", -1.0)
                )
                if better or tied_with_better_unknown:
                    class_best_tau = float(tau)
                    class_best_objective = value
                    class_best_metrics = metrics
            if class_best_objective > best_objective + 1e-12 or class_best_tau != thresholds[class_id]:
                thresholds[class_id] = class_best_tau
                best_objective = class_best_objective
                best_metrics = class_best_metrics
                improved = True
        if not improved:
            break

    y_pred = apply_classwise_threshold(y_pred_known, scores, thresholds)
    acc_k, acc_u, acc = acc_components(y_true, y_pred)
    return ClasswiseThresholdResult(thresholds=thresholds, acc_k=acc_k, acc_u=acc_u, acc=acc)


def calibrate_subset_tau(
    y_true: np.ndarray,
    y_pred_known: np.ndarray,
    scores: np.ndarray,
    true_labels: np.ndarray,
    subset_size: int = 3,
    max_candidates: int = 200,
    objective: str = "ACC_subset_mean",
) -> ThresholdResult:
    """Calibrate one global threshold for the unknown-subset ACC objective."""
    y_true = _as_array(y_true).astype(int)
    y_pred_known = _as_array(y_pred_known).astype(int)
    scores = _as_array(scores).astype(float)
    true_labels = _as_array(true_labels).astype(str)
    candidates = _subset_tau_candidates(scores, int(max_candidates))

    best_tau = float("-inf")
    best_value = -1.0
    best_metrics = {"ACC_k": -1.0, "ACC_u": -1.0, "ACC": -1.0, "ACC_u_subset_mean": -1.0}
    objective_key = str(objective)
    for tau in candidates:
        y_pred = apply_threshold(y_pred_known, scores, float(tau))
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "true_label": true_labels})
        subset_report = unknown_subset_acc_report(df, subset_size=int(subset_size))
        acc_k, acc_u, acc = acc_components(y_true, y_pred)
        metrics = {"ACC_k": acc_k, "ACC_u": acc_u, "ACC": acc, **subset_report}
        value = float(metrics.get(objective_key, float("nan")))
        if np.isnan(value):
            value = float(metrics.get("ACC_subset_mean", acc))
        better = value > best_value + 1e-12
        tied_with_better_unknown = (
            abs(value - best_value) <= 1e-12
            and metrics.get("ACC_u_subset_mean", -1.0) > best_metrics.get("ACC_u_subset_mean", -1.0)
        )
        if better or tied_with_better_unknown:
            best_tau = float(tau)
            best_value = value
            best_metrics = metrics

    return ThresholdResult(
        tau=best_tau,
        acc_k=float(best_metrics["ACC_k"]),
        acc_u=float(best_metrics["ACC_u"]),
        acc=float(best_metrics["ACC"]),
    )


def main_metrics(y_true: np.ndarray, y_pred_known: np.ndarray, scores: np.ndarray, tau: float) -> dict[str, float]:
    y_true = _as_array(y_true).astype(int)
    y_pred_known = _as_array(y_pred_known).astype(int)
    scores = _as_array(scores).astype(float)
    y_pred = apply_threshold(y_pred_known, scores, tau)
    acc_k, acc_u, acc = acc_components(y_true, y_pred)
    y_is_known = y_true != UNKNOWN_ID
    return {
        "tau": float(tau),
        "ACC_k": acc_k,
        "ACC_u": acc_u,
        "ACC": acc,
        "AUROC": binary_auroc(y_is_known, scores),
        "FPR@95TPR": fpr_at_tpr(y_is_known, scores, 0.95),
        "OSCR": oscr(y_true, y_pred_known, scores),
    }


def prediction_metrics(
    y_true: np.ndarray,
    y_pred_known: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    tau: float = float("nan"),
) -> dict[str, float]:
    y_true = _as_array(y_true).astype(int)
    y_pred_known = _as_array(y_pred_known).astype(int)
    scores = _as_array(scores).astype(float)
    y_pred = _as_array(y_pred).astype(int)
    acc_k, acc_u, acc = acc_components(y_true, y_pred)
    y_is_known = y_true != UNKNOWN_ID
    return {
        "tau": float(tau),
        "ACC_k": acc_k,
        "ACC_u": acc_u,
        "ACC": acc,
        "AUROC": binary_auroc(y_is_known, scores),
        "FPR@95TPR": fpr_at_tpr(y_is_known, scores, 0.95),
        "OSCR": oscr(y_true, y_pred_known, scores),
    }


def unknown_subset_acc_report(
    df: pd.DataFrame,
    subset_size: int = 3,
    unknown_id: int = UNKNOWN_ID,
) -> dict[str, float]:
    """Compute ACC_u and ACC over all unknown-label subsets.

    This follows the common bird-vocalization OSR protocol that samples every
    combination of k unknown species, computes unknown rejection accuracy for
    each subset, then reports mean +/- std across combinations. ACC_k is fixed
    for a given test set and ACC = 0.5 * ACC_k + 0.5 * ACC_u for each subset.
    """
    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")
    if "y_true" not in df.columns or "y_pred" not in df.columns or "true_label" not in df.columns:
        raise ValueError("unknown_subset_acc_report requires y_true, y_pred, and true_label columns.")

    known = df[df["y_true"].astype(int) != int(unknown_id)]
    unknown = df[df["y_true"].astype(int) == int(unknown_id)]
    acc_k = float((known["y_pred"].astype(int) == known["y_true"].astype(int)).mean()) if not known.empty else float("nan")
    labels = sorted(str(label) for label in unknown["true_label"].dropna().unique())

    if not labels:
        return {
            "unknown_subset_size": int(subset_size),
            "unknown_subset_count": 0,
            "ACC_u_subset_mean": float("nan"),
            "ACC_u_subset_std": float("nan"),
            "ACC_subset_mean": float("nan"),
            "ACC_subset_std": float("nan"),
        }

    effective_size = min(int(subset_size), len(labels))
    acc_u_values = []
    acc_values = []
    for combo in itertools.combinations(labels, effective_size):
        subset = unknown[unknown["true_label"].astype(str).isin(combo)]
        if subset.empty:
            continue
        acc_u = float((subset["y_pred"].astype(int) == int(unknown_id)).mean())
        if np.isnan(acc_k):
            acc = acc_u
        else:
            acc = 0.5 * acc_k + 0.5 * acc_u
        acc_u_values.append(acc_u)
        acc_values.append(acc)

    if not acc_u_values:
        return {
            "unknown_subset_size": int(effective_size),
            "unknown_subset_count": 0,
            "ACC_u_subset_mean": float("nan"),
            "ACC_u_subset_std": float("nan"),
            "ACC_subset_mean": float("nan"),
            "ACC_subset_std": float("nan"),
        }

    acc_u_array = np.asarray(acc_u_values, dtype=float)
    acc_array = np.asarray(acc_values, dtype=float)
    ddof = 1 if len(acc_u_array) > 1 else 0
    return {
        "unknown_subset_size": int(effective_size),
        "unknown_subset_count": int(len(acc_u_array)),
        "ACC_u_subset_mean": float(acc_u_array.mean()),
        "ACC_u_subset_std": float(acc_u_array.std(ddof=ddof)),
        "ACC_subset_mean": float(acc_array.mean()),
        "ACC_subset_std": float(acc_array.std(ddof=ddof)),
    }


def known_class_report(df: pd.DataFrame, label_names: list[str]) -> pd.DataFrame:
    known = df[df["y_true"] != UNKNOWN_ID].copy()
    rows = []
    for label_id, label in enumerate(label_names):
        true_mask = known["y_true"] == label_id
        pred_mask = known["y_pred"] == label_id
        tp = int((true_mask & pred_mask).sum())
        support = int(true_mask.sum())
        predicted = int(pred_mask.sum())
        recall = tp / support if support else float("nan")
        precision = tp / predicted if predicted else float("nan")
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        confusions = (
            known.loc[true_mask & (known["y_pred"] != label_id) & (known["y_pred"] != UNKNOWN_ID), "y_pred"]
            .value_counts()
            .head(2)
        )
        top2 = [
            f"{label_names[int(idx)]}:{int(count)}"
            for idx, count in confusions.items()
            if int(idx) >= 0 and int(idx) < len(label_names)
        ]
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "top2_confusions": "; ".join(top2),
            }
        )
    return pd.DataFrame(rows)


def unknown_class_report(df: pd.DataFrame, label_names: list[str]) -> pd.DataFrame:
    unknown = df[df["y_true"] == UNKNOWN_ID].copy()
    known = df[df["y_true"] != UNKNOWN_ID].copy()
    rows = []
    if unknown.empty:
        return pd.DataFrame()
    for label, group in unknown.groupby("true_label"):
        rejection_rate = float((group["y_pred"] == UNKNOWN_ID).mean())
        accepted = group[group["y_pred"] != UNKNOWN_ID]
        dist = accepted["y_pred"].value_counts(normalize=True).head(5)
        known_dist = [
            f"{label_names[int(idx)]}:{value:.3f}"
            for idx, value in dist.items()
            if int(idx) >= 0 and int(idx) < len(label_names)
        ]
        rows.append(
            {
                "unknown_label": label,
                "rejection_rate": rejection_rate,
                "false_acceptance_rate": 1.0 - rejection_rate,
                "support": int(len(group)),
                "misclassification_to_known": "; ".join(known_dist),
                "per_unknown_auroc": binary_auroc(
                    np.r_[np.ones(len(known), dtype=bool), np.zeros(len(group), dtype=bool)],
                    np.r_[known["score"].to_numpy(), group["score"].to_numpy()],
                ) if not known.empty else float("nan"),
                "median_mls": float(group["score"].median()),
                "mean_mls": float(group["score"].mean()),
            }
        )
        if "min_reconstruction_error" in group.columns:
            rows[-1]["median_min_reconstruction_error"] = float(group["min_reconstruction_error"].median())
            rows[-1]["mean_min_reconstruction_error"] = float(group["min_reconstruction_error"].mean())
        elif "reconstruction_error" in group.columns:
            rows[-1]["median_min_reconstruction_error"] = float(group["reconstruction_error"].median())
            rows[-1]["mean_min_reconstruction_error"] = float(group["reconstruction_error"].mean())
    return pd.DataFrame(rows)


def known_confusion_matrix(df: pd.DataFrame, label_names: list[str]) -> pd.DataFrame:
    known = df[(df["y_true"] != UNKNOWN_ID) & (df["y_pred"] != UNKNOWN_ID)]
    mat = np.zeros((len(label_names), len(label_names)), dtype=int)
    for row in known.itertuples():
        mat[int(row.y_true), int(row.y_pred)] += 1
    return pd.DataFrame(mat, index=label_names, columns=label_names)


def unknown_to_known_matrix(df: pd.DataFrame, label_names: list[str]) -> pd.DataFrame:
    unknown = df[(df["y_true"] == UNKNOWN_ID) & (df["y_pred"] != UNKNOWN_ID)]
    labels = sorted(unknown["true_label"].unique())
    mat = np.zeros((len(labels), len(label_names)), dtype=float)
    for i, label in enumerate(labels):
        group = unknown[unknown["true_label"] == label]
        denom = max(1, len(df[(df["y_true"] == UNKNOWN_ID) & (df["true_label"] == label)]))
        for pred, count in group["y_pred"].value_counts().items():
            if 0 <= int(pred) < len(label_names):
                mat[i, int(pred)] = count / denom
    return pd.DataFrame(mat, index=labels, columns=label_names)


def top_confusion_pairs(df: pd.DataFrame, label_names: list[str], top_n: int = 20) -> pd.DataFrame:
    rows = []
    known = df[(df["y_true"] != UNKNOWN_ID) & (df["y_pred"] != UNKNOWN_ID)]
    known = known[known["y_true"] != known["y_pred"]]
    for (src, dst), count in known.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False).head(top_n).items():
        rows.append({"source": label_names[int(src)], "target": label_names[int(dst)], "count": int(count), "type": "known-known"})
    unknown = df[(df["y_true"] == UNKNOWN_ID) & (df["y_pred"] != UNKNOWN_ID)]
    for (src, dst), count in unknown.groupby(["true_label", "y_pred"]).size().sort_values(ascending=False).head(top_n).items():
        rows.append({"source": str(src), "target": label_names[int(dst)], "count": int(count), "type": "unknown-known"})
    return pd.DataFrame(rows)


def reconstruction_report(df: pd.DataFrame) -> pd.DataFrame:
    required = {"reconstruction_error", "nearest_wrong_reconstruction_error", "true_label"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    rows = []
    for label, group in df.groupby("true_label"):
        rows.append(
            {
                "label": label,
                "correct_ae_error_mean": float(group["reconstruction_error"].mean()),
                "correct_ae_error_std": float(group["reconstruction_error"].std()),
                "nearest_wrong_ae_error_mean": float(group["nearest_wrong_reconstruction_error"].mean()),
                "nearest_wrong_ae_error_std": float(group["nearest_wrong_reconstruction_error"].std()),
            }
        )
    return pd.DataFrame(rows)
