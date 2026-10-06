"""The shared task's metrics, plus cluster-bootstrap confidence intervals.

Strict: macro-F1 and accuracy over the three classes (Yes, To some extent, No).
Lenient: "Yes" and "To some extent" merged, macro-F1 and accuracy over two classes.
Strict macro-F1 is the official ranking metric.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import LABELS

LENIENT = np.array([0, 0, 1])  # Yes, To some extent -> 0; No -> 1


def scores(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    lt, lp = LENIENT[y_true], LENIENT[y_pred]
    per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "strict_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "strict_acc": float(accuracy_score(y_true, y_pred)),
        "lenient_f1": float(f1_score(lt, lp, labels=[0, 1], average="macro", zero_division=0)),
        "lenient_acc": float(accuracy_score(lt, lp)),
        **{f"f1_{name}": float(v) for name, v in zip(LABELS, per_class)},
    }


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])


def fast_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """The four headline metrics from a confusion matrix (same values as ``scores``)."""
    if metric.startswith("lenient"):
        y_true, y_pred, k = LENIENT[y_true], LENIENT[y_pred], 2
    else:
        k = 3
    cm = np.bincount(k * y_true + y_pred, minlength=k * k).reshape(k, k)
    if metric.endswith("acc"):
        return float(np.trace(cm) / cm.sum())
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def _group_index(groups):
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    return members


def bootstrap(y_true, preds: dict, groups, metric="strict_f1", n_boot=2000, seed=0) -> dict:
    """Cluster bootstrap over conversations.

    ``preds`` maps a system name to its predictions. Returns, for each system, the
    point estimate and a 95% interval, resampling whole conversations so responses to
    the same conversation are never treated as independent.
    """
    y_true = np.asarray(y_true)
    preds = {name: np.asarray(p) for name, p in preds.items()}
    members = _group_index(groups)
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in preds}
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[g] for g in pick])
        for name, p in preds.items():
            draws[name].append(fast_metric(y_true[idx], p[idx], metric))
    out = {}
    for name, p in preds.items():
        d = np.array(draws[name])
        out[name] = {"value": scores(y_true, p)[metric], "lo": float(np.percentile(d, 2.5)),
                     "hi": float(np.percentile(d, 97.5)), "draws": d}
    return out


def paired_difference(y_true, pred_a, pred_b, groups, metric="strict_f1", n_boot=2000, seed=0) -> dict:
    """Bootstrap distribution of metric(a) - metric(b) on the same resampled conversations."""
    res = bootstrap(y_true, {"a": pred_a, "b": pred_b}, groups, metric, n_boot, seed)
    diff = res["a"]["draws"] - res["b"]["draws"]
    return {"diff": res["a"]["value"] - res["b"]["value"], "lo": float(np.percentile(diff, 2.5)),
            "hi": float(np.percentile(diff, 97.5)), "p_le_0": float(np.mean(diff <= 0))}
