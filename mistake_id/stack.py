"""Conversation-aware second stage (exploratory).

Every reply arrives together with the other tutors' replies to the same dialogue. The
first-stage classifier judges each reply on its own; this stage lets the decision depend
on how a reply compares with its siblings. A multinomial logistic regression takes, for
each reply, the first-stage log-probabilities, the mean probabilities of the other replies
to the same conversation, the difference between the two, and the reply's rank within its
conversation on P(Yes). It is cross-fitted: the model that predicts fold k is trained on the
out-of-fold features of the other folds, so no reply is ever predicted by a model that saw
its label.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def features(probs: np.ndarray, groups: np.ndarray) -> np.ndarray:
    logp = np.log(np.clip(probs, 1e-6, 1.0))
    sib_mean = np.zeros_like(probs)
    rank = np.zeros(len(probs))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) > 1:
            total = probs[idx].sum(0)
            sib_mean[idx] = (total - probs[idx]) / (len(idx) - 1)
            order = probs[idx, 0].argsort().argsort()
            rank[idx] = order / (len(idx) - 1)
        else:
            sib_mean[idx] = probs[idx]
            rank[idx] = 0.5
    return np.column_stack([logp, sib_mean, probs - sib_mean, rank])


def crossfit_stack(probs: np.ndarray, y: np.ndarray, groups: np.ndarray, fold: np.ndarray, C: float = 1.0):
    """Out-of-fold stacked probabilities, shape (n, 3)."""
    X = features(probs, groups)
    out = np.zeros_like(probs)
    for k in sorted(set(fold.tolist())):
        tr, te = fold != k, fold == k
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(X[tr], y[tr])
        out[np.ix_(np.flatnonzero(te), clf.classes_)] = clf.predict_proba(X[te])
    return out


def fit_stack(probs: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float = 1.0) -> LogisticRegression:
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(features(probs, groups), y)
    return clf
