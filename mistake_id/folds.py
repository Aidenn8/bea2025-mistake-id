"""Train/validation splits.

Three ways of holding out data are compared in the report:

* ``notebook``: the single 80/20 split the original submission used,
  ``datasets.Dataset.train_test_split(test_size=0.2, seed=42)`` over individual
  responses. Replayed exactly (the test indices are the first ceil(0.2 n) entries of
  ``np.random.default_rng(42).permutation(n)``; checked against datasets 3.5.0).
* ``leaky``: 5-fold cross-validation over individual responses. Responses to the same
  conversation land on both sides of the split.
* ``grouped``: 5-fold cross-validation where every response to a conversation stays in
  the same fold (stratified on the label so each fold has a similar class mix).
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.model_selection import KFold, StratifiedGroupKFold

N_FOLDS = 5


def notebook_split(n: int, seed: int = 42, test_size: float = 0.2) -> np.ndarray:
    """Fold array with 0 for validation rows and -1 for training rows."""
    n_test = math.ceil(test_size * n)
    perm = np.random.default_rng(seed).permutation(n)
    fold = np.full(n, -1)
    fold[perm[:n_test]] = 0
    return fold


def leaky_folds(n: int, seed: int = 0, k: int = N_FOLDS) -> np.ndarray:
    fold = np.empty(n, dtype=int)
    for i, (_, val) in enumerate(KFold(k, shuffle=True, random_state=seed).split(np.zeros(n))):
        fold[val] = i
    return fold


def grouped_folds(labels, groups, seed: int = 0, k: int = N_FOLDS) -> np.ndarray:
    labels, groups = np.asarray(labels), np.asarray(groups)
    fold = np.empty(len(labels), dtype=int)
    splitter = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
    for i, (_, val) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        fold[val] = i
    return fold


def make_folds(scheme: str, rows: list[dict], seed: int = 0) -> np.ndarray:
    if scheme == "notebook":
        return notebook_split(len(rows))
    if scheme == "leaky":
        return leaky_folds(len(rows), seed)
    if scheme == "grouped":
        return grouped_folds([r["label"] for r in rows], [r["group"] for r in rows], seed)
    raise ValueError(f"unknown split scheme {scheme!r}")


def fold_ids(fold: np.ndarray) -> list[int]:
    return sorted(int(f) for f in np.unique(fold) if f >= 0)
