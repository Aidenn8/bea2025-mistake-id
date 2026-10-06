"""Collecting saved runs, building ensembles and tuning the decision rule.

Every run directory (results/runs/NAME/SPLIT-sSEED/) holds one npz per held-out fold
with the probabilities that fold's model gave its held-out responses ("out-of-fold")
and the whole test set. Pooling the out-of-fold probabilities gives one honest
prediction for every development response.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import scores

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"


def load_run(name: str, split: str = "grouped", seed: int = 0) -> dict:
    d = RUNS / name / f"{split}-s{seed}"
    fold = np.load(d / "fold_assignment.npy")
    n = len(fold)
    oof = np.full((n, 3), np.nan, dtype=np.float64)
    test, metas = [], []
    for path in sorted(d.glob("fold[0-9].npz")):
        z = np.load(path)
        if len(z["val_idx"]):
            oof[z["val_idx"]] = z["val_probs"][:, 0]
        if z["test_probs"].size:
            test.append(z["test_probs"][:, 0])
        metas.append(json.loads(path.with_suffix(".json").read_text()))
    expected = sorted(int(f) for f in np.unique(fold) if f >= 0)
    done = sorted(m["fold"] for m in metas)
    return {"name": name, "split": split, "seed": seed, "fold": fold, "oof": oof,
            "test_folds": np.stack(test) if test else None,
            "test": np.mean(test, axis=0) if test else None,
            "meta": metas, "complete": done == expected or (split == "full" and done == [0])}


def evaluate(run: dict, y) -> dict:
    """Pooled out-of-fold scores plus the per-fold spread."""
    y = np.asarray(y)
    mask = ~np.isnan(run["oof"][:, 0])
    pred = run["oof"].argmax(-1)
    out = {"pooled": scores(y[mask], pred[mask]), "n": int(mask.sum())}
    per_fold = [scores(y[run["fold"] == k], pred[run["fold"] == k])
                for k in sorted(set(run["fold"][mask].tolist())) if k >= 0]
    out["per_fold"] = per_fold
    for key in ("strict_f1", "strict_acc", "lenient_f1", "lenient_acc"):
        vals = np.array([s[key] for s in per_fold])
        out[f"{key}_fold_mean"] = float(vals.mean())
        out[f"{key}_fold_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
    return out


def average(probs: list[np.ndarray], weights=None) -> np.ndarray:
    """Mean of probability arrays (an ensemble). NaN rows stay NaN."""
    w = np.ones(len(probs)) if weights is None else np.asarray(weights, dtype=float)
    return sum(wi * p for wi, p in zip(w, probs)) / w.sum()


# --- decision rule -----------------------------------------------------------------
# The classifier is trained for cross-entropy, not macro-F1. Shifting the log-probability
# of the two minority classes by constant offsets trades a little accuracy on "Yes"
# for recall on "To some extent" and "No". Offsets are chosen by grid search.

GRID = np.round(np.arange(-2.0, 3.0001, 0.05), 2)


def _macro_f1_grid(logp: np.ndarray, y: np.ndarray, grid=GRID) -> np.ndarray:
    """Strict macro-F1 for every (offset_some, offset_no) pair; shape (len(grid), len(grid))."""
    out = np.empty((len(grid), len(grid)))
    for i, a in enumerate(grid):
        shifted = logp + np.array([0.0, a, 0.0])
        for j, b in enumerate(grid):
            pred = (shifted + np.array([0.0, 0.0, b])).argmax(-1)
            cm = np.bincount(3 * y + pred, minlength=9).reshape(3, 3)
            tp = np.diag(cm)
            denom = cm.sum(0) + cm.sum(1)
            f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
            out[i, j] = f1.mean()
    return out


def tune_offsets(probs: np.ndarray, y, grid=GRID) -> np.ndarray:
    """Offsets (0, a, b) added to log-probabilities that maximise macro-F1 on (probs, y).

    Ties are broken towards the smallest total shift, so the rule stays close to
    plain argmax when the data do not ask for more.
    """
    y = np.asarray(y)
    logp = np.log(np.clip(probs, 1e-9, 1.0))
    f1 = _macro_f1_grid(logp, y, grid)
    best = f1.max()
    cand = np.argwhere(f1 >= best - 1e-12)
    shift = np.abs(grid[cand[:, 0]]) + np.abs(grid[cand[:, 1]])
    i, j = cand[shift.argmin()]
    return np.array([0.0, grid[i], grid[j]])


def apply_offsets(probs: np.ndarray, offsets) -> np.ndarray:
    return (np.log(np.clip(probs, 1e-9, 1.0)) + np.asarray(offsets)).argmax(-1)


def crossfit_offsets(probs: np.ndarray, y, fold: np.ndarray) -> tuple[np.ndarray, list]:
    """Honest estimate of the tuned rule: offsets for fold k are tuned on the other folds."""
    y = np.asarray(y)
    pred = np.empty(len(y), dtype=int)
    chosen = []
    for k in sorted(set(fold.tolist())):
        if k < 0:
            continue
        tr, te = fold != k, fold == k
        off = tune_offsets(probs[tr], y[tr])
        pred[te] = apply_offsets(probs[te], off)
        chosen.append(off.tolist())
    return pred, chosen
