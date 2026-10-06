"""Checks on the decision-rule tuning and ensembling helpers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mistake_id.metrics import scores  # noqa: E402
from mistake_id.results import _macro_f1_grid, apply_offsets, average, crossfit_offsets, tune_offsets  # noqa: E402


def _synthetic(n=3000, seed=0):
    """A calibrated-looking classifier whose argmax under-predicts the two rare classes."""
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.78, 0.07, 0.15])
    logits = rng.normal(0, 1.0, size=(n, 3))
    logits[np.arange(n), y] += 1.6
    logits[:, 0] += 2.0                      # strong bias towards the majority class
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True), y


def test_grid_matches_sklearn():
    p, y = _synthetic()
    logp = np.log(p)
    grid = np.array([-0.5, 0.0, 0.7])
    f1 = _macro_f1_grid(logp, y, grid)
    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            pred = (logp + np.array([0, a, b])).argmax(-1)
            assert abs(f1[i, j] - scores(y, pred)["strict_f1"]) < 1e-12


def test_offsets_raise_macro_f1_and_point_the_right_way():
    p, y = _synthetic()
    base = scores(y, p.argmax(-1))["strict_f1"]
    off = tune_offsets(p, y)
    tuned = scores(y, apply_offsets(p, off))["strict_f1"]
    assert tuned > base + 0.02
    assert off[0] == 0 and off[1] > 0 and off[2] > 0     # boosts both minority classes


def test_crossfit_never_uses_the_scored_fold():
    p, y = _synthetic()
    fold = np.arange(len(y)) % 5
    pred, chosen = crossfit_offsets(p, y, fold)
    assert len(chosen) == 5
    # Corrupting fold 0's labels must not change the offsets used to predict fold 0.
    y2 = y.copy()
    y2[fold == 0] = (y2[fold == 0] + 1) % 3
    pred2, chosen2 = crossfit_offsets(p, y2, fold)
    assert chosen2[0] == chosen[0]
    assert np.array_equal(pred2[fold == 0], pred[fold == 0])


def test_average_is_mean():
    a, b = np.full((2, 3), 0.2), np.full((2, 3), 0.4)
    assert np.allclose(average([a, b]), 0.3)
    assert np.allclose(average([a, b], [3, 1]), 0.25)
