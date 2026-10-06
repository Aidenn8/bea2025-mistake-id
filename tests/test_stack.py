"""Checks on the conversation-aware second stage."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mistake_id.stack import crossfit_stack, features  # noqa: E402


def test_feature_layout_and_sibling_mean():
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.6, 0.2, 0.2], [0.9, 0.05, 0.05]])
    groups = np.array([0, 0, 0, 1])
    X = features(probs, groups)
    assert X.shape == (4, 10)
    assert np.allclose(X[0, 3:6], (probs[1] + probs[2]) / 2)        # mean of the other two replies
    assert np.allclose(X[3, 3:6], probs[3])                          # a lone reply is its own sibling mean
    assert X[0, 9] == 1.0 and X[1, 9] == 0.0                         # rank on P(Yes) within the conversation


def test_crossfit_outputs_probabilities_and_uses_siblings():
    rng = np.random.default_rng(0)
    n_conv, k = 300, 8
    groups = np.repeat(np.arange(n_conv), k)
    # In "hard" conversations every reply looks a little less like Yes, and the true label is
    # No exactly when a reply is the least Yes-like in its conversation: only a model that
    # compares siblings can see that.
    base = rng.normal(0, 1, (n_conv * k, 3))
    base[:, 0] += 2.0 - 1.0 * np.repeat(rng.random(n_conv) > 0.5, k)
    p = np.exp(base)
    p /= p.sum(1, keepdims=True)
    y = np.zeros(len(p), dtype=int)
    for g in range(n_conv):
        idx = np.flatnonzero(groups == g)
        y[idx[p[idx, 0].argmin()]] = 2
    fold = groups % 5
    out = crossfit_stack(p, y, groups, fold)
    assert np.allclose(out.sum(1), 1.0) and (out >= 0).all()
    acc_stack = (out.argmax(1) == y).mean()
    acc_alone = (p.argmax(1) == y).mean()
    assert acc_stack > acc_alone + 0.05
