"""Cheap reference points, computed on the same folds as the neural runs.

    python scripts/baselines.py

* majority class ("Yes" for everything);
* tutor identity: each tutor's most common label in the training folds (an upper bound
  on what tutor style alone explains; tutor names are hidden in the test set);
* TF-IDF + logistic regression on the response, on the history, and on both, under the
  leaky (response-level) and grouped (conversation-level) splits. A history-only model
  never sees the response it is judging, so whatever it scores above the majority class
  under the leaky split comes from recognising conversations it was trained on.

Writes results/baselines.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mistake_id.data import load_rows  # noqa: E402
from mistake_id.folds import make_folds  # noqa: E402
from mistake_id.metrics import bootstrap, scores  # noqa: E402


def tfidf_oof(rows, fold, fields):
    y = np.array([r["label"] for r in rows])
    pred = np.empty(len(rows), dtype=int)
    for k in sorted(set(fold.tolist())):
        tr, te = fold != k, fold == k
        mats_tr, mats_te = [], []
        for field in fields:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
            text = [r[field] for r in rows]
            mats_tr.append(vec.fit_transform([t for t, m in zip(text, tr) if m]))
            mats_te.append(vec.transform([t for t, m in zip(text, te) if m]))
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
        clf.fit(hstack(mats_tr).tocsr(), y[tr])
        pred[te] = clf.predict(hstack(mats_te).tocsr())
    return pred


def main():
    rows = load_rows("dev")
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    folds = {s: make_folds(s, rows, 0) for s in ("grouped", "leaky")}
    out = {}

    out["majority"] = scores(y, np.zeros_like(y))

    fold = folds["grouped"]
    pred = np.empty(len(rows), dtype=int)
    for k in range(5):
        tr = fold != k
        top = {t: Counter(y[tr & (np.array([r["tutor"] for r in rows]) == t)]).most_common(1)[0][0]
               for t in {r["tutor"] for r in rows}}
        for i in np.flatnonzero(fold == k):
            pred[i] = top[rows[i]["tutor"]]
    out["tutor_identity"] = scores(y, pred)

    preds = {}
    for name, fields in (("response", ["response"]), ("history", ["history"]), ("both", ["history", "response"])):
        for split, f in folds.items():
            p = tfidf_oof(rows, f, fields)
            preds[f"tfidf_{name}_{split}"] = p
            out[f"tfidf_{name}_{split}"] = scores(y, p)
            print(f"tfidf {name:8s} {split:7s} strict F1 {out[f'tfidf_{name}_{split}']['strict_f1']:.4f}")

    ci = bootstrap(y, preds, groups, n_boot=1000)
    for k, v in ci.items():
        out[k]["strict_f1_ci"] = [v["lo"], v["hi"]]
    for k in ("majority", "tutor_identity"):
        print(f"{k:16s} strict F1 {out[k]['strict_f1']:.4f} acc {out[k]['strict_acc']:.4f}")
    (ROOT / "results" / "baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
