"""Write a Codabench submission (predictions.json inside predictions.json.zip).

    python scripts/make_submission.py OUT_DIR --labels results/test_predictions/NAME.npy

``--labels`` is an array of 1,547 class indices (0 Yes, 1 To some extent, 2 No) in the
order of ``load_rows("test")``. The JSON mirrors the test file, with an annotation added
to every tutor response, as the task's submission format requires. The file is checked
against the original before it is zipped.
"""
import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mistake_id.data import LABELS, load_raw, load_rows  # noqa: E402


def build(labels: np.ndarray) -> list[dict]:
    raw = load_raw("test")
    rows = load_rows("test")
    assert len(labels) == len(rows) == 1547
    it = iter(labels.tolist())
    out = []
    for conv in raw:
        responses = {}
        for tutor, item in conv["tutor_responses"].items():
            responses[tutor] = {"response": item["response"],
                                "annotation": {"Mistake_Identification": LABELS[next(it)]}}
        out.append({"conversation_id": conv["conversation_id"],
                    "conversation_history": conv["conversation_history"],
                    "tutor_responses": responses})
    assert next(it, None) is None
    return out


def check(sub: list[dict]):
    raw = load_raw("test")
    assert len(sub) == len(raw) == 191
    n = 0
    for a, b in zip(sub, raw):
        assert a["conversation_id"] == b["conversation_id"]
        assert a["conversation_history"] == b["conversation_history"]
        assert list(a["tutor_responses"]) == list(b["tutor_responses"])
        for t, r in a["tutor_responses"].items():
            assert r["response"] == b["tutor_responses"][t]["response"]
            assert r["annotation"]["Mistake_Identification"] in LABELS
            n += 1
    assert n == 1547


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()
    labels = np.load(args.labels).astype(int)
    sub = build(labels)
    check(sub)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    js = out / "predictions.json"
    js.write_text(json.dumps(sub, indent=1, ensure_ascii=False))
    with zipfile.ZipFile(out / "predictions.json.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(js, "predictions.json")
    counts = Counter(LABELS[i] for i in labels.tolist())
    print(f"wrote {out / 'predictions.json.zip'}: " + ", ".join(f"{k} {counts[k]}" for k in LABELS))


if __name__ == "__main__":
    main()
