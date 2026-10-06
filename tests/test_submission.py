"""The submission file mirrors the test file and carries one valid label per response."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_submission import build, check  # noqa: E402

from mistake_id.data import LABELS, load_rows  # noqa: E402


def test_round_trip():
    labels = np.random.default_rng(0).integers(0, 3, 1547)
    sub = build(labels)
    check(sub)
    rows = load_rows("test")
    flat = [(c["conversation_id"], t, r["annotation"]["Mistake_Identification"])
            for c in sub for t, r in c["tutor_responses"].items()]
    assert [(r["conversation_id"], r["tutor"]) for r in rows] == [(a, b) for a, b, _ in flat]
    assert [LABELS[i] for i in labels] == [lab for _, _, lab in flat]
