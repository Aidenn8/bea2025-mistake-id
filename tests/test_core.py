"""Checks on data loading, splits, metrics and input encoding."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mistake_id.data import LABELS, load_rows  # noqa: E402
from mistake_id.encode import Encoder, notebook_text  # noqa: E402
from mistake_id.folds import grouped_folds, leaky_folds, notebook_split  # noqa: E402
from mistake_id.metrics import scores  # noqa: E402


@pytest.fixture(scope="module")
def dev():
    return load_rows("dev")


@pytest.fixture(scope="module")
def test():
    return load_rows("test")


def test_sizes(dev, test):
    assert len(dev) == 2476 and len(test) == 1547
    assert len({r["group"] for r in dev}) == 287      # 300 ids, 13 pairs share a history
    assert len({r["conversation_id"] for r in dev}) == 300
    assert len({r["group"] for r in test}) == 191
    assert Counter(r["label"] for r in dev) == {0: 1932, 2: 370, 1: 174}


def test_notebook_split_matches_datasets():
    datasets = pytest.importorskip("datasets")
    split = datasets.Dataset.from_dict({"i": list(range(2476))}).train_test_split(test_size=0.2, seed=42)
    fold = notebook_split(2476)
    assert sorted(split["test"]["i"]) == sorted(np.flatnonzero(fold == 0).tolist())
    assert len(split["train"]) == int((fold == -1).sum()) == 1980


def test_grouped_folds_never_split_a_conversation(dev):
    fold = grouped_folds([r["label"] for r in dev], [r["group"] for r in dev], seed=0)
    by_group = {}
    for r, f in zip(dev, fold):
        by_group.setdefault(r["group"], set()).add(int(f))
    assert all(len(s) == 1 for s in by_group.values())
    for f in range(5):                      # every fold has every class
        assert set(np.asarray([r["label"] for r in dev])[fold == f]) == {0, 1, 2}


def test_leaky_folds_do_split_conversations(dev):
    fold = leaky_folds(len(dev), seed=0)
    spans = Counter(len({int(f) for r, f in zip(dev, fold) if r["group"] == g}) for g in range(287))
    assert spans[1] < 10                    # almost every conversation is spread over folds


def test_metrics_hand_example():
    y = [0, 0, 1, 2, 2, 0]
    p = [0, 1, 1, 2, 0, 0]
    s = scores(y, p)
    # per-class F1: Yes 2/3 (P 2/3, R 2/3); Some 2/3 (P 1/2, R 1); No 2/3 (P 1, R 1/2)
    assert s["strict_f1"] == pytest.approx(2 / 3)
    assert s["strict_acc"] == pytest.approx(4 / 6)
    # lenient: y = [0,0,0,1,1,0], p = [0,0,0,1,0,0]; class 0 F1 = 8/9, class 1 F1 = 2/3
    assert s["lenient_acc"] == pytest.approx(5 / 6)
    assert s["lenient_f1"] == pytest.approx((8 / 9 + 2 / 3) / 2)
    assert LABELS == ["Yes", "To some extent", "No"]


@pytest.mark.parametrize("model", ["bert-base-uncased", "microsoft/deberta-v3-base", "roberta-base"])
def test_pair_encoder_keeps_response_and_latest_turn(dev, test, model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    enc = Encoder(tok, "pair", max_len=512, head_tokens=96)
    rows = dev + test
    tail_mismatch = 0
    for row in rows:
        ids = enc(row)["input_ids"]
        assert len(ids) <= 512
        # The response is the first segment: its tokens follow the start token unchanged.
        r_ids = tok(row["response"], add_special_tokens=False)["input_ids"]
        if len(r_ids) <= 252:
            assert ids[1:1 + len(r_ids)] == r_ids
        # The end of the history (the student's latest turn) sits just before the final
        # special token. Allow rare one-token differences where a cut changes how the
        # tokenizer merges the neighbouring characters.
        h_ids = tok(row["history"], add_special_tokens=False)["input_ids"]
        tail_mismatch += ids[-21:-1] != h_ids[-20:]
    assert tail_mismatch <= 0.005 * len(rows)
    assert enc.stats["guard_trimmed"] <= 5


def test_notebook_encoder_matches_original(test):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = Encoder(tok, "notebook")
    cut = 0
    for row in test:
        ids = enc(row)["input_ids"]
        assert ids == tok(notebook_text(row), truncation=True, max_length=512)["input_ids"]
        cut += len(tok(notebook_text(row))["input_ids"]) > 512
    assert cut == 186                         # replies partly or fully cut in the submitted system
