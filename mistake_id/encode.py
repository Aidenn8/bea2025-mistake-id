"""Turning a (conversation history, tutor response) pair into model inputs.

``notebook``: the submitted system's format. History first, then the response, as one
string, cut on the right at 512 tokens. When a conversation is long the cut removes
the end of the input, which is exactly where the tutor response sits.

``pair``: the response is the first segment and is never cut (up to
``max_response`` tokens). The history is the second segment; if it does not fit, the
model keeps its first ``head_tokens`` tokens (the problem statement) and as much of
its end (the student's latest turn, which contains the mistake) as fits, with an
ellipsis in between. Cuts are made on character offsets of the original text, so
the result is an exact substring of the conversation.
"""
from __future__ import annotations

ELLIPSIS = " … "


def notebook_text(row: dict) -> str:
    return f"Conversation history: {row['history']}\nTutor: {row['response']}"


def _special_tokens_in_pair(tok) -> int:
    pair = len(tok("a", "b")["input_ids"])
    return pair - len(tok("a", add_special_tokens=False)["input_ids"]) - len(tok("b", add_special_tokens=False)["input_ids"])


def _keep_ends(text: str, offsets, head: int, tail: int) -> str:
    """Keep the first ``head`` and last ``tail`` tokens of ``text`` (by character offsets)."""
    n = len(offsets)
    if head + tail >= n:
        return text
    head_end = offsets[head - 1][1] if head > 0 else 0
    tail_start = offsets[n - tail][0] if tail > 0 else len(text)
    return text[:head_end].rstrip() + ELLIPSIS + text[tail_start:].lstrip()


class Encoder:
    def __init__(self, tok, fmt: str, max_len: int = 512, head_tokens: int = 96, max_response: int = 256,
                 margin: int = 8):
        if fmt not in ("notebook", "pair"):
            raise ValueError(f"unknown input format {fmt!r}")
        self.tok, self.fmt, self.max_len = tok, fmt, max_len
        self.head_tokens, self.max_response, self.margin = head_tokens, max_response, margin
        self.n_special = _special_tokens_in_pair(tok) if fmt == "pair" else 0
        self.stats = {"history_cut": 0, "response_cut": 0, "guard_trimmed": 0, "n": 0}

    def __call__(self, row: dict) -> dict:
        self.stats["n"] += 1
        if self.fmt == "notebook":
            return dict(self.tok(notebook_text(row), truncation=True, max_length=self.max_len))
        return self._pair(row)

    def _pair(self, row: dict) -> dict:
        tok = self.tok
        response = row["response"]
        r = tok(response, add_special_tokens=False, return_offsets_mapping=True)
        if len(r["input_ids"]) > self.max_response:
            response = _keep_ends(response, r["offset_mapping"], self.max_response - 4, 0)
            self.stats["response_cut"] += 1
        n_resp = len(tok(response, add_special_tokens=False)["input_ids"])

        history = row["history"]
        budget = self.max_len - self.n_special - n_resp - self.margin
        h = tok(history, add_special_tokens=False, return_offsets_mapping=True)
        if len(h["input_ids"]) > budget:
            head = min(self.head_tokens, budget // 3)
            history = _keep_ends(history, h["offset_mapping"], head, budget - head)
            self.stats["history_cut"] += 1

        # Re-tokenising the cut text can shift a token or two at the seams; trim any
        # overflow from the start of the history (never from the response).
        side = tok.truncation_side
        tok.truncation_side = "left"
        try:
            enc = tok(response, history, truncation="only_second", max_length=self.max_len)
        finally:
            tok.truncation_side = side
        full = len(tok(response, history)["input_ids"])
        if full > self.max_len:
            self.stats["guard_trimmed"] += 1
        return dict(enc)
