"""Load the BEA 2025 MRBench files and flatten them to one row per tutor response.

The development set has 300 conversations and 2,476 annotated tutor responses; the
test set has 191 conversations and 1,547 responses whose labels are hidden. Rows keep
the file order (conversations in order, tutors in dictionary order), which is the
order the original notebook used, so its train/validation split can be replayed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FILES = {"dev": "mrbench_v3_devset.json", "test": "mrbench_v3_testset.json"}

# Class index order used everywhere in this repository.
LABELS = ["Yes", "To some extent", "No"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

TARGET = "Mistake_Identification"
DIMENSIONS = ["Mistake_Identification", "Mistake_Location", "Providing_Guidance", "Actionability"]


def normalize_history(text: str) -> str:
    """Whitespace- and case-insensitive key for a conversation history."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip().lower()


def load_raw(split: str) -> list[dict]:
    path = DATA_DIR / FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python scripts/download_data.py` first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten(conversations: list[dict], labelled: bool) -> list[dict]:
    rows = []
    for conv in conversations:
        for tutor, item in conv["tutor_responses"].items():
            row = {
                "conversation_id": conv["conversation_id"],
                "tutor": tutor,
                "history": conv["conversation_history"],
                "response": item["response"],
                "history_key": normalize_history(conv["conversation_history"]),
            }
            if labelled:
                row["labels"] = {dim: LABEL2ID[item["annotation"][dim]] for dim in DIMENSIONS}
                row["label"] = row["labels"][TARGET]
            rows.append(row)
    return rows


def load_rows(split: str) -> list[dict]:
    """Rows for 'dev' (labelled) or 'test' (unlabelled), with a conversation group id.

    Groups are defined by the normalised history text, not by conversation_id: the
    development set has 13 pairs of conversations (ids ending _1/_2) that share the
    same history under different ids, so grouping by id would still leak.
    """
    rows = flatten(load_raw(split), labelled=(split == "dev"))
    keys: dict[str, int] = {}
    for row in rows:
        row["group"] = keys.setdefault(row["history_key"], len(keys))
    return rows
