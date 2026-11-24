"""Answer and retrieval metrics (SQuAD/MuSiQue-style normalization)."""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _prf(pred: str, gold: str) -> tuple[float, float, float]:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        v = float(p == g)
        return v, v, v
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0, 0.0, 0.0
    prec, rec = common / len(p), common / len(g)
    return prec, rec, 2 * prec * rec / (prec + rec)


def answer_metrics(pred: str, golds: list[str]) -> dict:
    """Max over gold aliases. `answer_recall` = fraction of gold tokens present in
    the prediction (robust to verbose answers); `contains` = normalized gold
    string appears verbatim in the prediction."""
    golds = [g for g in golds if g] or [""]
    npred = normalize_answer(pred)
    return {
        "em": max(float(npred == normalize_answer(g)) for g in golds),
        "f1": max(_prf(pred, g)[2] for g in golds),
        "answer_recall": max(_prf(pred, g)[1] for g in golds),
        "contains": max(float(bool(normalize_answer(g)) and normalize_answer(g) in npred)
                        for g in golds),
    }


def retrieval_recall(retrieved: list[str] | None, supporting: list[str]) -> float | None:
    if retrieved is None or not supporting:
        return None
    r = set(retrieved)
    return sum(1 for s in supporting if s in r) / len(supporting)
