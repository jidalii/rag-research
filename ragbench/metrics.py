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


def all_gold_retrieved(retrieved: list[str] | None, supporting: list[str]) -> float | None:
    """1.0 when every gold passage is in the context (H2's retrieval/reasoning split)."""
    rec = retrieval_recall(retrieved, supporting)
    return None if rec is None else float(rec == 1.0)


def recall_by_hop(retrieved: list[str] | None, gold_hop: dict[str, int]) -> dict | None:
    """{hop position: 1.0/0.0} for each gold paragraph (MuSiQue, H3)."""
    if retrieved is None or not gold_hop:
        return None
    r = set(retrieved)
    return {str(h): float(pid in r) for pid, h in gold_hop.items()}


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def fact_recall(context: str | None, facts: list[dict]) -> float | None:
    """Share of MultiHop-RAG evidence facts whose text appears in the context
    (case/punctuation-insensitive substring). Stricter than article-level recall:
    a retrieved chunk of the right article may not hold the fact."""
    if context is None or not facts:
        return None
    ctx = _squash(context)
    return sum(1 for f in facts if _squash(f["fact"]) in ctx) / len(facts)
