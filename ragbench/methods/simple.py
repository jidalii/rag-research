"""Baselines that need no index."""
from __future__ import annotations

import random

from .base import Method


class LLMOnly(Method):
    """Closed-book: question only, no documents, no tools."""
    name = "llm_only"

    def answer(self, ex, index):
        pred, raw = self.generate_answer(ex.question, None)
        return {"prediction": pred, "raw": raw, "retrieved_pids": None}


class FullContext(Method):
    """All of the question's own passages in the prompt (MuSiQue: the 20 given
    paragraphs, gold + distractors). Upper-bound-ish reference for retrieval."""
    name = "full_context"

    def answer(self, ex, index):
        if not ex.passages:
            raise ValueError("full_context needs per-question passages (MuSiQue).")
        passages = list(ex.passages)
        if self.params.get("shuffle", True):
            random.Random(ex.qid).shuffle(passages)
        ctx = "\n\n".join(f"[{p.title}] {p.text}" for p in passages)
        pred, raw = self.generate_answer(ex.question, ctx)
        return {"prediction": pred, "raw": raw, "retrieved_pids": [p.pid for p in passages]}
