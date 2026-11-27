"""Baselines that need no retrieval index: closed book, full context, oracle."""
from __future__ import annotations

import random
import re

from .base import Method


def _fmt(passages) -> str:
    return "\n\n".join(f"[{p.title}] {p.text}" for p in passages)


class _CorpusLookup(Method):
    """Base for methods that look passages up by pid. MultiHop-RAG questions carry
    no passages of their own, so the "index" is just {pid: Passage} (no LLM calls)."""
    needs_index = True

    def index_params(self) -> dict:
        return {}

    def build_index(self, passages):
        return {p.pid: p for p in passages}

    @staticmethod
    def _lookup(ex, index):
        by_pid = {p.pid: p for p in ex.passages} or index
        return by_pid


class LLMOnly(Method):
    """Closed-book: question only, no documents, no tools."""
    name = "llm_only"

    def answer(self, ex, index):
        pred, raw = self.generate_answer(ex.question, None)
        return {"prediction": pred, "raw": raw, "retrieved_pids": None}


class FullContext(_CorpusLookup):
    """MuSiQue: all of the question's own paragraphs (gold + distractors), shuffled.
    MultiHop-RAG: the gold articles plus seeded random distractor articles, up to
    `max_context_tokens` (default 32k). Gold is always kept, even over the cap."""
    name = "full_context"

    def __init__(self, params, llm, embedder, qa_cfg):
        super().__init__({"max_context_tokens": 32000, "shuffle": True, **params},
                         llm, embedder, qa_cfg)
        import tiktoken
        self.tok = tiktoken.get_encoding("cl100k_base")

    def answer(self, ex, index):
        rng = random.Random(ex.qid)
        over_cap = False
        if ex.passages:
            passages = list(ex.passages)
        else:
            cap = self.params["max_context_tokens"]
            gold = [index[pid] for pid in ex.supporting_pids]
            used = sum(len(self.tok.encode(f"[{p.title}] {p.text}")) for p in gold)
            over_cap = used > cap
            gold_set = set(ex.supporting_pids)
            others = [p for pid, p in index.items() if pid not in gold_set]
            rng.shuffle(others)
            passages = list(gold)
            for p in others:
                n = len(self.tok.encode(f"[{p.title}] {p.text}"))
                if used + n <= cap:
                    passages.append(p)
                    used += n
                if cap - used < 200:
                    break
        if self.params["shuffle"]:
            rng.shuffle(passages)
        pred, raw = self.generate_answer(ex.question, _fmt(passages))
        return {"prediction": pred, "raw": raw,
                "retrieved_pids": [p.pid for p in passages], "gold_over_cap": over_cap}


class Oracle(_CorpusLookup):
    """Only the gold passages, shuffled: the reasoning ceiling for a reader (H2)."""
    name = "oracle"

    def answer(self, ex, index):
        by_pid = self._lookup(ex, index)
        passages = [by_pid[pid] for pid in ex.supporting_pids]
        random.Random(ex.qid).shuffle(passages)
        pred, raw = self.generate_answer(ex.question, _fmt(passages))
        return {"prediction": pred, "raw": raw,
                "retrieved_pids": [p.pid for p in passages]}


_REF = re.compile(r"#(\d+)")


def render_subquestion(text: str, answers: list[str]) -> str:
    """MuSiQue sub-questions reference earlier hops as #k and are often written as
    'subject >> relation'. Fill #k with hop k's gold answer and turn the arrow form
    into a plain question."""
    text = _REF.sub(lambda m: answers[int(m.group(1)) - 1]
                    if int(m.group(1)) <= len(answers) else m.group(0), text).strip()
    if ">>" in text:
        subject, relation = (s.strip() for s in text.split(">>", 1))
        return f"What is the {relation} of {subject}?"
    return text


class LLMOnlyDecomposed(Method):
    """H4: closed book, each MuSiQue sub-question asked on its own with earlier hops'
    gold answers filled in. `prediction` is the last hop's answer (graded against the
    final gold answer); every hop is in `subq` and graded by the judge step."""
    name = "llm_only_decomposed"

    def answer(self, ex, index):
        steps = ex.meta.get("decomposition") or []
        if not steps:
            raise ValueError("llm_only_decomposed needs MuSiQue question_decomposition")
        gold, subq = [s["answer"] for s in steps], []
        from ..metrics import answer_metrics
        for s in steps:
            q = render_subquestion(s["question"], gold)
            pred, raw = self.generate_answer(q, None)
            subq.append({"hop": s["hop"], "question": q, "gold": s["answer"],
                         "prediction": pred, "raw": raw,
                         "metrics": answer_metrics(pred, [s["answer"]])})
        return {"prediction": subq[-1]["prediction"], "raw": subq[-1]["raw"],
                "retrieved_pids": None, "subq": subq}
