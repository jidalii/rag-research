"""Method interface. To add a method (GraphRAG, vanilla RAG, ...):

  1. subclass Method, set `name`, implement `answer()` and (if it indexes)
     `build_index()` + `needs_index = True`;
  2. register it in ragbench/methods/__init__.py.

Conventions:
  - LLM calls made while indexing use phase="index"; at question time phase="query".
  - `answer()` returns {"prediction", "raw", "retrieved_pids", ...}. `retrieved_pids`
    are corpus Passage.pid values (None if the method does not retrieve).
  - Indexes must be picklable so the runner can cache them on disk.
"""
from __future__ import annotations

import threading
from typing import Any

from ..datasets import Example, Passage
from ..embeddings import Embedder
from ..llm import LLMClient
from ..prompts import extract_answer, qa_prompt


class Method:
    name = "base"
    needs_index = False
    uses_embedder = False

    def __init__(self, params: dict, llm: LLMClient, embedder: Embedder | None, qa_cfg: dict):
        self.params = params
        self.llm = llm
        self.embedder = embedder
        # a method entry may override the [qa] settings (e.g. an llm_only_cot arm)
        self.answer_format = params.pop("answer_format", qa_cfg.get("answer_format", "short"))
        self.max_answer_tokens = params.pop("max_answer_tokens",
                                            qa_cfg.get("max_answer_tokens", 256))
        self._tl = threading.local()

    def index_params(self) -> dict:
        """Parameters that affect the index (used in the cache key)."""
        return dict(self.params)

    def pop_contexts(self) -> list[str]:
        """Contexts passed to generate_answer by this thread since the last call."""
        out = getattr(self._tl, "contexts", [])
        self._tl.contexts = []
        return out

    def pop_truncated(self) -> int:
        """Answers by this thread that hit max_answer_tokens since the last call."""
        n = getattr(self._tl, "truncated", 0)
        self._tl.truncated = 0
        return n

    def build_index(self, passages: list[Passage]) -> Any:
        return None

    def answer(self, ex: Example, index: Any) -> dict:
        raise NotImplementedError

    # helper shared by all methods
    def generate_answer(self, question: str, context: str | None) -> tuple[str, str]:
        # the runner reads the context back (same thread) to measure it uniformly:
        # context tokens, gold facts present, ... It is not stored in predictions.
        self._tl.contexts = getattr(self._tl, "contexts", []) + [context or ""]
        system, prompt = qa_prompt(question, context, self.answer_format)
        raw, truncated = self.llm.complete(prompt, system=system,
                                           max_tokens=self.max_answer_tokens, phase="query")
        self._tl.truncated = getattr(self._tl, "truncated", 0) + int(truncated)
        return extract_answer(raw, self.answer_format), raw
