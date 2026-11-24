"""Embedding backends. Kept independent of the generator LLM so every
LLM is compared over the same retrieval embedding space (Claude has no
embedding API)."""
from __future__ import annotations

import os
import threading

import numpy as np

from .llm import Usage, UsageTracker


class Embedder:
    def __init__(self, cfg: dict, tracker: UsageTracker):
        self.provider = cfg.get("provider", "local")
        self.model = cfg.get("model", "sentence-transformers/multi-qa-mpnet-base-cos-v1")
        self.batch_size = cfg.get("batch_size", 64)
        self.tracker = tracker
        self._lock = threading.Lock()
        if self.provider == "local":
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.model, device=cfg.get("device"))
        elif self.provider == "openai":
            from openai import OpenAI
            self._oa = OpenAI(api_key=os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY")))
        else:
            raise ValueError(f"Unknown embedding provider {self.provider!r}")

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    def embed(self, texts: list[str], phase: str = "embed") -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self.provider == "local":
            with self._lock:  # SentenceTransformer is not thread-safe
                vecs = self._st.encode(texts, batch_size=self.batch_size,
                                       normalize_embeddings=True, show_progress_bar=False)
            self.tracker.record(phase, Usage(calls=1))
            return np.asarray(vecs, dtype=np.float32)
        out = []
        for i in range(0, len(texts), self.batch_size):
            resp = self._oa.embeddings.create(model=self.model, input=texts[i:i + self.batch_size])
            out.extend(d.embedding for d in resp.data)
            self.tracker.record(phase, Usage(calls=1, input_tokens=resp.usage.prompt_tokens))
        v = np.asarray(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
