"""Shared pieces for the vanilla_rag / dos_rag / slot_rag tests. Not a test file.

A fake embedder and a fake LLM keep the tests offline: no API calls, no model download.
"""
import pickle

import numpy as np
import tiktoken

from ragbench.datasets import Example, Passage
from ragbench.llm import LLMClient, LLMConfig, Usage, UsageTracker
from ragbench.methods import get_method

TOK = tiktoken.get_encoding("cl100k_base")

# search results, most similar first: (document, position in the document)
RANKED = [("A", 8), ("B", 2), ("C", 5), ("A", 3), ("B", 1)]


def chunks(spec: str):
    """"A8 B2" -> [("A", 8), ("B", 2)]"""
    return [(s[0], int(s[1:])) for s in spec.split()]


def squash(text: str) -> str:
    """Drop whitespace, which the splitter may change, to compare what is left."""
    return "".join(text.split())


class WordEmbedder:
    """Cosine over counts of a few keywords, so the ranking is predictable."""
    VOCAB = ["alpha", "beta", "gamma", "delta"]
    key = "fake"

    def embed(self, texts, phase="embed"):
        v = np.array([[t.lower().count(w) for w in self.VOCAB] + [0.1] for t in texts],
                     dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class RecordingLLM(LLMClient):
    def __init__(self):
        super().__init__(LLMConfig(label="fake", provider="fake", model="fake"), UsageTracker())
        self.prompts = []

    def _generate(self, prompt, system, max_tokens):
        self.prompts.append(prompt)
        return "Paris", Usage(calls=1, input_tokens=1, output_tokens=1), False


def run(name, passages, question, **params):
    """Index the passages and answer one question: (output, index, prompt)."""
    llm = RecordingLLM()
    method = get_method(name)(params, llm, WordEmbedder(), {})
    index = pickle.loads(pickle.dumps(method.build_index(passages)))  # runner caches it
    out = method.answer(Example("q", question, ["Paris"], []), index)
    return out, index, llm.prompts[0]


def _sentences(*words):
    return " ".join(f"{w} " * 12 + "." for w in words)


PASSAGES = [Passage("p1", "", _sentences("beta", "delta", "alpha")),
            Passage("p2", "", _sentences("gamma", "alpha alpha beta"))]


def run_example(name):
    """A search over PASSAGES that returns several chunks from both documents."""
    return run(name, PASSAGES, "alpha alpha alpha beta", chunk_max_tokens=20, top_k=4)


def prompt_order(out) -> list[tuple[str, int]]:
    return [tuple(c) for c in out["retrieved_chunks"]]


def check_prompt_follows_order(out, index, prompt):
    """The prompt really holds the chunks in the reported order."""
    text_of = dict(zip(index.chunks, index.texts))
    at = [prompt.index(text_of[c]) for c in prompt_order(out)]
    assert at == sorted(at)


def check_same_selection_as_vanilla(name):
    """Compare a method with vanilla_rag; returns (vanilla order, method output)."""
    vanilla, _, _ = run_example("vanilla_rag")
    out, index, prompt = run_example(name)
    ranked = prompt_order(vanilla)
    assert len({pid for pid, _ in ranked}) == 2
    assert sorted(prompt_order(out)) == sorted(ranked)
    assert out["retrieved_pids"] == vanilla["retrieved_pids"]
    assert out["context_tokens"] == vanilla["context_tokens"]
    check_prompt_follows_order(out, index, prompt)
    return ranked, out


def check_token_budget(name):
    out, index, _ = run(name, PASSAGES, "alpha", chunk_max_tokens=20, top_k=10,
                        max_context_tokens=30)
    used = sum(len(TOK.encode(index.texts[index.chunks.index(c)])) for c in prompt_order(out))
    assert 0 < used <= 30
    # recall only counts documents that made it into the prompt
    assert set(out["retrieved_pids"]) == {pid for pid, _ in prompt_order(out)}
