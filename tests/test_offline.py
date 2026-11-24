"""Offline tests: no API calls. Run with `.venv/bin/python -m pytest -q`."""
from pathlib import Path

import pytest

from ragbench.datasets import load_dataset
from ragbench.llm import LLMClient, LLMConfig, Usage, UsageTracker
from ragbench.methods.raptor import Raptor, trim_to_last_sentence
from ragbench.methods.simple import LLMOnly
from ragbench.metrics import answer_metrics, retrieval_recall

DEV = Path("datasets/musique/musique_ans_v1.0_dev.jsonl")


class FakeLLM(LLMClient):
    """Echoes a deterministic 'summary'. If `truncate`, always reports hitting
    max_tokens and returns text cut mid-sentence."""

    def __init__(self, truncate=False):
        super().__init__(LLMConfig(label="fake", provider="fake", model="fake"), UsageTracker())
        self.truncate = truncate

    def _generate(self, prompt, system, max_tokens):
        if prompt.startswith("Question:") or "\nQuestion:" in prompt:
            return "Paris", Usage(calls=1, input_tokens=len(prompt) // 4, output_tokens=1), False
        body = prompt.split("\n\n", 1)[-1]
        words = body.split()[:40]
        text = " ".join(words) + ". This is the second sentence of the summary. And then it got cut"
        if self.truncate:
            return text, Usage(calls=1, input_tokens=len(prompt) // 4, output_tokens=max_tokens), True
        return text.rsplit(" And", 1)[0], Usage(calls=1, input_tokens=10, output_tokens=60), False


def test_trim_to_last_sentence():
    assert trim_to_last_sentence("A b c. D e f") == "A b c."
    assert trim_to_last_sentence('He said "hi." Then he') == 'He said "hi."'
    assert trim_to_last_sentence("v1.5 is out. Next") == "v1.5 is out."
    assert trim_to_last_sentence("no terminator at all") == "no terminator at all"


def test_metrics():
    m = answer_metrics("The Eiffel Tower", ["Eiffel Tower"])
    assert m["em"] == 1.0 and m["f1"] == 1.0
    m = answer_metrics("It is the Eiffel Tower in Paris", ["Eiffel Tower"])
    assert m["answer_recall"] == 1.0 and m["f1"] < 1.0 and m["contains"] == 1.0
    assert retrieval_recall(["a", "b"], ["a", "c"]) == 0.5
    assert retrieval_recall(None, ["a"]) is None


@pytest.mark.skipif(not DEV.exists(), reason="run scripts/prepare_data.py musique")
def test_musique_loader_pooled():
    ex, corpus = load_dataset({"name": "musique", "path": str(DEV), "limit": 5}, seed=0)
    assert len(ex) == 5
    assert all(17 <= len(e.passages) <= 20 for e in ex)
    assert all(2 <= len(e.supporting_pids) <= 4 for e in ex)
    assert len(corpus) == len({p.pid for p in corpus}) <= 100
    pids = {p.pid for p in corpus}
    assert all(s in pids for e in ex for s in e.supporting_pids)


def test_llm_only_no_context():
    llm = FakeLLM()
    m = LLMOnly({}, llm, None, {})
    from ragbench.datasets import Example
    out = m.answer(Example("q", "What is the capital of France?", ["Paris"], []), None)
    assert out["prediction"] == "Paris" and out["retrieved_pids"] is None


@pytest.mark.skipif(not DEV.exists(), reason="run scripts/prepare_data.py musique")
@pytest.mark.parametrize("truncate", [False, True])
def test_raptor_builds_and_summaries_end_on_sentence(truncate):
    from ragbench.embeddings import Embedder
    ex, corpus = load_dataset({"name": "musique", "path": str(DEV), "limit": 3}, seed=0)
    llm = FakeLLM(truncate=truncate)
    emb = Embedder({}, llm.tracker)
    m = Raptor({"num_layers": 2}, llm, emb, {})
    idx = m.build_index(corpus)
    layers = idx.stats["nodes_per_layer"]
    assert layers[0] >= len(corpus) and len(layers) >= 2, layers
    summaries = [n.text for l, ns in idx.tree.layer_to_nodes.items() if l > 0 for n in ns]
    assert summaries and all(s.rstrip().endswith(".") for s in summaries), summaries[:3]
    if truncate:
        assert idx.stats["summarizer"]["trimmed"] == len(summaries)
    assert llm.tracker.to_dict()["index"]["calls"] == len(summaries)

    out = m.answer(ex[0], idx)
    assert out["prediction"] == "Paris"
    assert out["context_tokens"] <= m.params["max_context_tokens"]
    assert set(out["retrieved_pids"]) <= set(out["retrieved_pids_expanded"])
