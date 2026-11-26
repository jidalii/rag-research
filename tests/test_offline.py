"""Offline tests: no API calls. Run with `.venv/bin/python -m pytest -q`."""
from pathlib import Path

import numpy as np
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


# --------------------------------------------------------------------------- #
# HippoRAG (GraphRAG) tests
# --------------------------------------------------------------------------- #
from ragbench.prompts import (openie_prompt, parse_entities, parse_triples,
                              passage_ner_prompt, query_ner_prompt)
from ragbench.methods.hipporag import (HippoRAG, build_graph, canon,
                                       personalized_pagerank, score_passages, GraphIndex)


def test_parse_entities_and_triples_good():
    ner = '{"named_entities": ["Radio City", "India", "Radio City"]}'
    assert parse_entities(ner) == ["Radio City", "India"]  # de-duplicated, order kept
    oie = '{"triples": [["Radio City", "located in", "India"], ["bad", "pair"]]}'
    assert parse_triples(oie) == [("Radio City", "located in", "India")]


def test_parse_oneshot_triples_with_missing_comma():
    # The verbatim HippoRAG one-shot triples have a missing comma after the
    # "English" row; the salvage parser must still recover all 12 triples.
    from ragbench.prompts import _HIPPO_ONESHOT_PASSAGE_TRIPLES
    triples = parse_triples(_HIPPO_ONESHOT_PASSAGE_TRIPLES)
    assert len(triples) == 12
    assert ("Radio City", "located in", "India") in triples


def test_parse_oneshot_entities():
    from ragbench.prompts import _HIPPO_ONESHOT_PASSAGE_ENTITIES
    ents = parse_entities(_HIPPO_ONESHOT_PASSAGE_ENTITIES)
    assert len(ents) == 7 and "PlanetRadiocity.com" in ents


@pytest.mark.parametrize("junk", ["", "unknown", "I cannot help", "{broken", "[1, 2, 3]"])
def test_parse_junk_returns_empty(junk):
    assert parse_entities(junk) == []
    assert parse_triples(junk) == []


def test_prompt_builders_are_single_turn_and_verbatim():
    sys_ner, user_ner = passage_ner_prompt("Some paragraph.")
    assert sys_ner.startswith("Your task is to extract named entities")
    assert "Radio City" in user_ner and "Some paragraph." in user_ner
    sys_oie, user_oie = openie_prompt("Some paragraph.", ["Foo"])
    assert "RDF" in sys_oie and '"Foo"' in user_oie
    sys_q, user_q = query_ner_prompt("Who founded Foo?")
    assert sys_q == "You're a very effective entity extraction system."
    assert "Arthur's Magazine" in user_q and "Who founded Foo?" in user_q


def test_build_graph_merges_shared_entities():
    extractions = [
        ("pid_a", ["Alice", "Paris"], [("Alice", "lives in", "Paris")]),
        ("pid_b", ["Bob", "paris"], [("Bob", "visited", "Paris")]),  # 'paris' canonicalizes
        ("pid_c", ["Lonely"], []),                                    # isolated node
    ]
    nodes, node_id, rows, cols, weights, node_pids = build_graph(extractions)
    assert canon("paris") in node_id and canon("Paris") in node_id
    paris = node_id["paris"]
    assert node_pids[paris] == {"pid_a", "pid_b"}       # merged across passages
    assert node_id["lonely"] in node_pids               # isolated entity still a node
    assert len(rows) == 2                               # two relation edges


def test_personalized_pagerank_and_scoring():
    # chain: A - B - C ; seed on A should rank A highest, then B.
    nodes = ["a", "b", "c"]
    idx = GraphIndex(pids=["p0"], nodes=nodes, node_id={n: i for i, n in enumerate(nodes)},
                     rows=[0, 1], cols=[1, 2], weights=[1.0, 1.0],
                     node_pids={0: {"p0"}, 1: {"p0"}, 2: {"p1"}},
                     node_emb=np.zeros((3, 1), dtype=np.float32), pid_text={})
    ppr = personalized_pagerank(idx, [0], alpha=0.5, max_iter=100, tol=1e-9)
    assert ppr[0] > ppr[1] > ppr[2]
    # empty seeds must not crash
    assert personalized_pagerank(idx, [], alpha=0.5, max_iter=100, tol=1e-9) == {}
    scores = score_passages(ppr, idx.node_pids, ["p0", "p1"])
    assert scores["p0"] > 0 and scores["p1"] > 0


class HippoFakeLLM(LLMClient):
    """Returns canned NER / OpenIE / query-NER JSON so HippoRAG can be built and
    queried offline. Answers 'Paris' to any QA prompt."""

    def __init__(self):
        super().__init__(LLMConfig(label="fake", provider="fake", model="fake"), UsageTracker())

    def _generate(self, prompt, system, max_tokens):
        u = Usage(calls=1, input_tokens=len(prompt) // 4, output_tokens=8)
        if system and system.startswith("Your task is to extract named entities"):
            return '{"named_entities": ["Alice", "Paris", "Alicia"]}', u, False
        if system and "RDF" in system:
            return '{"triples": [["Alice", "lives in", "Paris"]]}', u, False
        if system == "You're a very effective entity extraction system.":
            return '{"named_entities": ["Alice"]}', u, False
        return "Paris", u, False


@pytest.mark.skipif(not DEV.exists(), reason="run scripts/prepare_data.py musique")
def test_hipporag_build_index_and_answer_roundtrip():
    import pickle
    from ragbench.embeddings import Embedder
    ex, corpus = load_dataset({"name": "musique", "path": str(DEV), "limit": 3}, seed=0)
    llm = HippoFakeLLM()
    emb = Embedder({}, llm.tracker)
    # 0.35: this QA-tuned SBERT scores even "Alice"/"Alicia" only ~0.39 cosine,
    # so a low threshold is needed to exercise synonym-edge creation in the test.
    m = HippoRAG({"synonym_threshold": 0.35, "index_workers": 2}, llm, emb, {})
    idx = m.build_index(corpus)

    assert idx.stats["num_nodes"] >= 2
    assert idx.stats["ner_openie_calls"] == 2 * len(corpus)
    # index-phase calls = 2 LLM calls/passage (NER + OpenIE) + 1 batched embedder
    # call for the entity-synonym embeddings (Embedder records calls under "index").
    assert llm.tracker.to_dict()["index"]["calls"] == 2 * len(corpus) + 1
    # "Alice" and "Alicia" embed close together -> at least one synonym edge
    assert idx.stats["num_synonym_edges"] >= 1
    # every node's pids point at real corpus passages
    corpus_pids = {p.pid for p in corpus}
    assert all(pids <= corpus_pids for pids in idx.node_pids.values())
    # picklable
    idx2 = pickle.loads(pickle.dumps(idx))
    assert idx2.nodes == idx.nodes and idx2.stats == idx.stats

    out = m.answer(ex[0], idx)
    assert out["prediction"] == "Paris"
    assert set(out["retrieved_pids"]) <= set(out["retrieved_pids_expanded"])
    assert all(p in corpus_pids for p in out["retrieved_pids"])
    assert out["context_tokens"] <= m.params["max_context_tokens"]


def test_hipporag_answer_no_matching_entities_does_not_crash():
    # Query entities that link to no graph node -> empty seeds -> empty context.
    from ragbench.datasets import Example
    nodes = ["alice", "paris"]
    idx = GraphIndex(pids=["p0"], nodes=nodes, node_id={n: i for i, n in enumerate(nodes)},
                     rows=[0], cols=[1], weights=[1.0],
                     node_pids={0: {"p0"}, 1: {"p0"}},
                     node_emb=np.zeros((0, 0), dtype=np.float32),   # no embeddings -> no linking
                     pid_text={"p0": "Alice: lives in Paris"})
    m = HippoRAG({}, HippoFakeLLM(), None, {})
    out = m.answer(Example("q", "Who?", ["Paris"], []), idx)
    assert out["num_seed_nodes"] == 0
    assert out["retrieved_pids"] == [] and out["prediction"] == "Paris"
