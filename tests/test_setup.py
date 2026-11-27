"""Offline tests for the experiment-setup pieces: YAML configs, sweeps, new metrics,
new methods and the judge parser. No API calls."""
from pathlib import Path

import pytest

from ragbench.config import expand_sweeps, load_config
from ragbench.datasets import Example, Passage, load_dataset
from ragbench.judge import parse_verdict
from ragbench.llm import LLMConfig, UsageTracker
from ragbench.methods.simple import FullContext, Oracle, render_subquestion
from ragbench.metrics import all_gold_retrieved, fact_recall, recall_by_hop

DEV = Path("datasets/musique/musique_ans_v1.0_dev.jsonl")


def test_sweep_labels():
    arms = expand_sweeps([{"name": "raptor", "top_k": None,
                           "sweep": {"max_context_tokens": [1000, 2000]}}])
    assert [a["label"] for a in arms] == ["raptor@1k", "raptor@2k"]
    assert arms[1]["max_context_tokens"] == 2000 and "sweep" not in arms[1]
    with pytest.raises(ValueError):
        expand_sweeps([{"name": "x"}, {"name": "x"}])


def test_yaml_extends(tmp_path):
    (tmp_path / "base.yaml").write_text("llms: [{label: a, provider: mock, model: m}]\n"
                                        "run: {seed: 1, name: base}\n")
    (tmp_path / "exp.yaml").write_text("extends: [base.yaml]\nrun: {name: exp}\n"
                                       "methods: [{name: llm_only}]\n")
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg["run"] == {"seed": 1, "name": "exp"} and cfg["llms"][0]["label"] == "a"


def test_shipped_configs_resolve():
    for p in Path("configs").rglob("*.yaml"):
        if p.parent.name != "common":
            cfg = load_config(p)
            assert cfg["methods"] and cfg["llms"], p


def test_new_metrics():
    assert all_gold_retrieved(["a", "b"], ["a", "b"]) == 1.0
    assert all_gold_retrieved(["a"], ["a", "b"]) == 0.0
    assert all_gold_retrieved(None, ["a"]) is None
    assert recall_by_hop(["a"], {"a": 1, "b": 2}) == {"1": 1.0, "2": 0.0}
    facts = [{"pid": "x", "fact": "The CEO, Sam, resigned."}, {"pid": "y", "fact": "Other."}]
    assert fact_recall("blah the ceo sam resigned blah", facts) == 0.5


def test_render_subquestion():
    assert render_subquestion("#1 >> spouse", ["Steve Hillage"]) == \
        "What is the spouse of Steve Hillage?"
    assert render_subquestion("Where did #1 die?", ["Ada"]) == "Where did Ada die?"


def test_parse_verdict():
    assert parse_verdict("CORRECT") == 1.0
    assert parse_verdict("Incorrect.") == 0.0
    assert parse_verdict("INCORRECT") == 0.0
    assert parse_verdict("maybe") is None


class _Echo:
    """Stand-in LLM: records the prompt."""
    def __init__(self):
        self.tracker = UsageTracker()
        self.cfg = LLMConfig(label="e", provider="mock", model="m")

    def complete(self, prompt, **kw):
        self.last = prompt
        return "x", False


def test_full_context_mhr_cap_keeps_gold():
    corpus = {f"p{i}": Passage(f"p{i}", f"T{i}", "word " * 300) for i in range(50)}
    ex = Example("q1", "Q?", ["a"], ["p3", "p7"], meta={})
    m = FullContext({"max_context_tokens": 2000}, _Echo(), None, {})
    out = m.answer(ex, corpus)
    assert {"p3", "p7"} <= set(out["retrieved_pids"])
    assert 2 < len(out["retrieved_pids"]) <= 7         # ~300 tokens each, 2k cap
    assert Oracle({}, _Echo(), None, {}).answer(ex, corpus)["retrieved_pids"].__len__() == 2


@pytest.mark.skipif(not DEV.exists(), reason="run scripts/prepare_data.py musique")
def test_musique_decomposition_matches_gold():
    ex, _ = load_dataset({"name": "musique", "path": str(DEV), "limit": 50})
    for e in ex:
        assert set(e.meta["gold_hop"]) == set(e.supporting_pids)
        assert sorted(e.meta["gold_hop"].values()) == list(range(1, e.meta["hops"] + 1))
