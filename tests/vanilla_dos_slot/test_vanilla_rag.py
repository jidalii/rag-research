"""vanilla_rag: chunks in similarity order."""
from rag_helpers import (RANKED, WordEmbedder, check_prompt_follows_order,
                         check_token_budget, prompt_order, run_example)
from ragbench.methods.vanilla_dos_slot import order_by_similarity


def test_order():
    ranked = list(RANKED)
    assert order_by_similarity(ranked) == RANKED


def test_prompt_follows_similarity():
    out, index, prompt = run_example("vanilla_rag")
    query = WordEmbedder().embed(["alpha alpha alpha beta"])[0]
    scores = dict(zip(index.chunks, index.emb @ query))
    used = [scores[c] for c in prompt_order(out)]
    assert used == sorted(used, reverse=True)
    assert out["differs_from_vanilla"] is False
    check_prompt_follows_order(out, index, prompt)


def test_token_budget_and_retrieved_pids():
    check_token_budget("vanilla_rag")
