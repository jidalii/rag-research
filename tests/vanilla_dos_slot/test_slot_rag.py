"""slot_rag: reading order inside each document, each document keeping its positions."""
from rag_helpers import (RANKED, check_same_selection_as_vanilla, check_token_budget, chunks,
                         prompt_order)
from ragbench.methods.vanilla_dos_slot import order_within_slots


def test_order():
    ranked = list(RANKED)
    assert order_within_slots(ranked) == chunks("A3 B1 C5 A8 B2")
    assert ranked == RANKED  # input is not modified


def test_one_document_is_plain_reading_order():
    assert order_within_slots(chunks("A8 A2 A5")) == chunks("A2 A5 A8")


def test_distinct_documents_keep_similarity_order():
    assert order_within_slots(chunks("C5 A8 B2")) == chunks("C5 A8 B2")


def test_same_chunks_as_vanilla_in_slot_order():
    ranked, out = check_same_selection_as_vanilla("slot_rag")
    assert prompt_order(out) == order_within_slots(ranked)
    # every position still belongs to the same document
    assert [pid for pid, _ in prompt_order(out)] == [pid for pid, _ in ranked]


def test_token_budget_and_retrieved_pids():
    check_token_budget("slot_rag")
