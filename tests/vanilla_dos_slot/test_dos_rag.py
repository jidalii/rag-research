"""dos_rag: chunks grouped by document, in reading order inside each."""
from rag_helpers import (RANKED, check_same_selection_as_vanilla, check_token_budget, chunks,
                         prompt_order)
from ragbench.methods.vanilla_dos_slot import order_by_document


def test_order():
    ranked = list(RANKED)
    assert order_by_document(ranked) == chunks("A3 A8 B1 B2 C5")
    assert ranked == RANKED  # input is not modified


def test_one_document_is_plain_reading_order():
    assert order_by_document(chunks("A8 A2 A5")) == chunks("A2 A5 A8")


def test_distinct_documents_keep_similarity_order():
    assert order_by_document(chunks("C5 A8 B2")) == chunks("C5 A8 B2")


def test_same_chunks_as_vanilla_in_dos_order():
    ranked, out = check_same_selection_as_vanilla("dos_rag")
    assert prompt_order(out) == order_by_document(ranked)
    assert out["differs_from_vanilla"] is True


def test_token_budget_and_retrieved_pids():
    check_token_budget("dos_rag")
