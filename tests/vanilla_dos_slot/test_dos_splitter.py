"""The DOS RAG splitter and chunk numbering used by vanilla_rag, dos_rag and slot_rag."""
from pathlib import Path

import pytest

from rag_helpers import TOK, run, squash
from ragbench.datasets import load_dataset
from ragbench.methods.vanilla_dos_slot import split_text

DATA = Path(__file__).resolve().parents[2] / "datasets"
MUSIQUE = DATA / "musique" / "musique_ans_v1.0_dev.jsonl"
MULTIHOP = DATA / "multihop_rag"


def test_split_keeps_reading_order_around_unsplittable_text():
    words = " ".join(f"w{i}" for i in range(90))  # no , ; : to split on
    text = f"Alpha comes first. {words}. Gamma comes last."
    pieces = split_text(text, TOK, 40)
    assert squash("".join(pieces)) == squash(text)
    assert pieces[0].startswith("Alpha") and "Gamma" in pieces[-1]


def test_split_keeps_sentence_punctuation():
    text = "Is this a question? Yes it is! Here is a statement."
    assert split_text(text, TOK, 100) == [text]


@pytest.mark.parametrize("dataset, path", [("musique", MUSIQUE),
                                           ("multihop_rag", MULTIHOP / "corpus.json")])
def test_real_dataset_chunks(dataset, path):
    if not path.exists():
        pytest.skip(f"run scripts/prepare_data.py {dataset}")
    cfg = {"name": dataset, "path": str(MUSIQUE if dataset == "musique" else MULTIHOP), "limit": 3}
    examples, corpus = load_dataset(cfg, seed=0)
    corpus = corpus[:15]
    out, index, _ = run("dos_rag", corpus, examples[0].question)
    assert set(out["retrieved_pids"]) <= {p.pid for p in corpus}
    # every passage's chunks are numbered 0..n-1 and reproduce its text in order
    for p in corpus:
        mine = [(pos, t) for (pid, pos), t in zip(index.chunks, index.texts) if pid == p.pid]
        assert [pos for pos, _ in mine] == list(range(len(mine)))
        assert squash("".join(t for _, t in mine)) == squash(f"{p.title}: {p.text}")
