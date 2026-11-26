from __future__ import annotations

import re
from dataclasses import dataclass, field

import nltk
import numpy as np

from .base import Method

# --------------------------------------------------------------------------- #
# Chunking: the splitter of the original DOS RAG code.
#
# Copied from https://github.com/alex-laitenberger/dos-rag-eval (source/method/utils.py),
# MIT license, Copyright (c) 2025 Alex Laitenberger.
#
# Changes:
#   - Fix: when part of an over-long sentence has no , ; : to split on, upstream emits
#     its word slices before the sentences it is still collecting, so chunks leave
#     reading order. The collected text is now emitted first. Other text is chunked
#     exactly as upstream does.
#   - NLTK's sentence data is fetched on first use instead of on import.
# --------------------------------------------------------------------------- #

def _ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        nltk.data.find("tokenizers/punkt_tab")  # fail clearly if the download did not work


def split_text(text: str, tokenizer, max_tokens: int):
    """
    Splits a document into chunks of up to max_tokens.

    Parameters:
        text (str): The input text to split.
        tokenizer: Pre-initialized tiktoken tokenizer.
        max_tokens (int): Maximum tokens per chunk.

    Returns:
        list of str: List of text chunks.
    """

    _ensure_punkt()

    # Break the text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)

    # Helper function to calculate token count for a sentence
    def get_token_count(sentence):
        return len(tokenizer.encode(" " + sentence))

    # Helper function to handle adding chunks
    def handle_chunk(chunk):
        if chunk:
            chunks.append(" ".join(chunk))

    chunks, current_chunk, current_length = [], [], 0

    for sentence in sentences:
        if not sentence.strip():
            continue

        token_count = get_token_count(sentence)

        # Handle long sentences by further splitting
        if token_count > max_tokens:
            for sub_sentence in re.split(r"(?<=[,;:])\s", sentence):
                sub_token_count = get_token_count(sub_sentence)
                words = sub_sentence.split()

                if sub_token_count > max_tokens:
                    # ragbench fix: emit the text collected so far first
                    handle_chunk(current_chunk)
                    current_chunk, current_length = [], 0
                    for i in range(0, len(words), max_tokens):
                        handle_chunk(words[i:i + max_tokens])
                else:
                    if current_length + sub_token_count > max_tokens:
                        handle_chunk(current_chunk)
                        current_chunk, current_length = [], 0

                    current_chunk.append(sub_sentence)
                    current_length += sub_token_count
            continue

        # Add sentence to current chunk
        if current_length + token_count > max_tokens:
            handle_chunk(current_chunk)
            current_chunk, current_length = [], 0

        current_chunk.append(sentence)
        current_length += token_count

    handle_chunk(current_chunk)  # Add any remaining chunk
    return chunks


# --------------------------------------------------------------------------- #
# Ordering. `ranked` is a list of (pid, position) pairs, most similar first.
# --------------------------------------------------------------------------- #
def order_by_similarity(ranked: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return list(ranked)


def _group_by_document(ranked) -> dict[str, list[tuple[str, int]]]:
    """pid -> its chunks in reading order. Keys are in order of each document's best chunk."""
    groups: dict[str, list[tuple[str, int]]] = {}
    for chunk in ranked:
        groups.setdefault(chunk[0], []).append(chunk)
    return {pid: sorted(chunks, key=lambda c: c[1]) for pid, chunks in groups.items()}


def order_by_document(ranked: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return [c for chunks in _group_by_document(ranked).values() for c in chunks]


def order_within_slots(ranked: list[tuple[str, int]]) -> list[tuple[str, int]]:
    queues = {pid: iter(chunks) for pid, chunks in _group_by_document(ranked).items()}
    return [next(queues[pid]) for pid, _ in ranked]


# --------------------------------------------------------------------------- #
@dataclass
class ChunkIndex:
    chunks: list[tuple[str, int]]     # (passage pid, position within that passage)
    texts: list[str]
    emb: np.ndarray                   # (num_chunks, dim), normalized
    stats: dict = field(default_factory=dict)


class VanillaRAG(Method):
    """Retrieved chunks in similarity order, most similar first."""
    name = "vanilla_rag"
    needs_index = True
    uses_embedder = True
    order = staticmethod(order_by_similarity)

    DEFAULTS = {"chunk_max_tokens": 100, "top_k": 10, "max_context_tokens": 2000}

    def __init__(self, params, llm, embedder, qa_cfg):
        super().__init__({**self.DEFAULTS, **params}, llm, embedder, qa_cfg)
        import tiktoken
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def index_params(self) -> dict:
        # "splitter" is in the cache key so indexes from another splitter are not reused
        return {"chunk_max_tokens": self.params["chunk_max_tokens"], "splitter": "dos-rag"}

    def build_index(self, passages):
        chunks, texts = [], []
        for p in passages:
            text = f"{p.title}: {p.text}" if p.title else p.text
            pieces = split_text(text, self.tokenizer, self.params["chunk_max_tokens"])
            for position, piece in enumerate(pieces):
                chunks.append((p.pid, position))
                texts.append(piece)
        if not chunks:
            raise ValueError(f"{self.name}: empty corpus")
        emb = self.embedder.embed(texts, phase="index")
        stats = {"num_passages": len(passages), "num_chunks": len(chunks),
                 "chunk_tokens": sum(len(self.tokenizer.encode(t)) for t in texts)}
        return ChunkIndex(chunks, texts, emb, stats)

    def retrieve(self, question: str, index: ChunkIndex) -> list[int]:
        """Chunk indices, most similar first. Identical for all three methods."""
        query = self.embedder.embed([question])[0]
        ranking = np.argsort(-(index.emb @ query), kind="stable")[: self.params["top_k"]]
        selected, used = [], 0
        for i in ranking:
            tokens = len(self.tokenizer.encode(index.texts[i]))
            if used + tokens > self.params["max_context_tokens"]:
                break
            selected.append(int(i))
            used += tokens
        return selected

    def answer(self, ex, index: ChunkIndex):
        selected = self.retrieve(ex.question, index)
        ranked = [index.chunks[i] for i in selected]
        text_of = {index.chunks[i]: index.texts[i] for i in selected}
        ordered = self.order(ranked)
        # same chunk formatting as RAPTOR's get_text
        context = "".join(" ".join(text_of[c].splitlines()) + "\n\n" for c in ordered)
        pred, raw = self.generate_answer(ex.question, context or None)
        return {
            "prediction": pred, "raw": raw,
            # only documents with a chunk in the prompt, listed in similarity order
            "retrieved_pids": list(dict.fromkeys(pid for pid, _ in ranked)),
            "retrieved_chunks": [list(c) for c in ordered],    # in prompt order
            "differs_from_vanilla": ordered != ranked,
            "context_tokens": len(self.tokenizer.encode(context)),
        }


class DOSRAG(VanillaRAG):
    """Chunks grouped by document, in reading order inside each document."""
    name = "dos_rag"
    order = staticmethod(order_by_document)


class SlotRAG(VanillaRAG):
    """Reading order inside each document without moving documents' positions."""
    name = "slot_rag"
    order = staticmethod(order_within_slots)
