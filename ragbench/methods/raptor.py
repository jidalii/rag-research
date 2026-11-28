"""RAPTOR (Sarthi et al., ICLR 2024) on top of the vendored upstream code in ./raptor.

Reused from upstream without changes: UMAP + GMM clustering (RAPTOR_Clustering),
recursive summarization (ClusterTreeBuilder.construct_tree), Tree/Node, and
TreeRetriever (collapsed-tree or tree-traversal retrieval).

Adaptations (all in this file, upstream files are never edited):
  1. Summarizer / embedder adapters, so any provider in ragbench.llm (OpenAI,
     DeepSeek, Claude) builds the tree and token usage is counted as phase="index".
  2. Complete-sentence summaries. Upstream passes summarization_length (100) as a
     hard max_tokens, which cuts summaries mid-sentence. Here the length target is
     put in the prompt, max_tokens gets headroom, and if the model still hits the
     limit the text is trimmed back to its last complete sentence.
  3. Leaves are built per corpus passage (upstream split_text, 100 tokens), so every
     leaf knows which passage it came from; this is what makes retrieval recall
     measurable. Upstream build_from_text would chunk across passage boundaries.
  4. Leaf embeddings are computed in one batch; UMAP gets a fixed random_state
     (upstream leaves it unset, so trees differ between runs even with seeds).
"""
from __future__ import annotations

import contextlib
import copy
import functools
import logging
import random
import re
import threading
import time
import types
import warnings
from dataclasses import dataclass, field

import numpy as np

from .base import Method

log = logging.getLogger(__name__)

UPSTREAM_SUMMARY = ("Write a summary of the following, including as many key details as "
                    "possible: {context}:")
COMPLETE_SUMMARY = ("Write a summary of the following, including as many key details as "
                    "possible. Use at most about {words} words and end with a complete "
                    "sentence.\n\n{context}")
SUMMARY_SYSTEM = "You are a helpful assistant."

_SENT_END = re.compile(r"""[.!?。](?:["'”’)\]]*)(?=\s|$)""")


def trim_to_last_sentence(text: str) -> str:
    """Cut a truncated generation back to its last complete sentence."""
    text = text.rstrip()
    ends = list(_SENT_END.finditer(text))
    if not ends:
        return text
    return text[: ends[-1].end()].rstrip()


# --------------------------------------------------------------------------- #
# Upstream model adapters
# --------------------------------------------------------------------------- #
def _make_summarizer(llm, params: dict):
    from raptor.SummarizationModels import BaseSummarizationModel

    prompt_name = params.get("summary_prompt", "complete")
    template = COMPLETE_SUMMARY if prompt_name == "complete" else UPSTREAM_SUMMARY
    headroom = params.get("summary_max_tokens_multiplier", 4)

    class Summarizer(BaseSummarizationModel):
        def __init__(self):
            self.stats = {"summaries": 0, "hit_limit": 0, "trimmed": 0, "retried": 0,
                          "failed": 0}
            self.first_error = None
            self._lock = threading.Lock()

        def summarize(self, context, max_tokens=150):
            # Upstream runs this in a thread pool and never reads the futures, so an
            # exception here vanishes and the cluster silently gets no summary. Count
            # failures so build_index can refuse to return (and cache) a broken tree.
            try:
                return self._summarize(context, max_tokens)
            except Exception as e:
                with self._lock:
                    self.stats["failed"] += 1
                    self.first_error = self.first_error or repr(e)
                raise

        def _summarize(self, context, max_tokens):
            # max_tokens is upstream's summarization_length: a length *target*
            words = max(20, int(max_tokens * 0.75))
            prompt = template.format(context=context, words=words)
            budget = max_tokens * headroom
            text, truncated = llm.complete(prompt, system=SUMMARY_SYSTEM,
                                           max_tokens=budget, phase="index")
            retried = False
            if truncated and not _SENT_END.search(text):
                # nothing usable to trim back to: try once with double the budget
                retried = True
                text, truncated = llm.complete(prompt, system=SUMMARY_SYSTEM,
                                               max_tokens=budget * 2, phase="index")
            trimmed = False
            if truncated:
                new = trim_to_last_sentence(text)
                trimmed, text = new != text, new
            with self._lock:
                self.stats["summaries"] += 1
                self.stats["hit_limit"] += int(truncated)
                self.stats["trimmed"] += int(trimmed)
                self.stats["retried"] += int(retried)
            if not text.strip():
                raise RuntimeError("empty RAPTOR summary (refusal or provider error)")
            return text

    return Summarizer()


def _make_embedding_model(embedder):
    from raptor.EmbeddingModels import BaseEmbeddingModel

    class CachedEmbedding(BaseEmbeddingModel):
        """Upstream calls create_embedding(text) one string at a time; leaves are
        pre-embedded in a batch and served from this cache."""

        def __init__(self):
            self.cache: dict[str, np.ndarray] = {}
            self._lock = threading.Lock()

        def prefill(self, texts: list[str], phase: str = "embed"):
            todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
            if todo:
                for t, v in zip(todo, embedder.embed(todo, phase=phase)):
                    self.cache[t] = v

        def create_embedding(self, text):
            with self._lock:
                hit = self.cache.get(text)
            if hit is None:
                hit = embedder.embed([text])[0]
                with self._lock:
                    self.cache[text] = hit
            return hit

    return CachedEmbedding()


_UMAP_LOCK = threading.Lock()


@contextlib.contextmanager
def _seeded_umap(seed: int | None):
    if seed is None:
        yield
        return
    import umap as umap_module
    from raptor import cluster_utils
    with _UMAP_LOCK:
        orig = cluster_utils.umap
        cluster_utils.umap = types.SimpleNamespace(
            UMAP=functools.partial(umap_module.UMAP, random_state=seed))
        try:
            yield
        finally:
            cluster_utils.umap = orig


EMB_KEY = "EMB"


@dataclass
class RaptorIndex:
    tree: object                                  # raptor.tree_structures.Tree
    leaf_pid: dict[int, str]                      # leaf node index -> passage pid
    layer_of: dict[int, int]
    stats: dict = field(default_factory=dict)

    def descendant_pids(self, node_index: int) -> set[str]:
        node = self.tree.all_nodes[node_index]
        if not node.children:
            return {self.leaf_pid[node_index]}
        out: set[str] = set()
        for c in node.children:
            out |= self.descendant_pids(c)
        return out


# --------------------------------------------------------------------------- #
class Raptor(Method):
    name = "raptor"
    needs_index = True
    uses_embedder = True

    DEFAULTS = {
        # tree construction (upstream defaults)
        "chunk_max_tokens": 100,
        "summarization_length": 100,
        "num_layers": 5,
        "reduction_dimension": 10,
        "cluster_threshold": 0.1,
        "max_length_in_cluster": 3500,
        "summary_prompt": "complete",       # "complete" | "upstream"
        "summary_max_tokens_multiplier": 4,
        "parallel_summaries": True,
        "seed": 224,
        # retrieval (upstream TreeRetriever.retrieve)
        "collapse_tree": True,
        "top_k": 10,
        "max_context_tokens": 2000,
    }

    def __init__(self, params, llm, embedder, qa_cfg):
        super().__init__({**self.DEFAULTS, **params}, llm, embedder, qa_cfg)
        import tiktoken
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.emb_model = _make_embedding_model(embedder)

    def index_params(self) -> dict:
        keys = ["chunk_max_tokens", "summarization_length", "num_layers",
                "reduction_dimension", "cluster_threshold", "max_length_in_cluster",
                "summary_prompt", "summary_max_tokens_multiplier", "seed"]
        return {k: self.params[k] for k in keys}

    # ------------------------------------------------------------------ build
    def _builder(self, summarizer):
        from raptor import ClusterTreeBuilder, ClusterTreeConfig
        from raptor.tree_structures import Tree

        parallel = self.params["parallel_summaries"]

        class ChunkTreeBuilder(ClusterTreeBuilder):
            def build_from_chunks(self, chunks: list[str]):
                leaf_nodes = {}
                for i, text in enumerate(chunks):
                    _, node = self.create_node(i, text)
                    leaf_nodes[i] = node
                layer_to_nodes = {0: list(leaf_nodes.values())}
                all_nodes = copy.deepcopy(leaf_nodes)
                roots = self.construct_tree(all_nodes, all_nodes, layer_to_nodes,
                                            use_multithreading=parallel)
                return Tree(all_nodes, roots, leaf_nodes, self.num_layers, layer_to_nodes)

        p = self.params
        cfg = ClusterTreeConfig(
            reduction_dimension=p["reduction_dimension"],
            clustering_params={"threshold": p["cluster_threshold"],
                               "max_length_in_cluster": p["max_length_in_cluster"]},
            tokenizer=self.tokenizer,
            max_tokens=p["chunk_max_tokens"],
            num_layers=p["num_layers"],
            summarization_length=p["summarization_length"],
            summarization_model=summarizer,
            embedding_models={EMB_KEY: self.emb_model},
            cluster_embedding_model=EMB_KEY,
        )
        return ChunkTreeBuilder(cfg)

    def build_index(self, passages):
        from raptor.utils import reverse_mapping, split_text

        chunks, chunk_pid = [], []
        for p in passages:
            text = f"{p.title}: {p.text}" if p.title else p.text
            for c in split_text(text, self.tokenizer, self.params["chunk_max_tokens"]):
                chunks.append(c)
                chunk_pid.append(p.pid)
        if not chunks:
            raise ValueError("RAPTOR: empty corpus")

        seed = self.params["seed"]
        random.seed(seed)
        np.random.seed(seed)
        t0 = time.perf_counter()
        self.emb_model.prefill(chunks)
        summarizer = _make_summarizer(self.llm, self.params)
        builder = self._builder(summarizer)
        with _seeded_umap(seed), warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tree = builder.build_from_chunks(chunks)

        if summarizer.stats["failed"]:
            raise RuntimeError(f"RAPTOR: {summarizer.stats['failed']} summaries failed, so the "
                               f"tree is incomplete and is not cached. First error: "
                               f"{summarizer.first_error}")
        layer_of = reverse_mapping(tree.layer_to_nodes)
        summary_nodes = [n for l, ns in tree.layer_to_nodes.items() if l > 0 for n in ns]
        stats = {
            "num_passages": len(passages),
            "num_leaves": len(chunks),
            "nodes_per_layer": {int(k): len(v) for k, v in tree.layer_to_nodes.items()},
            "leaf_tokens": sum(len(self.tokenizer.encode(c)) for c in chunks),
            "summary_tokens": sum(len(self.tokenizer.encode(n.text)) for n in summary_nodes),
            "summarizer": summarizer.stats,
            "build_time_s": round(time.perf_counter() - t0, 2),
        }
        return RaptorIndex(tree, dict(enumerate(chunk_pid)), layer_of, stats)

    # ---------------------------------------------------------------- answer
    def answer(self, ex, index: RaptorIndex):
        from raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig

        retriever = TreeRetriever(
            TreeRetrieverConfig(tokenizer=self.tokenizer, context_embedding_model=EMB_KEY,
                                embedding_model=self.emb_model),
            index.tree,
        )
        p = self.params
        context, info = retriever.retrieve(
            # top_k = None: the token budget alone decides how many nodes fit
            ex.question, top_k=p["top_k"] or 1_000_000, max_tokens=p["max_context_tokens"],
            collapse_tree=p["collapse_tree"], return_layer_information=True,
        )
        node_ids = [r["node_index"] for r in info]
        leaf_pids = [index.leaf_pid[i] for i in node_ids if i in index.leaf_pid]
        expanded = set()
        for i in node_ids:
            expanded |= index.descendant_pids(i)
        pred, raw = self.generate_answer(ex.question, context)
        return {
            "prediction": pred, "raw": raw,
            "retrieved_pids": list(dict.fromkeys(leaf_pids)),
            "retrieved_pids_expanded": sorted(expanded),
            "retrieved_layers": [r["layer_number"] for r in info],
            "context_tokens": len(self.tokenizer.encode(context)),
        }
