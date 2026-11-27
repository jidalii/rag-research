"""HippoRAG (Gutierrez et al., NeurIPS 2024) over the shared corpus.

HippoRAG builds a knowledge graph from per-passage OpenIE (named-entity
recognition then relation-triple extraction), links each entity to similar
entities via embeddings (synonym edges), and answers a question by running
Personalized PageRank (PPR) seeded on the entities named in the question. Each
passage is then scored by the PPR mass of the entities it contains, and the
top-k passages become the answer context.

Faithful to upstream HippoRAG:
  - The verbatim v1 NER / OpenIE / query-NER prompts (see ragbench.prompts). The
    one-shot example is folded into the single user string because ragbench's
    LLMClient is single-turn; the instructions and example are kept word for word.
  - Entity knowledge graph + embedding-based synonym edges + Personalized PageRank.

Adaptations for this benchmark:
  - PPR uses NetworkX (`nx.pagerank`) rather than upstream's igraph.
  - All extraction goes through self.llm (phase="index" when building, "query"
    when answering) so tokens and $ are counted; embeddings use self.embedder.
  - Retrieval units are corpus passages, so retrieval_recall is paragraph-level
    like the other methods. retrieved_pids are the top-k passages; the looser
    retrieved_pids_expanded is every passage touched by a non-zero-mass entity.

Not yet implemented (deferred): HippoRAG's node-specificity (IDF) weighting and
IRCoT-style multi-step retrieval.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from ..prompts import (openie_prompt, parse_entities, parse_triples,
                       passage_ner_prompt, query_ner_prompt)
from .base import Method

log = logging.getLogger(__name__)


def canon(entity: str) -> str:
    """Canonical form of an entity string used as a graph node key."""
    return " ".join(entity.lower().split())


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
@dataclass
class GraphIndex:
    pids: list[str]                               # corpus passage ids
    nodes: list[str]                              # canonical entity strings
    node_id: dict[str, int]
    rows: list[int]                               # edge endpoints (node ids)
    cols: list[int]
    weights: list[float]                          # relation edges 1.0, synonym edges = cosine
    node_pids: dict[int, set[str]]                # node id -> passages it appeared in
    node_emb: np.ndarray                          # (num_nodes, dim) normalized, or (0, 0)
    pid_text: dict[str, str]                      # pid -> "title: text" for context
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Graph construction (pure; no LLM, no embeddings)
# --------------------------------------------------------------------------- #
def build_graph(passage_extractions: list[tuple[str, list[str], list[tuple[str, str, str]]]]):
    """Build the entity graph from per-passage (pid, entities, triples).

    Returns (nodes, node_id, rows, cols, weights, node_pids). Entities are
    canonicalized (lowercased, whitespace-collapsed) so the same entity across
    passages collapses to one node. Both NER entities and the subject/object of
    each triple become nodes; each triple adds one relation edge (weight 1.0).
    """
    nodes: list[str] = []
    node_id: dict[str, int] = {}
    node_pids: dict[int, set[str]] = {}

    def add_node(name: str, pid: str | None) -> int:
        key = canon(name)
        nid = node_id.get(key)
        if nid is None:
            nid = len(nodes)
            node_id[key] = nid
            nodes.append(key)
            node_pids[nid] = set()
        if pid is not None:
            node_pids[nid].add(pid)
        return nid

    # (src, dst) -> accumulated weight, so repeated relations don't inflate edges
    edge_w: dict[tuple[int, int], float] = {}
    for pid, entities, triples in passage_extractions:
        for e in entities:
            add_node(e, pid)
        for s, _r, o in triples:
            si = add_node(s, pid)
            oi = add_node(o, pid)
            if si == oi:
                continue
            key = (si, oi) if si < oi else (oi, si)
            edge_w[key] = edge_w.get(key, 0.0) + 1.0

    rows, cols, weights = [], [], []
    for (a, b), w in edge_w.items():
        rows.append(a)
        cols.append(b)
        weights.append(float(w))
    return nodes, node_id, rows, cols, weights, node_pids


# --------------------------------------------------------------------------- #
# Personalized PageRank (NetworkX) + passage scoring
# --------------------------------------------------------------------------- #
def personalized_pagerank(index: GraphIndex, seed_ids: list[int], *, alpha: float,
                          max_iter: int, tol: float) -> dict[int, float]:
    """PPR over the (undirected) entity graph, restarting on seed_ids. Returns
    {node_id: score} for the nodes reachable; empty dict when there are no seeds."""
    import networkx as nx

    seeds = [s for s in seed_ids if 0 <= s < len(index.nodes)]
    if not seeds:
        return {}
    g = nx.Graph()
    g.add_nodes_from(range(len(index.nodes)))
    for a, b, w in zip(index.rows, index.cols, index.weights):
        # parallel edges collapse; keep the larger weight
        if g.has_edge(a, b):
            g[a][b]["weight"] = max(g[a][b]["weight"], w)
        else:
            g.add_edge(a, b, weight=w)
    personalization = {s: 1.0 / len(seeds) for s in seeds}
    try:
        return nx.pagerank(g, alpha=alpha, personalization=personalization,
                           max_iter=max_iter, tol=tol, weight="weight")
    except nx.PowerIterationFailedConvergence:
        log.warning("PPR did not converge in %d iterations; using last estimate", max_iter)
        return nx.pagerank(g, alpha=alpha, personalization=personalization,
                           max_iter=max_iter * 10, tol=tol * 10, weight="weight")


def score_passages(ppr: dict[int, float], node_pids: dict[int, set[str]],
                   pids: list[str]) -> dict[str, float]:
    """Passage score = sum of the PPR mass of the entities that appear in it."""
    scores = {p: 0.0 for p in pids}
    for nid, mass in ppr.items():
        if mass <= 0:
            continue
        for pid in node_pids.get(nid, ()):
            if pid in scores:
                scores[pid] += mass
    return scores


# --------------------------------------------------------------------------- #
class HippoRAG(Method):
    name = "hipporag"
    needs_index = True
    uses_embedder = True

    DEFAULTS = {
        "synonym_threshold": 0.8,     # cosine; add synonym edges at/above this
        "top_k": 10,                  # passages used as context
        "ppr_damping": 0.5,           # NetworkX alpha
        "ppr_max_iter": 100,
        "ppr_tol": 1e-6,
        "link_top_k": 1,              # graph nodes linked per query entity
        "max_context_tokens": 2000,
        "openie_max_tokens": 1024,
        "ner_max_tokens": 512,
        "index_workers": 8,           # threads for per-passage extraction
        "seed": 224,
    }

    def __init__(self, params, llm, embedder, qa_cfg):
        super().__init__({**self.DEFAULTS, **params}, llm, embedder, qa_cfg)
        import tiktoken
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def index_params(self) -> dict:
        keys = ["synonym_threshold", "openie_max_tokens", "ner_max_tokens", "seed"]
        return {k: self.params[k] for k in keys}

    # ------------------------------------------------------------------ build
    def _extract(self, passage_text: str) -> tuple[list[str], list[tuple[str, str, str]], int]:
        """NER then post-NER OpenIE for one passage. Returns (entities, triples, n_calls)."""
        ner_sys, ner_user = passage_ner_prompt(passage_text)
        ner_raw = self.llm.generate(ner_user, system=ner_sys,
                                    max_tokens=self.params["ner_max_tokens"], phase="index")
        entities = parse_entities(ner_raw)
        oie_sys, oie_user = openie_prompt(passage_text, entities)
        oie_raw = self.llm.generate(oie_user, system=oie_sys,
                                    max_tokens=self.params["openie_max_tokens"], phase="index")
        triples = parse_triples(oie_raw)
        return entities, triples, 2

    def build_index(self, passages):
        if not passages:
            raise ValueError("HippoRAG: empty corpus")
        t0 = time.perf_counter()
        pid_text = {p.pid: (f"{p.title}: {p.text}" if p.title else p.text) for p in passages}

        parse_failures = 0
        results: list[tuple[str, list[str], list[tuple[str, str, str]]]] = [None] * len(passages)

        def work(i_p):
            i, p = i_p
            entities, triples, _ = self._extract(pid_text[p.pid])
            return i, p.pid, entities, triples

        # Extract triples from each passage
        workers = max(1, int(self.params["index_workers"]))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, pid, entities, triples in pool.map(work, enumerate(passages)):
                if not entities and not triples:
                    parse_failures += 1
                results[i] = (pid, entities, triples)

        nodes, node_id, rows, cols, weights, node_pids = build_graph(results)
        num_relation_edges = len(rows)

        # Synonym edges: embed entity strings, link near-duplicates above threshold.
        num_synonym_edges = 0
        node_emb = np.zeros((0, 0), dtype=np.float32)
        if nodes:
            node_emb = np.asarray(self.embedder.embed(nodes, phase="index"), dtype=np.float32)
            num_synonym_edges = self._add_synonym_edges(
                nodes, node_emb, rows, cols, weights,
                threshold=self.params["synonym_threshold"])

        stats = {
            "num_passages": len(passages),
            "num_nodes": len(nodes),
            "num_relation_edges": num_relation_edges,
            "num_synonym_edges": num_synonym_edges,
            "parse_failures": parse_failures,
            "ner_openie_calls": 2 * len(passages),
            "build_time_s": round(time.perf_counter() - t0, 2),
        }
        return GraphIndex(pids=[p.pid for p in passages], nodes=nodes, node_id=node_id,
                          rows=rows, cols=cols, weights=weights, node_pids=node_pids,
                          node_emb=node_emb, pid_text=pid_text, stats=stats)

    @staticmethod
    def _add_synonym_edges(nodes, node_emb, rows, cols, weights, *, threshold) -> int:
        """Add an edge between entity nodes whose embeddings are >= threshold cosine.
        node_emb is L2-normalized, so the dot product is cosine. Mutates rows/cols/
        weights in place and returns the number of synonym edges added."""
        n = len(nodes)
        if n < 2:
            return 0
        existing = {(a, b) if a < b else (b, a) for a, b in zip(rows, cols)}
        sims = node_emb @ node_emb.T
        added = 0
        iu = np.triu_indices(n, k=1)
        for a, b in zip(*iu):
            s = float(sims[a, b])
            if s >= threshold and (a, b) not in existing:
                rows.append(int(a))
                cols.append(int(b))
                weights.append(s)
                added += 1
        return added

    # ---------------------------------------------------------------- answer
    def _link_query_entities(self, index: GraphIndex, q_entities: list[str]) -> list[int]:
        """Map query entities to graph nodes by embedding similarity."""
        if not q_entities or index.node_emb.shape[0] == 0:
            return []
        q = np.asarray(self.embedder.embed(q_entities), dtype=np.float32)
        sims = q @ index.node_emb.T                      # (num_q, num_nodes)
        k = max(1, int(self.params["link_top_k"]))
        seeds: list[int] = []
        for row in sims:
            top = np.argsort(-row)[:k]
            seeds.extend(int(i) for i in top)
        return list(dict.fromkeys(seeds))

    def answer(self, ex, index: GraphIndex):
        q_sys, q_user = query_ner_prompt(ex.question)
        q_raw = self.llm.generate(q_user, system=q_sys,
                                  max_tokens=self.params["ner_max_tokens"], phase="query")
        q_entities = parse_entities(q_raw)
        seed_ids = self._link_query_entities(index, q_entities)

        ppr = personalized_pagerank(index, seed_ids, alpha=self.params["ppr_damping"],
                                    max_iter=self.params["ppr_max_iter"],
                                    tol=self.params["ppr_tol"])
        scores = score_passages(ppr, index.node_pids, index.pids)

        ranked = sorted(index.pids, key=lambda p: scores.get(p, 0.0), reverse=True)
        top = [p for p in ranked if scores.get(p, 0.0) > 0][: self.params["top_k"]]
        expanded = sorted(p for p, s in scores.items() if s > 0)

        # only passages that actually made it into the context count as retrieved
        context, top, truncated = self._build_context(index, top)
        pred, raw = self.generate_answer(ex.question, context or None)
        return {
            "prediction": pred,
            "raw": raw,
            "retrieved_pids": top,
            "retrieved_pids_expanded": expanded,
            "truncated_passage": truncated,
            "num_query_entities": len(q_entities),
            "num_seed_nodes": len(seed_ids),
            "context_tokens": len(self.tokenizer.encode(context)) if context else 0,
        }

    def _build_context(self, index: GraphIndex, pids: list[str]):
        """Highest-PPR passages until the token budget runs out. A passage longer than
        the whole budget (MultiHop-RAG articles) is cut to fit, so the budget holds for
        every method. Returns (context, pids in the context, truncated pid or None)."""
        budget = self.params["max_context_tokens"]
        texts, chosen, used, truncated = [], [], 0, None
        for pid in pids:
            text = index.pid_text.get(pid, "")
            if not text:
                continue
            toks = self.tokenizer.encode(text)
            if used + len(toks) > budget:
                if not chosen:  # first passage alone is over budget: keep its start
                    texts.append(self.tokenizer.decode(toks[:budget]))
                    chosen.append(pid)
                    truncated = pid
                break
            texts.append(text)
            chosen.append(pid)
            used += len(toks)
        return "\n\n".join(texts), chosen, truncated
