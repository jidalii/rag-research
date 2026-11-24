"""Dataset loaders. Every dataset is normalized to (examples, corpus).

Example.supporting_pids are the gold evidence passage ids used for retrieval recall.
Passage ids are content hashes, so the same paragraph appearing under several
MuSiQue questions collapses to a single corpus entry.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Passage:
    pid: str
    title: str
    text: str


@dataclass
class Example:
    qid: str
    question: str
    answers: list[str]                       # gold answer + aliases
    supporting_pids: list[str]
    passages: list[Passage] = field(default_factory=list)  # per-question context (MuSiQue)
    meta: dict = field(default_factory=dict)


def _pid(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:16]


def _read_json_or_jsonl(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    return json.loads(text)


def _read_ids(path: str) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def _select(items: list, cfg: dict, seed: int, key) -> list:
    """ids_file (shared subset) first, then limit/sample."""
    if cfg.get("ids_file"):
        wanted = _read_ids(cfg["ids_file"])
        by_id = {key(x): x for x in items}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise ValueError(f"{len(missing)} ids from {cfg['ids_file']} not in dataset, "
                             f"e.g. {missing[:3]}")
        items = [by_id[i] for i in wanted]
    return _sample(items, cfg.get("limit"), cfg.get("sample", "first"), seed)


def _sample(items: list, limit: int | None, sample: str, seed: int) -> list:
    if not limit or limit >= len(items):
        return items
    if sample == "random":
        return random.Random(seed).sample(items, limit)
    return items[:limit]


def _dedup(passages) -> list[Passage]:
    seen, out = set(), []
    for p in passages:
        if p.pid not in seen:
            seen.add(p.pid)
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
def load_musique(cfg: dict, seed: int):
    """MuSiQue (Trivedi et al. 2022), e.g. musique_ans_v1.0_dev.jsonl."""
    rows = _read_json_or_jsonl(Path(cfg["path"]))
    if cfg.get("answerable_only", True):
        rows = [r for r in rows if r.get("answerable", True)]
    all_rows = rows
    rows = _select(rows, cfg, seed, key=lambda r: r["id"])

    def to_example(r) -> Example:
        passages, support = [], []
        for p in r["paragraphs"]:
            pas = Passage(_pid(p["title"], p["paragraph_text"]), p["title"], p["paragraph_text"])
            passages.append(pas)
            if p.get("is_supporting"):
                support.append(pas.pid)
        hops = len(r.get("question_decomposition", [])) or None
        return Example(
            qid=r["id"], question=r["question"],
            answers=[r["answer"], *r.get("answer_aliases", [])],
            supporting_pids=support, passages=passages,
            meta={"hops": hops, "type": r["id"].split("__")[0]},
        )

    examples = [to_example(r) for r in rows]
    scope = cfg.get("corpus", "pooled")
    if scope == "full":
        corpus = _dedup(p for r in all_rows for p in to_example(r).passages)
    else:  # pooled (and per_question, where the runner uses ex.passages instead)
        corpus = _dedup(p for ex in examples for p in ex.passages)
    return examples, corpus


def load_multihop_rag(cfg: dict, seed: int):
    """MultiHop-RAG (Tang & Yang 2024): corpus.json + MultiHopRAG.json.

    Evidence is matched to corpus articles by title, so retrieval recall is
    article-level.
    """
    root = Path(cfg["path"])
    corpus_rows = _read_json_or_jsonl(root / cfg.get("corpus_file", "corpus.json"))
    query_rows = _read_json_or_jsonl(root / cfg.get("queries_file", "MultiHopRAG.json"))
    corpus = _dedup(Passage(_pid(d["title"]), d["title"], d["body"]) for d in corpus_rows)

    # stable ids from the position in the released file (before any filtering)
    for i, q in enumerate(query_rows):
        q.setdefault("id", f"mhr_{i:04d}")
    skip = set(cfg.get("skip_question_types", []))  # e.g. ["null_query"]
    query_rows = [q for q in query_rows if q.get("question_type") not in skip]
    query_rows = _select(query_rows, cfg, seed, key=lambda q: q["id"])
    examples = []
    for q in query_rows:
        support = sorted({_pid(e["title"]) for e in q.get("evidence_list", [])})
        examples.append(Example(
            qid=q["id"], question=q["query"], answers=[q["answer"]],
            supporting_pids=support,
            meta={"type": q.get("question_type"), "hops": len(support) or None},
        ))
    return examples, corpus


LOADERS = {"musique": load_musique, "multihop_rag": load_multihop_rag}


def load_dataset(cfg: dict, seed: int = 0):
    name = cfg["name"]
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset {name!r}; available: {list(LOADERS)}")
    return LOADERS[name](cfg, seed)
