"""Run experiments from a YAML (or TOML) config.

    python -m ragbench.run configs/experiments/musique_main.yaml
    python -m ragbench.run configs/experiments/musique_main.yaml --llm gpt-4o-mini --method raptor@2k --limit 20
    python -m ragbench.run configs/experiments/musique_main.yaml --dry-run

Every (llm x method arm) pair is one experiment, written to
    <output_dir>/<run.name>/<dataset>__<method label>__<llm label>/
        predictions.jsonl   one line per question (resumable: finished qids are skipped)
        summary.json        metrics + token usage + $ cost
and all summaries are collected into <output_dir>/<run.name>/summary.csv.
Judge verdicts are added afterwards by `python -m ragbench.judge`.

`run.index_llm` (an llm label) makes that LLM build every index; the indexes are
cached once and shared by all readers, and their cost is reported against it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from .config import load_config
from .datasets import load_dataset
from .llm import LLMConfig, Usage, UsageTracker, make_llm, usage_delta
from .methods import get_method
from .metrics import (all_gold_retrieved, answer_metrics, fact_recall, recall_by_hop,
                      retrieval_recall)

log = logging.getLogger("ragbench")
_ENC = None


def _tokens(text: str) -> int:
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text))


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Index cache: the pickle plus a JSON sidecar with the usage it cost to build,
# so a cached index still reports its full build cost.
# --------------------------------------------------------------------------- #
def load_or_build_index(method, passages, *, cache_dir: Path, key_parts: dict,
                        tracker: UsageTracker, use_cache: bool):
    key = _hash({**key_parts, "pids": sorted(p.pid for p in passages)})
    pkl, meta = cache_dir / f"{key}.pkl", cache_dir / f"{key}.json"
    if use_cache and pkl.exists() and meta.exists():
        with open(pkl, "rb") as f:
            index = pickle.load(f)
        info = json.loads(meta.read_text())
        return index, info["usage"], info.get("stats", {}), True
    before = tracker.snapshot()
    t0 = time.time()
    index = method.build_index(passages)
    usage = usage_delta(tracker.snapshot(), before)
    stats = dict(getattr(index, "stats", {}) or {})
    stats.setdefault("build_time_s", round(time.time() - t0, 2))
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(pkl, "wb") as f:
        pickle.dump(index, f)
    meta.write_text(json.dumps({"key": key_parts, "usage": usage, "stats": stats}, indent=1))
    return index, usage, stats, False


def _sum_usage(dicts) -> dict:
    tot: dict[str, Usage] = defaultdict(Usage)
    for d in dicts:
        for phase, u in (d or {}).items():
            tot[phase].add(Usage(**u))
    return {p: asdict(u) for p, u in tot.items()}


class _LazyLLM:
    """Builds the real client the first time it is used."""

    def __init__(self, cfg, tracker):
        self._cfg, self._tracker, self._client = cfg, tracker, None

    def __getattr__(self, name):
        if self._client is None:
            self._client = make_llm(self._cfg, self._tracker)
        return getattr(self._client, name)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


# --------------------------------------------------------------------------- #
# Per-question scoring and run-level summaries (shared with ragbench.judge)
# --------------------------------------------------------------------------- #
def score_record(rec: dict, out: dict, ex, contexts: list[str]) -> None:
    m = answer_metrics(out["prediction"], ex.answers)
    retrieved = out.get("retrieved_pids")
    m["retrieval_recall"] = retrieval_recall(retrieved, ex.supporting_pids)
    m["all_gold_retrieved"] = all_gold_retrieved(retrieved, ex.supporting_pids)
    if "retrieved_pids_expanded" in out:
        m["retrieval_recall_expanded"] = retrieval_recall(out["retrieved_pids_expanded"],
                                                          ex.supporting_pids)
    gold_hop = ex.meta.get("gold_hop") or {}
    rec["recall_by_hop"] = recall_by_hop(retrieved, gold_hop)
    if "retrieved_pids_expanded" in out:
        rec["recall_by_hop_expanded"] = recall_by_hop(out["retrieved_pids_expanded"], gold_hop)
    context = "\n\n".join(contexts)
    if ex.meta.get("facts") and retrieved is not None:
        m["fact_recall"] = fact_recall(context, ex.meta["facts"])
    rec["context_tokens"] = sum(_tokens(c) for c in contexts)
    rec["metrics"] = m


GROUP_METRICS = ("judge", "f1", "em", "retrieval_recall", "all_gold_retrieved")


def _group(recs, key) -> dict:
    groups = defaultdict(list)
    for r in recs:
        k = key(r)
        if k is not None:
            groups[str(k)].append(r)
    return {k: {"n": len(rs), **{m: _mean(r["metrics"].get(m) for r in rs)
                                 for m in GROUP_METRICS}}
            for k, rs in sorted(groups.items())}


def summarize_records(recs: list[dict]) -> dict:
    """Metric means overall and by hops / type / gold-retrieved flag / hop position."""
    names = sorted({k for r in recs for k in r["metrics"]})
    hop_recall = defaultdict(list)
    hop_recall_exp = defaultdict(list)
    for r in recs:
        for h, v in (r.get("recall_by_hop") or {}).items():
            hop_recall[h].append(v)
        for h, v in (r.get("recall_by_hop_expanded") or {}).items():
            hop_recall_exp[h].append(v)
    return {
        "metrics": {m: _mean(r["metrics"].get(m) for r in recs) for m in names},
        "metrics_by_hops": _group(recs, lambda r: r["meta"].get("hops")),
        "metrics_by_type": _group(recs, lambda r: r["meta"].get("type")),
        "metrics_by_gold_retrieved": _group(
            recs, lambda r: {1.0: "all", 0.0: "missing"}.get(
                r["metrics"].get("all_gold_retrieved"))),
        "recall_by_hop_position": {h: _mean(v) for h, v in sorted(hop_recall.items())},
        "recall_by_hop_position_expanded": {h: _mean(v)
                                            for h, v in sorted(hop_recall_exp.items())},
        "context_tokens_mean": _mean(r.get("context_tokens") for r in recs),
        "latency_s_mean": _mean(r.get("latency_s") for r in recs),
        "answer_truncated": sum(1 for r in recs if r.get("answer_truncated")),
        "empty_predictions": sum(1 for r in recs if not (r.get("prediction") or "").strip()),
    }


# --------------------------------------------------------------------------- #
def run_one(cfg: dict, llm_cfg: LLMConfig, mcfg: dict, examples, corpus, embedder_holder,
            out_root: Path, workers: int, index_llm_cfg: LLMConfig | None):
    ds_cfg, qa_cfg = cfg["dataset"], cfg.get("qa", {})
    mname = mcfg["name"]
    mlabel = mcfg.get("label", mname)
    params = {k: v for k, v in mcfg.items() if k not in ("name", "label", "corpus")}
    out_dir = out_root / f"{ds_cfg['name']}__{mlabel}__{llm_cfg.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    done = {}
    if pred_path.exists():
        for line in pred_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    done[r["qid"]] = r
    todo = [ex for ex in examples if ex.qid not in done]
    print(f"\n=== {out_dir.name}: {len(examples)} questions ({len(done)} already done)")

    tracker = UsageTracker()
    llm = make_llm(llm_cfg, tracker)
    Method = get_method(mname)
    embedder = embedder_holder(tracker) if Method.uses_embedder else None
    method = Method(params, llm, embedder, qa_cfg)

    # ---------------- index (built by index_llm when set, else by the reader)
    builder_cfg = index_llm_cfg or llm_cfg
    idx_tracker = UsageTracker() if index_llm_cfg else tracker
    scope = mcfg.get("corpus", ds_cfg.get("corpus", "pooled"))
    if scope == "per_question" and not all(ex.passages for ex in examples):
        raise ValueError("corpus = 'per_question' needs per-question passages (MuSiQue)")
    index_usage, index_stats, cache_hits = [], [], 0
    shared_index, per_q_index = None, {}
    if Method.needs_index:
        # created on first use, so loading a cached index needs no index_llm API key
        method.llm = _LazyLLM(builder_cfg, idx_tracker) if index_llm_cfg else llm
        cache_dir = (Path(cfg["run"].get("index_dir", "indexes")) / ds_cfg["name"] / mname
                     / builder_cfg.label)
        key_parts = {"method": mname, "params": method.index_params(),
                     "llm": [builder_cfg.provider, builder_cfg.model, builder_cfg.effort],
                     "embedder": getattr(embedder, "key", None), "v": 1}
        use_cache = cfg["run"].get("cache_indexes", True)
        if scope == "per_question":
            # all examples (not only todo) so index cost is reported for the whole run
            for ex in tqdm(examples, desc="index (per question)"):
                idx, u, st, hit = load_or_build_index(
                    method, ex.passages, cache_dir=cache_dir, key_parts=key_parts,
                    tracker=idx_tracker, use_cache=use_cache)
                if ex.qid not in done:
                    per_q_index[ex.qid] = idx
                index_usage.append(u); index_stats.append(st); cache_hits += hit
        else:
            print(f"index over {len(corpus)} passages (built by {builder_cfg.label}) ...")
            t0 = time.time()
            shared_index, u, st, hit = load_or_build_index(
                method, corpus, cache_dir=cache_dir, key_parts=key_parts,
                tracker=idx_tracker, use_cache=use_cache)
            index_usage.append(u); index_stats.append(st); cache_hits += hit
            print(f"index {'loaded from cache' if hit else 'built'} in {time.time()-t0:.1f}s")
        method.llm = llm  # answering always uses the reader

    # ---------------- answer
    def work(ex):
        idx = per_q_index.get(ex.qid, shared_index)
        rec = {"qid": ex.qid, "question": ex.question, "answers": ex.answers,
               "meta": ex.meta}
        method.pop_contexts()
        method.pop_truncated()
        t0 = time.time()
        with tracker.capture() as cap:
            try:
                out = method.answer(ex, idx)
            except Exception as e:  # keep going; failed qids are retried on rerun
                log.warning("question %s failed: %r", ex.qid, e)
                rec["error"] = repr(e)
                out = None
        rec["latency_s"] = round(time.time() - t0, 3)
        rec["usage"] = {p: asdict(u) for p, u in cap.items()}
        contexts = method.pop_contexts()
        rec["answer_truncated"] = method.pop_truncated()
        if out is None:
            return rec
        rec.update(out)
        score_record(rec, out, ex, contexts)
        return rec

    fresh = []
    with open(pred_path, "a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, ex) for ex in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="answer"):
            rec = fut.result()
            fresh.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if not rec.get("error"):
                done[rec["qid"]] = rec

    # ---------------- summary
    recs = [done[ex.qid] for ex in examples if ex.qid in done]
    n_err = len(examples) - len(recs)
    query_usage = _sum_usage(r.get("usage") for r in recs)
    idx_usage = _sum_usage(index_usage)
    cost = {
        "index_usd": builder_cfg.cost_usd(idx_usage.get("index", {})),
        "query_usd": llm_cfg.cost_usd(query_usage.get("query", {})),
    }
    if cost["index_usd"] is not None and cost["query_usd"] is not None:
        cost["total_usd"] = round(cost["index_usd"] + cost["query_usd"], 6)
    if cost["query_usd"] is not None:
        cost["query_usd_per_question"] = round(cost["query_usd"] / max(1, len(recs)), 8)
    summary = {
        "dataset": ds_cfg["name"], "corpus_scope": scope, "method": mlabel,
        "method_params": method.params, "llm": llm_cfg.label, "model": llm_cfg.model,
        "provider": llm_cfg.provider, "index_llm": builder_cfg.label if Method.needs_index
        else None, "n_questions": len(examples), "n_answered": len(recs),
        "n_errors": n_err, "corpus_passages": len(corpus) if scope != "per_question" else None,
        **summarize_records(recs),
        "usage": {"index": idx_usage, "query": query_usage},
        "index_cache_hits": cache_hits,
        "index_stats": index_stats[0] if len(index_stats) == 1 else
        {"n_indexes": len(index_stats)},
        "cost": cost,
        "wall_time_s": round(sum(r.get("latency_s") or 0 for r in recs), 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"metrics": summary["metrics"], "cost": cost}, indent=1))
    if n_err:
        print(f"WARNING: {n_err} questions failed; rerun the same command to retry them.")
    # an account-level error (no credits, bad key) fails every question the same way
    errs = [r["error"] for r in fresh if r.get("error")]
    if fresh and len(errs) == len(fresh) and any(k in errs[0] for k in FATAL_ERRORS):
        summary["fatal"] = errs[0][:200]
    return summary


FATAL_ERRORS = ("insufficient_quota", "no credits remaining", "invalid_api_key",
                "Incorrect API key", "Authentication", "PermissionDenied")

CSV_METRICS = ("judge", "f1", "em", "answer_recall", "contains", "retrieval_recall",
               "retrieval_recall_expanded", "all_gold_retrieved", "fact_recall")


def write_csv(out_root: Path):
    rows = []
    for p in sorted(out_root.glob("*/summary.json")):
        s = json.loads(p.read_text())
        u_idx = s["usage"]["index"].get("index", {})
        u_q = s["usage"]["query"].get("query", {})
        rows.append({
            "dataset": s["dataset"], "method": s["method"], "llm": s["llm"],
            "index_llm": s.get("index_llm"), "n": s["n_answered"], "errors": s["n_errors"],
            **{k: s["metrics"].get(k) for k in CSV_METRICS},
            "context_tokens": s.get("context_tokens_mean"),
            "latency_s": s.get("latency_s_mean"),
            "index_in_tok": u_idx.get("input_tokens", 0),
            "index_out_tok": u_idx.get("output_tokens", 0),
            "query_in_tok": u_q.get("input_tokens", 0),
            "query_out_tok": u_q.get("output_tokens", 0),
            "index_usd": s["cost"].get("index_usd"), "query_usd": s["cost"].get("query_usd"),
            "total_usd": s["cost"].get("total_usd"),
        })
    if rows:
        with open(out_root / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_root / 'summary.csv'}")


def resolve_index_llm(cfg: dict) -> LLMConfig | None:
    ref = cfg.get("run", {}).get("index_llm")
    if not ref:
        return None
    if isinstance(ref, dict):
        return LLMConfig.from_dict(ref)
    for d in cfg.get("llms", []):
        if d.get("label", d.get("model")) == ref:
            return LLMConfig.from_dict(d)
    raise ValueError(f"run.index_llm {ref!r} is not an llm label in the config")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--llm", action="append", help="only these llm labels (repeatable)")
    ap.add_argument("--method", action="append", help="only these method labels (repeatable)")
    ap.add_argument("--limit", type=int, help="override dataset.limit")
    ap.add_argument("--ids-file", help="override dataset.ids_file")
    ap.add_argument("--name", help="override run.name (output folder)")
    ap.add_argument("--workers", type=int, help="override run.max_workers")
    ap.add_argument("--dry-run", action="store_true", help="load data and print the plan only")
    args = ap.parse_args(argv)

    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    run = cfg.setdefault("run", {})
    if args.limit is not None:
        cfg["dataset"]["limit"] = args.limit
    if args.ids_file:
        cfg["dataset"]["ids_file"] = args.ids_file
    if args.name:
        run["name"] = args.name
    # per-experiment tweaks of shared llm entries, e.g. a larger num_ctx for long prompts
    for d in cfg.get("llms", []):
        d.update((cfg.get("llm_overrides") or {}).get(d.get("label", d.get("model")), {}))
    index_llm_cfg = resolve_index_llm(cfg)
    llms = [LLMConfig.from_dict(d) for d in cfg.get("llms", [])]
    methods = cfg.get("methods", [])
    only = args.llm or run.get("llm_labels")   # CLI wins over the config's selection
    if only:
        llms = [l for l in llms if l.label in only]
    if args.method:
        methods = [m for m in methods if m.get("label", m["name"]) in args.method]
    if not llms or not methods:
        sys.exit("no llms / methods selected")
    for m in methods:
        get_method(m["name"])  # fail fast on typos

    examples, corpus = load_dataset(cfg["dataset"], seed=run.get("seed", 0))
    out_root = Path(run.get("output_dir", "results")) / run.get("name", Path(args.config).stem)
    print(f"dataset={cfg['dataset']['name']} questions={len(examples)} "
          f"corpus_passages={len(corpus)} scope={cfg['dataset'].get('corpus', 'pooled')} "
          f"index_llm={index_llm_cfg.label if index_llm_cfg else 'reader'}")
    print(f"{len(llms)} llms x {len(methods)} arms:",
          [m.get("label", m["name"]) for m in methods])
    if args.dry_run:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    _emb = {}

    def embedder_holder(tracker):  # one embedding model shared by all experiments
        from .embeddings import Embedder
        if "e" not in _emb:
            _emb["e"] = Embedder(cfg.get("embedding", {}), tracker)
        _emb["e"].tracker = tracker
        return _emb["e"]

    workers = args.workers or run.get("max_workers", 8)
    for llm_cfg in llms:
        w = min(workers, llm_cfg.max_workers or workers)
        for mcfg in methods:
            s = run_one(cfg, llm_cfg, mcfg, examples, corpus, embedder_holder, out_root, w,
                        index_llm_cfg)
            if s.get("fatal"):
                print(f"SKIPPING the remaining arms of {llm_cfg.label}: {s['fatal']}")
                break
    write_csv(out_root)


if __name__ == "__main__":
    main()
