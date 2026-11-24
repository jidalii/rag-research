"""Run experiments from a TOML config.

    python -m ragbench.run configs/musique_pilot.toml
    python -m ragbench.run configs/musique_pilot.toml --llm gpt-4o-mini --method raptor --limit 20
    python -m ragbench.run configs/musique_pilot.toml --dry-run

Every ([[llms]] x [[methods]]) pair is one experiment, written to
    <output_dir>/<run.name>/<dataset>__<method label>__<llm label>/
        predictions.jsonl   one line per question (resumable: finished qids are skipped)
        summary.json        metrics + token usage + $ cost
and all summaries are collected into <output_dir>/<run.name>/summary.csv.
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
import tomllib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from .datasets import load_dataset
from .llm import LLMConfig, Usage, UsageTracker, make_llm, usage_delta
from .methods import get_method
from .metrics import answer_metrics, retrieval_recall

log = logging.getLogger("ragbench")


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
    index = method.build_index(passages)
    usage = usage_delta(tracker.snapshot(), before)
    stats = getattr(index, "stats", {})
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


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


# --------------------------------------------------------------------------- #
def run_one(cfg: dict, llm_cfg: LLMConfig, mcfg: dict, examples, corpus, embedder_holder,
            out_root: Path, workers: int):
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

    # ---------------- index
    # a method may override the dataset's corpus scope (e.g. corpus = "per_question")
    scope = mcfg.get("corpus", ds_cfg.get("corpus", "pooled"))
    if scope == "per_question" and not all(ex.passages for ex in examples):
        raise ValueError("corpus = 'per_question' needs per-question passages (MuSiQue)")
    index_usage, index_stats, cache_hits = [], [], 0
    shared_index, per_q_index = None, {}
    if Method.needs_index:
        cache_dir = Path(cfg["run"].get("index_dir", "indexes")) / ds_cfg["name"] / mname
        key_parts = {"method": mname, "params": method.index_params(),
                     "llm": [llm_cfg.provider, llm_cfg.model, llm_cfg.effort],
                     "embedder": getattr(embedder, "key", None), "v": 1}
        use_cache = cfg["run"].get("cache_indexes", True)
        if scope == "per_question":
            # all examples (not only todo) so index cost is reported for the whole run
            for ex in tqdm(examples, desc="index (per question)"):
                idx, u, st, hit = load_or_build_index(
                    method, ex.passages, cache_dir=cache_dir / llm_cfg.label,
                    key_parts=key_parts, tracker=tracker, use_cache=use_cache)
                if ex.qid not in done:
                    per_q_index[ex.qid] = idx
                index_usage.append(u); index_stats.append(st); cache_hits += hit
        else:
            print(f"building index over {len(corpus)} passages ...")
            t0 = time.time()
            shared_index, u, st, hit = load_or_build_index(
                method, corpus, cache_dir=cache_dir / llm_cfg.label,
                key_parts=key_parts, tracker=tracker, use_cache=use_cache)
            index_usage.append(u); index_stats.append(st); cache_hits += hit
            print(f"index {'loaded from cache' if hit else 'built'} in {time.time()-t0:.1f}s: "
                  f"{json.dumps(st.get('nodes_per_layer', {}))}")

    # ---------------- answer
    def work(ex):
        idx = per_q_index.get(ex.qid, shared_index)
        rec = {"qid": ex.qid, "question": ex.question, "answers": ex.answers,
               "meta": ex.meta}
        with tracker.capture() as cap:
            try:
                out = method.answer(ex, idx)
            except Exception as e:  # keep going; failed qids are retried on rerun
                log.exception("question %s failed", ex.qid)
                rec["error"] = repr(e)
                out = None
        rec["usage"] = {p: asdict(u) for p, u in cap.items()}
        if out is None:
            return rec
        rec.update(out)
        rec["metrics"] = answer_metrics(out["prediction"], ex.answers)
        rec["metrics"]["retrieval_recall"] = retrieval_recall(
            out.get("retrieved_pids"), ex.supporting_pids)
        if "retrieved_pids_expanded" in out:
            rec["metrics"]["retrieval_recall_expanded"] = retrieval_recall(
                out["retrieved_pids_expanded"], ex.supporting_pids)
        return rec

    with open(pred_path, "a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, ex) for ex in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="answer"):
            rec = fut.result()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if not rec.get("error"):
                done[rec["qid"]] = rec

    # ---------------- summary
    recs = [done[ex.qid] for ex in examples if ex.qid in done]
    n_err = len(examples) - len(recs)
    metric_names = sorted({k for r in recs for k in r["metrics"]})
    metrics = {m: _mean(r["metrics"].get(m) for r in recs) for m in metric_names}
    by_hops = defaultdict(list)
    for r in recs:
        by_hops[str(r["meta"].get("hops"))].append(r)
    metrics_by_hops = {h: {"n": len(rs), **{m: _mean(r["metrics"].get(m) for r in rs)
                                            for m in ("f1", "em", "retrieval_recall")}}
                       for h, rs in sorted(by_hops.items())}
    by_type = defaultdict(list)
    for r in recs:
        by_type[str(r["meta"].get("type"))].append(r)
    metrics_by_type = {t: {"n": len(rs), **{m: _mean(r["metrics"].get(m) for r in rs)
                                            for m in ("f1", "em", "retrieval_recall")}}
                       for t, rs in sorted(by_type.items())}
    query_usage = _sum_usage(r.get("usage") for r in recs)
    idx_usage = _sum_usage(index_usage)
    cost = {
        "index_usd": llm_cfg.cost_usd(idx_usage.get("index", {})),
        "query_usd": llm_cfg.cost_usd(query_usage.get("query", {})),
    }
    if cost["index_usd"] is not None and cost["query_usd"] is not None:
        cost["total_usd"] = round(cost["index_usd"] + cost["query_usd"], 6)
        cost["query_usd_per_question"] = round(cost["query_usd"] / max(1, len(recs)), 8)
    summary = {
        "dataset": ds_cfg["name"], "corpus_scope": scope, "method": mlabel,
        "method_params": method.params, "llm": llm_cfg.label, "model": llm_cfg.model,
        "provider": llm_cfg.provider, "n_questions": len(examples), "n_answered": len(recs),
        "n_errors": n_err, "corpus_passages": len(corpus) if scope != "per_question" else None,
        "metrics": metrics, "metrics_by_hops": metrics_by_hops,
        "metrics_by_type": metrics_by_type,
        "usage": {"index": idx_usage, "query": query_usage},
        "index_cache_hits": cache_hits,
        "index_stats": index_stats[0] if len(index_stats) == 1 else
        {"n_indexes": len(index_stats)},
        "cost": cost,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"metrics": metrics, "cost": cost}, indent=1))
    if n_err:
        print(f"WARNING: {n_err} questions failed; rerun the same command to retry them.")
    return summary


def write_csv(out_root: Path):
    rows = []
    for p in sorted(out_root.glob("*/summary.json")):
        s = json.loads(p.read_text())
        u_idx = s["usage"]["index"].get("index", {})
        u_q = s["usage"]["query"].get("query", {})
        rows.append({
            "dataset": s["dataset"], "method": s["method"], "llm": s["llm"],
            "n": s["n_answered"], "errors": s["n_errors"],
            **{k: s["metrics"].get(k) for k in
               ("f1", "em", "answer_recall", "contains", "retrieval_recall",
                "retrieval_recall_expanded")},
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


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--llm", action="append", help="only these llm labels (repeatable)")
    ap.add_argument("--method", action="append", help="only these method labels (repeatable)")
    ap.add_argument("--limit", type=int, help="override dataset.limit")
    ap.add_argument("--workers", type=int, help="override run.max_workers")
    ap.add_argument("--dry-run", action="store_true", help="load data and print the plan only")
    args = ap.parse_args(argv)

    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    cfg = tomllib.loads(Path(args.config).read_text())
    run = cfg.setdefault("run", {})
    if args.limit is not None:
        cfg["dataset"]["limit"] = args.limit
    llms = [LLMConfig.from_dict(d) for d in cfg.get("llms", [])]
    methods = cfg.get("methods", [])
    if args.llm:
        llms = [l for l in llms if l.label in args.llm]
    if args.method:
        methods = [m for m in methods if m.get("label", m["name"]) in args.method]
    if not llms or not methods:
        sys.exit("no [[llms]] / [[methods]] selected")
    for m in methods:
        get_method(m["name"])  # fail fast on typos

    examples, corpus = load_dataset(cfg["dataset"], seed=run.get("seed", 0))
    out_root = Path(run.get("output_dir", "results")) / run.get("name", Path(args.config).stem)
    print(f"dataset={cfg['dataset']['name']} questions={len(examples)} "
          f"corpus_passages={len(corpus)} scope={cfg['dataset'].get('corpus', 'pooled')}")
    print("experiments:", [f"{m.get('label', m['name'])} x {l.label}" for l in llms
                           for m in methods])
    if args.dry_run:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.toml").write_text(Path(args.config).read_text())

    _emb = {}

    def embedder_holder(tracker):  # one embedding model shared by all experiments
        from .embeddings import Embedder
        if "e" not in _emb:
            _emb["e"] = Embedder(cfg.get("embedding", {}), tracker)
        _emb["e"].tracker = tracker
        return _emb["e"]

    workers = args.workers or run.get("max_workers", 8)
    for llm_cfg in llms:
        for mcfg in methods:
            run_one(cfg, llm_cfg, mcfg, examples, corpus, embedder_holder, out_root, workers)
    write_csv(out_root)


if __name__ == "__main__":
    main()
