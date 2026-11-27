"""LLM-as-judge correctness, run after answering.

    python -m ragbench.judge results/musique_main
    python -m ragbench.judge results/musique_main --only musique__raptor@2k__gpt-4o-mini

Reads the `judge:` LLM from <run dir>/config.resolved.yaml, grades every finished
prediction (and every sub-question of llm_only_decomposed), writes judge.jsonl next to
predictions.jsonl, and re-writes summary.json / summary.csv with `judge` accuracy.
Verdicts are cached in <output_dir>/judge_cache.jsonl by (judge model, question, gold,
prediction), so re-running or re-judging identical answers costs nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from .llm import LLMConfig, UsageTracker, make_llm
from .run import summarize_records, write_csv

SYSTEM = (
    "You grade answers to questions. Decide whether the predicted answer means the same "
    "thing as any one of the gold answers. Ignore wording, capitalization, articles, "
    "extra explanation and formatting. It is incorrect if it names a different entity, "
    "number, date or yes/no value, if it hedges between several candidates, or if it "
    "does not answer. If the gold answer says the information is insufficient, the "
    "prediction is correct only if it also says the question cannot be answered."
)
PROMPT = ("Question: {question}\nGold answers: {gold}\nPredicted answer: {pred}\n\n"
          "Reply with exactly one word: CORRECT or INCORRECT.")


def parse_verdict(raw: str) -> float | None:
    t = raw.strip().upper()
    if t.startswith("INCORRECT") or "INCORRECT" in t.split():
        return 0.0
    if t.startswith("CORRECT") or "CORRECT" in t.split():
        return 1.0
    return None


class Judge:
    def __init__(self, cfg: LLMConfig, cache_path: Path):
        self.cfg = cfg
        self.tracker = UsageTracker()
        self.llm = make_llm(cfg, self.tracker)
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.cache[d["key"]] = d
        self.lock = threading.Lock()

    def grade(self, question: str, golds: list[str], pred: str) -> dict:
        key = hashlib.sha1(json.dumps([self.cfg.provider, self.cfg.model, question,
                                       sorted(golds), pred]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        if not (pred or "").strip():
            d = {"key": key, "correct": 0.0, "raw": "<empty prediction>"}
        else:
            raw = self.llm.generate(PROMPT.format(question=question, gold=" | ".join(golds),
                                                  pred=pred),
                                    system=SYSTEM, max_tokens=8, phase="judge")
            d = {"key": key, "correct": parse_verdict(raw), "raw": raw}
        with self.lock:
            self.cache[key] = d
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return d


def judge_dir(exp_dir: Path, judge: Judge, workers: int) -> None:
    recs = {}
    for line in (exp_dir / "predictions.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                recs[r["qid"]] = r
    recs = list(recs.values())

    def one(r):
        v = judge.grade(r["question"], r["answers"], r["prediction"])
        out = {"qid": r["qid"], "correct": v["correct"], "raw": v["raw"]}
        if r.get("subq"):
            out["subq"] = [judge.grade(s["question"], [s["gold"]], s["prediction"])["correct"]
                           for s in r["subq"]]
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(tqdm(pool.map(one, recs), total=len(recs), desc=exp_dir.name))
    (exp_dir / "judge.jsonl").write_text(
        "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in verdicts))

    by_q = {v["qid"]: v for v in verdicts}
    for r in recs:
        v = by_q[r["qid"]]
        r["metrics"]["judge"] = v["correct"]
        if "subq" in v:
            hops = [c for c in v["subq"] if c is not None]
            r["metrics"]["judge_all_hops"] = float(all(hops)) if hops else None
    summary_path = exp_dir / "summary.json"
    s = json.loads(summary_path.read_text())
    s.update(summarize_records(recs))
    s["judge"] = {"model": judge.cfg.model, "provider": judge.cfg.provider,
                  "unparsed": sum(1 for v in verdicts if v["correct"] is None)}
    summary_path.write_text(json.dumps(s, indent=2, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--only", action="append", help="only these experiment folders")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    load_dotenv()
    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    if not cfg.get("judge"):
        raise SystemExit("no `judge:` section in the run's config")
    jcfg = LLMConfig.from_dict({"label": "judge", **cfg["judge"]})
    judge = Judge(jcfg, run_dir.parent / "judge_cache.jsonl")
    for exp in sorted(p.parent for p in run_dir.glob("*/predictions.jsonl")):
        if args.only and exp.name not in args.only:
            continue
        judge_dir(exp, judge, args.workers)
    usage = judge.tracker.to_dict().get("judge", {})
    print(f"judge calls this run: {usage.get('calls', 0)}  "
          f"in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)}  "
          f"usd={jcfg.cost_usd(usage) if usage else 0}")
    write_csv(run_dir)


if __name__ == "__main__":
    main()
