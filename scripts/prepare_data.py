"""Download, verify and validate the benchmark datasets, and write shared question
subsets so every team member evaluates on exactly the same questions.

    .venv/bin/python scripts/prepare_data.py all
    .venv/bin/python scripts/prepare_data.py musique --splits dev train
    .venv/bin/python scripts/prepare_data.py multihop_rag
    .venv/bin/python scripts/prepare_data.py all --verify          # offline re-check
    .venv/bin/python scripts/prepare_data.py musique --subset pilot --n 100 --seed 0

Layout written:
    datasets/musique/musique_ans_v1.0_{dev,train}.jsonl
    datasets/multihop_rag/{corpus.json, MultiHopRAG.json}
    datasets/<name>/manifest.json            source, revision, sha256, stats
    datasets/<name>/subsets/<name>.txt       one question id per line

Use a subset from a config with:   [dataset] ids_file = "datasets/musique/subsets/pilot.txt"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "datasets"

# --------------------------------------------------------------------------- #
# Pinned sources
# --------------------------------------------------------------------------- #
MUSIQUE = {
    "source": "https://github.com/StonyBrookNLP/musique (official Google Drive release)",
    "gdrive_id": "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h",
    "zip_name": "musique_v1.0.zip",
    "files": {  # split -> (file in zip, sha256, expected lines)
        "dev": ("data/musique_ans_v1.0_dev.jsonl",
                "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b", 2417),
        "train": ("data/musique_ans_v1.0_train.jsonl",
                  "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a", 19938),
    },
}
MHR_REV = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
MULTIHOP_RAG = {
    "source": f"https://huggingface.co/datasets/yixuantt/MultiHopRAG @ {MHR_REV}",
    "files": {
        "corpus.json": "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
        "MultiHopRAG.json": "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
    },
    "url": "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/{rev}/{name}",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_sha(path: Path, expected: str | None) -> str:
    got = sha256(path)
    if expected and got != expected:
        raise SystemExit(f"checksum mismatch for {path}:\n  expected {expected}\n  got      {got}")
    return got


def download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def write_manifest(dest: Path, manifest: dict) -> None:
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# MuSiQue
# --------------------------------------------------------------------------- #
def prepare_musique(splits: list[str], verify_only: bool) -> dict:
    dest = DATA / "musique"
    dest.mkdir(parents=True, exist_ok=True)
    missing = [s for s in splits if not (dest / Path(MUSIQUE["files"][s][0]).name).exists()]
    if missing and verify_only:
        raise SystemExit(f"musique: missing splits {missing}; run without --verify")
    if missing:
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / MUSIQUE["zip_name"]
            gdown = shutil.which("gdown") or str(Path(sys.executable).parent / "gdown")
            print(f"musique: downloading {MUSIQUE['zip_name']} (~272 MB)")
            subprocess.run([gdown, MUSIQUE["gdrive_id"], "-O", str(zpath)], check=True)
            with zipfile.ZipFile(zpath) as z:
                for s in missing:
                    member = MUSIQUE["files"][s][0]
                    with z.open(member) as src, open(dest / Path(member).name, "wb") as out:
                        shutil.copyfileobj(src, out)
    manifest = {"source": MUSIQUE["source"], "files": {}}
    for s in splits:
        member, expected, n_expected = MUSIQUE["files"][s]
        path = dest / Path(member).name
        digest = check_sha(path, expected)
        n = sum(1 for line in open(path) if line.strip())
        if n != n_expected:
            raise SystemExit(f"musique {s}: {n} lines, expected {n_expected}")
        manifest["files"][path.name] = {"split": s, "sha256": digest, "questions": n}
    manifest["stats"] = validate_musique(dest / "musique_ans_v1.0_dev.jsonl") \
        if "dev" in splits else {}
    write_manifest(dest, manifest)
    return manifest


def validate_musique(path: Path) -> dict:
    from ragbench.datasets import load_dataset
    ex, corpus = load_dataset({"name": "musique", "path": str(path), "corpus": "full"})
    pids = {p.pid for p in corpus}
    # 2401/2417 dev questions have 20 paragraphs, the rest 17-19
    assert all(17 <= len(e.passages) <= 20 for e in ex), "expected 17-20 paragraphs"
    assert all(len(e.supporting_pids) == e.meta["hops"] for e in ex), "1 gold paragraph per hop"
    assert all(s in pids for e in ex for s in e.supporting_pids)
    assert len({e.qid for e in ex}) == len(ex), "duplicate question ids"
    stats = {
        "questions": len(ex),
        "paragraphs_per_question": dict(sorted(Counter(str(len(e.passages))
                                                       for e in ex).items())),
        "unique_paragraphs_full_split": len(corpus),
        "hops": dict(sorted(Counter(str(e.meta["hops"]) for e in ex).items())),
        "types": dict(sorted(Counter(e.meta["type"] for e in ex).items())),
    }
    print(f"musique dev ok: {json.dumps(stats)}")
    return stats


# --------------------------------------------------------------------------- #
# MultiHop-RAG
# --------------------------------------------------------------------------- #
def prepare_multihop_rag(verify_only: bool) -> dict:
    dest = DATA / "multihop_rag"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"source": MULTIHOP_RAG["source"], "files": {}}
    for name, expected in MULTIHOP_RAG["files"].items():
        path = dest / name
        if not path.exists():
            if verify_only:
                raise SystemExit(f"multihop_rag: missing {path}; run without --verify")
            download(MULTIHOP_RAG["url"].format(rev=MHR_REV, name=name), path)
        manifest["files"][name] = {"sha256": check_sha(path, expected)}
    manifest["stats"] = validate_multihop_rag(dest)
    write_manifest(dest, manifest)
    return manifest


def validate_multihop_rag(dest: Path) -> dict:
    from ragbench.datasets import load_dataset
    ex, corpus = load_dataset({"name": "multihop_rag", "path": str(dest)})
    pids = {p.pid for p in corpus}
    assert len(corpus) == 609, f"expected 609 articles, got {len(corpus)}"
    assert all(s in pids for e in ex for s in e.supporting_pids), "evidence not in corpus"
    assert len({e.qid for e in ex}) == len(ex), "duplicate question ids"
    stats = {
        "questions": len(ex),
        "articles": len(corpus),
        "types": dict(sorted(Counter(e.meta["type"] for e in ex).items())),
        "evidence_articles": dict(sorted(Counter(str(len(e.supporting_pids))
                                                 for e in ex).items())),
    }
    print(f"multihop_rag ok: {json.dumps(stats)}")
    return stats


# --------------------------------------------------------------------------- #
# Shared subsets
# --------------------------------------------------------------------------- #
def make_subset(name: str, subset: str, n: int, seed: int, stratify: bool,
                exclude_types: list[str]) -> Path:
    from ragbench.datasets import load_dataset
    if name == "musique":
        cfg = {"name": "musique", "path": str(DATA / "musique/musique_ans_v1.0_dev.jsonl")}
    else:
        cfg = {"name": "multihop_rag", "path": str(DATA / "multihop_rag")}
    ex, _ = load_dataset(cfg)
    ex = [e for e in ex if e.meta.get("type") not in set(exclude_types)]
    rng = random.Random(seed)
    if stratify:  # proportional per question type, so small samples keep the mix
        groups: dict[str, list] = {}
        for e in ex:
            groups.setdefault(str(e.meta.get("type")), []).append(e)
        picked = []
        for t, items in sorted(groups.items()):
            k = round(n * len(items) / len(ex))
            picked += rng.sample(items, min(k, len(items)))
        taken = {e.qid for e in picked}
        rest = [e for e in ex if e.qid not in taken]
        picked += rng.sample(rest, max(0, n - len(picked)))
        picked = picked[:n]
    else:
        picked = rng.sample(ex, min(n, len(ex)))
    ids = sorted(e.qid for e in picked)
    out = DATA / name / "subsets" / f"{subset}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# {name} subset '{subset}': n={len(ids)} seed={seed} stratify={stratify} "
              f"exclude_types={exclude_types}\n")
    out.write_text(header + "\n".join(ids) + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(ids)} ids, "
          f"types {dict(Counter(str(e.meta.get('type')) for e in picked))})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["musique", "multihop_rag", "all"])
    ap.add_argument("--splits", nargs="+", default=["dev"], choices=["dev", "train"],
                    help="MuSiQue splits (test has no answers, so it is not used)")
    ap.add_argument("--verify", action="store_true", help="no downloads; verify existing files")
    ap.add_argument("--subset", help="also write a question-id subset with this name")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-stratify", action="store_true")
    ap.add_argument("--exclude-types", nargs="*", default=[],
                    help="e.g. null_query for MultiHop-RAG")
    args = ap.parse_args()

    names = ["musique", "multihop_rag"] if args.dataset == "all" else [args.dataset]
    for name in names:
        if name == "musique":
            prepare_musique(args.splits, args.verify)
        else:
            prepare_multihop_rag(args.verify)
        if args.subset:
            make_subset(name, args.subset, args.n, args.seed, not args.no_stratify,
                        args.exclude_types)


if __name__ == "__main__":
    main()
