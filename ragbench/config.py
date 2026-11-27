"""Experiment configs: YAML (preferred) or TOML.

Two additions over a plain config:

  extends: [configs/common/llms.yaml]    shared sections merged in first; the file
                                         itself wins (dicts merge, lists replace)
  methods:
    - name: vanilla_rag
      sweep: {max_context_tokens: [1000, 2000, 4000]}
                                         one arm per value: vanilla_rag@1k, @2k, @4k
"""
from __future__ import annotations

import copy
import itertools
import tomllib
from pathlib import Path

import yaml


def _read(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text) or {}
    if path.suffix == ".toml":
        return tomllib.loads(text)
    raise ValueError(f"config must be .yaml/.yml/.toml: {path}")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(ref: str, here: Path) -> Path:
    p = Path(ref)
    for cand in (p, here.parent / p):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"extends: {ref} (from {here})")


def _short(key: str, val) -> str:
    if key == "max_context_tokens" and isinstance(val, int) and val % 1000 == 0:
        return f"{val // 1000}k"
    return f"{key}={val}"


def expand_sweeps(methods: list[dict]) -> list[dict]:
    out = []
    for m in methods:
        sweep = m.get("sweep")
        if not sweep:
            out.append(m)
            continue
        base = {k: v for k, v in m.items() if k != "sweep"}
        keys = list(sweep)
        for combo in itertools.product(*(sweep[k] for k in keys)):
            arm = dict(base, **dict(zip(keys, combo)))
            tag = ",".join(_short(k, v) for k, v in zip(keys, combo))
            arm["label"] = f"{base.get('label', base['name'])}@{tag}"
            out.append(arm)
    labels = [m.get("label", m["name"]) for m in out]
    dup = {l for l in labels if labels.count(l) > 1}
    if dup:
        raise ValueError(f"duplicate method labels {sorted(dup)}: give them a `label`")
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = _read(path)
    merged: dict = {}
    for ref in cfg.pop("extends", []) or []:
        merged = _merge(merged, load_config(_resolve(ref, path)))
    cfg = _merge(merged, cfg)
    if "methods" in cfg:
        cfg["methods"] = expand_sweeps(cfg["methods"])
    return cfg
