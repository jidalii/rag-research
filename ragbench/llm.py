"""Provider-agnostic chat LLM adapters with token accounting.

Supported providers (set `provider` in a [[llms]] table):
  - "openai"    : OpenAI Chat Completions (e.g. gpt-4o-mini)
  - "deepseek"  : DeepSeek, OpenAI-compatible endpoint (https://api.deepseek.com)
  - "anthropic" : Claude via the Anthropic Messages API

No tools are ever passed to any provider, so web search / code execution are
disabled for every method (required for the "LLM only" baseline).
"""
from __future__ import annotations

import contextlib
import os
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict


# --------------------------------------------------------------------------- #
# Usage tracking
# --------------------------------------------------------------------------- #
@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0          # includes reasoning/thinking tokens when billed
    reasoning_tokens: int = 0       # subset of output_tokens, where the API reports it
    cached_input_tokens: int = 0
    refusals: int = 0
    errors: int = 0

    def add(self, other: "Usage") -> None:
        for k, v in asdict(other).items():
            setattr(self, k, getattr(self, k) + v)


class UsageTracker:
    """Thread-safe token accounting, bucketed by phase ("index", "query", "embed")."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self.by_phase: dict[str, Usage] = defaultdict(Usage)

    def record(self, phase: str, usage: Usage) -> None:
        with self._lock:
            self.by_phase[phase].add(usage)
        for cap in getattr(self._local, "stack", []):
            cap[phase].add(usage)

    @contextlib.contextmanager
    def capture(self):
        """Also collect usage recorded *by the current thread* into a dict
        phase -> Usage (used for per-question query cost)."""
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        cap: dict[str, Usage] = defaultdict(Usage)
        self._local.stack.append(cap)
        try:
            yield cap
        finally:
            self._local.stack.remove(cap)

    def snapshot(self) -> dict:
        return self.to_dict()

    def merge(self, phase: str, usage_dict: dict) -> None:
        self.record(phase, Usage(**usage_dict))

    def to_dict(self) -> dict:
        with self._lock:
            return {p: asdict(u) for p, u in self.by_phase.items()}


@dataclass
class LLMConfig:
    label: str
    provider: str
    model: str
    temperature: float | None = 0.0
    # Some models (Claude with always-on thinking, reasoning models) spend output
    # tokens before answering; every request gets at least this many max_tokens.
    max_tokens_floor: int = 0
    effort: str | None = None           # anthropic: output_config.effort
    # Explicit thinking mode. None = provider default. DeepSeek V4 Flash thinks by
    # default (reasoning tokens are billed and eat max_tokens); Claude Opus 5.5 cannot
    # disable thinking (control it with `effort`), Claude Haiku 5.5 can at effort<=high.
    thinking: bool | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    price_input_per_m: float | None = None
    price_output_per_m: float | None = None
    price_cached_input_per_m: float | None = None
    max_retries: int = 6
    timeout: float = 120.0
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in known}
        kw["extra"] = {k: v for k, v in d.items() if k not in known}
        kw.setdefault("label", d.get("model"))
        return cls(**kw)

    def cost_usd(self, u: dict) -> float | None:
        if self.price_input_per_m is None or self.price_output_per_m is None:
            return None
        cached = u.get("cached_input_tokens", 0)
        cached_price = self.price_cached_input_per_m
        if cached_price is None:
            cached_price, cached = self.price_input_per_m, 0  # treat as uncached
        uncached = u.get("input_tokens", 0) - cached
        return (uncached * self.price_input_per_m
                + cached * cached_price
                + u.get("output_tokens", 0) * self.price_output_per_m) / 1e6


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #
class LLMClient(ABC):
    def __init__(self, cfg: LLMConfig, tracker: UsageTracker):
        self.cfg = cfg
        self.tracker = tracker

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 256, phase: str = "query") -> tuple[str, bool]:
        """Returns (text, truncated). `truncated` is True when generation stopped
        because it hit max_tokens."""
        max_tokens = max(max_tokens, self.cfg.max_tokens_floor)
        try:
            text, usage, truncated = self._generate(prompt, system, max_tokens)
        except Exception:
            self.tracker.record(phase, Usage(calls=1, errors=1))
            raise
        self.tracker.record(phase, usage)
        return text, truncated

    def generate(self, prompt: str, **kw) -> str:
        return self.complete(prompt, **kw)[0]

    @abstractmethod
    def _generate(self, prompt: str, system: str | None,
                  max_tokens: int) -> tuple[str, Usage, bool]:
        ...


class OpenAICompatClient(LLMClient):
    DEFAULTS = {
        "openai": (None, "OPENAI_API_KEY"),
        "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    }

    def __init__(self, cfg: LLMConfig, tracker: UsageTracker):
        super().__init__(cfg, tracker)
        from openai import OpenAI
        base_url, key_env = self.DEFAULTS.get(cfg.provider, (None, "OPENAI_API_KEY"))
        self.client = OpenAI(
            base_url=cfg.base_url or base_url,
            api_key=os.environ.get(cfg.api_key_env or key_env),
            max_retries=cfg.max_retries,
            timeout=cfg.timeout,
        )

    def _generate(self, prompt, system, max_tokens):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(model=self.cfg.model, messages=messages, max_tokens=max_tokens)
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature
        if self.cfg.provider == "deepseek" and self.cfg.thinking is not None:
            kwargs["extra_body"] = {"thinking": {
                "type": "enabled" if self.cfg.thinking else "disabled"}}
        kwargs.update(self.cfg.extra)
        resp = self.client.chat.completions.create(**kwargs)
        u = resp.usage
        cached = 0
        if u is not None:
            details = getattr(u, "prompt_tokens_details", None)
            cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
            cached = cached or (getattr(u, "prompt_cache_hit_tokens", 0) or 0)  # DeepSeek
        cdet = getattr(u, "completion_tokens_details", None) if u else None
        usage = Usage(
            calls=1,
            input_tokens=u.prompt_tokens if u else 0,
            output_tokens=u.completion_tokens if u else 0,
            reasoning_tokens=(getattr(cdet, "reasoning_tokens", 0) or 0) if cdet else 0,
            cached_input_tokens=cached,
        )
        choice = resp.choices[0]
        if choice.finish_reason == "content_filter":
            usage.refusals = 1
        return (choice.message.content or "").strip(), usage, choice.finish_reason == "length"


class AnthropicClient(LLMClient):
    def __init__(self, cfg: LLMConfig, tracker: UsageTracker):
        super().__init__(cfg, tracker)
        import anthropic
        kw = dict(max_retries=cfg.max_retries, timeout=cfg.timeout)
        if cfg.api_key_env:
            kw["api_key"] = os.environ.get(cfg.api_key_env)
        if cfg.base_url:
            kw["base_url"] = cfg.base_url
        self.client = anthropic.Anthropic(**kw)

    def _generate(self, prompt, system, max_tokens):
        kwargs = dict(model=self.cfg.model, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        # Current Claude models reject sampling params (temperature/top_p), so
        # cfg.temperature is intentionally not sent. Depth is set via effort.
        if self.cfg.effort:
            kwargs["output_config"] = {"effort": self.cfg.effort}
        if self.cfg.thinking is not None:
            kwargs["thinking"] = {"type": "adaptive" if self.cfg.thinking else "disabled"}
        kwargs.update(self.cfg.extra)
        resp = self.client.messages.create(**kwargs)
        u = resp.usage
        usage = Usage(
            calls=1,
            input_tokens=(u.input_tokens or 0) + (u.cache_read_input_tokens or 0)
            + (u.cache_creation_input_tokens or 0),
            output_tokens=u.output_tokens or 0,
            cached_input_tokens=u.cache_read_input_tokens or 0,
        )
        if resp.stop_reason == "refusal":
            usage.refusals = 1
            return "", usage, False
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text.strip(), usage, resp.stop_reason == "max_tokens"


class MockClient(LLMClient):
    """Offline, deterministic, free. For pipeline tests only, never for results."""

    def _generate(self, prompt, system, max_tokens):
        if "Question:" in prompt:
            text = "unknown"
        else:  # summary: first words of the input, then a sentence end
            text = " ".join(prompt.split()[-60:]).rstrip(".") + "."
        return text, Usage(calls=1, input_tokens=len(prompt) // 4,
                           output_tokens=len(text) // 4), False


def usage_delta(after: dict, before: dict) -> dict:
    out = {}
    for phase, u in after.items():
        b = before.get(phase, {})
        d = {k: v - b.get(k, 0) for k, v in u.items()}
        if any(d.values()):
            out[phase] = d
    return out


def make_llm(cfg: LLMConfig, tracker: UsageTracker) -> LLMClient:
    if cfg.provider in ("openai", "deepseek", "openai_compat"):
        return OpenAICompatClient(cfg, tracker)
    if cfg.provider == "anthropic":
        return AnthropicClient(cfg, tracker)
    if cfg.provider == "mock":
        return MockClient(cfg, tracker)
    raise ValueError(f"Unknown LLM provider: {cfg.provider!r}")
