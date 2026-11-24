"""QA prompts shared by every method so differences come from retrieval/indexing,
not from prompt wording."""
from __future__ import annotations

import re

SHORT_RULE = ("Respond with only the final answer as a short phrase (an entity, date, "
              "number, or a few words). Do not explain.")
COT_RULE = ("Think step by step, then on the last line write 'Answer: ' followed by the "
            "final answer as a short phrase.")

SYSTEM_CLOSED_BOOK = ("You are answering questions using only your own knowledge. "
                      "If unsure, give your best guess.")
SYSTEM_WITH_CONTEXT = ("You are answering questions using the provided context. The answer "
                       "may require combining facts from several passages.")


def qa_prompt(question: str, context: str | None, answer_format: str) -> tuple[str, str]:
    rule = COT_RULE if answer_format == "cot" else SHORT_RULE
    if context is None:
        return SYSTEM_CLOSED_BOOK + " " + rule, f"Question: {question}"
    return (SYSTEM_WITH_CONTEXT + " " + rule,
            f"Context:\n{context}\n\nQuestion: {question}")


def extract_answer(raw: str, answer_format: str) -> str:
    if answer_format == "cot":
        m = re.findall(r"answer\s*:\s*(.+)", raw, flags=re.IGNORECASE)
        if m:
            return m[-1].strip().strip("*").strip()
        lines = [l for l in raw.strip().splitlines() if l.strip()]
        return lines[-1].strip() if lines else ""
    return raw.strip().splitlines()[0].strip() if raw.strip() else ""


# RAPTOR summarization prompt (official repo wording + an explicit length target,
# instead of hard truncation via max_tokens, so thinking models aren't cut off).
RAPTOR_SUMMARY = ("Write a summary of the following, including as many key details as "
                  "possible (at most about {words} words):\n\n{context}")
