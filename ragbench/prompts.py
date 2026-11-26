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


# --------------------------------------------------------------------------- #
# HippoRAG (GraphRAG) prompts and parsing.
#
# These reproduce the verbatim v1 HippoRAG prompts (OSU-NLP-Group/HippoRAG,
# `legacy` branch): passage NER, post-NER OpenIE, and query NER. Upstream sends
# them as a multi-turn chat (system + a one-shot human/AI exchange + the real
# input). ragbench's LLMClient is single-turn (one system + one user string), so
# the one-shot example is folded into the user string; the system instruction and
# the example content are kept word for word.
# --------------------------------------------------------------------------- #
import json as _json

# One-shot example shared by passage NER and OpenIE (upstream `one_shot_passage`).
_HIPPO_ONESHOT_PASSAGE = (
    "Radio City\n"
    "Radio City is India's first private FM radio station and was started on 3 July 2001.\n"
    "It plays Hindi, English and regional songs.\n"
    "Radio City recently forayed into New Media in May 2008 with the launch of a music "
    "portal - PlanetRadiocity.com that offers music related news, videos, songs, and other "
    "music-related features."
)
_HIPPO_ONESHOT_PASSAGE_ENTITIES = (
    '{"named_entities":\n'
    '    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", '
    '"PlanetRadiocity.com"]\n'
    "}\n"
)
_HIPPO_ONESHOT_PASSAGE_TRIPLES = (
    '{"triples": [\n'
    '            ["Radio City", "located in", "India"],\n'
    '            ["Radio City", "is", "private FM radio station"],\n'
    '            ["Radio City", "started on", "3 July 2001"],\n'
    '            ["Radio City", "plays songs in", "Hindi"],\n'
    '            ["Radio City", "plays songs in", "English"]\n'
    '            ["Radio City", "forayed into", "New Media"],\n'
    '            ["Radio City", "launched", "PlanetRadiocity.com"],\n'
    '            ["PlanetRadiocity.com", "launched in", "May 2008"],\n'
    '            ["PlanetRadiocity.com", "is", "music portal"],\n'
    '            ["PlanetRadiocity.com", "offers", "news"],\n'
    '            ["PlanetRadiocity.com", "offers", "videos"],\n'
    '            ["PlanetRadiocity.com", "offers", "songs"]\n'
    "    ]\n"
    "}\n"
)

# Passage NER (upstream `ner_instruction` + one-shot + `ner_user_input`).
_HIPPO_NER_SYSTEM = ("Your task is to extract named entities from the given paragraph. \n"
                     "Respond with a JSON list of entities.\n")


def passage_ner_prompt(passage: str) -> tuple[str, str]:
    example = (f"Paragraph:\n```\n{_HIPPO_ONESHOT_PASSAGE}\n```\n\n"
               f"{_HIPPO_ONESHOT_PASSAGE_ENTITIES}")
    user = f"{example}\nNow do the same for the next paragraph.\n\nParagraph:```\n{passage}\n```"
    return _HIPPO_NER_SYSTEM, user


# Post-NER OpenIE (upstream `openie_post_ner_instruction` + one-shot + `openie_post_ner_frame`).
_HIPPO_OPENIE_SYSTEM = (
    "Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. \n"
    "Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. \n\n"
    "Pay attention to the following requirements:\n"
    "- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.\n"
    "- Clearly resolve pronouns to their specific names to maintain clarity.\n\n"
)


def _openie_frame(passage: str, named_entity_json: str) -> str:
    return ("Convert the paragraph into a JSON dict, it has a named entity list and a triple "
            f"list.\nParagraph:\n```\n{passage}\n```\n\n{named_entity_json}\n")


def openie_prompt(passage: str, entities: list[str]) -> tuple[str, str]:
    example = (_openie_frame(_HIPPO_ONESHOT_PASSAGE, _HIPPO_ONESHOT_PASSAGE_ENTITIES)
               + _HIPPO_ONESHOT_PASSAGE_TRIPLES)
    named_entity_json = _json.dumps({"named_entities": entities})
    user = (f"{example}\nNow do the same for the next paragraph.\n\n"
            f"{_openie_frame(passage, named_entity_json)}")
    return _HIPPO_OPENIE_SYSTEM, user


# Query NER (upstream `named_entity_extraction_parallel.py` one-shot).
_HIPPO_QUERY_NER_SYSTEM = "You're a very effective entity extraction system."
_HIPPO_QUERY_NER_ONESHOT_IN = (
    "Please extract all named entities that are important for solving the questions below.\n"
    "Place the named entities in json format.\n\n"
    "Question: Which magazine was started first Arthur's Magazine or First for Women?\n\n"
)
_HIPPO_QUERY_NER_ONESHOT_OUT = '{"named_entities": ["First for Women", "Arthur\'s Magazine"]}'


def query_ner_prompt(question: str) -> tuple[str, str]:
    user = (f"{_HIPPO_QUERY_NER_ONESHOT_IN}{_HIPPO_QUERY_NER_ONESHOT_OUT}\n\n"
            f"Question: {question}\n")
    return _HIPPO_QUERY_NER_SYSTEM, user


# --------------------------------------------------------------------------- #
# Tolerant JSON parsing: strict json.loads first, then salvage the first JSON
# object/array in the text (tolerating trailing / missing commas), then [].
# --------------------------------------------------------------------------- #
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
# missing comma between two adjacent list items: "] [" -> "], [" (the one-shot
# triples example itself has a missing comma after the "English" row).
_MISSING_COMMA = re.compile(r"\]\s*\[")


def _loads_salvage(raw: str):
    if not raw or not raw.strip():
        return None
    try:
        return _json.loads(raw)
    except Exception:
        pass
    start = min([i for i in (raw.find("{"), raw.find("[")) if i != -1], default=-1)
    if start == -1:
        return None
    opener = raw[start]
    closer = "}" if opener == "{" else "]"
    end = raw.rfind(closer)
    if end <= start:
        return None
    blob = raw[start:end + 1]
    for candidate in (blob, _TRAILING_COMMA.sub(r"\1", blob)):
        for fixed in (candidate, _MISSING_COMMA.sub("], [", candidate)):
            try:
                return _json.loads(fixed)
            except Exception:
                continue
    return None


def parse_entities(raw: str) -> list[str]:
    """Parse an NER response into a list of entity strings. Returns [] on failure."""
    data = _loads_salvage(raw)
    if isinstance(data, dict):
        data = data.get("named_entities", [])
    if not isinstance(data, list):
        return []
    out, seen = [], set()
    for e in data:
        if not isinstance(e, str):
            continue
        e = e.strip()
        if e and e not in seen:
            seen.add(e)
            out.append(e)
    return out


def parse_triples(raw: str) -> list[tuple[str, str, str]]:
    """Parse an OpenIE response into (subject, relation, object) triples. [] on failure."""
    data = _loads_salvage(raw)
    if isinstance(data, dict):
        data = data.get("triples", [])
    if not isinstance(data, list):
        return []
    out = []
    for t in data:
        if not isinstance(t, (list, tuple)) or len(t) != 3:
            continue
        if not all(isinstance(x, str) for x in t):
            continue
        s, r, o = (x.strip() for x in t)
        if s and r and o:
            out.append((s, r, o))
    return out
