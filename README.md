# RAG Research

How well do RAG approaches handle **multi-hop** questions? Measured by answer F1,
retrieval recall, and token cost (indexing vs. answering).

| Item | Details |
| --- | --- |
| Datasets | MuSiQue (first), MultiHop-RAG (later) |
| Methods | RAPTOR (Jida), GraphRAG (Micah), vanilla / DOS RAG (Shreyansh), LLM only, LLM + full doc |
| LLMs | GPT-4o-mini, DeepSeek V4 Flash, Claude |
| Metrics | F1 / EM / answer recall, retrieval recall, tokens + $ for indexing and for queries |

## Getting started (teammates)

```bash
git clone git@github.com:jidalii/rag-research.git && cd rag-research
uv venv --python 3.11
uv pip install --python .venv/bin/python -e . pytest
cp .env.example .env                                  # add only the API keys you need
.venv/bin/python scripts/prepare_data.py all          # download + verify both datasets
.venv/bin/python -m pytest -q                         # offline tests, no API calls
.venv/bin/python -m ragbench.run configs/smoke_mock.toml   # whole pipeline, mock LLM, free
```

> If another project's virtualenv is active, `uv pip install` writes into *that* one.
> Always pass `--python .venv/bin/python`, or run `deactivate` first.

## Datasets

`scripts/prepare_data.py` downloads each dataset from a pinned source, checks SHA-256
checksums, validates it through the same loaders the experiments use, and writes a
`manifest.json`. Re-running it never downloads again; `--verify` only re-checks.

```bash
.venv/bin/python scripts/prepare_data.py all                    # both datasets (MuSiQue dev)
.venv/bin/python scripts/prepare_data.py musique --splits dev train
.venv/bin/python scripts/prepare_data.py all --verify           # offline integrity check
```

| | MuSiQue (answerable, v1.0) | MultiHop-RAG |
| --- | --- | --- |
| Source | official Google Drive release (StonyBrookNLP/musique) | HF `yixuantt/MultiHopRAG` @ `71ac0d0` |
| Files | `datasets/musique/musique_ans_v1.0_{dev,train}.jsonl` | `datasets/multihop_rag/{corpus.json,MultiHopRAG.json}` |
| Questions | dev 2,417 (2-hop 1,252 / 3-hop 760 / 4-hop 405) | 2,556 (comparison 856, inference 816, temporal 583, null 301) |
| Corpus | 17–20 paragraphs per question (gold + distractors); 21,100 unique in dev | 609 news articles shared by all questions |
| Gold evidence | `is_supporting` paragraphs (one per hop) | `evidence_list` articles, matched by title |
| Retrieval recall | paragraph level | article level |

The MuSiQue test split has no answers, so we use dev. MultiHop-RAG queries have no ids in
the release; the loader assigns `mhr_0000`… by position in the file, so ids are stable.

**Shared question subsets.** To make results comparable across methods and teammates,
everyone runs on the same question ids, committed under `datasets/<name>/subsets/`:

| Subset | MuSiQue | MultiHop-RAG |
| --- | --- | --- |
| `pilot_100.txt` | 100 questions | 100 questions |
| `main_1000.txt` | 1,000 questions | 1,000 questions |

Both are sampled with seed 0 and stratified by question type (MuSiQue `2hop`…`4hop3`,
MultiHop-RAG query type), so small samples keep the full mix. Use one from a config:

```toml
[dataset]
ids_file = "datasets/musique/subsets/pilot_100.txt"
```

Make a new one with `scripts/prepare_data.py musique --verify --subset NAME --n N --seed S`
(add `--exclude-types null_query` to drop MultiHop-RAG's unanswerable queries).

## Running experiments

Each run is described by one TOML file. Each `[[llms]]` × `[[methods]]` pair is one experiment.

```bash
.venv/bin/python -m ragbench.run configs/smoke_mock.toml                 # offline, free
.venv/bin/python -m ragbench.run configs/musique_pilot.toml --dry-run    # print plan only
.venv/bin/python -m ragbench.run configs/musique_smoke.toml --llm deepseek-flash
.venv/bin/python -m ragbench.run configs/musique_pilot.toml --llm gpt-4o-mini --method raptor --limit 20
```

Outputs go to `results/<run.name>/`:

- `<dataset>__<method>__<llm>/predictions.jsonl`: one line per question with the prediction, retrieved passage ids, metrics and token usage.
- `summary.json`: mean metrics, metrics by hop count and question type, index/query tokens and $.
- `summary.csv`: one row per experiment.

Runs can be resumed: rerunning the same command skips finished questions and retries failed ones. RAPTOR trees are cached in `indexes/`, and a JSON file next to each tree records the tokens it took to build. The index cost is therefore reported even when the tree comes from the cache.

### Config sections (see `configs/musique_pilot.toml`)

| Section | Keys |
| --- | --- |
| `[run]` | `name`, `output_dir`, `index_dir`, `cache_indexes`, `max_workers`, `seed` |
| `[dataset]` | `name` (`musique` \| `multihop_rag`), `path`, `ids_file`, `limit`, `sample` (`first` \| `random`), `corpus` (`pooled` \| `per_question` \| `full`) |
| `[embedding]` | `provider` (`local` \| `openai`), `model`. Shared by every LLM. |
| `[qa]` | `answer_format` (`short` \| `cot`), `max_answer_tokens` |
| `[[llms]]` | `label`, `provider` (`openai` \| `deepseek` \| `anthropic` \| `mock`), `model`, `temperature`, `thinking`, `effort` (Claude), `max_tokens_floor`, `price_*_per_m` |
| `[[methods]]` | `name`, optional `label` and `corpus` override, plus method parameters |

No method gets tools (no web search or code execution). This matters most for the LLM-only baseline.

## Design decisions

**Corpus scope.** Each MuSiQue question ships with up to 20 paragraphs (2–4 gold, the rest distractors), about 2k tokens. An index built over only those paragraphs is tiny. With a 2k-token retrieval budget it returns almost everything, so retrieval recall is close to 1 and says nothing. `corpus = "pooled"` (the default) builds one shared index over the de-duplicated paragraphs of all sampled questions. That gives 1,802 passages for `pilot_100` and 11,774 for `main_1000`, close to HippoRAG's MuSiQue setup. `per_question` is still available for comparison.

**RAPTOR and multi-hop.** RAPTOR is a way of building the index. It retrieves once, with no step-by-step decomposition, so it can run on multi-hop data. Its summary nodes can combine facts from several passages, which is the hypothesis to test.

**Retrieval recall for RAPTOR.** Two numbers are reported:

- `retrieval_recall`: gold passages hit by retrieved *leaf* nodes.
- `retrieval_recall_expanded`: a retrieved summary node also counts every passage beneath it. This is an upper bound and is generous.

**Fixed embeddings.** Claude has no embedding API. All LLMs therefore share one embedding model (upstream RAPTOR's SBERT), so differences between runs come from the LLM (summaries and answers), not from retrieval embeddings.

**Thinking tokens.** DeepSeek V4 Flash thinks by default. The configs set `thinking = false`, as in adaptive-raptor, where the default left RAPTOR summaries empty. Claude Opus 5.5 always thinks and is controlled with `effort`. Thinking tokens are billed as output and are included in the token counts.

## RAPTOR implementation

`raptor/` is upstream RAPTOR copied verbatim (commit `7da1d48`, see `raptor/UPSTREAM.md`), following the same approach as adaptive-raptor. The adapters live in `ragbench/methods/raptor.py`. The upstream code is used for UMAP+GMM clustering, recursive summarization, `Tree`, and `TreeRetriever` (collapsed tree: `top_k=10`, `max_context_tokens=2000`). Differences from upstream:

1. **Summaries never stop mid-sentence.** Upstream passes `summarization_length=100` as a hard `max_tokens`, which cuts summaries off. Here the length goes into the prompt ("at most about N words, end with a complete sentence") and `max_tokens` gets 4× headroom. If the model still hits the limit, the text is trimmed back to its last complete sentence; if nothing can be trimmed, the call is retried once with twice the budget. Counts are in `index_stats.summarizer`. Set `summary_prompt = "upstream"` to use the original prompt word for word.
2. Leaves are split per passage (upstream `split_text`, 100 tokens), so each leaf maps to its source passage and retrieval recall can be computed.
3. Summaries can be generated by any provider, and their tokens are counted.
4. UMAP is seeded so trees can be reproduced.

## Adding a method (GraphRAG, vanilla RAG, ...)

Every method gets the same questions, corpus, QA prompt, metrics and token accounting
from the runner; a method only decides **how to index** and **what context to retrieve**.

**1. Write the class** in `ragbench/methods/<your_method>.py`. A minimal vanilla RAG:

```python
import numpy as np
from .base import Method


class VanillaRAG(Method):
    name = "vanilla_rag"
    needs_index = True        # runner calls build_index() and caches the result
    uses_embedder = True      # runner passes the shared embedder as self.embedder

    def build_index(self, passages):
        # passages: list[Passage(pid, title, text)], already de-duplicated.
        # LLM calls made here must use phase="index" (e.g. GraphRAG entity extraction):
        #   self.llm.generate(prompt, system=..., max_tokens=..., phase="index")
        texts = [f"{p.title}: {p.text}" for p in passages]
        return {"pids": [p.pid for p in passages], "texts": texts,
                "emb": self.embedder.embed(texts)}           # must be picklable

    def answer(self, ex, index):
        q = self.embedder.embed([ex.question])[0]
        top = np.argsort(-(index["emb"] @ q))[: self.params.get("top_k", 5)]
        context = "\n\n".join(index["texts"][i] for i in top)
        pred, raw = self.generate_answer(ex.question, context)  # shared prompt, phase="query"
        return {
            "prediction": pred,
            "raw": raw,
            "retrieved_pids": [index["pids"][i] for i in top],  # -> retrieval_recall
        }
```

**2. Register it** in `ragbench/methods/__init__.py`:

```python
from .vanilla_rag import VanillaRAG
REGISTRY[VanillaRAG.name] = VanillaRAG
```

**3. Add it to a config.** Every key except `name`, `label` and `corpus` arrives in
`self.params`:

```toml
[[methods]]
name = "vanilla_rag"
top_k = 5
# label = "vanilla_rag_k10"   # optional: run several variants of one method
# corpus = "per_question"     # optional: override dataset.corpus for this method
```

**4. Test it for free, then for real.** Add it to `configs/smoke_mock.toml` and run that
(mock LLM, no keys), then `configs/musique_smoke.toml --llm gpt-4o-mini` (3 questions).

**Rules that keep results comparable**

- Make every LLM call through `self.llm` (`generate`, or `complete` to also get a
  truncation flag), with `phase="index"` while indexing and `phase="query"` when
  answering. Calling a provider SDK directly bypasses token and $ accounting.
- Produce the final answer with `self.generate_answer(question, context)` so all methods
  share one QA prompt and answer format.
- Return `retrieved_pids` as corpus `Passage.pid` values. If your retrieval units are
  smaller (chunks) or derived (summaries, graph communities), map them back to source
  passages; you can also return `retrieved_pids_expanded` for a looser variant, as RAPTOR
  does.
- Put every parameter that changes the index in `index_params()` (defaults to all params)
  so cached indexes are rebuilt when it changes.
- Use `self.embedder` rather than your own embedding model, unless the method requires
  one; if so, say so in the results.
- Methods are called from several threads (`run.max_workers`); keep `answer()` free of
  shared mutable state.

## Layout

```text
ragbench/
  run.py          CLI + runner (TOML -> experiments, resume, index cache, summaries)
  llm.py          OpenAI / DeepSeek / Anthropic / mock adapters + token accounting
  datasets.py     MuSiQue + MultiHop-RAG loaders -> (examples, corpus)
  metrics.py      EM / F1 / answer recall / retrieval recall
  embeddings.py   local SBERT or OpenAI embeddings
  prompts.py      shared QA prompts
  methods/        llm_only, full_context, raptor  (+ your method)
raptor/           vendored upstream RAPTOR (do not edit)
configs/          experiment TOMLs
scripts/          prepare_data.py (download, verify, subsets)
tests/            offline tests
```
