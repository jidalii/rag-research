# Experiment Setup: Multi-hop RAG Study

Status: **draft for review** (2026-11-27). Companion to the hypotheses doc (H1–H4).
Nothing below is implemented yet beyond what `main` already has; section 10 lists the code
changes that start once this is approved.

**In one paragraph.** We run 6 core methods (plus 3 LLM-only variants and an oracle) × 3
LLMs × 2 datasets on the shared `main_1000` question sets. Retrieval methods are swept over
three context budgets (1k / 2k / 4k tokens). Every answer is scored by token F1/EM and by one
fixed LLM judge. That is 96 experiment runs, about 105k answer calls and 105k judge calls.
Estimated API cost is about $45 for GPT-4o-mini and DeepSeek V4 Flash plus the judge; the third
LLM is Qwen3 8B run locally with Ollama, which costs nothing but takes an estimated 2 to 4 days
of machine time (section 9). Who builds the indexes, the local model size and the judge model
are the main decisions for review (section 11, D0 to D2). Every experiment is described by a
YAML config.

## 1. Experiment matrix

Each row is one *arm*. An arm × LLM × dataset is one run directory under `results/`.

| Arm | Context given to the reader | Budgets | MuSiQue | MultiHop-RAG | Used by |
| --- | --- | --- | --- | --- | --- |
| `llm_only` | none (closed book) | – | yes | yes | H1 floor, H4 |
| `llm_only_cot` | none, chain-of-thought answer | – | yes | – | H4 |
| `llm_only_decomposed` | none, each sub-question asked alone | – | yes | – | H4 |
| `full_context` | MuSiQue: the question's own 17–20 paragraphs. MHR: gold articles + distractors up to 32k tokens | – | yes | yes | H1 |
| `oracle` | gold passages only, shuffled | – | yes | yes | H2 |
| `vanilla_rag` | top chunks, similarity order | 1k, 2k, 4k | yes | yes | H1, H2, H3 |
| `dos_rag` | same chunks, grouped by document in reading order | 1k, 2k, 4k | yes | yes | H1, H2, H3 |
| `raptor` | collapsed-tree nodes (leaves + summaries) | 1k, 2k, 4k | yes | yes | H1, H2, H3 |
| `hipporag` | passages ranked by personalized PageRank | 1k, 2k, 4k | yes | yes | H1, H2, H3 |

Per LLM: 17 arms on MuSiQue, 15 on MultiHop-RAG, so 32 runs × 3 LLMs = **96 runs**.
`slot_rag` stays in the harness but is outside the core matrix (decision D7).

## 2. Datasets and question sets

| | MuSiQue | MultiHop-RAG |
| --- | --- | --- |
| Split / source | answerable v1.0 dev (pinned, see `datasets/musique/manifest.json`) | HF `yixuantt/MultiHopRAG` @ `71ac0d0` |
| Questions | `subsets/main_1000.txt` (pilot: `pilot_100.txt`) | `subsets/main_1000.txt`: 335 comparison, 319 inference, 228 temporal, 118 null |
| Retrieval corpus | `pooled`: de-duplicated paragraphs of the 1,000 questions = 11,774 passages | all 609 articles (1.39M tokens; mean 2.3k, max 16k per article) |
| Gold evidence | `is_supporting` paragraphs, one per hop, with hop position from `question_decomposition` | `evidence_list` articles (article level) and their `fact` sentences |
| Hop count | 2 / 3 / 4 from the decomposition | number of distinct evidence articles |
| Question shape | type prefix: `2hop`, `3hop1` (chain), `3hop2`, `4hop1` (chain), `4hop2`, `4hop3` | query type |

Rules:

- **Null queries** (118 MHR questions whose answer is "insufficient information") are run with
  everything else but excluded from H1–H4 accuracy. They are reported separately for the
  deferred hallucination hypothesis. So H1–H4 use 1,000 MuSiQue and 882 MHR questions.
- **Full context on MuSiQue** is about 2.2k tokens on average (max 3.7k), so it falls between
  the 2k and 4k budgets.
- **Full context on MHR** cannot be the whole corpus (1.39M tokens). It is the question's gold
  articles plus distractor articles sampled with a fixed seed, filled to a 32k-token cap. Gold
  evidence alone averages 5.9k tokens (max 33k); if gold exceeds the cap, gold is kept and the
  overflow is reported (D3).
- **Budgets vs. MHR gold size.** With gold evidence averaging 5.9k tokens, the 1k–4k budgets
  cannot hold whole gold articles. That is fine for chunk retrieval (the evidence fact is a
  sentence), but it is why MHR retrieval recall is also reported at fact level: a gold
  article counts as hit when its evidence `fact` appears in the context (D5).

## 3. Methods and fixed parameters

Shared by all methods:

- **Embedder:** `sentence-transformers/multi-qa-mpnet-base-cos-v1` (local, the same for every
  LLM, so retrieval does not depend on the reader).
- **QA prompt:** the shared `ragbench/prompts.py` prompt, `answer_format = "short"`,
  `max_answer_tokens = 256`. The only CoT arm is `llm_only_cot`.
- **Context budget:** each retrieval method adds units in its own ranking order until the next
  unit would exceed `max_context_tokens` (cl100k tokens). `top_k` no longer limits the context;
  the budget does. Providers' own input-token counts are recorded too.
- **No tools** for any method (no web search, no code execution).

| Method | Index (built once per LLM and dataset, reused across budgets) | Query-time parameters |
| --- | --- | --- |
| `vanilla_rag`, `dos_rag` | 100-token chunks with title prefix (no LLM calls) | budget |
| `raptor` | upstream defaults: 100-token leaves, summaries of ~100 words that end on a full sentence, 5 layers, UMAP dim 10, GMM threshold 0.1, cluster cap 3,500 tokens, seed 224 | collapsed tree, budget |
| `hipporag` | per-passage NER + OpenIE, synonym edges at cosine ≥ 0.8, seed 224 | PPR damping 0.5, link top-1, budget |
| `full_context` | none | MHR cap 32k |
| `oracle` | none | – |
| `llm_only*` | none | – |

Gold-in-context rule for chunked methods: a gold passage counts as retrieved when **any** of
its chunks is in the context (D6). For RAPTOR and HippoRAG we also report the "expanded"
variant (a retrieved summary or graph node credits the passages beneath it).

## 4. Models and decoding

| Label | Model id | Settings | Price in config (USD / 1M in / out) |
| --- | --- | --- | --- |
| `gpt-4o-mini` | `gpt-4o-mini` | temperature 0 | 0.15 / 0.60 |
| `deepseek-flash` | `deepseek-flash` (V4 Flash) | temperature 0, thinking off | 0.15 / 0.60 (off-peak; verify) |
| `qwen3-8b` | `qwen3:8b` via local Ollama, Q4_K_M (pin the tag and digest before the main run) | temperature 0, fixed seed, thinking off (`think: false`), `num_ctx` set per run | 0 / 0 (local) |

**Local Qwen.** Ollama runs on the M2 Pro with 16 GB of memory, so an 8B model at 4-bit (about
5 GB) is the largest that fits with room for a 32k-token context: the KV cache at 32k is about
5 GB more. Rules for the local runs:

- **Context window.** Ollama's default `num_ctx` is small and silently cuts long prompts from
  the front. Every Qwen run sets `num_ctx` above its largest prompt (8,192 for the 1k–4k
  budgets, 36,864 for MHR `full_context`), and the harness fails a question if Ollama's
  `prompt_eval_count` is lower than the prompt's token count.
- **Thinking off.** Qwen3 is a hybrid reasoning model; `think: false` keeps it a
  non-reasoning reader like the other two.
- **One request at a time** (`max_workers: 1`, `OLLAMA_NUM_PARALLEL=1`), since parallel slots
  each need their own KV cache.
- **Cost is reported as tokens and wall-clock hours**, not dollars. Index and query token counts
  are still recorded, so H1's cost comparison still works in tokens.

**Model strength order.** H1, H2 and H4 talk about "stronger" models. The
three LLMs are close in tier, so we do not assume an order. We rank them by their measured
`oracle` accuracy (gold passages only) on each dataset and use that ranking in the analysis.

Thinking tokens are billed as output and counted. Prices only feed the cost meter and must be
checked before results are reported. **Determinism check:** temperature 0 is not fully
deterministic, so one arm (`vanilla_rag` at 2k, `pilot_100`) is repeated 3 times per LLM and
the answer flip rate is reported.

## 5. Metrics recorded

Per question (`predictions.jsonl`), in addition to what is recorded today:

| Field | Meaning | Hypothesis |
| --- | --- | --- |
| `metrics.em`, `f1`, `answer_recall`, `contains` | string metrics (already recorded) | all |
| `judge.correct`, `judge.raw`, `judge.model` | LLM-judge verdict (written by the judge step) | all |
| `context_tokens` | cl100k tokens of the context actually sent | H1 |
| `retrieval_recall`, `retrieval_recall_expanded` | share of gold passages in context | H1, H3 |
| `all_gold_retrieved` | every gold passage in context | H2 |
| `recall_by_hop` | `{hop_position: 0/1}` per gold paragraph (MuSiQue), leaf and expanded | H3 |
| `fact_recall` | MHR: share of evidence facts present in the context | H3 |
| `subq` | `llm_only_decomposed`: per sub-question prediction and judge verdict | H4 |
| `latency_s` | wall-clock seconds for the question | cost |
| `usage` | tokens per phase (already recorded) | cost |

Per run (`summary.json`): accuracy (judge and F1) overall, by hop, by type and by gold-retrieved
flag; index build tokens, dollars and wall-clock time; cache hits.

## 6. LLM judge

- **One judge model for every run**, at temperature 0, ideally not one of the three tested
  models (D2).
- **Input:** question, gold answer and aliases, prediction. **Output:** `CORRECT` or
  `INCORRECT` plus a one-line reason. MHR null queries: an abstention ("insufficient
  information" or equivalent) is correct.
- **Runs after answering**, as a separate step over existing `predictions.jsonl` files. It is
  resumable and caches verdicts by hash of (question, gold, prediction, judge model), so
  identical predictions are judged once.
- **Calibration:** 100 answers sampled across arms, LLMs and both datasets, labelled by two
  teammates without seeing the judge's verdict. Report human–human and human–judge agreement
  (Cohen's κ). Target κ ≥ 0.8; the human–judge disagreement rate is the judge error used in
  H2's edge cases.

## 7. Statistics

- **Paired comparisons:** all arms answer the same questions. Pairwise arm differences use a
  paired bootstrap (10,000 resamples over questions) for 95% intervals and McNemar's test,
  Holm-corrected within each family of comparisons.
- **H1 interaction:** mixed-effects logistic regression,
  `correct ~ method * model + (1 | question)`, per dataset and budget.
- **Sample size:** at n = 1,000 a single accuracy has a 95% interval of about ±3 points (±9 at
  n = 100), so `pilot_100` is only for debugging.

## 8. YAML configs and how to run

Experiments move from TOML to YAML (`.yaml` / `.yml`; existing TOML keeps working). Two
additions over the current format: `extends` pulls in shared sections (LLMs, embedder), and
`sweep` expands one method entry into one arm per value.

```yaml
# configs/common/llms.yaml (shared by every experiment)
llms:
  - {label: gpt-4o-mini, provider: openai, model: gpt-4o-mini, temperature: 0.0,
     price_input_per_m: 0.15, price_output_per_m: 0.60}
  - {label: deepseek-flash, provider: deepseek, model: deepseek-flash, temperature: 0.0,
     thinking: false, price_input_per_m: 0.15, price_output_per_m: 0.60}
  - {label: qwen3-8b, provider: ollama, model: "qwen3:8b", temperature: 0.0, seed: 0,
     thinking: false, num_ctx: 8192, base_url: "http://localhost:11434",
     max_workers: 1}     # MHR full_context overrides num_ctx: 36864
embedding: {provider: local, model: sentence-transformers/multi-qa-mpnet-base-cos-v1}
judge: {provider: openai, model: TBD, temperature: 0.0}   # D2
```

```yaml
# configs/experiments/musique_main.yaml
extends: [configs/common/llms.yaml]
run: {name: musique_main, output_dir: results, index_dir: indexes, index_llm: deepseek-flash,
      max_workers: 8, seed: 0}
dataset:
  name: musique
  path: datasets/musique/musique_ans_v1.0_dev.jsonl
  ids_file: datasets/musique/subsets/main_1000.txt
  corpus: pooled
qa: {answer_format: short, max_answer_tokens: 256}
methods:
  - {name: llm_only}
  - {name: llm_only, label: llm_only_cot, answer_format: cot}
  - {name: llm_only_decomposed}
  - {name: full_context}
  - {name: oracle}
  - name: vanilla_rag
    sweep: {max_context_tokens: [1000, 2000, 4000]}   # -> vanilla_rag@1k, @2k, @4k
  - name: dos_rag
    sweep: {max_context_tokens: [1000, 2000, 4000]}
  - name: raptor
    sweep: {max_context_tokens: [1000, 2000, 4000]}
  - name: hipporag
    sweep: {max_context_tokens: [1000, 2000, 4000]}
```

```bash
.venv/bin/python -m ragbench.run configs/experiments/musique_main.yaml --dry-run   # list runs + call estimate
.venv/bin/python -m ragbench.run configs/experiments/musique_main.yaml --llm gpt-4o-mini
.venv/bin/python -m ragbench.judge results/musique_main            # adds judge verdicts
.venv/bin/python -m ragbench.analyze results/musique_main results/multihop_rag_main
```

Planned config files: `configs/common/llms.yaml`, `configs/experiments/{musique_main,
multihop_rag_main, determinism}.yaml`, and `configs/smoke_mock.yaml` (mock LLM, offline).
Runs stay resumable; the fully resolved config (after `extends` and `sweep`) is written to
`results/<run>/config.resolved.yaml`.

## 9. Run plan and cost

Order:

1. **Code + offline tests** (section 10's list), including a mock-LLM smoke run of both YAML
   configs.
2. **Pilot:** `pilot_100`, GPT-4o-mini, all arms (about $3). Check every field above is
   filled; draw the 100 judge-calibration answers from here plus a DeepSeek pilot.
   Run the same pilot on local Qwen to measure tokens per second and confirm no prompt is
   cut by `num_ctx`.
3. **Judge calibration** (hand labels) before any large judge run.
4. **Main:** `main_1000`, all three LLMs, both datasets. The local Qwen runs go in the
   background while the API runs proceed.
5. **Analysis** and the determinism check.

Approximate volume per LLM (estimates from measured context sizes; index calls from the
MHR RAPTOR mock build and HippoRAG's two calls per passage):

| Phase | Calls | Input tokens | Output tokens (no thinking) |
| --- | --- | --- | --- |
| Answering, MuSiQue | ~19.6k (17 arms + ~2.65 sub-questions per question) | ~32M | ~0.6M |
| Answering, MHR | ~15k | ~67M (32M of it is `full_context`) | ~0.3M |
| Index, RAPTOR (both datasets) | ~2.3k summaries | ~3M | ~0.3M |
| Index, HippoRAG (both datasets) | ~24.8k | ~23M | ~4M |
| **Total** | **~62k** | **~125M** | **~5M** |

Approximate cost (config prices, unverified):

| LLM | Cost | Notes |
| --- | --- | --- |
| GPT-4o-mini | ~$22 | |
| DeepSeek V4 Flash | ~$22 | off-peak price |
| Qwen3 8B (local) | $0 | about 2 to 4 days of machine time; see below |
| Judge | ~$4 to ~$100+ | ~105k calls × ~250 tokens; depends on the judge model (D2) |

**Local run time.** Answering takes about 99M input tokens per LLM. At an assumed 300 to 600
tokens per second of prompt processing for an 8B 4-bit model on the M2 Pro, that is roughly 46
to 92 hours, plus about 35k short answers. Building the indexes locally would add about 26M
input and 5M output tokens; output at roughly 30 tokens per second makes that about two extra
days, so the local model only answers and uses the indexes built by the fixed index LLM (D0).
Dropping MHR `full_context` for Qwen saves about a third of the time (D1). The pilot measures
the real speed before the main run is scheduled.

`full_context` on MHR (32k per question) and HippoRAG indexing are the two biggest items for
every model.

## 10. Code changes after approval

1. **YAML loader** with `extends` and `sweep`, the resolved config written per run, PyYAML
   added to dependencies; TOML still accepted.
2. **Ollama provider:** `provider: ollama` using Ollama's chat API with `think`, `seed` and
   `options.num_ctx`; per-LLM `max_workers`; a check that `prompt_eval_count` covers the whole
   prompt; wall-clock time per question and per run.
3. **Index LLM** (D0): an optional `index_llm` label in the config, so indexes are built and
   cached once by that LLM and shared by every reader. Index cost is reported against it.
4. **Shared budget:** every retrieval method fills to `max_context_tokens` in rank order and
   reports `context_tokens`; RAPTOR's `index_params()` drops query-time keys (`top_k`,
   `max_context_tokens`, `collapse_tree`) so all budgets reuse one tree.
5. **Loader:** MuSiQue decomposition (sub-questions, answers, gold paragraph per hop) into
   `Example.meta`; MHR evidence facts.
6. **Metrics:** `all_gold_retrieved`, `recall_by_hop` (leaf and expanded), MHR `fact_recall`,
   `latency_s`, and summaries by gold-retrieved flag.
7. **Methods:** `oracle`, `llm_only_decomposed`, MHR `full_context` with the 32k cap.
8. **Judge:** `python -m ragbench.judge`, resumable and cached.
9. **Analysis:** `python -m ragbench.analyze` for bootstrap intervals, McNemar, the mixed-effects
   fit, per-hop/type/budget tables and the H1 accuracy-vs-budget figure.
10. **Offline tests** for each of the above with the mock LLM.

## 11. Decisions for review

| # | Decision | Proposal | Alternatives |
| --- | --- | --- | --- |
| D0 | Who builds the RAPTOR and HippoRAG indexes | **Decided: DeepSeek V4 Flash (API) builds every index** (`run.index_llm`); all readers share it | each LLM builds its own index |
| D1 | Local Qwen size and scope | `qwen3:8b`, full matrix, answering only (2 to 4 days) | skip MHR `full_context` for Qwen (about a third faster); `qwen3:14b` at 4-bit (about 9 GB, stronger but slower and tight on 16 GB at 32k context); `qwen3:4b` (fastest) |
| D2 | Judge model | One stronger model outside the three tested ones, fixed for the whole study | GPT-4o-mini (cheap, but weak and judges its own answers) |
| D3 | MHR full context | Gold + seeded distractors to 32k tokens | 16k cap; each model's longest context (not comparable across models) |
| D4 | Budgets | 1k / 2k / 4k on both datasets | add 8k on MHR, where gold evidence averages 5.9k tokens |
| D5 | MHR retrieval recall | Report article-level and fact-level | article-level only |
| D6 | Gold passage "retrieved" for chunked methods | any chunk in context | all chunks in context |
| D7 | `slot_rag` | outside the core matrix | include at all budgets (+12 runs) |
| D8 | HippoRAG on MHR | extract per 512-token chunk, since `openie_max_tokens` = 1024 cannot cover a 16k-token article (Micah) | per article, as now |
| D9 | Thinking arm for H4 | add DeepSeek with thinking on for the three `llm_only*` arms (cheap) | no thinking arm |
| D10 | Judge calibration labellers | two teammates, 100 answers each, same set | one labeller |
