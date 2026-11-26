# Feature Log

Changes made compared with the original DOS RAG GitHub repository.

## Original source

- **Repository:** [alex-laitenberger/dos-rag-eval](https://github.com/alex-laitenberger/dos-rag-eval)
- **Paper:** [Stronger Baselines for Retrieval-Augmented Generation with Long-Context Language Models](https://aclanthology.org/2025.emnlp-main.1656/)
- **License:** MIT

## 1. Three chunk orderings

The original always puts retrieved chunks in document order. There are now three methods. They retrieve the same chunks and differ only in the order. For search results `A8, B2, C5, A3, B1` (best first; letter = document, number = position in it):

| Method        | Rule                                                               | Prompt order     |
| ------------- | ------------------------------------------------------------------ | ---------------- |
| `vanilla_rag` | Similarity to the question                                         | `A8 B2 C5 A3 B1` |
| `dos_rag`     | Grouped by document, in reading order inside each                  | `A3 A8 B1 B2 C5` |
| `slot_rag`    | Each document keeps its positions, filled in its own reading order | `A3 B1 C5 A8 B2` |

`slot_rag` is new. It applies reading order without grouping documents, so the two effects can be compared.

```python
queues = {pid: iter(chunks) for pid, chunks in _group_by_document(ranked).items()}
return [next(queues[pid]) for pid, _ in ranked]
```

## 2. Search across many documents

The original searches one long document per question. The methods now search a collection of documents and run on MuSiQue and MultiHop-RAG.

- **What a document is:** a paragraph in MuSiQue, a whole article in MultiHop-RAG.
- **Order between documents:** there is no original order, so DOS puts the document with the best-ranked chunk first. This is a rule added here.
- **MuSiQue limit:** reading order is restored only inside each short paragraph. The data gives no order between paragraphs.

## 3. Part of the team's shared harness

The methods plug into the team's experiment runner, so they share its datasets, answer prompt, scoring and token and cost tracking.

- **Embeddings:** the team's shared model replaces the original Snowflake model.
- **Retrieval limits:** up to 10 chunks and 2,000 tokens.
- **Retrieval recall:** counts only documents with a chunk in the prompt.
- **Order tracking:** each answer records the chunks in prompt order and whether that order differs from vanilla.

## 4. Splitter fix to keep reading order

The original splitter is used, with chunks of about 100 tokens. It could place a long stretch of text with nothing to split on ahead of earlier sentences, so chunk numbers stopped matching reading order. The earlier text is now saved first. Other text is chunked as before.

```python
if sub_token_count > max_tokens:
    handle_chunk(current_chunk)  # save earlier text first
    current_chunk, current_length = [], 0
```

The sentence-splitting library is pinned to the original's version (NLTK 3.9.1), because a newer version split some text differently. Its sentence data now downloads on first use instead of on import.

## 5. Tests

New tests cover each ordering, the splitter, the token limit, and chunk numbering on both datasets. They also check that all three methods use the same chunks. Fake models keep them offline and free.

