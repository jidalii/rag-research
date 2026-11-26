# Feature Log

Changes made to `vanilla-dos-rag/` compared with the original GitHub repository.

## Writing rules

- Log the modifications made to the original GitHub repository.
- Focus on code and features. Don't add a documentation section.
- Describe each feature.
- Include a few code snippets where useful, without overloading the log.
- Don't list exact files.

## Original source

- **Repository:** [alex-laitenberger/dos-rag-eval](https://github.com/alex-laitenberger/dos-rag-eval)
- **Paper:** [Stronger Baselines for Retrieval-Augmented Generation with Long-Context Language Models](https://aclanthology.org/2025.emnlp-main.1656/)
- **License:** MIT, see [`../LICENSE`](../LICENSE)

## 1. Switch between DOS RAG and vanilla RAG

The original code always puts retrieved chunks in document order. A new `--mode` option on the experiment scripts adds vanilla RAG as a second choice.

| Mode            | Order of retrieved chunks in the prompt   |
| --------------- | ----------------------------------------- |
| `dos` (default) | Position in the document                  |
| `vanilla`       | Similarity to the question, highest first |

Both modes use the same embeddings and select the same chunks. Only the order differs:

```python
if mode == "dos":
    ordered_nodes = sorted(selected_nodes, key=lambda node: node.index)
else:
    ordered_nodes = selected_nodes  # already most similar first
```

To run each mode, for example on QuALITY:

```bash
python -m source.experiments.quality.run_experiment --mode dos
python -m source.experiments.quality.run_experiment --mode vanilla
```

Answer filenames start with `dos-rag-` or `vanilla-rag-`, so the results can be told apart. An unknown mode raises an error.

## 2. Tests

Added two tests to `test/test_rag.py`:

- `test_retrieve_modes_differ_only_in_order`: both modes select the same chunks in different orders, and `dos` is the default.
- `test_retrieve_invalid_mode`: an unknown mode raises an error.
