# Results

All three systems were evaluated on the same deterministic sample of 300
TriviaQA `rc.nocontext` validation questions. The reported values were
independently recomputed from the saved prediction records by
`code/scripts/audit_results.py`; every audit consistency check passed.

| Task | System | Exact Match | F1 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | BM25 + heuristic answer extraction | 3.67% | 9.49% | 28.67% | 38.00% | 42.00% | 47.67% |
| 2 | MiniLM dense retrieval + Qwen RAG | 32.67% | 37.99% | 24.67% | 40.00% | 43.67% | N/A |
| 3 | BM25/MiniLM reciprocal-rank fusion + Qwen RAG | 34.33% | 40.42% | 30.33% | 42.00% | 46.00% | N/A |

## Summary

Task 3 produced the strongest overall result, improving on Task 2 by 1.66
percentage points in Exact Match and 2.43 points in F1. Its hybrid retrieval
also improved Recall@5 from 43.67% to 46.00%. Both RAG systems substantially
outperformed Task 1's heuristic answer extraction, while Task 1 achieved the
best reported Recall@1 among the first two individual retrievers.

## Task 3 groundedness check

The groundedness checker labels an answer as supported when its complete
normalized phrase occurs in at least one of the five retrieved context
passages supplied to the generator.

| Gold answer outcome | Supported | Flagged |
| --- | ---: | ---: |
| Exactly correct | 88 | 15 |
| Incorrect | 144 | 53 |

- Groundedness-check accuracy: 47.00%
- Supported predictions: 232
- Flagged predictions: 68
- Correct answers with high lexical overlap: 85.44%
- Incorrect answers with low lexical overlap: 26.90%

The check is useful as a warning signal because 53 of the 68 flagged answers
were incorrect, but lexical support alone is not sufficient evidence that a
generated answer is correct.

## Reproducing and verifying the results

Run the complete pipeline:

```bash
python code/run_all.py --device cuda --seed 42
```

Recompute and verify the saved metrics:

```bash
python code/scripts/audit_results.py
```

The detailed setup, dataset preparation, and per-task commands are documented
in `README.md` and the task-specific README files under `code/`.
