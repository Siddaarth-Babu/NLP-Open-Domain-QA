#!/usr/bin/env python3
"""Run Task 1 BM25 retrieve-and-extract and write predictions plus metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.evaluation.metrics import normalize_answer, score_predictions
from task1.bm25 import BM25Index
from task1.io import read_jsonl, write_jsonl
from task1.text import extract_answer


def contains_answer(passage: str, answers: list[str]) -> bool:
    """Match a normalized answer as a complete token phrase, not a substring."""
    normalized_passage = f" {normalize_answer(passage)} "
    return any(
        (normalized_answer := normalize_answer(answer))
        and f" {normalized_answer} " in normalized_passage
        for answer in answers
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/processed/triviaqa_validation_300.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/wikipedia_paragraphs.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/indexes/task1_bm25.json"))
    parser.add_argument("--predictions-output", type=Path, default=Path("artifacts/task1_predictions.jsonl"))
    parser.add_argument("--metrics-output", type=Path, default=Path("artifacts/task1_metrics.json"))
    parser.add_argument("--answer-k", type=int, default=5)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--max-questions", type=int, help="Use only for a smoke test; omit for reported metrics.")
    args = parser.parse_args()
    if args.answer_k < 1 or any(k < 1 for k in args.recall_k):
        raise SystemExit("all k values must be positive")
    examples = read_jsonl(args.questions)
    if args.max_questions is not None:
        if args.max_questions < 1:
            raise SystemExit("max-questions must be positive")
        examples = examples[:args.max_questions]
    corpus, index = read_jsonl(args.corpus), BM25Index.load(args.index)
    documents = {str(document["id"]): document for document in corpus}
    maximum_k = max(args.answer_k, *args.recall_k)
    predictions: list[dict[str, object]] = []
    recall_hits = {k: 0 for k in args.recall_k}
    for example in examples:
        results = index.search(str(example["question"]), maximum_k)
        retrieved = [{**documents[document_id], "bm25_score": score} for document_id, score in results]
        for k in args.recall_k:
            recall_hits[k] += int(any(contains_answer(str(document["text"]), list(example["answers"])) for document in retrieved[:k]))
        answer_passages = retrieved[:args.answer_k]
        prediction = extract_answer(str(example["question"]), str(answer_passages[0]["text"])) if answer_passages else ""
        predictions.append({"id": example["id"], "prediction": prediction, "retrieved_passage_ids": [document["id"] for document in retrieved], "top_passage": answer_passages[0]["text"] if answer_passages else ""})
    write_jsonl(predictions, args.predictions_output)
    prediction_map = {str(record["id"]): str(record["prediction"]) for record in predictions}
    metrics = score_predictions(examples, prediction_map)
    metrics["recall_at_k"] = {str(k): recall_hits[k] / len(examples) for k in args.recall_k}
    metrics["answer_k"] = args.answer_k
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
