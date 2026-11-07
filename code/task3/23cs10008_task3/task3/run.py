#!/usr/bin/env python3
"""Run Task 3: RRF hybrid retrieval, RAG, and lexical groundedness verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.evaluation.metrics import exact_match, normalize_answer, score_predictions
from shared.reproducibility import set_inference_seed
from task1.bm25 import BM25Index
from task1.io import read_jsonl, write_jsonl
from task2.dense import encode_texts, load_encoder, load_index
from task2.run import generate_answer, load_generator, truncate_passage
from task3.hybrid import has_lexical_support, reciprocal_rank_fusion


def contains_gold_answer(passage: str, answers: list[str]) -> bool:
    normalized_passage = f" {normalize_answer(passage)} "
    return any((normalized_answer := normalize_answer(answer)) and f" {normalized_answer} " in normalized_passage for answer in answers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/processed/triviaqa_validation_300.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/wikipedia_paragraphs.jsonl"))
    parser.add_argument("--bm25-index", type=Path, default=Path("data/indexes/task1_bm25.json"))
    parser.add_argument("--dense-index", type=Path, default=Path("data/indexes/task2_minilm.faiss"))
    parser.add_argument("--retriever-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--retriever-revision", default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--generator-revision", default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--passage-tokens", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-questions", type=int, help="Use only for a smoke test; omit for reported metrics.")
    parser.add_argument("--predictions-output", type=Path, default=Path("artifacts/task3_predictions.jsonl"))
    parser.add_argument("--metrics-output", type=Path, default=Path("artifacts/task3_metrics.json"))
    args = parser.parse_args()
    if min(args.rrf_constant, args.candidate_k, args.retrieval_k, args.passage_tokens, args.max_new_tokens) < 1:
        raise SystemExit("all numeric retrieval and generation settings must be positive")
    set_inference_seed(args.seed)

    examples, corpus = read_jsonl(args.questions), read_jsonl(args.corpus)
    if args.max_questions is not None:
        examples = examples[: args.max_questions]
    if not examples:
        raise SystemExit("No evaluation questions were selected.")
    documents = {str(row["id"]): row for row in corpus}
    bm25 = BM25Index.load(args.bm25_index)
    encoder, (dense_index, dense_ids) = load_encoder(args.retriever_model, args.device, args.retriever_revision), load_index(args.dense_index)
    query_vectors = encode_texts(encoder, [str(row["question"]) for row in examples], batch_size=64)
    _, dense_positions = dense_index.search(query_vectors, args.candidate_k)
    tokenizer, model = load_generator(args.generator_model, not args.no_4bit, args.generator_revision)

    predictions: list[dict[str, object]] = []
    recall_hits = {k: 0 for k in (1, 3, 5) if k <= args.retrieval_k}
    checker_correct = checker_incorrect = correct_total = incorrect_total = 0
    correct_supported = incorrect_unsupported = 0

    for example, positions in zip(examples, dense_positions, strict=True):
        question = str(example["question"])
        bm25_ids = [document_id for document_id, _ in bm25.search(question, args.candidate_k)]
        dense_ranked_ids = [dense_ids[position] for position in positions if position >= 0]
        fused = reciprocal_rank_fusion([bm25_ids, dense_ranked_ids], constant=args.rrf_constant)
        retrieved = [{**documents[document_id], "rrf_score": score} for document_id, score in fused[: args.retrieval_k]]
        for k in recall_hits:
            recall_hits[k] += int(any(contains_gold_answer(str(row["text"]), list(example["answers"])) for row in retrieved[:k]))

        prediction = generate_answer(tokenizer, model, question, retrieved, args.passage_tokens, args.max_new_tokens)
        prompt_passages = [truncate_passage(tokenizer, str(row["text"]), args.passage_tokens) for row in retrieved]
        supported = has_lexical_support(prediction, prompt_passages)
        exact = bool(exact_match(prediction, list(example["answers"])))
        if exact:
            correct_total += 1
            correct_supported += int(supported)
            checker_correct += int(supported)
        else:
            incorrect_total += 1
            incorrect_unsupported += int(not supported)
            checker_incorrect += int(not supported)
        predictions.append({"id": example["id"], "prediction": prediction, "groundedness": "supported" if supported else "flagged_low_lexical_overlap", "retrieved_passage_ids": [row["id"] for row in retrieved], "rrf_scores": [row["rrf_score"] for row in retrieved], "top_passages": [row["text"] for row in retrieved]})

    write_jsonl(predictions, args.predictions_output)
    metrics = score_predictions(examples, {str(row["id"]): str(row["prediction"]) for row in predictions})
    metrics.update({"recall_at_k": {str(k): recall_hits[k] / len(examples) for k in recall_hits}, "retrieval_k": args.retrieval_k, "candidate_k": args.candidate_k, "rrf_constant": args.rrf_constant, "passage_tokens": args.passage_tokens, "max_new_tokens": args.max_new_tokens, "seed": args.seed, "retriever_model": args.retriever_model, "retriever_revision": args.retriever_revision, "generator_model": args.generator_model, "generator_revision": args.generator_revision, "quantization": "4-bit NF4" if not args.no_4bit else "none", "groundedness": {"check": f"normalized generated answer must occur as a whole phrase in at least one {args.passage_tokens}-token retrieved context passage", "groundedness_check_accuracy": (checker_correct + checker_incorrect) / len(examples), "correct_answers": correct_total, "correct_answers_with_high_lexical_overlap": correct_supported / correct_total if correct_total else 0.0, "incorrect_answers": incorrect_total, "incorrect_answers_with_low_lexical_overlap": incorrect_unsupported / incorrect_total if incorrect_total else 0.0, "supported_predictions": sum(row["groundedness"] == "supported" for row in predictions), "flagged_predictions": sum(row["groundedness"] != "supported" for row in predictions)}})
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
