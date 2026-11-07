#!/usr/bin/env python3
"""Independently recompute saved Task 1-3 metrics and consistency checks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.evaluation.metrics import exact_match, normalize_answer, score_predictions
from task1.io import read_jsonl


EXPECTED_K = {"task1": (1, 3, 5, 10), "task2": (1, 3, 5), "task3": (1, 3, 5)}


def contains_answer(text: str, answers: list[str]) -> bool:
    normalized_text = f" {normalize_answer(text)} "
    return any(
        (answer := normalize_answer(candidate)) and f" {answer} " in normalized_text
        for candidate in answers
    )


def close(left: float, right: object) -> bool:
    return isinstance(right, (int, float)) and math.isclose(left, float(right), rel_tol=0.0, abs_tol=1e-12)


def audit_task(task: str, examples: list[dict[str, object]], documents: dict[str, dict[str, object]], artifacts: Path) -> dict[str, object]:
    predictions_path = artifacts / f"{task}_predictions.jsonl"
    metrics_path = artifacts / f"{task}_metrics.json"
    predictions = read_jsonl(predictions_path)
    reported = json.loads(metrics_path.read_text(encoding="utf-8"))
    prediction_by_id = {str(row["id"]): str(row["prediction"]) for row in predictions}
    recomputed = score_predictions(examples, prediction_by_id)

    recall: dict[str, float] = {}
    for k in EXPECTED_K[task]:
        hits = 0
        for example, prediction in zip(examples, predictions, strict=True):
            retrieved_ids = [str(value) for value in prediction["retrieved_passage_ids"]][:k]
            hits += int(any(contains_answer(str(documents[document_id]["text"]), list(example["answers"])) for document_id in retrieved_ids))
        recall[str(k)] = hits / len(examples)
    recomputed["recall_at_k"] = recall

    checks = {
        "count": len(predictions) == len(examples) == int(reported["count"]),
        "ordered_unique_ids": [str(row["id"]) for row in predictions] == [str(row["id"]) for row in examples]
        and len(prediction_by_id) == len(predictions),
        "exact_match": close(float(recomputed["exact_match"]), reported.get("exact_match")),
        "f1": close(float(recomputed["f1"]), reported.get("f1")),
        "recall_at_k": all(close(value, reported.get("recall_at_k", {}).get(k)) for k, value in recall.items()),
    }

    result: dict[str, object] = {"recomputed": recomputed, "checks": checks, "consistent": all(checks.values())}
    if task == "task3":
        correct = supported_correct = incorrect = flagged_incorrect = supported = flagged = 0
        for example, prediction in zip(examples, predictions, strict=True):
            is_correct = bool(exact_match(str(prediction["prediction"]), list(example["answers"])))
            is_supported = prediction["groundedness"] == "supported"
            correct += int(is_correct)
            supported_correct += int(is_correct and is_supported)
            incorrect += int(not is_correct)
            flagged_incorrect += int(not is_correct and not is_supported)
            supported += int(is_supported)
            flagged += int(not is_supported)
        groundedness = {
            "groundedness_check_accuracy": (supported_correct + flagged_incorrect) / len(examples),
            "correct_answers": correct,
            "correct_answers_with_high_lexical_overlap": supported_correct / correct if correct else 0.0,
            "incorrect_answers": incorrect,
            "incorrect_answers_with_low_lexical_overlap": flagged_incorrect / incorrect if incorrect else 0.0,
            "supported_predictions": supported,
            "flagged_predictions": flagged,
        }
        reported_groundedness = reported.get("groundedness", {})
        groundedness_consistent = all(close(float(value), reported_groundedness.get(key)) for key, value in groundedness.items())
        result["groundedness"] = groundedness
        result["checks"]["groundedness_aggregation"] = groundedness_consistent
        result["consistent"] = bool(result["consistent"]) and groundedness_consistent
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/processed/triviaqa_validation_300.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/wikipedia_paragraphs.jsonl"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results_audit.json"))
    args = parser.parse_args()
    examples = read_jsonl(args.questions)
    documents = {str(row["id"]): row for row in read_jsonl(args.corpus)}
    report = {task: audit_task(task, examples, documents, args.artifacts) for task in EXPECTED_K}
    report["all_consistent"] = all(bool(report[task]["consistent"]) for task in EXPECTED_K)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_consistent"]:
        raise SystemExit("One or more saved result files are inconsistent; inspect the checks above.")


if __name__ == "__main__":
    main()
