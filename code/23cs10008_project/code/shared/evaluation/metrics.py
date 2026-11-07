"""String QA metrics with support for all accepted answer aliases."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(answer) for answer in gold_answers))


def token_f1(prediction: str, gold_answer: str) -> float:
    prediction_tokens, gold_tokens = normalize_answer(prediction).split(), normalize_answer(gold_answer).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    matched = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if not matched:
        return 0.0
    precision, recall = matched / len(prediction_tokens), matched / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    return max((token_f1(prediction, answer) for answer in gold_answers), default=0.0)


def score_predictions(examples: Sequence[dict[str, object]], predictions: dict[str, str]) -> dict[str, float]:
    if not examples:
        raise ValueError("cannot score an empty evaluation set")
    em_scores, f1_scores = [], []
    for example in examples:
        answers = example["answers"]
        if not isinstance(answers, list) or not all(isinstance(answer, str) for answer in answers):
            raise ValueError(f"example {example['id']} has invalid answers")
        prediction = predictions.get(str(example["id"]), "")
        em_scores.append(exact_match(prediction, answers))
        f1_scores.append(best_f1(prediction, answers))
    return {"count": float(len(examples)), "exact_match": sum(em_scores) / len(examples), "f1": sum(f1_scores) / len(examples)}
