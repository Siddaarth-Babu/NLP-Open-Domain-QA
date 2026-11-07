"""Alias-aware answer scoring and explicitly labeled retrieval proxies.

Recall is success@k: fraction of questions for which at least one retrieved
passage contains a normalized gold alias as a whole token sequence. It is NOT
the fraction of all relevant documents retrieved, nor a semantic relevance label.
"""
import re
import string
from collections import Counter


def normalize(text):
    # Conventional QA normalization: lowercase, punctuation, articles, whitespace.
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def gold_answers(answer):
    values = [answer["value"]] + list(answer.get("aliases", []))
    return list(dict.fromkeys(str(v) for v in values if normalize(v)))


def token_f1(prediction, reference):
    a, b = normalize(prediction).split(), normalize(reference).split()
    if not a or not b:
        return float(a == b)
    overlap = sum((Counter(a) & Counter(b)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(b)
    return 2 * precision * recall / (precision + recall)


def answer_scores(prediction, references):
    if not references:
        raise ValueError("Every scored question needs a nonempty gold-answer list")
    return {
        "em": float(any(normalize(prediction) == normalize(g) for g in references)),
        "f1": max(token_f1(prediction, g) for g in references),
    }


def contains_alias(text, references):
    haystack = " " + normalize(text) + " "
    return any(" " + normalize(g) + " " in haystack for g in references if normalize(g))


def recall_at_k(passages, references, k):
    return float(any(contains_alias(p["text"], references) for p in passages[:k]))


def overlap_score(answer, evidence):
    """Token-multiset answer coverage; empty/abstained answers have score zero.

    This is only a heuristic. It does not check subject-relation consistency,
    negation, contradictions, or whether the question has actually been answered.
    """
    a, e = Counter(normalize(answer).split()), Counter(normalize(evidence).split())
    return sum((a & e).values()) / sum(a.values()) if a else 0.0


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty experiment")
    keys = ["em", "f1", "recall@1", "recall@3", "recall@5", "recall@10"]
    return {"n": len(rows), **{k: 100 * sum(r[k] for r in rows) / len(rows) for k in keys}}
