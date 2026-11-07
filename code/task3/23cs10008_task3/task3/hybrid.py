"""RRF fusion and transparent lexical groundedness checks for Task 3."""

from __future__ import annotations

from collections.abc import Iterable

from shared.evaluation.metrics import normalize_answer


def reciprocal_rank_fusion(rankings: Iterable[Iterable[str]], constant: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked document IDs with RRF using one-based ranks."""
    if constant < 1:
        raise ValueError("RRF constant must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[str(document_id)] = scores.get(str(document_id), 0.0) + 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def has_lexical_support(answer: str, passages: Iterable[str]) -> bool:
    """Require the normalized generated answer as a whole phrase in evidence."""
    normalized_answer = normalize_answer(answer)
    if not normalized_answer or normalized_answer == "i dont know":
        return False
    return any(f" {normalized_answer} " in f" {normalize_answer(passage)} " for passage in passages)
