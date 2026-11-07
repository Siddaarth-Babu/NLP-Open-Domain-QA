"""Deterministic text processing and answer-span heuristics for Task 1."""

from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer shared by indexing and BM25 retrieval."""
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


def split_paragraphs(article_text: str, minimum_characters: int = 80) -> list[str]:
    """Split MediaWiki plaintext extracts into useful paragraph documents."""
    paragraphs = (re.sub(r"\s+", " ", part).strip() for part in article_text.split("\n"))
    return [paragraph for paragraph in paragraphs if len(paragraph) >= minimum_characters]


def best_sentence(question: str, passage: str) -> str:
    """Select the sentence with greatest lexical overlap with the question."""
    question_terms = set(tokenize(question))
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", passage) if sentence.strip()]
    if not sentences:
        return passage.strip()
    return max(sentences, key=lambda sentence: (len(question_terms & set(tokenize(sentence))), -len(sentence)))


def extract_answer(question: str, passage: str) -> str:
    """Return a short lexical span from the best-overlap sentence.

    This intentionally simple baseline favors numbers and capitalized entity spans;
    Task 1 is a retrieve-and-*heuristically*-extract system, not a learned reader.
    """
    sentence = best_sentence(question, passage)
    question_terms = set(tokenize(question))
    number_spans = re.findall(r"\b\d{1,4}(?:[,.]\d+)?(?:\s?(?:BC|AD|CE|BCE|%|km|m|million|billion))?\b", sentence, re.IGNORECASE)
    if number_spans and any(word in tokenize(question) for word in ("when", "year", "how", "many", "much")):
        return number_spans[0]
    entity_spans = re.findall(r"\b(?:[A-Z][\w'’-]*)(?:\s+(?:[A-Z][\w'’-]*|of|the|and|de|von|van))*", sentence)
    useful = [span.strip() for span in entity_spans if set(tokenize(span)) - question_terms]
    if useful:
        return max(useful, key=lambda span: (len(tokenize(span)), -sentence.find(span)))
    words = tokenize(sentence)
    remaining = [word for word in words if word not in question_terms]
    return " ".join(remaining[:5]) or sentence
