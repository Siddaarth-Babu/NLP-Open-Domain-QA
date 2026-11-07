"""A transparent, serializable BM25 implementation for paragraph retrieval."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from task1.text import tokenize


class BM25Index:
    def __init__(self, document_ids: list[str], document_lengths: list[int], postings: dict[str, list[list[int]]], average_document_length: float, k1: float = 1.5, b: float = 0.75):
        self.document_ids = document_ids
        self.document_lengths = document_lengths
        self.postings = postings
        self.average_document_length = average_document_length
        self.k1 = k1
        self.b = b

    @classmethod
    def build(cls, documents: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        postings: dict[str, list[list[int]]] = defaultdict(list)
        document_lengths: list[int] = []
        for document_number, document in enumerate(documents):
            terms = tokenize(str(document["text"]))
            document_lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                postings[term].append([document_number, frequency])
        average = sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
        return cls([str(document["id"]) for document in documents], document_lengths, dict(postings), average, k1, b)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if k < 1 or not self.document_ids:
            return []
        scores: defaultdict[int, float] = defaultdict(float)
        total_documents = len(self.document_ids)
        for term in set(tokenize(query)):
            term_postings = self.postings.get(term, [])
            if not term_postings:
                continue
            document_frequency = len(term_postings)
            idf = math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
            for document_number, frequency in term_postings:
                length_ratio = self.document_lengths[document_number] / self.average_document_length if self.average_document_length else 0.0
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[document_number] += idf * frequency * (self.k1 + 1) / denominator
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.document_ids[item[0]]))[:k]
        return [(self.document_ids[document_number], score) for document_number, score in ranked]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        return cls(**json.loads(path.read_text(encoding="utf-8")))
