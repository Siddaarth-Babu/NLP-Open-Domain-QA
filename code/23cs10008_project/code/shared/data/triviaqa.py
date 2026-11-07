"""Utilities for creating a deterministic TriviaQA evaluation subset."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def accepted_answers(answer: Mapping[str, Any]) -> list[str]:
    """Return canonical answer and all non-empty aliases without duplicates."""
    candidates = [answer.get("value", ""), *answer.get("aliases", [])]
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if isinstance(candidate, str) and (cleaned := candidate.strip()) and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique


def make_records(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert dataset rows to the compact project JSONL schema."""
    records: list[dict[str, Any]] = []
    for row in rows:
        answers = accepted_answers(row["answer"])
        if answers:
            records.append({"id": str(row.get("question_id", row.get("id", len(records)))), "question": row["question"].strip(), "answers": answers})
    return records


def deterministic_sample(records: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    """Sample records reproducibly and sort by ID for stable files."""
    if not 1 <= sample_size <= len(records):
        raise ValueError(f"sample_size must be between 1 and {len(records)}")
    return sorted(random.Random(seed).sample(records, sample_size), key=lambda record: record["id"])


def write_jsonl(records: Iterable[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
