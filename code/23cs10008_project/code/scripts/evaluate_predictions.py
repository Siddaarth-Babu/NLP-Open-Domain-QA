#!/usr/bin/env python3
"""Evaluate JSONL predictions against a prepared TriviaQA subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.evaluation.metrics import score_predictions


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", type=Path)
    parser.add_argument("predictions", type=Path, help="JSONL records containing id and prediction")
    args = parser.parse_args()
    examples = read_jsonl(args.examples)
    predictions = {str(row["id"]): str(row["prediction"]) for row in read_jsonl(args.predictions)}
    print(json.dumps(score_predictions(examples, predictions), indent=2))


if __name__ == "__main__":
    main()
