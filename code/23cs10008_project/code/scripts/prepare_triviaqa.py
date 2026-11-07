#!/usr/bin/env python3
"""Download and persist a fixed TriviaQA rc.nocontext evaluation subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data.triviaqa import deterministic_sample, make_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/processed/triviaqa_validation_300.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Install dependencies with: python -m pip install -r requirements.txt") from error
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=args.split)
    records = make_records(dataset)
    sample = deterministic_sample(records, args.sample_size, args.seed)
    write_jsonl(sample, args.output)
    print(f"Wrote {len(sample)} examples to {args.output} (split={args.split}, seed={args.seed}).")


if __name__ == "__main__":
    main()
