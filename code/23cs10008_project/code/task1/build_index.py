#!/usr/bin/env python3
"""Build a serialized BM25 index over a Wikipedia paragraph JSONL corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task1.bm25 import BM25Index
from task1.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/wikipedia_paragraphs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/indexes/task1_bm25.json"))
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()
    if args.k1 <= 0 or not 0 <= args.b <= 1:
        raise SystemExit("k1 must be positive and b must be in [0, 1]")
    documents = read_jsonl(args.corpus)
    if not documents:
        raise SystemExit("Corpus is empty.")
    index = BM25Index.build(documents, k1=args.k1, b=args.b)
    index.save(args.output)
    print(f"Indexed {len(documents)} paragraphs and {len(index.postings)} unique terms at {args.output}.")


if __name__ == "__main__":
    main()
