#!/usr/bin/env python3
"""Embed the shared Wikipedia corpus with MiniLM and write a FAISS cosine index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task1.io import read_jsonl
from task2.dense import encode_texts, load_encoder, save_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/wikipedia_paragraphs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/indexes/task2_minilm.faiss"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model-revision", default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("batch-size must be positive")
    documents = read_jsonl(args.corpus)
    if not documents:
        raise SystemExit("Corpus is empty.")
    encoder = load_encoder(args.model, args.device, args.model_revision)
    vectors = encode_texts(encoder, [str(row["text"]) for row in documents], args.batch_size)
    save_index(vectors, [str(row["id"]) for row in documents], args.output)
    metadata = {"model": args.model, "model_revision": args.model_revision, "similarity": "cosine via normalized embeddings and IndexFlatIP", "documents": len(documents), "dimensions": int(vectors.shape[1])}
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
