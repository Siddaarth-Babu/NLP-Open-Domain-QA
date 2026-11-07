#!/usr/bin/env python3
"""Prepare missing assets, run Tasks 1-3 in order, then audit all results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def run(*arguments: str) -> None:
    command = [PYTHON, *arguments]
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def is_older(target: Path, source: Path) -> bool:
    return not target.exists() or target.stat().st_mtime < source.stat().st_mtime


def dense_index_needs_build(index: Path, corpus: Path, rebuild: bool) -> bool:
    ids = index.with_suffix(index.suffix + ".ids.json")
    metadata = index.with_suffix(index.suffix + ".metadata.json")
    if rebuild or any(is_older(path, corpus) for path in (index, ids, metadata)):
        return True
    try:
        saved = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return saved.get("model_revision") != MINILM_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-indexes", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda"], default="cuda")
    args = parser.parse_args()

    questions = ROOT / "data/processed/triviaqa_validation_300.jsonl"
    corpus = ROOT / "data/processed/wikipedia_paragraphs.jsonl"
    manifest = ROOT / "data/processed/wikipedia_discovery_manifest.jsonl"
    bm25 = ROOT / "data/indexes/task1_bm25.json"
    dense = ROOT / "data/indexes/task2_minilm.faiss"

    if not questions.exists():
        run("code/scripts/prepare_triviaqa.py", "--split", "validation", "--sample-size", "300", "--seed", str(args.seed))
    if not corpus.exists() or not manifest.exists():
        run("code/task1/build_corpus.py", "--pages-per-question", "3", "--delay-seconds", "1.0", "--retry-forever")
    if args.rebuild_indexes or is_older(bm25, corpus):
        run("code/task1/build_index.py", "--k1", "1.5", "--b", "0.75")

    run("code/task1/run.py", "--answer-k", "5", "--recall-k", "1", "3", "5", "10")

    if dense_index_needs_build(dense, corpus, args.rebuild_indexes):
        run("code/task2/build_index.py", "--device", args.device, "--batch-size", "128")
    run("code/task2/run.py", "--device", args.device, "--seed", str(args.seed), "--recall-k", "1", "3", "5")
    run("code/task3/run.py", "--device", args.device, "--seed", str(args.seed))
    run("code/scripts/audit_results.py")


if __name__ == "__main__":
    main()
