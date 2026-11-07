"""Task runners with per-question checkpoints and comparable evaluation."""
import time
from datetime import datetime, timezone
from pathlib import Path
from common.io import ROOT, config, digest, file_hash, read_json, read_jsonl, write_json, write_jsonl, set_seed
from common.metrics import answer_scores, recall_at_k, summarize


def load_data():
    questions = read_jsonl(ROOT / "data/questions.jsonl")
    corpus = read_jsonl(ROOT / "data/corpus.jsonl")
    if not questions or not corpus:
        raise RuntimeError("Run python code/run.py prepare first")
    manifest = read_json(ROOT / "data/corpus_manifest.json")
    if file_hash(ROOT / "data/corpus.jsonl") != manifest["corpus_sha256"]:
        raise ValueError("Corpus was modified after freezing")
    if file_hash(ROOT / "data/questions.jsonl") != manifest["questions_sha256"]:
        raise ValueError("Question sample changed after corpus construction; use a fresh data directory")
    return questions, corpus


def run_task(task, resources=None):
    cfg = config()
    environment = set_seed(cfg["seed"], cfg["threads"])
    questions, corpus = load_data()
    resources = {} if resources is None else resources
    started = time.monotonic()
    if task in ("task1", "task3") and "bm25" not in resources:
        from task1.baseline import BM25Retriever
        resources["bm25"] = BM25Retriever(corpus, cfg["bm25_k1"], cfg["bm25_b"])
    if task == "task1" and "extractor" not in resources:
        from task1.baseline import ClassicalExtractor
        resources["extractor"] = ClassicalExtractor()
    if task in ("task2", "task3"):
        if "dense" not in resources:
            from task2.dense import DenseRetriever
            resources["dense"] = DenseRetriever(corpus, cfg, ROOT / "cache", file_hash(ROOT / "data/corpus.jsonl"))
        if "generator" not in resources:
            from task2.generator import Generator
            resources["generator"] = Generator(cfg)
    identity = digest({"config": cfg, "torch": environment["torch"], "cuda": environment["cuda"],
                       "corpus": file_hash(ROOT / "data/corpus.jsonl"),
                       "questions": file_hash(ROOT / "data/questions.jsonl"), "task": task,
                       "implementation": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / "code").rglob("*.py"))
                                          if p.name not in ("make_report.py", "analyze.py", "run.py")}})
    folder = ROOT / "results" / task
    checkpoint_dir = folder / "records"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, question in enumerate(questions, 1):
        checkpoint = checkpoint_dir / f'{question["question_id"]}.json'
        if checkpoint.exists() and read_json(checkpoint).get("run_identity") == identity:
            rows.append(read_json(checkpoint))
            continue
        tick = time.monotonic()
        q = question["question"]
        bm25 = resources["bm25"].search(q, cfg["retrieval_depth"]) if task in ("task1", "task3") else None
        dense = resources["dense"].search(q, cfg["retrieval_depth"]) if task in ("task2", "task3") else None
        if task == "task1":
            ranking = bm25
        elif task == "task2":
            ranking = dense
        else:
            from task3.hybrid import reciprocal_rank_fusion
            ranking = reciprocal_rank_fusion(bm25, dense, cfg["rrf_constant"], cfg["retrieval_depth"])
        retrieved = [corpus[i] for i, _ in ranking]
        if task == "task1":
            answer, sentence = resources["extractor"].answer(q, retrieved[0]["text"])
            evidence, input_tokens = retrieved[:cfg["answer_top_k"]], None
        else:
            answer, evidence, input_tokens = resources["generator"].answer(q, retrieved)
            sentence = None
        from task3.groundedness import check
        row = {"question_id": question["question_id"], "question": q, "answers": question["answers"],
               "prediction": answer, **answer_scores(answer, question["answers"]),
               **{f"recall@{k}": recall_at_k(retrieved, question["answers"], k) for k in (1, 3, 5, 10)},
               "ranking": [{"passage_id": corpus[i]["passage_id"], "score": score} for i, score in ranking],
               "evidence": evidence, "selected_sentence": sentence, "input_tokens": input_tokens,
               "seconds": time.monotonic() - tick, "run_identity": identity,
               **check(answer, evidence, cfg["overlap_threshold"])}
        write_json(checkpoint, row)
        rows.append(row)
        if number % 10 == 0 or number == 1:
            print(f'{task}: {number}/{len(questions)} | prediction={answer!r} | {row["seconds"]:.2f}s', flush=True)
    write_jsonl(folder / "predictions.jsonl", rows)
    summary = summarize(rows)
    summary.update({"task": task, "run_identity": identity, "config": cfg, "environment": environment,
                    "corpus_content_digest": digest(corpus), "questions_content_digest": digest(questions),
                    "wall_seconds_this_invocation": time.monotonic() - started,
                    "sum_question_seconds": sum(r["seconds"] for r in rows),
                    "abstentions": sum(r["abstained"] for r in rows),
                    "retrieval_metric": "answer-containing passage success@k (alias-matching proxy)",
                    "finished_utc": datetime.now(timezone.utc).isoformat()})
    write_json(folder / "metrics.json", summary)
    print(summary, flush=True)
    return resources
