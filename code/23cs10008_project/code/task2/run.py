#!/usr/bin/env python3
"""Run Task 2: MiniLM/FAISS retrieval followed by deterministic 4-bit RAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.evaluation.metrics import normalize_answer, score_predictions
from shared.reproducibility import set_inference_seed
from task1.io import read_jsonl, write_jsonl
from task2.dense import encode_texts, load_encoder, load_index


def contains_answer(passage: str, answers: list[str]) -> bool:
    normalized_passage = f" {normalize_answer(passage)} "
    return any((answer := normalize_answer(value)) and f" {answer} " in normalized_passage for value in answers)


def truncate_passage(tokenizer, passage: str, token_limit: int) -> str:
    ids = tokenizer(passage, add_special_tokens=False, truncation=True, max_length=token_limit)["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)


def load_generator(model_name: str, use_4bit: bool, revision: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Task 2 generation requires CUDA. Start Docker with --gpus all.")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    return tokenizer, AutoModelForCausalLM.from_pretrained(model_name, revision=revision, **kwargs)


def generate_answer(tokenizer, model, question: str, passages: list[dict[str, object]], passage_tokens: int, max_new_tokens: int) -> str:
    context = "\n\n".join(f"[{number}] {truncate_passage(tokenizer, str(row['text']), passage_tokens)}" for number, row in enumerate(passages, 1))
    user_message = f"Passages:\n{context}\n\nQuestion: {question}"
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer using only the supplied passages. Return only the shortest answer phrase: no full sentence, explanation, or label. Multiword names, titles, and terms are allowed. If the passages do not support an answer, reply exactly: I don't know."},
            {"role": "user", "content": user_message},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500).to(model.device)
    with torch.inference_mode():
        # Greedy decoding is deterministic; temperature applies only to sampling,
        # so it is intentionally omitted when do_sample=False.
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    completion = output[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(completion, skip_special_tokens=True).strip()
    return decoded.splitlines()[0] if decoded else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/processed/triviaqa_validation_300.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/wikipedia_paragraphs.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/indexes/task2_minilm.faiss"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model-revision", default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--generator-revision", default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--passage-tokens", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--predictions-output", type=Path, default=Path("artifacts/task2_predictions.jsonl"))
    parser.add_argument("--metrics-output", type=Path, default=Path("artifacts/task2_metrics.json"))
    args = parser.parse_args()
    if min(args.retrieval_k, args.passage_tokens, args.max_new_tokens, *args.recall_k) < 1:
        raise SystemExit("retrieval-k, recall-k, passage-tokens, and max-new-tokens must be positive")
    set_inference_seed(args.seed)
    examples, corpus = read_jsonl(args.questions), read_jsonl(args.corpus)
    if args.max_questions is not None:
        examples = examples[: args.max_questions]
    if not examples:
        raise SystemExit("No evaluation questions were selected.")
    documents = {str(row["id"]): row for row in corpus}
    encoder, (index, ids) = load_encoder(args.model, args.device, args.model_revision), load_index(args.index)
    query_vectors = encode_texts(encoder, [str(row["question"]) for row in examples], batch_size=64)
    maximum_k = max(args.retrieval_k, *args.recall_k)
    scores, positions = index.search(query_vectors, maximum_k)
    tokenizer, model = load_generator(args.generator_model, not args.no_4bit, args.generator_revision)
    predictions, recall_hits = [], {k: 0 for k in args.recall_k}
    for example, row_scores, row_positions in zip(examples, scores, positions, strict=True):
        retrieved = [{**documents[ids[position]], "dense_score": float(score)} for score, position in zip(row_scores, row_positions, strict=True) if position >= 0]
        for k in args.recall_k:
            recall_hits[k] += int(any(contains_answer(str(row["text"]), list(example["answers"])) for row in retrieved[:k]))
        answer_passages = retrieved[: args.retrieval_k]
        prediction = generate_answer(tokenizer, model, str(example["question"]), answer_passages, args.passage_tokens, args.max_new_tokens)
        predictions.append({"id": example["id"], "prediction": prediction, "retrieved_passage_ids": [row["id"] for row in retrieved], "top_passages": [row["text"] for row in retrieved]})
    write_jsonl(predictions, args.predictions_output)
    metrics = score_predictions(examples, {str(row["id"]): str(row["prediction"]) for row in predictions})
    metrics.update({"recall_at_k": {str(k): recall_hits[k] / len(examples) for k in args.recall_k}, "retrieval_k": args.retrieval_k, "passage_tokens": args.passage_tokens, "max_new_tokens": args.max_new_tokens, "seed": args.seed, "retriever_model": args.model, "retriever_revision": args.model_revision, "generator_model": args.generator_model, "generator_revision": args.generator_revision, "quantization": "4-bit NF4" if not args.no_4bit else "none"})
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
