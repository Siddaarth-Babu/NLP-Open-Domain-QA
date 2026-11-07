"""Produce comparison tables, charts, qualitative examples, and support audits."""
import csv
import random
from collections import Counter
from pathlib import Path
from common.io import ROOT, config, read_json, read_jsonl, write_json, digest


def percent(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def support_metrics(rows, task):
    correct = [r for r in rows if r["em"] == 1]
    incorrect = [r for r in rows if r["em"] == 0]
    retained = [r for r in rows if r["high_overlap"]]
    metrics = {"n": len(rows), "correct_n": len(correct), "incorrect_n": len(incorrect),
               "correct_with_high_overlap_percent": percent(sum(r["high_overlap"] for r in correct), len(correct)),
               "incorrect_with_low_overlap_percent": percent(sum(not r["high_overlap"] for r in incorrect), len(incorrect)),
               "retained_coverage_percent": percent(len(retained), len(rows)),
               "retained_em_percent": percent(sum(r["em"] for r in retained), len(retained)),
               "flagged_as_unsupported": sum(not r["high_overlap"] for r in rows),
               "groundedness_check_accuracy_percent": None,
               "groundedness_note": "Requires independent answer-in-context support labels, not gold-answer correctness."}
    label_path = ROOT / "results/support_labels.csv"
    if label_path.exists():
        labels = list(csv.DictReader(label_path.open(encoding="utf-8")))
        mapping = {r["question_id"]: r for r in rows}
        selected = [r for r in labels if r["task"] == task and r.get("supported") in ("0", "1")]
        if len({r["question_id"] for r in selected}) != len(selected):
            raise ValueError("Duplicate support audit labels")
        if any(r["question_id"] not in mapping for r in selected):
            raise ValueError("Audit labels reference questions outside this experiment")
        if any(r.get("claim_digest") != digest({"question": mapping[r["question_id"]]["question"], "prediction": mapping[r["question_id"]]["prediction"], "evidence": [e["text"] for e in mapping[r["question_id"]]["evidence"]]}) for r in selected):
            raise ValueError("Support labels are stale: review the changed question, answer, or evidence before analysis")
        if selected:
            nonempty = [r for r in selected if mapping[r["question_id"]]["prediction"].strip()]
            metrics["groundedness_check_accuracy_percent"] = percent(sum(
                int(mapping[r["question_id"]]["high_overlap"]) == int(r["supported"]) for r in nonempty), len(nonempty))
            metrics["support_audit_n"] = len(nonempty)
            metrics["support_audit_total_n"] = len(selected)
            metrics["audit_abstentions_excluded"] = len(selected) - len(nonempty)
            metrics["support_audit_annotators"] = sorted({r.get("annotator", "unspecified") for r in selected})
            metrics["groundedness_note"] = "Accuracy on independently reviewed nonempty pilot answers; abstentions excluded; not full-dataset accuracy."
    return metrics


def analyze():
    cfg = config()
    metrics, all_rows, support, examples = {}, {}, {}, {}
    for task in ("task1", "task2", "task3"):
        path = ROOT / "results" / task / "metrics.json"
        if not path.exists():
            continue
        metrics[task] = read_json(path)
        rows = read_jsonl(path.parent / "predictions.jsonl")
        all_rows[task] = rows
        if task != "task1":
            support[task] = support_metrics(rows, task)
        # Deterministic, disclosed examples from outcome strata (not representative).
        groups = {"correct": [r for r in rows if r["em"]],
                  "incorrect": [r for r in rows if not r["em"]],
                  "high_overlap_incorrect": [r for r in rows if r["high_overlap"] and not r["em"]],
                  "low_overlap_correct": [r for r in rows if not r["high_overlap"] and r["em"]]}
        if task == "task3" and "task2" in all_rows:
            dense_by_id = {r["question_id"]: r for r in all_rows["task2"]}
            # Prefer changed outcomes to explain the controlled retrieval comparison.
            groups["correct"].sort(key=lambda r: dense_by_id[r["question_id"]]["em"])
            groups["incorrect"].sort(key=lambda r: -dense_by_id[r["question_id"]]["em"])
        examples[task] = {name: group[:2] for name, group in groups.items()}
    out = ROOT / "results"
    if len({m["corpus_content_digest"] for m in metrics.values()}) > 1 or len({m["questions_content_digest"] for m in metrics.values()}) > 1:
        raise ValueError("Tasks used different corpus/question contents. Freeze data, then rerun all tasks before comparison.")
    out.mkdir(exist_ok=True)
    if "task2" in all_rows and "task3" in all_rows:
        dense = {r["question_id"]: r for r in all_rows["task2"]}
        hybrid = {r["question_id"]: r for r in all_rows["task3"]}
        if dense.keys() != hybrid.keys():
            raise ValueError("Dense and hybrid runs must score the same question IDs")
        pairs = [(dense[qid], hybrid[qid]) for qid in dense]
        identical = [(a, b) for a, b in pairs if [e["passage_id"] for e in a["evidence"]] == [e["passage_id"] for e in b["evidence"]]]
        paired = {"n": len(pairs),
                  "both_correct": sum(a["em"] and b["em"] for a, b in pairs),
                  "dense_only_correct": sum(a["em"] and not b["em"] for a, b in pairs),
                  "hybrid_only_correct": sum(not a["em"] and b["em"] for a, b in pairs),
                  "both_incorrect": sum(not a["em"] and not b["em"] for a, b in pairs),
                  "identical_top5_prompts": len(identical),
                  "identical_prompt_output_disagreements": sum(a["prediction"] != b["prediction"] for a, b in identical)}
        write_json(out / "paired_comparison.json", paired)
    write_json(out / "comparison.json", metrics)
    write_json(out / "groundedness.json", support)
    write_json(out / "qualitative_examples.json", examples)
    fields = ["task", "n", "em", "f1", "recall@1", "recall@3", "recall@5", "recall@10"]
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics.values())
    table = "| System | N | EM | F1 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for task, row in metrics.items():
        table += f'| {task} | {row["n"]} | ' + " | ".join(f"{row[k]:.2f}" for k in fields[2:]) + " |\n"
    (out / "comparison.md").write_text(table + "\nAll scores are percentages. Recall is an answer-containing passage proxy.\n", encoding="utf-8")
    # Support labeling template is separate from answers/EM to reduce annotation bias.
    audit_path = out / "support_audit.json"
    if "task2" in all_rows and "task3" in all_rows:
        selected_ids = random.Random(cfg["seed"] + 1).sample([r["question_id"] for r in all_rows["task2"]], min(10, len(all_rows["task2"])))
        audit = []
        for task in ("task2", "task3"):
            for row in all_rows[task]:
                if row["question_id"] in selected_ids:
                    audit.append({"task": task, "question_id": row["question_id"], "question": row["question"],
                                  "prediction": row["prediction"], "evidence": row["evidence"],
                                  "run_identity": row["run_identity"],
                                  "claim_digest": digest({"question": row["question"], "prediction": row["prediction"], "evidence": [e["text"] for e in row["evidence"]]}),
                                  "supported": None, "annotation_rule": "Does the evidence entail the proposed answer to this question? Empty answers = 0."})
        write_json(audit_path, audit)
        with (out / "support_labels_TEMPLATE.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["task", "question_id", "supported", "annotator", "rationale", "run_identity", "claim_digest"])
            writer.writeheader()
            writer.writerows({"task": r["task"], "question_id": r["question_id"], "run_identity": r["run_identity"], "claim_digest": r["claim_digest"]} for r in audit)
    if metrics:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
        names = list(metrics)
        x = list(range(len(names)))
        axes[0].bar([v - .18 for v in x], [metrics[n]["em"] for n in names], .36, label="EM", color="#255f85")
        axes[0].bar([v + .18 for v in x], [metrics[n]["f1"] for n in names], .36, label="Token F1", color="#55a7a0")
        axes[0].set_xticks(x, names)
        axes[0].set(ylabel="Score (%)", title="Answer quality", ylim=(0, 100))
        axes[0].legend(frameon=False)
        for n, color in zip(names, ["#255f85", "#55a7a0", "#b26b36"]):
            axes[1].plot([1, 3, 5, 10], [metrics[n][f"recall@{k}"] for k in (1, 3, 5, 10)], marker="o", label=n, color=color)
        axes[1].set(xlabel="Retrieved passages (k)", ylabel="Questions with answer-containing passage (%)",
                    title="Retrieval success proxy", ylim=(0, 100), xticks=[1, 3, 5, 10])
        axes[1].legend(frameon=False)
        fig.savefig(out / "comparison.png", dpi=180)
        plt.close(fig)
    print(table, flush=True)


if __name__ == "__main__":
    analyze()
