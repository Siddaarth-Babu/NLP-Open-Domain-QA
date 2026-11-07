"""Answer/evidence overlap is a warning signal, never a proof of entailment."""
from common.metrics import overlap_score


def check(answer, evidence, threshold=0.8):
    score = overlap_score(answer, "\n".join(p["text"] for p in evidence))
    return {"overlap": score, "high_overlap": bool(answer.strip()) and score >= threshold,
            "abstained": not bool(answer.strip())}
