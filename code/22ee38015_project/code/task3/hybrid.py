"""RRF combines ranks rather than incomparable BM25/cosine raw scores."""


def reciprocal_rank_fusion(bm25, dense, constant=60, depth=50):
    scores = {}
    for ranking in (bm25, dense):
        seen = set()
        for rank, (index, _) in enumerate(ranking, 1):
            if index in seen:
                raise ValueError("A retriever returned duplicate passage IDs")
            seen.add(index)
            scores[index] = scores.get(index, 0.0) + 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:depth]
