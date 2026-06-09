from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, top_n: int = 5000
) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [
        doc_id
        for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    ]
