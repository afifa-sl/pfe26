"""Fusion hybride Dense + Sparse via Reciprocal Rank Fusion (RRF)."""
from typing import List, Dict, Any


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Combine les résultats dense et sparse avec RRF.

    Score RRF = Σ 1 / (k + rank_i)
    Formule originale: Cormack et al., 2009
    k=60 est la valeur par défaut recommandée.
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(dense_results, 1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(sparse_results, 1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for rank, doc_id in enumerate(sorted_ids, 1):
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = round(rrf_scores[doc_id], 6)
        doc["rank"] = rank
        results.append(doc)

    return results
