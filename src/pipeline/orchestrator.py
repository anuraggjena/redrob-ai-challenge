from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.graph.evidence_graph import build_evidence_graph
from src.graph.graph_scorer import score_graph
from src.pipeline.config_loader import load_job_requirements
from src.reasoning.reason_generator import generate_reasoning
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex
from src.retrieval.query_builder import build_query_from_jd
from src.retrieval.rrf_fusion import reciprocal_rank_fusion
from src.scoring.ensemble_scorer import load_weights, score_batch
from src.scoring.ltr_ranker import LTRRanker
from src.understanding.models import CandidateProfile
from src.understanding.parser import load_candidates
from src.utils.artifacts import candidate_set_hash

_RETRIEVAL_TOP_K = 10_000
_RECALL_POOL_SIZE = 5_000
_LTR_POOL_SIZE = 2_000
_HONEYPOT_THRESHOLD = 0.7


def score_from_rank(rank: int) -> float:
    return round(1.0 - (rank - 1) * 0.008, 4)


def _update_semantic_features(
    features_by_id: dict[str, dict[str, float]],
    dense_ranking: list[str],
    bm25_ranking: list[str],
    recall_pool: list[str],
) -> None:
    dense_rank = {candidate_id: rank for rank, candidate_id in enumerate(dense_ranking, start=1)}
    bm25_rank = {candidate_id: rank for rank, candidate_id in enumerate(bm25_ranking, start=1)}
    rrf_scores: dict[str, float] = {}
    for ranking in (dense_ranking, bm25_ranking):
        for rank, candidate_id in enumerate(ranking, start=1):
            rrf_scores[candidate_id] = rrf_scores.get(candidate_id, 0.0) + 1.0 / (60 + rank)

    max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
    max_dense = max(len(dense_ranking), 1)
    max_bm25 = max(len(bm25_ranking), 1)

    for candidate_id in recall_pool:
        features = features_by_id.setdefault(candidate_id, {})
        dense_score = 1.0 - (dense_rank.get(candidate_id, max_dense + 1) - 1) / max_dense
        bm25_score = 1.0 - (bm25_rank.get(candidate_id, max_bm25 + 1) - 1) / max_bm25
        rrf_score = rrf_scores.get(candidate_id, 0.0) / max_rrf

        features["semantic_dense_score"] = max(0.0, min(1.0, dense_score))
        features["semantic_bm25_score"] = max(0.0, min(1.0, bm25_score))
        features["semantic_rrf_score"] = max(0.0, min(1.0, rrf_score))
        features["semantic_fit_composite"] = (
            0.4 * features["semantic_dense_score"]
            + 0.3 * features["semantic_bm25_score"]
            + 0.3 * features["semantic_rrf_score"]
        )


class RankingPipeline:
    def __init__(self, artifacts_dir: Path, config_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.config_dir = Path(config_dir)
        self.job_requirements_path = self.config_dir / "job_requirements.yaml"
        self.weights = load_weights(self.config_dir / "feature_weights.yaml")
        self.ltr_ranker = LTRRanker(self.artifacts_dir / "ltr_model.lgb")

    def _load_query_embedding(self) -> np.ndarray:
        query_path = self.artifacts_dir / "query_embedding.npy"
        if not query_path.exists():
            raise FileNotFoundError(
                f"Missing precomputed query embedding: {query_path}. "
                "Run scripts/build_indexes.py during offline precompute."
            )
        return np.load(query_path)

    def run(
        self, candidates_path: Path, top_n: int = 100
    ) -> tuple[list[dict[str, Any]], dict[str, CandidateProfile]]:
        profiles = load_candidates(candidates_path)
        profiles_by_id = {profile.candidate_id: profile for profile in profiles}
        candidate_ids = [profile.candidate_id for profile in profiles]
        valid_ids = set(candidate_ids)

        features_by_id = self._load_features_from_artifacts(candidate_ids)
        recall_pool, dense_hits, bm25_hits = self._retrieve(candidate_ids, valid_ids)
        if dense_hits or bm25_hits:
            _update_semantic_features(features_by_id, dense_hits, bm25_hits, recall_pool)

        coarse_scores = score_batch(candidate_ids, features_by_id, self.weights)
        recall_scores = {cid: coarse_scores[cid] for cid in recall_pool if cid in coarse_scores}
        outside_scores = {
            cid: coarse_scores[cid] for cid in candidate_ids if cid not in recall_scores
        }
        merged_scores = {**outside_scores, **recall_scores}

        top_2k = self._select_top(merged_scores, _LTR_POOL_SIZE)
        final_ranked = self._ltr_rerank(top_2k, features_by_id)
        top_filtered = self._apply_honeypot_filter(final_ranked, features_by_id, top_n)

        rows: list[dict[str, Any]] = [
            {"candidate_id": candidate_id, "raw_score": float(score)}
            for candidate_id, score in top_filtered
        ]
        return rows, profiles_by_id

    def _load_features_from_artifacts(self, candidate_ids: list[str]) -> dict[str, dict[str, float]]:
        parquet_path = self.artifacts_dir / "features.parquet"
        meta_path = self.artifacts_dir / "features_meta.json"
        current_hash = candidate_set_hash(candidate_ids)

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Missing precomputed features: {parquet_path}. "
                "Run scripts/build_features.py before ranking."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing features metadata: {meta_path}. "
                "Re-run scripts/build_features.py."
            )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stored_hash = meta.get("candidate_hash")
        if stored_hash != current_hash:
            raise ValueError(
                "Feature cache hash mismatch. "
                f"Expected {current_hash[:12]}..., found {str(stored_hash)[:12]}... "
                "Rebuild artifacts with build_features.py for this candidate file."
            )

        frame = pd.read_parquet(parquet_path)
        id_set = set(candidate_ids)
        frame = frame[frame["candidate_id"].isin(id_set)]
        if len(frame) != len(candidate_ids):
            raise ValueError(
                f"Feature parquet missing rows: expected {len(candidate_ids)}, found {len(frame)}"
            )

        features_by_id: dict[str, dict[str, float]] = {}
        for record in frame.to_dict(orient="records"):
            candidate_id = str(record.pop("candidate_id"))
            features_by_id[candidate_id] = {key: float(value) for key, value in record.items()}
        return features_by_id

    def _retrieve(
        self,
        candidate_ids: list[str],
        valid_ids: set[str],
    ) -> tuple[list[str], list[str], list[str]]:
        faiss_path = self.artifacts_dir / "faiss.index"
        bm25_dir = self.artifacts_dir / "bm25"
        if not faiss_path.exists() or not (bm25_dir / "bm25.pkl").exists():
            return list(candidate_ids), [], []

        dense_index = DenseIndex()
        dense_index.load(self.artifacts_dir)

        bm25_index = BM25Index()
        bm25_index.load(bm25_dir)

        dense_query, bm25_tokens = build_query_from_jd(self.job_requirements_path)
        query_embedding = self._load_query_embedding()

        dense_hits = [
            cid
            for cid in dense_index.search(
                query_embedding,
                top_k=min(_RETRIEVAL_TOP_K, len(candidate_ids)),
            )
            if cid in valid_ids
        ]
        bm25_hits = [
            cid
            for cid in bm25_index.query(
                " ".join(bm25_tokens),
                top_k=min(_RETRIEVAL_TOP_K, len(candidate_ids)),
            )
            if cid in valid_ids
        ]

        recall_pool = [
            cid
            for cid in reciprocal_rank_fusion(
                [dense_hits, bm25_hits],
                k=60,
                top_n=min(_RECALL_POOL_SIZE, len(candidate_ids)),
            )
            if cid in valid_ids
        ]
        return recall_pool, dense_hits, bm25_hits

    @staticmethod
    def _select_top(scores: dict[str, float], limit: int) -> list[str]:
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [candidate_id for candidate_id, _ in ranked[:limit]]

    def _ltr_rerank(
        self,
        candidate_ids: list[str],
        features_by_id: dict[str, dict[str, float]],
    ) -> list[tuple[str, float]]:
        if not candidate_ids:
            return []

        if self.ltr_ranker.is_loaded:
            ranked = self.ltr_ranker.rank(
                candidate_ids,
                features_by_id,
                top_n=len(candidate_ids),
            )
            if ranked:
                return ranked

        ensemble_scores = score_batch(candidate_ids, features_by_id, self.weights)
        ranked = sorted(
            ensemble_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [(candidate_id, float(score)) for candidate_id, score in ranked]

    @staticmethod
    def _apply_honeypot_filter(
        ranked: list[tuple[str, float]],
        features_by_id: dict[str, dict[str, float]],
        top_n: int,
    ) -> list[tuple[str, float]]:
        selected: list[tuple[str, float]] = []
        selected_ids: set[str] = set()

        for candidate_id, score in ranked:
            honeypot_probability = float(
                features_by_id.get(candidate_id, {}).get("honeypot_probability", 0.0)
            )
            if honeypot_probability > _HONEYPOT_THRESHOLD:
                continue
            selected.append((candidate_id, score))
            selected_ids.add(candidate_id)
            if len(selected) >= top_n:
                return selected

        for candidate_id, score in ranked:
            if candidate_id in selected_ids:
                continue
            selected.append((candidate_id, score))
            selected_ids.add(candidate_id)
            if len(selected) >= top_n:
                break

        if len(selected) < top_n:
            raise RuntimeError(
                f"Honeypot filter could not fill top {top_n}; only {len(selected)} candidates available."
            )
        return selected
