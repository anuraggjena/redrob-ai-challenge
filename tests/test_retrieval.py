from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rrf_fusion_orders_overlap():
    from src.retrieval.rrf_fusion import reciprocal_rank_fusion

    dense = ["A", "B", "C", "D"]
    sparse = ["B", "A", "E", "C"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_n=3)
    assert fused[0] in {"A", "B"}


def test_bm25_index_returns_ids():
    from src.retrieval.bm25_index import BM25Index

    docs = [
        "python retrieval ranking faiss",
        "java backend spring",
        "python search embeddings",
    ]
    ids = ["c1", "c2", "c3"]
    index = BM25Index()
    index.build(docs, ids)
    results = index.query("python retrieval", top_k=2)
    assert results[0] == "c1"
    assert len(results) == 2


def test_query_builder_loads_jd():
    from src.retrieval.query_builder import build_query_from_jd

    dense, tokens = build_query_from_jd(ROOT / "config" / "job_requirements.yaml")
    assert "retrieval" in dense.lower()
    assert "retrieval" in tokens
    assert len(dense) > 50
    assert len(tokens) > 10
