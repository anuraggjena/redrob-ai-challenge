import pandas as pd
import pytest

from scripts.evaluate import compute_composite, evaluate_ranking, ndcg_at_k


def test_ndcg_perfect_ranking():
    labels_df = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "label": [5.0, 4.0, 2.0, 0.0],
        }
    )
    submission_df = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "rank": [1, 2, 3, 4],
            "score": [0.95, 0.85, 0.55, 0.10],
            "reasoning": ["", "", "", ""],
        }
    )

    report = evaluate_ranking(labels_df, submission_df, k_values=[10])
    assert report["ndcg@10"] == pytest.approx(1.0, abs=1e-6)

    worse_submission = submission_df.copy()
    worse_submission.loc[worse_submission["candidate_id"] == "d", "rank"] = 1
    worse_submission.loc[worse_submission["candidate_id"] == "a", "rank"] = 4
    worse = evaluate_ranking(labels_df, worse_submission, k_values=[10])
    assert worse["ndcg@10"] < report["ndcg@10"]


def test_composite_formula():
    metrics = {
        "ndcg@10": 0.8,
        "ndcg@50": 0.7,
        "map": 0.6,
        "p@10": 0.5,
    }
    composite = compute_composite(metrics)
    expected = 0.50 * 0.8 + 0.30 * 0.7 + 0.15 * 0.6 + 0.05 * 0.5
    assert composite == pytest.approx(expected)


def test_ndcg_at_k_empty():
    assert ndcg_at_k([], [], k=10) == 0.0
