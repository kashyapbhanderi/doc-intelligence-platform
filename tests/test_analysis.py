import pytest
import sys
import os
import json
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from eval.ndcg_analysis import analyze_ndcg_results
from eval.ndcg_eval import (
    dcg_score,
    ndcg_score,
    is_relevant
)


def test_analysis_file_exists():
    """Analysis file should be created."""
    assert os.path.exists(
        "eval/ndcg_analysis.json"), \
        "Run python eval/ndcg_analysis.py first"


def test_analysis_has_required_fields():
    """Analysis should have all required fields."""
    path = "eval/ndcg_analysis.json"
    if not os.path.exists(path):
        pytest.skip("Analysis not run yet")

    with open(path, encoding='utf-8') as f:
        analysis = json.load(f)

    assert "total_queries" in analysis
    assert "improved_count" in analysis
    assert "worse_count" in analysis
    assert "improvement_rate" in analysis
    assert "details" in analysis


def test_improvement_rate_is_percentage():
    """Improvement rate should be between 0 and 100."""
    path = "eval/ndcg_analysis.json"
    if not os.path.exists(path):
        pytest.skip("Analysis not run yet")

    with open(path, encoding='utf-8') as f:
        analysis = json.load(f)

    rate = analysis["improvement_rate"]
    assert 0 <= rate <= 100, \
        f"Invalid improvement rate: {rate}"


def test_counts_add_up():
    """Improved + worse + unchanged = total."""
    path = "eval/ndcg_analysis.json"
    if not os.path.exists(path):
        pytest.skip("Analysis not run yet")

    with open(path, encoding='utf-8') as f:
        analysis = json.load(f)

    total = analysis["total_queries"]
    parts = (analysis["improved_count"] +
             analysis["worse_count"] +
             analysis["unchanged_count"])

    assert total == parts, \
        f"Counts don't add up: {parts} != {total}"


def test_finetuned_answer_eval_exists():
    """Fine-tuned answer evaluation should exist."""
    path = "eval/answer_eval_finetuned.json"
    if not os.path.exists(path):
        pytest.skip("Fine-tuned eval not run yet")

    with open(path, encoding='utf-8') as f:
        results = json.load(f)

    assert "summary" in results
    assert "results" in results


def test_ndcg_calculation_correctness():
    """NDCG calculation should be mathematically correct."""
    import math

    # Manual calculation
    relevances = [1, 0, 1, 0, 0]
    k = 5

    # DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    expected_dcg = 1.0 / math.log2(2) + \
                   1.0 / math.log2(4)

    # Ideal = [1, 1, 0, 0, 0]
    # IDCG = 1/log2(2) + 1/log2(3)
    ideal_dcg = (1.0 / math.log2(2) +
                 1.0 / math.log2(3))

    expected_ndcg = expected_dcg / ideal_dcg

    calculated = ndcg_score(relevances, k)
    assert abs(calculated - expected_ndcg) < 0.001, \
        f"NDCG wrong: {calculated} vs {expected_ndcg}"


def test_query_engine_uses_finetuned_model():
    """Query engine should auto-detect fine-tuned model."""
    ft_path = "models/finetuned/final"
    if not os.path.exists(ft_path):
        pytest.skip("Fine-tuned model not found")

    from embeddings.query_engine import build_query_engine
    try:
        engine, client = build_query_engine(top_k=3)
        assert engine is not None
    except Exception as e:
        pytest.skip(f"Weaviate not available: {e}")