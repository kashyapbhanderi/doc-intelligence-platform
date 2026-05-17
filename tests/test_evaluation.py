import pytest
import sys
import os
import json
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from eval.ndcg_eval import dcg_score, ndcg_score, is_relevant


def test_dcg_perfect_ranking():
    """Perfect ranking — relevant doc at position 1."""
    relevances = [1, 0, 0, 0, 0]
    score = dcg_score(relevances, k=5)
    assert score > 0


def test_dcg_zero_for_no_relevant():
    """No relevant docs should give DCG of 0."""
    relevances = [0, 0, 0, 0, 0]
    score = dcg_score(relevances, k=5)
    assert score == 0.0


def test_ndcg_perfect_score():
    """Perfect ranking should give NDCG of 1.0."""
    relevances = [1, 0, 0, 0, 0]
    score = ndcg_score(relevances, k=5)
    assert score == 1.0


def test_ndcg_zero_for_no_relevant():
    """No relevant docs should give NDCG of 0."""
    relevances = [0, 0, 0, 0, 0]
    score = ndcg_score(relevances, k=5)
    assert score == 0.0


def test_ndcg_between_zero_and_one():
    """NDCG should always be between 0 and 1."""
    relevances = [0, 1, 0, 0, 1]
    score = ndcg_score(relevances, k=5)
    assert 0.0 <= score <= 1.0


def test_is_relevant_source_match():
    """Result from same source should be relevant."""
    result = {"source": "lora.pdf",
              "text": "some text about LoRA"}
    qa = {"source": "lora.pdf",
          "answer": "LoRA reduces trainable parameters"}
    assert is_relevant(result, qa) == 1


def test_is_relevant_different_source():
    """Result from different source should be not relevant."""
    result = {"source": "bert.pdf",
              "text": "BERT uses bidirectional attention"}
    qa = {"source": "lora.pdf",
          "answer": "LoRA reduces trainable parameters"}
    assert is_relevant(result, qa) == 0


def test_qa_dataset_exists():
    """Q&A dataset file should exist after generation."""
    qa_path = "eval/qa_dataset.json"
    assert os.path.exists(qa_path), (
        "Run python eval/generate_qa.py first"
    )


def test_qa_dataset_has_50_pairs():
    """Dataset should have at least 50 Q&A pairs."""
    qa_path = "eval/qa_dataset.json"
    if not os.path.exists(qa_path):
        pytest.skip("Q&A dataset not generated yet")

    with open(qa_path, encoding='utf-8') as f:
        pairs = json.load(f)
    assert len(pairs) >= 50


def test_qa_pairs_have_required_fields():
    """Every Q&A pair must have question and answer."""
    qa_path = "eval/qa_dataset.json"
    if not os.path.exists(qa_path):
        pytest.skip("Q&A dataset not generated yet")

    with open(qa_path, encoding='utf-8') as f:
        pairs = json.load(f)

    for pair in pairs:
        assert "question" in pair
        assert "answer" in pair
        assert "source" in pair
        # After cleaning, all pairs should be good quality
        assert len(pair["question"]) >= 10, \
            f"Short question found: '{pair['question']}'"
        assert len(pair["answer"]) >= 15, \
            f"Short answer found: '{pair['answer']}'"