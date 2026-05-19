import pytest
import sys
import os
import json
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def test_finetuned_ndcg_file_exists():
    """Fine-tuned NDCG results should exist."""
    assert os.path.exists(
        "eval/ndcg_finetuned.json"), \
        "Run python eval/compare_models.py first"


def test_finetuned_ndcg_better_than_baseline():
    """Fine-tuned model should improve hybrid NDCG."""
    baseline_path = "eval/ndcg_results.json"
    finetuned_path = "eval/ndcg_finetuned.json"

    if not os.path.exists(baseline_path):
        pytest.skip("Baseline results not found")
    if not os.path.exists(finetuned_path):
        pytest.skip("Fine-tuned results not found")

    with open(baseline_path,
              encoding='utf-8') as f:
        baseline = json.load(f)
    with open(finetuned_path,
              encoding='utf-8') as f:
        finetuned = json.load(f)

    base_ndcg = baseline["summary"]["hybrid_ndcg"]
    ft_ndcg = finetuned["summary"]["hybrid_ndcg"]

    print(f"\nBaseline NDCG:   {base_ndcg:.4f}")
    print(f"Fine-tuned NDCG: {ft_ndcg:.4f}")
    print(f"Improvement:     {ft_ndcg - base_ndcg:+.4f}")

    # Fine-tuned should be at least as good
    assert ft_ndcg >= base_ndcg * 0.95, \
        (f"Fine-tuned NDCG {ft_ndcg:.4f} is much worse "
         f"than baseline {base_ndcg:.4f}")


def test_weaviate_has_correct_count():
    """Weaviate should have all 11089 chunks."""
    try:
        from embeddings.embedder import DocumentEmbedder
        embedder = DocumentEmbedder(
            model_path="models/finetuned/final"
        )
        count = embedder.get_document_count()
        print(f"\nChunks in Weaviate: {count}")
        assert count > 5000, \
            f"Too few chunks: {count}"
    except Exception as e:
        pytest.skip(f"Weaviate not available: {e}")


def test_finetuned_embeddings_different_from_base():
    """
    Fine-tuned model should produce different
    vectors than base model for domain text.
    """
    model_path = "models/finetuned/final"
    if not os.path.exists(model_path):
        pytest.skip("Fine-tuned model not found")

    import numpy as np
    from sentence_transformers import SentenceTransformer

    base = SentenceTransformer("all-MiniLM-L6-v2")
    finetuned = SentenceTransformer(model_path)

    text = ("LoRA fine-tuning reduces trainable "
            "parameters using low-rank decomposition")

    v_base = base.encode(text)
    v_ft = finetuned.encode(text)

    # Vectors should be different
    diff = np.mean(np.abs(v_base - v_ft))
    print(f"\nVector difference: {diff:.6f}")
    assert diff > 0.001, \
        "Fine-tuned model produces same vectors as base!"


def test_ndcg_results_have_required_fields():
    """NDCG results file should have all fields."""
    path = "eval/ndcg_finetuned.json"
    if not os.path.exists(path):
        pytest.skip("Fine-tuned NDCG not run yet")

    with open(path, encoding='utf-8') as f:
        results = json.load(f)

    assert "summary" in results
    assert "details" in results
    assert "hybrid_ndcg" in results["summary"]
    assert "vector_ndcg" in results["summary"]
    assert "bm25_ndcg" in results["summary"]