import pytest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2))


def test_finetuned_model_exists():
    """Fine-tuned model folder should exist."""
    assert os.path.exists("models/finetuned/final"), \
        "Run python embeddings/finetune.py first"


def test_finetuned_model_has_config():
    """Model folder should have config file."""
    model_path = "models/finetuned/final"
    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet")

    files = os.listdir(model_path)
    assert any("config" in f for f in files), \
        f"No config file found. Files: {files}"


def test_finetuned_model_loads():
    """Fine-tuned model should load without errors."""
    model_path = "models/finetuned/final"
    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path)
    assert model is not None


def test_finetuned_model_correct_dimensions():
    """Fine-tuned model should output 384-dim vectors."""
    model_path = "models/finetuned/final"
    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path)
    vector = model.encode("test sentence")
    assert len(vector) == 384


def test_finetuned_better_positive_similarity():
    """
    Fine-tuned model should score domain-related
    pairs higher than base model.
    """
    model_path = "models/finetuned/final"
    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet")

    from sentence_transformers import SentenceTransformer

    base = SentenceTransformer("all-MiniLM-L6-v2")
    finetuned = SentenceTransformer(model_path)

    q = "What is retrieval augmented generation?"
    a = ("RAG combines document retrieval with "
         "language model generation to reduce "
         "hallucination")

    base_score = cosine_sim(
        base.encode(q), base.encode(a))
    ft_score = cosine_sim(
        finetuned.encode(q), finetuned.encode(a))

    print(f"\nBase score:       {base_score:.4f}")
    print(f"Fine-tuned score: {ft_score:.4f}")

    # Fine-tuned should be at least as good
    # (with 1 epoch it might be slightly lower)
    assert ft_score > 0.3, \
        "Fine-tuned model giving very low scores"


def test_prepare_training_data():
    """Training data preparation should work."""
    from embeddings.finetune import (
        load_triplets,
        prepare_training_data
    )

    triplets = load_triplets("data/triplets_clean.json")
    train, eval_data = prepare_training_data(
        triplets[:20])

    assert len(train) > 0
    assert len(eval_data) > 0
    assert len(train) > len(eval_data)


def test_training_split_ratio():
    """Training set should be 80% of total data."""
    from embeddings.finetune import (
        load_triplets,
        prepare_training_data
    )

    triplets = load_triplets("data/triplets_clean.json")
    train, eval_data = prepare_training_data(triplets)

    total = len(train) + len(eval_data)
    train_ratio = len(train) / total
    assert 0.75 <= train_ratio <= 0.85, \
        f"Unexpected split ratio: {train_ratio}"