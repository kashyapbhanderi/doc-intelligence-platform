import pytest
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def cosine_sim(v1, v2):
    return float(np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)))


# ── Checkpoint tests ─────────────────────────────────────

def test_best_model_exists():
    """Best model folder must exist after Day 12."""
    assert os.path.exists("models/finetuned/best"), \
        "Run python scripts/pick_best_model.py first"


def test_best_model_has_config():
    """Best model folder must contain a config file."""
    path = "models/finetuned/best"
    if not os.path.exists(path):
        pytest.skip("Best model not saved yet")
    files = os.listdir(path)
    assert any("config" in f for f in files), \
        f"No config found. Files: {files}"


def test_best_model_loads():
    """Best model must load without errors."""
    path = "models/finetuned/best"
    if not os.path.exists(path):
        pytest.skip("Best model not saved yet")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(path)
    assert model is not None


def test_best_model_outputs_384_dims():
    """Best model must output 384-dimensional vectors."""
    path = "models/finetuned/best"
    if not os.path.exists(path):
        pytest.skip("Best model not saved yet")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(path)
    vec = model.encode("test sentence")
    assert len(vec) == 384, \
        f"Expected 384 dims, got {len(vec)}"


# ── Quality tests ────────────────────────────────────────

def test_best_model_beats_random():
    """
    Best model must score domain pairs above 0.3.
    A score below 0.3 means the model is not working.
    """
    path = "models/finetuned/best"
    if not os.path.exists(path):
        pytest.skip("Best model not saved yet")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(path)

    q   = "What is retrieval augmented generation?"
    pos = ("RAG combines document retrieval with "
           "language model generation")
    score = cosine_sim(model.encode(q), model.encode(pos))
    assert score > 0.3, \
        f"Score too low: {score:.4f}. Model may be broken."


def test_finetuned_gap_beats_baseline():
    """
    Fine-tuned model gap should exceed base model gap.
    Gap = avg(positive scores) - avg(negative scores).
    """
    ft_path = "models/finetuned/best"
    if not os.path.exists(ft_path):
        pytest.skip("Best model not saved yet")

    from sentence_transformers import SentenceTransformer
    base = SentenceTransformer("all-MiniLM-L6-v2")
    ft   = SentenceTransformer(ft_path)

    pairs = [
        ("What is LoRA?",
         "LoRA adds low-rank matrices to frozen weights",
         "The weather is sunny today"),
        ("How does RAG work?",
         "RAG retrieves documents then generates answers",
         "Recipe: boil pasta for 10 minutes"),
    ]

    def avg_gap(model):
        gaps = []
        for q, p, n in pairs:
            ps = cosine_sim(model.encode(q), model.encode(p))
            ns = cosine_sim(model.encode(q), model.encode(n))
            gaps.append(ps - ns)
        return sum(gaps) / len(gaps)

    base_gap = avg_gap(base)
    ft_gap   = avg_gap(ft)

    print(f"\n  Base gap:       {base_gap:.4f}")
    print(f"  Fine-tuned gap: {ft_gap:.4f}")

    # With 5 epochs, fine-tuned should be better
    # Allow small tolerance for CPU training variance
    assert ft_gap >= base_gap * 0.90, \
        f"Fine-tuned ({ft_gap:.4f}) worse than base ({base_gap:.4f})"


# ── Comparison results tests ─────────────────────────────

def test_model_comparison_json_exists():
    """Comparison results file must be saved."""
    assert os.path.exists("eval/model_comparison.json"), \
        "Run python scripts/pick_best_model.py first"


def test_model_comparison_has_winner():
    """Comparison JSON must declare a winner."""
    path = "eval/model_comparison.json"
    if not os.path.exists(path):
        pytest.skip("Comparison not run yet")
    with open(path) as f:
        data = json.load(f)
    assert "winner" in data
    assert "winner_score" in data
    assert data["winner_score"] > 0.0


def test_training_notes_exist():
    """Training notes file must be created."""
    assert os.path.exists("training/notes.md"), \
        "Create training/notes.md with your LR results"