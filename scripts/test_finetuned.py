import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from sentence_transformers import SentenceTransformer
import numpy as np


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )


def test_model(model_path, model_name):
    """Test a model on domain-specific pairs."""
    model = SentenceTransformer(model_path)
    print(f"\nTesting: {model_name}")
    print("-" * 50)

    # These pairs should score HIGH (related)
    positive_pairs = [
        ("What is LoRA fine-tuning?",
         "LoRA reduces trainable parameters by adding low-rank matrices to frozen model weights"),
        ("How does attention work?",
         "Attention mechanism allows the model to focus on relevant parts of input sequence"),
        ("What is RAG?",
         "Retrieval augmented generation combines document retrieval with language model generation"),
    ]

    # These pairs should score LOW (unrelated)
    negative_pairs = [
        ("What is LoRA fine-tuning?",
         "The stock market showed volatility today"),
        ("How does attention work?",
         "Recipe: mix flour, eggs, and butter together"),
    ]

    pos_scores = []
    neg_scores = []

    print("Positive pairs (should score HIGH):")
    for q, a in positive_pairs:
        v1 = model.encode(q)
        v2 = model.encode(a)
        score = cosine_similarity(v1, v2)
        pos_scores.append(score)
        icon = "✅" if score > 0.4 else "⚠️"
        print(f"  {icon} {score:.4f} | {q[:40]}")

    print("\nNegative pairs (should score LOW):")
    for q, a in negative_pairs:
        v1 = model.encode(q)
        v2 = model.encode(a)
        score = cosine_similarity(v1, v2)
        neg_scores.append(score)
        icon = "✅" if score < 0.4 else "⚠️"
        print(f"  {icon} {score:.4f} | {q[:40]}")

    avg_pos = sum(pos_scores) / len(pos_scores)
    avg_neg = sum(neg_scores) / len(neg_scores)
    gap = avg_pos - avg_neg

    print(f"\nAvg positive score: {avg_pos:.4f}")
    print(f"Avg negative score: {avg_neg:.4f}")
    print(f"Gap (higher=better): {gap:.4f}")

    return avg_pos, avg_neg, gap


if __name__ == "__main__":
    print("Comparing base vs fine-tuned model")
    print("=" * 60)

    # Test base model
    base_pos, base_neg, base_gap = test_model(
        "all-MiniLM-L6-v2",
        "BASE MODEL (all-MiniLM-L6-v2)"
    )

    # Test fine-tuned model
    ft_path = "models/finetuned/final"
    if os.path.exists(ft_path):
        ft_pos, ft_neg, ft_gap = test_model(
            ft_path,
            "FINE-TUNED MODEL"
        )

        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<25} {'Base':>8} {'Fine-tuned':>12} {'Change':>8}")
        print("-" * 55)
        print(f"{'Avg positive score':<25} "
              f"{base_pos:>8.4f} "
              f"{ft_pos:>12.4f} "
              f"{ft_pos-base_pos:>+8.4f}")
        print(f"{'Avg negative score':<25} "
              f"{base_neg:>8.4f} "
              f"{ft_neg:>12.4f} "
              f"{ft_neg-base_neg:>+8.4f}")
        print(f"{'Gap (pos - neg)':<25} "
              f"{base_gap:>8.4f} "
              f"{ft_gap:>12.4f} "
              f"{ft_gap-base_gap:>+8.4f}")

        if ft_gap > base_gap:
            print("\n✅ Fine-tuned model is BETTER!")
            print("   Positive pairs are more similar")
            print("   Negative pairs are less similar")
        else:
            print("\n⚠️  More training needed.")
            print("   Run 3 epochs tomorrow for improvement.")
    else:
        print(f"\nFine-tuned model not found at {ft_path}")
        print("Run: python embeddings/finetune.py first")