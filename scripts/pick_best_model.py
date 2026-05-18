import os
import sys
import json
import shutil
import numpy as np
sys.path.insert(0, os.path.abspath('.'))

import mlflow
from sentence_transformers import SentenceTransformer


def cosine_sim(v1, v2):
    return float(
        np.dot(v1, v2) /
        (np.linalg.norm(v1) * np.linalg.norm(v2))
    )


def score_model(model_path: str) -> float:
    """
    Score a model on 5 domain-specific pairs.
    Returns average gap between positive and negative scores.
    Higher gap = better model.
    """
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return 0.0

    model = SentenceTransformer(model_path)

    pairs = [
        (
            "What is LoRA fine-tuning?",
            "LoRA reduces trainable parameters by adding low-rank matrices",
            "The stock market closed higher today"
        ),
        (
            "How does attention mechanism work?",
            "Attention allows the model to focus on relevant parts of input",
            "Recipe ingredients include flour and butter"
        ),
        (
            "What is retrieval augmented generation?",
            "RAG combines document retrieval with language model generation",
            "Weather forecast shows rain tomorrow"
        ),
        (
            "How does BERT differ from GPT?",
            "BERT is bidirectional while GPT is autoregressive left-to-right",
            "The train departs at 9am from platform 3"
        ),
        (
            "What is contrastive learning?",
            "Contrastive learning trains models by comparing similar and dissimilar examples",
            "The garden needs watering every two days"
        ),
    ]

    gaps = []
    for q, pos, neg in pairs:
        q_vec   = model.encode(q)
        pos_vec = model.encode(pos)
        neg_vec = model.encode(neg)
        pos_score = cosine_sim(q_vec, pos_vec)
        neg_score = cosine_sim(q_vec, neg_vec)
        gaps.append(pos_score - neg_score)

    return sum(gaps) / len(gaps)


def compare_and_save_best():
    """
    Compare all candidate checkpoints and save the best
    to models/finetuned/best.
    """
    candidates = {
        "base_model":    "all-MiniLM-L6-v2",
        "lr2e-5_ep5":    "models/finetuned/lr2e-5_ep5",
        "lr3e-5_ep5":    "models/finetuned/lr3e-5_ep5",
    }

    print("=" * 55)
    print("  MODEL COMPARISON")
    print("=" * 55)
    print(f"  {'Model':<20} {'Gap Score':>12}  {'Status':>10}")
    print(f"  {'-'*20} {'-'*12}  {'-'*10}")

    scores = {}
    for name, path in candidates.items():
        score = score_model(path)
        scores[name] = score
        status = "✅ exists" if os.path.exists(path) or name == "base_model" else "❌ missing"
        print(f"  {name:<20} {score:>12.4f}  {status:>10}")

    print("=" * 55)

    # Pick winner (highest gap = best separation)
    winner = max(scores, key=scores.get)
    winner_path = candidates[winner]
    winner_score = scores[winner]

    print(f"\n  Winner: {winner}  (gap = {winner_score:.4f})")

    # Save best model
    best_path = "models/finetuned/best"
    if os.path.exists(best_path):
        shutil.rmtree(best_path)

    if winner == "base_model":
        print("\n  ⚠️  Base model won — fine-tuning needs more epochs.")
        print("     Copying base model to best/ path for consistency.")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.save(best_path)
    else:
        shutil.copytree(winner_path, best_path)
        print(f"\n  ✅ Best model copied → {best_path}")

    # Save comparison results to JSON
    results = {
        "winner": winner,
        "winner_path": winner_path,
        "winner_score": round(winner_score, 4),
        "all_scores": {k: round(v, 4) for k, v in scores.items()},
    }
    os.makedirs("eval", exist_ok=True)
    with open("eval/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved → eval/model_comparison.json")
    print(f"\n  Best model ready at: {best_path}")
    print(f"  Use it tomorrow in Day 13 benchmarking.")
    return winner, winner_score


if __name__ == "__main__":
    compare_and_save_best()
    